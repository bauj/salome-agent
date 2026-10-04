from executor.shaper.sketch import exec_records
from executor.shaper.state import (
    feature_shape, measure, new_name, publish_study, record, sketch_of, state, tool,
    transaction,
)

_AXES = ("OX", "OY", "OZ")


def _base_selections(executor, base) -> tuple[list, list]:
    """A sketch (all its regions) or a published feature (all its results) as
    SHAPER selection source, plus the vars it references."""
    st = state(executor)
    names = base if isinstance(base, list) else [base]
    sels, refs = [], []
    for name in names:
        if name in st.sketches:
            sels.append(f'model.selection("COMPOUND", "all-in-{name}")')
        elif name in st.objects:
            sels += _result_selections(st, name)
        else:
            raise ValueError(_unknown(st, name))
        refs.append(name)
    return sels, refs


def _result_selections(st, name: str) -> list:
    feat = st.ns[name].feature()
    return [f'model.selection("{r.shape().shapeTypeStr()}", {r.data().name()!r})'
            for r in feat.results() if r.shape() is not None]


def _unknown(st, name) -> str:
    known = ", ".join([*st.sketches, *st.objects]) or "none yet"
    return f"{name!r} is not a shaper sketch or feature — built so far: {known}"


def _direction(direction) -> str:
    if direction in (None, ""):
        return "model.selection()"
    if direction not in _AXES:
        raise ValueError(f"direction must be one of {', '.join(_AXES)} or omitted "
                         f"(the sketch normal), got {direction!r}")
    return f'model.selection("EDGE", "PartSet/{direction}")'


def _build(executor, name: str, fn: str, code: str, refs: list, kind: str) -> dict:
    """Create one feature from `code`; on a SHAPER error remove it again and fail."""
    st = state(executor)
    # Recompute first: a selection by name is resolved when the feature is
    # created, and result.py runs the whole history in one transaction.
    rec = record(fn, "model.do()\n" + code, out=[name], refs=refs, publish=True)
    try:
        exec_records(st, [rec])
        feat = st.ns[name].feature()
        err = feat.error()
        if not err and not any(r.shape() is not None for r in feat.results()):
            err = "the operation produced nothing"
    except Exception as e:  # noqa: BLE001
        err = str(e) or type(e).__name__
    if err:
        if name in st.ns:
            with transaction():
                st.ns["shaper_doc"].removeFeature(st.ns.pop(name).feature())
        return {"ok": False, "error": f"{fn}: {err}"}
    st.objects[name] = kind
    publish_study(executor)
    shape = feature_shape(executor, name)
    info = measure(executor, name, shape)
    # a later geom stage reaches the result by name (a one-way copy)
    executor.geom_bindings[name] = shape
    return {"ok": True, "records": [rec], "name": name, "measure": info,
            "result_shape": name, "shape_info": info}


def _size(args: dict) -> tuple:
    size = args.get("size")
    if isinstance(size, bool) or not isinstance(size, (int, float)):
        raise ValueError(f"size must be a number, got {size!r}")
    back = args.get("from_size", 0)
    if isinstance(back, bool) or not isinstance(back, (int, float)):
        raise ValueError(f"from_size must be a number, got {back!r}")
    return size, back


@tool("make_face", category="feature",
      description=(
          "Make faces from closed regions of a sketch: `regions` is a list of region "
          "indices from the sketch's latest reply (indices change when the sketch is "
          "edited), or \"all\". One face per region."),
      schema={"type": "object", "properties": {
          "sketch": {"type": "string"},
          "regions": {"anyOf": [{"type": "array", "items": {"type": "integer"}},
                                {"type": "string", "enum": ["all"]}]}},
          "required": ["sketch", "regions"]})
def tool_make_face(executor, args: dict):
    import ModelAPI
    st = state(executor)
    sk = sketch_of(executor, args.get("sketch"))
    regions = args.get("regions")
    name = new_name(executor, "sh_face")
    if regions == "all":
        sels = [f'model.selection("COMPOUND", "all-in-{sk.name}")']
    else:
        res = sk.api.feature().firstResult()
        n = ModelAPI.modelAPI_ResultConstruction(res).facesNum() if res is not None else 0
        if not isinstance(regions, list) or not regions:
            raise ValueError("regions must be a non-empty list of indices or \"all\"")
        bad = [i for i in regions if not isinstance(i, int) or not 0 <= i < n]
        if bad:
            raise ValueError(f"no region {bad} in {sk.name} — it has regions "
                             f"0..{n - 1}" if n else f"{sk.name} has no closed region")
        sels = [f'model.selection("FACE", {nm!r})' for nm in _region_names(st, sk, regions)]
    code = f"{name} = model.addFace(shaper_doc, [{', '.join(sels)}])"
    return _build(executor, name, "addFace", code, [sk.name], "face")


def _region_names(st, sk, regions: list) -> list:
    """SHAPER's naming string for each region (what its own dumps write), found
    by selecting the region by object in a throwaway Face feature."""
    import ModelAPI
    res = sk.api.feature().firstResult()
    rc = ModelAPI.modelAPI_ResultConstruction(res)
    with transaction() as m:
        probe = m.addFace(st.ns["shaper_doc"], [m.selection(res, rc.face(i)) for i in regions])
    try:
        lst = probe.feature().selectionList("base_objects")
        return [lst.value(k).namingName() for k in range(lst.size())]
    finally:
        with transaction():
            st.ns["shaper_doc"].removeFeature(probe.feature())


@tool("extrude", category="feature",
      description=(
          "Extrude a face or a whole sketch (all its closed regions) by `size` along "
          "`direction` (OX, OY, OZ; default the sketch normal). from_size extends it the "
          "other way. Returns the solid's name and its measure."),
      schema={"type": "object", "properties": {
          "base": {"type": "string", "description": "a sketch or a make_face result"},
          "size": {"type": "number"},
          "from_size": {"type": "number"},
          "direction": {"type": "string", "enum": list(_AXES)}},
          "required": ["base", "size"]})
def tool_extrude(executor, args: dict):
    sels, refs = _base_selections(executor, args.get("base"))
    size, back = _size(args)
    name = new_name(executor, "extrusion")
    code = (f"{name} = model.addExtrusion(shaper_doc, [{', '.join(sels)}], "
            f"{_direction(args.get('direction'))}, {size!r}, {back!r})")
    return _build(executor, name, "addExtrusion", code, refs, "solid")


def _boolean_extrusion(executor, args: dict, op: str) -> dict:
    st = state(executor)
    sels, refs = _base_selections(executor, args.get("base"))
    target = args.get("target")
    if target not in st.objects:
        raise ValueError(_unknown(st, target))
    targets = _result_selections(st, target)
    before = measure(executor, target)["volume"]
    direction = _direction(args.get("direction"))
    prefix = "extrusion_cut" if op == "Cut" else "extrusion_fuse"
    name = new_name(executor, prefix)
    if args.get("through_all") and op == "Cut":
        code = (f"{name} = model.addExtrusionCut(shaper_doc, [{', '.join(sels)}], "
                f"{direction}, [{', '.join(targets)}])")
    else:
        size, back = _size(args)
        code = (f"{name} = model.addExtrusion{op}(shaper_doc, [{', '.join(sels)}], "
                f"{direction}, {size!r}, {back!r}, [{', '.join(targets)}])")
    out = _build(executor, name, f"addExtrusion{op}", code, refs + [target], "solid")
    if out["ok"]:
        after = out["measure"]["volume"]
        if abs(after - before) <= 1e-6 * max(1.0, abs(before)):
            out["delta"] = (f"{name}: the volume did not change ({before:g}) — the "
                            f"extrusion does not reach {target}")
    return out


@tool("extrude_cut", category="feature",
      description=(
          "Pocket: extrude a face or sketch and remove it from the solid `target`. Give "
          "`size` (and optionally from_size), or through_all: true to cut through the "
          "whole target."),
      schema={"type": "object", "properties": {
          "base": {"type": "string"}, "target": {"type": "string"},
          "size": {"type": "number"}, "from_size": {"type": "number"},
          "through_all": {"type": "boolean"},
          "direction": {"type": "string", "enum": list(_AXES)}},
          "required": ["base", "target"]})
def tool_extrude_cut(executor, args: dict):
    return _boolean_extrusion(executor, args, "Cut")


@tool("extrude_fuse", category="feature",
      description="Boss: extrude a face or sketch by `size` and fuse it to the solid `target`.",
      schema={"type": "object", "properties": {
          "base": {"type": "string"}, "target": {"type": "string"},
          "size": {"type": "number"}, "from_size": {"type": "number"},
          "direction": {"type": "string", "enum": list(_AXES)}},
          "required": ["base", "target", "size"]})
def tool_extrude_fuse(executor, args: dict):
    return _boolean_extrusion(executor, args, "Fuse")


@tool("revolve", category="feature",
      description=(
          "Revolve a face or sketch by `angle` degrees about `axis`: OX, OY, OZ, or a "
          "line of a sketch as \"<sketch>:<line id>\"."),
      schema={"type": "object", "properties": {
          "base": {"type": "string"}, "axis": {"type": "string"},
          "angle": {"type": "number"}},
          "required": ["base", "axis", "angle"]})
def tool_revolve(executor, args: dict):
    sels, refs = _base_selections(executor, args.get("base"))
    axis = args.get("axis")
    if axis in _AXES:
        axis_src = f'model.selection("EDGE", "PartSet/{axis}")'
    elif isinstance(axis, str) and ":" in axis:
        sk_name, _, lid = axis.partition(":")
        sk = sketch_of(executor, sk_name)
        ent = sk.entities.get(lid)
        if ent is None or ent.type != "line":
            raise ValueError(f"{axis!r}: {lid!r} is not a line of {sk_name}")
        axis_src = f"{ent.var}.result()"
        refs += [ent.var, sk.name]
    else:
        raise ValueError(f"axis must be OX, OY, OZ or \"<sketch>:<line id>\", got {axis!r}")
    angle = args.get("angle")
    if isinstance(angle, bool) or not isinstance(angle, (int, float)):
        raise ValueError(f"angle must be a number, got {angle!r}")
    name = new_name(executor, "revolution")
    code = (f"{name} = model.addRevolution(shaper_doc, [{', '.join(sels)}], {axis_src}, "
            f"{angle!r}, 0)")
    return _build(executor, name, "addRevolution", code, refs, "solid")


@tool("measure", category="query",
      description=(
          "Measure published shaper objects by name: for a feature its bbox, size, "
          "centre, area, volume, face and edge counts and validity; for a sketch its "
          "dof, loose_ends and regions."),
      schema={"type": "object", "properties": {
          "names": {"type": "array", "items": {"type": "string"}}},
          "required": ["names"]})
def tool_measure(executor, args: dict):
    from executor.shaper.state import status
    st = state(executor)
    out = {}
    for name in args.get("names") or []:
        if name in st.sketches:
            out[name] = status(executor, st.sketches[name])
        elif name in st.objects:
            out[name] = measure(executor, name)
        else:
            out[name] = None
    return {"ok": True, "records": [], "measurements": out}
