import math

from executor.shaper.sketch import exec_records
from executor.shaper.state import (
    CONSTRUCTION, PUBLISH_LINE, construction_info, forget, measure, new_name, publish_study,
    raw_box, record, sketch_frame, sketch_of, state, tool, transaction,
)

_AXES = ("OX", "OY", "OZ")


def _nested_regions(sk) -> bool:
    """A region with a hole in it: the hole is a region of its own, so taking
    every region fills the hole (task 314)."""
    import GeomAPI
    import ModelAPI
    res = sk.api.feature().firstResult()
    if res is None:
        return False
    rc = ModelAPI.modelAPI_ResultConstruction(res)
    return any(len(rc.face(i).subShapes(GeomAPI.GeomAPI_Shape.WIRE)) > 1
               for i in range(rc.facesNum()))


def _base_selections(executor, base) -> tuple[list, list]:
    """A sketch (all its regions) or a published feature (all its results) as
    SHAPER selection source, plus the vars it references."""
    st = state(executor)
    names = base if isinstance(base, list) else [base]
    sels, refs = [], []
    for name in names:
        if not isinstance(name, str):
            raise ValueError(f"base must be a published name, got {name!r}")
        if name in st.sketches:
            if _nested_regions(st.sketches[name]):
                raise ValueError(
                    f"{name} has a closed region inside another: taking the whole sketch "
                    "would fill the hole too. make_face on the material regions (the "
                    "sketch reply lists them), then use that face")
            sels.append(f'model.selection("COMPOUND", "all-in-{name}")')
        elif name in st.objects:
            _not_construction(st, name)
            sels += _result_selections(st, name)
        else:
            raise ValueError(_unknown(st, name))
        refs.append(name)
    return sels, refs


def _result_selections(st, name: str) -> list:
    feat = st.ns[name].feature()
    results = [r for r in feat.results() if r.shape() is not None]
    if results and all(r.isConcealed() for r in results):
        owners = {a.owner().data().name() for r in results for a in r.data().refsToMe()}
        users = [o for o in st.objects if o in owners]
        by, use = ((", ".join(users), users[-1]) if users
                   else ("a later operation", "that operation's result"))
        raise ValueError(f"{name} was used up by {by}, which replaced it — use {use}, "
                         f"or build a new shape instead of {name}")
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


def _build(executor, name: str, fn: str, code: str, refs: list, kind: str,
           helpers: tuple = ()) -> dict:
    """Create one feature from `code`; on a SHAPER error remove it again and fail.
    `helpers` are features `code` makes before `name`, named `<name>_...`."""
    st = state(executor)
    # Recompute first: a selection by name is resolved when the feature is
    # created, and result.py runs the whole history in one transaction.
    rec = record(fn, "model.do()\n" + code, out=[*helpers, name], refs=refs, publish=True)
    rec["publish"] = [name]
    try:
        exec_records(st, [rec])
        feat = st.ns[name].feature()
        err = feat.error()
        if not err and not any(r.shape() is not None for r in feat.results()):
            err = "the operation produced nothing"
    except Exception as e:  # noqa: BLE001
        err = str(e) or type(e).__name__
    info = None
    if not err:
        st.objects[name] = kind
        try:
            publish_study(executor)
            info = (construction_info(executor, name) if kind in CONSTRUCTION
                    else measure(executor, name))
        except Exception as e:  # noqa: BLE001
            st.objects.pop(name)
            err = str(e) or type(e).__name__
    if err:
        with transaction():
            for var in (name, *reversed(helpers)):
                if var in st.ns:
                    st.ns["shaper_doc"].removeFeature(st.ns.pop(var).feature())
        return {"ok": False, "error": f"{fn}: {err}"}
    return {"ok": True, "records": [rec], "name": name, "measure": info,
            "result_shape": name, "shape_info": info}


def _size(args: dict) -> tuple:
    size = args.get("size")
    if isinstance(size, bool) or not isinstance(size, (int, float)):
        raise ValueError(f"size must be a number, got {size!r}")
    back = args.get("from_size", 0)
    if isinstance(back, bool) or not isinstance(back, (int, float)):
        raise ValueError(f"from_size must be a number, got {back!r}")
    if size + back == 0:
        raise ValueError(f"size {size} with from_size {back} has zero length: from_size "
                         "extends the other way, so it adds to size — for a length the other "
                         f"way use size {-size}, or size 0 with from_size {size}")
    return size, back


@tool("make_face", category="sketch",
      description=(
          "Make faces from closed regions of a sketch: `regions` is a list of region "
          "indices from the sketch's latest reply (indices change when the sketch is "
          "edited), or \"all\". Several regions are merged into one face where they "
          "touch, so the regions of overlapping shapes give one outline."),
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
    res = sk.api.feature().firstResult()
    n = ModelAPI.modelAPI_ResultConstruction(res).facesNum() if res is not None else 0
    if regions == "all":
        if _nested_regions(sk):
            raise ValueError(f"{sk.name} has a closed region inside another: \"all\" would "
                             "fill the hole. List the material regions' indices instead "
                             "(the sketch reply lists them)")
        sels = [f'model.selection("COMPOUND", "all-in-{sk.name}")']
        count = n
    else:
        if not isinstance(regions, list) or not regions:
            raise ValueError("regions must be a non-empty list of indices or \"all\"")
        bad = [i for i in regions if not isinstance(i, int) or not 0 <= i < n]
        if bad:
            raise ValueError(f"no region {bad} in {sk.name} — it has regions "
                             f"0..{n - 1}" if n else f"{sk.name} has no closed region")
        sels = [f'model.selection("FACE", {nm!r})' for nm in _region_names(st, sk, regions)]
        count = len(regions)
    if count < 2:
        code = f"{name} = model.addFace(shaper_doc, [{', '.join(sels)}])"
        return _build(executor, name, "addFace", code, [sk.name], "face")
    # One face per region, then fused with the shared edges removed: one face
    # per connected group of regions.
    parts = f"{name}_regions"
    names = [parts] + [f"{parts}_{i}" for i in range(2, count + 1)]
    code = (f"{parts} = model.addFace(shaper_doc, [{', '.join(sels)}])\n"
            + PUBLISH_LINE.format(var=parts, name=parts) + "\nmodel.do()\n"
            + f"{name} = model.addFuse(shaper_doc, ["
            + ", ".join(f'model.selection("FACE", {nm!r})' for nm in names) + "], True)")
    return _build(executor, name, "addFace", code, [sk.name], "face", helpers=(parts,))


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


@tool("extrude", category="swept",
      description=(
          "Extrude a face or a whole sketch (all its closed regions) by `size` along "
          "`direction` (OX, OY, OZ; default the sketch normal). from_size extends it the "
          "other way: the length is size + from_size, both positive for a two-sided "
          "extrusion. Returns the solid's name and its measure."),
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
    _not_construction(st, target)
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
            forget(executor, [name])
            what = "removed" if op == "Cut" else "added"
            base = args.get("base")
            along = args.get("direction")
            if not along:
                along = "the sketch normal"
                if base in st.sketches:
                    n = [round(c, 6) + 0.0 for c in sketch_frame(st.sketches[base])[3]]
                    along += f" {n}"
            head = f"addExtrusion{op}: nothing was {what} — the extrusion of {base} along {along}"
            box = _extrusion_box(executor, name, sels, refs, direction, args, target)
            if box is None:
                why = (f"left {target} unchanged, so it was undone (where the extrusion "
                       f"lies could not be checked).")
            elif not _boxes_overlap(box, out["measure"]["bbox"]):
                why = (f"does not reach {target}, so it was undone. Give direction, or "
                       f"from_size to extrude the other way.")
            elif op == "Cut":
                why = (f"lies only where {target} has no material (for example a hole or "
                       f"cut-out already there), so it was undone. Measure {target} and "
                       f"compare it with what you expect before cutting again.")
            else:
                why = (f"lies only inside {target}'s material, so it was undone. Measure "
                       f"{target} and compare it with what you expect before adding again.")
            return {"ok": False, "error": f"{head} {why}"}
    return out


def _extrusion_box(executor, name: str, sels: list, refs: list, direction: str,
                   args: dict, target: str) -> list | None:
    """Bounding box of the plain extrusion a boolean extrusion swept, built and
    removed again; through_all is swept far enough to cross the whole target."""
    if args.get("through_all"):
        try:
            boxes = [raw_box(executor, n) for n in [*refs, target]]
        except Exception:  # noqa: BLE001
            return None
        lo = [min(b[i] for b in boxes) for i in range(3)]
        hi = [max(b[i + 3] for b in boxes) for i in range(3)]
        size, back = math.dist(lo, hi) + 1.0, 0
    else:
        size, back = _size(args)
    probe = f"{name}_probe"
    code = (f"{probe} = model.addExtrusion(shaper_doc, [{', '.join(sels)}], "
            f"{direction}, {size!r}, {back!r})")
    built = _build(executor, probe, "addExtrusion", code, refs, "solid")
    if not built["ok"]:
        return None
    forget(executor, [probe])
    return built["measure"]["bbox"]


def _boxes_overlap(a: list, b: list) -> bool:
    """Boxes as [xmin, xmax, ymin, ymax, zmin, zmax] share some volume, not
    just a face."""
    return all(min(a[i + 1], b[i + 1]) - max(a[i], b[i]) > 1e-6 for i in (0, 2, 4))


@tool("extrude_cut", category="swept",
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


@tool("extrude_fuse", category="swept",
      description="Boss: extrude a face or sketch by `size` and fuse it to the solid `target`.",
      schema={"type": "object", "properties": {
          "base": {"type": "string"}, "target": {"type": "string"},
          "size": {"type": "number"}, "from_size": {"type": "number"},
          "direction": {"type": "string", "enum": list(_AXES)}},
          "required": ["base", "target", "size"]})
def tool_extrude_fuse(executor, args: dict):
    return _boolean_extrusion(executor, args, "Fuse")


def _operands(executor, names, field: str) -> tuple[list, list]:
    st = state(executor)
    if isinstance(names, str):
        names = [names]
    if not isinstance(names, list) or not names:
        raise ValueError(f"{field} must be a non-empty list of names")
    sels = []
    for name in names:
        if name in st.sketches:
            raise ValueError(f"{name!r} is a sketch — make a face from it first "
                             "(make_face) or a solid (extrude)")
        if name not in st.objects:
            raise ValueError(_unknown(st, name))
        _not_construction(st, name)
        sels += _result_selections(st, name)
    return sels, list(names)


def _not_construction(st, name: str) -> None:
    kind = st.objects.get(name)
    if kind in CONSTRUCTION:
        raise ValueError(f"{name!r} is a construction {kind}, not a shape — use it as the origin, "
                         "corner or axis of a primitive, or a plane to sketch on")


def _boolean(executor, args: dict, op: str) -> dict:
    objs, refs = _operands(executor, args.get("objects"), "objects")
    name = new_name(executor, op.lower())
    if op == "Fuse":
        if len(refs) < 2:
            raise ValueError("fuse needs at least two objects")
        code = f"{name} = model.addFuse(shaper_doc, [{', '.join(objs)}], True)"
        return _build(executor, name, "addFuse", code, refs, "boolean")
    tools, trefs = _operands(executor, args.get("tools"), "tools")
    infos = [measure(executor, n) for n in refs]
    before = _amount(infos)
    tool_boxes = [measure(executor, n)["bbox"] for n in trefs] if op == "Cut" else []
    code = f"{name} = model.add{op}(shaper_doc, [{', '.join(objs)}], [{', '.join(tools)}])"
    out = _build(executor, name, f"add{op}", code, refs + trefs, "boolean")
    if out["ok"] and op == "Cut":
        if abs(_amount([out["measure"]]) - before) <= 1e-6 * max(1.0, before):
            if any(_boxes_overlap(t, m["bbox"]) for t in tool_boxes for m in infos):
                why = (f"{', '.join(trefs)} lies only where {', '.join(refs)} has no "
                       f"material (for example a hole or cut-out already there)")
            else:
                why = f"{', '.join(trefs)} does not overlap {', '.join(refs)}"
            out["delta"] = f"{name}: nothing was removed — {why}"
    return out


def _amount(infos: list) -> float:
    """Volume of solids, area of faces."""
    return sum(m.get("volume") or m.get("area") or 0 for m in infos)


_BOOL_SCHEMA = {"type": "object", "properties": {
    "objects": {"type": "array", "items": {"type": "string"}},
    "tools": {"type": "array", "items": {"type": "string"}}},
    "required": ["objects", "tools"]}


@tool("fuse", category="boolean",
      description=(
          "Fuse faces or solids into one: `objects` is two or more published names. "
          "Shared edges between coplanar faces are removed."),
      schema={"type": "object", "properties": {
          "objects": {"type": "array", "items": {"type": "string"}}},
          "required": ["objects"]})
def tool_fuse(executor, args: dict):
    return _boolean(executor, args, "Fuse")


@tool("cut", category="boolean",
      description="Remove the faces or solids `tools` from the faces or solids `objects`.",
      schema=_BOOL_SCHEMA)
def tool_cut(executor, args: dict):
    return _boolean(executor, args, "Cut")


@tool("common", category="boolean",
      description="Keep only where `objects` and `tools` overlap (faces or solids).",
      schema=_BOOL_SCHEMA)
def tool_common(executor, args: dict):
    return _boolean(executor, args, "Common")


@tool("revolve", category="swept",
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
        elif st.objects.get(name) in CONSTRUCTION:
            out[name] = construction_info(executor, name)
        elif name in st.objects:
            out[name] = measure(executor, name)
            results = st.ns[name].feature().results()
            if results and all(r.isConcealed() for r in results):
                out[name]["concealed"] = True
        else:
            out[name] = None
    return {"ok": True, "records": [], "measurements": out}
