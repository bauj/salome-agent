from executor.shaper.features import _build, _operands
from executor.shaper.primitives import _num
from executor.shaper.state import SUB_TYPES, forget, new_name, state, subshapes, tool

_MAX_EXPLODE = 50
_USED = " `shape` is used up: refer to the result by its new name."

_WHERE = {"description": (
    "\"all\"; {\"sorted_by\": \"x|y|z\", \"take\": \"max|min\", \"count\": N}; "
    "{\"by_size\": \"longest|shortest|largest|smallest\", \"count\": N}; or "
    "{\"on_plane\": {\"point\": [x, y, z], \"normal\": [x, y, z]}, \"state\": "
    "\"on|in|out|onin|onout\"} (in = the side the normal points away from)")}
_IDS = {"description": "\"all\" or a list of ids from get_subshapes (\"E3\", \"F2\", ...)",
        "anyOf": [{"type": "string", "enum": ["all"]},
                  {"type": "array", "items": {"type": "string"}}]}


def _shape_name(executor, args: dict, field: str = "shape") -> str:
    name = args.get(field)
    if not isinstance(name, str):
        raise ValueError(f"{field} must be one published name, got {name!r}")
    _operands(executor, [name], field)
    return name


def _plane_test(where: dict):
    op = where.get("on_plane")
    if not isinstance(op, dict):
        raise ValueError("on_plane must be {point: [x, y, z], normal: [x, y, z]}")
    p0 = [_num(v, "on_plane point") for v in op.get("point") or []]
    n = [_num(v, "on_plane normal") for v in op.get("normal") or []]
    if len(p0) != 3 or len(n) != 3 or not any(n):
        raise ValueError("on_plane needs point [x, y, z] and a non-zero normal [x, y, z]")
    norm = sum(c * c for c in n) ** 0.5
    side = lambda p: sum((p[i] - p0[i]) * n[i] for i in range(3)) / norm
    state_ = where.get("state", "on")
    tests = {"on": lambda d: abs(d) < 1e-6, "in": lambda d: d < -1e-6,
             "out": lambda d: d > 1e-6, "onin": lambda d: d < 1e-6,
             "onout": lambda d: d > -1e-6}
    if state_ not in tests:
        raise ValueError(f"state must be one of {', '.join(tests)}, got {state_!r}")
    return lambda pts: all(tests[state_](side(p)) for p in pts)


def select(subs: list, where, points_of) -> list:
    """The subs a `where` predicate picks. points_of(sub) gives points on a
    sub-shape (its vertices and centre of mass) for the on_plane test."""
    if isinstance(where, str) and where.strip().startswith("{"):
        import json
        where = json.loads(where)
    if where in (None, "all"):
        return list(subs)
    if not isinstance(where, dict):
        raise ValueError(f"where must be \"all\" or a predicate object, got {where!r}")
    count = lambda: _count(where.get("count"))
    if "sorted_by" in where:
        axis = {"x": 0, "y": 1, "z": 2}.get(where["sorted_by"]) \
            if isinstance(where["sorted_by"], str) else None
        take = where.get("take", "max")
        if axis is None or take not in ("max", "min"):
            raise ValueError("sorted_by must be x, y or z and take max or min")
        return sorted(subs, key=lambda u: u.centre[axis], reverse=take == "max")[:count()]
    if "by_size" in where:
        kind = where["by_size"]
        if kind not in ("longest", "shortest", "largest", "smallest"):
            raise ValueError("by_size must be longest, shortest, largest or smallest")
        return sorted(subs, key=lambda u: u.size,
                      reverse=kind in ("longest", "largest"))[:count()]
    if "on_plane" in where:
        test = _plane_test(where)
        return [u for u in subs if test(points_of(u))]
    raise ValueError(f"unknown where {where!r}")


def _count(value) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value != int(value) \
            or value < 1:
        raise ValueError(f"count must be a whole number of at least 1, got {value!r}")
    return int(value)


def _points(sub) -> list:
    import GeomAPI
    from GeomAlgoAPI import GeomAlgoAPI_ShapeTools as T
    xyz = lambda p: [p.x(), p.y(), p.z()]
    pts = [xyz(GeomAPI.GeomAPI_Vertex(v).point())
           for v in sub.shape.subShapes(GeomAPI.GeomAPI_Shape.VERTEX)]
    if sub.shape.shapeType() != GeomAPI.GeomAPI_Shape.VERTEX:
        pts.append(xyz(T.centreOfMass(sub.shape)))
    return pts


def _pick(executor, name: str, sub_type: str, ids, where) -> list:
    """The Subs named by `ids` ("all" or a list) or picked by `where`."""
    subs = subshapes(executor, name, sub_type)
    if ids is not None and where is not None:
        raise ValueError("give either the ids or where, not both")
    if ids is None and where is None:
        raise ValueError(f"select the {sub_type}s: ids from get_subshapes, \"all\", "
                         "or a where predicate")
    if where is not None:
        picked = select(subs, where, _points)
    elif ids == "all":
        picked = subs
    else:
        if not isinstance(ids, list) or not ids or \
                not all(isinstance(i, str) for i in ids):
            raise ValueError(f"ids must be \"all\" or a list like [\"E1\", \"E4\"], got {ids!r}")
        by_id = {u.id: u for u in subs}
        bad = [i for i in ids if i not in by_id]
        if bad:
            raise ValueError(f"{name} has no {sub_type} {', '.join(bad)} — it has "
                             f"{subs[0].id}..{subs[-1].id}" if subs else
                             f"{name} has no {sub_type}")
        picked = [by_id[i] for i in dict.fromkeys(ids)]
    if not picked:
        raise ValueError(f"no {sub_type} of {name} matches {where!r}")
    return picked


def _sels(sub_type: str, picked: list) -> str:
    return ", ".join(f'model.selection("{sub_type.upper()}", {u.naming!r})' for u in picked)


@tool("get_subshapes", category="query",
      description=("Ids of the vertices, edges, faces or solids of `shape` (\"E1\", "
                   "\"F3\", ...), numbered by centre so the same shape always gives the "
                   "same ids, with each one's centre and size (length, area or volume). "
                   "`where` narrows them: by a centre coordinate, by size, or by a plane "
                   "-- state \"on\" keeps only those lying in it, e.g. a rim circle and not "
                   "the seams that cross it. Pass the ids to fillet_edges, chamfer_edges, "
                   "make_thick_solid, get_subshape, or create_sketch (a face)."),
      schema={"type": "object", "properties": {
          "shape": {"type": "string"},
          "sub_type": {"type": "string", "enum": list(SUB_TYPES)},
          "where": _WHERE}, "required": ["shape", "sub_type"]})
def tool_get_subshapes(executor, args: dict):
    name = _shape_name(executor, args)
    subs = subshapes(executor, name, args.get("sub_type"))
    picked = select(subs, args.get("where", "all"), _points)
    return {"ok": True, "records": [], "shape": name,
            "subshapes": [{"id": u.id, "centre": u.centre, "size": u.size} for u in picked]}


@tool("get_subshape", category="feature",
      description=("A new shape copied from sub-shapes of `shape` (e.g. some of its "
                   "faces): `ids` from get_subshapes, all of one type. `shape` stays."),
      schema={"type": "object", "properties": {
          "shape": {"type": "string"},
          "ids": {"type": "array", "items": {"type": "string"}}},
          "required": ["shape", "ids"]})
def tool_get_subshape(executor, args: dict):
    name = _shape_name(executor, args)
    ids = args.get("ids")
    if not isinstance(ids, list) or not ids or not all(isinstance(i, str) for i in ids):
        raise ValueError(f"ids must be a list like [\"F1\", \"F2\"], got {ids!r}")
    letters = {i[:1] for i in ids}
    types = {v: k for k, v in SUB_TYPES.items()}
    if len(letters) != 1 or next(iter(letters)) not in types:
        raise ValueError("ids must all be of one type (all E.., all F.., ...)")
    sub_type = types[next(iter(letters))]
    picked = _pick(executor, name, sub_type, ids, None)
    out = new_name(executor, "copy")
    code = f"{out} = model.addCopy(shaper_doc, [{_sels(sub_type, picked)}], 1)"
    return _build(executor, out, "addCopy", code, [name], "copy")


@tool("explode", category="feature",
      description=(f"Copy every sub-shape of `sub_type` of `shape` as its own named "
                   f"shape, in id order (at most {_MAX_EXPLODE}). `shape` stays. Use "
                   "get_subshapes when ids are enough."),
      schema={"type": "object", "properties": {
          "shape": {"type": "string"},
          "sub_type": {"type": "string", "enum": list(SUB_TYPES)}},
          "required": ["shape", "sub_type"]})
def tool_explode(executor, args: dict):
    name = _shape_name(executor, args)
    sub_type = args.get("sub_type")
    subs = subshapes(executor, name, sub_type)
    if len(subs) > _MAX_EXPLODE:
        raise ValueError(f"{name} has {len(subs)} {sub_type}s, more than {_MAX_EXPLODE}: "
                         "pick some with get_subshapes and get_subshape")
    records, names = [], []
    for u in subs:
        out = new_name(executor, f"{name}_{sub_type}")
        code = f"{out} = model.addCopy(shaper_doc, [{_sels(sub_type, [u])}], 1)"
        r = _build(executor, out, "addCopy", code, [name], "copy")
        if not r["ok"]:
            forget(executor, [n["name"] for n in names])
            return r
        records += r["records"]
        names.append({"name": out, "id": u.id})
    return {"ok": True, "records": records, "names": names}


def _edge_feature(executor, args: dict, prefix: str, fn: str, call) -> dict:
    name = _shape_name(executor, args)
    picked = _pick(executor, name, "edge", args.get("edges"), args.get("where"))
    out = new_name(executor, prefix)
    code = f"{out} = model.{fn}(shaper_doc, [{_sels('edge', picked)}], {call})"
    r = _build(executor, out, fn, code, [name], "solid")
    if r["ok"]:
        r["edges"] = [u.id for u in picked]
    return r


@tool("fillet_edges", category="feature",
      description=("Round edges of `shape` with `radius`: `edges` (ids from "
                   "get_subshapes, or \"all\") or a `where` predicate (as get_subshapes). "
                   "Fails if the radius is too large for the shape." + _USED),
      schema={"type": "object", "properties": {
          "shape": {"type": "string"}, "radius": {"type": "number"},
          "edges": _IDS, "where": _WHERE}, "required": ["shape", "radius"]})
def tool_fillet_edges(executor, args: dict):
    r = _num(args.get("radius"), "radius", positive=True)
    return _edge_feature(executor, args, "fillet", "addFillet", repr(r))


@tool("chamfer_edges", category="feature",
      description=("Bevel edges of `shape` by distances d1 and d2 (default d1) along its "
                   "two faces: `edges` (ids or \"all\") or a `where` predicate." + _USED),
      schema={"type": "object", "properties": {
          "shape": {"type": "string"}, "d1": {"type": "number"}, "d2": {"type": "number"},
          "edges": _IDS, "where": _WHERE}, "required": ["shape", "d1"]})
def tool_chamfer_edges(executor, args: dict):
    d1 = _num(args.get("d1"), "d1", positive=True)
    d2 = d1 if args.get("d2") is None else _num(args["d2"], "d2", positive=True)
    return _edge_feature(executor, args, "chamfer", "addChamfer", f"True, {d1!r}, {d2!r}")


@tool("make_thick_solid", category="feature",
      description=("Hollow the solid `shape` into a wall of `thickness`, open at `faces` "
                   "(ids from get_subshapes, or a `where` predicate; at least one). "
                   "inside (default true) keeps the outer size; false grows the part by "
                   "the thickness." + _USED),
      schema={"type": "object", "properties": {
          "shape": {"type": "string"}, "thickness": {"type": "number"},
          "faces": {"type": "array", "items": {"type": "string"}}, "where": _WHERE,
          "inside": {"type": "boolean"}}, "required": ["shape", "thickness"]})
def tool_make_thick_solid(executor, args: dict):
    name = _shape_name(executor, args)
    t = _num(args.get("thickness"), "thickness", positive=True)
    inside = args.get("inside", True)
    if not isinstance(inside, bool):
        raise ValueError(f"inside must be true or false, got {inside!r}")
    solids = subshapes(executor, name, "solid")
    if len(solids) != 1:
        raise ValueError(f"{name} holds {len(solids)} solids; make_thick_solid takes one")
    picked = _pick(executor, name, "face", args.get("faces"), args.get("where"))
    out = new_name(executor, "thickness")
    code = (f"{out} = model.addHollowedSolid(shaper_doc, "
            f"model.selection(\"SOLID\", {solids[0].naming!r}), {t!r}, "
            f"[{_sels('face', picked)}], {inside})")
    return _build(executor, out, "addHollowedSolid", code, [name], "solid")


def _tools_sels(executor, names, field: str) -> tuple[list, list]:
    """Cutting tools: shapes, or construction planes."""
    st = state(executor)
    if isinstance(names, str):
        names = [names]
    if not isinstance(names, list) or not names or not all(isinstance(n, str) for n in names):
        raise ValueError(f"{field} must be a non-empty list of names")
    sels, refs = [], []
    for n in names:
        if st.objects.get(n) == "plane":
            sels.append(f'model.selection("FACE", {n!r})')
            refs.append(n)
        else:
            s, r = _operands(executor, [n], field)
            sels += s
            refs += r
    return sels, refs


@tool("partition", category="boolean",
      description=("Cut `shapes` (and `tools`, construction planes allowed) by each other "
                   "and keep every piece, sharing the faces where they meet: one result. "
                   "The inputs are used up."),
      schema={"type": "object", "properties": {
          "shapes": {"type": "array", "items": {"type": "string"}},
          "tools": {"type": "array", "items": {"type": "string"}}},
          "required": ["shapes"]})
def tool_partition(executor, args: dict):
    sels, refs = _operands(executor, args.get("shapes"), "shapes")
    if args.get("tools"):
        t_sels, t_refs = _tools_sels(executor, args["tools"], "tools")
        sels, refs = sels + t_sels, refs + t_refs
    if len(sels) < 2:
        raise ValueError("partition needs at least two shapes, or a shape and a tool")
    out = new_name(executor, "partition")
    code = f"{out} = model.addPartition(shaper_doc, [{', '.join(sels)}])"
    return _build(executor, out, "addPartition", code, refs, "solid")


@tool("split", category="boolean",
      description=("Split `objects` into pieces with `tools` (shapes or construction "
                   "planes): every piece is kept in one result; the tools are not. The "
                   "objects are used up."),
      schema={"type": "object", "properties": {
          "objects": {"type": "array", "items": {"type": "string"}},
          "tools": {"type": "array", "items": {"type": "string"}}},
          "required": ["objects", "tools"]})
def tool_split(executor, args: dict):
    sels, refs = _operands(executor, args.get("objects"), "objects")
    t_sels, t_refs = _tools_sels(executor, args.get("tools"), "tools")
    out = new_name(executor, "split")
    code = (f"{out} = model.addSplit(shaper_doc, [{', '.join(sels)}], "
            f"[{', '.join(t_sels)}])")
    return _build(executor, out, "addSplit", code, refs + t_refs, "solid")

