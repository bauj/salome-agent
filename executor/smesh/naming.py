from executor.smesh.state import (
    REPUBLISH, GeoGroup, group_ref, record, run, shape_ref, state, tool)

_GEOM_TYPES = {"VERTEX": "vertex", "EDGE": "edge", "FACE": "face", "SOLID": "solid"}


@tool("get_subshapes", category="naming",
      description=("Ids of the vertices, edges, faces or solids of `shape`, with each "
                   "one's centre and size (length, area or volume). `where` narrows them: "
                   "by a centre coordinate, by size, or by a plane. Pass the ids to "
                   "name_subshapes."),
      schema={"type": "object", "properties": {
          "shape": {"type": "string"},
          "sub_type": {"type": "string", "enum": ["vertex", "edge", "face", "solid"]},
          "where": {"description": "\"all\", or {\"sorted_by\":\"x|y|z\",\"take\":\"max|min\",\"count\":N}, or {\"by_size\":\"longest|shortest|largest|smallest\",\"count\":N}, or {\"on_plane\":{\"point\":[x,y,z],\"normal\":[x,y,z]},\"state\":\"on\"}"}},
          "required": ["shape", "sub_type"]})
def tool_get_subshapes(executor, args: dict):
    _, owner = shape_ref(executor, args.get("shape"))
    if owner == "shaper":
        from executor.shaper.subshapes import tool_get_subshapes as shaper_get
        return shaper_get(executor, args)
    from executor import geom
    r = geom.call_tool(executor, "get_subshapes", {"where": "all", **args})
    if not r.get("ok"):
        return r
    st = state(executor)
    geompy, shape = st.ns["geompy"], st.ns[args["shape"]]
    col = {"edge": 0, "face": 1, "solid": 2}.get(args.get("sub_type"))
    subs = []
    for i, c in zip(r["ids"], r["centres"]):
        size = (None if col is None else
                round(geompy.BasicProperties(geompy.GetSubShape(shape, [i]))[col], 4))
        subs.append({"id": i, "centre": c, "size": size})
    return {"ok": True, "records": [], "shape": args["shape"], "subshapes": subs}


def _geom_group_code(name, shape, ids, sub_type) -> str:
    return (f"{name} = geompy.CreateGroup({shape}, geompy.ShapeType[{sub_type.upper()!r}])\n"
            f"geompy.UnionIDs({name}, {list(ids)!r})\n"
            f"geompy.addToStudyInFather({shape}, {name}, {name!r})")


def _taken(executor) -> set:
    st = state(executor)
    sh = executor.__dict__.get("shaper_state")
    return (set(st.geo_groups) | set(st.meshes) | set(executor.geom_bindings)
            | (set(sh.objects) if sh else set()))


@tool("name_subshapes", category="naming",
      description=("Publish sub-shapes of `shape` (`ids` from get_subshapes, all of one "
                   "type) as one named group. Use the name in local_size, edge_segments, "
                   "viscous_layers and group_on_geometry."),
      schema={"type": "object", "properties": {
          "shape": {"type": "string"},
          "ids": {"type": "array", "items": {"type": ["string", "integer"]}},
          "name": {"type": "string"}}, "required": ["shape", "ids", "name"]})
def tool_name_subshapes(executor, args: dict):
    from executor.shaper.groups import check_group_name, make_group
    shape, ids, name = args.get("shape"), args.get("ids"), args.get("name")
    expr, owner = shape_ref(executor, shape)
    st = state(executor)
    check_group_name(name, _taken(executor))
    if owner == "shaper":
        r = make_group(executor, shape, ids, name)
        if not r["ok"]:
            return r
        rec = dict(r["records"][0])
        rec["code"] = rec["code"] + f"\n_name({name}, {name!r})\n" + REPUBLISH
        rec["publish"] = None
        st.geo_groups[name] = GeoGroup(shape, "shaper", r["sub_type"])
        return {"ok": True, "records": [rec], "name": name, "sub_type": r["sub_type"],
                "count": r["count"], "measure": r.get("measure")}
    if not isinstance(ids, list) or not ids or not all(isinstance(i, int) for i in ids):
        raise ValueError(f"ids of a GEOM shape are integers from get_subshapes, got {ids!r}")
    geompy = st.ns["geompy"]
    types = {_GEOM_TYPES.get(str(geompy.GetSubShape(st.ns[expr], [i]).GetShapeType()))
             for i in ids}
    if len(types) != 1 or None in types:
        raise ValueError("ids must all be vertices, all edges, all faces or all solids")
    sub_type = types.pop()
    code = _geom_group_code(name, expr, ids, sub_type)
    run(st, code)
    st.geo_groups[name] = GeoGroup(shape, "geom", sub_type)
    props = geompy.BasicProperties(st.ns[name])
    rec = record("CreateGroup", code, out=[name], refs=[shape])
    return {"ok": True, "records": [rec], "name": name, "sub_type": sub_type,
            "count": len(ids), "measure": {"length": props[0], "area": props[1],
                                           "volume": props[2]}}


@tool("shape_info", category="query",
      description="Type, volume, area and face count of published shapes and named groups.",
      schema={"type": "object", "properties": {
          "names": {"type": "array", "items": {"type": "string"}}}, "required": ["names"]})
def tool_shape_info(executor, args: dict):
    from executor.smesh.compute import geom_copy
    st = state(executor)
    geompy = st.ns["geompy"]
    out = {}
    for name in args.get("names") or []:
        expr, _ = shape_ref(executor, name)
        try:
            shape = geom_copy(st, eval(expr, st.ns))
        except LookupError as e:
            out[name] = {"error": str(e)}
            continue
        props = geompy.BasicProperties(shape)
        out[name] = {"type": str(shape.GetShapeType()).lower(),
                     "length": round(props[0], 4), "area": round(props[1], 4),
                     "volume": round(props[2], 4),
                     "faces": geompy.NumberOfFaces(shape)}
    return {"ok": True, "records": [], "shapes": out}
