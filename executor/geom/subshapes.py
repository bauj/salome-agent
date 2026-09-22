from executor.geom.helpers import tool, _next_name, _publish
from executor.geom.results import _centre, _select_subshapes


@tool("explode", category="feature",
      description='Publish every subshape of the given sub_type (edge, face, ...) of `shape` as its own named object, sorted by centre for a deterministic order. Use get_subshapes instead if you only need ids.',
      schema={"type": "object", "properties": {"shape": {"type": "string", "description": "name of the shape to decompose"}, "sub_type": {"type": "string", "enum": ["vertex", "edge", "wire", "face", "shell", "solid"]}}, "required": ["shape", "sub_type"]},
      arg_types={})
def tool_explode(executor, args: dict):
    geompy = executor.namespace["geompy"]
    shape_name = args["shape"]
    shape = executor.geom_bindings[shape_name]
    sub_type = args["sub_type"]
    type_code = int(geompy.ShapeType[sub_type.upper()])
    subs = geompy.SubShapeAllSortedCentres(shape, type_code)
    names = []
    for i, sub in enumerate(subs, start=1):
        nm = f"{shape_name}_{sub_type}_{i}"
        _publish(executor, sub, nm)
        names.append(nm)
    return {
        "ok": True,
        "records": [{
            "fn": "SubShapeAllSortedCentres",
            "args": [{"kind": "ref", "value": shape_name},
                     {"kind": "literal", "value": type_code}],
            "out_vars": names,
            "publish": names,
            "emit": "list_call",
        }],
        "sanity": {"ok": True},
        "names": names,
    }


@tool("get_subshape", category="feature",
      description='Build one new named shape from a list of subshape ids of `shape` (e.g. a group of faces). Get the ids from get_subshapes or explode.',
      schema={"type": "object", "properties": {"shape": {"type": "string"}, "ids": {"type": "array", "items": {"type": "integer"}, "description": "subshape ids from explode or get_subshapes"}}, "required": ["shape", "ids"]},
      arg_types={})
def tool_get_subshape(executor, args: dict):
    geompy = executor.namespace["geompy"]
    shape_name = args["shape"]
    shape = executor.geom_bindings[shape_name]
    ids = [int(i) for i in args["ids"]]
    sub = geompy.GetSubShape(shape, ids)
    name = _next_name(executor, "sub")
    _publish(executor, sub, name)
    return {
        "ok": True,
        "records": [{
            "fn": "GetSubShape",
            "args": [{"kind": "ref", "value": shape_name},
                     {"kind": "literal", "value": ids}],
            "out_vars": [name],
            "publish": [name],
        }],
        "sanity": {"ok": True},
    }


@tool("get_subshapes", category="query",
      description='Find subshape ids of `shape` without changing geometry. Returns {ids, centres}. Pass the ids to fillet_edges, chamfer_edges or get_subshape. where = "all", or select by a centre axis (sorted_by/take/count), by size (by_size/count), or by a plane (on_plane/state).',
      schema={"type": "object", "properties": {"shape": {"type": "string"}, "sub_type": {"type": "string", "enum": ["vertex", "edge", "wire", "face", "shell", "solid"]}, "where": {"description": "\"all\", or {\"sorted_by\":\"x|y|z\",\"take\":\"max|min\",\"count\":N}, or {\"by_size\":\"longest|shortest|largest|smallest\",\"count\":N}, or {\"on_plane\":{\"point\":[x,y,z],\"normal\":[x,y,z]},\"state\":\"on\"}"}}, "required": ["shape", "sub_type", "where"]},
      arg_types={})
def tool_get_subshapes(executor, args: dict):
    geompy = executor.namespace["geompy"]
    shape = executor.geom_bindings[args["shape"]]
    type_code = int(geompy.ShapeType[args["sub_type"].upper()])
    try:
        selected = _select_subshapes(geompy, shape, type_code, args["where"])
    except ValueError as e:
        return {"ok": False, "error": f"get_subshapes: {e}"}
    return {
        "ok": True,
        "records": [],
        "sanity": {"ok": True},
        "ids": [i for i, _ in selected],
        "centres": [_centre(geompy, s) for _, s in selected],
    }


@tool("min_distance", category="query",
      description='Report the shortest distance (mm) between two published shapes.',
      schema={"type": "object", "properties": {"shape1": {"type": "string"}, "shape2": {"type": "string"}}, "required": ["shape1", "shape2"]},
      arg_types={})
def tool_min_distance(executor, args: dict):
    geompy = executor.namespace["geompy"]
    s1 = executor.geom_bindings[args["shape1"]]
    s2 = executor.geom_bindings[args["shape2"]]
    d = geompy.MinDistance(s1, s2)
    if isinstance(d, (tuple, list)):
        d = d[0]
    return {"ok": True, "records": [], "sanity": {"ok": True}, "distance": round(d, 4)}


@tool("basic_properties", category="query",
      description='Report [length, area, volume] of `shape` (mm): length for an edge/wire, area for a face/shell, volume for a solid; the others are 0.',
      schema={"type": "object", "properties": {"shape": {"type": "string"}}, "required": ["shape"]},
      arg_types={})
def tool_basic_properties(executor, args: dict):
    geompy = executor.namespace["geompy"]
    shape = executor.geom_bindings[args["shape"]]
    length, area, volume = geompy.BasicProperties(shape)
    return {"ok": True, "records": [], "sanity": {"ok": True},
            "properties": {"length": round(length, 4), "area": round(area, 4),
                           "volume": round(volume, 4)}}
