from executor.geom.helpers import _check_shape, tool
from executor.geom.results import _shape_info


@tool("check_shape", category="query",
      description='Report whether `shape` is a valid geometry, with a diagnostic message.',
      schema={"type": "object", "properties": {"shape": {"type": "string"}}, "required": ["shape"]},
      arg_types={})
def tool_check_shape(executor, args: dict):
    geompy = executor.namespace["geompy"]
    shape = executor.geom_bindings[args["shape"]]
    is_valid, msg = _check_shape(geompy, shape)
    return {"ok": bool(is_valid), "records": [], "sanity": {"ok": bool(is_valid), "message": msg}}


@tool("list_study_objects", category="query",
      description='List the names of every object currently published in the study.',
      schema={"type": "object", "properties": {}},
      arg_types={})
def tool_list_study_objects(executor, args: dict):
    return {"ok": True, "records": [], "sanity": {"ok": True},
            "names": list(executor.geom_bindings.keys())}


@tool("measure", category="query",
      description='Report bbox/size/centre (mm) of one or more published shapes by name. Use it to check where a shape landed and to see upstream shapes you did not build.',
      schema={"type": "object", "properties": {"names": {"type": "array", "items": {"type": "string"}, "description": "published shape names to measure"}}, "required": ["names"]},
      arg_types={})
def tool_measure(executor, args: dict):
    geompy = executor.namespace["geompy"]
    out = {}
    for name in args["names"]:
        shape = executor.geom_bindings.get(name)
        out[name] = _shape_info(geompy, shape) if shape is not None else None
    return {"ok": True, "records": [], "sanity": {"ok": True}, "measurements": out}
