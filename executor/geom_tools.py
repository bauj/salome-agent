from typing import Any, Callable, TypedDict


class ArgRef(TypedDict):
    kind: str    # "literal" | "ref" | "list" | "study"
    value: Any


class CallRecord(TypedDict):
    fn: str
    args: list[ArgRef]
    out_vars: list[str]
    publish: list[str] | None


_counters: dict[str, int] = {}


def _next_name(prefix: str) -> str:
    _counters[prefix] = _counters.get(prefix, 0) + 1
    return f"{prefix}_{_counters[prefix]}"


def _resolve(executor, arg: ArgRef):
    if arg["kind"] == "literal":
        return arg["value"]
    if arg["kind"] == "ref":
        return executor.geom_bindings[arg["value"]]
    if arg["kind"] == "list":
        return [_resolve(executor, a) for a in arg["value"]]
    if arg["kind"] == "study":
        return executor.namespace["find_object"](arg["value"])
    raise ValueError(f"unknown ArgRef kind {arg['kind']!r}")


def _publish(executor, shape, name: str):
    geompy = executor.namespace["geompy"]
    geompy.addToStudy(shape, name)
    executor.geom_bindings[name] = shape


def tool_make_box(executor, args: dict):
    geompy = executor.namespace["geompy"]
    box = geompy.MakeBoxDXDYDZ(args["dx"], args["dy"], args["dz"])
    name = _next_name("box")
    _publish(executor, box, name)
    return {
        "ok": True,
        "records": [{
            "fn": "MakeBoxDXDYDZ",
            "args": [{"kind": "literal", "value": args["dx"]},
                     {"kind": "literal", "value": args["dy"]},
                     {"kind": "literal", "value": args["dz"]}],
            "out_vars": [name],
            "publish": [name],
        }],
        "sanity": {"ok": True},
    }


def tool_fillet_edges(executor, args: dict):
    geompy = executor.namespace["geompy"]
    shape = executor.geom_bindings[args["shape"]]
    edge_ids = (
        list(range(1, len(geompy.SubShapeAllIDs(shape, geompy.ShapeType["EDGE"])) + 1))
        if args["edges"] == "all" else args["edges"]
    )
    filleted = geompy.MakeFilletEdges(shape, args["radius"], edge_ids)
    # theReturnStatus=1 -> (isValid, description); the default (0) returns
    # a bare bool, which broke unpacking here until caught in testing.
    is_valid, msg = geompy.CheckShape(filleted, True, 1)
    if not is_valid:
        # Not published yet at this point, so there is nothing to clean
        # up — the invariant decision 16 wants (no orphan survives a
        # failed call) holds by construction rather than by an explicit
        # delete step.
        return {"ok": False, "error": f"fillet_edges: CheckShape failed on result ({msg})"}
    name = _next_name("fillet")
    _publish(executor, filleted, name)
    return {
        "ok": True,
        "records": [{
            "fn": "MakeFilletEdges",
            "args": [{"kind": "ref", "value": args["shape"]},
                     {"kind": "literal", "value": args["radius"]},
                     {"kind": "literal", "value": edge_ids}],
            "out_vars": [name],
            "publish": [name],
        }],
        "sanity": {"ok": True},
    }


def tool_check_shape(executor, args: dict):
    geompy = executor.namespace["geompy"]
    shape = executor.geom_bindings[args["shape"]]
    is_valid, msg = geompy.CheckShape(shape, True, 1)
    return {"ok": bool(is_valid), "records": [], "sanity": {"ok": bool(is_valid), "message": msg}}


def tool_list_study_objects(executor, args: dict):
    return {"ok": True, "records": [], "sanity": {"ok": True},
            "names": list(executor.geom_bindings.keys())}


# Registry: tool name -> handler(executor, args) -> ToolResult.
# make_cylinder/boolean_*/chamfer_edges/finish_stage follow the same shape
# as tool_make_box/tool_fillet_edges above and are added the same way —
# not stubbed here, added incrementally as later stages need them.
TOOLS: dict[str, Callable] = {
    "make_box": tool_make_box,
    "fillet_edges": tool_fillet_edges,
    "check_shape": tool_check_shape,
    "list_study_objects": tool_list_study_objects,
}

# Per-tool arg coercion: LLM tool-call args arrive JSON-typed (numbers can
# come back as strings, e.g. {"height": "40"}), but geompy does not coerce —
# a string arg is interpreted as a notebook variable-name reference and
# fails with a misleading RuntimeError ("Variable with name '20' doesn't
# exist!!!"). Add an entry here when a new tool takes numeric args.
ARG_TYPES: dict[str, dict[str, Callable]] = {
    "make_box": {"dx": float, "dy": float, "dz": float},
    "fillet_edges": {"radius": float},
}

# JSON Schema per tool, sent to the LLM as each function's "parameters" so
# it knows argument names/types/required-ness. Without this the LLM has to
# guess shapes — real bug found live: it returned a numeric-looking string
# for a param that needed to stay a list, and codegen silently mis-iterated
# over the string's characters instead of erroring.
TOOL_SCHEMAS: dict[str, dict] = {
    "make_box": {
        "type": "object",
        "properties": {
            "dx": {"type": "number"}, "dy": {"type": "number"}, "dz": {"type": "number"},
        },
        "required": ["dx", "dy", "dz"],
    },
    "fillet_edges": {
        "type": "object",
        "properties": {
            "shape": {"type": "string", "description": "published name of the shape to fillet"},
            "edges": {
                "description": "\"all\" or a list of edge subshape indices",
                "oneOf": [{"type": "string", "enum": ["all"]},
                          {"type": "array", "items": {"type": "integer"}}],
            },
            "radius": {"type": "number"},
        },
        "required": ["shape", "edges", "radius"],
    },
    "check_shape": {
        "type": "object",
        "properties": {"shape": {"type": "string"}},
        "required": ["shape"],
    },
    "list_study_objects": {"type": "object", "properties": {}},
}


def _coerce_args(name: str, args: dict) -> dict:
    types = ARG_TYPES.get(name)
    if not types:
        return args
    coerced = dict(args)
    for key, cast in types.items():
        if key in coerced:
            coerced[key] = cast(coerced[key])
    return coerced


def call_tool(executor, name: str, args: dict) -> dict:
    handler = TOOLS.get(name)
    if handler is None:
        return {"ok": False, "error": f"unknown tool {name!r}"}
    try:
        args = _coerce_args(name, args)
        return handler(executor, args)
    except Exception as e:
        return {"ok": False, "error": str(e)}


def list_tools() -> list[dict]:
    return [{"name": n, "parameters": TOOL_SCHEMAS.get(n, {"type": "object", "properties": {}})}
            for n in TOOLS]
