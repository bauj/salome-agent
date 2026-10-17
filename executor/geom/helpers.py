from typing import Any, TypedDict
from typing_extensions import NotRequired


from executor.registry import ToolRegistry

_VALID_CATEGORIES = {"primitive", "sketch", "swept", "boolean", "transform",
                     "feature", "display", "query"}
registry = ToolRegistry(valid_categories=_VALID_CATEGORIES)
tool = registry.tool


class ArgRef(TypedDict):
    kind: str    # "literal" | "ref" | "list" | "study" | "salomeds_color"
    value: Any


class CallRecord(TypedDict):
    fn: str
    args: list[ArgRef]
    out_vars: list[str]
    publish: list[str] | None
    emit: NotRequired[str]        # "geompy_call" (default) | "method_call" | "gui_call"
    recv: NotRequired[str]        # ref name of the receiver, for method_call / gui_call


_counters: dict[str, int] = {}
# Names a replayed tool call must reuse, in allocation order (Task 1 of the
# agentic-chat plan). call_tool fills it for one call and clears it after.
_pinned: list[str] = []


def _executor_counters(executor=None) -> dict[str, int]:
    if executor is not None and hasattr(executor, "_counters"):
        return executor._counters
    return _counters


def _executor_pinned(executor=None) -> list[str]:
    if executor is not None and hasattr(executor, "_pinned") and executor._pinned:
        return executor._pinned
    if _pinned:
        return _pinned
    if executor is not None and hasattr(executor, "_pinned"):
        return executor._pinned
    return _pinned


def reset_counters(executor=None) -> None:
    _counters.clear()
    if executor is not None and hasattr(executor, "_counters"):
        executor._counters.clear()


def _next_name(executor, prefix: str) -> str:
    counters = _executor_counters(executor)
    pinned = _executor_pinned(executor)
    stem = f"{executor.name_prefix}{prefix}_"
    for i, name in enumerate(pinned):
        suffix = name[len(stem):]
        if name.startswith(stem) and suffix.isdigit():
            del pinned[i]
            counters[prefix] = max(counters.get(prefix, 0), int(suffix))
            return name
    counters[prefix] = counters.get(prefix, 0) + 1
    return f"{stem}{counters[prefix]}"


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


def _check_shape(geompy, shape) -> tuple:
    """(is_valid, message) from CheckShape with the geometry check. That check
    can die in CORBA (MARSHAL_InvalidEnumValue) on an invalid shape; the
    topology-only check still reports what is wrong."""
    try:
        return geompy.CheckShape(shape, True, 1)
    except Exception as e:
        is_valid, msg = geompy.CheckShape(shape, False, 1)
        return is_valid, f"{msg}\n(geometry check failed: {e}; topology checked only)".strip()


def _publish(executor, shape, name: str):
    geompy = executor.namespace["geompy"]
    geompy.addToStudy(shape, name)
    executor.geom_bindings[name] = shape


def _gui_call(executor, shape, method: str, value) -> None:
    """Best-effort GUI display call (setColor via ImportComponentGUI, etc.).

    Everything is guarded: outside a GUI process ImportComponentGUI can
    raise, and the whole point of these calls is that they are optional
    (the generated result.py wraps them in try/except too).
    """
    try:
        import salome
        gg = executor.namespace.get("gg")
        if gg is None:
            gg = executor.namespace["gg"] = salome.ImportComponentGUI("GEOM")
        getattr(gg, method)(salome.ObjectToID(shape), value)
    except Exception:
        pass


def _parse_coords(arg):
    """A 3-number coordinate literal -> (x, y, z) floats, else None.

    Accepts [x, y, z], (x, y, z), "x,y,z" and "(x, y, z)" — a real LLM passes a
    point/vector argument as any of these instead of the name of a shape it
    built, which otherwise fails with an opaque KeyError on geom_bindings.
    """
    if isinstance(arg, (list, tuple)) and len(arg) == 3:
        parts = arg
    elif isinstance(arg, str) and "," in arg:
        parts = arg.strip(" []()").split(",")
    else:
        return None
    try:
        x, y, z = (float(p) for p in parts)
    except (TypeError, ValueError):
        return None
    return x, y, z


def _resolve_axis(executor, arg, records: list) -> str:
    """Resolve an axis arg to a ref name.

    "OX"/"OY"/"OZ" or a coordinate literal -> emit a MakeVectorDXDYDZ CallRecord
    and return its name. Otherwise arg is a ref name — return it unchanged.
    """
    coords = _parse_coords(arg)
    if coords is not None:
        dx, dy, dz = coords
        geompy = executor.namespace["geompy"]
        name = _next_name(executor, "vec")
        executor.geom_bindings[name] = geompy.MakeVectorDXDYDZ(dx, dy, dz)
        records.append({
            "fn": "MakeVectorDXDYDZ",
            "args": [{"kind": "literal", "value": dx},
                     {"kind": "literal", "value": dy},
                     {"kind": "literal", "value": dz}],
            "out_vars": [name],
            "publish": None,
        })
        return name
    if arg == "OX":
        geompy = executor.namespace["geompy"]
        vec = geompy.MakeVectorDXDYDZ(1, 0, 0)
        name = _next_name(executor, "vec")
        executor.geom_bindings[name] = vec
        records.append({
            "fn": "MakeVectorDXDYDZ",
            "args": [{"kind": "literal", "value": 1},
                     {"kind": "literal", "value": 0},
                     {"kind": "literal", "value": 0}],
            "out_vars": [name],
            "publish": None,
        })
        return name
    elif arg == "OY":
        geompy = executor.namespace["geompy"]
        vec = geompy.MakeVectorDXDYDZ(0, 1, 0)
        name = _next_name(executor, "vec")
        executor.geom_bindings[name] = vec
        records.append({
            "fn": "MakeVectorDXDYDZ",
            "args": [{"kind": "literal", "value": 0},
                     {"kind": "literal", "value": 1},
                     {"kind": "literal", "value": 0}],
            "out_vars": [name],
            "publish": None,
        })
        return name
    elif arg == "OZ":
        geompy = executor.namespace["geompy"]
        vec = geompy.MakeVectorDXDYDZ(0, 0, 1)
        name = _next_name(executor, "vec")
        executor.geom_bindings[name] = vec
        records.append({
            "fn": "MakeVectorDXDYDZ",
            "args": [{"kind": "literal", "value": 0},
                     {"kind": "literal", "value": 0},
                     {"kind": "literal", "value": 1}],
            "out_vars": [name],
            "publish": None,
        })
        return name
    else:
        # It's a ref name
        return arg


def _resolve_plane(executor, arg, records: list) -> str:
    """Resolve a plane arg to a ref name.

    If arg is "XOY", "YOZ", or "ZOX", emit a MakePlane CallRecord and return its name.
    Otherwise, arg is a ref name — return it unchanged.
    """
    if arg == "XOY":
        geompy = executor.namespace["geompy"]
        origin = geompy.MakeVertex(0, 0, 0)
        normal = geompy.MakeVectorDXDYDZ(0, 0, 1)
        plane = geompy.MakePlane(origin, normal, 2000)
        name = _next_name(executor, "plane")
        executor.geom_bindings[name] = plane
        records.append({
            "fn": "MakePlane",
            "args": [{"kind": "literal", "value": 0},
                     {"kind": "literal", "value": 0},
                     {"kind": "literal", "value": 0},
                     {"kind": "literal", "value": 0},
                     {"kind": "literal", "value": 0},
                     {"kind": "literal", "value": 1},
                     {"kind": "literal", "value": 2000}],
            "out_vars": [name],
            "publish": None,
        })
        return name
    elif arg == "YOZ":
        geompy = executor.namespace["geompy"]
        origin = geompy.MakeVertex(0, 0, 0)
        normal = geompy.MakeVectorDXDYDZ(1, 0, 0)
        plane = geompy.MakePlane(origin, normal, 2000)
        name = _next_name(executor, "plane")
        executor.geom_bindings[name] = plane
        records.append({
            "fn": "MakePlane",
            "args": [{"kind": "literal", "value": 0},
                     {"kind": "literal", "value": 0},
                     {"kind": "literal", "value": 0},
                     {"kind": "literal", "value": 1},
                     {"kind": "literal", "value": 0},
                     {"kind": "literal", "value": 0},
                     {"kind": "literal", "value": 2000}],
            "out_vars": [name],
            "publish": None,
        })
        return name
    elif arg == "ZOX":
        geompy = executor.namespace["geompy"]
        origin = geompy.MakeVertex(0, 0, 0)
        normal = geompy.MakeVectorDXDYDZ(0, 1, 0)
        plane = geompy.MakePlane(origin, normal, 2000)
        name = _next_name(executor, "plane")
        executor.geom_bindings[name] = plane
        records.append({
            "fn": "MakePlane",
            "args": [{"kind": "literal", "value": 0},
                     {"kind": "literal", "value": 0},
                     {"kind": "literal", "value": 0},
                     {"kind": "literal", "value": 0},
                     {"kind": "literal", "value": 1},
                     {"kind": "literal", "value": 0},
                     {"kind": "literal", "value": 2000}],
            "out_vars": [name],
            "publish": None,
        })
        return name
    else:
        # It's a ref name
        return arg


def _resolve_point(executor, arg, records: list) -> str:
    """Resolve a point arg to a ref name.

    "O" or a coordinate literal ([x,y,z] / "x,y,z" / "(x, y, z)") -> emit a
    MakeVertex CallRecord and return its name. Otherwise arg is a ref name —
    return it unchanged.
    """
    coords = (0, 0, 0) if arg == "O" else _parse_coords(arg)
    if coords is not None:
        x, y, z = coords
        geompy = executor.namespace["geompy"]
        name = _next_name(executor, "pt")
        executor.geom_bindings[name] = geompy.MakeVertex(x, y, z)
        records.append({
            "fn": "MakeVertex",
            "args": [{"kind": "literal", "value": x},
                     {"kind": "literal", "value": y},
                     {"kind": "literal", "value": z}],
            "out_vars": [name],
            "publish": None,
        })
        return name
    return arg


def _check_sub_ids(tool: str, geompy, shape, shape_name: str, ids, kind: str) -> None:
    """Reject ids that are not `kind` subshapes of `shape` before OCC sees
    them: OCC answers a wrong id with "TopoDS::Edge" or "Offset driver failed",
    and models guess 0-based indexes (task 218). The error carries the real ids."""
    valid = list(geompy.SubShapeAllIDs(shape, geompy.ShapeType[kind]))
    bad = [i for i in ids if i not in valid]
    if bad:
        shown = ", ".join(map(str, valid[:24])) + (", ..." if len(valid) > 24 else "")
        raise ValueError(
            f"{tool}: {bad} are not {kind.lower()} ids of {shape_name} — its "
            f"{kind.lower()} ids are [{shown}]. Ids are GEOM subshape ids (from "
            f"get_subshapes), not 0-based indexes.")
