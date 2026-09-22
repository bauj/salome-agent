"""The geom tool package. Importing it registers every tool."""
from executor.geom.helpers import (  # noqa: F401
    ArgRef, CallRecord, _counters, _gui_call, _next_name, _parse_coords,
    _publish, _resolve, _resolve_axis, _resolve_plane, _resolve_point,
    registry, reset_counters,
)
from executor.geom.results import (  # noqa: F401
    _attach_shape_info, _delta_note, _select_subshapes,
)
from executor.geom.primitives import (  # noqa: F401
    tool_make_box, tool_make_box_two_pnt, tool_make_cylinder, tool_make_sphere,
    tool_make_cone, tool_make_torus,
)
from executor.geom.sketch import (  # noqa: F401
    tool_make_face, tool_make_disk, tool_make_point, tool_make_vector,
    tool_make_line, tool_make_plane, tool_make_circle,
)
from executor.geom.swept import (  # noqa: F401
    tool_make_prism, tool_make_revolution, tool_make_pipe,
    tool_make_thick_solid, tool_make_filling,
)
from executor.geom.booleans import (  # noqa: F401
    tool_fuse, tool_cut, tool_common, tool_partition,
)
from executor.geom.transforms import (  # noqa: F401
    tool_translate, tool_rotate, tool_mirror, tool_scale, tool_multi_translate,
    tool_multi_rotate, tool_offset_shape, tool_position, tool_align,
)
from executor.geom.features import (  # noqa: F401
    tool_fillet_edges, tool_chamfer_edges,
)
from executor.geom.display import (  # noqa: F401
    tool_set_color, tool_set_transparency, tool_set_display_mode,
)
from executor.geom.queries import (  # noqa: F401
    tool_check_shape, tool_list_study_objects, tool_measure,
)
# subshapes.py holds explode/get_subshape/get_subshapes/min_distance/
# basic_properties (tools 39-43 of the pre-split wire order) and is imported
# last so registration order reproduces that order — see task-6-brief ruling 1.
from executor.geom.subshapes import (  # noqa: F401
    tool_explode, tool_get_subshape, tool_get_subshapes, tool_min_distance,
    tool_basic_properties,
)

# Registration order = the order these submodules are imported above, which
# must match the order list_tools() served before the split (guarded by
# tests/fixtures/golden/list_tools.json).
TOOLS = registry.handlers()
TOOL_CATEGORY = registry.categories()
ARG_TYPES = registry.arg_types()
TOOL_SCHEMAS = registry.schemas()
TOOL_DESCRIPTIONS = registry.descriptions()


def _coerce_args(name: str, args: dict) -> dict:
    return registry.coerce_args(name, args)


def list_tools() -> list[dict]:
    return registry.list_tools()


def call_tool(executor, name: str, args: dict) -> dict:
    handler = TOOLS.get(name)
    if handler is None:
        return {"ok": False, "error": f"unknown tool {name!r}"}
    try:
        args = _coerce_args(name, args)
        result = handler(executor, args)
        if result.get("ok"):
            _attach_shape_info(executor, result)
            try:
                note = _delta_note(executor, name, args, result)
            except Exception:
                note = None
            if note:
                result["delta"] = note
        return result
    except Exception as e:
        return {"ok": False, "error": str(e)}
