"""Compatibility shim — the tools moved to executor.geom.

Kept for one commit so the ~1800-line test file and the five production
import sites stay untouched while the split is proven. Deleted in Task 7.
"""
from executor.geom import *  # noqa: F401,F403
from executor.geom import (  # noqa: F401 — private names star-import skips
    _attach_shape_info, _coerce_args, _counters, _delta_note, _gui_call,
    _next_name, _parse_coords, _publish, _resolve, _resolve_axis,
    _resolve_plane, _resolve_point, _select_subshapes,
)
