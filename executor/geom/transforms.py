import math

from executor.geom.helpers import (
    _check_shape,
    tool, _next_name, _publish, _resolve_axis, _resolve_plane, _resolve_point,
)
from executor.geom.results import _shape_info


@tool("translate", category="transform",
      description='A COPY of `shape` moved by (dx, dy, dz); the source is unchanged -- refer to the new name afterwards.',
      schema={"type": "object", "properties": {"shape": {"type": "string"}, "dx": {"type": "number"}, "dy": {"type": "number"}, "dz": {"type": "number"}}, "required": ["shape", "dx", "dy", "dz"]},
      arg_types={"dx": float, "dy": float, "dz": float})
def tool_translate(executor, args: dict):
    geompy = executor.namespace["geompy"]
    shape = executor.geom_bindings[args["shape"]]
    result = geompy.MakeTranslation(shape, args["dx"], args["dy"], args["dz"])
    is_valid, msg = _check_shape(geompy, result)
    if not is_valid:
        return {"ok": False, "error": f"translate: CheckShape failed ({msg})"}
    name = _next_name(executor, "tr")
    _publish(executor, result, name)
    return {
        "ok": True,
        "records": [{
            "fn": "MakeTranslation",
            "args": [{"kind": "ref", "value": args["shape"]},
                     {"kind": "literal", "value": args["dx"]},
                     {"kind": "literal", "value": args["dy"]},
                     {"kind": "literal", "value": args["dz"]}],
            "out_vars": [name],
            "publish": [name],
        }],
        "sanity": {"ok": True},
    }


@tool("rotate", category="transform",
      description='A COPY of `shape` rotated about `axis` by angle_deg; the source is unchanged.',
      schema={"type": "object", "properties": {"shape": {"type": "string"}, "axis": {"description": "\"OX\"/\"OY\"/\"OZ\" or published vector name", "oneOf": [{"type": "string", "enum": ["OX", "OY", "OZ"]}, {"type": "string"}]}, "angle_deg": {"type": "number"}}, "required": ["shape", "axis", "angle_deg"]},
      arg_types={"angle_deg": float})
def tool_rotate(executor, args: dict):
    geompy = executor.namespace["geompy"]
    shape = executor.geom_bindings[args["shape"]]
    records = []
    axis = _resolve_axis(executor, args["axis"], records)
    angle_rad = math.radians(args["angle_deg"])
    result = geompy.MakeRotation(shape, executor.geom_bindings[axis], angle_rad)
    is_valid, msg = _check_shape(geompy, result)
    if not is_valid:
        return {"ok": False, "error": f"rotate: CheckShape failed ({msg})"}
    name = _next_name(executor, "rot")
    _publish(executor, result, name)
    records.append({
        "fn": "MakeRotation",
        "args": [{"kind": "ref", "value": args["shape"]},
                 {"kind": "ref", "value": axis},
                 {"kind": "literal", "value": angle_rad}],
        "out_vars": [name],
        "publish": [name],
    })
    return {
        "ok": True,
        "records": records,
        "sanity": {"ok": True},
    }


@tool("mirror", category="transform",
      description='A COPY of `shape` mirrored across a plane, axis or point (`by` picks which; `entity` names it or a shortcut).',
      schema={"type": "object", "properties": {"shape": {"type": "string"}, "by": {"type": "string", "enum": ["plane", "axis", "point"], "default": "plane"}, "entity": {"description": "plane/axis/point name, or XOY/YOZ/ZOX/OX/OY/OZ/O for shortcuts", "type": "string"}}, "required": ["shape", "entity"]},
      arg_types={})
def tool_mirror(executor, args: dict):
    geompy = executor.namespace["geompy"]
    shape = executor.geom_bindings[args["shape"]]
    by = args.get("by", "plane")
    records = []

    if by == "axis":
        axis = _resolve_axis(executor, args["entity"], records)
        result = geompy.MakeMirrorByAxis(shape, executor.geom_bindings[axis])
        fn = "MakeMirrorByAxis"
        entity_ref = axis
    elif by == "point":
        point = _resolve_point(executor, args["entity"], records)
        result = geompy.MakeMirrorByPoint(shape, executor.geom_bindings[point])
        fn = "MakeMirrorByPoint"
        entity_ref = point
    else:  # by == "plane" (default)
        plane = _resolve_plane(executor, args["entity"], records)
        result = geompy.MakeMirrorByPlane(shape, executor.geom_bindings[plane])
        fn = "MakeMirrorByPlane"
        entity_ref = plane

    is_valid, msg = _check_shape(geompy, result)
    if not is_valid:
        return {"ok": False, "error": f"mirror: CheckShape failed ({msg})"}
    name = _next_name(executor, "mir")
    _publish(executor, result, name)
    records.append({
        "fn": fn,
        "args": [{"kind": "ref", "value": args["shape"]},
                 {"kind": "ref", "value": entity_ref}],
        "out_vars": [name],
        "publish": [name],
    })
    return {
        "ok": True,
        "records": records,
        "sanity": {"ok": True},
    }


@tool("scale", category="transform",
      description='A COPY of `shape` scaled about `center` -- uniform `factor`, or per-axis factor_x/y/z.',
      schema={"type": "object", "properties": {"shape": {"type": "string"}, "factor": {"type": "number", "description": "uniform scale factor"}, "factor_x": {"type": "number", "description": "X-axis scale factor"}, "factor_y": {"type": "number", "description": "Y-axis scale factor"}, "factor_z": {"type": "number", "description": "Z-axis scale factor"}, "center": {"type": "string", "description": "center point (O or point name)", "default": "O"}}, "required": ["shape"]},
      arg_types={"factor": float, "factor_x": float, "factor_y": float, "factor_z": float})
def tool_scale(executor, args: dict):
    geompy = executor.namespace["geompy"]
    shape = executor.geom_bindings[args["shape"]]
    center = _resolve_point(executor, args.get("center", "O"), [])
    center_obj = executor.geom_bindings[center]

    if "factor" in args:
        result = geompy.MakeScaleTransform(shape, center_obj, args["factor"])
        fn = "MakeScaleTransform"
        args_list = [{"kind": "ref", "value": args["shape"]},
                     {"kind": "ref", "value": center},
                     {"kind": "literal", "value": args["factor"]}]
    else:
        result = geompy.MakeScaleAlongAxes(shape, center_obj, args["factor_x"], args["factor_y"], args["factor_z"])
        fn = "MakeScaleAlongAxes"
        args_list = [{"kind": "ref", "value": args["shape"]},
                     {"kind": "ref", "value": center},
                     {"kind": "literal", "value": args["factor_x"]},
                     {"kind": "literal", "value": args["factor_y"]},
                     {"kind": "literal", "value": args["factor_z"]}]

    is_valid, msg = _check_shape(geompy, result)
    if not is_valid:
        return {"ok": False, "error": f"scale: CheckShape failed ({msg})"}
    name = _next_name(executor, "scale")
    _publish(executor, result, name)
    return {
        "ok": True,
        "records": [{
            "fn": fn,
            "args": args_list,
            "out_vars": [name],
            "publish": [name],
        }],
        "sanity": {"ok": True},
    }


@tool("multi_translate", category="transform",
      description='A COPY of `shape` replicated `count` times along `vector` at spacing `step` (add vector2/step2/count2 for a 2D grid).',
      schema={"type": "object", "properties": {"shape": {"type": "string"}, "vector": {"description": "OX/OY/OZ or vector name", "oneOf": [{"type": "string", "enum": ["OX", "OY", "OZ"]}, {"type": "string"}]}, "step": {"type": "number"}, "count": {"type": "integer"}, "vector2": {"type": "string", "description": "optional second vector for 2D"}, "step2": {"type": "number", "description": "optional second step"}, "count2": {"type": "integer", "description": "optional second count"}}, "required": ["shape", "vector", "step", "count"]},
      arg_types={"step": float, "count": int, "step2": float, "count2": int})
def tool_multi_translate(executor, args: dict):
    geompy = executor.namespace["geompy"]
    shape = executor.geom_bindings[args["shape"]]
    records = []
    vector = _resolve_axis(executor, args["vector"], records)
    vector_obj = executor.geom_bindings[vector]

    if "vector2" in args:
        vector2 = _resolve_axis(executor, args["vector2"], records)
        vector2_obj = executor.geom_bindings[vector2]
        result = geompy.MakeMultiTranslation2D(
            shape, vector_obj, args["step"], args["count"], vector2_obj, args["step2"], args["count2"]
        )
        fn = "MakeMultiTranslation2D"
        args_list = [{"kind": "ref", "value": args["shape"]},
                     {"kind": "ref", "value": vector},
                     {"kind": "literal", "value": args["step"]},
                     {"kind": "literal", "value": args["count"]},
                     {"kind": "ref", "value": vector2},
                     {"kind": "literal", "value": args["step2"]},
                     {"kind": "literal", "value": args["count2"]}]
    else:
        result = geompy.MakeMultiTranslation1D(shape, vector_obj, args["step"], args["count"])
        fn = "MakeMultiTranslation1D"
        args_list = [{"kind": "ref", "value": args["shape"]},
                     {"kind": "ref", "value": vector},
                     {"kind": "literal", "value": args["step"]},
                     {"kind": "literal", "value": args["count"]}]

    is_valid, msg = _check_shape(geompy, result)
    if not is_valid:
        return {"ok": False, "error": f"multi_translate: CheckShape failed ({msg})"}
    name = _next_name(executor, "mtr")
    _publish(executor, result, name)
    records.append({
        "fn": fn,
        "args": args_list,
        "out_vars": [name],
        "publish": [name],
    })
    return {
        "ok": True,
        "records": records,
        "sanity": {"ok": True},
    }


@tool("multi_rotate", category="transform",
      description='A COPY of `shape` replicated `count` times about `axis` (angle_deg = step, default full circle).',
      schema={"type": "object", "properties": {"shape": {"type": "string"}, "axis": {"description": "OX/OY/OZ or axis name", "oneOf": [{"type": "string", "enum": ["OX", "OY", "OZ"]}, {"type": "string"}]}, "count": {"type": "integer", "description": "number of copies"}, "angle_deg": {"type": "number", "description": "optional angle step (degrees)"}}, "required": ["shape", "axis", "count"]},
      arg_types={"count": int, "angle_deg": float})
def tool_multi_rotate(executor, args: dict):
    geompy = executor.namespace["geompy"]
    shape = executor.geom_bindings[args["shape"]]
    records = []
    axis = _resolve_axis(executor, args["axis"], records)
    axis_obj = executor.geom_bindings[axis]

    if "angle_deg" in args:
        angle_rad = math.radians(args["angle_deg"])
        result = geompy.MultiRotate1DByStep(shape, axis_obj, angle_rad, args["count"])
        fn = "MultiRotate1DByStep"
        args_list = [{"kind": "ref", "value": args["shape"]},
                     {"kind": "ref", "value": axis},
                     {"kind": "literal", "value": angle_rad},
                     {"kind": "literal", "value": args["count"]}]
    else:
        result = geompy.MultiRotate1DNbTimes(shape, axis_obj, args["count"])
        fn = "MultiRotate1DNbTimes"
        args_list = [{"kind": "ref", "value": args["shape"]},
                     {"kind": "ref", "value": axis},
                     {"kind": "literal", "value": args["count"]}]

    is_valid, msg = _check_shape(geompy, result)
    if not is_valid:
        return {"ok": False, "error": f"multi_rotate: CheckShape failed ({msg})"}
    name = _next_name(executor, "mrot")
    _publish(executor, result, name)
    records.append({
        "fn": fn,
        "args": args_list,
        "out_vars": [name],
        "publish": [name],
    })
    return {
        "ok": True,
        "records": records,
        "sanity": {"ok": True},
    }


@tool("offset_shape", category="transform",
      description='A COPY of `shape` with every face offset outward (+) or inward (-) by `offset`.',
      schema={"type": "object", "properties": {"shape": {"type": "string"}, "offset": {"type": "number"}}, "required": ["shape", "offset"]},
      arg_types={"offset": float})
def tool_offset_shape(executor, args: dict):
    geompy = executor.namespace["geompy"]
    shape = executor.geom_bindings[args["shape"]]
    result = geompy.MakeOffset(shape, args["offset"])
    is_valid, msg = _check_shape(geompy, result)
    if not is_valid:
        return {"ok": False, "error": f"offset_shape: CheckShape failed ({msg})"}
    name = _next_name(executor, "off")
    _publish(executor, result, name)
    return {
        "ok": True,
        "records": [{
            "fn": "MakeOffset",
            "args": [{"kind": "ref", "value": args["shape"]},
                     {"kind": "literal", "value": args["offset"]}],
            "out_vars": [name],
            "publish": [name],
        }],
        "sanity": {"ok": True},
    }


@tool("position", category="transform",
      description='A COPY of `shape` with its local frame placed at target `origin` with axes `x_dir`/`y_dir` -- use to assemble parts.',
      schema={"type": "object", "properties": {"shape": {"type": "string", "description": "shape to place (a copy is made)"}, "origin": {"type": "array", "items": {"type": "number"}, "description": "target frame origin [x,y,z]; default [0,0,0]"}, "x_dir": {"type": "array", "items": {"type": "number"}, "description": "target frame X-axis direction; default [1,0,0]"}, "y_dir": {"type": "array", "items": {"type": "number"}, "description": "target frame Y-axis direction; default [0,1,0]"}}, "required": ["shape"]},
      arg_types={})
def tool_position(executor, args: dict):
    """Modify Location: place a shape so its local frame lands on the given
    target frame (origin + X/Y axis directions). Copy semantics -- the
    source is untouched. Start LCS is the global frame (None)."""
    geompy = executor.namespace["geompy"]
    shape = executor.geom_bindings[args["shape"]]
    origin = [float(v) for v in args.get("origin", [0.0, 0.0, 0.0])]
    x_dir = [float(v) for v in args.get("x_dir", [1.0, 0.0, 0.0])]
    y_dir = [float(v) for v in args.get("y_dir", [0.0, 1.0, 0.0])]
    marker = geompy.MakeMarker(*origin, *x_dir, *y_dir)
    marker_name = _next_name(executor, "lcs")
    _publish(executor, marker, marker_name)
    result = geompy.MakePosition(shape, None, marker)
    is_valid, msg = _check_shape(geompy, result)
    if not is_valid:
        return {"ok": False, "error": f"position: CheckShape failed ({msg})"}
    name = _next_name(executor, "pos")
    _publish(executor, result, name)
    return {
        "ok": True,
        "records": [
            {"fn": "MakeMarker",
             "args": [{"kind": "literal", "value": v} for v in (*origin, *x_dir, *y_dir)],
             "out_vars": [marker_name], "publish": [marker_name]},
            {"fn": "MakePosition",
             "args": [{"kind": "ref", "value": args["shape"]},
                      {"kind": "literal", "value": None},
                      {"kind": "ref", "value": marker_name}],
             "out_vars": [name], "publish": [name]},
        ],
        "sanity": {"ok": True},
    }


_AXIS_IDX = {"x": 0, "y": 1, "z": 2}


def _extent(bbox: list, axis: str, anchor: str) -> float:
    lo, hi = bbox[2 * _AXIS_IDX[axis]], bbox[2 * _AXIS_IDX[axis] + 1]
    return {"min": lo, "max": hi, "center": (lo + hi) / 2}[anchor]


@tool("align", category="transform",
      description="Move `shape` so chosen faces/centres line up with `to`. `match` is a list of {axis, my, target, gap}: `shape` is translated along that axis until its `my` (min|max|center) meets `to`'s `target` plus `gap` mm. Axes not listed stay put. Use for assembly -- placing a part against, on top of, flush with, or centred on another; the tool does the bounding-box arithmetic and emits a plain translation.",
      schema={"type": "object", "properties": {"shape": {"type": "string", "description": "shape to move (a copy is made)"}, "to": {"type": "string", "description": "reference shape to align against"}, "match": {"type": "array", "description": "one alignment constraint per axis; axes not listed are unchanged", "items": {"type": "object", "properties": {"axis": {"type": "string", "enum": ["x", "y", "z"]}, "my": {"type": "string", "enum": ["min", "max", "center"], "description": "which extreme of `shape` on this axis"}, "target": {"type": "string", "enum": ["min", "max", "center"], "description": "which extreme of `to` on this axis"}, "gap": {"type": "number", "description": "signed offset in mm, default 0"}}, "required": ["axis", "my", "target"]}}}, "required": ["shape", "to", "match"]},
      arg_types={})
def tool_align(executor, args: dict):
    """Relative placement: translate `shape` so its chosen extremes meet
    `to`'s. The model states the spatial relationship; the tool reads both
    bounding boxes and computes the move. Emits a plain MakeTranslation."""
    geompy = executor.namespace["geompy"]
    shape = executor.geom_bindings.get(args["shape"])
    target = executor.geom_bindings.get(args["to"])
    if shape is None:
        return {"ok": False, "error": f"align: unknown shape {args['shape']!r}"}
    if target is None:
        return {"ok": False, "error": f"align: unknown shape {args['to']!r}"}
    s_info = _shape_info(geompy, shape)
    t_info = _shape_info(geompy, target)
    if s_info is None or t_info is None:
        return {"ok": False, "error": "align: could not measure one of the shapes"}
    delta = {"x": 0.0, "y": 0.0, "z": 0.0}
    seen = set()
    for c in args["match"]:
        axis = c["axis"]
        if axis not in _AXIS_IDX:
            return {"ok": False, "error": f"align: bad axis {axis!r}"}
        if axis in seen:
            return {"ok": False, "error": f"align: two constraints on axis {axis!r}"}
        seen.add(axis)
        gap = float(c.get("gap") or 0)
        want = _extent(t_info["bbox"], axis, c["target"]) + gap
        have = _extent(s_info["bbox"], axis, c["my"])
        delta[axis] = round(want - have, 6)
    result = geompy.MakeTranslation(shape, delta["x"], delta["y"], delta["z"])
    is_valid, msg = _check_shape(geompy, result)
    if not is_valid:
        return {"ok": False, "error": f"align: CheckShape failed ({msg})"}
    name = _next_name(executor, "tr")
    _publish(executor, result, name)
    return {
        "ok": True,
        "records": [{
            "fn": "MakeTranslation",
            "args": [{"kind": "ref", "value": args["shape"]},
                     {"kind": "literal", "value": delta["x"]},
                     {"kind": "literal", "value": delta["y"]},
                     {"kind": "literal", "value": delta["z"]}],
            "out_vars": [name],
            "publish": [name],
        }],
        "sanity": {"ok": True},
    }
