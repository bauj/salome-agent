import math

from executor.geom.helpers import tool, _next_name, _publish, _resolve_axis


@tool("make_prism", category="swept",
      description='Linear extrusion of a face/wire/edge by the vector (dx, dy, dz).',
      schema={"type": "object", "properties": {"base": {"type": "string", "description": "face/wire/edge to extrude"}, "dx": {"type": "number"}, "dy": {"type": "number"}, "dz": {"type": "number"}}, "required": ["base", "dx", "dy", "dz"]},
      arg_types={"dx": float, "dy": float, "dz": float})
def tool_make_prism(executor, args: dict):
    geompy = executor.namespace["geompy"]
    base = executor.geom_bindings[args["base"]]
    prism = geompy.MakePrismDXDYDZ(base, args["dx"], args["dy"], args["dz"])
    is_valid, msg = geompy.CheckShape(prism, True, 1)
    if not is_valid:
        return {"ok": False, "error": f"make_prism: CheckShape failed ({msg})"}
    name = _next_name(executor, "prism")
    _publish(executor, prism, name)
    return {
        "ok": True,
        "records": [{
            "fn": "MakePrismDXDYDZ",
            "args": [{"kind": "ref", "value": args["base"]},
                     {"kind": "literal", "value": args["dx"]},
                     {"kind": "literal", "value": args["dy"]},
                     {"kind": "literal", "value": args["dz"]}],
            "out_vars": [name],
            "publish": [name],
        }],
        "sanity": {"ok": True},
    }


@tool("make_revolution", category="swept",
      description='Revolve a face/wire/edge about `axis` by angle_deg. The profile must not touch the axis.',
      schema={"type": "object", "properties": {"base": {"type": "string", "description": "face/wire/edge to revolve"}, "axis": {"description": "\"OX\"/\"OY\"/\"OZ\" or published vector name", "oneOf": [{"type": "string", "enum": ["OX", "OY", "OZ"]}, {"type": "string"}]}, "angle_deg": {"type": "number", "description": "rotation angle in degrees"}}, "required": ["base", "axis", "angle_deg"]},
      arg_types={"angle_deg": float})
def tool_make_revolution(executor, args: dict):
    geompy = executor.namespace["geompy"]
    base = executor.geom_bindings[args["base"]]
    records = []
    axis = _resolve_axis(executor, args["axis"], records)
    angle_rad = math.radians(args["angle_deg"])
    rev = geompy.MakeRevolution(base, executor.geom_bindings[axis], angle_rad)
    is_valid, msg = geompy.CheckShape(rev, True, 1)
    if not is_valid:
        return {"ok": False, "error": f"make_revolution: CheckShape failed ({msg})"}
    name = _next_name(executor, "rev")
    _publish(executor, rev, name)
    records.append({
        "fn": "MakeRevolution",
        "args": [{"kind": "ref", "value": args["base"]},
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


@tool("make_pipe", category="swept",
      description='Sweep a cross-section `base` along a `path` wire/edge.',
      schema={"type": "object", "properties": {"base": {"type": "string", "description": "cross-section to sweep"}, "path": {"type": "string", "description": "wire/edge to sweep along"}}, "required": ["base", "path"]},
      arg_types={})
def tool_make_pipe(executor, args: dict):
    geompy = executor.namespace["geompy"]
    base = executor.geom_bindings[args["base"]]
    path = executor.geom_bindings[args["path"]]
    pipe = geompy.MakePipe(base, path)
    is_valid, msg = geompy.CheckShape(pipe, True, 1)
    if not is_valid:
        return {"ok": False, "error": f"make_pipe: CheckShape failed ({msg})"}
    name = _next_name(executor, "pipe")
    _publish(executor, pipe, name)
    return {
        "ok": True,
        "records": [{
            "fn": "MakePipe",
            "args": [{"kind": "ref", "value": args["base"]},
                     {"kind": "ref", "value": args["path"]}],
            "out_vars": [name],
            "publish": [name],
        }],
        "sanity": {"ok": True},
    }


@tool("make_thick_solid", category="swept",
      description='Hollow a solid into a shell of wall thickness `thickness`. `faces` lists face subshape IDs to leave open (required -- a closed solid cannot be thickened).',
      schema={"type": "object", "properties": {"shape": {"type": "string", "description": "shape to thicken"}, "thickness": {"type": "number"}, "faces": {"type": "array", "items": {"type": "integer"}, "description": "face subshape IDs to remove (open) -- required; a closed solid cannot be thickened without at least one"}, "inside": {"type": "boolean", "default": False}}, "required": ["shape", "thickness", "faces"]},
      arg_types={"thickness": float})
def tool_make_thick_solid(executor, args: dict):
    geompy = executor.namespace["geompy"]
    shape = executor.geom_bindings[args["shape"]]
    inside = args.get("inside", False)
    # MakeThickSolid on a closed solid fails ("Offset driver failed") unless
    # some faces are removed to open it -- theFacesIDs is effectively
    # required. Callers pass explicit face subshape IDs (nicer selection
    # comes with task 176b).
    faces = args["faces"]
    thick = geompy.MakeThickSolid(shape, args["thickness"], faces, inside)
    is_valid, msg = geompy.CheckShape(thick, True, 1)
    if not is_valid:
        return {"ok": False, "error": f"make_thick_solid: CheckShape failed ({msg})"}
    name = _next_name(executor, "thick")
    _publish(executor, thick, name)
    return {
        "ok": True,
        "records": [{
            "fn": "MakeThickSolid",
            "args": [{"kind": "ref", "value": args["shape"]},
                     {"kind": "literal", "value": args["thickness"]},
                     {"kind": "literal", "value": faces},
                     {"kind": "literal", "value": inside}],
            "out_vars": [name],
            "publish": [name],
        }],
        "sanity": {"ok": True},
    }


@tool("make_filling", category="swept",
      description='A surface filling a set of boundary contour edges/wires.',
      schema={"type": "object", "properties": {"contours": {"type": "array", "items": {"type": "string"}, "description": "edges/wires to fill"}}, "required": ["contours"]},
      arg_types={})
def tool_make_filling(executor, args: dict):
    geompy = executor.namespace["geompy"]
    contours = [executor.geom_bindings[c] for c in args["contours"]]
    fill = geompy.MakeFilling(contours)
    is_valid, msg = geompy.CheckShape(fill, True, 1)
    if not is_valid:
        return {"ok": False, "error": f"make_filling: CheckShape failed ({msg})"}
    name = _next_name(executor, "fill")
    _publish(executor, fill, name)
    return {
        "ok": True,
        "records": [{
            "fn": "MakeFilling",
            "args": [{"kind": "list", "value": [{"kind": "ref", "value": c} for c in args["contours"]]}],
            "out_vars": [name],
            "publish": [name],
        }],
        "sanity": {"ok": True},
    }
