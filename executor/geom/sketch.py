from executor.geom.helpers import tool, _next_name, _publish, _resolve_point, _resolve_axis


@tool("make_face", category="sketch",
      description='Planar rectangular face height x width. orientation 1=XOY, 2=YOZ, 3=ZOX.',
      schema={"type": "object", "properties": {"height": {"type": "number"}, "width": {"type": "number"}, "orientation": {"type": "integer", "enum": [1, 2, 3], "default": 1}}, "required": ["height", "width"]},
      arg_types={"height": float, "width": float, "orientation": int})
def tool_make_face(executor, args: dict):
    geompy = executor.namespace["geompy"]
    orientation = args.get("orientation", 1)
    face = geompy.MakeFaceHW(args["height"], args["width"], orientation)
    is_valid, msg = geompy.CheckShape(face, True, 1)
    if not is_valid:
        return {"ok": False, "error": f"make_face: CheckShape failed ({msg})"}
    name = _next_name(executor, "face")
    _publish(executor, face, name)
    return {
        "ok": True,
        "records": [{
            "fn": "MakeFaceHW",
            "args": [{"kind": "literal", "value": args["height"]},
                     {"kind": "literal", "value": args["width"]},
                     {"kind": "literal", "value": orientation}],
            "out_vars": [name],
            "publish": [name],
        }],
        "sanity": {"ok": True},
    }


@tool("make_disk", category="sketch",
      description='Planar circular face of the given radius. orientation 1=XOY, 2=YOZ, 3=ZOX.',
      schema={"type": "object", "properties": {"radius": {"type": "number"}, "orientation": {"type": "integer", "enum": [1, 2, 3], "default": 1}}, "required": ["radius"]},
      arg_types={"radius": float, "orientation": int})
def tool_make_disk(executor, args: dict):
    geompy = executor.namespace["geompy"]
    orientation = args.get("orientation", 1)
    disk = geompy.MakeDiskR(args["radius"], orientation)
    is_valid, msg = geompy.CheckShape(disk, True, 1)
    if not is_valid:
        return {"ok": False, "error": f"make_disk: CheckShape failed ({msg})"}
    name = _next_name(executor, "disk")
    _publish(executor, disk, name)
    return {
        "ok": True,
        "records": [{
            "fn": "MakeDiskR",
            "args": [{"kind": "literal", "value": args["radius"]},
                     {"kind": "literal", "value": orientation}],
            "out_vars": [name],
            "publish": [name],
        }],
        "sanity": {"ok": True},
    }


@tool("make_point", category="sketch",
      description='A vertex at (x, y, z).',
      schema={"type": "object", "properties": {"x": {"type": "number"}, "y": {"type": "number"}, "z": {"type": "number"}}, "required": ["x", "y", "z"]},
      arg_types={"x": float, "y": float, "z": float})
def tool_make_point(executor, args: dict):
    geompy = executor.namespace["geompy"]
    pt = geompy.MakeVertex(args["x"], args["y"], args["z"])
    # Skip CheckShape for vertices (always valid)
    name = _next_name(executor, "pt")
    _publish(executor, pt, name)
    return {
        "ok": True,
        "records": [{
            "fn": "MakeVertex",
            "args": [{"kind": "literal", "value": args["x"]},
                     {"kind": "literal", "value": args["y"]},
                     {"kind": "literal", "value": args["z"]}],
            "out_vars": [name],
            "publish": [name],
        }],
        "sanity": {"ok": True},
    }


@tool("make_vector", category="sketch",
      description='A free vector with components (dx, dy, dz) from the origin.',
      schema={"type": "object", "properties": {"dx": {"type": "number"}, "dy": {"type": "number"}, "dz": {"type": "number"}}, "required": ["dx", "dy", "dz"]},
      arg_types={"dx": float, "dy": float, "dz": float})
def tool_make_vector(executor, args: dict):
    geompy = executor.namespace["geompy"]
    vec = geompy.MakeVectorDXDYDZ(args["dx"], args["dy"], args["dz"])
    # Skip CheckShape for vectors (always valid)
    name = _next_name(executor, "vec")
    _publish(executor, vec, name)
    return {
        "ok": True,
        "records": [{
            "fn": "MakeVectorDXDYDZ",
            "args": [{"kind": "literal", "value": args["dx"]},
                     {"kind": "literal", "value": args["dy"]},
                     {"kind": "literal", "value": args["dz"]}],
            "out_vars": [name],
            "publish": [name],
        }],
        "sanity": {"ok": True},
    }


@tool("make_line", category="sketch",
      description='A straight line segment between two points.',
      schema={"type": "object", "properties": {"pnt1": {"type": "string", "description": "first point"}, "pnt2": {"type": "string", "description": "second point"}}, "required": ["pnt1", "pnt2"]},
      arg_types={})
def tool_make_line(executor, args: dict):
    geompy = executor.namespace["geompy"]
    records: list = []
    n1 = _resolve_point(executor, args["pnt1"], records)
    n2 = _resolve_point(executor, args["pnt2"], records)
    line = geompy.MakeLineTwoPnt(executor.geom_bindings[n1], executor.geom_bindings[n2])
    name = _next_name(executor, "line")
    _publish(executor, line, name)
    records.append({
        "fn": "MakeLineTwoPnt",
        "args": [{"kind": "ref", "value": n1}, {"kind": "ref", "value": n2}],
        "out_vars": [name],
        "publish": [name],
    })
    return {"ok": True, "records": records, "sanity": {"ok": True}}


@tool("make_plane", category="sketch",
      description='An unbounded plane through `point` with the given `normal` vector (trimmed for display at trim_size).',
      schema={"type": "object", "properties": {"point": {"type": "string", "description": "point on plane"}, "normal": {"type": "string", "description": "normal vector"}, "trim_size": {"type": "number", "default": 2000}}, "required": ["point", "normal"]},
      arg_types={"trim_size": float})
def tool_make_plane(executor, args: dict):
    geompy = executor.namespace["geompy"]
    records: list = []
    pn = _resolve_point(executor, args["point"], records)
    vn = _resolve_axis(executor, args["normal"], records)
    trim_size = args.get("trim_size", 2000)
    plane = geompy.MakePlane(executor.geom_bindings[pn], executor.geom_bindings[vn], trim_size)
    is_valid, msg = geompy.CheckShape(plane, True, 1)
    if not is_valid:
        return {"ok": False, "error": f"make_plane: CheckShape failed ({msg})"}
    name = _next_name(executor, "plane")
    _publish(executor, plane, name)
    records.append({
        "fn": "MakePlane",
        "args": [{"kind": "ref", "value": pn},
                 {"kind": "ref", "value": vn},
                 {"kind": "literal", "value": trim_size}],
        "out_vars": [name],
        "publish": [name],
    })
    return {"ok": True, "records": records, "sanity": {"ok": True}}


@tool("make_circle", category="sketch",
      description='A circular edge (wire), centre point, `normal` axis, radius.',
      schema={"type": "object", "properties": {"center": {"type": "string", "description": "center point"}, "normal": {"type": "string", "description": "normal vector"}, "radius": {"type": "number"}}, "required": ["center", "normal", "radius"]},
      arg_types={"radius": float})
def tool_make_circle(executor, args: dict):
    geompy = executor.namespace["geompy"]
    records: list = []
    cn = _resolve_point(executor, args["center"], records)
    vn = _resolve_axis(executor, args["normal"], records)
    circ = geompy.MakeCircle(executor.geom_bindings[cn], executor.geom_bindings[vn], args["radius"])
    name = _next_name(executor, "circ")
    _publish(executor, circ, name)
    records.append({
        "fn": "MakeCircle",
        "args": [{"kind": "ref", "value": cn},
                 {"kind": "ref", "value": vn},
                 {"kind": "literal", "value": args["radius"]}],
        "out_vars": [name],
        "publish": [name],
    })
    return {"ok": True, "records": records, "sanity": {"ok": True}}


@tool("make_polygon_face", category="sketch",
      description=('Planar face bounded by a closed polygon through `points` (3 or more corners, '
                   'in order; the last joins back to the first). Extrude it with make_prism for a '
                   'gable roof, wedge or any prismatic profile.'),
      schema={"type": "object", "properties": {"points": {"type": "array", "items": {"type": "string"}, "minItems": 3, "description": "corners in order: point names or \"x,y,z\""}}, "required": ["points"]},
      arg_types={})
def tool_make_polygon_face(executor, args: dict):
    if len(args["points"]) < 3:
        return {"ok": False, "error": "make_polygon_face: needs at least 3 points"}
    geompy = executor.namespace["geompy"]
    records: list = []
    pts = [_resolve_point(executor, p, records) for p in args["points"]]
    wire = geompy.MakePolyline([executor.geom_bindings[n] for n in pts], True)
    wname = _next_name(executor, "wire")
    executor.geom_bindings[wname] = wire
    records.append({
        "fn": "MakePolyline",
        "args": [{"kind": "list", "value": [{"kind": "ref", "value": n} for n in pts]},
                 {"kind": "literal", "value": True}],
        "out_vars": [wname],
        "publish": None,
    })
    try:
        face = geompy.MakeFaceWires([wire], 1)
    except Exception as e:
        return {"ok": False, "error": (f"make_polygon_face: no planar face ({e}) — all points "
                                       "must lie in the same plane")}
    is_valid, msg = geompy.CheckShape(face, True, 1)
    if not is_valid:
        return {"ok": False, "error": f"make_polygon_face: CheckShape failed ({msg})"}
    name = _next_name(executor, "face")
    _publish(executor, face, name)
    records.append({
        "fn": "MakeFaceWires",
        "args": [{"kind": "list", "value": [{"kind": "ref", "value": wname}]},
                 {"kind": "literal", "value": 1}],
        "out_vars": [name],
        "publish": [name],
    })
    return {"ok": True, "records": records, "sanity": {"ok": True}}
