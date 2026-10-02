from executor.geom.helpers import _check_shape, tool, _next_name, _publish, _resolve_point


@tool("make_box", category="primitive",
      description='Axis-aligned solid box with one corner at the origin, extending +X/+Y/+Z by dx/dy/dz.',
      schema={"type": "object", "properties": {"dx": {"type": "number"}, "dy": {"type": "number"}, "dz": {"type": "number"}}, "required": ["dx", "dy", "dz"]},
      arg_types={"dx": float, "dy": float, "dz": float})
def tool_make_box(executor, args: dict):
    geompy = executor.namespace["geompy"]
    box = geompy.MakeBoxDXDYDZ(args["dx"], args["dy"], args["dz"])
    name = _next_name(executor, "box")
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


@tool("make_box_two_pnt", category="primitive",
      description='Axis-aligned solid box spanning two opposite corner points.',
      schema={"type": "object", "properties": {"pnt1": {"type": "string", "description": "first corner: a point name or \"x,y,z\""}, "pnt2": {"type": "string", "description": "opposite corner: a point name or \"x,y,z\""}}, "required": ["pnt1", "pnt2"]},
      arg_types={})
def tool_make_box_two_pnt(executor, args: dict):
    geompy = executor.namespace["geompy"]
    records: list = []
    n1 = _resolve_point(executor, args["pnt1"], records)
    n2 = _resolve_point(executor, args["pnt2"], records)
    box = geompy.MakeBoxTwoPnt(executor.geom_bindings[n1], executor.geom_bindings[n2])
    is_valid, msg = _check_shape(geompy, box)
    if not is_valid:
        return {"ok": False, "error": f"make_box_two_pnt: CheckShape failed ({msg})"}
    name = _next_name(executor, "box")
    _publish(executor, box, name)
    records.append({
        "fn": "MakeBoxTwoPnt",
        "args": [{"kind": "ref", "value": n1}, {"kind": "ref", "value": n2}],
        "out_vars": [name],
        "publish": [name],
    })
    return {"ok": True, "records": records, "sanity": {"ok": True}}


@tool("make_cylinder", category="primitive",
      description='Solid cylinder, base centred at the origin, axis along +Z.',
      schema={"type": "object", "properties": {"radius": {"type": "number"}, "height": {"type": "number"}}, "required": ["radius", "height"]},
      arg_types={"radius": float, "height": float})
def tool_make_cylinder(executor, args: dict):
    geompy = executor.namespace["geompy"]
    cyl = geompy.MakeCylinderRH(args["radius"], args["height"])
    is_valid, msg = _check_shape(geompy, cyl)
    if not is_valid:
        return {"ok": False, "error": f"make_cylinder: CheckShape failed ({msg})"}
    name = _next_name(executor, "cyl")
    _publish(executor, cyl, name)
    return {
        "ok": True,
        "records": [{
            "fn": "MakeCylinderRH",
            "args": [{"kind": "literal", "value": args["radius"]},
                     {"kind": "literal", "value": args["height"]}],
            "out_vars": [name],
            "publish": [name],
        }],
        "sanity": {"ok": True},
    }


@tool("make_sphere", category="primitive",
      description='Solid sphere centred at the origin (or at `center` if given).',
      schema={"type": "object", "properties": {"radius": {"type": "number"}, "center": {"type": "string", "description": "optional center point"}}, "required": ["radius"]},
      arg_types={"radius": float})
def tool_make_sphere(executor, args: dict):
    geompy = executor.namespace["geompy"]
    records: list = []
    if "center" in args:
        cn = _resolve_point(executor, args["center"], records)
        sph = geompy.MakeSpherePntR(executor.geom_bindings[cn], args["radius"])
        record_args = [{"kind": "ref", "value": cn},
                       {"kind": "literal", "value": args["radius"]}]
        fn = "MakeSpherePntR"
    else:
        sph = geompy.MakeSphereR(args["radius"])
        record_args = [{"kind": "literal", "value": args["radius"]}]
        fn = "MakeSphereR"
    is_valid, msg = _check_shape(geompy, sph)
    if not is_valid:
        return {"ok": False, "error": f"make_sphere: CheckShape failed ({msg})"}
    name = _next_name(executor, "sph")
    _publish(executor, sph, name)
    records.append({
        "fn": fn,
        "args": record_args,
        "out_vars": [name],
        "publish": [name],
    })
    return {"ok": True, "records": records, "sanity": {"ok": True}}


@tool("make_cone", category="primitive",
      description='Solid (truncated) cone, base radius1 at the origin, radius2 at height, axis along +Z. radius2=0 for a sharp tip.',
      schema={"type": "object", "properties": {"radius1": {"type": "number"}, "radius2": {"type": "number"}, "height": {"type": "number"}}, "required": ["radius1", "radius2", "height"]},
      arg_types={"radius1": float, "radius2": float, "height": float})
def tool_make_cone(executor, args: dict):
    geompy = executor.namespace["geompy"]
    cone = geompy.MakeConeR1R2H(args["radius1"], args["radius2"], args["height"])
    is_valid, msg = _check_shape(geompy, cone)
    if not is_valid:
        return {"ok": False, "error": f"make_cone: CheckShape failed ({msg})"}
    name = _next_name(executor, "cone")
    _publish(executor, cone, name)
    return {
        "ok": True,
        "records": [{
            "fn": "MakeConeR1R2H",
            "args": [{"kind": "literal", "value": args["radius1"]},
                     {"kind": "literal", "value": args["radius2"]},
                     {"kind": "literal", "value": args["height"]}],
            "out_vars": [name],
            "publish": [name],
        }],
        "sanity": {"ok": True},
    }


@tool("make_torus", category="primitive",
      description='Solid torus centred at the origin, axis along +Z; radius_major = ring centre-line, radius_minor = tube.',
      schema={"type": "object", "properties": {"radius_major": {"type": "number"}, "radius_minor": {"type": "number"}}, "required": ["radius_major", "radius_minor"]},
      arg_types={"radius_major": float, "radius_minor": float})
def tool_make_torus(executor, args: dict):
    geompy = executor.namespace["geompy"]
    torus = geompy.MakeTorusRR(args["radius_major"], args["radius_minor"])
    is_valid, msg = _check_shape(geompy, torus)
    if not is_valid:
        return {"ok": False, "error": f"make_torus: CheckShape failed ({msg})"}
    name = _next_name(executor, "torus")
    _publish(executor, torus, name)
    return {
        "ok": True,
        "records": [{
            "fn": "MakeTorusRR",
            "args": [{"kind": "literal", "value": args["radius_major"]},
                     {"kind": "literal", "value": args["radius_minor"]}],
            "out_vars": [name],
            "publish": [name],
        }],
        "sanity": {"ok": True},
    }
