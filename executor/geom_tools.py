import math
from typing import Any, Callable, TypedDict
from typing_extensions import NotRequired


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


def reset_counters() -> None:
    _counters.clear()


def _next_name(executor, prefix: str) -> str:
    _counters[prefix] = _counters.get(prefix, 0) + 1
    return f"{executor.name_prefix}{prefix}_{_counters[prefix]}"


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


def _resolve_axis(executor, arg, records: list) -> str:
    """Resolve an axis arg to a ref name.

    If arg is "OX", "OY", or "OZ", emit a MakeVectorDXDYDZ CallRecord and return its name.
    Otherwise, arg is a ref name — return it unchanged.
    """
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

    If arg is "O", emit a MakeVertex CallRecord and return its name.
    Otherwise, arg is a ref name — return it unchanged.
    """
    if arg == "O":
        geompy = executor.namespace["geompy"]
        pt = geompy.MakeVertex(0, 0, 0)
        name = _next_name(executor, "pt")
        executor.geom_bindings[name] = pt
        records.append({
            "fn": "MakeVertex",
            "args": [{"kind": "literal", "value": 0},
                     {"kind": "literal", "value": 0},
                     {"kind": "literal", "value": 0}],
            "out_vars": [name],
            "publish": None,
        })
        return name
    else:
        # It's a ref name
        return arg


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


def tool_fillet_edges(executor, args: dict):
    geompy = executor.namespace["geompy"]
    shape = executor.geom_bindings[args["shape"]]
    edge_ids = (
        geompy.SubShapeAllIDs(shape, geompy.ShapeType["EDGE"])
        if args["edges"] == "all" else args["edges"]
    )
    # geompy.ShapeType["EDGE"] evaluates to 6 (hardcoded literal for result.py)
    edge_type = 6  # geompy.ShapeType["EDGE"]
    filleted = geompy.MakeFillet(shape, args["radius"], edge_type, edge_ids)
    # theReturnStatus=1 -> (isValid, description); the default (0) returns
    # a bare bool, which broke unpacking here until caught in testing.
    is_valid, msg = geompy.CheckShape(filleted, True, 1)
    if not is_valid:
        # Not published yet at this point, so there is nothing to clean
        # up — the invariant decision 16 wants (no orphan survives a
        # failed call) holds by construction rather than by an explicit
        # delete step.
        return {"ok": False, "error": f"fillet_edges: CheckShape failed on result ({msg})"}
    name = _next_name(executor, "fillet")
    _publish(executor, filleted, name)
    return {
        "ok": True,
        "records": [{
            "fn": "MakeFillet",
            "args": [{"kind": "ref", "value": args["shape"]},
                     {"kind": "literal", "value": args["radius"]},
                     {"kind": "literal", "value": edge_type},
                     {"kind": "literal", "value": edge_ids}],
            "out_vars": [name],
            "publish": [name],
        }],
        "sanity": {"ok": True},
    }


def tool_chamfer_edges(executor, args: dict):
    geompy = executor.namespace["geompy"]
    shape = executor.geom_bindings[args["shape"]]
    d1 = args["d1"]
    d2 = args.get("d2", d1)
    edge_ids = (
        geompy.SubShapeAllIDs(shape, geompy.ShapeType["EDGE"])
        if args["edges"] == "all" else args["edges"]
    )
    # MakeChamferEdges(theShape, theD1, theD2, theEdges) -- no shape-type arg,
    # unlike MakeFillet.
    chamfered = geompy.MakeChamferEdges(shape, d1, d2, edge_ids)
    is_valid, msg = geompy.CheckShape(chamfered, True, 1)
    if not is_valid:
        return {"ok": False, "error": f"chamfer_edges: CheckShape failed on result ({msg})"}
    name = _next_name(executor, "chamfer")
    _publish(executor, chamfered, name)
    return {
        "ok": True,
        "records": [{
            "fn": "MakeChamferEdges",
            "args": [{"kind": "ref", "value": args["shape"]},
                     {"kind": "literal", "value": d1},
                     {"kind": "literal", "value": d2},
                     {"kind": "literal", "value": edge_ids}],
            "out_vars": [name],
            "publish": [name],
        }],
        "sanity": {"ok": True},
    }


def tool_make_box_two_pnt(executor, args: dict):
    geompy = executor.namespace["geompy"]
    pnt1 = executor.geom_bindings[args["pnt1"]]
    pnt2 = executor.geom_bindings[args["pnt2"]]
    box = geompy.MakeBoxTwoPnt(pnt1, pnt2)
    is_valid, msg = geompy.CheckShape(box, True, 1)
    if not is_valid:
        return {"ok": False, "error": f"make_box_two_pnt: CheckShape failed ({msg})"}
    name = _next_name(executor, "box")
    _publish(executor, box, name)
    return {
        "ok": True,
        "records": [{
            "fn": "MakeBoxTwoPnt",
            "args": [{"kind": "ref", "value": args["pnt1"]},
                     {"kind": "ref", "value": args["pnt2"]}],
            "out_vars": [name],
            "publish": [name],
        }],
        "sanity": {"ok": True},
    }


def tool_make_cylinder(executor, args: dict):
    geompy = executor.namespace["geompy"]
    cyl = geompy.MakeCylinderRH(args["radius"], args["height"])
    is_valid, msg = geompy.CheckShape(cyl, True, 1)
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


def tool_make_sphere(executor, args: dict):
    geompy = executor.namespace["geompy"]
    if "center" in args:
        center = executor.geom_bindings[args["center"]]
        sph = geompy.MakeSpherePntR(center, args["radius"])
        record_args = [{"kind": "ref", "value": args["center"]},
                       {"kind": "literal", "value": args["radius"]}]
        fn = "MakeSpherePntR"
    else:
        sph = geompy.MakeSphereR(args["radius"])
        record_args = [{"kind": "literal", "value": args["radius"]}]
        fn = "MakeSphereR"
    is_valid, msg = geompy.CheckShape(sph, True, 1)
    if not is_valid:
        return {"ok": False, "error": f"make_sphere: CheckShape failed ({msg})"}
    name = _next_name(executor, "sph")
    _publish(executor, sph, name)
    return {
        "ok": True,
        "records": [{
            "fn": fn,
            "args": record_args,
            "out_vars": [name],
            "publish": [name],
        }],
        "sanity": {"ok": True},
    }


def tool_make_cone(executor, args: dict):
    geompy = executor.namespace["geompy"]
    cone = geompy.MakeConeR1R2H(args["radius1"], args["radius2"], args["height"])
    is_valid, msg = geompy.CheckShape(cone, True, 1)
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


def tool_make_torus(executor, args: dict):
    geompy = executor.namespace["geompy"]
    torus = geompy.MakeTorusRR(args["radius_major"], args["radius_minor"])
    is_valid, msg = geompy.CheckShape(torus, True, 1)
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


def tool_make_line(executor, args: dict):
    geompy = executor.namespace["geompy"]
    pnt1 = executor.geom_bindings[args["pnt1"]]
    pnt2 = executor.geom_bindings[args["pnt2"]]
    line = geompy.MakeLineTwoPnt(pnt1, pnt2)
    name = _next_name(executor, "line")
    _publish(executor, line, name)
    return {
        "ok": True,
        "records": [{
            "fn": "MakeLineTwoPnt",
            "args": [{"kind": "ref", "value": args["pnt1"]},
                     {"kind": "ref", "value": args["pnt2"]}],
            "out_vars": [name],
            "publish": [name],
        }],
        "sanity": {"ok": True},
    }


def tool_make_plane(executor, args: dict):
    geompy = executor.namespace["geompy"]
    point = executor.geom_bindings[args["point"]]
    normal = executor.geom_bindings[args["normal"]]
    trim_size = args.get("trim_size", 2000)
    plane = geompy.MakePlane(point, normal, trim_size)
    is_valid, msg = geompy.CheckShape(plane, True, 1)
    if not is_valid:
        return {"ok": False, "error": f"make_plane: CheckShape failed ({msg})"}
    name = _next_name(executor, "plane")
    _publish(executor, plane, name)
    return {
        "ok": True,
        "records": [{
            "fn": "MakePlane",
            "args": [{"kind": "ref", "value": args["point"]},
                     {"kind": "ref", "value": args["normal"]},
                     {"kind": "literal", "value": trim_size}],
            "out_vars": [name],
            "publish": [name],
        }],
        "sanity": {"ok": True},
    }


def tool_make_circle(executor, args: dict):
    geompy = executor.namespace["geompy"]
    center = executor.geom_bindings[args["center"]]
    normal = executor.geom_bindings[args["normal"]]
    circ = geompy.MakeCircle(center, normal, args["radius"])
    name = _next_name(executor, "circ")
    _publish(executor, circ, name)
    return {
        "ok": True,
        "records": [{
            "fn": "MakeCircle",
            "args": [{"kind": "ref", "value": args["center"]},
                     {"kind": "ref", "value": args["normal"]},
                     {"kind": "literal", "value": args["radius"]}],
            "out_vars": [name],
            "publish": [name],
        }],
        "sanity": {"ok": True},
    }


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


def tool_fuse(executor, args: dict):
    geompy = executor.namespace["geompy"]
    shapes = [executor.geom_bindings[s] for s in args["shapes"]]
    if len(shapes) == 2:
        result = geompy.MakeFuse(shapes[0], shapes[1])
        fn = "MakeFuse"
    else:
        result = geompy.MakeFuseList(shapes)
        fn = "MakeFuseList"
    is_valid, msg = geompy.CheckShape(result, True, 1)
    if not is_valid:
        return {"ok": False, "error": f"fuse: CheckShape failed ({msg})"}
    name = _next_name(executor, "fuse")
    _publish(executor, result, name)

    if len(shapes) == 2:
        args_list = [{"kind": "ref", "value": args["shapes"][0]},
                     {"kind": "ref", "value": args["shapes"][1]}]
    else:
        args_list = [{"kind": "list", "value": [{"kind": "ref", "value": s} for s in args["shapes"]]}]

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


def tool_cut(executor, args: dict):
    geompy = executor.namespace["geompy"]
    main = executor.geom_bindings[args["main"]]
    tools = [executor.geom_bindings[t] for t in args["tools"]]

    if len(tools) == 1:
        result = geompy.MakeCut(main, tools[0])
        fn = "MakeCut"
        args_list = [{"kind": "ref", "value": args["main"]},
                     {"kind": "ref", "value": args["tools"][0]}]
    else:
        result = geompy.MakeCutList(main, tools)
        fn = "MakeCutList"
        args_list = [{"kind": "ref", "value": args["main"]},
                     {"kind": "list", "value": [{"kind": "ref", "value": t} for t in args["tools"]]}]

    is_valid, msg = geompy.CheckShape(result, True, 1)
    if not is_valid:
        return {"ok": False, "error": f"cut: CheckShape failed ({msg})"}
    name = _next_name(executor, "cut")
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


def tool_common(executor, args: dict):
    geompy = executor.namespace["geompy"]
    shapes = [executor.geom_bindings[s] for s in args["shapes"]]
    result = geompy.MakeCommonList(shapes)
    is_valid, msg = geompy.CheckShape(result, True, 1)
    if not is_valid:
        return {"ok": False, "error": f"common: CheckShape failed ({msg})"}
    name = _next_name(executor, "common")
    _publish(executor, result, name)
    return {
        "ok": True,
        "records": [{
            "fn": "MakeCommonList",
            "args": [{"kind": "list", "value": [{"kind": "ref", "value": s} for s in args["shapes"]]}],
            "out_vars": [name],
            "publish": [name],
        }],
        "sanity": {"ok": True},
    }


def tool_partition(executor, args: dict):
    geompy = executor.namespace["geompy"]
    shapes = [executor.geom_bindings[s] for s in args["shapes"]]
    tools = [executor.geom_bindings[t] for t in args.get("tools", [])]
    result = geompy.MakePartition(shapes, tools)
    is_valid, msg = geompy.CheckShape(result, True, 1)
    if not is_valid:
        return {"ok": False, "error": f"partition: CheckShape failed ({msg})"}
    name = _next_name(executor, "part")
    _publish(executor, result, name)
    tools_refs = args.get("tools", [])
    return {
        "ok": True,
        "records": [{
            "fn": "MakePartition",
            "args": [{"kind": "list", "value": [{"kind": "ref", "value": s} for s in args["shapes"]]},
                     {"kind": "list", "value": [{"kind": "ref", "value": t} for t in tools_refs]}],
            "out_vars": [name],
            "publish": [name],
        }],
        "sanity": {"ok": True},
    }


def tool_translate(executor, args: dict):
    geompy = executor.namespace["geompy"]
    shape = executor.geom_bindings[args["shape"]]
    result = geompy.MakeTranslation(shape, args["dx"], args["dy"], args["dz"])
    is_valid, msg = geompy.CheckShape(result, True, 1)
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


def tool_rotate(executor, args: dict):
    geompy = executor.namespace["geompy"]
    shape = executor.geom_bindings[args["shape"]]
    records = []
    axis = _resolve_axis(executor, args["axis"], records)
    angle_rad = math.radians(args["angle_deg"])
    result = geompy.MakeRotation(shape, executor.geom_bindings[axis], angle_rad)
    is_valid, msg = geompy.CheckShape(result, True, 1)
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

    is_valid, msg = geompy.CheckShape(result, True, 1)
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

    is_valid, msg = geompy.CheckShape(result, True, 1)
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

    is_valid, msg = geompy.CheckShape(result, True, 1)
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

    is_valid, msg = geompy.CheckShape(result, True, 1)
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


def tool_offset_shape(executor, args: dict):
    geompy = executor.namespace["geompy"]
    shape = executor.geom_bindings[args["shape"]]
    result = geompy.MakeOffset(shape, args["offset"])
    is_valid, msg = geompy.CheckShape(result, True, 1)
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
    is_valid, msg = geompy.CheckShape(result, True, 1)
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


def tool_set_color(executor, args: dict):
    # Live call for immediate GUI update
    shape = executor.geom_bindings[args["shape"]]
    # Import SALOMEDS at call time
    from salome.kernel import SALOMEDS
    color = SALOMEDS.Color(args["red"] / 255.0, args["green"] / 255.0, args["blue"] / 255.0)
    shape.SetColor(color)

    # Record for codegen: no out_vars, no publish, method_call emit
    return {
        "ok": True,
        "records": [{
            "fn": "SetColor",
            "args": [{"kind": "salomeds_color", "value": [args["red"], args["green"], args["blue"]]}],
            "out_vars": [],
            "publish": None,
            "emit": "method_call",
            "recv": args["shape"],
        }],
        "sanity": {"ok": True},
    }


def tool_set_transparency(executor, args: dict):
    shape = executor.geom_bindings[args["shape"]]
    # Best-effort live GUI update. Headless (no GUI process),
    # ImportComponentGUI itself can raise -- keep the whole thing guarded,
    # the codegen `gui_call` emit is wrapped the same way.
    _gui_call(executor, shape, "setTransparency", args["transparency"])

    # Record for codegen: no out_vars, no publish, gui_call emit
    return {
        "ok": True,
        "records": [{
            "fn": "setTransparency",
            "args": [{"kind": "literal", "value": args["transparency"]}],
            "out_vars": [],
            "publish": None,
            "emit": "gui_call",
            "recv": args["shape"],
        }],
        "sanity": {"ok": True},
    }


def tool_set_display_mode(executor, args: dict):
    shape = executor.geom_bindings[args["shape"]]
    _gui_call(executor, shape, "setDisplayMode", args["mode"])

    # Record for codegen: no out_vars, no publish, gui_call emit
    return {
        "ok": True,
        "records": [{
            "fn": "setDisplayMode",
            "args": [{"kind": "literal", "value": args["mode"]}],
            "out_vars": [],
            "publish": None,
            "emit": "gui_call",
            "recv": args["shape"],
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
    "make_box_two_pnt": tool_make_box_two_pnt,
    "make_cylinder": tool_make_cylinder,
    "make_sphere": tool_make_sphere,
    "make_cone": tool_make_cone,
    "make_torus": tool_make_torus,
    "make_face": tool_make_face,
    "make_disk": tool_make_disk,
    "make_point": tool_make_point,
    "make_vector": tool_make_vector,
    "make_line": tool_make_line,
    "make_plane": tool_make_plane,
    "make_circle": tool_make_circle,
    "make_prism": tool_make_prism,
    "make_revolution": tool_make_revolution,
    "make_pipe": tool_make_pipe,
    "make_thick_solid": tool_make_thick_solid,
    "make_filling": tool_make_filling,
    "fuse": tool_fuse,
    "cut": tool_cut,
    "common": tool_common,
    "partition": tool_partition,
    "translate": tool_translate,
    "rotate": tool_rotate,
    "mirror": tool_mirror,
    "scale": tool_scale,
    "multi_translate": tool_multi_translate,
    "multi_rotate": tool_multi_rotate,
    "offset_shape": tool_offset_shape,
    "position": tool_position,
    "fillet_edges": tool_fillet_edges,
    "chamfer_edges": tool_chamfer_edges,
    "set_color": tool_set_color,
    "set_transparency": tool_set_transparency,
    "set_display_mode": tool_set_display_mode,
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
    "make_box_two_pnt": {},
    "make_cylinder": {"radius": float, "height": float},
    "make_sphere": {"radius": float},
    "make_cone": {"radius1": float, "radius2": float, "height": float},
    "make_torus": {"radius_major": float, "radius_minor": float},
    "make_face": {"height": float, "width": float, "orientation": int},
    "make_disk": {"radius": float, "orientation": int},
    "make_point": {"x": float, "y": float, "z": float},
    "make_vector": {"dx": float, "dy": float, "dz": float},
    "make_line": {},
    "make_plane": {"trim_size": float},
    "make_circle": {"radius": float},
    "make_prism": {"dx": float, "dy": float, "dz": float},
    "make_revolution": {"angle_deg": float},
    "make_pipe": {},
    "make_thick_solid": {"thickness": float},
    "make_filling": {},
    "fuse": {},
    "cut": {},
    "common": {},
    "partition": {},
    "translate": {"dx": float, "dy": float, "dz": float},
    "rotate": {"angle_deg": float},
    "mirror": {},
    "scale": {"factor": float, "factor_x": float, "factor_y": float, "factor_z": float},
    "multi_translate": {"step": float, "count": int, "step2": float, "count2": int},
    "multi_rotate": {"count": int, "angle_deg": float},
    "offset_shape": {"offset": float},
    "fillet_edges": {"radius": float},
    "chamfer_edges": {"d1": float, "d2": float},
    "set_color": {"red": int, "green": int, "blue": int},
    "set_transparency": {"transparency": float},
    "set_display_mode": {"mode": int},
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
    "make_box_two_pnt": {
        "type": "object",
        "properties": {
            "pnt1": {"type": "string", "description": "first corner point"},
            "pnt2": {"type": "string", "description": "opposite corner point"},
        },
        "required": ["pnt1", "pnt2"],
    },
    "make_cylinder": {
        "type": "object",
        "properties": {
            "radius": {"type": "number"},
            "height": {"type": "number"},
        },
        "required": ["radius", "height"],
    },
    "make_sphere": {
        "type": "object",
        "properties": {
            "radius": {"type": "number"},
            "center": {"type": "string", "description": "optional center point"},
        },
        "required": ["radius"],
    },
    "make_cone": {
        "type": "object",
        "properties": {
            "radius1": {"type": "number"},
            "radius2": {"type": "number"},
            "height": {"type": "number"},
        },
        "required": ["radius1", "radius2", "height"],
    },
    "make_torus": {
        "type": "object",
        "properties": {
            "radius_major": {"type": "number"},
            "radius_minor": {"type": "number"},
        },
        "required": ["radius_major", "radius_minor"],
    },
    "make_face": {
        "type": "object",
        "properties": {
            "height": {"type": "number"},
            "width": {"type": "number"},
            "orientation": {"type": "integer", "enum": [1, 2, 3], "default": 1},
        },
        "required": ["height", "width"],
    },
    "make_disk": {
        "type": "object",
        "properties": {
            "radius": {"type": "number"},
            "orientation": {"type": "integer", "enum": [1, 2, 3], "default": 1},
        },
        "required": ["radius"],
    },
    "make_point": {
        "type": "object",
        "properties": {
            "x": {"type": "number"},
            "y": {"type": "number"},
            "z": {"type": "number"},
        },
        "required": ["x", "y", "z"],
    },
    "make_vector": {
        "type": "object",
        "properties": {
            "dx": {"type": "number"},
            "dy": {"type": "number"},
            "dz": {"type": "number"},
        },
        "required": ["dx", "dy", "dz"],
    },
    "make_line": {
        "type": "object",
        "properties": {
            "pnt1": {"type": "string", "description": "first point"},
            "pnt2": {"type": "string", "description": "second point"},
        },
        "required": ["pnt1", "pnt2"],
    },
    "make_plane": {
        "type": "object",
        "properties": {
            "point": {"type": "string", "description": "point on plane"},
            "normal": {"type": "string", "description": "normal vector"},
            "trim_size": {"type": "number", "default": 2000},
        },
        "required": ["point", "normal"],
    },
    "make_circle": {
        "type": "object",
        "properties": {
            "center": {"type": "string", "description": "center point"},
            "normal": {"type": "string", "description": "normal vector"},
            "radius": {"type": "number"},
        },
        "required": ["center", "normal", "radius"],
    },
    "make_prism": {
        "type": "object",
        "properties": {
            "base": {"type": "string", "description": "face/wire/edge to extrude"},
            "dx": {"type": "number"},
            "dy": {"type": "number"},
            "dz": {"type": "number"},
        },
        "required": ["base", "dx", "dy", "dz"],
    },
    "make_revolution": {
        "type": "object",
        "properties": {
            "base": {"type": "string", "description": "face/wire/edge to revolve"},
            "axis": {
                "description": "\"OX\"/\"OY\"/\"OZ\" or published vector name",
                "oneOf": [{"type": "string", "enum": ["OX", "OY", "OZ"]},
                          {"type": "string"}],
            },
            "angle_deg": {"type": "number", "description": "rotation angle in degrees"},
        },
        "required": ["base", "axis", "angle_deg"],
    },
    "make_pipe": {
        "type": "object",
        "properties": {
            "base": {"type": "string", "description": "cross-section to sweep"},
            "path": {"type": "string", "description": "wire/edge to sweep along"},
        },
        "required": ["base", "path"],
    },
    "make_thick_solid": {
        "type": "object",
        "properties": {
            "shape": {"type": "string", "description": "shape to thicken"},
            "thickness": {"type": "number"},
            "faces": {"type": "array", "items": {"type": "integer"},
                      "description": "face subshape IDs to remove (open) -- required; a closed solid cannot be thickened without at least one"},
            "inside": {"type": "boolean", "default": False},
        },
        "required": ["shape", "thickness", "faces"],
    },
    "make_filling": {
        "type": "object",
        "properties": {
            "contours": {"type": "array", "items": {"type": "string"}, "description": "edges/wires to fill"},
        },
        "required": ["contours"],
    },
    "fuse": {
        "type": "object",
        "properties": {
            "shapes": {"type": "array", "items": {"type": "string"}, "description": "shapes to fuse (2 or more)"},
        },
        "required": ["shapes"],
    },
    "cut": {
        "type": "object",
        "properties": {
            "main": {"type": "string", "description": "main shape"},
            "tools": {"type": "array", "items": {"type": "string"}, "description": "tool shapes (1 or more)"},
        },
        "required": ["main", "tools"],
    },
    "common": {
        "type": "object",
        "properties": {
            "shapes": {"type": "array", "items": {"type": "string"}, "description": "shapes to intersect"},
        },
        "required": ["shapes"],
    },
    "partition": {
        "type": "object",
        "properties": {
            "shapes": {"type": "array", "items": {"type": "string"}, "description": "shapes to partition"},
            "tools": {"type": "array", "items": {"type": "string"}, "description": "optional tool shapes", "default": []},
        },
        "required": ["shapes"],
    },
    "translate": {
        "type": "object",
        "properties": {
            "shape": {"type": "string"},
            "dx": {"type": "number"},
            "dy": {"type": "number"},
            "dz": {"type": "number"},
        },
        "required": ["shape", "dx", "dy", "dz"],
    },
    "rotate": {
        "type": "object",
        "properties": {
            "shape": {"type": "string"},
            "axis": {
                "description": "\"OX\"/\"OY\"/\"OZ\" or published vector name",
                "oneOf": [{"type": "string", "enum": ["OX", "OY", "OZ"]},
                          {"type": "string"}],
            },
            "angle_deg": {"type": "number"},
        },
        "required": ["shape", "axis", "angle_deg"],
    },
    "mirror": {
        "type": "object",
        "properties": {
            "shape": {"type": "string"},
            "by": {"type": "string", "enum": ["plane", "axis", "point"], "default": "plane"},
            "entity": {
                "description": "plane/axis/point name, or XOY/YOZ/ZOX/OX/OY/OZ/O for shortcuts",
                "type": "string",
            },
        },
        "required": ["shape", "entity"],
    },
    "scale": {
        "type": "object",
        "properties": {
            "shape": {"type": "string"},
            "factor": {"type": "number", "description": "uniform scale factor"},
            "factor_x": {"type": "number", "description": "X-axis scale factor"},
            "factor_y": {"type": "number", "description": "Y-axis scale factor"},
            "factor_z": {"type": "number", "description": "Z-axis scale factor"},
            "center": {"type": "string", "description": "center point (O or point name)", "default": "O"},
        },
        "required": ["shape"],
    },
    "multi_translate": {
        "type": "object",
        "properties": {
            "shape": {"type": "string"},
            "vector": {
                "description": "OX/OY/OZ or vector name",
                "oneOf": [{"type": "string", "enum": ["OX", "OY", "OZ"]},
                          {"type": "string"}],
            },
            "step": {"type": "number"},
            "count": {"type": "integer"},
            "vector2": {"type": "string", "description": "optional second vector for 2D"},
            "step2": {"type": "number", "description": "optional second step"},
            "count2": {"type": "integer", "description": "optional second count"},
        },
        "required": ["shape", "vector", "step", "count"],
    },
    "multi_rotate": {
        "type": "object",
        "properties": {
            "shape": {"type": "string"},
            "axis": {
                "description": "OX/OY/OZ or axis name",
                "oneOf": [{"type": "string", "enum": ["OX", "OY", "OZ"]},
                          {"type": "string"}],
            },
            "count": {"type": "integer", "description": "number of copies"},
            "angle_deg": {"type": "number", "description": "optional angle step (degrees)"},
        },
        "required": ["shape", "axis", "count"],
    },
    "offset_shape": {
        "type": "object",
        "properties": {
            "shape": {"type": "string"},
            "offset": {"type": "number"},
        },
        "required": ["shape", "offset"],
    },
    "position": {
        "type": "object",
        "properties": {
            "shape": {"type": "string", "description": "shape to place (a copy is made)"},
            "origin": {"type": "array", "items": {"type": "number"},
                       "description": "target frame origin [x,y,z]; default [0,0,0]"},
            "x_dir": {"type": "array", "items": {"type": "number"},
                      "description": "target frame X-axis direction; default [1,0,0]"},
            "y_dir": {"type": "array", "items": {"type": "number"},
                      "description": "target frame Y-axis direction; default [0,1,0]"},
        },
        "required": ["shape"],
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
    "chamfer_edges": {
        "type": "object",
        "properties": {
            "shape": {"type": "string", "description": "published name of the shape to chamfer"},
            "d1": {"type": "number", "description": "first chamfer distance"},
            "d2": {"type": "number", "description": "second chamfer distance (defaults to d1)"},
            "edges": {
                "description": "\"all\" or a list of edge subshape indices",
                "oneOf": [{"type": "string", "enum": ["all"]},
                          {"type": "array", "items": {"type": "integer"}}],
            },
        },
        "required": ["shape", "d1", "edges"],
    },
    "set_color": {
        "type": "object",
        "properties": {
            "shape": {"type": "string"},
            "red": {"type": "integer", "minimum": 0, "maximum": 255},
            "green": {"type": "integer", "minimum": 0, "maximum": 255},
            "blue": {"type": "integer", "minimum": 0, "maximum": 255},
        },
        "required": ["shape", "red", "green", "blue"],
    },
    "set_transparency": {
        "type": "object",
        "properties": {
            "shape": {"type": "string"},
            "transparency": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        },
        "required": ["shape", "transparency"],
    },
    "set_display_mode": {
        "type": "object",
        "properties": {
            "shape": {"type": "string"},
            "mode": {"type": "integer", "enum": [0, 1, 2], "description": "0=wireframe, 1=shading, 2=shading+edges"},
        },
        "required": ["shape", "mode"],
    },
    "check_shape": {
        "type": "object",
        "properties": {"shape": {"type": "string"}},
        "required": ["shape"],
    },
    "list_study_objects": {"type": "object", "properties": {}},
}

# One line per tool: what it produces and the implicit placement/orientation
# defaults that the JSON Schema alone does not convey. Sent to the LLM as each
# function's "description". Keep every key in TOOLS listed here (guarded by a
# unit test).
TOOL_DESCRIPTIONS: dict[str, str] = {
    "make_box": "Axis-aligned solid box with one corner at the origin, extending +X/+Y/+Z by dx/dy/dz.",
    "make_box_two_pnt": "Axis-aligned solid box spanning two opposite corner points.",
    "make_cylinder": "Solid cylinder, base centred at the origin, axis along +Z.",
    "make_sphere": "Solid sphere centred at the origin (or at `center` if given).",
    "make_cone": "Solid (truncated) cone, base radius1 at the origin, radius2 at height, axis along +Z. radius2=0 for a sharp tip.",
    "make_torus": "Solid torus centred at the origin, axis along +Z; radius_major = ring centre-line, radius_minor = tube.",
    "make_face": "Planar rectangular face height x width. orientation 1=XOY, 2=YOZ, 3=ZOX.",
    "make_disk": "Planar circular face of the given radius. orientation 1=XOY, 2=YOZ, 3=ZOX.",
    "make_point": "A vertex at (x, y, z).",
    "make_vector": "A free vector with components (dx, dy, dz) from the origin.",
    "make_line": "An infinite line through two points.",
    "make_plane": "An unbounded plane through `point` with the given `normal` vector (trimmed for display at trim_size).",
    "make_circle": "A circular edge (wire), centre point, `normal` axis, radius.",
    "make_prism": "Linear extrusion of a face/wire/edge by the vector (dx, dy, dz).",
    "make_revolution": "Revolve a face/wire/edge about `axis` by angle_deg. The profile must not touch the axis.",
    "make_pipe": "Sweep a cross-section `base` along a `path` wire/edge.",
    "make_thick_solid": "Hollow a solid into a shell of wall thickness `thickness`. `faces` lists face subshape IDs to leave open (required -- a closed solid cannot be thickened).",
    "make_filling": "A surface filling a set of boundary contour edges/wires.",
    "fuse": "Boolean union of two or more shapes into one.",
    "cut": "Boolean subtraction: `main` minus every shape in `tools`.",
    "common": "Boolean intersection of two or more shapes.",
    "partition": "Split `shapes` by each other (and optional `tools`), keeping all resulting pieces in one compound.",
    "translate": "A COPY of `shape` moved by (dx, dy, dz); the source is unchanged -- refer to the new name afterwards.",
    "rotate": "A COPY of `shape` rotated about `axis` by angle_deg; the source is unchanged.",
    "mirror": "A COPY of `shape` mirrored across a plane, axis or point (`by` picks which; `entity` names it or a shortcut).",
    "scale": "A COPY of `shape` scaled about `center` -- uniform `factor`, or per-axis factor_x/y/z.",
    "multi_translate": "A COPY of `shape` replicated `count` times along `vector` at spacing `step` (add vector2/step2/count2 for a 2D grid).",
    "multi_rotate": "A COPY of `shape` replicated `count` times about `axis` (angle_deg = step, default full circle).",
    "offset_shape": "A COPY of `shape` with every face offset outward (+) or inward (-) by `offset`.",
    "position": "A COPY of `shape` with its local frame placed at target `origin` with axes `x_dir`/`y_dir` -- use to assemble parts.",
    "fillet_edges": "Round edges of `shape` with the given `radius`. `edges` = \"all\" or a list of edge subshape indices.",
    "chamfer_edges": "Bevel edges of `shape` by distances d1/d2. `edges` = \"all\" or a list of edge subshape indices.",
    "set_color": "Set the display colour of `shape` (r/g/b 0-255). Apply once, last, on the final shape.",
    "set_transparency": "Set display transparency of `shape` (0 opaque .. 1 invisible).",
    "set_display_mode": "Set display mode of `shape` (0=wireframe, 1=shading, 2=shading+edges).",
    "check_shape": "Report whether `shape` is a valid geometry, with a diagnostic message.",
    "list_study_objects": "List the names of every object currently published in the study.",
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


def _shape_info(geompy, shape) -> dict | None:
    """World-space bounding box, size and centre (bbox midpoint) of a shape,
    in mm, rounded. Spatial feedback for the specialist -- lets it catch a
    misplaced part ("my wheel is at the origin") without a rendered view."""
    if shape is None:
        return None
    try:
        xmin, xmax, ymin, ymax, zmin, zmax = geompy.BoundingBox(shape)
    except Exception:
        return None
    r = lambda v: round(v, 4)
    return {
        "bbox": [r(xmin), r(xmax), r(ymin), r(ymax), r(zmin), r(zmax)],
        "size": [r(xmax - xmin), r(ymax - ymin), r(zmax - zmin)],
        "centre": [r((xmin + xmax) / 2), r((ymin + ymax) / 2), r((zmin + zmax) / 2)],
    }


def _attach_shape_info(executor, result: dict) -> None:
    published = [n for rec in result.get("records", []) for n in (rec.get("publish") or [])]
    if not published:
        return
    name = published[-1]
    info = _shape_info(executor.namespace["geompy"], executor.geom_bindings.get(name))
    if info:
        result["result_shape"] = name
        result["shape_info"] = info


def call_tool(executor, name: str, args: dict) -> dict:
    handler = TOOLS.get(name)
    if handler is None:
        return {"ok": False, "error": f"unknown tool {name!r}"}
    try:
        args = _coerce_args(name, args)
        result = handler(executor, args)
        if result.get("ok"):
            _attach_shape_info(executor, result)
        return result
    except Exception as e:
        return {"ok": False, "error": str(e)}


def list_tools() -> list[dict]:
    return [{"name": n,
             "parameters": TOOL_SCHEMAS.get(n, {"type": "object", "properties": {}}),
             "description": TOOL_DESCRIPTIONS.get(n, "")}
            for n in TOOLS]
