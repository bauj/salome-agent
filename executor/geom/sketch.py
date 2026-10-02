import math

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


def _not_a_plane(coords) -> str | None:
    """Why these corners cannot bound one planar face, or None if they can."""
    o = coords[0]
    vecs = [tuple(c[i] - o[i] for i in range(3)) for c in coords[1:]]
    scale = max((abs(v) for vec in vecs for v in vec), default=0.0) or 1.0
    normal = None
    for a in vecs:
        for b in vecs:
            n = (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])
            if max(abs(x) for x in n) > 1e-9 * scale * scale:
                normal = n
                break
        if normal:
            break
    if normal is None:
        return "the points are collinear, they span no plane"
    length = sum(x * x for x in normal) ** 0.5
    for vec in vecs:
        if abs(sum(vec[i] * normal[i] for i in range(3))) / length > 1e-6 * scale:
            return "the points do not all lie in the same plane"
    return None


@tool("make_polygon_face", category="sketch",
      description=('Planar face bounded by a closed polygon through `points` (3 or more corners, '
                   'in order; the last joins back to the first). Extrude it with make_prism for a '
                   'gable roof, wedge or any prismatic profile. Straight sides only: for an '
                   'outline with arcs or curves use make_profile_face.'),
      schema={"type": "object", "properties": {"points": {"type": "array", "items": {"type": "string"}, "minItems": 3, "description": "corners in order: point names or \"x,y,z\""}}, "required": ["points"]},
      arg_types={})
def tool_make_polygon_face(executor, args: dict):
    if len(args["points"]) < 3:
        return {"ok": False, "error": "make_polygon_face: needs at least 3 points"}
    geompy = executor.namespace["geompy"]
    records: list = []
    pts = [_resolve_point(executor, p, records) for p in args["points"]]
    # MakeFaceWires(..., isPlanarWanted=1) does not refuse a non-planar wire —
    # seen live, it quietly builds a curved face — so check the corners here.
    problem = _not_a_plane([geompy.PointCoordinates(executor.geom_bindings[n]) for n in pts])
    if problem:
        return {"ok": False, "error": f"make_polygon_face: {problem}"}
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


@tool("make_arc", category="sketch",
      description=('A circular arc edge, in one of two forms: `center` + `start` + `end` (radius = '
                   'centre to start; `end` only sets the direction the arc stops at; it takes the '
                   'shorter way round, `reverse` the longer; for a half circle use the other form) '
                   'or `start` + `through` + `end` (the arc passes through all three). Points are names or "x,y,z". The result gives the arc\'s real end points '
                   'and length: start the next segment there. Chain arcs and make_line segments with '
                   'make_wire, then make_face_from_wires; for a whole outline make_profile_face '
                   'does it in one call.'),
      schema={"type": "object", "properties": {"start": {"type": "string", "description": "start point"}, "end": {"type": "string", "description": "end point (centre form: sets the direction only)"}, "center": {"type": "string", "description": "centre point (centre form)"}, "through": {"type": "string", "description": "a point the arc passes through, between start and end (3-point form)"}, "reverse": {"type": "boolean", "default": False, "description": "centre form: take the longer way round"}}, "required": ["start", "end"]},
      arg_types={})
def tool_make_arc(executor, args: dict):
    if ("center" in args) == ("through" in args):
        return {"ok": False, "error": ("make_arc: give either `center` (centre form) or `through` "
                                       "(3-point form), not both or neither")}
    geompy = executor.namespace["geompy"]
    records: list = []
    sn = _resolve_point(executor, args["start"], records)
    en = _resolve_point(executor, args["end"], records)
    b = executor.geom_bindings
    s, e = geompy.PointCoordinates(b[sn]), geompy.PointCoordinates(b[en])
    if math.dist(s, e) < 1e-9:
        return {"ok": False, "error": ("make_arc: start and end are the same point; for a full "
                                       "circle use make_circle")}
    if "center" in args:
        cn = _resolve_point(executor, args["center"], records)
        if math.dist(geompy.PointCoordinates(b[cn]), s) < 1e-9:
            return {"ok": False, "error": "make_arc: the centre is on the start point (radius 0)"}
        reverse = bool(args.get("reverse", False))
        arc = geompy.MakeArcCenter(b[cn], b[sn], b[en], reverse)
        fn = "MakeArcCenter"
        call_args = [{"kind": "ref", "value": cn}, {"kind": "ref", "value": sn},
                     {"kind": "ref", "value": en}, {"kind": "literal", "value": reverse}]
    else:
        mn = _resolve_point(executor, args["through"], records)
        if _not_a_plane([s, geompy.PointCoordinates(b[mn]), e]):
            return {"ok": False, "error": ("make_arc: start, through and end are collinear, no "
                                           "arc passes through them")}
        arc = geompy.MakeArc(b[sn], b[mn], b[en])
        fn = "MakeArc"
        call_args = [{"kind": "ref", "value": n} for n in (sn, mn, en)]
    name = _next_name(executor, "arc")
    _publish(executor, arc, name)
    records.append({"fn": fn, "args": call_args, "out_vars": [name], "publish": [name]})
    ends = [[round(v, 6) + 0.0 for v in geompy.PointCoordinates(geompy.MakeVertexOnCurve(arc, t))]
            for t in (0.0, 1.0)]
    return {"ok": True, "records": records, "sanity": {"ok": True},
            "arc": {"start": ends[0], "end": ends[1],
                    "length": round(geompy.BasicProperties(arc)[0], 6)}}


def _loose_ends(edge_ends, tol: float = 1e-6) -> list:
    """End points that only one edge reaches. `edge_ends`: per edge, the
    coordinates of its vertices (one vertex for a closed edge like a circle)."""
    groups: list = []   # [point, how many edge ends meet there]
    for ends in edge_ends:
        if len(ends) < 2:
            continue
        for p in ends:
            for g in groups:
                if math.dist(g[0], p) <= tol:
                    g[1] += 1
                    break
            else:
                groups.append([tuple(p), 1])
    return [g[0] for g in groups if g[1] % 2]


def _shape_edge_ends(geompy, shape) -> list:
    kinds = geompy.ShapeType
    return [[geompy.PointCoordinates(v) for v in geompy.SubShapeAll(edge, kinds["VERTEX"])]
            for edge in geompy.SubShapeAll(shape, kinds["EDGE"])]


def _fmt_point(p) -> str:
    return str([round(v, 6) for v in p])


@tool("make_wire", category="sketch",
      description=('Join edges (make_line, make_arc, ... segments) end to end into one wire. The '
                   'result says whether it is closed; a closed wire can be made into a face with '
                   'make_face_from_wires.'),
      schema={"type": "object", "properties": {"edges": {"type": "array", "items": {"type": "string"}, "minItems": 1, "description": "edges or wires, each touching the next"}}, "required": ["edges"]},
      arg_types={})
def tool_make_wire(executor, args: dict):
    names = args["edges"]
    if not names:
        return {"ok": False, "error": "make_wire: needs at least one edge"}
    geompy = executor.namespace["geompy"]
    shapes = [executor.geom_bindings[n] for n in names]
    loose = _loose_ends([ends for s in shapes for ends in _shape_edge_ends(geompy, s)])
    tolerance = 1e-07
    try:
        wire = geompy.MakeWire(shapes, tolerance)
    except Exception as e:
        gaps = sorted((math.dist(a, b), a, b) for i, a in enumerate(loose) for b in loose[i + 1:])
        where = (f"; the closest loose ends are {_fmt_point(gaps[0][1])} and "
                 f"{_fmt_point(gaps[0][2])}, {gaps[0][0]:.6g} apart — make each segment start "
                 "exactly where the previous one ends") if gaps else ""
        return {"ok": False, "error": f"make_wire: the edges do not join up ({e}){where}"}
    name = _next_name(executor, "wire")
    _publish(executor, wire, name)
    result = {"ok": True, "sanity": {"ok": True}, "closed": not loose,
              "records": [{"fn": "MakeWire",
                           "args": [{"kind": "list", "value": [{"kind": "ref", "value": n} for n in names]},
                                    {"kind": "literal", "value": tolerance}],
                           "out_vars": [name], "publish": [name]}]}
    if loose:
        result["loose_ends"] = [[round(v, 6) for v in p] for p in loose]
    return result


_PLANAR_KINDS = {"PLANE", "PLANAR", "POLYGON", "DISK_CIRCLE", "DISK_ELLIPSE"}


def _planar_face(geompy, wires, tool: str):
    """(face, None) for a planar face bounded by `wires`, else (None, error)."""
    try:
        face = geompy.MakeFaceWires(wires, 1)
    except Exception as e:
        return None, (f"{tool}: no face ({e}) — the wires must be closed, in one plane, holes "
                      "inside the outer contour")
    # Like MakePolyline faces, MakeFaceWires(..., 1) quietly builds a curved
    # face from wires that are not in one plane, so check what it made.
    kind = str(geompy.KindOfShape(face)[0]).rsplit(".", 1)[-1]
    if kind not in _PLANAR_KINDS:
        return None, f"{tool}: the wires do not lie in one plane (GEOM built a {kind} face)"
    is_valid, msg = geompy.CheckShape(face, True, 1)
    if not is_valid:
        return None, f"{tool}: CheckShape failed ({msg})"
    return face, None


@tool("make_face_from_wires", category="sketch",
      description=('Planar face bounded by closed wires: the first is the outer contour, any others '
                   'are holes (make_wire results, or a make_circle edge for a round hole). Extrude '
                   'it with make_prism or revolve it with make_revolution.'),
      schema={"type": "object", "properties": {"wires": {"type": "array", "items": {"type": "string"}, "minItems": 1, "description": "outer contour first, then holes"}}, "required": ["wires"]},
      arg_types={})
def tool_make_face_from_wires(executor, args: dict):
    names = args["wires"]
    if not names:
        return {"ok": False, "error": "make_face_from_wires: needs at least one wire"}
    geompy = executor.namespace["geompy"]
    shapes = [executor.geom_bindings[n] for n in names]
    for n, s in zip(names, shapes):
        loose = _loose_ends(_shape_edge_ends(geompy, s))
        if loose:
            return {"ok": False, "error": (f"make_face_from_wires: wire {n} is open, loose ends at "
                                           f"{', '.join(_fmt_point(p) for p in loose)} — close it "
                                           "before making a face")}
    face, error = _planar_face(geompy, shapes, "make_face_from_wires")
    if error:
        return {"ok": False, "error": error}
    name = _next_name(executor, "face")
    _publish(executor, face, name)
    return {"ok": True, "sanity": {"ok": True},
            "area": round(geompy.BasicProperties(face)[1], 6),
            "records": [{"fn": "MakeFaceWires",
                         "args": [{"kind": "list", "value": [{"kind": "ref", "value": n} for n in names]},
                                  {"kind": "literal", "value": 1}],
                         "out_vars": [name], "publish": [name]}]}
