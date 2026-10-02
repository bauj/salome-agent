import math

from executor.geom.helpers import (
    tool, _next_name, _parse_coords, _publish, _resolve_axis, _resolve_point,
)
from executor.geom.sketch import (
    _fmt_point, _loose_ends, _not_a_plane, _planar_face, _shape_edge_ends,
)

_SAME = 1e-7


def _ref(name):
    return {"kind": "ref", "value": name}


def _lit(value):
    return {"kind": "literal", "value": value}


def _refs(names):
    return {"kind": "list", "value": [_ref(n) for n in names]}


def _edge_ends(geompy, edge) -> list:
    return [[round(v, 6) + 0.0 for v in geompy.PointCoordinates(geompy.MakeVertexOnCurve(edge, t))]
            for t in (0.0, 1.0)]


def _coords(executor, arg):
    lit = (0.0, 0.0, 0.0) if arg == "O" else _parse_coords(arg)
    if lit is not None:
        return lit
    return tuple(executor.namespace["geompy"].PointCoordinates(executor.geom_bindings[arg]))


@tool("make_ellipse", category="sketch",
      description=('An ellipse edge, in one of two forms: `center` + `normal` + `r_major` + '
                   '`r_minor` (a full ellipse; `major_dir` orients the major axis) or `center` + '
                   '`start` + `end` (a quarter ellipse from `start` to `end`, two points at right '
                   'angles from the centre at the ends of the semi-axes). Points are names or '
                   '"x,y,z".'),
      schema={"type": "object", "properties": {
          "center": {"type": "string", "description": "centre point"},
          "normal": {"type": "string", "description": "normal of the ellipse plane (full form)"},
          "r_major": {"type": "number", "description": "major radius (full form)"},
          "r_minor": {"type": "number", "description": "minor radius, <= r_major (full form)"},
          "major_dir": {"type": "string", "description": "direction of the major axis (full form)"},
          "start": {"type": "string", "description": "end of one semi-axis (quarter form)"},
          "end": {"type": "string", "description": "end of the other semi-axis (quarter form)"}},
          "required": ["center"]},
      arg_types={"r_major": float, "r_minor": float})
def tool_make_ellipse(executor, args: dict):
    full = [k for k in ("normal", "r_major", "r_minor") if k in args]
    quarter = [k for k in ("start", "end") if k in args]
    if (len(full) == 3) == (len(quarter) == 2) or (full and quarter):
        return {"ok": False, "error": ("make_ellipse: give either `normal` + `r_major` + `r_minor` "
                                       "(full ellipse) or `start` + `end` (quarter ellipse)")}
    geompy = executor.namespace["geompy"]
    records: list = []
    b = executor.geom_bindings
    cn = _resolve_point(executor, args["center"], records)
    if full:
        r_major, r_minor = args["r_major"], args["r_minor"]
        if r_minor <= 0:
            return {"ok": False, "error": "make_ellipse: the radii must be positive"}
        if r_minor > r_major:
            return {"ok": False, "error": (f"make_ellipse: r_minor {r_minor:g} is bigger than "
                                           f"r_major {r_major:g}; swap them and turn `major_dir` "
                                           "a quarter turn")}
        vn = _resolve_axis(executor, args["normal"], records)
        call_args = [_ref(cn), _ref(vn), _lit(r_major), _lit(r_minor)]
        shapes = [b[cn], b[vn], r_major, r_minor]
        if "major_dir" in args:
            mn = _resolve_axis(executor, args["major_dir"], records)
            call_args.append(_ref(mn))
            shapes.append(b[mn])
        edge = geompy.MakeEllipse(*shapes)
        fn = "MakeEllipse"
    else:
        sn = _resolve_point(executor, args["start"], records)
        en = _resolve_point(executor, args["end"], records)
        c, s, e = (geompy.PointCoordinates(b[n]) for n in (cn, sn, en))
        a = [s[i] - c[i] for i in range(3)]
        d = [e[i] - c[i] for i in range(3)]
        la, ld = math.hypot(*a), math.hypot(*d)
        if la < _SAME or ld < _SAME:
            return {"ok": False, "error": "make_ellipse: `start` or `end` is on the centre"}
        cos = sum(a[i] * d[i] for i in range(3)) / (la * ld)
        if abs(cos) > 1e-6:
            angle = math.degrees(math.acos(max(-1.0, min(1.0, cos))))
            return {"ok": False, "error": (f"make_ellipse: `start` and `end` are {angle:.6g} "
                                           "degrees apart seen from the centre; a quarter ellipse "
                                           "needs them at 90 (the ends of its two semi-axes)")}
        edge = geompy.MakeArcOfEllipse(b[cn], b[sn], b[en])
        fn = "MakeArcOfEllipse"
        call_args = [_ref(cn), _ref(sn), _ref(en)]
    name = _next_name(executor, "ellipse")
    _publish(executor, edge, name)
    records.append({"fn": fn, "args": call_args, "out_vars": [name], "publish": [name]})
    info = {"length": round(geompy.BasicProperties(edge)[0], 6)}
    if quarter:
        info["start"], info["end"] = _edge_ends(geompy, edge)
    return {"ok": True, "records": records, "sanity": {"ok": True}, "ellipse": info}


def _repeated_point(coords) -> str | None:
    for i, (p, q) in enumerate(zip(coords, coords[1:]), 1):
        if math.dist(p, q) <= _SAME:
            return f"points {i} and {i + 1} are the same point {_fmt_point(p)}"
    return None


@tool("make_polyline", category="sketch",
      description=('A wire of straight segments through `points` in order; `closed` joins the last '
                   'back to the first. Open, it is a path for make_pipe or a piece for make_wire; '
                   'for a filled polygon use make_polygon_face.'),
      schema={"type": "object", "properties": {
          "points": {"type": "array", "items": {"type": "string"}, "minItems": 2, "description": "point names or \"x,y,z\", in order"},
          "closed": {"type": "boolean", "default": False}},
          "required": ["points"]},
      arg_types={})
def tool_make_polyline(executor, args: dict):
    if len(args["points"]) < 2:
        return {"ok": False, "error": "make_polyline: needs at least 2 points"}
    geompy = executor.namespace["geompy"]
    records: list = []
    pts = [_resolve_point(executor, p, records) for p in args["points"]]
    problem = _repeated_point([geompy.PointCoordinates(executor.geom_bindings[n]) for n in pts])
    if problem:
        return {"ok": False, "error": f"make_polyline: {problem}"}
    closed = bool(args.get("closed", False))
    wire = geompy.MakePolyline([executor.geom_bindings[n] for n in pts], closed)
    name = _next_name(executor, "polyline")
    _publish(executor, wire, name)
    records.append({"fn": "MakePolyline", "args": [_refs(pts), _lit(closed)],
                    "out_vars": [name], "publish": [name]})
    return {"ok": True, "records": records, "sanity": {"ok": True},
            "length": round(geompy.BasicProperties(wire)[0], 6)}


@tool("make_curve", category="sketch",
      description=('A smooth curve edge from `points`: kind "interpolation" (default) passes '
                   'through every point, "bezier" only through the first and last (the others pull '
                   'it like control points). `closed` makes a loop; `start_tangent` + `end_tangent` '
                   '(directions, open interpolation only) set the direction at the ends. The result '
                   'gives its real end points and length.'),
      schema={"type": "object", "properties": {
          "points": {"type": "array", "items": {"type": "string"}, "minItems": 2, "description": "point names or \"x,y,z\", in order"},
          "kind": {"type": "string", "enum": ["interpolation", "bezier"], "default": "interpolation"},
          "closed": {"type": "boolean", "default": False},
          "start_tangent": {"type": "string", "description": "direction at the first point"},
          "end_tangent": {"type": "string", "description": "direction at the last point"}},
          "required": ["points"]},
      arg_types={})
def tool_make_curve(executor, args: dict):
    kind = args.get("kind", "interpolation")
    if kind not in ("interpolation", "bezier"):
        return {"ok": False, "error": f"make_curve: kind must be \"interpolation\" or \"bezier\", not {kind!r}"}
    if len(args["points"]) < 2:
        return {"ok": False, "error": "make_curve: needs at least 2 points"}
    closed = bool(args.get("closed", False))
    tangents = [k for k in ("start_tangent", "end_tangent") if k in args]
    if tangents and (len(tangents) == 1 or kind != "interpolation" or closed):
        return {"ok": False, "error": ("make_curve: `start_tangent` and `end_tangent` go together, "
                                       "on an open interpolation curve")}
    geompy = executor.namespace["geompy"]
    records: list = []
    b = executor.geom_bindings
    pts = [_resolve_point(executor, p, records) for p in args["points"]]
    problem = _repeated_point([geompy.PointCoordinates(b[n]) for n in pts])
    if problem:
        return {"ok": False, "error": f"make_curve: {problem}"}
    shapes = [b[n] for n in pts]
    if tangents:
        t1 = _resolve_axis(executor, args["start_tangent"], records)
        t2 = _resolve_axis(executor, args["end_tangent"], records)
        edge = geompy.MakeInterpolWithTangents(shapes, b[t1], b[t2])
        fn, call_args = "MakeInterpolWithTangents", [_refs(pts), _ref(t1), _ref(t2)]
    elif kind == "bezier":
        edge = geompy.MakeBezier(shapes, closed)
        fn, call_args = "MakeBezier", [_refs(pts), _lit(closed)]
    else:
        edge = geompy.MakeInterpol(shapes, closed, False)
        fn, call_args = "MakeInterpol", [_refs(pts), _lit(closed), _lit(False)]
    name = _next_name(executor, "curve")
    _publish(executor, edge, name)
    records.append({"fn": fn, "args": call_args, "out_vars": [name], "publish": [name]})
    start, end = _edge_ends(geompy, edge)
    return {"ok": True, "records": records, "sanity": {"ok": True},
            "curve": {"start": start, "end": end,
                      "length": round(geompy.BasicProperties(edge)[0], 6)}}


class _ProfileError(Exception):
    pass


def _plane_normal(coords):
    """Unit normal of the plane through `coords`, or None if they are collinear."""
    o = coords[0]
    vecs = [tuple(c[i] - o[i] for i in range(3)) for c in coords[1:]]
    scale = max((abs(v) for vec in vecs for v in vec), default=0.0) or 1.0
    best = None
    for a in vecs:
        for b in vecs:
            n = (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])
            length = math.hypot(*n)
            if length > 1e-9 * scale * scale and (best is None or length > best[0]):
                best = (length, n)
    if best is None:
        return None
    return tuple(round(v / best[0], 9) + 0.0 for v in best[1])


class _Tracer:
    """Builds the edges of one profile's outlines, sharing vertices between them."""

    def __init__(self, executor, records):
        self.ex, self.records = executor, records
        self.g = executor.namespace["geompy"]
        self.known: list = []     # (coords, vertex name) of points this call made or used
        self.used: list = []      # every point the outer outline uses, for its plane
        self.fillets: list = []   # (corner coords, radius) to round once the face exists

    def point(self, arg, coords=None):
        coords = coords or _coords(self.ex, arg)
        for c, n in self.known:
            if math.dist(c, coords) <= _SAME:
                return n, c
        name = _resolve_point(self.ex, arg, self.records)
        self.known.append((coords, name))
        return name, coords

    def edge(self, fn, call_args, shape) -> str:
        name = _next_name(self.ex, "edge")
        self.ex.geom_bindings[name] = shape
        self.records.append({"fn": fn, "args": call_args, "out_vars": [name], "publish": None})
        return name

    def outline(self, outline, label: str) -> tuple[str, bool]:
        """(wire name, whether a straight line had to close it)."""
        if not isinstance(outline, dict) or "start" not in outline:
            raise _ProfileError(f"{label}: needs a `start` point")
        segments = outline.get("segments") or []
        if not segments:
            raise _ProfileError(f"{label}: needs at least one segment")
        b = self.ex.geom_bindings
        start_n, start_c = self.point(outline["start"])
        cur_n, cur_c = start_n, start_c
        self.used.append(start_c)
        edges = []
        for i, seg in enumerate(segments, 1):
            where = f"{label} segment {i}"
            if not isinstance(seg, dict) or "to" not in seg:
                raise _ProfileError(f"{where}: needs a `to` point")
            forms = [k for k in ("center", "through", "via") if k in seg]
            if len(forms) > 1:
                raise _ProfileError(f"{where}: give at most one of `center`, `through`, `via`")
            to_c = _coords(self.ex, seg["to"])
            if math.dist(to_c, cur_c) <= _SAME:
                raise _ProfileError(f"{where}: ends where it starts, at {_fmt_point(cur_c)}")
            try:
                if not forms:
                    to_n, to_c = self.point(seg["to"], to_c)
                    shape = self.g.MakeLineTwoPnt(b[cur_n], b[to_n])
                    edges.append(self.edge("MakeLineTwoPnt", [_ref(cur_n), _ref(to_n)], shape))
                elif forms == ["center"]:
                    c_n, c_c = self.point(seg["center"])
                    self.used.append(c_c)
                    r, d = math.dist(c_c, cur_c), math.dist(c_c, to_c)
                    if r <= _SAME or d <= _SAME:
                        raise _ProfileError(f"{where}: the centre is on an end of the arc")
                    if abs(d - r) > _SAME:
                        on = [round(c_c[k] + (to_c[k] - c_c[k]) * r / d, 9) for k in range(3)]
                        if abs(d - r) > 1e-3 * r:
                            raise _ProfileError(
                                f"{where}: `to` {_fmt_point(to_c)} is {d:.6g} from the centre but "
                                f"the arc's radius (centre to where the segment starts, "
                                f"{_fmt_point(cur_c)}) is {r:.6g}; the point at radius {r:.6g} in "
                                f"that direction is {_fmt_point(on)}")
                        to_n, to_c = self.point(on, tuple(on))
                    else:
                        to_n, to_c = self.point(seg["to"], to_c)
                    u = [cur_c[k] - c_c[k] for k in range(3)]
                    v = [to_c[k] - c_c[k] for k in range(3)]
                    if sum(u[k] * v[k] for k in range(3)) < 0 and _not_a_plane([c_c, cur_c, to_c]):
                        raise _ProfileError(f"{where}: a half circle has no shorter way round; "
                                            "give `through` (a point on the arc) instead of `center`")
                    reverse = bool(seg.get("reverse", False))
                    shape = self.g.MakeArcCenter(b[c_n], b[cur_n], b[to_n], reverse)
                    edges.append(self.edge("MakeArcCenter",
                                           [_ref(c_n), _ref(cur_n), _ref(to_n), _lit(reverse)], shape))
                elif forms == ["through"]:
                    m_n, m_c = self.point(seg["through"])
                    self.used.append(m_c)
                    if _not_a_plane([cur_c, m_c, to_c]):
                        raise _ProfileError(f"{where}: its start, `through` and `to` are collinear, "
                                            "no arc passes through them")
                    to_n, to_c = self.point(seg["to"], to_c)
                    shape = self.g.MakeArc(b[cur_n], b[m_n], b[to_n])
                    edges.append(self.edge("MakeArc", [_ref(cur_n), _ref(m_n), _ref(to_n)], shape))
                else:
                    via = seg["via"]
                    if not isinstance(via, list) or not via:
                        raise _ProfileError(f"{where}: `via` must be a list of points")
                    via_n = []
                    for p in via:
                        n, c = self.point(p)
                        via_n.append(n)
                        self.used.append(c)
                    to_n, to_c = self.point(seg["to"], to_c)
                    names = [cur_n, *via_n, to_n]
                    shape = self.g.MakeInterpol([b[n] for n in names], False, False)
                    edges.append(self.edge("MakeInterpol",
                                           [_refs(names), _lit(False), _lit(False)], shape))
            except (_ProfileError, KeyError):
                raise
            except Exception as e:
                raise _ProfileError(f"{where}: {e}") from e
            if "fillet" in seg:
                r = seg["fillet"]
                if isinstance(r, bool) or not isinstance(r, (int, float)) or r <= 0:
                    raise _ProfileError(f"{where}: `fillet` must be a positive radius")
                self.fillets.append((to_c, float(r)))
            self.used.append(to_c)
            cur_n, cur_c = to_n, to_c
        closed_with_line = math.dist(cur_c, start_c) > _SAME
        if closed_with_line:
            shape = self.g.MakeLineTwoPnt(b[cur_n], b[start_n])
            edges.append(self.edge("MakeLineTwoPnt", [_ref(cur_n), _ref(start_n)], shape))
        tolerance = 1e-07
        try:
            wire = self.g.MakeWire([b[n] for n in edges], tolerance)
        except Exception as e:
            raise _ProfileError(f"{label}: the segments do not join into one wire ({e})") from e
        loose = _loose_ends(_shape_edge_ends(self.g, wire))
        if loose:
            raise _ProfileError(f"{label}: the outline does not close, loose ends at "
                                f"{', '.join(_fmt_point(p) for p in loose)}")
        name = _next_name(self.ex, "wire")
        b[name] = wire
        self.records.append({"fn": "MakeWire", "args": [_refs(edges), _lit(tolerance)],
                             "out_vars": [name], "publish": None})
        return name, closed_with_line


def _fillet_corners(executor, face, name, wires, fillets, records):
    """Round the traced corners with MakeFillet2D, one call per radius, so OCC
    finds the tangent points. Records the plain face and each step; the last
    shape gets `name`. Returns (shape, error)."""
    geompy = executor.namespace["geompy"]
    radii = list(dict.fromkeys(r for _, r in fillets))
    names = [_next_name(executor, "face") for _ in radii] + [name]
    shapes = [face]
    for i, r in enumerate(radii):
        corners = [c for c, rr in fillets if rr == r]
        verts = geompy.SubShapeAll(shapes[-1], geompy.ShapeType["VERTEX"])
        ids = []
        for c in corners:
            match = [v for v in verts if math.dist(geompy.PointCoordinates(v), c) <= 1e-6]
            if not match:
                return None, f"make_profile_face: no corner left at {_fmt_point(c)} to fillet"
            ids.append(geompy.GetSubShapeID(shapes[-1], match[0]))
        try:
            shapes.append(geompy.MakeFillet2D(shapes[-1], r, ids))
        except Exception as e:
            where = ", ".join(_fmt_point(c) for c in corners)
            return None, (f"make_profile_face: fillet {r:g} at {where} failed ({e}): the radius "
                          "is too big for the segments meeting there, or they already join smoothly")
        records.append({"fn": "MakeFillet2D", "args": [_ref(names[i]), _lit(r), _lit(ids)],
                        "out_vars": [names[i + 1]], "publish": None})
    records.insert(len(records) - len(radii),
                   {"fn": "MakeFaceWires", "args": [_refs(wires), _lit(1)],
                    "out_vars": [names[0]], "publish": None})
    records[-1]["publish"] = [name]
    for n, shape in zip(names[:-1], shapes[:-1]):
        executor.geom_bindings[n] = shape
    return shapes[-1], None


_POINT = {"type": "string", "description": "point name or \"x,y,z\""}
_OUTLINE = {"type": "object", "properties": {
    "start": _POINT,
    "segments": {"type": "array", "minItems": 1, "items": {"type": "object", "properties": {
        "to": {"type": "string", "description": "where the segment ends"},
        "center": {"type": "string", "description": "circular arc about this centre (shorter way round)"},
        "reverse": {"type": "boolean", "default": False, "description": "centre arc: the longer way round"},
        "through": {"type": "string", "description": "circular arc through this point"},
        "via": {"type": "array", "items": {"type": "string"}, "description": "smooth curve through these points"},
        "fillet": {"type": "number", "description": "round the corner at `to` with this radius"}},
        "required": ["to"]}}},
    "required": ["start", "segments"]}


@tool("make_profile_face", category="sketch",
      description=('Planar face from an outline traced from `start` through `segments`, each '
                   'starting where the previous one ends and ending at its `to`: a straight line, '
                   'or a circular arc with `center` (shorter way round, `reverse` for the longer; '
                   'for a half circle give `through` instead) or `through` (a point on the arc), '
                   'or a smooth curve through `via` points. `fillet` rounds the corner at a segment\'s '
                   '`to` with a tangent arc of that radius: give corners as drawn and the tool '
                   'finds the tangent points (end the last segment at `start` to round that one). '
                   'If the last `to` is not back at `start` '
                   'a straight line closes it. `holes`: outlines of the same form, '
                   '{"center", "radius"} circles, or {"wire": name} of a closed wire. The outline and '
                   'each hole may run either way round. Builds edges, wire and '
                   'face in one call: prefer it to make_line/make_arc/make_wire/make_face_from_wires '
                   'for a whole outline. Extrude it with make_prism or revolve it with '
                   'make_revolution.'),
      schema={"type": "object", "properties": {
          "start": _OUTLINE["properties"]["start"],
          "segments": _OUTLINE["properties"]["segments"],
          "holes": {"type": "array", "items": {"type": "object", "properties": {
              "start": _POINT,
              "segments": _OUTLINE["properties"]["segments"],
              "center": {"type": "string", "description": "round hole centre"},
              "radius": {"type": "number", "description": "round hole radius"},
              "wire": {"type": "string", "description": "name of an existing closed wire"}}},
              "description": "an outline ({start, segments}), a circle ({center, radius}) or {wire}"}},
          "required": ["start", "segments"]},
      arg_types={})
def tool_make_profile_face(executor, args: dict):
    geompy = executor.namespace["geompy"]
    records: list = []
    tracer = _Tracer(executor, records)
    b = executor.geom_bindings
    closed_by_line = []
    try:
        outer, by_line = tracer.outline(args, "outline")
        if by_line:
            closed_by_line.append("outline")
        wires = [outer]
        normal = None
        for i, hole in enumerate(args.get("holes") or [], 1):
            label = f"hole {i}"
            if isinstance(hole, str):
                hole = {"wire": hole}
            if not isinstance(hole, dict):
                raise _ProfileError(f"{label}: give {{start, segments}}, {{center, radius}} or {{wire}}")
            if "segments" in hole:
                name, by_line = tracer.outline(hole, label)
                if by_line:
                    closed_by_line.append(label)
            elif "radius" in hole:
                if "center" not in hole or not hole["radius"] or float(hole["radius"]) <= 0:
                    raise _ProfileError(f"{label}: a round hole needs `center` and a positive `radius`")
                if normal is None:
                    normal = _plane_normal(tracer.used)
                    if normal is None:
                        raise _ProfileError("the outline is a straight line, it has no plane "
                                            "to put a round hole in")
                    normal_n = _resolve_axis(executor, list(normal), records)
                c_n, _ = tracer.point(hole["center"])
                radius = float(hole["radius"])
                name = _next_name(executor, "circ")
                b[name] = geompy.MakeCircle(b[c_n], b[normal_n], radius)
                records.append({"fn": "MakeCircle", "args": [_ref(c_n), _ref(normal_n), _lit(radius)],
                                "out_vars": [name], "publish": None})
            elif "wire" in hole:
                name = hole["wire"]
                loose = _loose_ends(_shape_edge_ends(geompy, b[name]))
                if loose:
                    raise _ProfileError(f"{label}: wire {name} is open, loose ends at "
                                        f"{', '.join(_fmt_point(p) for p in loose)}")
            else:
                raise _ProfileError(f"{label}: give {{start, segments}}, {{center, radius}} or {{wire}}")
            wires.append(name)
    except _ProfileError as e:
        return {"ok": False, "error": f"make_profile_face: {e}"}
    face, error = _planar_face(geompy, [b[n] for n in wires], "make_profile_face")
    if error:
        return {"ok": False, "error": error}
    name = _next_name(executor, "face")
    face, error = _fillet_corners(executor, face, name, wires, tracer.fillets, records)
    if error:
        return {"ok": False, "error": error}
    _publish(executor, face, name)
    result = {"ok": True, "records": records, "sanity": {"ok": True},
              "area": round(geompy.BasicProperties(face)[1], 6)}
    if closed_by_line:
        result["closed_with_line"] = closed_by_line
    return result
