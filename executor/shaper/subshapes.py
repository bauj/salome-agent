import math

from executor.shaper.features import _build, _operands
from executor.shaper.primitives import _num
from executor.shaper.state import (SUB_TYPES, Sub, forget, new_name, raw_box, results_of, state,
                                   subshapes, tool)

_MAX_EXPLODE = 50
_MEET = 0.95   # OCC segfaults when two fillets on one face meet (2r ~ face width)
# OCC's fillet (ChFi3d) carries a fillet on along edges it finds smooth: faces whose
# normals differ by at most _G1 across the edge (ChFi3d::IsTangentFaces), or by less
# than _FADE at a vertex (ChFi3d_FilBuilder's default angular tolerance).
_G1, _FADE = 0.1, 0.01
_BAND = 0.1    # a sampled curved-face angle within 10% of a limit is not decided
_USED = " `shape` is used up: refer to the result by its new name."

_WHERE = {"description": (
    "\"all\"; {\"sorted_by\": \"x|y|z\", \"take\": \"max|min\", \"count\": N (default 1)}; "
    "{\"by_size\": \"longest|shortest|largest|smallest\", \"count\": N (default 1)}; or "
    "{\"on_plane\": {\"point\": [x, y, z], \"normal\": [x, y, z]}, \"state\": "
    "\"on|in|out|onin|onout\"} (in = the side the normal points away from)")}
_IDS = {"description": "\"all\" or a list of ids from get_subshapes (\"E3\", \"F2\", ...)",
        "anyOf": [{"type": "string", "enum": ["all"]},
                  {"type": "array", "items": {"type": "string"}}]}


def _shape_name(executor, args: dict, field: str = "shape") -> str:
    name = args.get(field)
    if not isinstance(name, str):
        raise ValueError(f"{field} must be one published name, got {name!r}")
    _operands(executor, [name], field)
    return name


def _plane_test(where: dict):
    op = where.get("on_plane")
    if not isinstance(op, dict):
        raise ValueError("on_plane must be {point: [x, y, z], normal: [x, y, z]}")
    p0, n = op.get("point"), op.get("normal")
    if not (isinstance(p0, list) and isinstance(n, list) and len(p0) == len(n) == 3):
        raise ValueError("on_plane needs point [x, y, z] and normal [x, y, z]")
    p0 = [_num(v, "on_plane point") for v in p0]
    n = [_num(v, "on_plane normal") for v in n]
    if not any(n):
        raise ValueError("on_plane needs point [x, y, z] and a non-zero normal [x, y, z]")
    norm = sum(c * c for c in n) ** 0.5
    side = lambda p: sum((p[i] - p0[i]) * n[i] for i in range(3)) / norm
    state_ = where.get("state", "on")
    tests = {"on": lambda d: abs(d) < 1e-6, "in": lambda d: d < -1e-6,
             "out": lambda d: d > 1e-6, "onin": lambda d: d < 1e-6,
             "onout": lambda d: d > -1e-6}
    if not isinstance(state_, str) or state_ not in tests:
        raise ValueError(f"state must be one of {', '.join(tests)}, got {state_!r}")
    return lambda pts: all(tests[state_](side(p)) for p in pts)


def select(subs: list, where, points_of) -> list:
    """The subs a `where` predicate picks. points_of(sub) gives points on a
    sub-shape (its vertices and centre of mass) for the on_plane test."""
    if isinstance(where, str) and where.strip().startswith("{"):
        import json
        where = json.loads(where)
    if where in (None, "all"):
        return list(subs)
    if not isinstance(where, dict):
        raise ValueError(f"where must be \"all\" or a predicate object, got {where!r}")
    count = lambda: _count(where.get("count", 1))
    if "sorted_by" in where:
        axis = {"x": 0, "y": 1, "z": 2}.get(where["sorted_by"]) \
            if isinstance(where["sorted_by"], str) else None
        take = where.get("take", "max")
        if axis is None or take not in ("max", "min"):
            raise ValueError("sorted_by must be x, y or z and take max or min")
        return sorted(subs, key=lambda u: u.centre[axis], reverse=take == "max")[:count()]
    if "by_size" in where:
        kind = where["by_size"]
        if kind not in ("longest", "shortest", "largest", "smallest"):
            raise ValueError("by_size must be longest, shortest, largest or smallest")
        return sorted(subs, key=lambda u: u.size,
                      reverse=kind in ("longest", "largest"))[:count()]
    if "on_plane" in where:
        test = _plane_test(where)
        return [u for u in subs if test(points_of(u))]
    raise ValueError(f"where {where!r} has none of sorted_by, by_size or on_plane — use "
                     f"one of: {_WHERE['description']}; e.g. {{\"sorted_by\": \"z\", "
                     f"\"take\": \"max\"}} for the highest one")


def _count(value) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)) \
            or not math.isfinite(value) or value != int(value) or value < 1:
        raise ValueError(f"count must be a whole number of at least 1, got {value!r}")
    return int(value)


def _points(sub) -> list:
    import GeomAPI
    from GeomAlgoAPI import GeomAlgoAPI_ShapeTools as T
    xyz = lambda p: [p.x(), p.y(), p.z()]
    pts = [xyz(GeomAPI.GeomAPI_Vertex(v).point())
           for v in sub.shape.subShapes(GeomAPI.GeomAPI_Shape.VERTEX)]
    if sub.shape.shapeType() != GeomAPI.GeomAPI_Shape.VERTEX:
        pts.append(xyz(T.centreOfMass(sub.shape)))
    return pts


def _pick(executor, name: str, sub_type: str, ids, where, subs=None) -> list:
    """The Subs named by `ids` ("all" or a list) or picked by `where`."""
    subs = subshapes(executor, name, sub_type) if subs is None else subs
    if ids is not None and where is not None:
        raise ValueError("give either the ids or where, not both")
    if ids is None and where is None:
        raise ValueError(f"select the {sub_type}s: ids from get_subshapes, \"all\", "
                         "or a where predicate")
    if where is not None:
        picked = select(subs, where, _points)
    elif ids == "all":
        picked = subs
    else:
        if not isinstance(ids, list) or not ids or \
                not all(isinstance(i, str) for i in ids):
            raise ValueError(f"ids must be \"all\" or a list like [\"E1\", \"E4\"], got {ids!r}")
        by_id = {u.id: u for u in subs}
        bad = [i for i in ids if i not in by_id]
        if bad:
            raise ValueError(f"{name} has no {sub_type} {', '.join(bad)} — it has "
                             f"{subs[0].id}..{subs[-1].id}" if subs else
                             f"{name} has no {sub_type}")
        picked = [by_id[i] for i in dict.fromkeys(ids)]
    if not picked:
        raise ValueError(f"no {sub_type} of {name} matches {where!r}")
    return picked


def _sels(sub_type: str, picked: list) -> str:
    return ", ".join(f'model.selection("{sub_type.upper()}", {u.naming!r})' for u in picked)


@tool("get_subshapes", category="query",
      description=("Ids of the vertices, edges, faces or solids of `shape` (\"E1\", "
                   "\"F3\", ...), numbered by centre so the same shape always gives the "
                   "same ids, with each one's centre and size (length, area or volume). "
                   "`where` narrows them: by a centre coordinate, by size, or by a plane "
                   "-- state \"on\" keeps only those lying in it, e.g. a rim circle and not "
                   "the seams that cross it. Pass the ids to fillet_edges, chamfer_edges, "
                   "make_thick_solid, get_subshape, or create_sketch (a face). An edge "
                   "whose two faces meet smoothly, with no corner (e.g. where a fillet "
                   "ends), has \"smooth\": true (null: could not tell); fillet_edges and "
                   "chamfer_edges cannot round it."),
      schema={"type": "object", "properties": {
          "shape": {"type": "string"},
          "sub_type": {"type": "string", "enum": list(SUB_TYPES)},
          "where": _WHERE}, "required": ["shape", "sub_type"]})
def tool_get_subshapes(executor, args: dict):
    stray = sorted(set(args) - {"shape", "sub_type", "where"})
    if stray:
        raise ValueError(f"get_subshapes takes shape, sub_type and where, not "
                         f"{', '.join(stray)} — a filter goes in `where`")
    name = _shape_name(executor, args)
    subs = subshapes(executor, name, args.get("sub_type"))
    picked = select(subs, args.get("where", "all"), _points)
    out = [{"id": u.id, "centre": u.centre, "size": u.size} for u in picked]
    if args.get("sub_type") == "edge":
        smooth = _edge_graph(subs, _faces(executor, name))[2]
        for d in out:
            if smooth[d["id"]] is True or smooth[d["id"]] is None:
                d["smooth"] = smooth[d["id"]]
    return {"ok": True, "records": [], "shape": name, "subshapes": out}


def _faces(executor, name: str) -> list:
    """The faces of `name` as Subs without ids or names, for _edge_graph."""
    import GeomAPI
    out = []
    for res in results_of(executor, name):
        for f in res.shape().subShapes(GeomAPI.GeomAPI_Shape.FACE):
            if not any(f.isSame(u.shape) for u in out):
                out.append(Sub(None, f, "", None, None))
    return out


@tool("get_subshape", category="feature",
      description=("A new shape copied from sub-shapes of `shape` (e.g. some of its "
                   "faces): `ids` from get_subshapes, all of one type. `shape` stays."),
      schema={"type": "object", "properties": {
          "shape": {"type": "string"},
          "ids": {"type": "array", "items": {"type": "string"}}},
          "required": ["shape", "ids"]})
def tool_get_subshape(executor, args: dict):
    name = _shape_name(executor, args)
    ids = args.get("ids")
    if not isinstance(ids, list) or not ids or not all(isinstance(i, str) for i in ids):
        raise ValueError(f"ids must be a list like [\"F1\", \"F2\"], got {ids!r}")
    letters = {i[:1] for i in ids}
    types = {v: k for k, v in SUB_TYPES.items()}
    if len(letters) != 1 or next(iter(letters)) not in types:
        raise ValueError("ids must all be of one type (all E.., all F.., ...)")
    sub_type = types[next(iter(letters))]
    picked = _pick(executor, name, sub_type, ids, None)
    out = new_name(executor, "copy")
    code = f"{out} = model.addCopy(shaper_doc, [{_sels(sub_type, picked)}], 1)"
    return _build(executor, out, "addCopy", code, [name], "copy")


@tool("explode", category="feature",
      description=(f"Copy every sub-shape of `sub_type` of `shape` as its own named "
                   f"shape, in id order (at most {_MAX_EXPLODE}). `shape` stays. Use "
                   "get_subshapes when ids are enough."),
      schema={"type": "object", "properties": {
          "shape": {"type": "string"},
          "sub_type": {"type": "string", "enum": list(SUB_TYPES)}},
          "required": ["shape", "sub_type"]})
def tool_explode(executor, args: dict):
    name = _shape_name(executor, args)
    sub_type = args.get("sub_type")
    subs = subshapes(executor, name, sub_type)
    if not subs:
        raise ValueError(f"{name} has no {sub_type}")
    if len(subs) > _MAX_EXPLODE:
        raise ValueError(f"{name} has {len(subs)} {sub_type}s, more than {_MAX_EXPLODE}: "
                         "pick some with get_subshapes and get_subshape")
    records, names = [], []
    for u in subs:
        out = new_name(executor, sub_type)
        code = f"{out} = model.addCopy(shaper_doc, [{_sels(sub_type, [u])}], 1)"
        r = _build(executor, out, "addCopy", code, [name], "copy")
        if not r["ok"]:
            forget(executor, [n["name"] for n in names])
            return r
        records += r["records"]
        names.append({"name": out, "id": u.id})
    return {"ok": True, "records": records, "names": names}


def fillets_meet(radius: float, pairs: list):
    """The refusal for fillets of `radius` on edge pairs (id, id, distance) that
    bound one face and would meet across it, or None."""
    for a, b, d in pairs:
        if d > 1e-7 and 2 * radius >= _MEET * d:
            return (f"fillets of radius {radius:g} on {a} and {b} would meet across the "
                    f"{d:.4g} mm face between them, which SHAPER cannot build; use a "
                    f"radius below {_MEET * d / 2:.4g}")
    return None


def meets_earlier(radius: float, pairs: list):
    """The refusal for a fillet of `radius` on an edge whose face also borders an
    earlier fillet, for (edge id, fillet face id, distance to that fillet's edge
    on the shared face), or None."""
    for a, g, d in pairs:
        if d > 1e-7 and radius >= _MEET * d:
            return (f"a fillet of radius {radius:g} on {a} would meet the earlier fillet "
                    f"{g} across the {d:.4g} mm of face between them, which SHAPER "
                    f"cannot build; use a radius below {_MEET * d:.4g}")
    return None


def no_corner(ids: list, on: dict):
    """The refusal for picked edges whose faces meet smoothly, given each edge's
    face ids, or None."""
    if not ids:
        return None
    tail = ("; pick edges where faces meet at an angle (get_subshapes marks the "
            "smooth ones)")
    if len(ids) == 1:
        fs = list(dict.fromkeys(on[ids[0]]))
        where = (f"its faces {fs[0]} and {fs[1]} meet smoothly there (as where an "
                 f"earlier fillet ends)" if len(fs) == 2 else f"it is a seam of {fs[0]}")
        return f"{ids[0]} has no corner to round: {where}{tail}"
    return (f"{', '.join(ids)} have no corner to round: the two faces of each meet "
            f"smoothly there (as where an earlier fillet ends){tail}")


def zero_wall(walls: list) -> str:
    """The refusal for picked edges that end where two faces touch back to back,
    for (edge id, point, face of the edge id, other face id)."""
    e, p, f, g = walls[0]
    at = ", ".join(f"{c:.4g}" for c in p)
    msg = (f"cannot round {e}: at ({at}) it ends on {g}, which touches {f} back to back "
           f"there, so the wall between them has zero thickness and SHAPER's fillet fails "
           f"without a reason; move or resize the shape that made {g} so its face does "
           f"not touch {f}, or leave {e} sharp")
    more = list(dict.fromkeys(w[0] for w in walls[1:] if w[0] != e))
    if more:
        msg += f" ({', '.join(more)} end on such a wall too)"
    return msg


def silent_fail(corners: list, unknown: list) -> str:
    """What to say when SHAPER builds nothing and gives no error."""
    if corners:
        return (f"SHAPER failed without giving a reason on {', '.join(corners)}; their "
                f"faces meet at a corner, so this is not a smooth edge. Check that each "
                f"edge's ends do not touch another face (a wall of zero thickness), or "
                f"try a smaller radius or fewer edges at once")
    return ("SHAPER found no corner to round at the picked edges"
            + (f": the faces of {', '.join(unknown)} may meet smoothly there"
               if unknown else "") + "; pick edges where faces meet at an angle")


def cuts_face(radius: float, cuts: list, added: list):
    """The refusal for a built fillet of `radius` that cut faces in two, for
    (face id, chain edge id, nearest other edge id, distance), or None."""
    if not cuts:
        return None
    f, a, b, d = min(cuts, key=lambda c: c[3])
    more = sorted({c[0] for c in cuts} - {f})
    msg = (f"the fillet of radius {radius:g} on {a} ran across {f} and cut it in two: "
           f"{f} is only {d:.4g} mm wide between {a} and {b} there, and fillets next to "
           f"a cut face fail later; nothing was built, use a radius below {d:.4g}")
    if more:
        msg += f" (it also cut {', '.join(more)})"
    if a in added:
        msg += (f". SHAPER also rounds {', '.join(added)} with the picked edges: they go "
                f"on from them at under {math.degrees(_G1):.1f} degrees")
    return msg


def _angle(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(y * y for y in b))
    return math.acos(max(-1.0, min(1.0, dot / (na * nb))))


def _carries(e, t_e, x, t_x, others, smooth, fading, v):
    """Whether OCC carries a fillet on `e` on to `x` at vertex `v`: True,
    False, or None when it could not be told."""
    verdicts = [True if smooth[x] is False else None if smooth[x] is None else False]
    verdicts += [True if smooth[o] is True else False if smooth[o] in (False, "free")
                 else None for o in others]
    if t_e is None or t_x is None:
        verdicts.append(None)
    else:
        a = _angle([-c for c in t_e], t_x)
        fade = [fading(e, v), fading(x, v)]
        if None in fade:
            verdicts.append(None)
        else:
            verdicts.append(a < (_FADE if True in fade else math.pi / 2))
    return False if False in verdicts else None if None in verdicts else True


def fillet_chain(picked, ends, at, smooth, fading):
    """The edges OCC's fillet rounds besides `picked`, and the vertices where
    that could not be told. Like ChFi3d_Builder::PerformElement, it goes on from
    an edge to the next one at a vertex when that one is a crease, every other
    edge there is smooth, and the fillet does not turn back.
    ends[e]: [(vertex, unit tangent leaving the vertex along e)]; at[v]: the
    edges at v; smooth[e]: True, False, "free" (one face) or None (not told);
    fading(e, v): whether the faces of e meet tangentially at v (or None)."""
    chain, added, unsure = set(picked), [], []
    todo = [(e, v, t) for e in picked for v, t in ends[e]]
    while todo:
        e, v, t_e = todo.pop()
        cands = [x for x in at[v] if x != e]
        go, maybe = [], False
        for x in cands:
            t_x = [t for w, t in ends[x] if w == v]
            others = [o for o in cands if o != x]
            got = _carries(e, t_e, x, t_x[0] if len(t_x) == 1 else None, others,
                           smooth, fading, v)
            if got:
                go.append(x)
            maybe = maybe or got is None
        if len(go) != 1 or maybe:
            if go or maybe:
                unsure.append(v)
            continue
        x = go[0]
        if x in chain:
            continue
        chain.add(x)
        added.append(x)
        todo += [(x, w, t) for w, t in ends[x] if w != v]
    return added, unsure


def _unit(a, b):
    d = [b[i] - a[i] for i in range(3)]
    n = math.sqrt(sum(c * c for c in d))
    return [c / n for c in d] if n > 1e-12 else None


def _edge_graph(edges: list, faces: list):
    """The topology fillet_chain walks for `edges` of a shape whose faces are
    `faces` (Subs), plus the faces of each edge."""
    import GeomAPI
    S = GeomAPI.GeomAPI_Shape
    xyz = lambda p: [p.x(), p.y(), p.z()]
    on = {u.id: [] for u in edges}       # a seam shows up twice in its one face
    for f in faces:
        for s in f.shape.subShapes(S.EDGE):
            for u in edges:
                if u.shape.isSame(s):
                    on[u.id].append(f)
                    break

    def normal(f):
        g = GeomAPI.GeomAPI_Face(f.shape)
        if not g.isPlanar():
            return None
        d = g.getPlane().direction()
        n = xyz(d)
        return [-c for c in n] if f.shape.orientation() == S.REVERSED else n

    verts, ends, smooth = [], {}, {}
    for u in edges:
        e = GeomAPI.GeomAPI_Edge(u.shape)
        fs = on[u.id]
        if len(fs) < 2:
            smooth[u.id] = "free"
        elif fs[0].shape.isSame(fs[1].shape):
            smooth[u.id] = True
        else:
            n1, n2 = normal(fs[0]), normal(fs[1])
            if n1 and n2:
                smooth[u.id] = _angle(n1, n2) <= _G1
            else:
                smooth[u.id] = _continuous(fs[0].shape, fs[1].shape, _edge_points(u.shape),
                                           _G1)
        ends[u.id] = []
        if e.isDegenerated():
            continue
        c = GeomAPI.GeomAPI_Curve(e)
        a, b = c.startParam(), c.endParam()
        p0, p1 = xyz(c.getPoint(a)), xyz(c.getPoint(b))
        step = (b - a) * 1e-6
        for v in u.shape.subShapes(S.VERTEX):
            k = next((i for i, w in enumerate(verts) if w.isSame(v)), None)
            if k is None:
                k = len(verts)
                verts.append(v)
            p = xyz(GeomAPI.GeomAPI_Vertex(v).point())
            if e.isClosed():
                ends[u.id].append((k, None))
            elif math.dist(p, p0) <= math.dist(p, p1):
                ends[u.id].append((k, _unit(p0, xyz(c.getPoint(a + step)))))
            else:
                ends[u.id].append((k, _unit(p1, xyz(c.getPoint(b - step)))))
    at = {}
    for u in edges:
        for k, _ in ends[u.id]:
            at.setdefault(k, [])
            if u.id not in at[k]:
                at[k].append(u.id)

    def fading(eid, k):
        fs = on[eid]
        if len(fs) < 2 or fs[0].shape.isSame(fs[1].shape):
            return False
        n1, n2 = normal(fs[0]), normal(fs[1])
        if n1 and n2:
            return _angle(n1, n2) < _FADE
        return _continuous(fs[0].shape, fs[1].shape,
                           [xyz(GeomAPI.GeomAPI_Vertex(verts[k]).point())], _FADE)

    return ends, at, smooth, fading, on


def _continuous(f1, f2, points: list, limit: float):
    """Whether the faces f1, f2 meet within `limit` rad at the points: True,
    False, or None when a normal is unknown or the angle is near the limit."""
    worst = 0.0
    for q in points:
        n1, n2 = _surface_normal(f1, q), _surface_normal(f2, q)
        if n1 is None or n2 is None:
            return None
        a = _angle(n1, n2)
        worst = max(worst, min(a, math.pi - a))
    if worst <= limit * (1 - _BAND):
        return True
    if worst > limit * (1 + _BAND):
        return False
    return None


def _edge_points(edge, n: int = 7) -> list:
    import GeomAPI
    c = GeomAPI.GeomAPI_Curve(GeomAPI.GeomAPI_Edge(edge))
    a, b = c.startParam(), c.endParam()
    return [[q.x(), q.y(), q.z()] for q in
            (c.getPoint(a + (b - a) * k / (n - 1)) for k in range(n))]


def _surface_normal(face, p):
    """The normal (either way) of a plane, cylinder, cone, sphere or torus face
    at the point p on it, or None for other surfaces."""
    import GeomAPI
    g = GeomAPI.GeomAPI_Face(face)
    xyz = lambda q: [q.x(), q.y(), q.z()]
    sub = lambda a, b: [a[i] - b[i] for i in range(3)]
    dot = lambda a, b: sum(a[i] * b[i] for i in range(3))
    radial = lambda w, ax: sub(w, [dot(w, ax) * c for c in ax])
    if g.isPlanar():
        return xyz(g.getPlane().direction())
    c, s, t, k = g.getCylinder(), g.getSphere(), g.getTorus(), g.getCone()
    if c is not None:
        n = radial(sub(p, xyz(c.location())), xyz(c.axis()))
    elif s is not None:
        n = sub(p, xyz(s.center()))
    elif t is not None:
        rho = radial(sub(p, xyz(t.center())), xyz(t.direction()))
        r = math.sqrt(dot(rho, rho))
        if r < 1e-9:
            return None
        n = sub(p, [xyz(t.center())[i] + t.majorRadius() * rho[i] / r for i in range(3)])
    elif k is not None:
        ax, w = xyz(k.axis()), sub(p, xyz(k.apex()))
        rho = radial(w, ax)
        r = math.sqrt(dot(rho, rho))
        if r < 1e-9:
            return None
        a, side = k.semiAngle(), 1 if dot(w, ax) >= 0 else -1
        n = [rho[i] / r * math.cos(a) - side * ax[i] * math.sin(a) for i in range(3)]
    else:
        return None
    return n if dot(n, n) > 1e-18 else None


def _outside(solids: list, p: list, tol: float) -> bool | None:
    """Whether the point p lies outside every solid (None if not told)."""
    import GeomAPI
    from GeomAlgoAPI import GeomAlgoAPI_PointBuilder, GeomAlgoAPI_ShapeTools as T
    v = GeomAlgoAPI_PointBuilder.vertex(GeomAPI.GeomAPI_Pnt(*p))
    ds = [T.minimalDistance(v, s) for s in solids]
    if any(d is None or d < 0 for d in ds):
        return None
    return all(d > tol for d in ds)


def _step(box: list) -> float:
    """A step small against a shape of bounding box [xmin, ymin, zmin, xmax,
    ymax, zmax]: 1e-4 of its diagonal."""
    return max(math.dist(box[:3], box[3:]) * 1e-4, 1e-5)


def _zero_walls(picked: list, faces: list, on: dict, solids: list,
                step: float) -> tuple[list, list]:
    """(edge id, point, face of the edge id, other face id) for each end of a
    picked edge where another face is tangent to one of the edge's faces and
    there is no material on either side of them (a wall of zero thickness); and
    the ids of edges whose ends could not be checked (a normal not known).
    Face orientation is not used: SHAPER's plane normals are oriented, its
    curved ones (as _surface_normal gives them) are not."""
    import GeomAPI
    S = GeomAPI.GeomAPI_Shape
    verts = {f.id: f.shape.subShapes(S.VERTEX) for f in faces}
    walls, unchecked = [], []
    for u in picked:
        own = on[u.id]
        unsure = False
        for v in u.shape.subShapes(S.VERTEX):
            q = GeomAPI.GeomAPI_Vertex(v).point()
            p = [q.x(), q.y(), q.z()]
            others = [f for f in faces if not any(f.shape.isSame(o.shape) for o in own)
                      and any(v.isSame(w) for w in verts[f.id])]
            for f in others:
                nf = _surface_normal(f.shape, p)
                for o in own:
                    no = _surface_normal(o.shape, p)
                    if nf is None or no is None:
                        unsure = True
                        continue
                    if min(_angle(nf, no), math.pi - _angle(nf, no)) > 1e-6:
                        continue
                    r = math.sqrt(sum(c * c for c in no))
                    sides = [_outside(solids, [p[i] + k * step * no[i] / r for i in range(3)],
                                      step / 10) for k in (1, -1)]
                    if None in sides:
                        unsure = True
                    elif all(sides):
                        walls.append((u.id, p, o.id, f.id))
        if unsure and not any(w[0] == u.id for w in walls):
            unchecked.append(u.id)
    return walls, unchecked


def _earlier_fillets(chain: list, on: dict) -> tuple[list, bool]:
    """(edge id, fillet face id, distance) from each chain edge to the line along
    which an earlier fillet face rolls on one of the chain edge's faces (where it
    meets that face smoothly, not where it ends on it); and whether a line could
    not be told."""
    from GeomAlgoAPI import GeomAlgoAPI_ShapeTools as T
    import GeomAPI
    E = GeomAPI.GeomAPI_Shape.EDGE
    fillets = {f.id: f for fs in on.values() for f in fs if "GF:Fillet" in f.naming}
    out, unsure = [], False
    for u in chain:
        for f in {f.id: f for f in on[u.id]}.values():
            for g in fillets.values():
                if g.id == f.id:
                    continue
                for s in f.shape.subShapes(E):
                    if not any(s.isSame(t) for t in g.shape.subShapes(E)):
                        continue
                    rolls = _continuous(f.shape, g.shape, _edge_points(s), _G1)
                    unsure = unsure or rolls is None
                    if rolls:
                        out.append((u.id, g.id, T.minimalDistance(u.shape, s)))
    return out, unsure


def _elementary(face) -> bool:
    import GeomAPI
    g = GeomAPI.GeomAPI_Face(face)
    return g.isPlanar() or any(x is not None for x in (
        g.getCylinder(), g.getSphere(), g.getTorus(), g.getCone()))


def _within(piece, face, tol: float = 1e-5) -> bool:
    """Whether the boundary of `piece` lies on `face`."""
    import GeomAPI
    from GeomAlgoAPI import GeomAlgoAPI_ShapeTools as T
    from GeomAlgoAPI import GeomAlgoAPI_PointBuilder as PB
    for e in piece.subShapes(GeomAPI.GeomAPI_Shape.EDGE):
        if GeomAPI.GeomAPI_Edge(e).isDegenerated():
            continue
        if any(T.minimalDistance(face, PB.vertex(*q)) > tol for q in _edge_points(e, 3)):
            return False
    return True


def _touch(a, b) -> bool:
    import GeomAPI
    V = GeomAPI.GeomAPI_Shape.VERTEX
    return any(v.isSame(w) for v in a.subShapes(V) for w in b.subShapes(V))


def _cut_faces(chain: list, edges: list, on: dict, shapes: list) -> tuple[list, list]:
    """(face id, chain edge id, nearest edge id, distance) for each face of a
    chain edge that comes out of the fillet result `shapes` as several faces,
    with the nearest edge of that face that does not touch the chain edge; and
    the ids of faces that could not be checked."""
    import GeomAPI
    from GeomAlgoAPI import GeomAlgoAPI_ShapeTools as T
    out_faces = [f for s in shapes for f in s.subShapes(GeomAPI.GeomAPI_Shape.FACE)]
    cuts, unsure = [], []
    for f in {f.id: f for u in chain for f in on[u.id]}.values():
        if not _elementary(f.shape):
            unsure.append(f.id)
            continue
        g = GeomAPI.GeomAPI_Face(f.shape)
        pieces = [p for p in out_faces if g.isSameGeometry(p) and _within(p, f.shape)]
        if len(pieces) < 2:
            continue
        mine = [u for u in chain if any(x.id == f.id for x in on[u.id])]
        rest = [e for e in edges if any(x.id == f.id for x in on[e.id])]
        near = [(u.id, e.id, T.minimalDistance(u.shape, e.shape)) for u in mine
                for e in rest if not _touch(u.shape, e.shape)]
        if near:
            cuts.append((f.id, *min(near, key=lambda n: n[2])))
    return cuts, unsure


def _face_pairs(executor, name: str, picked: list) -> list:
    """(id, id, distance) for each two picked edges that bound one face and do
    not touch."""
    import GeomAPI
    from GeomAlgoAPI import GeomAlgoAPI_ShapeTools as T
    E, F = GeomAPI.GeomAPI_Shape.EDGE, GeomAPI.GeomAPI_Shape.FACE
    seen, out = set(), []
    for res in results_of(executor, name):
        for f in res.shape().subShapes(F):
            on = [u for u in picked if any(u.shape.isSame(e) for e in f.subShapes(E))]
            for i, a in enumerate(on):
                for b in on[i + 1:]:
                    if (a.id, b.id) not in seen:
                        seen.add((a.id, b.id))
                        out.append((a.id, b.id, T.minimalDistance(a.shape, b.shape)))
    return out


def _fillet_check(executor, name: str, edges: list, picked: list, radius: float, graph):
    """The edges SHAPER rounds besides `picked`, a note when that could not be
    told everywhere, the chain of rounded edges and the faces of each edge;
    raises when the fillets would meet."""
    ends, at, smooth, fading, on = graph
    by_id = {u.id: u for u in edges}
    added, unsure = fillet_chain([u.id for u in picked], ends, at, smooth, fading)
    chain = picked + [by_id[i] for i in added]
    earlier, unsure_earlier = _earlier_fillets(chain, on)
    refusal = fillets_meet(radius, _face_pairs(executor, name, chain)) or \
        meets_earlier(radius, earlier)
    if refusal:
        if added:
            refusal += (f". SHAPER also rounds {', '.join(added)} with the picked edges: "
                        f"they go on from them at under {math.degrees(_G1):.1f} degrees")
        raise ValueError(refusal)
    notes = []
    if unsure:
        notes.append(f"could not tell at {len(unsure)} vertex(es) whether SHAPER carries "
                     "the fillet on to further edges")
    if unsure_earlier:
        notes.append("could not tell where an earlier fillet rolls on a face next to "
                     "these edges")
    note = "; ".join(notes) + " (curved faces): not checked for fillets meeting" \
        if notes else None
    return added, note, chain, on


def _edge_feature(executor, args: dict, prefix: str, fn: str, call, radius=None) -> dict:
    name = _shape_name(executor, args)
    edges = subshapes(executor, name, "edge")
    picked = _pick(executor, name, "edge", args.get("edges"), args.get("where"), edges)
    graph = _edge_graph(edges, subshapes(executor, name, "face"))
    smooth, on = graph[2], graph[4]
    # OCC skips edges without a corner when others have one, and fails silently
    # when none has
    flat = [u.id for u in picked if smooth[u.id] is True]
    if len(flat) == len(picked):
        raise ValueError(no_corner(flat, {i: [f.id for f in on[i]] for i in flat}))
    skipped = [i for i in flat if not on[i][0].shape.isSame(on[i][-1].shape)]
    real = [u for u in picked if u.id not in flat]
    solids = [r.shape() for r in results_of(executor, name) if r.shape() is not None]
    walls, unchecked = _zero_walls(real, subshapes(executor, name, "face"), on, solids,
                                   _step(raw_box(executor, name)))
    if walls:
        raise ValueError(zero_wall(walls))
    unknown = [u.id for u in picked if smooth[u.id] is None]
    corners = [u.id for u in real if smooth[u.id] is False]
    nothing = silent_fail(corners, unknown)
    added, notes, check = [], [], None
    if unchecked:
        notes.append(f"could not check the ends of {', '.join(unchecked)} for a wall of "
                     "zero thickness (curved faces)")
    if skipped:
        notes.append(f"SHAPER skipped {', '.join(skipped)}: the faces there meet "
                     "smoothly, with no corner to round")
    if radius is not None:
        added, note, chain, on = _fillet_check(
            executor, name, edges, [u for u in picked if u.id not in flat], radius, graph)
        notes += [note] if note else []

        def check(feat):
            shapes = [r.shape() for r in feat.results() if r.shape() is not None]
            cuts, unsure = _cut_faces(chain, edges, on, shapes)
            if unsure:
                notes.append(f"{', '.join(unsure)} not a plane, cylinder, cone, sphere or "
                             "torus: not checked for being cut in two by the fillet")
            return cuts_face(radius, cuts, added)
    out = new_name(executor, prefix)
    code = f"{out} = model.{fn}(shaper_doc, [{_sels('edge', picked)}], {call})"
    r = _build(executor, out, fn, code, [name], "solid", check=check, nothing=nothing)
    if r["ok"]:
        r["edges"] = [u.id for u in picked]
        if added:
            r["also_rounded"] = added
        if notes:
            r["note"] = "; ".join(notes + ([r["note"]] if r.get("note") else []))
    return r


@tool("fillet_edges", category="feature",
      description=("Round edges of `shape` with `radius`: `edges` (ids from "
                   "get_subshapes, or \"all\") or a `where` predicate (as get_subshapes). "
                   "Fails if the radius is too large for the shape. A fillet also runs on "
                   "along edges that continue a picked one almost straight; the reply "
                   "lists them as also_rounded." + _USED),
      schema={"type": "object", "properties": {
          "shape": {"type": "string"}, "radius": {"type": "number"},
          "edges": _IDS, "where": _WHERE}, "required": ["shape", "radius"]})
def tool_fillet_edges(executor, args: dict):
    r = _num(args.get("radius"), "radius", positive=True)
    return _edge_feature(executor, args, "fillet", "addFillet", repr(r), radius=r)


@tool("chamfer_edges", category="feature",
      description=("Bevel edges of `shape` by distances d1 and d2 (default d1) along its "
                   "two faces: `edges` (ids or \"all\") or a `where` predicate." + _USED),
      schema={"type": "object", "properties": {
          "shape": {"type": "string"}, "d1": {"type": "number"}, "d2": {"type": "number"},
          "edges": _IDS, "where": _WHERE}, "required": ["shape", "d1"]})
def tool_chamfer_edges(executor, args: dict):
    d1 = _num(args.get("d1"), "d1", positive=True)
    d2 = d1 if args.get("d2") is None else _num(args["d2"], "d2", positive=True)
    return _edge_feature(executor, args, "chamfer", "addChamfer", f"True, {d1!r}, {d2!r}")


@tool("make_thick_solid", category="feature",
      description=("Hollow the solid `shape` into a wall of `thickness`, open at `faces` "
                   "(ids from get_subshapes, or a `where` predicate; at least one). "
                   "inside (default true) keeps the outer size; false grows the part by "
                   "the thickness." + _USED),
      schema={"type": "object", "properties": {
          "shape": {"type": "string"}, "thickness": {"type": "number"},
          "faces": {"type": "array", "items": {"type": "string"}}, "where": _WHERE,
          "inside": {"type": "boolean"}}, "required": ["shape", "thickness"]})
def tool_make_thick_solid(executor, args: dict):
    name = _shape_name(executor, args)
    t = _num(args.get("thickness"), "thickness", positive=True)
    inside = args.get("inside", True)
    if not isinstance(inside, bool):
        raise ValueError(f"inside must be true or false, got {inside!r}")
    solids = subshapes(executor, name, "solid")
    if len(solids) != 1:
        raise ValueError(f"{name} holds {len(solids)} solids; make_thick_solid takes one")
    picked = _pick(executor, name, "face", args.get("faces"), args.get("where"))
    out = new_name(executor, "thickness")
    code = (f"{out} = model.addHollowedSolid(shaper_doc, "
            f"model.selection(\"SOLID\", {solids[0].naming!r}), {t!r}, "
            f"[{_sels('face', picked)}], {inside})")
    return _build(executor, out, "addHollowedSolid", code, [name], "solid")


def _no_repeats(*groups) -> None:
    names = [n for g in groups for n in (g if isinstance(g, list) else [g])]
    twice = sorted({n for n in names if names.count(n) > 1 and isinstance(n, str)})
    if twice:
        raise ValueError(f"{', '.join(twice)} given more than once")


def _tools_sels(executor, names, field: str) -> tuple[list, list]:
    """Cutting tools: shapes, or construction planes."""
    st = state(executor)
    if isinstance(names, str):
        names = [names]
    if not isinstance(names, list) or not names or not all(isinstance(n, str) for n in names):
        raise ValueError(f"{field} must be a non-empty list of names")
    sels, refs = [], []
    for n in names:
        if st.objects.get(n) == "plane":
            sels.append(f'model.selection("FACE", {n!r})')
            refs.append(n)
        else:
            s, r = _operands(executor, [n], field)
            sels += s
            refs += r
    return sels, refs


@tool("partition", category="boolean",
      description=("Cut `shapes` (and `tools`, construction planes allowed) by each other "
                   "and keep every piece, sharing the faces where they meet: one result. "
                   "The inputs are used up."),
      schema={"type": "object", "properties": {
          "shapes": {"type": "array", "items": {"type": "string"}},
          "tools": {"type": "array", "items": {"type": "string"}}},
          "required": ["shapes"]})
def tool_partition(executor, args: dict):
    tools = args.get("tools")
    if tools is not None and not isinstance(tools, list):
        raise ValueError(f"tools must be a list of names, got {tools!r}")
    _no_repeats(args.get("shapes"), tools or [])
    sels, refs = _operands(executor, args.get("shapes"), "shapes")
    if tools:
        t_sels, t_refs = _tools_sels(executor, args["tools"], "tools")
        sels, refs = sels + t_sels, refs + t_refs
    if len(sels) < 2:
        raise ValueError("partition needs at least two shapes, or a shape and a tool")
    out = new_name(executor, "partition")
    code = f"{out} = model.addPartition(shaper_doc, [{', '.join(sels)}])"
    return _build(executor, out, "addPartition", code, refs, "solid")


@tool("split", category="boolean",
      description=("Split `objects` into pieces with `tools` (shapes or construction "
                   "planes): every piece is kept in one result; the tools are not. The "
                   "objects are used up."),
      schema={"type": "object", "properties": {
          "objects": {"type": "array", "items": {"type": "string"}},
          "tools": {"type": "array", "items": {"type": "string"}}},
          "required": ["objects", "tools"]})
def tool_split(executor, args: dict):
    _no_repeats(args.get("objects"), args.get("tools"))
    sels, refs = _operands(executor, args.get("objects"), "objects")
    t_sels, t_refs = _tools_sels(executor, args.get("tools"), "tools")
    out = new_name(executor, "split")
    code = (f"{out} = model.addSplit(shaper_doc, [{', '.join(sels)}], "
            f"[{', '.join(t_sels)}])")
    return _build(executor, out, "addSplit", code, refs + t_refs, "solid")

