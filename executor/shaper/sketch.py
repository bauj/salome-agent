import math
import re

from executor.shaper.state import (
    CONSTRUCTION, PUBLISH_LINE, Constraint, Entity, Sketch, check_id, construction_info,
    KIND_TYPES, forget, geometry, new_name, offset_plane, record, sketch_frame, subshapes,
    restore, run, sketch_of, snapshot, solver_error, state, status, tool, transaction,
)

# plane -> the world axes lying in it, offered as sketch references
_PLANES = {"XOY": ("OX", "OY"), "XOZ": ("OX", "OZ"), "YOZ": ("OY", "OZ")}
_ENTITY_TYPES = ("line", "circle", "arc", "point", "ellipse", "spline", "rectangle",
                 "polyline", "projection")


def exec_records(st, records: list) -> None:
    """Run records exactly as result.py will: their code, then their publish line.
    If one fails, the features the batch already made are removed again, so the
    document never holds what no record describes."""
    made = []
    try:
        with transaction():
            for r in records:
                made += r.get("out_vars") or ()
                run(st, r["code"])
                for v in r.get("publish") or ():
                    run(st, PUBLISH_LINE.format(var=v, name=v))
    except Exception:
        with transaction():
            for var in reversed(made):
                obj = st.ns.pop(var, None)
                if obj is not None:
                    try:
                        st.ns["shaper_doc"].removeFeature(obj.feature())
                    except Exception:  # noqa: BLE001 — already gone with its owner
                        pass
        raise


def _drop_helpers(st, records: list) -> None:
    """Remove the helper features (`<var>_x = ...`) a failed batch left behind: they
    are not outputs of any record, so exec_records does not know them."""
    names = [m for r in records for m in re.findall(r"^(\w+_[a-z]\w*) = ", r["code"], re.M)
             if m in st.ns]
    with transaction():
        for n in reversed(names):
            obj = st.ns.pop(n, None)
            try:
                st.ns["shaper_doc"].removeFeature(obj.feature())
            except Exception:  # noqa: BLE001 — already gone with its owner
                pass


def _num(v, what: str):
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ValueError(f"{what} must be a number, got {v!r}")
    return v


def _xy(v, what: str) -> list:
    if not isinstance(v, (list, tuple)) or len(v) != 2:
        raise ValueError(f"{what} must be [x, y], got {v!r}")
    return [_num(v[0], what), _num(v[1], what)]


def _reply(executor, sk: Sketch, records: list, before: dict | None, **extra) -> dict:
    st = status(executor, sk, before)
    return {"ok": True, "records": records, **st, **extra,
            "result_shape": sk.name,
            "shape_info": {"dof": st["dof"], "loose_ends": st["loose_ends"]}}


def _taken(sk: Sketch) -> set:
    return set(sk.entities) | set(sk.constraints) | {"ORIGIN", *sk.axes}


# --- create_sketch ----------------------------------------------------------

@tool("create_sketch", category="sketch",
      description=(
          "Start a new empty sketch on a base plane (\"XOY\", \"XOZ\", \"YOZ\") or on a "
          "plane parallel to one ({\"base\": \"XOY\", \"offset\": 15}; the offset runs along "
          "+Z for XOY, +Y for XOZ, +X for YOZ). Returns its "
          "published name. Every sketch has the references ORIGIN (a point) and the two "
          "world axes lying in its plane (XOY: OX, OY; XOZ: OX, OZ; YOZ: OY, OZ), usable "
          "in constraints to anchor the geometry. On XOY sketch coordinates [x, y] are "
          "world X, Y. A sketch can also go on a make_plane result (its name) or on a "
          "flat face of a solid ({\"face\": \"F6\", \"of\": \"box_1\"}, ids from "
          "get_subshapes): then the reply gives its `frame` and sketch coordinates "
          "[u, v] are the world point origin + u * x_dir + v * y_dir; ORIGIN is the "
          "world origin projected onto the plane, and only world axes parallel to the "
          "plane are offered."),
      schema={"type": "object", "properties": {
          "plane": {"description": ("\"XOY\" | \"XOZ\" | \"YOZ\", {base, offset}, a "
                                    "make_plane name, or {face, of}"),
                    "anyOf": [{"type": "string"},
                              {"type": "object", "properties": {
                                  "base": {"type": "string", "enum": list(_PLANES)},
                                  "offset": {"type": "number"}},
                               "required": ["base", "offset"]},
                              {"type": "object", "properties": {
                                  "face": {"type": "string"}, "of": {"type": "string"}},
                               "required": ["face", "of"]}]}},
          "required": ["plane"]})
def tool_create_sketch(executor, args: dict):
    st = state(executor)
    plane = args.get("plane", "XOY")
    name = new_name(executor, "sketch")
    records = []
    refs = []
    normal = None
    if isinstance(plane, dict) and "face" in plane:
        face, of = plane.get("face"), plane.get("of")
        if not isinstance(face, str):
            raise ValueError(f"face must be an id like \"F2\", got {face!r}")
        if not isinstance(of, str) or (of not in st.objects):
            raise ValueError(f"of must name a built shape, got {of!r}")
        if st.objects[of] in CONSTRUCTION:
            raise ValueError(f"{of!r} is a construction {st.objects[of]}, not a solid")
        subs = {u.id: u for u in subshapes(executor, of, "face")}
        if face not in subs:
            raise ValueError(f"{of} has no face {face!r} — get_subshapes(sub_type=face) "
                             f"lists them ({', '.join(subs)})")
        normal = _face_normal(subs[face].shape)
        if normal is None:
            raise ValueError(f"{face} of {of} is not flat: a sketch needs a flat face")
        target = f'model.selection("FACE", {subs[face].naming!r})'
        refs = [of]
    elif isinstance(plane, str) and st.objects.get(plane) == "plane":
        normal = construction_info(executor, plane)["normal"]
        target = f'model.selection("FACE", {plane!r})'
        refs = [plane]
    elif isinstance(plane, dict):
        base, offset = plane.get("base"), plane.get("offset")
        if base not in _PLANES:
            raise ValueError(f"plane base must be one of {', '.join(_PLANES)}, got {base!r}")
        offset = _num(offset, "plane offset")
        pv = f"{name}_plane"
        records.append(record("addPlane", (
            f'{pv} = model.addPlane(shaper_doc, {offset_plane(base, offset)})\n'
            f'_name({pv}, {pv!r})'), out=[pv], recv=name))
        target = f'model.selection("FACE", {pv!r})'
        refs = [pv]
    elif plane in _PLANES:
        base = plane
        target = f"model.defaultPlane({plane!r})"
    else:
        raise ValueError(f"plane must be one of {', '.join(_PLANES)}, {{base, offset}}, "
                         f"a make_plane name or {{face, of}}, got {plane!r}")
    # a face or plane is selected by name: recompute first, as _build does
    pre = "model.do()\n" if normal is not None else ""
    records.append(record("addSketch", f"{pre}{name} = model.addSketch(shaper_doc, {target})",
                          out=[name], refs=refs, publish=True))
    if normal is None:
        axes = _PLANES[base]
    else:
        axes = tuple(ax for ax, i in (("OX", 0), ("OY", 1), ("OZ", 2)) if abs(normal[i]) < 1e-9)
    records.append(record("addProjection", (
        f'{name}_ORIGIN = {name}.addProjection(model.selection("VERTEX", '
        f'"PartSet/Origin"), False)'), out=[f"{name}_ORIGIN"], recv=name))
    for ax in axes:
        records.append(record("addProjection", (
            f'{name}_{ax} = {name}.addProjection(model.selection("EDGE", '
            f'"PartSet/{ax}"), False)'), out=[f"{name}_{ax}"], recv=name))
    exec_records(st, records)
    sk = Sketch(name, st.ns[name], axes)
    extra = {}
    st.sketches[name] = sk
    if normal is not None:
        try:
            o, x, y, n = sketch_frame(sk)
        except Exception:
            forget(executor, [name])
            raise
        sk.frame = (o, x, y)
        r6 = lambda v: [round(c, 6) + 0.0 for c in v]
        extra["frame"] = {"origin": r6(o), "x_dir": r6(x), "y_dir": r6(y), "normal": r6(n)}
    return _reply(executor, sk, records, None, **extra)


def _face_normal(face) -> list | None:
    import GeomAPI
    f = GeomAPI.GeomAPI_Face(face)
    if not f.isPlanar():
        return None
    d = f.getPlane().direction()
    return [d.x(), d.y(), d.z()]


# --- sketch_add -------------------------------------------------------------

def _entity_code(sk: Sketch, e: dict, var: str) -> str:
    t = e["type"]
    if t == "line":
        (x1, y1), (x2, y2) = e["start"], e["end"]
        call = f"{sk.name}.addLine({x1!r}, {y1!r}, {x2!r}, {y2!r})"
    elif t == "circle":
        (cx, cy), r = e["center"], e["radius"]
        call = f"{sk.name}.addCircle({cx!r}, {cy!r}, {r!r})"
    elif t == "arc":
        (cx, cy), (sx, sy), (ex, ey) = e["center"], e["start"], e["end"]
        call = (f"{sk.name}.addArc({cx!r}, {cy!r}, {sx!r}, {sy!r}, {ex!r}, {ey!r}, "
                f"{bool(e.get('reversed'))!r})")
    else:
        x, y = e["at"]
        call = f"{sk.name}.addPoint({x!r}, {y!r})"
    return f"{var} = {call}\n_name({var}, {var!r})"


def _line_code(sk: Sketch, var: str, a, b) -> str:
    return f"{var} = {sk.name}.addLine({a[0]!r}, {a[1]!r}, {b[0]!r}, {b[1]!r})\n_name({var}, {var!r})"


def _points_list(e: dict, w: str, least: int, distinct: bool = False) -> list:
    pts = e.get("points")
    if not isinstance(pts, list) or len(pts) < least:
        raise ValueError(f"{w}: points must be a list of at least {least} [x, y]")
    pts = [_xy(p, f"{w} point") for p in pts]
    pairs = list(zip(pts, pts[1:])) + ([(pts[-1], pts[0])] if e.get("closed") else [])
    if distinct:
        pairs = [(p, q) for i, p in enumerate(pts) for q in pts[i + 1:]]
    if any(math.dist(p, q) < 1e-9 for p, q in pairs):
        raise ValueError(f"{w}: two "
                         + ("" if distinct else "consecutive ") + "points are the same")
    return pts


def _plan_entity(executor, sk: Sketch, e, taken: set) -> tuple[str, list]:
    """Code adding entity `e`, and the (id, var) of every entity it makes: a
    rectangle or polyline makes one line per side, ids <id>1, <id>2, ..."""
    if not isinstance(e, dict):
        raise ValueError(f"each entity is an object, got {e!r}")
    check_id(e.get("id"), "entity")
    w = e["id"]
    t = e.get("type")
    if not isinstance(t, str) or t not in _ENTITY_TYPES:
        raise ValueError(f"{w}: type must be one of {', '.join(_ENTITY_TYPES)}, got {t!r}")
    var = f"{sk.name}_{w}"
    closed = e.get("closed", False)
    if not isinstance(closed, bool):
        raise ValueError(f"{w}: closed must be true or false")
    if closed and t not in ("polyline", "spline"):
        raise ValueError(f"{w}: closed applies to a polyline or a spline, not a {t}")
    if not isinstance(e.get("construction", t == "projection"), bool):
        raise ValueError(f"{w}: construction must be true or false")
    if t in ("line", "circle", "arc", "point"):
        _check_entity(e)
        return _entity_code(sk, e, var), [(w, var)]
    if t == "ellipse":
        c = _xy(e.get("center"), f"{w}.center")
        big = _num(e.get("major_radius"), f"{w}.major_radius")
        small = _num(e.get("minor_radius"), f"{w}.minor_radius")
        if not 0 < small < big:
            raise ValueError(f"{w}: need 0 < minor_radius < major_radius "
                             "(a circle is type circle)")
        a = math.radians(_num(e.get("angle", 0), f"{w}.angle"))
        m = [c[0] + big * math.cos(a), c[1] + big * math.sin(a)]
        p = [c[0] - small * math.sin(a), c[1] + small * math.cos(a)]
        r = lambda v: repr(round(v, 9) + 0.0)
        code = (f"{var}_m = {sk.name}.addEllipse({r(c[0])}, {r(c[1])}, {r(m[0])}, {r(m[1])}, "
                f"{r(p[0])}, {r(p[1])}, True)\nmodel.do()\n"
                f"{var} = _wrap(model.lastSubFeature({sk.name}, \"SketchEllipse\"))\n"
                f"_name({var}, {var!r})")
        return code, [(w, var)]
    if t == "spline":
        pts = _points_list(e, w, 3 if closed else 2, distinct=True)
        code = "".join(f"{var}_q{k} = {sk.name}.addPoint({x!r}, {y!r})\n"
                       f"{var}_q{k}.setAuxiliary(True)\n" for k, (x, y) in enumerate(pts, 1))
        refs = ", ".join(f"{var}_q{k}.coordinates()" for k in range(1, len(pts) + 1))
        code += (f"{var} = {sk.name}.addInterpolation([{refs}]"
                 + (", periodic=True" if closed else "") + f")\n_name({var}, {var!r})")
        return code, [(w, var)]
    if t == "projection":
        of, sub = e.get("of"), e.get("edge") or e.get("vertex")
        if e.get("edge") and e.get("vertex"):
            raise ValueError(f"{w}: give edge or vertex, not both")
        if not isinstance(e.get("construction", True), bool):
            raise ValueError(f"{w}: construction must be true or false")
        st = state(executor)
        if not isinstance(of, str) or of not in st.objects:
            raise ValueError(f"{w}: of must name a built shape, got {of!r}")
        sub_type = "edge" if e.get("edge") else "vertex"
        subs = {u.id: u for u in subshapes(executor, of, sub_type)}
        if not isinstance(sub, str) or sub not in subs:
            raise ValueError(f"{w}: give edge (or vertex): an id of {of} from get_subshapes, "
                             f"got {sub!r}")
        _projectable(sk, subs[sub].shape, w)
        keep = not e.get("construction", True)
        code = (f"model.do()\n{var}_p = {sk.name}.addProjection(model.selection("
                f"\"{sub_type.upper()}\", {subs[sub].naming!r}), {keep})\nmodel.do()\n"
                f"{var} = _wrap({var}_p.createdFeature())\n_name({var}, {var!r})")
        return code, [(w, var)]
    pts = (_points_list(e, w, 3 if closed else 2) if t == "polyline" else None)
    if t == "rectangle":
        a, b = _xy(e.get("corner1"), f"{w}.corner1"), _xy(e.get("corner2"), f"{w}.corner2")
        if abs(a[0] - b[0]) < 1e-9 or abs(a[1] - b[1]) < 1e-9:
            raise ValueError(f"{w}: the corners must differ in x and in y")
        code = (f"{var}_r = {sk.name}.addRectangle({a[0]!r}, {a[1]!r}, {b[0]!r}, {b[1]!r})\n"
                + "\n".join(f"{var}{k} = _wrap({var}_r.lines()[{k - 1}])\n"
                            f"_name({var}{k}, '{var}{k}')" for k in range(1, 5)))
        made = [(f"{w}{k}", f"{var}{k}") for k in range(1, 5)]
    else:   # polyline
        segs = list(zip(pts, pts[1:])) + ([(pts[-1], pts[0])] if closed else [])
        made = [(f"{w}{k}", f"{var}{k}") for k in range(1, len(segs) + 1)]
        code = "\n".join(_line_code(sk, v, a, b) for (_, v), (a, b) in zip(made, segs))
        joins = list(zip(made, made[1:])) + ([(made[-1], made[0])] if closed else [])
        code += "".join(f"\n{sk.name}.setCoincident({p[1]}.endPoint(), {q[1]}.startPoint())"
                        for p, q in joins)
    clash = [i for i, _ in made if i in taken]
    if clash:
        raise ValueError(f"{w}: makes the ids {', '.join(i for i, _ in made)}, but "
                         f"{', '.join(clash)} is taken — choose another id")
    return code, made


def _projectable(sk: Sketch, shape, w: str) -> None:
    """A straight edge along the sketch normal projects to a point, which SHAPER
    cannot make."""
    import GeomAPI
    if shape.shapeType() != GeomAPI.GeomAPI_Shape.EDGE:
        return
    edge = GeomAPI.GeomAPI_Edge(shape)
    if edge.isCircle() or edge.isArc():
        c, n = edge.circle().normal(), sketch_frame(sk)[3]
        if abs(c.x() * n[0] + c.y() * n[1] + c.z() * n[2]) < 1e-9:
            raise ValueError(f"{w}: that circle is seen edge-on by the sketch plane and "
                             "projects to a line; use a sketch plane it faces")
        return
    if not edge.isLine():
        return
    a, b = edge.firstPoint(), edge.lastPoint()
    d = [b.x() - a.x(), b.y() - a.y(), b.z() - a.z()]
    n = sketch_frame(sk)[3]
    length = math.sqrt(sum(c * c for c in d))
    if length and abs(sum(p * q for p, q in zip(d, n))) / length > 1 - 1e-9:
        raise ValueError(f"{w}: that edge is square to the sketch plane and projects to a "
                         "point; project one of its vertices instead (vertex: \"V..\")")


def _check_entity(e: dict) -> None:
    t, w = e["type"], e["id"]
    if t == "line":
        s, en = _xy(e.get("start"), f"{w}.start"), _xy(e.get("end"), f"{w}.end")
        if math.dist(s, en) < 1e-9:
            raise ValueError(f"{w}: start and end are the same point")
    elif t == "circle":
        _xy(e.get("center"), f"{w}.center")
        if _num(e.get("radius"), f"{w}.radius") <= 0:
            raise ValueError(f"{w}: radius must be positive")
    elif t == "arc":
        c = _xy(e.get("center"), f"{w}.center")
        s, en = _xy(e.get("start"), f"{w}.start"), _xy(e.get("end"), f"{w}.end")
        r1, r2 = math.dist(c, s), math.dist(c, en)
        if r1 < 1e-9:
            raise ValueError(f"{w}: start is on the center")
        if abs(r1 - r2) > 1e-6 * max(1.0, r1):
            raise ValueError(f"{w}: start and end must be the same distance from the "
                             f"center (got {r1:.6g} and {r2:.6g})")
    else:
        _xy(e.get("at"), f"{w}.at")


def _register(st, sk: Sketch, made: list, construction: bool) -> None:
    for i, var in made:
        kind = st.ns[var].feature().getKind()
        sk.entities[i] = Entity(i, KIND_TYPES.get(kind, "curve"), var, st.ns[var], construction)


_PT = {"type": "array", "items": {"type": "number"}}


@tool("sketch_add", category="sketch",
      description=(
          "Add entities to a sketch, each with an id you choose (unique in the sketch, "
          "letters and digits, e.g. L1, C2): "
          "{id, type: \"line\", start: [x, y], end: [x, y]}, "
          "{id, type: \"circle\", center, radius}, "
          "{id, type: \"arc\", center, start, end, reversed?} (counter-clockwise from start "
          "to end; reversed: clockwise), "
          "{id, type: \"point\", at}, "
          "{id, type: \"ellipse\", center, major_radius, minor_radius, angle?} (angle of "
          "the major axis, degrees), "
          "{id, type: \"spline\", points: [[x, y], ...], closed?} (smooth, through every point), "
          "{id, type: \"rectangle\", corner1, corner2} and "
          "{id, type: \"polyline\", points, closed?} (one line per side, ids <id>1, <id>2, "
          "..., already joined), "
          "{id, type: \"projection\", edge (or vertex): an id from get_subshapes, of: a "
          "shape} (a solid's edge seen in this sketch; a construction reference unless "
          "construction: false). "
          "construction: true marks a helper (a centre line, an axis of symmetry): it can "
          "be constrained against but forms no region and is never an open end. Lines and "
          "arcs are not joined by being drawn end to end: add coincident constraints for "
          "that. The reply gives dof, loose_ends, regions (closed areas with index, area, a "
          "point inside)."),
      schema={"type": "object", "properties": {
          "sketch": {"type": "string"},
          "entities": {"type": "array", "items": {"type": "object", "properties": {
              "id": {"type": "string"},
              "type": {"type": "string", "enum": list(_ENTITY_TYPES)},
              "start": _PT, "end": _PT, "center": _PT, "at": _PT,
              "corner1": _PT, "corner2": _PT,
              "points": {"type": "array", "items": _PT},
              "radius": {"type": "number"}, "major_radius": {"type": "number"},
              "minor_radius": {"type": "number"}, "angle": {"type": "number"},
              "edge": {"type": "string"}, "vertex": {"type": "string"},
              "of": {"type": "string"},
              "reversed": {"type": "boolean"}, "closed": {"type": "boolean"},
              "construction": {"type": "boolean"}},
              "required": ["id", "type"]}}},
          "required": ["sketch", "entities"]})
def tool_sketch_add(executor, args: dict):
    st = state(executor)
    sk = sketch_of(executor, args.get("sketch"))
    entities = args.get("entities") or []
    if not isinstance(entities, list) or not entities:
        raise ValueError("entities must be a non-empty list")
    taken = _taken(sk)
    plans = []
    for e in entities:
        if isinstance(e, dict) and isinstance(e.get("id"), str) and e["id"] in taken:
            raise ValueError(f"id {e['id']!r} is already used in this sketch — ids are "
                             f"unique per sketch (taken: {', '.join(sorted(taken))})")
        code, made = _plan_entity(executor, sk, e, taken)
        taken |= {e["id"]} | {i for i, _ in made}
        plans.append((e, code, made))
    before = snapshot(sk)
    records = []
    for e, code, made in plans:
        if e.get("construction") and e["type"] not in ("projection",):
            code += "".join(f"\n{v}.setAuxiliary(True)" for _, v in made)
        records.append(record("add" + e["type"].capitalize(), code,
                              out=[v for _, v in made], recv=sk.name))
    # A rename does not mark the sketch for recompute, so without this its
    # result keeps SHAPER's own name (SketchLine_N) for the last entity, and a
    # face selected by that name is lost in result.py.
    records.append(record("execute", f"{sk.name}.feature().execute()", recv=sk.name))
    try:
        exec_records(st, records)
    except Exception:
        _drop_helpers(st, records)
        raise
    for e, _, made in plans:
        construction = (e.get("construction", True) if e["type"] == "projection"
                        else bool(e.get("construction")))
        _register(st, sk, made, construction)
    return _reply(executor, sk, records, before)


# --- sketch_constrain -------------------------------------------------------

# type -> (SketchAPI method, the kinds of its refs, takes a value)
_POINT, _LINE, _ROUND, _CURVE = "point", "line", "round", "curve"
_CONSTRAINTS = {
    "coincident": ("setCoincident", None, False),
    "horizontal": ("setHorizontal", (_LINE,), False),
    "vertical": ("setVertical", (_LINE,), False),
    "parallel": ("setParallel", (_LINE, _LINE), False),
    "perpendicular": ("setPerpendicular", (_LINE, _LINE), False),
    "collinear": ("setCollinear", (_LINE, _LINE), False),
    "tangent": ("setTangent", None, False),
    "equal": ("setEqual", None, False),
    "midpoint": ("setMiddlePoint", (_POINT, _LINE), False),
    "fixed": ("setFixed", None, False),
    "length": ("setLength", (_LINE,), True),
    "distance": ("setDistance", None, True),
    "horizontal_distance": ("setHorizontalDistance", (_POINT, _POINT), True),
    "vertical_distance": ("setVerticalDistance", (_POINT, _POINT), True),
    "radius": ("setRadius", (_ROUND,), True),
    "angle": ("setAngle", (_LINE, _LINE), True),
}


class _Ref:
    def __init__(self, text, kind, src, base, var):
        self.text, self.kind, self.src, self.base, self.var = text, kind, src, base, var


def _resolve_ref(sk: Sketch, text) -> _Ref:
    if not isinstance(text, str):
        raise ValueError(f"a ref is a string such as \"L1\" or \"L1.end\", got {text!r}")
    if text == "ORIGIN":
        var = f"{sk.name}_ORIGIN"
        return _Ref(text, _POINT, f"SketchAPI_Point({var}.createdFeature()).coordinates()",
                    None, var)
    if text in sk.axes:
        var = f"{sk.name}_{text}"
        return _Ref(text, _LINE, f"{var}.createdFeature().result()", None, var)
    base, _, sub = text.partition(".")
    ent = sk.entities.get(base)
    if ent is None:
        known = ", ".join([*sk.entities, "ORIGIN", *sk.axes])
        raise ValueError(f"unknown ref {text!r} — this sketch has: {known}")
    v = ent.var
    if not sub:
        if ent.type == "line":
            return _Ref(text, _LINE, f"{v}.result()", base, v)
        if ent.type in ("circle", "arc"):
            return _Ref(text, _ROUND, f"{v}.results()[1]", base, v)
        if ent.type in ("ellipse", "spline", "curve"):
            return _Ref(text, _CURVE, f"{v}.result()", base, v)
        return _Ref(text, _POINT, f"{v}.coordinates()", base, v)
    open_ends = ent.type in ("spline", "curve") and "start" in geometry(ent)
    subs = {"line": {"start": "startPoint", "end": "endPoint"},
            "arc": {"start": "startPoint", "end": "endPoint", "center": "center"},
            "circle": {"center": "center"}, "ellipse": {"center": "center"},
            "spline": {"start": "startPoint", "end": "endPoint"} if open_ends else {},
            "curve": {"start": "startPoint", "end": "endPoint"} if open_ends else {},
            }.get(ent.type, {})
    if sub not in subs:
        valid = ", ".join(f"{base}.{s}" for s in subs) or f"{base} itself"
        raise ValueError(f"{text!r}: a {ent.type} has no {sub!r} — use {valid}")
    return _Ref(text, _POINT, f"{v}.{subs[sub]}()", base, v)


def _plan_constraint(sk: Sketch, c, used: set) -> tuple:
    if not isinstance(c, dict):
        raise ValueError(f"each constraint is an object, got {c!r}")
    t = c.get("type")
    if t not in _CONSTRAINTS:
        raise ValueError(f"unknown constraint type {t!r} — types: {', '.join(_CONSTRAINTS)}")
    method, kinds, valued = _CONSTRAINTS[t]
    raw = c.get("refs")
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{t}: refs must be a non-empty list")
    refs = [_resolve_ref(sk, r) for r in raw]
    got = [r.kind for r in refs]
    want = _expected(t, kinds, got)
    if want is not None:
        shown = ", ".join(f"{r.text} ({r.kind})" for r in refs)
        raise ValueError(f"{t} takes {want}; got {shown}")
    refs = _ordered(t, refs)
    args = [r.src for r in refs]
    if valued:
        args.append(repr(_num(c.get("value"), f"{t} value")))
    cid = c.get("id")
    if cid is None:
        while True:
            sk.next_k += 1
            cid = f"K{sk.next_k}"
            if cid not in used:
                break
    else:
        check_id(cid, "constraint")
        if cid in used:
            raise ValueError(f"constraint id {cid!r} is already used in this sketch")
    used.add(cid)
    var = f"{sk.name}_{cid}"
    code = f"{var} = {sk.name}.{method}({', '.join(args)})"
    return cid, t, var, code, refs, c.get("value")


def _expected(t, kinds, got) -> str | None:
    if kinds is not None:
        if sorted(got) == sorted(kinds):
            return None
        return " and ".join(kinds).replace(_ROUND, "circle/arc")
    n = len(got)
    if t == "coincident":
        return None if n == 2 and _POINT in got else "two refs, at least one a point"
    if t == "tangent":
        return (None if n == 2 and _ROUND in got and _POINT not in got
                else "two curves, at least one a circle or arc")
    if t == "equal":
        return (None if n == 2 and (got == [_LINE, _LINE] or set(got) == {_ROUND})
                else "two lines, or two circles/arcs")
    if t == "fixed":
        return None if n == 1 else "one ref"
    if t == "distance":
        ok = n == 2 and _POINT in got and _ROUND not in got
        return None if ok else "a point and a point or line"
    return None


def _ordered(t, refs):
    """SketchAPI wants the point first (coincident, distance, midpoint) and the
    line first for tangent."""
    if t in ("coincident", "distance", "midpoint") and refs[0].kind != _POINT:
        return [refs[1], refs[0]]
    if t == "tangent" and refs[1].kind == _LINE:
        return [refs[1], refs[0]]
    return refs


@tool("sketch_constrain", category="sketch",
      description=(
          "Add constraints to a sketch: [{id?, type, refs, value?}]. refs name entities "
          "(\"L1\", \"C2\"), their points (\"L1.start\", \"L1.end\", \"A3.center\"), or "
          "ORIGIN and the plane's axes (OX, OY, OZ). Types: coincident (two points, or a "
          "point and a curve), horizontal / vertical (line), parallel / perpendicular / "
          "collinear (two lines), tangent (two curves, one round), equal (two lines or two "
          "circles/arcs), midpoint (point, line), fixed (any), length (line, value), "
          "distance (point and point or line, value), horizontal_distance / "
          "vertical_distance (two points, value), radius (circle or arc, value), angle "
          "(two lines, value in degrees). Applied one at a time: a constraint that "
          "conflicts with the ones before it is removed and listed in `rejected` with the "
          "solver's message; the others stay. The solver may move geometry: `moved` gives "
          "the new coordinates of what moved."),
      schema={"type": "object", "properties": {
          "sketch": {"type": "string"},
          "constraints": {"type": "array", "items": {"type": "object", "properties": {
              "id": {"type": "string"},
              "type": {"type": "string", "enum": list(_CONSTRAINTS)},
              "refs": {"type": "array", "items": {"type": "string"}},
              "value": {"type": "number"}},
              "required": ["type", "refs"]}}},
          "required": ["sketch", "constraints"]})
def tool_sketch_constrain(executor, args: dict):
    st = state(executor)
    sk = sketch_of(executor, args.get("sketch"))
    constraints = args.get("constraints") or []
    if not isinstance(constraints, list) or not constraints:
        raise ValueError("constraints must be a non-empty list")
    used = _taken(sk)
    k0 = sk.next_k
    try:
        planned = [_plan_constraint(sk, c, used) for c in constraints]
    except ValueError:
        sk.next_k = k0
        raise
    before = snapshot(sk)
    records, rejected = [], []
    for cid, t, var, code, refs, value in planned:
        # result.py solves after each constraint too, as the live calls do
        rec = record("set" + t, code + "\nmodel.do()", out=[var],
                     refs=[r.var for r in refs], recv=sk.name)
        prior = snapshot(sk)
        err = ""
        try:
            with transaction():
                run(st, rec["code"])
        except Exception as e:  # noqa: BLE001 — reported as a rejected constraint
            err = str(e) or type(e).__name__
        if not err:
            err = solver_error(sk)
        if err:
            # result.py never sees this constraint, so neither may the geometry
            with transaction():
                obj = st.ns.pop(var, None)
                if obj is not None:
                    st.ns["shaper_doc"].removeFeature(obj.feature())
                restore(sk, prior)
            rejected.append({"id": cid, "type": t, "message": err})
            continue
        records.append(rec)
        sk.constraints[cid] = Constraint(cid, t, var, st.ns[var],
                                         [r.text for r in refs], value)
    return _reply(executor, sk, records, before, rejected=rejected)


# --- sketch_remove ----------------------------------------------------------

@tool("sketch_remove", category="sketch",
      description=(
          "Remove entities or constraints from a sketch by id. Removing an entity also "
          "removes every constraint that uses it; `removed` lists all of them."),
      schema={"type": "object", "properties": {
          "sketch": {"type": "string"},
          "ids": {"type": "array", "items": {"type": "string"}}},
          "required": ["sketch", "ids"]})
def tool_sketch_remove(executor, args: dict):
    st = state(executor)
    sk = sketch_of(executor, args.get("sketch"))
    ids = args.get("ids") or []
    if isinstance(ids, str):
        ids = [ids]
    unknown = [i for i in ids if i not in sk.entities and i not in sk.constraints]
    if unknown:
        known = ", ".join([*sk.entities, *sk.constraints]) or "nothing"
        raise ValueError(f"not in this sketch: {', '.join(map(str, unknown))} — it has: {known}")
    ents = [i for i in ids if i in sk.entities]
    cons = list(dict.fromkeys(
        [i for i in ids if i in sk.constraints]
        + [cid for cid, c in sk.constraints.items()
           if any(r.partition(".")[0] in ents for r in c.refs)]))
    before = snapshot(sk)
    records = []
    for i in cons + ents:
        obj = sk.constraints.get(i) or sk.entities.get(i)
        records.append(record("removeFeature", f"shaper_doc.removeFeature({obj.var}.feature())",
                              refs=[obj.var], recv=sk.name))
    exec_records(st, records)
    for i in cons:
        sk.constraints.pop(i)
    for i in ents:
        sk.entities.pop(i)
    before = {k: v for k, v in before.items() if k in sk.entities}
    return _reply(executor, sk, records, before, removed=[*ents, *cons])


# --- sketch_status ----------------------------------------------------------

@tool("sketch_status", category="sketch",
      description=(
          "The full state of a sketch: every entity with its current (solved) "
          "coordinates, every constraint, dof, loose_ends and regions."),
      schema={"type": "object", "properties": {"sketch": {"type": "string"}},
              "required": ["sketch"]})
def tool_sketch_status(executor, args: dict):
    sk = sketch_of(executor, args.get("sketch"))
    entities = {i: {"type": e.type, **({"construction": True} if e.construction else {}),
                    **geometry(e)} for i, e in sk.entities.items()}
    constraints = {i: {"type": c.type, "refs": c.refs,
                       **({"value": c.value} if c.value is not None else {})}
                   for i, c in sk.constraints.items()}
    return _reply(executor, sk, [], None, entities=entities, constraints=constraints)


# --- sketch_fillet / sketch_copy --------------------------------------------

def _made_by(st, sk: Sketch, op_code: str, op_var: str, getter: str, prefix: str,
             fn: str, sources: list) -> tuple[list, list]:
    """Run an operation that makes sketch entities, then name each one
    <prefix>1, <prefix>2, ...: two records, both run exactly as result.py runs them.
    A pattern of a copy lists that copy among its own: left out."""
    new = f"{op_var}_new"
    op_code += (f"\n{new} = [_x for _x in {op_var}.{getter} "
                f"if _x.feature().data().name() not in {tuple(sources)!r}]")
    first = record(fn, op_code, out=[op_var], recv=sk.name)
    had_error = bool(solver_error(sk))
    try:
        exec_records(st, [first])
    except Exception:
        _drop_helpers(st, [first])
        raise
    n = len(st.ns[new])
    if n == 0:
        _drop_helpers(st, [first])
        with transaction():
            st.ns["shaper_doc"].removeFeature(st.ns.pop(op_var).feature())
        raise ValueError("the operation made no entity")
    made = [(f"{prefix}{k}", f"{sk.name}_{prefix}{k}") for k in range(1, n + 1)]
    code = "\n".join(f"{v} = _wrap({new}[{k}])\n_name({v}, {v!r})"
                     for k, (_, v) in enumerate(made))
    second = record("name", code + f"\n{sk.name}.feature().execute()",
                    out=[v for _, v in made], recv=sk.name)
    try:
        exec_records(st, [second])
        err = None if had_error else solver_error(sk)
    except Exception:
        with transaction():
            st.ns["shaper_doc"].removeFeature(st.ns.pop(op_var).feature())
        _drop_helpers(st, [first])
        raise
    if err:
        with transaction():
            for _, v in made:
                st.ns.pop(v, None)
            st.ns["shaper_doc"].removeFeature(st.ns.pop(op_var).feature())
        _drop_helpers(st, [first])
        raise ValueError(f"the copy conflicts with the constraints of the sketch ({err}); "
                         "nothing was added")
    return [first, second], made


def _new_prefix(sk: Sketch, prefix) -> str:
    check_id(prefix, "id")
    taken = _taken(sk)
    clash = sorted(i for i in taken if i == prefix or
                   (i.startswith(prefix) and i[len(prefix):].isdigit()))
    if clash:
        raise ValueError(f"id {prefix!r}: the new entities are named {prefix}1, {prefix}2, "
                         f"... and {', '.join(clash)} is taken — choose another id")
    return prefix


@tool("sketch_fillet", category="sketch",
      description=(
          "Round the corner where two lines meet with an arc of `radius`: `at` is the "
          "corner (\"L1.end\"); both lines are trimmed to the arc, which gets the id `id` "
          "and stays tangent to them. Nothing else moves."),
      schema={"type": "object", "properties": {
          "sketch": {"type": "string"}, "at": {"type": "string"},
          "radius": {"type": "number"}, "id": {"type": "string"}},
          "required": ["sketch", "at", "radius", "id"]})
def tool_sketch_fillet(executor, args: dict):
    st = state(executor)
    sk = sketch_of(executor, args.get("sketch"))
    fid = args.get("id")
    check_id(fid, "entity")
    if fid in _taken(sk):
        raise ValueError(f"id {fid!r} is already used in this sketch")
    ref = _resolve_ref(sk, args.get("at"))
    first, _, end1 = ref.text.partition(".")
    if ref.kind != _POINT or end1 not in ("start", "end") or sk.entities[first].type != "line":
        raise ValueError(f"at must be the end of a line, like \"L1.end\", got {ref.text!r}")
    if sk.entities[first].construction:
        raise ValueError(f"{first} is a construction line: round a corner of the profile")
    r = _num(args.get("radius"), "radius")
    if r <= 0:
        raise ValueError("radius must be positive")
    geos = snapshot(sk)
    p = geos[first][end1]
    meets = [(i, k) for i, e in sk.entities.items() if i != first and e.type == "line"
             and not e.construction for k in ("start", "end") if math.dist(geos[i][k], p) < 1e-6]
    if len(meets) != 1:
        raise ValueError(f"{ref.text} must be the corner of exactly two drawn lines; "
                         f"{len(meets)} other lines end there")
    second, end2 = meets[0]
    other = lambda i, k: geos[i]["end" if k == "start" else "start"]
    unit = lambda a: [c / math.hypot(*a) for c in a]
    u = unit([other(first, end1)[i] - p[i] for i in range(2)])
    v = unit([other(second, end2)[i] - p[i] for i in range(2)])
    half = math.acos(max(-1.0, min(1.0, u[0] * v[0] + u[1] * v[1]))) / 2
    if half < 1e-6 or math.pi / 2 - half < 1e-6:
        raise ValueError(f"{first} and {second} are in line at {ref.text}: no corner to round")
    t = r / math.tan(half)
    for i, k in ((first, end1), (second, end2)):
        if t >= math.dist(p, other(i, k)) * (1 - 1e-6):
            raise ValueError(f"radius {r:g} is too large: it needs {t:.4g} of {i}, which is "
                             f"only {math.dist(p, other(i, k)):.4g} long")
    t1 = [p[i] + u[i] * t for i in range(2)]
    t2 = [p[i] + v[i] * t for i in range(2)]
    w = unit([u[0] + v[0], u[1] + v[1]])
    c = [p[i] + w[i] * r / math.sin(half) for i in range(2)]
    rev = (t1[0] - c[0]) * (t2[1] - c[1]) - (t1[1] - c[1]) * (t2[0] - c[0]) < 0
    l1, l2 = sk.entities[first].var, sk.entities[second].var
    pt = {"start": "startPoint", "end": "endPoint"}
    attr = {"start": "StartPoint", "end": "EndPoint"}
    var = f"{sk.name}_{fid}"
    f6 = lambda x: repr(round(x, 9) + 0.0)
    code = (f"_unjoin({sk.name}, ({l1!r}, {attr[end1]!r}), ({l2!r}, {attr[end2]!r}))\n"
            f"{l1}.{pt[end1]}().setValue({f6(t1[0])}, {f6(t1[1])})\n"
            f"{l2}.{pt[end2]}().setValue({f6(t2[0])}, {f6(t2[1])})\n"
            f"{var} = {sk.name}.addArc({f6(c[0])}, {f6(c[1])}, {f6(t1[0])}, {f6(t1[1])}, "
            f"{f6(t2[0])}, {f6(t2[1])}, {rev})\n_name({var}, {var!r})\n"
            f"{sk.name}.setCoincident({var}.startPoint(), {l1}.{pt[end1]}())\n"
            f"{sk.name}.setCoincident({var}.endPoint(), {l2}.{pt[end2]}())\n"
            f"{sk.name}.setTangent({l1}.result(), {var}.results()[1])\n"
            f"{sk.name}.setTangent({l2}.result(), {var}.results()[1])\n"
            f"{sk.name}.setRadius({var}.results()[1], {r!r})\nmodel.do()\n"
            f"{sk.name}.feature().execute()")
    rec = record("fillet", code, out=[var], refs=[l1, l2], recv=sk.name)
    try:
        exec_records(st, [rec])
        err = solver_error(sk)
    except Exception:
        with transaction():
            restore(sk, geos)
        raise
    if err:
        with transaction():
            obj = st.ns.pop(var, None)
            if obj is not None:
                st.ns["shaper_doc"].removeFeature(obj.feature())
            restore(sk, geos)
        raise ValueError(f"the fillet conflicts with the constraints on {first} or {second} "
                         f"({err}); remove them first")
    # our own coincident constraint on that corner went with it
    corner = {f"{first}.{end1}", f"{second}.{end2}"}
    for cid, k in list(sk.constraints.items()):
        if k.type == "coincident" and set(k.refs) == corner:
            sk.constraints.pop(cid)
            st.ns.pop(k.var, None)
    _register(st, sk, [(fid, var)], False)
    return _reply(executor, sk, [rec], geos)


def _sources(sk: Sketch, ids) -> tuple[list, list]:
    if not isinstance(ids, list) or not ids or not all(isinstance(i, str) for i in ids):
        raise ValueError(f"of must be a list of entity ids, got {ids!r}")
    if len(set(ids)) != len(ids):
        raise ValueError(f"of lists the same entity twice: {ids!r}")
    srcs, refs = [], []
    for i in ids:
        ref = _resolve_ref(sk, i)
        if ref.base is None or "." in ref.text or ref.kind == _POINT:
            raise ValueError(f"{i!r}: copy curves (lines, arcs, circles, ...), not points")
        srcs.append(ref.src)
        refs.append(ref.var)
    return srcs, refs


@tool("sketch_copy", category="sketch",
      description=(
          "Make copies of curves of a sketch (`of`: entity ids); the new entities are "
          "named <id>1, <id>2, ... `op`: \"mirror\" across the line `axis` (an entity id or "
          "an axis of the plane); \"offset\" by `distance` (a parallel curve, the side "
          "set by the sign); \"linear\" `count` copies in all every `vector` [dx, dy]; "
          "\"circular\" `count` copies in all round `center` [x, y], evenly over the full "
          "circle or every `angle` degrees. Copies stay tied to the originals."),
      schema={"type": "object", "properties": {
          "sketch": {"type": "string"},
          "op": {"type": "string", "enum": ["mirror", "offset", "linear", "circular"]},
          "of": {"type": "array", "items": {"type": "string"}},
          "id": {"type": "string"}, "axis": {"type": "string"},
          "distance": {"type": "number"}, "vector": _PT, "center": _PT,
          "count": {"type": "integer"}, "angle": {"type": "number"}},
          "required": ["sketch", "op", "of", "id"]})
def tool_sketch_copy(executor, args: dict):
    st = state(executor)
    sk = sketch_of(executor, args.get("sketch"))
    prefix = _new_prefix(sk, args.get("id"))
    srcs, refs = _sources(sk, args.get("of"))
    objs = f"[{', '.join(srcs)}]"
    op = args.get("op")
    ov = f"{sk.name}_{prefix}_op"
    count = lambda: _copies(args.get("count"))
    own = {"mirror": {"axis"}, "offset": {"distance"}, "linear": {"vector", "count"},
           "circular": {"center", "count", "angle"}}
    if op in own:
        stray = sorted(k for k in ("axis", "distance", "vector", "center", "count", "angle")
                       if args.get(k) is not None and k not in own[op])
        if stray:
            raise ValueError(f"{', '.join(stray)} does not apply to op {op}")
    if op == "mirror":
        axis = _resolve_ref(sk, args.get("axis"))
        if axis.kind != _LINE:
            raise ValueError(f"axis must be a line of the sketch, got {args.get('axis')!r}")
        if axis.base is not None and axis.base in args["of"]:
            raise ValueError(f"{axis.base} is the axis: it cannot also be mirrored")
        code, getter = f"{ov} = {sk.name}.addMirror({axis.src}, {objs})\nmodel.do()", "mirrored()"
    elif op == "offset":
        d = _num(args.get("distance"), "distance")
        if d == 0:
            raise ValueError("distance must not be 0")
        if d < 0:
            for i in args["of"]:
                ent = sk.entities[i]
                if ent.type in ("circle", "arc") and -d >= geometry(ent)["radius"] - 1e-9:
                    raise ValueError(f"{i}: an inward offset of {-d:g} reaches its radius")
        code = f"{ov} = {sk.name}.addOffset({objs}, {abs(d)!r}, {d < 0})\nmodel.do()"
        getter = "offset()"
    elif op == "linear":
        dx, dy = _xy(args.get("vector"), "vector")
        if abs(dx) < 1e-9 and abs(dy) < 1e-9:
            raise ValueError("vector must not be [0, 0]")
        n = count()
        code = (f"{ov}_a = {sk.name}.addPoint(0.0, 0.0)\n{ov}_a.setAuxiliary(True)\n"
                f"{ov}_b = {sk.name}.addPoint({dx!r}, {dy!r})\n{ov}_b.setAuxiliary(True)\n"
                f"{ov} = {sk.name}.addTranslation({objs}, {ov}_a.coordinates(), "
                f"{ov}_b.coordinates(), {n})\nmodel.do()")
        getter = "translated()"
    elif op == "circular":
        cx, cy = _xy(args.get("center"), "center")
        n = count()
        if args.get("angle") is None:
            angle, full = 360.0, True
        else:
            angle, full = _num(args["angle"], "angle"), False
            if angle % 360 == 0 or abs(angle) * (n - 1) >= 360 - 1e-9:
                raise ValueError("copies would overlap: keep angle x (count - 1) under 360, "
                                 "or leave angle out to spread them round the full circle")
        code = (f"{ov}_c = {sk.name}.addPoint({cx!r}, {cy!r})\n{ov}_c.setAuxiliary(True)\n"
                f"{ov} = {sk.name}.addRotation({objs}, {ov}_c.coordinates(), {angle!r}, "
                f"{n}, {full})\nmodel.do()")
        getter = "rotated()"
    else:
        raise ValueError(f"op must be mirror, offset, linear or circular, got {op!r}")
    before = snapshot(sk)
    records, made = _made_by(st, sk, code, ov, getter, prefix, "add" + op.capitalize(), refs)
    records[0]["args"] = [{"kind": "expr", "value": "", "refs": sorted(set(refs))}]
    _register(st, sk, made, False)
    return _reply(executor, sk, records, before, made=[i for i, _ in made])


def _copies(value) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)) \
            or not math.isfinite(value) or value != int(value) or not 2 <= value <= 200:
        raise ValueError(f"count must be a whole number from 2 to 200, got {value!r}")
    return int(value)
