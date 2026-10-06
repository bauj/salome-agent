import math

from executor.shaper.state import (
    PUBLISH_LINE, Constraint, Entity, Sketch, check_id, geometry, new_name, offset_plane, record,
    restore, run, sketch_of, snapshot, solver_error, state, status, tool, transaction,
)

# plane -> the world axes lying in it, offered as sketch references
_PLANES = {"XOY": ("OX", "OY"), "XOZ": ("OX", "OZ"), "YOZ": ("OY", "OZ")}
_ENTITY_TYPES = ("line", "circle", "arc", "point")


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
          "world X, Y."),
      schema={"type": "object", "properties": {
          "plane": {"description": "\"XOY\" | \"XOZ\" | \"YOZ\", or {base, offset}",
                    "anyOf": [{"type": "string", "enum": list(_PLANES)},
                              {"type": "object", "properties": {
                                  "base": {"type": "string", "enum": list(_PLANES)},
                                  "offset": {"type": "number"}},
                               "required": ["base", "offset"]}]}},
          "required": ["plane"]})
def tool_create_sketch(executor, args: dict):
    st = state(executor)
    plane = args.get("plane", "XOY")
    name = new_name(executor, "sketch")
    records = []
    refs = []
    if isinstance(plane, dict):
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
        raise ValueError(f"plane must be one of {', '.join(_PLANES)} or "
                         f"{{base, offset}}, got {plane!r}")
    records.append(record("addSketch", f"{name} = model.addSketch(shaper_doc, {target})",
                          out=[name], refs=refs, publish=True))
    axes = _PLANES[base]
    records.append(record("addProjection", (
        f'{name}_ORIGIN = {name}.addProjection(model.selection("VERTEX", '
        f'"PartSet/Origin"), False)'), out=[f"{name}_ORIGIN"], recv=name))
    for ax in axes:
        records.append(record("addProjection", (
            f'{name}_{ax} = {name}.addProjection(model.selection("EDGE", '
            f'"PartSet/{ax}"), False)'), out=[f"{name}_{ax}"], recv=name))
    exec_records(st, records)
    sk = Sketch(name, st.ns[name], axes)
    st.sketches[name] = sk
    return _reply(executor, sk, records, None)


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
    code = f"{var} = {call}\n_name({var}, {var!r})"
    if e.get("construction"):
        code += f"\n{var}.setAuxiliary(True)"
    return code


def _check_entity(e, taken: set) -> dict:
    if not isinstance(e, dict):
        raise ValueError(f"each entity is an object, got {e!r}")
    check_id(e.get("id"), "entity")
    if e["id"] in taken:
        raise ValueError(f"id {e['id']!r} is already used in this sketch — ids are "
                         f"unique per sketch (taken: {', '.join(sorted(taken))})")
    t = e.get("type")
    if t not in _ENTITY_TYPES:
        raise ValueError(f"{e['id']}: type must be one of {', '.join(_ENTITY_TYPES)}, got {t!r}")
    w = e["id"]
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
    return e


@tool("sketch_add", category="sketch",
      description=(
          "Add entities to a sketch, each with an id you choose (unique in the sketch, "
          "letters and digits, e.g. L1, C2): "
          "{id, type: \"line\", start: [x, y], end: [x, y]}, "
          "{id, type: \"circle\", center, radius}, "
          "{id, type: \"arc\", center, start, end, reversed?} (counter-clockwise from start "
          "to end; reversed: clockwise), "
          "{id, type: \"point\", at}. "
          "construction: true marks a helper (a centre line, an axis of symmetry): it can "
          "be constrained against but forms no region and is never an open end. Entities "
          "are not joined by being drawn end to end: add coincident constraints for that. "
          "The reply gives dof, loose_ends, regions (closed areas with index, area, a "
          "point inside)."),
      schema={"type": "object", "properties": {
          "sketch": {"type": "string"},
          "entities": {"type": "array", "items": {"type": "object", "properties": {
              "id": {"type": "string"},
              "type": {"type": "string", "enum": list(_ENTITY_TYPES)},
              "start": {"type": "array", "items": {"type": "number"}},
              "end": {"type": "array", "items": {"type": "number"}},
              "center": {"type": "array", "items": {"type": "number"}},
              "radius": {"type": "number"},
              "at": {"type": "array", "items": {"type": "number"}},
              "reversed": {"type": "boolean"},
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
    for e in entities:
        _check_entity(e, taken)
        taken.add(e["id"])
    before = snapshot(sk)
    records = []
    for e in entities:
        var = f"{sk.name}_{e['id']}"
        records.append(record("add" + e["type"].capitalize(), _entity_code(sk, e, var),
                              out=[var], recv=sk.name))
    # A rename does not mark the sketch for recompute, so without this its
    # result keeps SHAPER's own name (SketchLine_N) for the last entity, and a
    # face selected by that name is lost in result.py.
    records.append(record("execute", f"{sk.name}.feature().execute()", recv=sk.name))
    exec_records(st, records)
    for e in entities:
        var = f"{sk.name}_{e['id']}"
        sk.entities[e["id"]] = Entity(e["id"], e["type"], var, st.ns[var],
                                      bool(e.get("construction")))
    return _reply(executor, sk, records, before)


# --- sketch_constrain -------------------------------------------------------

# type -> (SketchAPI method, the kinds of its refs, takes a value)
_POINT, _LINE, _ROUND = "point", "line", "round"
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
        return _Ref(text, _POINT, f"{v}.coordinates()", base, v)
    subs = {"line": {"start": "startPoint", "end": "endPoint"},
            "arc": {"start": "startPoint", "end": "endPoint", "center": "center"},
            "circle": {"center": "center"}}.get(ent.type, {})
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
