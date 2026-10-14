"""The run's SHAPER document, the sketches built in it, and what every tool
shares: running SHAPER code, recording it, and reporting a sketch's state.

Every change is made by executing the same source line that goes into
result.py, so the live document and the generated script cannot drift."""
import math
import re
from contextlib import contextmanager

from executor.registry import ToolRegistry
from executor.geom.helpers import _next_name

registry = ToolRegistry(valid_categories={
    "primitive", "sketch", "swept", "boolean", "transform", "feature", "display", "query"})
tool = registry.tool

# Renames a feature and its results to the published name, so selections by
# name ("all-in-<sketch>", a face's naming string) read the same live and in
# result.py.
NAME_HELPER = '''def _name(obj, n):
    model.do()
    feat = obj.feature()
    feat.data().setName(n)
    for i, r in enumerate(feat.results()):
        r.data().setName(n if i == 0 else f"{n}_{i + 1}")


def _unjoin(sk, a, b):
    """Remove the coincidences tying the point a to the point b, each given as
    (feature name, attribute id): a corner being rounded."""
    import ModelAPI
    cf = ModelAPI.featureToCompositeFeature(sk.feature())
    for i in reversed(range(cf.numberOfSubs())):
        f = cf.subFeature(i)
        if not f.getKind().startswith("SketchConstraintCoincidence"):
            continue
        ends = set()
        for k in ("ConstraintEntityA", "ConstraintEntityB"):
            ra = f.refattr(k)
            if ra.isInitialized() and not ra.isObject() and ra.attr() is not None:
                ends.add((ModelAPI.objectToFeature(ra.attr().owner()).data().name(),
                          ra.attr().id()))
        if ends == {a, b}:
            sk.feature().document().removeFeature(f)


def _wrap(ent):
    """A sketch entity SHAPER made itself (a fillet arc, a mirrored copy) as its API object."""
    f = ent if hasattr(ent, "getKind") else ent.feature()
    return getattr(SketchAPI, "SketchAPI_" + f.getKind()[len("Sketch"):])(f)
'''

PREAMBLE = ('from salome.shaper import model\n'
            'import SketchAPI\n'
            'from SketchAPI import SketchAPI_Point\n'
            'model.begin()\n'
            'shaper_doc = model.addPart(model.moduleDocument()).document()\n'
            'model.end()\n' + NAME_HELPER + 'model.begin()\n')
POSTAMBLE = 'model.end()\nmodel.publishToShaperStudy()\n'
PUBLISH_LINE = "_name({var}, {name!r})"

_ID = re.compile(r"[A-Za-z][A-Za-z0-9_]*$")
TOL = 1e-5
# Set once this server process has opened a SHAPER document: until then every
# hook (clear, forget) leaves SHAPER unloaded.
_USED = False


class Entity:
    def __init__(self, id_, type_, var, api, construction):
        self.id, self.type, self.var, self.api = id_, type_, var, api
        self.construction = construction


class Constraint:
    def __init__(self, id_, type_, var, api, refs, value=None):
        self.id, self.type, self.var, self.api, self.refs = id_, type_, var, api, refs
        self.value = value


class Sketch:
    def __init__(self, name, api, axes, frame=None):
        self.name, self.api, self.axes = name, api, axes
        # (origin, x_dir, y_dir) of a sketch on a face or a construction plane:
        # its coordinates are local to that frame
        self.frame = frame
        self.entities: dict[str, Entity] = {}
        self.constraints: dict[str, Constraint] = {}
        self.next_k = 0
        # while a closed sketch is edited again: the later features' check
        # status from before, to tell which ones the edit breaks
        self.baseline: dict | None = None


class ShaperState:
    def __init__(self, ns):
        self.ns = ns
        self.sketches: dict[str, Sketch] = {}
        # published feature name -> kind ("face", "solid", ...)
        self.objects: dict[str, str] = {}
        # group name -> the shape its sub-shapes belong to (make_group)
        self.groups: dict[str, str] = {}
        # the sketch being edited (executor/shaper/editing.py), or None
        self.editing: str | None = None


def model():
    from salome.shaper import model as m
    return m


@contextmanager
def transaction():
    """One SHAPER operation, closed even when its body raises: an operation left
    open breaks every later call on the server."""
    m = model()
    m.begin()
    try:
        yield m
    finally:
        m.end()


def state(executor) -> ShaperState:
    global _USED
    st = executor.__dict__.get("shaper_state")
    if st is None:
        import SketchAPI
        from SketchAPI import SketchAPI_Point
        _USED = True
        ns = {"model": model(), "SketchAPI": SketchAPI, "SketchAPI_Point": SketchAPI_Point}
        with transaction() as m:
            ns["shaper_doc"] = m.addPart(m.moduleDocument()).document()
        exec(NAME_HELPER, ns)
        st = executor.__dict__["shaper_state"] = ShaperState(ns)
    return st


def reset(executor) -> None:
    """A new run: forget the document handle. The Part stays, as GEOM objects
    stay in the study."""
    executor.__dict__.pop("shaper_state", None)


def clear(executor) -> None:
    """Remove every Part from the SHAPER session (clear_study)."""
    reset(executor)
    if not _USED:
        return
    with transaction() as m:
        part_set = m.moduleDocument()
        for feat in list(part_set.allFeatures()):
            if feat.getKind() == "Part":
                part_set.removeFeature(feat)


def _helpers(st, name: str) -> list:
    """The helper variables of `name`, named `<name>_...`: not those of another
    object whose own name starts with `<name>_` (forgetting `base` must leave
    the group `base_f` and its helpers alone)."""
    owners = set(st.objects) | set(st.sketches)
    return [v for v in st.ns if v.startswith(f"{name}_") and v not in owners
            and max((o for o in owners if v.startswith(f"{o}_")), key=len) == name]


def _subs(feat) -> list:
    import ModelAPI
    comp = ModelAPI.featureToCompositeFeature(feat)
    return [comp.subFeature(i) for i in range(comp.numberOfSubs())] if comp else []


def _flush_redisplay() -> None:
    import EventsAPI
    loop = EventsAPI.Events_Loop
    loop.loop().flush(loop.eventByName("ObjectsToRedisplay"))


def _hide(feats: list) -> None:
    """With the SHAPER module active its viewer draws feature results itself,
    and a feature removed from Python stays drawn: hide them while they are
    still valid, so the viewer erases them."""
    todo = list(feats)
    while todo:
        feat = todo.pop()
        todo += _subs(feat)
        for res in feat.results():
            res.setDisplayed(False)
    _flush_redisplay()


def show(executor, names: list, shown: bool) -> list:
    """Show or hide SHAPER results in SHAPER's own presentation, the one its
    tree's eye controls; nothing is drawn until SHAPER's GUI is loaded."""
    st = executor.__dict__.get("shaper_state")
    if st is None:
        return []
    feats, done = [], []
    for name in names:
        if name in st.sketches:
            feats.append(st.sketches[name].api.feature())
        elif name in st.objects:
            feats.append(st.ns[name].feature())
        else:
            continue
        done.append(name)
    # Outside an operation: SHAPER rolls back a flag-only operation at its end.
    for feat in feats:
        for res in feat.results():
            res.setDisplayed(shown)
    if feats:
        _flush_redisplay()
    return done


def forget(executor, names: list) -> list:
    """Remove published shaper objects (sketches, features) and their study
    entries; names this module did not build are ignored."""
    st = executor.__dict__.get("shaper_state")
    if st is None:
        return []
    gone = []
    with transaction():
        for name in names:
            if name in st.sketches:
                if st.editing == name:
                    st.editing = None
                helpers = _helpers(st, name)
                feats = [st.sketches.pop(name).api.feature()]
                plane = st.ns.get(f"{name}_plane")
                if plane is not None:
                    feats.append(plane.feature())
                for var in helpers:
                    st.ns.pop(var)
            elif name in st.objects:
                helpers = _helpers(st, name)
                st.objects.pop(name)
                st.groups.pop(name, None)
                feats = [st.ns[name].feature()]
                for var in helpers:
                    feats.append(st.ns.pop(var).feature())   # make_face's region helper
            else:
                continue
            _hide(feats)
            for feat in feats:
                st.ns["shaper_doc"].removeFeature(feat)
            st.ns.pop(name, None)
            gone.append(name)
    if gone:
        _remove_study_objects(executor, gone)
    return gone


def _remove_study_objects(executor, names: list) -> None:
    salome = executor.namespace["salome"]
    study = salome.myStudy
    builder = study.NewBuilder()
    for name in names:
        for sobj in study.FindObjectByName(name, "SHAPERSTUDY"):
            builder.RemoveObjectWithChildren(sobj)


def new_name(executor, prefix: str) -> str:
    return _next_name(executor, prefix)


def run(st: ShaperState, code: str) -> None:
    exec(code, st.ns)


def record(fn: str, code: str, *, out=(), refs=(), recv=None, publish=False) -> dict:
    rec = {"fn": fn, "emit": "stmt", "code": code, "out_vars": list(out),
           "args": [{"kind": "expr", "value": "", "refs": sorted(set(refs))}] if refs else [],
           "publish": list(out) if publish else None}
    if recv is not None:
        rec["recv"] = recv
    return rec


def check_id(id_: str, what: str) -> None:
    if not isinstance(id_, str) or not _ID.fullmatch(id_):
        raise ValueError(f"{what} id {id_!r} must start with a letter and use only "
                         "letters, digits and underscores (e.g. L1, C2, K3)")
    if id_.endswith("_op"):
        raise ValueError(f"{what} id {id_!r}: ids ending in _op are reserved for sketch_copy")


def publish_study(executor) -> None:
    model().publishToShaperStudy()


# --- geometry snapshots -----------------------------------------------------

def _pt(p) -> list:
    return [round(p.x(), 6), round(p.y(), 6)]


def geometry(ent: Entity) -> dict:
    a = ent.api
    if ent.type == "line":
        return {"start": _pt(a.startPoint()), "end": _pt(a.endPoint())}
    if ent.type == "circle":
        return {"center": _pt(a.center()), "radius": round(a.radius().value(), 6)}
    if ent.type == "arc":
        c = _pt(a.center())
        s = _pt(a.startPoint())
        return {"center": c, "start": s, "end": _pt(a.endPoint()),
                "radius": round(math.dist(c, s), 6)}
    if ent.type == "ellipse":
        return {"center": _pt(a.center()), "major_radius": round(a.majorRadius().value(), 6),
                "minor_radius": round(a.minorRadius().value(), 6)}
    if ent.type in ("spline", "curve"):
        if a.feature().getKind() == "SketchBSplinePeriodic":
            return {"closed": True}
        return {"start": _pt(a.startPoint()), "end": _pt(a.endPoint())}
    return {"at": _pt(a.coordinates())}


# a sketch feature kind -> the entity type the tools call it
KIND_TYPES = {"SketchLine": "line", "SketchCircle": "circle", "SketchArc": "arc",
              "SketchPoint": "point", "SketchEllipse": "ellipse",
              "SketchBSpline": "spline", "SketchBSplinePeriodic": "spline"}


def snapshot(sk: Sketch) -> dict:
    return {i: geometry(e) for i, e in sk.entities.items()}


def restore(sk: Sketch, geos: dict) -> None:
    """Put entities back where `geos` (a snapshot) had them."""
    for i, geo in moved(geos, snapshot(sk)).items():
        if sk.entities[i].type not in ("line", "circle", "arc", "point"):
            continue    # an ellipse or spline is moved through its own points
        old, a = geos[i], sk.entities[i].api
        if "start" in old:
            a.startPoint().setValue(*old["start"])
            a.endPoint().setValue(*old["end"])
        if "center" in old:
            a.center().setValue(*old["center"])
        if sk.entities[i].type == "circle":
            a.radius().setValue(old["radius"])
        if "at" in old:
            a.coordinates().setValue(*old["at"])


def moved(before: dict, after: dict) -> dict:
    out = {}
    for i, geo in after.items():
        old = before.get(i)
        if old is None:
            continue
        if any(_far(old[k], geo[k]) for k in geo if k in old):
            out[i] = geo
    return out


def _far(a, b) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return a != b
    if isinstance(a, list):
        return any(abs(x - y) > 1e-6 for x, y in zip(a, b))
    return abs(a - b) > 1e-6


# --- sketch state -----------------------------------------------------------

def dof(sk: Sketch) -> int:
    try:
        return int(model().dof(sk.api))
    except Exception:
        return -1


def solver_error(sk: Sketch) -> str:
    try:
        return re.sub(r"<[^>]+>", "", sk.api.solverError().value()).strip()
    except Exception:
        return ""


def _ends(ent: Entity, geo: dict) -> list:
    if ent.type in ("line", "arc") or (ent.type in ("spline", "curve") and "start" in geo):
        return [(f"{ent.id}.start", geo["start"]), (f"{ent.id}.end", geo["end"])]
    return []


def sketch_frame(sk: Sketch) -> tuple:
    """(origin, x_dir, y_dir, normal) of a sketch, from its feature."""
    import GeomDataAPI
    f = sk.api.feature()
    o = GeomDataAPI.geomDataAPI_Point(f.attribute("Origin")).pnt()
    x = GeomDataAPI.geomDataAPI_Dir(f.attribute("DirX")).dir()
    n = GeomDataAPI.geomDataAPI_Dir(f.attribute("Norm")).dir()
    xyz = lambda p: [p.x(), p.y(), p.z()]
    return xyz(o), xyz(x), xyz(n.cross(x)), xyz(n)


def _on_curve(p, ent: Entity, sk: Sketch) -> bool:
    """p (sketch coordinates) lies on an ellipse or spline: measured on its edge."""
    import GeomAPI
    from GeomAlgoAPI import GeomAlgoAPI_PointBuilder as PB, GeomAlgoAPI_ShapeTools as T
    res = ent.api.feature().lastResult()
    if res is None or res.shape() is None:
        return False
    o, x, y, _ = sketch_frame(sk)
    w = [o[i] + p[0] * x[i] + p[1] * y[i] for i in range(3)]
    return T.minimalDistance(PB.vertex(GeomAPI.GeomAPI_Pnt(*w)), res.shape()) < TOL


def _on(p, ent: Entity, geo: dict, sk: Sketch | None = None) -> bool:
    if ent.type in ("ellipse", "spline", "curve"):
        return sk is not None and _on_curve(p, ent, sk)
    if ent.type == "line":
        (ax, ay), (bx, by) = geo["start"], geo["end"]
        dx, dy = bx - ax, by - ay
        L2 = dx * dx + dy * dy
        t = 0.0 if L2 == 0 else max(0.0, min(1.0, ((p[0] - ax) * dx + (p[1] - ay) * dy) / L2))
        return math.dist(p, (ax + t * dx, ay + t * dy)) < TOL
    if ent.type == "circle":
        return abs(math.dist(p, geo["center"]) - geo["radius"]) < TOL
    if ent.type == "arc":
        c = geo["center"]
        if abs(math.dist(p, c) - geo["radius"]) >= TOL:
            return False
        ang = lambda q: math.atan2(q[1] - c[1], q[0] - c[0]) % (2 * math.pi)
        a0, a1, a = ang(geo["start"]), ang(geo["end"]), ang(p)
        if ent.api.reversed().value():
            a0, a1 = a1, a0
        span = (a1 - a0) % (2 * math.pi)
        return (a - a0) % (2 * math.pi) <= span + 1e-9
    return math.dist(p, geo["at"]) < TOL


def loose_ends(sk: Sketch, geos: dict) -> list:
    """End points of drawn (non-construction) curves that touch no other drawn
    curve: an open contour."""
    drawn = [e for e in sk.entities.values() if not e.construction and e.type != "point"]
    out = []
    for e in drawn:
        for ref, p in _ends(e, geos[e.id]):
            others = [o for o in drawn if o is not e]
            touches = any(_on(p, o, geos[o.id], sk) for o in others)
            if not touches and e.type == "arc":
                # an arc closing on itself is not open
                touches = math.dist(geos[e.id]["start"], geos[e.id]["end"]) < TOL
            if not touches:
                out.append({"ref": ref, "at": p})
    return out


def _point_inside_frame(face, frame) -> list | None:
    """As _point_inside, on a sketch with its own frame: the point in sketch
    coordinates [u, v]."""
    import GeomAPI
    from GeomAlgoAPI import GeomAlgoAPI_PointBuilder as PB, GeomAlgoAPI_ShapeTools as T
    o, xd, yd = frame
    uv = lambda p: [sum((p[i] - o[i]) * d[i] for i in range(3)) for d in (xd, yd)]
    world = lambda u, v: [o[i] + u * xd[i] + v * yd[i] for i in range(3)]
    pts = [uv([q.x(), q.y(), q.z()]) for q in
           (GeomAPI.GeomAPI_Vertex(vx).point() for vx in face.subShapes(GeomAPI.GeomAPI_Shape.VERTEX))]
    edges = face.subShapes(GeomAPI.GeomAPI_Shape.EDGE)
    for e in edges:     # a circle has one vertex: widen by its box corners
        ok, *b = e.computeSize()
        if ok:
            pts += [uv([b[i], b[j], b[k]]) for i in (0, 3) for j in (1, 4) for k in (2, 5)]
    if not pts:
        return None
    lo = [min(p[i] for p in pts) for i in range(2)]
    hi = [max(p[i] for p in pts) for i in range(2)]
    for n in (8, 16):
        best, best_d = None, -1.0
        for a in range(1, n):
            for b in range(1, n):
                u = lo[0] + (hi[0] - lo[0]) * a / n
                v = lo[1] + (hi[1] - lo[1]) * b / n
                vert = PB.vertex(GeomAPI.GeomAPI_Pnt(*world(u, v)))
                if not T.isSubShapeInsideShape(vert, face):
                    continue
                d = min(T.minimalDistance(vert, e) for e in edges)
                if d > best_d:
                    best, best_d = [u, v], d
        if best is not None:
            return [round(c, 4) + 0.0 for c in best]
    return None


def _point_inside(face) -> list | None:
    """A point well inside a planar region: of a grid over its box (7 x 7, then
    15 x 15 for a thin region), the inside point farthest from its edges."""
    import GeomAPI
    from GeomAlgoAPI import GeomAlgoAPI_PointBuilder as PB, GeomAlgoAPI_ShapeTools as T
    ok, *box = face.computeSize()
    if not ok:
        return None
    lo, hi = box[:3], box[3:]
    flat = [i for i in range(3) if hi[i] - lo[i] < 1e-5]
    if len(flat) != 1:     # sketches lie on XOY / XOZ / YOZ or planes parallel to them
        return None
    u, v = [i for i in range(3) if i not in flat]
    edges = face.subShapes(GeomAPI.GeomAPI_Shape.EDGE)
    for n in (8, 16):
        best, best_d = None, -1.0
        for a in range(1, n):
            for b in range(1, n):
                p = [(lo[i] + hi[i]) / 2 for i in range(3)]
                p[u] = lo[u] + (hi[u] - lo[u]) * a / n
                p[v] = lo[v] + (hi[v] - lo[v]) * b / n
                vert = PB.vertex(GeomAPI.GeomAPI_Pnt(*p))
                if not T.isSubShapeInsideShape(vert, face):
                    continue
                d = min(T.minimalDistance(vert, e) for e in edges)
                if d > best_d:
                    best, best_d = p, d
        if best is not None:
            return [round(c, 4) + 0.0 for c in best]
    return None


def regions(executor, sk: Sketch) -> list:
    """Closed regions of the sketch: index, area, a point inside."""
    import ModelAPI
    from GeomAlgoAPI import GeomAlgoAPI_ShapeTools
    res = sk.api.feature().firstResult()
    if res is None:
        return []
    rc = ModelAPI.modelAPI_ResultConstruction(res)
    out = []
    for i in range(rc.facesNum()):
        face = rc.face(i)
        entry = {"index": i, "area": round(GeomAlgoAPI_ShapeTools.area(face), 4)}
        try:
            inside = (_point_inside_frame(face, sk.frame) if sk.frame
                      else _point_inside(face))
        except Exception:  # noqa: BLE001 — the area alone still identifies the region
            inside = None
        if inside is not None:
            entry["inside"] = inside
        out.append(entry)
    return out


def status(executor, sk: Sketch, before: dict | None = None) -> dict:
    geos = snapshot(sk)
    out = {"sketch": sk.name, "dof": dof(sk),
           "loose_ends": loose_ends(sk, geos), "regions": regions(executor, sk)}
    if before is not None:
        out["moved"] = moved(before, geos)
    return out


def history_index(st: ShaperState) -> dict:
    """Feature name -> its place in the document's history."""
    return {f.data().name(): i for i, f in enumerate(st.ns["shaper_doc"].allFeatures())}


def sketch_of(executor, name) -> Sketch:
    st = state(executor)
    sk = st.sketches.get(name)
    if sk is None:
        known = ", ".join(st.sketches) or "none yet"
        raise ValueError(f"no sketch named {name!r} — sketches: {known}")
    return sk


# --- shapes -----------------------------------------------------------------
# Measured in SHAPER itself: no GEOM copy (a GEOM copy per call was never
# released, ~28 KB each).

def results_of(executor, name: str) -> list:
    """The published feature's result objects that hold a shape."""
    feat = state(executor).ns[name].feature()
    return [r for r in feat.results() if r.shape() is not None]


def shape_type(executor, name: str) -> str:
    feat = state(executor).ns[name].feature()
    if feat.getKind() == "Group":
        return feat.selectionList("group_list").selectionType().lower()
    shapes = [r.shape() for r in results_of(executor, name)]
    return shapes[0].shapeTypeStr().lower() if len(shapes) == 1 else "compound"


def consumed(executor, name: str) -> bool:
    """True once a later feature used the result up (a boolean's operand): it is
    no longer in the study."""
    return all(r.isConcealed() for r in results_of(executor, name))


def _real(s) -> bool:
    """False for a degenerated edge: the zero-length edge OCC puts where a
    revolved profile touches its axis. SHAPER cannot name or select it."""
    import GeomAPI
    return not (s.shapeType() == GeomAPI.GeomAPI_Shape.EDGE and s.edge().isDegenerated())


def _unique(shape, kind) -> int:
    """Sub-shapes of `kind`, each once: SHAPER lists a seam edge twice."""
    seen = []
    for s in shape.subShapes(kind):
        if _real(s) and not any(s.isSame(t) for t in seen):
            seen.append(s)
    return len(seen)


FLAT = 1e-4


def _precise_box(executor, results) -> list | None:
    """[xmin, ymin, zmin, xmax, ymax, zmax] of all results, exact on curved faces.
    computeSize() encloses a curve's control points (a torus of outer radius 60
    reads 64.94); SHAPER's bounding-box feature computes it exactly. It is made
    in an operation that is then aborted, so it leaves no feature and no undo
    entry. The feature also builds a box between the two corners, which fails
    with an "Application errors" popup on a flat shape, so a result thinner
    than the measure's rounding keeps its fast box (a flat result with a shape
    tolerance above 5e-5 still reaches the feature)."""
    import ModelAPI
    m = model()
    session = ModelAPI.ModelAPI_Session.get()
    if session.isOperation():
        return None
    doc = state(executor).ns["shaper_doc"]
    box = None
    session.startOperation("salome-agent measure")
    try:
        for r in results:
            ok, *b = r.shape().computeSize()
            if not ok:
                return None
            if min(b[i + 3] - b[i] for i in range(3)) >= FLAT:
                feat = m.getBoundingBox(doc, m.selection(r.shape().shapeTypeStr(), r.data().name()))
                res = feat.feature().firstResult()
                if res is None or res.shape() is None:
                    return None
                ok, *b = res.shape().computeSize()
                if not ok:
                    return None
            box = b if box is None else [min(box[i], b[i]) for i in range(3)] + \
                [max(box[i], b[i]) for i in range(3, 6)]
    finally:
        session.abortOperation()
    return box


def raw_box(executor, name: str) -> list:
    """[xmin, ymin, zmin, xmax, ymax, zmax], unrounded."""
    results = results_of(executor, name)
    if not results:
        raise ValueError(f"{name} has no result")
    box = _precise_box(executor, results)
    if box is None:     # fast box: only while another SHAPER operation is open
        sizes = [r.shape().computeSize()[1:] for r in results]
        box = [min(s[i] for s in sizes) for i in range(3)] + \
            [max(s[i] for s in sizes) for i in range(3, 6)]
    return box


def measure(executor, name: str) -> dict:
    from GeomAlgoAPI import GeomAlgoAPI_ShapeTools as T
    import GeomAPI
    results = results_of(executor, name)
    xmin, ymin, zmin, xmax, ymax, zmax = raw_box(executor, name)
    r4 = lambda v: round(v, 4) + 0.0
    shapes = [r.shape() for r in results]
    info = {
        "bbox": [r4(xmin), r4(xmax), r4(ymin), r4(ymax), r4(zmin), r4(zmax)],
        "size": [r4(xmax - xmin), r4(ymax - ymin), r4(zmax - zmin)],
        "centre": [r4((xmin + xmax) / 2), r4((ymin + ymax) / 2), r4((zmin + zmax) / 2)],
        "area": round(sum(T.area(s) for s in shapes), 4),
        "volume": round(sum(T.volume(s) for s in shapes), 4),
        "faces": sum(_unique(s, GeomAPI.GeomAPI_Shape.FACE) for s in shapes),
        "edges": sum(_unique(s, GeomAPI.GeomAPI_Shape.EDGE) for s in shapes),
        "valid": all(T.isShapeValid(s) for s in shapes),
    }
    if not info["valid"]:
        info["check"] = "SHAPER's shape check fails on this result"
    return info


CONSTRUCTION = frozenset({"point", "axis", "plane"})


def offset_plane(base: str, offset) -> str:
    """addPlane arguments for the plane parallel to `base` at `offset` along +X,
    +Y or +Z: SHAPER's XOZ normal points to -Y, so its offset is reversed."""
    return (f'model.selection("FACE", "PartSet/{base}"), {offset!r}, '
            f'{base == "XOZ"}')


def construction_info(executor, name: str) -> dict:
    """Where a construction point, axis or plane is: they have no size to measure."""
    import GeomAPI
    kind = state(executor).objects[name]
    shape = results_of(executor, name)[0].shape()
    xyz = lambda p: [round(p.x(), 4) + 0.0, round(p.y(), 4) + 0.0, round(p.z(), 4) + 0.0]
    if kind == "point":
        return {"kind": kind, "at": xyz(GeomAPI.GeomAPI_Vertex(shape).point())}
    if kind == "axis":
        edge = GeomAPI.GeomAPI_Edge(shape)
        a, b = xyz(edge.firstPoint()), xyz(edge.lastPoint())
        n = math.dist(a, b) or 1.0
        return {"kind": kind, "through": a,
                "direction": [round((q - p) / n, 6) + 0.0 for p, q in zip(a, b)]}
    pln = GeomAPI.GeomAPI_Face(shape).getPlane()
    return {"kind": kind, "point": xyz(pln.location()), "normal": xyz(pln.direction())}


def check_results(executor, names: list) -> dict:
    """finish_stage's re-check: a feature whose inputs changed after it was
    made (a sketch edited under a face) no longer builds."""
    st = state(executor)
    out = {}
    for name in names:
        if name in st.sketches or st.objects.get(name) in CONSTRUCTION:
            out[name] = {"status": "skip"}
        elif name not in st.objects or name not in st.ns:
            out[name] = {"status": "missing"}
        else:
            feat = st.ns[name].feature()
            err = feat.error()
            if err or not any(r.shape() is not None for r in feat.results()):
                out[name] = {"status": "broken", "detail": err or "it has no result"}
            else:
                out[name] = shape_status(measure(executor, name))
    return out


def shape_status(info: dict) -> dict:
    """As GEOM's results.shape_status, from a SHAPER measure."""
    if not info["valid"]:
        return {"status": "invalid", "detail": info.get("check") or "shape check failed"}
    if info["faces"] == 0 and info["edges"] == 0:
        return {"status": "empty", "detail": "it holds no geometry"}
    if info["faces"] > 0 and info["area"] <= 1e-9:
        return {"status": "empty", "detail": "its faces have zero area"}
    return {"status": "ok"}


# --- sub-shapes -------------------------------------------------------------
# The model names an edge or face by a short id ("E3", "F2"), numbered by
# centre (x, then y, then z) so the same shape always gives the same ids;
# result.py gets SHAPER's own naming string, which survives a rebuild.

SUB_TYPES = {"vertex": "V", "edge": "E", "face": "F", "solid": "S"}


class Sub:
    def __init__(self, id_, shape, naming, centre, size):
        self.id, self.shape, self.naming = id_, shape, naming
        self.centre, self.size = centre, size


def _naming_names(executor, result, shapes: list) -> list:
    """SHAPER's naming string for each sub-shape of `result`, read from a
    throwaway Group that selects them by object."""
    doc = state(executor).ns["shaper_doc"]
    with transaction() as m:
        probe = m.addGroup(doc, [m.selection(result, s) for s in shapes])
    try:
        lst = probe.feature().selectionList("group_list")
        if lst.size() != len(shapes):
            raise RuntimeError(f"SHAPER named {lst.size()} of {len(shapes)} sub-shapes")
        names = []
        for k, s in enumerate(shapes):
            got = lst.value(k)
            if got.value() is None or not got.value().isSame(s):
                raise RuntimeError("SHAPER named a different sub-shape than the one asked for")
            names.append(got.namingName())
        return names
    finally:
        with transaction():
            doc.removeFeature(probe.feature())


def subshapes(executor, name: str, sub_type: str) -> list:
    """Every sub-shape of `sub_type` of the published feature `name`, as Subs
    with their ids."""
    import GeomAPI
    from GeomAlgoAPI import GeomAlgoAPI_ShapeTools as T
    if not isinstance(sub_type, str) or sub_type not in SUB_TYPES:
        raise ValueError(f"sub_type must be one of {', '.join(SUB_TYPES)}, got {sub_type!r}")
    kind = getattr(GeomAPI.GeomAPI_Shape, sub_type.upper())
    names, seen = [], []
    for res in results_of(executor, name):
        uniq = []
        for s in res.shape().subShapes(kind):
            if _real(s) and not any(s.isSame(t) for t in seen):   # shared between results too
                uniq.append(s)
                seen.append(s)
        if uniq:
            names += list(zip(uniq, _naming_names(executor, res, uniq)))
    out = []
    for s, naming in names:
        ok, *b = s.computeSize()
        centre = [round((b[i] + b[i + 3]) / 2, 4) + 0.0 for i in range(3)]
        size = {"vertex": 0.0, "edge": T.length(s), "face": T.area(s),
                "solid": T.volume(s)}[sub_type]
        out.append(Sub(None, s, naming, centre, round(size, 4) + 0.0))
    # ties (coaxial rims, faces with one box centre) by size, then by name
    out.sort(key=lambda u: (tuple(u.centre), u.size, u.naming))
    for i, u in enumerate(out, start=1):
        u.id = f"{SUB_TYPES[sub_type]}{i}"
    return out
