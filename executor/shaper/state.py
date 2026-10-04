"""The run's SHAPER document, the sketches built in it, and what every tool
shares: running SHAPER code, recording it, and reporting a sketch's state.

Every change is made by executing the same source line that goes into
result.py, so the live document and the generated script cannot drift."""
import math
import re

from executor.registry import ToolRegistry
from executor.geom.helpers import _next_name

registry = ToolRegistry(valid_categories={"sketch", "feature", "query"})
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
'''

PREAMBLE = ('from salome.shaper import model\n'
            'from SketchAPI import SketchAPI_Point\n'
            'model.begin()\n'
            'shaper_doc = model.addPart(model.moduleDocument()).document()\n'
            'model.end()\n' + NAME_HELPER + 'model.begin()\n')
POSTAMBLE = 'model.end()\nmodel.publishToShaperStudy()\n'
PUBLISH_LINE = "_name({var}, {name!r})"

_ID = re.compile(r"[A-Za-z][A-Za-z0-9]*$")
TOL = 1e-5


class Entity:
    def __init__(self, id_, type_, var, api, construction):
        self.id, self.type, self.var, self.api = id_, type_, var, api
        self.construction = construction


class Constraint:
    def __init__(self, id_, type_, var, api, refs, value=None):
        self.id, self.type, self.var, self.api, self.refs = id_, type_, var, api, refs
        self.value = value


class Sketch:
    def __init__(self, name, api, axes):
        self.name, self.api, self.axes = name, api, axes
        self.entities: dict[str, Entity] = {}
        self.constraints: dict[str, Constraint] = {}
        self.next_k = 0


class ShaperState:
    def __init__(self, ns):
        self.ns = ns
        self.sketches: dict[str, Sketch] = {}
        # published feature name -> kind ("face", "solid", ...)
        self.objects: dict[str, str] = {}


def model():
    from salome.shaper import model as m
    return m


def state(executor) -> ShaperState:
    st = executor.__dict__.get("shaper_state")
    if st is None:
        from SketchAPI import SketchAPI_Point
        m = model()
        ns = {"model": m, "SketchAPI_Point": SketchAPI_Point}
        m.begin()
        ns["shaper_doc"] = m.addPart(m.moduleDocument()).document()
        m.end()
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
    try:
        m = model()
    except ImportError:
        return
    part_set = m.moduleDocument()
    m.begin()
    for feat in list(part_set.allFeatures()):
        if feat.getKind() == "Part":
            part_set.removeFeature(feat)
    m.end()


def forget(executor, names: list) -> list:
    """Remove published shaper objects (sketches, features) and their study
    entries; names this module did not build are ignored."""
    st = executor.__dict__.get("shaper_state")
    if st is None:
        return []
    gone = []
    m = model()
    m.begin()
    for name in names:
        if name in st.sketches:
            feat = st.sketches.pop(name).api.feature()
        elif name in st.objects:
            st.objects.pop(name)
            feat = st.ns[name].feature()
        else:
            continue
        st.ns["shaper_doc"].removeFeature(feat)
        st.ns.pop(name, None)
        executor.geom_bindings.pop(name, None)
        gone.append(name)
    m.end()
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
    if not isinstance(id_, str) or not _ID.match(id_):
        raise ValueError(f"{what} id {id_!r} must start with a letter and use only "
                         "letters and digits (e.g. L1, C2, K3)")


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
    return {"at": _pt(a.coordinates())}


def snapshot(sk: Sketch) -> dict:
    return {i: geometry(e) for i, e in sk.entities.items()}


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
    if ent.type in ("line", "arc"):
        return [(f"{ent.id}.start", geo["start"]), (f"{ent.id}.end", geo["end"])]
    return []


def _on(p, ent: Entity, geo: dict) -> bool:
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
            touches = any(_on(p, o, geos[o.id]) for o in others)
            if not touches and e.type == "arc":
                # an arc closing on itself is not open
                touches = math.dist(geos[e.id]["start"], geos[e.id]["end"]) < TOL
            if not touches:
                out.append({"ref": ref, "at": p})
    return out


def regions(executor, sk: Sketch) -> list:
    """Closed regions of the sketch: index, area, a point inside."""
    import ModelAPI
    from GeomAlgoAPI import GeomAlgoAPI_ShapeTools
    res = sk.api.feature().firstResult()
    if res is None:
        return []
    rc = ModelAPI.modelAPI_ResultConstruction(res)
    out = []
    geompy = executor.namespace["geompy"]
    for i in range(rc.facesNum()):
        face = rc.face(i)
        entry = {"index": i, "area": round(GeomAlgoAPI_ShapeTools.area(face), 4)}
        try:
            v = geompy.MakeVertexInsideFace(to_geom(geompy, face))
            entry["inside"] = [round(c, 4) for c in geompy.PointCoordinates(v)]
        except Exception:
            pass
        out.append(entry)
    return out


def status(executor, sk: Sketch, before: dict | None = None) -> dict:
    geos = snapshot(sk)
    out = {"sketch": sk.name, "dof": dof(sk),
           "loose_ends": loose_ends(sk, geos), "regions": regions(executor, sk)}
    if before is not None:
        out["moved"] = moved(before, geos)
    return out


def sketch_of(executor, name) -> Sketch:
    st = state(executor)
    sk = st.sketches.get(name)
    if sk is None:
        known = ", ".join(st.sketches) or "none yet"
        raise ValueError(f"no sketch named {name!r} — sketches: {known}")
    return sk


# --- shapes -----------------------------------------------------------------

def to_geom(geompy, shape):
    """A SHAPER shape as a GEOM copy (BREP stream). One-way: later sketch edits
    do not reach it."""
    return geompy.RestoreShape(shape.getShapeStream(False).encode())


def feature_shape(executor, name: str):
    """The GEOM copy of a published feature's results (one compound if several)."""
    st = state(executor)
    geompy = executor.namespace["geompy"]
    feat = st.ns[name].feature()
    shapes = [to_geom(geompy, r.shape()) for r in feat.results() if r.shape() is not None]
    if not shapes:
        raise ValueError(f"{name} has no result: {feat.error() or 'empty'}")
    return shapes[0] if len(shapes) == 1 else geompy.MakeCompound(shapes)


def measure(executor, name: str) -> dict:
    from executor.geom.helpers import _check_shape
    from executor.geom.results import _shape_info
    geompy = executor.namespace["geompy"]
    shape = feature_shape(executor, name)
    info = _shape_info(geompy, shape) or {}
    length, area, volume = geompy.BasicProperties(shape)
    valid, msg = _check_shape(geompy, shape)
    info.update({"area": round(area, 4), "volume": round(volume, 4),
                 "faces": geompy.NumberOfFaces(shape), "edges": geompy.NumberOfEdges(shape),
                 "valid": bool(valid)})
    if not valid:
        info["check"] = msg
    return info
