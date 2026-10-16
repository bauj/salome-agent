from executor.shaper.features import _unknown
from executor.shaper.state import (
    CONSTRUCTION, measure, publish_study, record, results_of, run, shape_status, state, tool,
    transaction,
)


def _feature(executor, name) -> str:
    st = state(executor)
    if name in st.sketches:
        raise ValueError(f"{name!r} is a sketch — make a face or a solid from it first")
    if name not in st.objects:
        raise ValueError(_unknown(st, name))
    return name


def _shapes(executor, name, allow=()) -> list:
    """Result shapes of a feature; a construction axis or plane is drawn as a
    finite stand-in, so its length, area or distance would mean nothing."""
    name = _feature(executor, name)
    kind = state(executor).objects[name]
    if kind in CONSTRUCTION and kind not in allow:
        raise ValueError(f"{name!r} is a construction {kind} — it has no size; "
                         "measure gives where it is")
    return [r.shape() for r in results_of(executor, name)]


@tool("check_shape", category="query",
      description="Report whether `shape` is valid geometry, with a diagnostic message.",
      schema={"type": "object", "properties": {"shape": {"type": "string"}},
              "required": ["shape"]})
def tool_check_shape(executor, args: dict):
    name = _feature(executor, args.get("shape"))
    st = state(executor)
    err = st.ns[name].feature().error()
    if err:
        return {"ok": True, "records": [], "valid": False, "message": err}
    if st.objects[name] in CONSTRUCTION:
        return {"ok": True, "records": [], "valid": True,
                "message": f"a construction {st.objects[name]}"}
    status = shape_status(measure(executor, name))
    return {"ok": True, "records": [], "valid": status["status"] == "ok",
            "message": status.get("detail") or "valid"}


@tool("list_study_objects", category="query",
      description=("List the shaper objects built so far: sketches, and every feature "
                   "with its kind; `used_up` marks results a later operation consumed."),
      schema={"type": "object", "properties": {}})
def tool_list_study_objects(executor, args: dict):
    st = state(executor)
    objects = []
    for name, kind in st.objects.items():
        results = st.ns[name].feature().results()
        entry = {"name": name, "kind": kind}
        if results and all(r.isConcealed() for r in results):
            entry["used_up"] = True
        objects.append(entry)
    return {"ok": True, "records": [], "sketches": list(st.sketches), "objects": objects}


def _faces_or_self(shape) -> list:
    import GeomAPI
    faces = shape.subShapes(GeomAPI.GeomAPI_Shape.FACE)
    return list(faces) if faces else [shape]


def _inside(a, b) -> bool:
    """A vertex of `a` lies inside the solid `b`."""
    import GeomAPI
    from GeomAlgoAPI import GeomAlgoAPI_ShapeTools as T
    if not b.subShapes(GeomAPI.GeomAPI_Shape.SOLID):
        return False
    verts = a.subShapes(GeomAPI.GeomAPI_Shape.VERTEX) or [a]
    return T.isSubShapeInsideShape(verts[0], b)


def min_distance(shapes1: list, shapes2: list) -> float:
    """Shortest distance between two shape groups, measured face to face: SHAPER's
    minimalDistance on two overlapping solids crashes the process. Zero when one
    lies inside the other."""
    from GeomAlgoAPI import GeomAlgoAPI_ShapeTools as T
    best = min(T.minimalDistance(f, g)
               for a in shapes1 for f in _faces_or_self(a)
               for b in shapes2 for g in _faces_or_self(b))
    if best > 1e-7 and any(_inside(a, b) or _inside(b, a) for a in shapes1 for b in shapes2):
        return 0.0
    return best


@tool("min_distance", category="query",
      description="Report the shortest distance (mm) between two published shapes; 0 if they touch or overlap.",
      schema={"type": "object", "properties": {
          "shape1": {"type": "string"}, "shape2": {"type": "string"}},
          "required": ["shape1", "shape2"]})
def tool_min_distance(executor, args: dict):
    d = min_distance(_shapes(executor, args.get("shape1"), allow=("point",)),
                     _shapes(executor, args.get("shape2"), allow=("point",)))
    return {"ok": True, "records": [], "distance": round(d, 6) + 0.0}


@tool("basic_properties", category="query",
      description=("Report length, area and volume (mm) of `shape`: length is the total "
                   "edge length of a wire or edge (0 once it has faces), area the face "
                   "area, volume the solid volume."),
      schema={"type": "object", "properties": {"shape": {"type": "string"}},
              "required": ["shape"]})
def tool_basic_properties(executor, args: dict):
    import GeomAPI
    from GeomAlgoAPI import GeomAlgoAPI_ShapeTools as T
    shapes = _shapes(executor, args.get("shape"))
    length = area = volume = 0.0
    for s in shapes:
        if s.subShapes(GeomAPI.GeomAPI_Shape.FACE):
            area += T.area(s)
            volume += T.volume(s)
        else:
            length += sum(T.length(e) for e in s.subShapes(GeomAPI.GeomAPI_Shape.EDGE))
    r = lambda v: round(v, 4) + 0.0
    return {"ok": True, "records": [], "length": r(length), "area": r(area),
            "volume": r(volume)}


# --- display ----------------------------------------------------------------

def _display(executor, name, fn: str, call: str) -> dict:
    name = _feature(executor, name)
    st = state(executor)
    if st.objects[name] in CONSTRUCTION:
        raise ValueError(f"{name!r} is a construction {st.objects[name]}; colour a shape")
    code = f"for _r in {name}.results():\n    _r.{call}"
    rec = record(fn, code, refs=[name], recv=name)
    with transaction():
        run(st, code)
    publish_study(executor)
    return {"ok": True, "records": [rec], "result_shape": name}


@tool("set_color", category="display",
      description=("Set the colour of `shape` (red/green/blue 0-255). Apply it last, on "
                   "the final shape: a later operation makes a new shape."),
      schema={"type": "object", "properties": {
          "shape": {"type": "string"},
          "red": {"type": "integer", "minimum": 0, "maximum": 255},
          "green": {"type": "integer", "minimum": 0, "maximum": 255},
          "blue": {"type": "integer", "minimum": 0, "maximum": 255}},
          "required": ["shape", "red", "green", "blue"]})
def tool_set_color(executor, args: dict):
    rgb = [args.get(k) for k in ("red", "green", "blue")]
    if any(isinstance(v, bool) or not isinstance(v, int) or not 0 <= v <= 255 for v in rgb):
        raise ValueError(f"red, green and blue must be integers 0-255, got {rgb}")
    return _display(executor, args.get("shape"), "setColor",
                    f"setColor({rgb[0]}, {rgb[1]}, {rgb[2]})")


@tool("set_transparency", category="display",
      description="Set the transparency of `shape` (0 opaque .. 1 invisible).",
      schema={"type": "object", "properties": {
          "shape": {"type": "string"},
          "transparency": {"type": "number", "minimum": 0.0, "maximum": 1.0}},
          "required": ["shape", "transparency"]})
def tool_set_transparency(executor, args: dict):
    t = args.get("transparency")
    if isinstance(t, bool) or not isinstance(t, (int, float)) or not 0 <= t <= 1:
        raise ValueError(f"transparency must be a number from 0 to 1, got {t!r}")
    return _display(executor, args.get("shape"), "setTransparency",
                    f"setTransparency({float(t)!r})")
