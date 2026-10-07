from executor.shaper.features import _base_selections, _build, _operands
from executor.shaper.primitives import _XYZ, _point
from executor.shaper.state import new_name, tool

_POINTS = {"type": "array", "items": _XYZ, "minItems": 2,
           "description": "points: [x, y, z] or make_point names"}


def _points(executor, points, name: str, least: int, closed: bool = False) -> tuple[list, str, list, tuple]:
    if not isinstance(points, list) or len(points) < least:
        raise ValueError(f"points must be a list of at least {least} points")
    if any(p == q for p, q in zip(points, points[1:])) or \
            (closed and len(points) > 1 and points[0] == points[-1]):
        raise ValueError("two consecutive points are the same")
    sels, code, refs, helpers = [], "", [], ()
    for i, p in enumerate(points, start=1):
        s, c, r, h = _point(executor, p, f"{name}_p{i}", f"point {i}")
        sels.append(s)
        code += c
        refs += r
        helpers += h
    return sels, code, refs, helpers


def _flag(args: dict, key: str) -> bool:
    v = args.get(key, False)
    if not isinstance(v, bool):
        raise ValueError(f"{key} must be true or false, got {v!r}")
    return v


@tool("make_polyline", category="sketch",
      description=("A 3D wire of straight segments through `points` in order; closed: true "
                   "joins the last point back to the first. Use as a pipe path."),
      schema={"type": "object", "properties": {"points": _POINTS, "closed": {"type": "boolean"}},
              "required": ["points"]})
def tool_make_polyline(executor, args: dict):
    closed = _flag(args, "closed")
    name = new_name(executor, "polyline")
    sels, code, refs, helpers = _points(executor, args.get("points"), name, 3 if closed else 2, closed)
    code += f"{name} = model.addPolyline3D(shaper_doc, [{', '.join(sels)}], {closed})"
    return _build(executor, name, "addPolyline3D", code, refs, "wire", helpers=helpers)


@tool("make_curve", category="sketch",
      description=("A smooth 3D curve through every one of `points` in order; closed: true "
                   "makes a loop. Use as a pipe path or a filling boundary."),
      schema={"type": "object", "properties": {"points": _POINTS, "closed": {"type": "boolean"}},
              "required": ["points"]})
def tool_make_curve(executor, args: dict):
    closed = _flag(args, "closed")
    name = new_name(executor, "curve")
    sels, code, refs, helpers = _points(executor, args.get("points"), name, 3 if closed else 2, closed)
    code += f"{name} = model.addInterpolation(shaper_doc, [{', '.join(sels)}], {closed}, False)"
    return _build(executor, name, "addInterpolation", code, refs, "wire", helpers=helpers)


@tool("make_pipe", category="swept",
      description=("Sweep `base` (a face, a sketch or a wire) along `path` (a make_polyline / "
                   "make_curve result or another wire or edge). The base should sit at the "
                   "start of the path, across it."),
      schema={"type": "object", "properties": {
          "base": {"type": "string"}, "path": {"type": "string"}},
          "required": ["base", "path"]})
def tool_make_pipe(executor, args: dict):
    for k in ("base", "path"):
        if not isinstance(args.get(k), str):
            raise ValueError(f"{k} must be one name, got {args.get(k)!r}")
    if args["base"] == args["path"]:
        raise ValueError("base and path must be two different shapes")
    sels, refs = _base_selections(executor, args.get("base"))
    path, prefs = _operands(executor, [args.get("path")], "path")
    if len(path) != 1:
        raise ValueError(f"{args.get('path')!r} has {len(path)} results; a path is one wire")
    name = new_name(executor, "pipe")
    code = f"{name} = model.addPipe(shaper_doc, [{', '.join(sels)}], {path[0]})"
    return _build(executor, name, "addPipe", code, refs + prefs, "solid")


@tool("make_loft", category="swept",
      description=("A solid or surface through two sections, `first` and `second` (faces "
                   "or closed wires, e.g. a circle and a square on parallel planes)."),
      schema={"type": "object", "properties": {
          "first": {"type": "string"}, "second": {"type": "string"}},
          "required": ["first", "second"]})
def tool_make_loft(executor, args: dict):
    for k in ("first", "second"):
        if not isinstance(args.get(k), str):
            raise ValueError(f"{k} must be one name, got {args.get(k)!r}")
    if args.get("first") == args.get("second"):
        raise ValueError("first and second must be two different sections")
    a, ra = _base_selections(executor, args.get("first"))
    b, rb = _base_selections(executor, args.get("second"))
    if len(a) != 1 or len(b) != 1:
        raise ValueError("each section must be one face or wire")
    name = new_name(executor, "loft")
    code = f"{name} = model.addLoft(shaper_doc, {a[0]}, {b[0]})"
    return _build(executor, name, "addLoft", code, ra + rb, "solid")


@tool("make_filling", category="swept",
      description=("A surface spanning boundary curves: `edges` lists two or more wires or "
                   "edges (make_curve, make_polyline, ...), in order."),
      schema={"type": "object", "properties": {
          "edges": {"type": "array", "items": {"type": "string"}}},
          "required": ["edges"]})
def tool_make_filling(executor, args: dict):
    edges = args.get("edges")
    if not isinstance(edges, list) or len(edges) < 2 or not all(isinstance(x, str) for x in edges) \
            or len(set(edges)) != len(edges):
        raise ValueError("edges must list two or more different names")
    sels, refs = _operands(executor, edges, "edges")
    name = new_name(executor, "filling")
    code = f"{name} = model.addFilling(shaper_doc, [{', '.join(sels)}])"
    return _build(executor, name, "addFilling", code, refs, "face")
