import math

from executor.shaper.features import _build
from executor.shaper.state import (
    PUBLISH_LINE, construction_info, new_name, offset_plane, state, tool,
)

_AXES = ("OX", "OY", "OZ")
_PLANES = ("XOY", "XOZ", "YOZ")

_XYZ = {"description": "[x, y, z] or the name of a make_point result",
        "anyOf": [{"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3},
                  {"type": "string"}]}
_AXIS = {"type": "string", "description": "OX, OY, OZ or the name of a make_axis result"}


def _num(value, what: str, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{what} must be a number, got {value!r}")
    if positive and value <= 0:
        raise ValueError(f"{what} must be greater than 0, got {value!r}")
    return value


def _coords(value, what: str) -> list | None:
    """[x, y, z] from a list or an "x,y,z" string; None if `value` is neither."""
    if isinstance(value, str) and value.count(",") == 2:
        try:
            value = [float(v) for v in value.split(",")]
        except ValueError:
            return None
    if isinstance(value, (list, tuple)):
        if len(value) != 3:
            raise ValueError(f"{what} must be [x, y, z], got {value!r}")
        return [_num(v, what) for v in value]
    return None


def _point(executor, value, var: str, what: str) -> tuple[str, str, list, tuple]:
    """A point argument as (selection, code making it, refs, helpers): a
    make_point name, or coordinates turned into a helper point `var`."""
    st = state(executor)
    xyz = _coords(value, what)
    if xyz is None and not isinstance(value, str):
        raise ValueError(f"{what} must be [x, y, z] or a make_point name, got {value!r}")
    if xyz is not None:
        code = (f"{var} = model.addPoint(shaper_doc, {', '.join(repr(v) for v in xyz)})\n"
                + PUBLISH_LINE.format(var=var, name=var) + "\n")
        return f'model.selection("VERTEX", {var!r})', code, [], (var,)
    if st.objects.get(value) == "point":
        return f'model.selection("VERTEX", {value!r})', "", [value], ()
    if value in st.objects or value in st.sketches:
        raise ValueError(f"{what}: {value!r} is not a point — give [x, y, z] or a "
                         "make_point name")
    raise ValueError(f"{what} must be [x, y, z] or a make_point name, got {value!r}")


def _xyz_of(executor, value) -> list | None:
    """Coordinates of a point argument, given or of a make_point result."""
    xyz = _coords(value, "point")
    if xyz is None and isinstance(value, str) and state(executor).objects.get(value) == "point":
        xyz = construction_info(executor, value)["at"]
    return xyz


def _distinct(executor, a, b, need_all_axes: bool) -> None:
    pa, pb = _xyz_of(executor, a), _xyz_of(executor, b)
    if pa is None or pb is None:
        return
    if need_all_axes and any(abs(p - q) < 1e-7 for p, q in zip(pa, pb)):
        raise ValueError(f"the corners {pa} and {pb} must differ in x, y and z")
    if math.dist(pa, pb) < 1e-7:
        raise ValueError(f"the two points are the same: {pa}")


def _one_form(args: dict, *forms: tuple) -> None:
    given = [f for f in forms if any(args.get(k) is not None for k in f)]
    if len(given) > 1:
        raise ValueError("give either " + " or ".join(", ".join(f) for f in forms)
                         + ", not both")


def _axis(executor, value, what: str = "axis") -> tuple[str, list]:
    if not isinstance(value, str):
        raise ValueError(f"{what} must be OX, OY, OZ or a make_axis name, got {value!r}")
    if value in _AXES:
        return f'model.selection("EDGE", "PartSet/{value}")', []
    st = state(executor)
    if st.objects.get(value) == "axis":
        return f'model.selection("EDGE", {value!r})', [value]
    if value in st.objects or value in st.sketches:
        raise ValueError(f"{what}: {value!r} is not an axis — give OX, OY, OZ or a "
                         "make_axis name")
    raise ValueError(f"{what} must be OX, OY, OZ or a make_axis name, got {value!r}")


def _placed(executor, args: dict, name: str, default_axis: str = "OZ"):
    """Origin point and axis of a primitive: (selections, code, refs, helpers)."""
    origin = args.get("origin")
    if origin is None:
        point, code, refs, helpers = 'model.selection("VERTEX", "PartSet/Origin")', "", [], ()
    else:
        point, code, refs, helpers = _point(executor, origin, f"{name}_origin", "origin")
    axis, arefs = _axis(executor, args.get("axis") or default_axis)
    return point, axis, code, refs + arefs, helpers


def _primitive(executor, prefix: str, fn: str, args: dict, sizes: list) -> dict:
    name = new_name(executor, prefix)
    point, axis, code, refs, helpers = _placed(executor, args, name)
    code += (f"{name} = model.{fn}(shaper_doc, {point}, {axis}, "
             f"{', '.join(repr(s) for s in sizes)})")
    return _build(executor, name, fn, code, refs, "solid", helpers=helpers)


_PLACE_PROPS = {"origin": _XYZ, "axis": _AXIS}
_PLACE_TEXT = (" `origin` (default the world origin) and `axis` (default OZ) place it.")


@tool("make_box", category="primitive",
      description=("Axis-aligned solid box with one corner at the origin, extending "
                   "+X/+Y/+Z by dx/dy/dz. For a box elsewhere use make_box_two_pnt."),
      schema={"type": "object", "properties": {
          "dx": {"type": "number"}, "dy": {"type": "number"}, "dz": {"type": "number"}},
          "required": ["dx", "dy", "dz"]})
def tool_make_box(executor, args: dict):
    dims = [_num(args.get(k), k, positive=True) for k in ("dx", "dy", "dz")]
    name = new_name(executor, "box")
    code = f"{name} = model.addBox(shaper_doc, {', '.join(repr(d) for d in dims)})"
    return _build(executor, name, "addBox", code, [], "solid")


@tool("make_box_two_pnt", category="primitive",
      description="Axis-aligned solid box spanning two opposite corners pnt1 and pnt2.",
      schema={"type": "object", "properties": {"pnt1": _XYZ, "pnt2": _XYZ},
              "required": ["pnt1", "pnt2"]})
def tool_make_box_two_pnt(executor, args: dict):
    _distinct(executor, args.get("pnt1"), args.get("pnt2"), need_all_axes=True)
    name = new_name(executor, "box")
    s1, c1, r1, h1 = _point(executor, args.get("pnt1"), f"{name}_p1", "pnt1")
    s2, c2, r2, h2 = _point(executor, args.get("pnt2"), f"{name}_p2", "pnt2")
    code = c1 + c2 + f"{name} = model.addBox(shaper_doc, {s1}, {s2})"
    return _build(executor, name, "addBox", code, r1 + r2, "solid", helpers=h1 + h2)


@tool("make_cylinder", category="primitive",
      description="Solid cylinder: base centred at `origin`, height along `axis`." + _PLACE_TEXT,
      schema={"type": "object", "properties": {
          "radius": {"type": "number"}, "height": {"type": "number"}, **_PLACE_PROPS},
          "required": ["radius", "height"]})
def tool_make_cylinder(executor, args: dict):
    sizes = [_num(args.get("radius"), "radius", True), _num(args.get("height"), "height", True)]
    return _primitive(executor, "cylinder", "addCylinder", args, sizes)


@tool("make_sphere", category="primitive",
      description="Solid sphere centred at `center` (default the world origin).",
      schema={"type": "object", "properties": {
          "radius": {"type": "number"}, "center": _XYZ}, "required": ["radius"]})
def tool_make_sphere(executor, args: dict):
    radius = _num(args.get("radius"), "radius", True)
    name = new_name(executor, "sphere")
    if args.get("center") is None:
        point, code, refs, helpers = 'model.selection("VERTEX", "PartSet/Origin")', "", [], ()
    else:
        point, code, refs, helpers = _point(executor, args["center"], f"{name}_origin", "center")
    code += f"{name} = model.addSphere(shaper_doc, {point}, {radius!r})"
    return _build(executor, name, "addSphere", code, refs, "solid", helpers=helpers)


@tool("make_cone", category="primitive",
      description=("Solid (truncated) cone: radius1 at `origin`, radius2 at `height` "
                   "along `axis`; radius2 = 0 for a sharp tip." + _PLACE_TEXT),
      schema={"type": "object", "properties": {
          "radius1": {"type": "number"}, "radius2": {"type": "number"},
          "height": {"type": "number"}, **_PLACE_PROPS},
          "required": ["radius1", "radius2", "height"]})
def tool_make_cone(executor, args: dict):
    r1, r2 = _num(args.get("radius1"), "radius1"), _num(args.get("radius2"), "radius2")
    if r1 < 0 or r2 < 0 or r1 == r2 == 0:
        raise ValueError(f"radius1 and radius2 must be >= 0 and not both 0, got {r1}, {r2}")
    sizes = [r1, r2, _num(args.get("height"), "height", True)]
    return _primitive(executor, "cone", "addCone", args, sizes)


@tool("make_torus", category="primitive",
      description=("Solid torus centred at `origin` around `axis`: radius_major is the "
                   "ring's centre line, radius_minor the tube." + _PLACE_TEXT),
      schema={"type": "object", "properties": {
          "radius_major": {"type": "number"}, "radius_minor": {"type": "number"},
          **_PLACE_PROPS},
          "required": ["radius_major", "radius_minor"]})
def tool_make_torus(executor, args: dict):
    big = _num(args.get("radius_major"), "radius_major", True)
    small = _num(args.get("radius_minor"), "radius_minor", True)
    if small >= big:
        raise ValueError(f"radius_minor ({small}) must be smaller than radius_major ({big})")
    return _primitive(executor, "torus", "addTorus", args, [big, small])


# --- construction -----------------------------------------------------------

@tool("make_point", category="sketch",
      description="A construction point at (x, y, z), usable as a corner, origin or centre.",
      schema={"type": "object", "properties": {
          "x": {"type": "number"}, "y": {"type": "number"}, "z": {"type": "number"}},
          "required": ["x", "y", "z"]})
def tool_make_point(executor, args: dict):
    xyz = [_num(args.get(k), k) for k in ("x", "y", "z")]
    name = new_name(executor, "point")
    code = f"{name} = model.addPoint(shaper_doc, {', '.join(repr(v) for v in xyz)})"
    return _build(executor, name, "addPoint", code, [], "point")


@tool("make_axis", category="sketch",
      description=("A construction axis (a direction in space), usable as the axis of a "
                   "primitive: through pnt1 and pnt2, or through `point` (default the "
                   "origin) along `direction` [dx, dy, dz]."),
      schema={"type": "object", "properties": {
          "pnt1": _XYZ, "pnt2": _XYZ, "point": _XYZ,
          "direction": {"type": "array", "items": {"type": "number"},
                        "minItems": 3, "maxItems": 3}}})
def tool_make_axis(executor, args: dict):
    _one_form(args, ("pnt1", "pnt2"), ("point", "direction"))
    name = new_name(executor, "axis")
    if args.get("pnt1") is not None or args.get("pnt2") is not None:
        _distinct(executor, args.get("pnt1"), args.get("pnt2"), need_all_axes=False)
        s1, c1, r1, h1 = _point(executor, args.get("pnt1"), f"{name}_p1", "pnt1")
        s2, c2, r2, h2 = _point(executor, args.get("pnt2"), f"{name}_p2", "pnt2")
        code = c1 + c2 + f"{name} = model.addAxis(shaper_doc, {s1}, {s2})"
        return _build(executor, name, "addAxis", code, r1 + r2, "axis", helpers=h1 + h2)
    d = _coords(args.get("direction"), "direction")
    if d is None or not any(d):
        raise ValueError("give pnt1 and pnt2, or a non-zero direction [dx, dy, dz]")
    dirs = ", ".join(repr(v) for v in d)
    if args.get("point") is None:
        code = f"{name} = model.addAxis(shaper_doc, {dirs})"
        return _build(executor, name, "addAxis", code, [], "axis")
    s, c, r, h = _point(executor, args["point"], f"{name}_origin", "point")
    code = c + f"{name} = model.addAxis(shaper_doc, {s}, {dirs})"
    return _build(executor, name, "addAxis", code, r, "axis", helpers=h)


def plane_coefficients(point: list, normal: list) -> list:
    """a, b, c, d of the plane a*x + b*y + c*z + d = 0 through `point`."""
    if not any(normal):
        raise ValueError("normal must not be [0, 0, 0]")
    d = -sum(p * n for p, n in zip(point, normal))
    return [*normal, d + 0.0]


@tool("make_plane", category="sketch",
      description=("A construction plane: through `point` [x, y, z] with `normal` "
                   "[nx, ny, nz], or parallel to a base plane (XOY, XOZ, YOZ) at `offset` "
                   "along +Z, +Y, +X."),
      schema={"type": "object", "properties": {
          "point": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3},
          "normal": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3},
          "base": {"type": "string", "enum": list(_PLANES)},
          "offset": {"type": "number"}}})
def tool_make_plane(executor, args: dict):
    _one_form(args, ("point", "normal"), ("base", "offset"))
    name = new_name(executor, "plane")
    if args.get("base") is not None:
        base = args["base"]
        if base not in _PLANES:
            raise ValueError(f"base must be one of {', '.join(_PLANES)}, got {base!r}")
        offset = _num(args.get("offset", 0), "offset")
        code = f"{name} = model.addPlane(shaper_doc, {offset_plane(base, offset)})"
        return _build(executor, name, "addPlane", code, [], "plane")
    point = _coords(args.get("point"), "point")
    normal = _coords(args.get("normal"), "normal")
    if point is None or normal is None:
        raise ValueError("give point and normal, or base and offset")
    coeffs = ", ".join(repr(v) for v in plane_coefficients(point, normal))
    code = f"{name} = model.addPlane(shaper_doc, {coeffs})"
    return _build(executor, name, "addPlane", code, [], "plane")

