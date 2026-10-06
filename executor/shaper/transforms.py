import math

from executor.shaper.features import _build, _operands
from executor.shaper.primitives import _AXIS, _XYZ, _axis, _num, _point
from executor.shaper.state import new_name, raw_box, state, tool

_MAX_COPIES = 500

_MOVED = (" `shape` is used up: refer to the result by its new name.")
_SHAPE = {"type": "string"}


def _source(executor, args: dict) -> tuple[list, list]:
    shape = args.get("shape")
    if not isinstance(shape, str):
        raise ValueError(f"shape must be one published name, got {shape!r}")
    return _operands(executor, [shape], "shape")


def _count(value, what: str, least: int = 2) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)) \
            or not math.isfinite(value) or value != int(value):
        raise ValueError(f"{what} must be a whole number, got {value!r}")
    if not least <= value <= _MAX_COPIES:
        raise ValueError(f"{what} must be from {least} to {_MAX_COPIES}, got {value!r}")
    return int(value)


def _unchanged(name: str, why: str) -> dict:
    return {"ok": True, "records": [], "name": name,
            "note": f"{why}: nothing was done, {name} keeps its name"}


def _translation(executor, name: str, sels: list, refs: list, d: list,
                 code: str = "", helpers: tuple = ()) -> dict:
    code += (f"{name} = model.addTranslation(shaper_doc, [{', '.join(sels)}], "
             f"{', '.join(repr(v) for v in d)})")
    return _build(executor, name, "addTranslation", code, refs, "transform", helpers=helpers)


@tool("translate", category="transform",
      description="Move `shape` by (dx, dy, dz)." + _MOVED,
      schema={"type": "object", "properties": {
          "shape": _SHAPE, "dx": {"type": "number"}, "dy": {"type": "number"},
          "dz": {"type": "number"}}, "required": ["shape", "dx", "dy", "dz"]})
def tool_translate(executor, args: dict):
    d = [_num(args.get(k), k) for k in ("dx", "dy", "dz")]
    sels, refs = _source(executor, args)
    if not any(d):
        return _unchanged(refs[0], "a move by (0, 0, 0)")
    return _translation(executor, new_name(executor, "translation"), sels, refs, d)


@tool("rotate", category="transform",
      description="Rotate `shape` by angle_deg about `axis`." + _MOVED,
      schema={"type": "object", "properties": {
          "shape": _SHAPE, "axis": _AXIS, "angle_deg": {"type": "number"}},
          "required": ["shape", "axis", "angle_deg"]})
def tool_rotate(executor, args: dict):
    angle = _num(args.get("angle_deg"), "angle_deg")
    sels, refs = _source(executor, args)
    axis, arefs = _axis(executor, args.get("axis"))
    if angle % 360 == 0:
        return _unchanged(refs[0], f"a rotation by {angle:g} degrees")
    name = new_name(executor, "rotation")
    code = f"{name} = model.addRotation(shaper_doc, [{', '.join(sels)}], {axis}, {angle!r})"
    return _build(executor, name, "addRotation", code, refs + arefs, "transform")


_BASIS = {"XOY": ("FACE", "PartSet/XOY"), "XOZ": ("FACE", "PartSet/XOZ"),
          "ZOX": ("FACE", "PartSet/XOZ"), "YOZ": ("FACE", "PartSet/YOZ"),
          "OX": ("EDGE", "PartSet/OX"), "OY": ("EDGE", "PartSet/OY"),
          "OZ": ("EDGE", "PartSet/OZ"), "O": ("VERTEX", "PartSet/Origin")}
_BASIS_KIND = {"plane": "FACE", "axis": "EDGE", "point": "VERTEX"}


def _basis(executor, entity) -> tuple[str, list]:
    if not isinstance(entity, str):
        raise ValueError(f"entity must be a name, got {entity!r}")
    if entity in _BASIS:
        kind, path = _BASIS[entity]
        return f'model.selection("{kind}", "{path}")', []
    kind = state(executor).objects.get(entity)
    if kind in _BASIS_KIND:
        return f'model.selection("{_BASIS_KIND[kind]}", {entity!r})', [entity]
    raise ValueError(f"entity must be XOY, XOZ, YOZ, OX, OY, OZ, O or a make_plane / "
                     f"make_axis / make_point name, got {entity!r}")


@tool("mirror", category="transform",
      description=("Mirror `shape` across a plane, an axis or a point: `entity` is XOY, "
                   "XOZ, YOZ, OX, OY, OZ, O (the origin) or a make_plane / make_axis / "
                   "make_point name. keep_original: true gives one result holding both "
                   "the shape and its mirror image; otherwise the shape is moved."
                   + _MOVED),
      schema={"type": "object", "properties": {
          "shape": _SHAPE, "entity": {"type": "string"},
          "keep_original": {"type": "boolean"}}, "required": ["shape", "entity"]})
def tool_mirror(executor, args: dict):
    sels, refs = _source(executor, args)
    basis, brefs = _basis(executor, args.get("entity"))
    keep = args.get("keep_original", False)
    if not isinstance(keep, bool):
        raise ValueError(f"keep_original must be true or false, got {keep!r}")
    name = new_name(executor, "symmetry")
    code = (f"{name} = model.addSymmetry(shaper_doc, [{', '.join(sels)}], {basis}, {keep})")
    return _build(executor, name, "addSymmetry", code, refs + brefs, "transform")


@tool("scale", category="transform",
      description=("Scale `shape` about `center` (default the origin): a uniform `factor`, "
                   "or factor_x / factor_y / factor_z." + _MOVED),
      schema={"type": "object", "properties": {
          "shape": _SHAPE, "factor": {"type": "number"}, "factor_x": {"type": "number"},
          "factor_y": {"type": "number"}, "factor_z": {"type": "number"}, "center": _XYZ},
          "required": ["shape"]})
def tool_scale(executor, args: dict):
    per_axis = [args.get(k) for k in ("factor_x", "factor_y", "factor_z")]
    if args.get("factor") is not None:
        if any(v is not None for v in per_axis):
            raise ValueError("give either factor or factor_x/y/z, not both")
        factors = [_num(args["factor"], "factor", positive=True)]
    elif any(v is not None for v in per_axis):
        factors = [_num(1 if v is None else v, k, positive=True)
                   for v, k in zip(per_axis, ("factor_x", "factor_y", "factor_z"))]
    else:
        raise ValueError("give factor, or factor_x / factor_y / factor_z")
    sels, refs = _source(executor, args)
    name = new_name(executor, "scale")
    if args.get("center") is None:
        center, code, crefs, helpers = 'model.selection("VERTEX", "PartSet/Origin")', "", [], ()
    else:
        center, code, crefs, helpers = _point(executor, args["center"], f"{name}_center",
                                              "center")
    code += (f"{name} = model.addScale(shaper_doc, [{', '.join(sels)}], {center}, "
             f"{', '.join(repr(f) for f in factors)})")
    return _build(executor, name, "addScale", code, refs + crefs, "transform", helpers=helpers)


@tool("multi_translate", category="transform",
      description=("Repeat `shape` `count` times along `vector` (OX, OY, OZ or a make_axis "
                   "name) every `step` mm, the original being the first; add vector2 / "
                   "step2 / count2 for a grid. The result holds every copy." + _MOVED),
      schema={"type": "object", "properties": {
          "shape": _SHAPE, "vector": _AXIS, "step": {"type": "number"},
          "count": {"type": "integer"}, "vector2": _AXIS, "step2": {"type": "number"},
          "count2": {"type": "integer"}},
          "required": ["shape", "vector", "step", "count"]})
def tool_multi_translate(executor, args: dict):
    step, count = _num(args.get("step"), "step"), _count(args.get("count"), "count")
    if step == 0:
        raise ValueError("step must not be 0: every copy would sit on the first")
    sels, refs = _source(executor, args)
    axis, arefs = _axis(executor, args.get("vector"), "vector")
    rest = f"{axis}, {step!r}, {count}"
    if any(args.get(k) is not None for k in ("vector2", "step2", "count2")):
        step2 = _num(args.get("step2"), "step2")
        count2 = _count(args.get("count2"), "count2")
        if step2 == 0:
            raise ValueError("step2 must not be 0: every copy would sit on the first")
        if count * count2 > _MAX_COPIES:
            raise ValueError(f"count x count2 must be at most {_MAX_COPIES}")
        if args.get("vector2") == args.get("vector"):
            raise ValueError("vector2 must differ from vector to make a grid")
        axis2, arefs2 = _axis(executor, args.get("vector2"), "vector2")
        rest += f", {axis2}, {step2!r}, {count2}"
        arefs += arefs2
    name = new_name(executor, "multi_translation")
    code = f"{name} = model.addMultiTranslation(shaper_doc, [{', '.join(sels)}], {rest})"
    return _build(executor, name, "addMultiTranslation", code, refs + arefs, "transform")


@tool("multi_rotate", category="transform",
      description=("Repeat `shape` `count` times about `axis`, the original being the "
                   "first: evenly round the full circle, or every angle_deg degrees. The "
                   "result holds every copy." + _MOVED),
      schema={"type": "object", "properties": {
          "shape": _SHAPE, "axis": _AXIS, "count": {"type": "integer"},
          "angle_deg": {"type": "number"}}, "required": ["shape", "axis", "count"]})
def tool_multi_rotate(executor, args: dict):
    count = _count(args.get("count"), "count")
    sels, refs = _source(executor, args)
    axis, arefs = _axis(executor, args.get("axis"))
    rest = f"{axis}, {count}"
    if args.get("angle_deg") is not None:
        step = _num(args["angle_deg"], "angle_deg")
        if step % 360 == 0 or abs(step) * (count - 1) >= 360 - 1e-9:
            raise ValueError(f"{count} copies every {step:g} degrees overlap: keep "
                             "angle_deg x (count - 1) under 360, or leave angle_deg out "
                             "to spread them round the full circle")
        rest = f"{axis}, {step!r}, {count}"
    name = new_name(executor, "multi_rotation")
    code = f"{name} = model.addMultiRotation(shaper_doc, [{', '.join(sels)}], {rest})"
    return _build(executor, name, "addMultiRotation", code, refs + arefs, "transform")


@tool("offset_shape", category="transform",
      description=("Offset every face of `shape` outward (+) or inward (-) by `offset` mm; "
                   "outward corners come out rounded." + _MOVED),
      schema={"type": "object", "properties": {
          "shape": _SHAPE, "offset": {"type": "number"}}, "required": ["shape", "offset"]})
def tool_offset_shape(executor, args: dict):
    offset = _num(args.get("offset"), "offset")
    if offset == 0:
        raise ValueError("offset must not be 0")
    sels, refs = _source(executor, args)
    if len(sels) != 1:
        raise ValueError(f"{args['shape']!r} has {len(sels)} results; offset takes one shape")
    name = new_name(executor, "offset")
    code = f"{name} = model.addOffset(shaper_doc, {sels[0]}, {offset!r})"
    return _build(executor, name, "addOffset", code, refs, "transform")


# --- position ---------------------------------------------------------------

def _unit(v: list, what: str, tol: float = 1e-12) -> list:
    n = math.sqrt(sum(c * c for c in v))
    if n < tol:
        raise ValueError(f"{what} must not be [0, 0, 0]")
    return [c / n for c in v]


def frame_rotation(x_dir: list, y_dir: list) -> tuple[list, float]:
    """Axis (through the origin) and angle in degrees of the rotation taking the
    world X and Y axes to x_dir and y_dir. y_dir is made perpendicular to x_dir."""
    x = _unit(x_dir, "x_dir")
    y0 = _unit(y_dir, "y_dir")
    dot = sum(a * b for a, b in zip(x, y0))
    y = _unit([b - dot * a for a, b in zip(x, y0)], "y_dir (it is parallel to x_dir)",
              tol=1e-9)
    z = [x[1] * y[2] - x[2] * y[1], x[2] * y[0] - x[0] * y[2], x[0] * y[1] - x[1] * y[0]]
    R = [[x[i], y[i], z[i]] for i in range(3)]       # columns are the new axes
    anti = [R[2][1] - R[1][2], R[0][2] - R[2][0], R[1][0] - R[0][1]]
    sin2 = math.sqrt(sum(a * a for a in anti))        # 2 sin(angle)
    angle = math.atan2(sin2, R[0][0] + R[1][1] + R[2][2] - 1)
    if angle < 1e-9:
        return [0.0, 0.0, 1.0], 0.0
    if math.pi - angle > 1e-6:
        axis = anti
    else:   # a half turn: the axis is the column of R + I with the largest norm
        cols = [[R[r][c] + (r == c) for r in range(3)] for c in range(3)]
        axis = max(cols, key=lambda c: sum(v * v for v in c))
    return [round(a, 12) + 0.0 for a in _unit(axis, "axis")], math.degrees(angle)


@tool("position", category="transform",
      description=("Place `shape` so the world frame lands on a target frame: `origin` "
                   "(default [0, 0, 0]) with axes x_dir (default [1, 0, 0]) and y_dir "
                   "(default [0, 1, 0]). Use to set a part, built at the origin, into an "
                   "assembly." + _MOVED),
      schema={"type": "object", "properties": {
          "shape": _SHAPE,
          "origin": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3},
          "x_dir": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3},
          "y_dir": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3}},
          "required": ["shape"]})
def tool_position(executor, args: dict):
    origin = _vec3(args, "origin", [0, 0, 0])
    axis, angle = frame_rotation(_vec3(args, "x_dir", [1, 0, 0]),
                                 _vec3(args, "y_dir", [0, 1, 0]))
    sels, refs = _source(executor, args)
    if angle == 0 and not any(origin):
        return _unchanged(refs[0], "the target frame is the world frame")
    name = new_name(executor, "position")
    if angle == 0:
        return _translation(executor, name, sels, refs, origin)
    ax, rot = f"{name}_axis", f"{name}_rot"
    code = (f"{ax} = model.addAxis(shaper_doc, {', '.join(repr(a) for a in axis)})\n"
            f"_name({ax}, {ax!r})\n"
            f"{rot} = model.addRotation(shaper_doc, [{', '.join(sels)}], "
            f"model.selection(\"EDGE\", {ax!r}), {angle!r})\n"
            f"_name({rot}, {rot!r})\n")
    return _translation(executor, name, [f"*{rot}.results()"], refs, origin,
                        code=code, helpers=(ax, rot))


def _vec3(args: dict, key: str, default: list) -> list:
    v = args.get(key)
    if v is None:
        return default
    if not isinstance(v, (list, tuple)) or len(v) != 3:
        raise ValueError(f"{key} must be [x, y, z], got {v!r}")
    return [_num(c, key) for c in v]


# --- align ------------------------------------------------------------------

_AXIS_IDX = {"x": 0, "y": 1, "z": 2}


def _bbox(executor, name: str) -> list:
    """Unrounded [xmin, xmax, ymin, ymax, zmin, zmax]: a rounded box leaves a
    sliver gap or overlap where a flush placement is asked for."""
    b = raw_box(executor, name)
    return [b[0], b[3], b[1], b[4], b[2], b[5]]


def align_delta(shape_bbox: list, to_bbox: list, match: list) -> list:
    """The translation that brings each named extreme of shape_bbox onto the
    one of to_bbox (bbox = [xmin, xmax, ymin, ymax, zmin, zmax])."""
    ext = lambda b, i, at: {"min": b[2 * i], "max": b[2 * i + 1],
                            "center": (b[2 * i] + b[2 * i + 1]) / 2}[at]
    if not isinstance(match, list) or not match:
        raise ValueError("match must be a non-empty list of {axis, my, target, gap}")
    delta, seen = [0.0, 0.0, 0.0], set()
    for c in match:
        axis = c.get("axis") if isinstance(c, dict) else None
        if not isinstance(axis, str) or axis not in _AXIS_IDX:
            raise ValueError(f"match axis must be x, y or z, got {axis!r}")
        if axis in seen:
            raise ValueError(f"two match entries for axis {axis}")
        seen.add(axis)
        if not all(isinstance(c.get(k), str) and c[k] in ("min", "max", "center")
                   for k in ("my", "target")):
            raise ValueError("my and target must be min, max or center")
        i = _AXIS_IDX[axis]
        gap = 0 if c.get("gap") is None else _num(c["gap"], "gap")
        delta[i] = ext(to_bbox, i, c["target"]) + gap - ext(shape_bbox, i, c["my"])
        if abs(delta[i]) < 1e-9:
            delta[i] = 0.0
    return delta


@tool("align", category="transform",
      description=("Move `shape` so chosen faces or centres line up with `to`. `match` is "
                   "a list of {axis, my, target, gap}: along that axis, `shape`'s `my` "
                   "(min | max | center) is brought to `to`'s `target` plus `gap` mm. Axes "
                   "not listed stay put. Use to place a part against, on top of, flush "
                   "with or centred on another; the tool does the box arithmetic." + _MOVED),
      schema={"type": "object", "properties": {
          "shape": _SHAPE, "to": {"type": "string"},
          "match": {"type": "array", "items": {"type": "object", "properties": {
              "axis": {"type": "string", "enum": ["x", "y", "z"]},
              "my": {"type": "string", "enum": ["min", "max", "center"]},
              "target": {"type": "string", "enum": ["min", "max", "center"]},
              "gap": {"type": "number"}}, "required": ["axis", "my", "target"]}}},
          "required": ["shape", "to", "match"]})
def tool_align(executor, args: dict):
    sels, refs = _source(executor, args)
    to = args.get("to")
    if not isinstance(to, str):
        raise ValueError(f"to must be one published name, got {to!r}")
    if to == refs[0]:
        raise ValueError("to must be another shape than shape")
    _operands(executor, [to], "to")
    delta = align_delta(_bbox(executor, refs[0]), _bbox(executor, to), args.get("match"))
    if not any(delta):
        return _unchanged(refs[0], f"{refs[0]} is already in place")
    out = _translation(executor, new_name(executor, "translation"), sels, refs, delta)
    if out["ok"]:
        out["moved_by"] = delta
    return out
