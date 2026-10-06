import math

from executor.shaper.features import _build, _operands
from executor.shaper.primitives import _AXIS, _XYZ, _axis, _num, _point
from executor.shaper.state import measure, new_name, state, tool

_MOVED = (" `shape` is used up: refer to the result by its new name.")
_SHAPE = {"type": "string"}


def _source(executor, args: dict) -> tuple[list, list]:
    shape = args.get("shape")
    if not isinstance(shape, str):
        raise ValueError(f"shape must be one published name, got {shape!r}")
    return _operands(executor, [shape], "shape")


def _count(value, what: str, least: int = 2) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value != int(value):
        raise ValueError(f"{what} must be a whole number, got {value!r}")
    if value < least:
        raise ValueError(f"{what} must be at least {least}, got {value!r}")
    return int(value)


def _translation(executor, name: str, sels: list, refs: list, d: list, fn="addTranslation",
                 code: str = "", helpers: tuple = ()) -> dict:
    code += (f"{name} = model.addTranslation(shaper_doc, [{', '.join(sels)}], "
             f"{', '.join(repr(v) for v in d)})")
    return _build(executor, name, fn, code, refs, "transform", helpers=helpers)


@tool("translate", category="transform",
      description="Move `shape` by (dx, dy, dz)." + _MOVED,
      schema={"type": "object", "properties": {
          "shape": _SHAPE, "dx": {"type": "number"}, "dy": {"type": "number"},
          "dz": {"type": "number"}}, "required": ["shape", "dx", "dy", "dz"]})
def tool_translate(executor, args: dict):
    d = [_num(args.get(k), k) for k in ("dx", "dy", "dz")]
    sels, refs = _source(executor, args)
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
    name = new_name(executor, "rotation")
    code = f"{name} = model.addRotation(shaper_doc, [{', '.join(sels)}], {axis}, {angle!r})"
    return _build(executor, name, "addRotation", code, refs + arefs, "transform")


_BASIS = {"XOY": ("FACE", "PartSet/XOY"), "XOZ": ("FACE", "PartSet/XOZ"),
          "ZOX": ("FACE", "PartSet/XOZ"), "YOZ": ("FACE", "PartSet/YOZ"),
          "OX": ("EDGE", "PartSet/OX"), "OY": ("EDGE", "PartSet/OY"),
          "OZ": ("EDGE", "PartSet/OZ"), "O": ("VERTEX", "PartSet/Origin")}
_BASIS_KIND = {"plane": "FACE", "axis": "EDGE", "point": "VERTEX"}


def _basis(executor, entity) -> tuple[str, list]:
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
    sels, refs = _source(executor, args)
    axis, arefs = _axis(executor, args.get("vector"), "vector")
    rest = f"{axis}, {step!r}, {count}"
    if any(args.get(k) is not None for k in ("vector2", "step2", "count2")):
        step2 = _num(args.get("step2"), "step2")
        count2 = _count(args.get("count2"), "count2")
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
        rest = f"{axis}, {_num(args['angle_deg'], 'angle_deg')!r}, {count}"
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

def _unit(v: list, what: str) -> list:
    n = math.sqrt(sum(c * c for c in v))
    if n < 1e-12:
        raise ValueError(f"{what} must not be [0, 0, 0]")
    return [c / n for c in v]


def frame_rotation(x_dir: list, y_dir: list) -> tuple[list, float]:
    """Axis (through the origin) and angle in degrees of the rotation taking the
    world X and Y axes to x_dir and y_dir. y_dir is made perpendicular to x_dir."""
    x = _unit(x_dir, "x_dir")
    dot = sum(a * b for a, b in zip(x, y_dir))
    y = _unit([b - dot * a for a, b in zip(x, y_dir)], "y_dir (it is parallel to x_dir)")
    z = [x[1] * y[2] - x[2] * y[1], x[2] * y[0] - x[0] * y[2], x[0] * y[1] - x[1] * y[0]]
    R = [[x[i], y[i], z[i]] for i in range(3)]       # columns are the new axes
    cos = max(-1.0, min(1.0, (R[0][0] + R[1][1] + R[2][2] - 1) / 2))
    angle = math.acos(cos)
    if angle < 1e-9:
        return [0.0, 0.0, 1.0], 0.0
    if math.pi - angle > 1e-6:
        axis = [R[2][1] - R[1][2], R[0][2] - R[2][0], R[1][0] - R[0][1]]
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
    vec = lambda k, d: [_num(v, k) for v in (args.get(k) or d)]
    origin = vec("origin", [0, 0, 0])
    axis, angle = frame_rotation(vec("x_dir", [1, 0, 0]), vec("y_dir", [0, 1, 0]))
    if len(origin) != 3:
        raise ValueError(f"origin must be [x, y, z], got {origin}")
    sels, refs = _source(executor, args)
    name = new_name(executor, "position")
    if angle == 0:
        return _translation(executor, name, sels, refs, origin, fn="addPlacement")
    ax, rot = f"{name}_axis", f"{name}_rot"
    code = (f"{ax} = model.addAxis(shaper_doc, {', '.join(repr(a) for a in axis)})\n"
            f"_name({ax}, {ax!r})\n"
            f"{rot} = model.addRotation(shaper_doc, [{', '.join(sels)}], "
            f"model.selection(\"EDGE\", {ax!r}), {angle!r})\n"
            f"_name({rot}, {rot!r})\n")
    return _translation(executor, name, [f"*{rot}.results()"], refs, origin,
                        fn="addPlacement", code=code, helpers=(ax, rot))


# --- align ------------------------------------------------------------------

_AXIS_IDX = {"x": 0, "y": 1, "z": 2}


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
        if axis not in _AXIS_IDX:
            raise ValueError(f"match axis must be x, y or z, got {axis!r}")
        if axis in seen:
            raise ValueError(f"two match entries for axis {axis}")
        seen.add(axis)
        if c.get("my") not in ("min", "max", "center") or \
                c.get("target") not in ("min", "max", "center"):
            raise ValueError("my and target must be min, max or center")
        i = _AXIS_IDX[axis]
        gap = _num(c.get("gap", 0) or 0, "gap")
        delta[i] = round(ext(to_bbox, i, c["target"]) + gap - ext(shape_bbox, i, c["my"]), 6)
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
    _operands(executor, [to], "to")
    delta = align_delta(measure(executor, refs[0])["bbox"], measure(executor, to)["bbox"],
                        args.get("match"))
    if not any(delta):
        return {"ok": True, "records": [], "name": refs[0],
                "note": f"{refs[0]} is already in place; nothing was moved"}
    out = _translation(executor, new_name(executor, "translation"), sels, refs, delta)
    if out["ok"]:
        out["moved_by"] = delta
    return out
