"""The vision agent's structured spec (task 286): mechanical validation, and the
renderings the planner, geom stages, amend and the report get."""
import math
import re

TOL = 0.01
PLANE_AXES = {"XY": (0, 1), "YZ": (1, 2), "XZ": (0, 2)}
PLANE_FIXED = {"XY": 2, "YZ": 0, "XZ": 1}
PRIMITIVES = ("box", "cone", "cylinder", "sphere", "torus")
FEATURES = ("hole", "fillet", "chamfer", "pattern")

_NUM = re.compile(r"^~?\s*-?\d+(\.\d+)?$")


def parse_num(v):
    """A number, or an estimate given as "~N"; None when it is neither."""
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str) and _NUM.match(v.strip()):
        return float(v.strip().lstrip("~").strip())
    return None


def parse_point(v):
    """"x,y,z" (any coordinate may be "~N") -> (x, y, z); None otherwise."""
    if not isinstance(v, str):
        return None
    bits = v.split(",")
    if len(bits) != 3:
        return None
    vals = [parse_num(b.strip()) for b in bits]
    return None if None in vals else tuple(vals)


def _fmt(p) -> str:
    return "(" + ", ".join(f"{c:g}" for c in p) + ")"


def _flat(p, plane):
    a, b = PLANE_AXES[plane]
    return (p[a], p[b])


def _off_line(p, q, r) -> float:
    """Distance from q to the line p-r (inf when p and r coincide)."""
    d = math.dist(p, r)
    if d < TOL:
        return math.inf
    return abs((r[0] - p[0]) * (q[1] - p[1]) - (r[1] - p[1]) * (q[0] - p[0])) / d


def _circumcircle(p, q, r):
    ax, ay = p
    bx, by = q
    cx, cy = r
    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < 1e-12:
        return None
    ux = ((ax * ax + ay * ay) * (by - cy) + (bx * bx + by * by) * (cy - ay)
          + (cx * cx + cy * cy) * (ay - by)) / d
    uy = ((ax * ax + ay * ay) * (cx - bx) + (bx * bx + by * by) * (ax - cx)
          + (cx * cx + cy * cy) * (bx - ax)) / d
    return (ux, uy), math.dist((ux, uy), p)


def _check_outline(outline, plane, where, err):
    """Check one {start, segments} outline; return the in-plane points that bound
    it (arcs as their full circle), or None when it cannot be traced."""
    start = parse_point(outline.get("start"))
    segs = outline.get("segments")
    if start is None:
        err(f'{where}.start must be an "x,y,z" point')
        return None
    if not isinstance(segs, list) or not segs:
        err(f"{where}.segments must be a non-empty list")
        return None
    fixed = PLANE_FIXED[plane]

    def on_plane(p, label):
        if abs(p[fixed] - start[fixed]) > TOL:
            err(f"{where} {label}: {_fmt(p)} is not in the {plane} plane of start {_fmt(start)}")

    extent = [_flat(start, plane)]
    corners = [_flat(start, plane)]
    straight_only = True
    cur = start
    for k, seg in enumerate(segs, 1):
        label = f"segment {k}"
        to = parse_point(seg.get("to")) if isinstance(seg, dict) else None
        if to is None:
            err(f'{where} {label}: "to" must be an "x,y,z" point')
            return None
        on_plane(to, label)
        forms = [key for key in ("center", "through", "via") if seg.get(key) is not None]
        if len(forms) > 1:
            err(f"{where} {label}: give one of center, through or via, not {' and '.join(forms)}")
        elif forms == ["center"]:
            c = parse_point(seg["center"])
            if c is None:
                err(f'{where} {label}: center must be an "x,y,z" point')
            else:
                straight_only = False
                r0, r1 = math.dist(cur, c), math.dist(to, c)
                if abs(r0 - r1) > TOL:
                    err(f"{where} {label}: arc start {_fmt(cur)} is {r0:g} from centre {_fmt(c)} "
                        f"but its end {_fmt(to)} is {r1:g}; both ends must lie on one radius")
                cx, cy = _flat(c, plane)
                r = max(r0, r1)
                extent += [(cx - r, cy - r), (cx + r, cy + r)]
        elif forms == ["through"]:
            t = parse_point(seg["through"])
            if t is None:
                err(f'{where} {label}: through must be an "x,y,z" point')
            else:
                straight_only = False
                on_plane(t, label)
                p, q, r = _flat(cur, plane), _flat(t, plane), _flat(to, plane)
                circle = _circumcircle(p, q, r)
                if _off_line(p, q, r) <= TOL or circle is None:
                    err(f"{where} {label}: through point {_fmt(t)} is on the line from "
                        f"{_fmt(cur)} to {_fmt(to)}, so it defines no arc")
                else:
                    (cx, cy), rad = circle
                    extent += [(cx - rad, cy - rad), (cx + rad, cy + rad)]
        elif forms == ["via"]:
            via = ([parse_point(v) for v in seg["via"]]
                   if isinstance(seg["via"], list) and seg["via"] else [None])
            if None in via:
                err(f'{where} {label}: via must be a list of "x,y,z" points')
            else:
                straight_only = False
                extent += [_flat(v, plane) for v in via]
        extent.append(_flat(to, plane))
        corners.append(_flat(to, plane))
        cur = to
    distinct = []
    for c in corners:
        if all(math.dist(c, d) > TOL for d in distinct):
            distinct.append(c)
    flat_line = (len(distinct) < 3 or
                 all(_off_line(distinct[0], q, distinct[1]) <= TOL for q in distinct[2:]))
    if len(distinct) < 2 or (straight_only and flat_line):
        err(f"{where}: the outline is degenerate (its corners lie on one line)")
    return extent


def outline_extent(outline, plane):
    if not isinstance(outline, dict) or plane not in PLANE_AXES:
        return None
    return _check_outline(outline, plane, "", lambda m: None)


def _box(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def _check_hole(hole, plane, box, where, err):
    x0, y0, x1, y1 = box

    def inside(q, label):
        if not (x0 - TOL <= q[0] <= x1 + TOL and y0 - TOL <= q[1] <= y1 + TOL):
            err(f"{where}: {label} {_fmt(q)} lies outside the outline "
                f"({x0:g}..{x1:g} by {y0:g}..{y1:g} in the {plane} plane)")
            return False
        return True

    if not isinstance(hole, dict):
        err(f"{where} must be an object")
    elif "segments" in hole:
        pts = _check_outline(hole, plane, where, err)
        for q in pts or []:
            if not inside(q, "point"):
                break
    elif "slot" in hole:
        s = hole["slot"]
        if not isinstance(s, dict) or (parse_num(s.get("width")) or 0) <= 0:
            err(f"{where}.slot.width must be a positive number")
        elif "from" in s or "to" in s:
            ends = [parse_point(s.get("from")), parse_point(s.get("to"))]
            if None in ends:
                err(f'{where}.slot: "from" and "to" must be "x,y,z" points')
            else:
                for e in ends:
                    if not inside(_flat(e, plane), "slot end"):
                        break
        elif "center" in s:
            c = parse_point(s.get("center"))
            angles = [parse_num(s.get("from_angle")), parse_num(s.get("to_angle"))]
            if c is None or (parse_num(s.get("radius")) or 0) <= 0 or None in angles:
                err(f'{where}.slot: an arc slot needs "center" as "x,y,z", a positive '
                    f'"radius" and "from_angle"/"to_angle" in degrees')
            else:
                inside(_flat(c, plane), "slot centre")
        else:
            err(f"{where}.slot: give {{from, to, width}} or "
                f"{{center, radius, width, from_angle, to_angle}}")
    elif "center" in hole:
        c = parse_point(hole.get("center"))
        r = parse_num(hole.get("radius"))
        if c is None or r is None or r <= 0:
            err(f'{where}: a round hole needs "center" as "x,y,z" and a positive "radius"')
        else:
            inside(_flat(c, plane), "centre")
    elif "wire" in hole:
        err(f"{where}: a spec hole cannot name a wire; give its outline")
    else:
        err(f"{where}: give an outline {{start, segments}}, a circle {{center, radius}} or a {{slot}}")


def _check_dims(dims, where, err):
    if not isinstance(dims, dict):
        err(f"{where} must be an object of named sizes")
        return
    for k, v in dims.items():
        if parse_num(v) is None:
            err(f'{where}.{k} must be a number or "~N"')


def _check_profile_base(base, dim, err):
    plane = base.get("plane")
    if plane not in PLANE_AXES:
        err('base.plane must be "XY", "YZ" or "XZ"')
        return
    ext, rev = base.get("extrude"), base.get("revolve")
    if ext and rev:
        err("base: give extrude or revolve, not both")
    if dim == "2d" and (ext or rev):
        err("base: a 2d target has no extrude or revolve")
    if ext:
        if not isinstance(ext, dict) or not parse_num(ext.get("height")):
            err("base.extrude.height must be a non-zero number")
        elif ext.get("direction") is not None and parse_point(ext["direction"]) is None:
            err('base.extrude.direction must be an "x,y,z" vector')
    if rev:
        if (not isinstance(rev, dict) or parse_point(rev.get("point")) is None
                or parse_point(rev.get("direction")) is None):
            err('base.revolve needs "point" and "direction" as "x,y,z"')
        elif not parse_num(rev.get("angle")):
            err("base.revolve.angle must be a non-zero number of degrees")
    prof = base.get("profile")
    if not isinstance(prof, dict):
        err("base.profile must be an object {start, segments, holes}")
        return
    pts = _check_outline(prof, plane, "base.profile", err)
    if not pts:
        return
    box = _box(pts)
    for h, hole in enumerate(prof.get("holes") or [], 1):
        _check_hole(hole, plane, box, f"base.profile.holes[{h}]", err)


def _check_feature(f, j, n, err):
    w = f"features[{j}]"
    if not isinstance(f, dict) or f.get("type") not in FEATURES:
        err(f"{w}.type must be one of {', '.join(FEATURES)}")
        return
    t = f["type"]
    if t == "hole":
        if parse_point(f.get("center")) is None:
            err(f'{w}.center must be an "x,y,z" point')
        if (parse_num(f.get("diameter")) or 0) <= 0:
            err(f"{w}.diameter must be a positive number")
        depth = f.get("depth")
        if depth != "through" and (parse_num(depth) or 0) <= 0:
            err(f'{w}.depth must be a positive number or "through"')
        if f.get("axis") is not None and parse_point(f["axis"]) is None:
            err(f'{w}.axis must be an "x,y,z" vector')
    elif t in ("fillet", "chamfer"):
        key = "radius" if t == "fillet" else "size"
        if (parse_num(f.get(key)) or 0) <= 0:
            err(f"{w}.{key} must be a positive number")
        if not isinstance(f.get("where"), str) or not f["where"].strip():
            err(f"{w}.where must say which edges")
    else:
        of = f.get("of")
        if isinstance(of, bool) or not isinstance(of, int) or not 1 <= of <= n or of == j:
            err(f"{w}.of must be the 1-based index of another feature (1-{n})")
        if f.get("kind") not in ("circular", "linear"):
            err(f'{w}.kind must be "circular" or "linear"')
        c = f.get("count")
        if isinstance(c, bool) or not isinstance(c, int) or c < 2:
            err(f"{w}.count must be an integer >= 2")


def _check_part(part, dim, ids, err):
    base = part.get("base")
    kind = base.get("kind") if isinstance(base, dict) else None
    if kind == "profile":
        _check_profile_base(base, dim, err)
    elif kind in ("primitive", "approx"):
        if dim == "2d":
            err("base: a 2d target is built from profile parts only")
        if kind == "primitive" and base.get("shape") not in PRIMITIVES:
            err(f"base.shape must be one of {', '.join(PRIMITIVES)}")
        if kind == "approx" and (not isinstance(base.get("description"), str)
                                 or not base["description"].strip()):
            err("base.description must say what the shape is")
        _check_dims(base.get("dims"), "base.dims", err)
    else:
        err('base.kind must be "profile", "primitive" or "approx"')
    pl = part.get("placement")
    if not isinstance(pl, dict):
        err("placement must be an object {origin, normal}")
    else:
        for key in ("origin", "normal"):
            if parse_point(pl.get(key)) is None:
                err(f'placement.{key} must be an "x,y,z" point')
        if pl.get("x_dir") is not None and parse_point(pl["x_dir"]) is None:
            err('placement.x_dir must be an "x,y,z" vector')
        n = parse_point(pl.get("normal"))
        if n is not None and math.hypot(*n) < TOL:
            err("placement.normal must not be zero")
    feats = part.get("features") or []
    if not isinstance(feats, list):
        err("features must be a list")
        feats = []
    for j, f in enumerate(feats, 1):
        _check_feature(f, j, len(feats), err)
    for k, op in enumerate(part.get("ops") or [], 1):
        if not isinstance(op, dict) or op.get("op") not in ("cut", "fuse"):
            err(f'ops[{k}].op must be "cut" or "fuse"')
        elif op.get("with") not in ids or op.get("with") == part.get("id"):
            err(f"ops[{k}].with must name another part ({', '.join(sorted(ids))})")


def validate_spec(spec) -> list[tuple[str | None, str]]:
    """Mechanical checks of a vision spec; [] when it is sound. Each error is
    (part id or None, message)."""
    if not isinstance(spec, dict):
        return [(None, "the spec must be a JSON object")]
    errs: list[tuple[str | None, str]] = []
    dim_block = spec.get("dimensionality")
    dim = dim_block.get("value") if isinstance(dim_block, dict) else None
    if dim not in ("2d", "3d"):
        errs.append((None, 'dimensionality.value must be "2d" or "3d"'))
    parts = spec.get("parts")
    if not isinstance(parts, list) or not parts:
        errs.append((None, "parts must be a non-empty list"))
        return errs
    ids = {p["id"] for p in parts if isinstance(p, dict) and isinstance(p.get("id"), str)}
    seen = set()
    for i, part in enumerate(parts, 1):
        if not isinstance(part, dict):
            errs.append((None, f"parts[{i}] must be an object"))
            continue
        pid = part.get("id")
        if not isinstance(pid, str) or not pid:
            errs.append((None, f"parts[{i}] needs a string id"))
            continue
        if pid in seen:
            errs.append((pid, "duplicate part id"))
            continue
        seen.add(pid)
        _check_part(part, dim, ids, lambda m, pid=pid: errs.append((pid, m)))
    return errs
