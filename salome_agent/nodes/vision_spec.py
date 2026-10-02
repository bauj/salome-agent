"""The vision agent's structured spec (task 286): mechanical validation, and the
renderings the planner, geom stages, amend and the report get."""
import copy
import json
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


def _arc_points(c, r, a0, sweep, chord=0.5):
    """Points along an arc from angle a0 (radians) through sweep."""
    n = max(2, math.ceil(abs(sweep) * r / chord) + 1)
    return [(c[0] + r * math.cos(a0 + sweep * k / (n - 1)),
             c[1] + r * math.sin(a0 + sweep * k / (n - 1))) for k in range(n)]


def _sample_outline(outline, plane):
    """The outline as a closed polyline of in-plane points, arcs traced the way
    make_profile_face traces them; None when it cannot be parsed."""
    start = parse_point(outline.get("start"))
    if start is None or not isinstance(outline.get("segments"), list):
        return None
    cur = _flat(start, plane)
    pts = [cur]
    for seg in outline["segments"]:
        to = parse_point(seg.get("to")) if isinstance(seg, dict) else None
        if to is None:
            return None
        to = _flat(to, plane)
        if seg.get("center") is not None:
            c = parse_point(seg["center"])
            if c is None:
                return None
            c = _flat(c, plane)
            a0 = math.atan2(cur[1] - c[1], cur[0] - c[0])
            a1 = math.atan2(to[1] - c[1], to[0] - c[0])
            sweep = (a1 - a0 + math.pi) % (2 * math.pi) - math.pi
            if seg.get("reverse"):
                sweep -= math.copysign(2 * math.pi, sweep)
            pts += _arc_points(c, math.dist(cur, c), a0, sweep, 1.0)[1:]
        elif seg.get("through") is not None:
            t = parse_point(seg["through"])
            circle = _circumcircle(cur, _flat(t, plane), to) if t else None
            if circle is None:
                return None
            c, r = circle
            ang = [math.atan2(q[1] - c[1], q[0] - c[0]) for q in (cur, _flat(t, plane), to)]
            ccw = (ang[2] - ang[0]) % (2 * math.pi)
            sweep = ccw if (ang[1] - ang[0]) % (2 * math.pi) < ccw else ccw - 2 * math.pi
            pts += _arc_points(c, r, ang[0], sweep, 1.0)[1:]
        elif isinstance(seg.get("via"), list):
            via = [parse_point(v) for v in seg["via"]]
            if None in via:
                return None
            pts += [_flat(v, plane) for v in via] + [to]
        else:
            pts.append(to)
        cur = to
    if math.dist(pts[-1], pts[0]) > TOL:
        pts.append(pts[0])
    return pts


def _seg_dist(p, a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]
    L = dx * dx + dy * dy
    t = 0.0 if L == 0 else max(0.0, min(1.0, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / L))
    return math.dist(p, (a[0] + t * dx, a[1] + t * dy))


def _poly_dist(A, B):
    """Smallest distance between two polylines (a single point is a polyline too)."""
    def one_way(P, Q):
        if len(Q) == 1:
            return min(math.dist(p, Q[0]) for p in P)
        return min(_seg_dist(p, Q[i], Q[i + 1]) for p in P for i in range(len(Q) - 1))
    return min(one_way(A, B), one_way(B, A))


def _inside_polygon(q, poly):
    inside = False
    for (x0, y0), (x1, y1) in zip(poly, poly[1:]):
        if (y0 > q[1]) != (y1 > q[1]) and q[0] < x0 + (q[1] - y0) * (x1 - x0) / (y1 - y0):
            inside = not inside
    return inside


def _check_hole(hole, plane, box, where, err):
    """Check one hole; for a round hole or a slot that sits in the outline's box,
    return (core polyline, half-width) for the overlap checks."""
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
            elif all(inside(_flat(e, plane), "slot end") for e in ends):
                return [_flat(e, plane) for e in ends], parse_num(s["width"]) / 2
        elif "center" in s:
            c = parse_point(s.get("center"))
            r = parse_num(s.get("radius"))
            angles = [parse_num(s.get("from_angle")), parse_num(s.get("to_angle"))]
            if c is None or (r or 0) <= 0 or None in angles:
                err(f'{where}.slot: an arc slot needs "center" as "x,y,z", a positive '
                    f'"radius" and "from_angle"/"to_angle" in degrees')
            elif inside(_flat(c, plane), "slot centre"):
                sweep = math.radians((angles[1] - angles[0]) % 360)
                return (_arc_points(_flat(c, plane), r, math.radians(angles[0]), sweep),
                        parse_num(s["width"]) / 2)
        else:
            err(f"{where}.slot: give {{from, to, width}} or "
                f"{{center, radius, width, from_angle, to_angle}}")
    elif "center" in hole:
        c = parse_point(hole.get("center"))
        r = parse_num(hole.get("radius"))
        if c is None or r is None or r <= 0:
            err(f'{where}: a round hole needs "center" as "x,y,z" and a positive "radius"')
        elif inside(_flat(c, plane), "centre"):
            return [_flat(c, plane)], r
    elif "wire" in hole:
        err(f"{where}: a spec hole cannot name a wire; give its outline")
    else:
        err(f"{where}: give an outline {{start, segments}}, a circle {{center, radius}} or a {{slot}}")
    return None


def _check_hole_layout(regions, boundary, err):
    """Round holes and slots must sit inside the outline without touching it or
    each other (else GEOM builds no single face)."""
    placed = []
    for h, (core, r) in regions:
        where = f"base.profile.holes[{h}]"
        if not all(_inside_polygon(q, boundary) for q in core):
            label = "centre" if len(core) == 1 else "slot centre line"
            err(f"{where}: {label} {_fmt(core[0])} lies outside the outline"
                if len(core) == 1 else f"{where}: {label} leaves the outline")
            continue
        d = _poly_dist(core, boundary)
        if d < r - TOL:
            err(f"{where}: the hole crosses the outline by {r - d:.3g}")
            continue
        placed.append((h, core, r))
    for i, (hi, ci, ri) in enumerate(placed):
        for hj, cj, rj in placed[i + 1:]:
            d = _poly_dist(ci, cj)
            if d < ri + rj - TOL:
                err(f"base.profile.holes[{hi}] and holes[{hj}] overlap by {ri + rj - d:.3g}; "
                    "holes must not touch each other")


def _check_dims(dims, where, err):
    if not isinstance(dims, dict):
        err(f"{where} must be an object of named sizes")
        return
    for k, v in dims.items():
        if parse_num(v) is None:
            err(f'{where}.{k} must be a number or "~N"')


def _check_fillet_arcs(outline, plane, where, err):
    """A centre arc tangent to the straight lines on both sides is a rounded
    corner: make_profile_face takes it as the corner plus "fillet", which spares
    the geom stage working out the arc itself."""
    start = parse_point(outline["start"])
    pts = [_flat(start, plane)] + [_flat(parse_point(s["to"]), plane) for s in outline["segments"]]
    segs = list(outline["segments"])
    if math.dist(pts[-1], pts[0]) > TOL:
        segs.append({"to": outline["start"]})
        pts.append(pts[0])
    n = len(segs)

    def straight(i):
        return not any(segs[i % n].get(k) is not None for k in ("center", "through", "via"))

    def unit(a, b):
        d = math.dist(a, b)
        return None if d < TOL else ((b[0] - a[0]) / d, (b[1] - a[1]) / d)

    for k, seg in enumerate(segs):
        if seg.get("center") is None or not straight(k - 1) or not straight(k + 1):
            continue
        c = _flat(parse_point(seg["center"]), plane)
        a, b = pts[k], pts[k + 1]
        before = unit(pts[k - 1] if k else pts[-2], a)
        after = unit(b, pts[(k + 2) % len(pts)] if k + 2 < len(pts) else pts[1])
        ra, rb = unit(c, a), unit(c, b)
        if None in (before, after, ra, rb):
            continue
        if abs(before[0] * ra[0] + before[1] * ra[1]) > 0.02 or \
                abs(after[0] * rb[0] + after[1] * rb[1]) > 0.02:
            continue
        cross = before[0] * after[1] - before[1] * after[0]
        if abs(cross) < 1e-6:
            continue
        t = ((b[0] - a[0]) * after[1] - (b[1] - a[1]) * after[0]) / cross
        corner = (a[0] + t * before[0], a[1] + t * before[1])
        err(f"{where} segment {k + 1}: this arc rounds the corner at {_fmt(corner)} between two "
            f'straight lines; end the line before it at that corner with "fillet": '
            f"{math.dist(c, a):g} and drop the arc")


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
    outline_errors = []
    pts = _check_outline(prof, plane, "base.profile", outline_errors.append)
    for m in outline_errors:
        err(m)
    if not pts:
        return
    if not outline_errors:
        _check_fillet_arcs(prof, plane, "base.profile", err)
    box = _box(pts)
    regions = []
    for h, hole in enumerate(prof.get("holes") or [], 1):
        region = _check_hole(hole, plane, box, f"base.profile.holes[{h}]", err)
        if region:
            regions.append((h, region))
    boundary = _sample_outline(prof, plane)
    if boundary and len(boundary) > 3 and not outline_errors:
        _check_hole_layout(regions, boundary, err)


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


def format_errors(errors) -> str:
    return "\n".join(f"- {pid}: {msg}" if pid else f"- {msg}" for pid, msg in errors)


def attach_conflicts(spec, errors) -> dict:
    """The spec with each remaining validator error stored where it applies."""
    if not isinstance(spec, dict):
        return {"summary": json.dumps(spec, ensure_ascii=False), "parts": [],
                "conflicts": [m for _, m in errors]}
    out = copy.deepcopy(spec)
    by_id = {p.get("id"): p for p in out.get("parts") or [] if isinstance(p, dict)}
    for pid, msg in errors:
        target = by_id.get(pid, out)
        target.setdefault("conflicts", []).append(msg)
    return out


def _parts(spec) -> list[dict]:
    return [p for p in spec.get("parts") or [] if isinstance(p, dict)]


def all_conflicts(spec: dict) -> list[str]:
    out = list(spec.get("conflicts") or [])
    for p in _parts(spec):
        out += [f"{p.get('id')}: {c}" for c in p.get("conflicts") or []]
    return out


def _target_line(spec) -> str | None:
    dim = spec.get("dimensionality")
    if not isinstance(dim, dict) or dim.get("value") not in ("2d", "3d"):
        return None
    line = f"Target: {dim['value'].upper()}"
    if dim["value"] == "2d":
        planes = {(p.get("base") or {}).get("plane") for p in _parts(spec)} - {None}
        line += f" in the {'/'.join(sorted(planes)) or 'XY'} plane"
    if dim.get("assumed"):
        line += f" (assumed: {dim.get('reason') or 'the image gave no depth'})"
    return line


def _json(v) -> str:
    return json.dumps(v, ensure_ascii=False)


def _part_line(p) -> str:
    base = p.get("base") if isinstance(p.get("base"), dict) else {}
    kind = base.get("kind")
    head = f"{p.get('id')} {p.get('name') or ''}".rstrip()
    if kind == "profile":
        desc = f"profile in {base.get('plane')}"
        if base.get("extrude"):
            desc += f", extruded {(base['extrude'] or {}).get('height')}"
        elif base.get("revolve"):
            desc += f", revolved {(base['revolve'] or {}).get('angle')} deg"
        else:
            desc += ", flat face"
        prof = base.get("profile") if isinstance(base.get("profile"), dict) else {}
        pts = outline_extent(prof, base.get("plane"))
        if pts:
            x0, y0, x1, y1 = _box(pts)
            desc += f", {x1 - x0:g} x {y1 - y0:g}"
        holes = len(prof.get("holes") or [])
        if holes:
            desc += f", {holes} hole{'s' if holes > 1 else ''}"
    elif kind == "primitive":
        desc = f"{base.get('shape')} {_json(base.get('dims') or {})}"
    elif kind == "approx":
        desc = f"approx. {base.get('description')} {_json(base.get('dims') or {})}"
    else:
        desc = "unknown base"
    feats = [f.get("type") for f in p.get("features") or [] if isinstance(f, dict)]
    if feats:
        counts = {}
        for t in feats:
            counts[t] = counts.get(t, 0) + 1
        desc += "; features: " + ", ".join(f"{n} {t}" for t, n in counts.items())
    if p.get("relation"):
        desc += f"; {p['relation']}"
    if p.get("conflicts"):
        n = len(p["conflicts"])
        desc += f"; {n} conflict{'s' if n > 1 else ''}"
    return f"- {head}: {desc}"


def _context_lines(spec) -> list[str]:
    lines = []
    target = _target_line(spec)
    if target:
        lines.append(target)
    if spec.get("summary"):
        lines.append(f"Summary: {spec['summary']}")
    if spec.get("datum"):
        lines.append(f"Datum: {spec['datum']}")
    if spec.get("uncertainties"):
        lines.append("Uncertainties: " + "; ".join(str(u) for u in spec["uncertainties"]))
    if spec.get("conflicts"):
        lines.append("Conflicts: " + "; ".join(str(c) for c in spec["conflicts"]))
    return lines


def compact_spec(spec: dict) -> str:
    """The planner's view: parts by id and kind, no numbers to restate."""
    lines = ["Reference image spec (read from the user's image(s); each geom stage is "
             "handed the exact data of the parts it lists):"]
    lines += _context_lines(spec)
    parts = _parts(spec)
    if parts:
        lines.append("Parts:")
        lines += [_part_line(p) for p in parts]
    return "\n".join(lines)


def parts_block(spec: dict, ids: list[str]) -> str:
    """A geom stage's view: its parts' exact JSON."""
    chosen = [p for p in _parts(spec) if p.get("id") in ids]
    lines = ["Reference image spec for this stage's part(s), read from the user's "
             "image(s) by an image-analysis agent (you do not see the images):"]
    lines += _context_lines(spec)
    lines.append("```json\n" + json.dumps(chosen, ensure_ascii=False, indent=1) + "\n```")
    return "\n".join(lines)


def amend_block(spec: dict) -> str:
    return (compact_spec(spec) + "\nParts data:\n```json\n"
            + json.dumps(_parts(spec), ensure_ascii=False, indent=1) + "\n```")


def part_ids(stage: dict) -> list[str]:
    ids = stage.get("parts") or []
    return [ids] if isinstance(ids, str) else [i for i in ids if isinstance(i, str)]


def stage_part_ids(plan: list[dict], idx: int, spec: dict) -> list[str]:
    """The parts stage idx builds; every part when no stage of the plan lists any
    (a planner that ignored `parts` must not leave geom without the spec)."""
    if any(part_ids(s) for s in plan):
        return part_ids(plan[idx])
    return [p.get("id") for p in _parts(spec)]


def unclaimed_parts(plan: list[dict], spec: dict) -> list[str]:
    claimed = {i for s in plan for i in part_ids(s)}
    if not claimed:
        return []
    return [p.get("id") for p in _parts(spec) if p.get("id") not in claimed]
