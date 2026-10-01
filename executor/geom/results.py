from executor.geom.helpers import registry


def _shape_info(geompy, shape) -> dict | None:
    """World-space bounding box, size and centre (bbox midpoint) of a shape,
    in mm, rounded. Spatial feedback for the specialist -- lets it catch a
    misplaced part ("my wheel is at the origin") without a rendered view.
    Precise: the default box encloses curved faces' control poles (a filleted
    40-radius cone reads +-42.73 instead of +-39.48), and ~10 ms is nothing."""
    if shape is None:
        return None
    try:
        xmin, xmax, ymin, ymax, zmin, zmax = geompy.BoundingBox(shape, True)
    except Exception:
        return None
    r = lambda v: round(v, 4)
    return {
        "bbox": [r(xmin), r(xmax), r(ymin), r(ymax), r(zmin), r(zmax)],
        "size": [r(xmax - xmin), r(ymax - ymin), r(zmax - zmin)],
        "centre": [r((xmin + xmax) / 2), r((ymin + ymax) / 2), r((zmin + zmax) / 2)],
    }


def _attach_shape_info(executor, result: dict) -> None:
    published = [n for rec in result.get("records", []) for n in (rec.get("publish") or [])]
    if not published:
        return
    name = published[-1]
    info = _shape_info(executor.namespace["geompy"], executor.geom_bindings.get(name))
    if info:
        result["result_shape"] = name
        result["shape_info"] = info


_DELTA_RIGID = {"translate", "rotate", "mirror", "multi_translate",
                "multi_rotate", "position", "align"}


def _bp_volume(geompy, shape):
    try:
        return float(geompy.BasicProperties(shape)[2])
    except Exception:
        return None


def _reldiff(a: float, b: float) -> float:
    return abs(a - b) / max(abs(b), 1e-9)


def _bboxes_all_disjoint(geompy, shapes, tol: float = 1e-4) -> bool:
    """True only if every pair of operand bounding boxes is separated by more
    than `tol` on at least one axis — i.e. no two parts can be touching. Used
    as a cheap broad phase before the fuse "parts do not overlap or touch"
    note: face contact contributes zero overlap volume, so the volume check
    alone cannot tell an abutting fuse from a scattered one (task 222).
    Conservative: any BoundingBox failure returns False (suppress the note)."""
    boxes = []
    for s in shapes:
        info = _shape_info(geompy, s)
        if info is None:
            return False
        boxes.append(info["bbox"])
    if len(boxes) < 2:
        return False
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i], boxes[j]
            sep = any(b[2 * k] - a[2 * k + 1] > tol or a[2 * k] - b[2 * k + 1] > tol
                      for k in range(3))
            if not sep:
                return False
    return True


def _delta_note(executor, name: str, args: dict, result: dict) -> str | None:
    """One-line anomaly note for a boolean or transform whose numeric outcome
    looks wrong — a cut that removed nothing, a fuse of parts that never
    touched, a rigid move that changed volume. Silent (None) when normal, so
    it costs the specialist nothing on a well-behaved call (task 215)."""
    # registry.categories() rather than a module-level TOOL_CATEGORY: the
    # combined dict now lives in executor.geom.__init__, built after every
    # tool submodule (incl. this one) has registered; going through the
    # registry avoids a results -> __init__ import cycle.
    if registry.categories().get(name) not in ("boolean", "transform"):
        return None
    geompy = executor.namespace["geompy"]
    gb = executor.geom_bindings
    out_name = result.get("result_shape")
    out = gb.get(out_name) if out_name else None
    if out is None:
        return None
    out_v = _bp_volume(geompy, out)
    if out_v is None:
        return None

    if name == "fuse":
        vs = [_bp_volume(geompy, gb[s]) for s in args.get("shapes", []) if s in gb]
        if len(vs) >= 2 and all(v is not None for v in vs) and sum(vs) > 0:
            if _reldiff(out_v, sum(vs)) <= 0.005:
                shapes = [gb[s] for s in args.get("shapes", []) if s in gb]
                if _bboxes_all_disjoint(geompy, shapes):
                    return ("fuse: the result volume equals the sum of the operands "
                            "and their bounding boxes are disjoint — the parts are "
                            "separated, not joined; check their placement")
        return None

    if name == "cut":
        mv = _bp_volume(geompy, gb.get(args.get("main")))
        if mv and mv > 0:
            frac = out_v / mv
            if frac >= 0.995:
                return (f"cut: removed almost nothing (result is {frac * 100:.1f}% "
                        "of the original volume)")
            if frac <= 0.005:
                return (f"cut: removed almost everything (result is {frac * 100:.2f}% "
                        "of the original volume)")
        return None

    if name == "common":
        vs = [v for v in (_bp_volume(geompy, gb[s])
                          for s in args.get("shapes", []) if s in gb) if v]
        if vs and out_v <= 0.005 * min(vs):
            return ("common: the intersection is empty or negligible — the shapes "
                    "barely overlap")
        return None

    if name == "partition":
        return None

    # transforms — the source shape is always args["shape"]
    src = gb.get(args.get("shape"))
    if src is None:
        return None
    src_v = _bp_volume(geompy, src)

    if name == "scale":
        if src_v and src_v > 0:
            if "factor" in args:
                exp = src_v * float(args["factor"]) ** 3
            else:
                exp = (src_v * float(args.get("factor_x", 1))
                       * float(args.get("factor_y", 1)) * float(args.get("factor_z", 1)))
            if exp > 0 and _reldiff(out_v, exp) > 0.02:
                return (f"scale: result volume {out_v:.4g} does not match the "
                        f"{exp:.4g} expected for that factor")
        return None

    if name in _DELTA_RIGID:
        if src_v and src_v > 0 and _reldiff(out_v, src_v) > 0.01:
            return (f"{name}: volume changed {src_v:.4g} -> {out_v:.4g} under a "
                    "rigid move, which should not happen")
        # Whole bbox, not just the centre: a rotation about an axis through the
        # centre keeps the centre but turns the box (torus 28x28x8 -> 28x8x28).
        si = _shape_info(geompy, src)
        oi = result.get("shape_info")
        if si and oi and len(oi["bbox"]) == 6 and all(
                abs(oi["bbox"][k] - si["bbox"][k]) < 1e-6 for k in range(6)):
            return (f"{name}: the new shape sits exactly where the source is — "
                    "the move had no effect")
        return None
    return None


def _centre(geompy, shape) -> list:
    xmin, xmax, ymin, ymax, zmin, zmax = geompy.BoundingBox(shape, True)
    r = lambda v: round(v, 4)
    return [r((xmin + xmax) / 2), r((ymin + ymax) / 2), r((zmin + zmax) / 2)]


def _state_enum(state: str):
    from salome.geom import geomBuilder
    g = geomBuilder.GEOM
    return {"on": g.ST_ON, "in": g.ST_IN, "out": g.ST_OUT,
            "onin": g.ST_ONIN, "onout": g.ST_ONOUT}[state]


def _select_subshapes(geompy, shape, type_code: int, where) -> list:
    """Resolve a `where` predicate to a list of (id, subshape) pairs.

    `where` is "all" or a dict (a JSON string is decoded first). Raises
    ValueError on an unknown predicate.
    """
    if isinstance(where, str) and where.strip().startswith("{"):
        import json
        where = json.loads(where)

    subs = list(geompy.SubShapeAllSortedCentres(shape, type_code))
    all_ids = list(geompy.SubShapeAllSortedCentresIDs(shape, type_code))
    pairs = list(zip(all_ids, subs))

    if where == "all":
        return pairs
    if "sorted_by" in where:
        axis = {"x": 0, "y": 1, "z": 2}[where["sorted_by"]]
        take = where.get("take", "max")
        n = int(where["count"])
        ordered = sorted(pairs, key=lambda p: _centre(geompy, p[1])[axis],
                         reverse=(take == "max"))
        return ordered[:n]
    if "by_size" in where:
        kind = where["by_size"]
        n = int(where["count"])
        metric_idx = 1 if kind in ("largest", "smallest") else 0
        biggest = kind in ("longest", "largest")
        ordered = sorted(pairs,
                         key=lambda p: geompy.BasicProperties(p[1])[metric_idx],
                         reverse=biggest)
        return ordered[:n]
    if "on_plane" in where:
        op = where["on_plane"]
        pnt = geompy.MakeVertex(*[float(v) for v in op["point"]])
        nrm = geompy.MakeVectorDXDYDZ(*[float(v) for v in op["normal"]])
        st = _state_enum(where.get("state", "on"))
        sel_ids = set(geompy.GetShapesOnPlaneWithLocationIDs(
            shape, type_code, nrm, pnt, st))
        return [p for p in pairs if p[0] in sel_ids]
    raise ValueError(f"unknown where {where!r}")
