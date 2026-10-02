from executor.geom.helpers import _check_shape, tool, _check_sub_ids, _next_name, _publish
from executor.geom.results import _select_subshapes


def _normalise_edge_ids(tool: str, edges, shape, geompy) -> list:
    """Coerce the `edges` arg of fillet/chamfer into a list of int edge ids.

    Accepts "all", an int, a list, or a string like "1,2,3" / "(1,2,3)" /
    "[1, 2, 3]" (a real LLM has produced all of these). Anything else raises a
    clear error rather than letting a bare string reach geompy, which returns
    an opaque `CORBA.BAD_PARAM ... Expecting sequence, got <class 'str'>`.
    """
    if edges == "all":
        return geompy.SubShapeAllIDs(shape, geompy.ShapeType["EDGE"])
    if isinstance(edges, int):
        edges = [edges]
    if isinstance(edges, str):
        edges = [p for p in edges.strip(" []()").replace(" ", "").split(",") if p]
    if not isinstance(edges, (list, tuple)) or not edges:
        raise ValueError(
            f'{tool}: pass `edges` ("all" or a list of edge ids) or a `where` '
            f'predicate, got edges={edges!r}')
    try:
        return [int(e) for e in edges]
    except (TypeError, ValueError):
        raise ValueError(f'{tool}: every entry in `edges` must be an integer, got {edges!r}')


def _edge_ids_for(tool: str, executor, args: dict, shape):
    """Edge ids for fillet/chamfer: a `where` predicate (inline get_subshapes)
    if given, otherwise the `edges` arg via _normalise_edge_ids."""
    geompy = executor.namespace["geompy"]
    if "where" in args and args["where"] is not None:
        try:
            selected = _select_subshapes(
                geompy, shape, int(geompy.ShapeType["EDGE"]), args["where"])
        except ValueError as e:
            raise ValueError(f"{tool}: {e}")
        return [i for i, _ in selected]
    ids = _normalise_edge_ids(tool, args.get("edges"), shape, geompy)
    if args.get("edges") != "all":
        _check_sub_ids(tool, geompy, shape, args["shape"], ids, "EDGE")
    return ids


def _shortest_edge(geompy, shape, ids) -> float | None:
    """Length of the shortest of the given edges, measured — never found by
    trying fillet radii: a pathological MakeFillet hangs OCC and wedges the
    server instead of raising."""
    try:
        return min(float(geompy.BasicProperties(geompy.GetSubShape(shape, [i]))[0])
                   for i in ids)
    except Exception:  # noqa: BLE001 — best-effort hint only
        return None


_MAX_FACES_FOR_WIDTH = 100


def _narrowest_neighbour(geompy, shape, ids) -> tuple[float, float, bool] | None:
    """(radius ceiling, face width, both sides selected) from the narrowest face
    next to a selected edge: its width is the distance to the nearest edge of
    that face that does not touch the selected one (a 2 mm rim for a shelled
    vase). A fillet on both edges of such a face gets half of it."""
    try:
        edge_t, face_t = geompy.ShapeType["EDGE"], geompy.ShapeType["FACE"]
        faces = geompy.SubShapeAll(shape, face_t)
        if len(faces) > _MAX_FACES_FOR_WIDTH:
            return None
        selected, best = set(ids), None
        for eid in ids:
            edge = geompy.GetSubShape(shape, [eid])
            for face in faces:
                if geompy.MinDistance(face, edge) > 1e-7:
                    continue
                face_edges = geompy.SubShapeAll(face, edge_t)
                face_ids = [geompy.GetSubShapeID(shape, e) for e in face_edges]
                if eid not in face_ids:
                    continue
                for oid, other in zip(face_ids, face_edges):
                    if oid == eid:
                        continue
                    width = float(geompy.MinDistance(edge, other))
                    if width <= 1e-7:
                        continue
                    both = oid in selected
                    ceiling = width / 2 if both else width
                    if best is None or ceiling < best[0]:
                        best = (ceiling, width, both)
        return best
    except Exception:  # noqa: BLE001 — best-effort hint only
        return None


def _fillet_advice(geompy, shape, ids) -> str:
    shortest = _shortest_edge(geompy, shape, ids)
    narrow = _narrowest_neighbour(geompy, shape, ids)
    if narrow and (shortest is None or narrow[0] < shortest / 2):
        # Half the geometric ceiling: a concave edge (inner rim of a shell)
        # fails well before it, and one number stops the model second-guessing.
        ceiling, width, _ = narrow
        return (f"the radius is too large — a face next to the selected edges is only "
                f"{width:.4g} mm wide. Use radius {ceiling / 2:.2g}: it fits here")
    if shortest:
        return (f"the radius is too large — the shortest selected edge is "
                f"{shortest:.4g} mm. Use radius {shortest / 4:.2g}: it fits here")
    return "try a smaller radius or check the edge ids"


@tool("fillet_edges", category="feature",
      description='Round edges of `shape` with the given `radius`. Select edges with `edges` ("all" or a list of edge ids from get_subshapes/explode) or inline with a `where` predicate (same as get_subshapes). Fails if the radius is too large for the shape.',
      schema={"type": "object", "properties": {"shape": {"type": "string", "description": "published name of the shape to fillet"}, "edges": {"description": "\"all\" or a list of edge subshape ids", "oneOf": [{"type": "string", "enum": ["all"]}, {"type": "array", "items": {"type": "integer"}}]}, "where": {"description": "instead of `edges`, select the edges inline: same predicate as get_subshapes -- {\"sorted_by\":\"z\",\"take\":\"max\",\"count\":4}, {\"by_size\":\"longest\",\"count\":N}, or {\"on_plane\":{\"point\":[x,y,z],\"normal\":[x,y,z]},\"state\":\"on\"}"}, "radius": {"type": "number"}}, "required": ["shape", "radius"]},
      arg_types={"radius": float})
def tool_fillet_edges(executor, args: dict):
    geompy = executor.namespace["geompy"]
    shape = executor.geom_bindings[args["shape"]]
    try:
        edge_ids = _edge_ids_for("fillet_edges", executor, args, shape)
    except ValueError as e:
        return {"ok": False, "error": str(e)}
    # geompy.ShapeType["EDGE"] evaluates to 6 (hardcoded literal for result.py)
    edge_type = 6  # geompy.ShapeType["EDGE"]
    try:
        filleted = geompy.MakeFillet(shape, args["radius"], edge_type, edge_ids)
    except Exception as e:  # noqa: BLE001 — geompy/CORBA errors are opaque; add context
        # OCC also segfaults (caught by GEOM) on a radius too big for a thin rim.
        radius_trouble = any(k in str(e) for k in ("radius", "SIGSEGV"))
        advice = (_fillet_advice(geompy, shape, edge_ids) if radius_trouble
                  else "try a smaller radius or check the edge ids")
        return {"ok": False, "error": f"fillet_edges: MakeFillet failed ({e}); {advice}"}
    # theReturnStatus=1 -> (isValid, description); the default (0) returns
    # a bare bool, which broke unpacking here until caught in testing.
    is_valid, msg = _check_shape(geompy, filleted)
    if not is_valid:
        # Not published yet at this point, so there is nothing to clean
        # up — the invariant decision 16 wants (no orphan survives a
        # failed call) holds by construction rather than by an explicit
        # delete step.
        return {"ok": False, "error": f"fillet_edges: CheckShape failed on result ({msg})"}
    name = _next_name(executor, "fillet")
    _publish(executor, filleted, name)
    return {
        "ok": True,
        "records": [{
            "fn": "MakeFillet",
            "args": [{"kind": "ref", "value": args["shape"]},
                     {"kind": "literal", "value": args["radius"]},
                     {"kind": "literal", "value": edge_type},
                     {"kind": "literal", "value": edge_ids}],
            "out_vars": [name],
            "publish": [name],
        }],
        "sanity": {"ok": True},
    }


@tool("chamfer_edges", category="feature",
      description='Bevel edges of `shape` by distances d1/d2. Select edges with `edges` ("all" or a list of edge ids) or inline with a `where` predicate (same as get_subshapes).',
      schema={"type": "object", "properties": {"shape": {"type": "string", "description": "published name of the shape to chamfer"}, "d1": {"type": "number", "description": "first chamfer distance"}, "d2": {"type": "number", "description": "second chamfer distance (defaults to d1)"}, "edges": {"description": "\"all\" or a list of edge subshape ids", "oneOf": [{"type": "string", "enum": ["all"]}, {"type": "array", "items": {"type": "integer"}}]}, "where": {"description": "instead of `edges`, select the edges inline: same predicate as get_subshapes"}}, "required": ["shape", "d1"]},
      arg_types={"d1": float, "d2": float})
def tool_chamfer_edges(executor, args: dict):
    geompy = executor.namespace["geompy"]
    shape = executor.geom_bindings[args["shape"]]
    d1 = args["d1"]
    d2 = args.get("d2", d1)
    try:
        edge_ids = _edge_ids_for("chamfer_edges", executor, args, shape)
    except ValueError as e:
        return {"ok": False, "error": str(e)}
    # MakeChamferEdges(theShape, theD1, theD2, theEdges) -- no shape-type arg,
    # unlike MakeFillet.
    try:
        chamfered = geompy.MakeChamferEdges(shape, d1, d2, edge_ids)
    except Exception as e:  # noqa: BLE001 — geompy/CORBA errors are opaque; add context
        return {"ok": False, "error": f"chamfer_edges: MakeChamferEdges failed ({e}); "
                f"try smaller distances or check the edge ids"}
    is_valid, msg = _check_shape(geompy, chamfered)
    if not is_valid:
        return {"ok": False, "error": f"chamfer_edges: CheckShape failed on result ({msg})"}
    name = _next_name(executor, "chamfer")
    _publish(executor, chamfered, name)
    return {
        "ok": True,
        "records": [{
            "fn": "MakeChamferEdges",
            "args": [{"kind": "ref", "value": args["shape"]},
                     {"kind": "literal", "value": d1},
                     {"kind": "literal", "value": d2},
                     {"kind": "literal", "value": edge_ids}],
            "out_vars": [name],
            "publish": [name],
        }],
        "sanity": {"ok": True},
    }
