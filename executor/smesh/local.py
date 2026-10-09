from executor.smesh.setup import _on, _pos
from executor.smesh.state import group_ref, mesh_of, record, run, state, tool

_MESH = {"mesh": {"type": "string"}}
_ON = {"type": "array", "items": {"type": "string"}}


def local_size_code(netgen, m, group, gexpr, size) -> str:
    if netgen:
        return f"{netgen}.SetLocalSizeOnShape({gexpr}, {size!r})"
    var = f"{m}_a1_{group}"
    return f"{var} = {m}.Segment(geom={gexpr})\n{var}.LocalLength({size!r})"


def edge_segments_code(var, m, gexpr, args) -> str:
    if (args.get("count") is None) == (args.get("length") is None):
        raise ValueError("give either count or length")
    lines = [f"{var} = {m}.Segment(geom={gexpr})"]
    if args.get("count") is not None:
        n = args["count"]
        if isinstance(n, bool) or not isinstance(n, int) or n < 1:
            raise ValueError(f"count must be a whole number >= 1, got {n!r}")
        ratio = _pos(args, "ratio", False)
        lines.append(f"{var}.NumberOfSegments({n}{'' if ratio is None else f', {ratio!r}'})")
    else:
        lines.append(f"{var}.LocalLength({_pos(args, 'length')!r})")
    if args.get("propagate"):
        lines.append(f"{var}.Propagation()")
    return "\n".join(lines)


def viscous_code(algo_var, dim, args, face_exprs) -> str:
    if args.get("faces_are") not in ("with_layers", "without_layers"):
        raise ValueError("faces_are must be with_layers or without_layers")
    n = args.get("layers")
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError(f"layers must be a whole number >= 1, got {n!r}")
    ids = " + ".join(f"_ids({e})" for e in face_exprs) or "[]"
    method = "ViscousLayers" if dim == 3 else "ViscousLayers2D"
    return (f"{algo_var}.{method}({_pos(args, 'thickness')!r}, {n}, "
            f"{_pos(args, 'stretch')!r}, {ids}, {args['faces_are'] == 'without_layers'})")


@tool("local_size", category="local",
      description="Element size `size` on the named groups in `on`, different from the mesh's own.",
      schema={"type": "object", "properties": {**_MESH, "on": _ON, "size": {"type": "number"}},
              "required": ["mesh", "on", "size"]})
def tool_local_size(executor, args: dict):
    m = args.get("mesh")
    info = mesh_of(executor, m)
    if info.setup is None:
        raise ValueError(f"{m} has no setup yet: call mesh_tetra, mesh_hexa, ... first")
    size = _pos(args, "size")
    lines, refs = [], []
    for g in _on(executor, args.get("on")):
        lines.append(local_size_code(info.netgen, m, g, group_ref(executor, g)[0], size))
        if not info.netgen:
            info.submeshes[g] = f"{m}_a1_{g}"
            info.algos[(1, g)] = f"{m}_a1_{g}"
        refs.append(g)
    code = "\n".join(lines)
    run(state(executor), code)
    return {"ok": True, "records": [record("local_size", code, recv=m, refs=refs)], "mesh": m}


@tool("edge_segments", category="local",
      description=("Cut the edges in `on` into `count` segments (or pieces of `length`); "
                   "`ratio` grades them from one end; `propagate` carries the same cut to "
                   "the opposite edges of four-sided faces."),
      schema={"type": "object", "properties": {
          **_MESH, "on": _ON, "count": {"type": "integer"}, "length": {"type": "number"},
          "ratio": {"type": "number"}, "propagate": {"type": "boolean"}},
          "required": ["mesh", "on"]})
def tool_edge_segments(executor, args: dict):
    m = args.get("mesh")
    info = mesh_of(executor, m)
    lines, refs = [], []
    for g in _on(executor, args.get("on")):
        var = f"{m}_a1_{g}"
        lines.append(edge_segments_code(var, m, group_ref(executor, g)[0], args))
        info.submeshes[g] = var
        info.algos[(1, g)] = var
        refs.append(g)
    code = "\n".join(lines)
    run(state(executor), code)
    return {"ok": True, "records": [record("edge_segments", code, recv=m, refs=refs)], "mesh": m}


# NETGEN 1D-2D-3D takes no viscous layers: on a mesh_tetra mesh the 3D part is
# split into NETGEN 1D-2D on the faces and NETGEN 3D in the volume first.
# The face parameters are copied from the tetra setup's, value by value.
_SPLIT_TETRA = ("{m}.RemoveHypothesis({m}_a3)\n"
                "{m}_a2 = {m}.Triangle(algo=smeshBuilder.NETGEN_1D2D)\n"
                "{m}_netgen2 = {m}_a2.Parameters()\n"
                "{m}_netgen2.SetMaxSize({max!r})\n"
                "{m}_netgen2.SetMinSize({min!r})\n"
                "{m}_netgen2.SetFineness({fineness!r})\n"
                "{m}_netgen2.SetGrowthRate({growth!r})\n"
                "{m}_a3 = {m}.Tetrahedron(algo=smeshBuilder.NETGEN_3D)")


@tool("viscous_layers", category="local",
      description=("`layers` thin layers of total `thickness`, each `stretch` times the one "
                   "before, along walls. `faces_are` says whether `faces` are the walls "
                   "that get layers or the ones left without (inlets, outlets)."),
      schema={"type": "object", "properties": {
          **_MESH, "thickness": {"type": "number"}, "layers": {"type": "integer"},
          "stretch": {"type": "number"}, "faces": _ON,
          "faces_are": {"type": "string", "enum": ["with_layers", "without_layers"]}},
          "required": ["mesh", "thickness", "layers", "stretch", "faces", "faces_are"]})
def tool_viscous_layers(executor, args: dict):
    m = args.get("mesh")
    info = mesh_of(executor, m)
    if info.dim not in (2, 3):
        raise ValueError(f"{m} has no setup yet: call mesh_tetra, mesh_surface, ... first")
    groups = _on(executor, args.get("faces"))
    lines = []
    if info.setup == "mesh_tetra" and (2, None) not in info.algos:
        p = state(executor).ns[info.netgen]
        lines.append(_SPLIT_TETRA.format(m=m, max=p.GetMaxSize(), min=p.GetMinSize(),
                                         fineness=p.GetFineness(), growth=p.GetGrowthRate()))
        info.algos[(2, None)] = f"{m}_a2"
        info.netgen = f"{m}_netgen2"
    algo = info.algos.get((info.dim, None))
    if algo is None:
        raise ValueError(f"{m} has no dimension-{info.dim} algorithm to carry the layers")
    lines.append(viscous_code(algo, info.dim, args, [group_ref(executor, g)[0] for g in groups]))
    code = "\n".join(lines)
    run(state(executor), code)
    return {"ok": True, "records": [record("viscous_layers", code, recv=m, refs=groups)],
            "mesh": m}


@tool("set_submesh_order", category="local",
      description="Which local setup wins where two meet: names in priority order.",
      schema={"type": "object", "properties": {**_MESH, "order": _ON},
              "required": ["mesh", "order"]})
def tool_set_submesh_order(executor, args: dict):
    m = args.get("mesh")
    info = mesh_of(executor, m)
    order = args.get("order") or []
    missing = [g for g in order if g not in info.submeshes]
    if len(order) < 2 or missing:
        raise ValueError(f"order needs two or more of {m}'s local setups: "
                         f"{', '.join(info.submeshes) or 'none yet'}")
    subs = ", ".join(f"{info.submeshes[g]}.GetSubMesh()" for g in order)
    code = f"{m}.SetMeshOrder([[{subs}]])"
    run(state(executor), code)
    return {"ok": True, "records": [record("set_submesh_order", code, recv=m)], "mesh": m}
