from executor.smesh.state import (
    MeshInfo, group_ref, mesh_of, new_name, record, run, shape_ref, state, tool)


@tool("create_mesh", category="setup",
      description=("An empty mesh of `shape` (a published solid, face, or group of them). "
                   "The reply gives its name."),
      schema={"type": "object", "properties": {"shape": {"type": "string"}},
              "required": ["shape"]})
def tool_create_mesh(executor, args: dict):
    shape = args.get("shape")
    expr, owner = shape_ref(executor, shape)
    st = state(executor)
    m = new_name(executor, "mesh")
    code = f"{m} = smesh.Mesh({expr}, {m!r})"
    run(st, code)
    st.meshes[m] = MeshInfo(shape, owner)
    return {"ok": True, "records": [record("Mesh", code, out=[m], refs=[shape])],
            "name": m, "shape": shape}


FINENESS = {"very_coarse": "VeryCoarse", "coarse": "Coarse", "moderate": "Moderate",
            "fine": "Fine", "very_fine": "VeryFine"}

RESET_CODE = ("for _h in list({m}.GetHypothesisList({m}.GetShape())):\n"
              "    {m}.RemoveHypothesis(_h)")


def _pos(args: dict, key: str, required: bool = True):
    v = args.get(key)
    if v is None and not required:
        return None
    if isinstance(v, bool) or not isinstance(v, (int, float)) or v <= 0:
        raise ValueError(f"{key} must be a positive number, got {v!r}")
    return v


def _netgen_options(p: str, args: dict) -> list:
    lines = [f"{p}.SetMaxSize({_pos(args, 'max_size')})"]
    if _pos(args, "min_size", False) is not None:
        lines.append(f"{p}.SetMinSize({args['min_size']})")
    if args.get("fineness") is not None:
        if args["fineness"] not in FINENESS:
            raise ValueError(f"fineness must be one of {', '.join(FINENESS)}")
        lines.append(f"{p}.SetFineness(smeshBuilder.{FINENESS[args['fineness']]})")
    if _pos(args, "growth_rate", False) is not None:
        lines.append(f"{p}.SetGrowthRate({args['growth_rate']})")
    return lines


def tetra_code(m: str, args: dict) -> str:
    p = f"{m}_netgen"
    lines = [f"{m}_a3 = {m}.Tetrahedron(algo=smeshBuilder.NETGEN_1D2D3D)",
             f"{p} = {m}_a3.Parameters()", *_netgen_options(p, args)]
    if args.get("second_order") is not None:
        lines.append(f"{p}.SetSecondOrder({bool(args['second_order'])})")
    return "\n".join(lines)


def surface_code(m: str, args: dict) -> str:
    p = f"{m}_netgen"
    lines = [f"{m}_a2 = {m}.Triangle(algo=smeshBuilder.NETGEN_1D2D)",
             f"{p} = {m}_a2.Parameters()", *_netgen_options(p, args)]
    if args.get("quads") is not None:
        lines.append(f"{p}.SetQuadAllowed({bool(args['quads'])})")
    return "\n".join(lines)


def hexa_code(m: str, args: dict) -> str:
    if (args.get("segments") is None) == (args.get("segment_length") is None):
        raise ValueError("give either segments or segment_length")
    if args.get("segments") is not None:
        n = args["segments"]
        if isinstance(n, bool) or not isinstance(n, int) or n < 1:
            raise ValueError(f"segments must be a whole number >= 1, got {n!r}")
        one = f"{m}_a1 = {m}.Segment().NumberOfSegments({n})"
    else:
        one = f"{m}_a1 = {m}.Segment().LocalLength({_pos(args, 'segment_length')})"
    return "\n".join([one, f"{m}_a2 = {m}.Quadrangle()", f"{m}_a3 = {m}.Hexahedron()"])


def body_fitted_code(m: str, args: dict) -> str:
    s = str(_pos(args, "size"))
    return f"{m}_a3 = {m}.BodyFitted().SetGrid({s!r}, {s!r}, {s!r}, 4.0)"


# algorithm -> (dimension, the Mesh method and its algo argument)
ALGORITHMS = {
    "segment": (1, "Segment", None),
    "composite_segment": (1, "Segment", "smeshBuilder.COMPOSITE"),
    "netgen_1d2d": (2, "Triangle", "smeshBuilder.NETGEN_1D2D"),
    "netgen_2d": (2, "Triangle", "smeshBuilder.NETGEN_2D"),
    "quadrangle": (2, "Quadrangle", None),
    "quadrangle_medial_axis": (2, "Quadrangle", "smeshBuilder.QUAD_MA_PROJ"),
    "radial_quadrangle": (2, "Quadrangle", "smeshBuilder.RADIAL_QUAD"),
    "polygon_per_face": (2, "Polygon", None),
    "netgen_1d2d3d": (3, "Tetrahedron", "smeshBuilder.NETGEN_1D2D3D"),
    "netgen_3d": (3, "Tetrahedron", "smeshBuilder.NETGEN_3D"),
    "hexahedron": (3, "Hexahedron", None),
    "prism": (3, "Prism", None),
    "radial_prism": (3, "Prism", None),
    "body_fitting": (3, "BodyFitted", None),
    "polyhedron_per_solid": (3, "Polyhedron", None),
}

# kind -> (dimension, method, [(param, required)], how the params become args)
HYPOTHESES = {
    "number_of_segments": (1, "NumberOfSegments", [("count", True), ("ratio", False)]),
    "local_length": (1, "LocalLength", [("length", True)]),
    "arithmetic_progression": (1, "Arithmetic1D", [("start", True), ("end", True)]),
    "geometric_progression": (1, "GeometricProgression", [("start", True), ("ratio", True)]),
    "start_end_length": (1, "StartEndLength", [("start", True), ("end", True)]),
    "propagation": (1, "Propagation", []),
    "quadratic_mesh": (1, "QuadraticMesh", []),
    "max_element_area": (2, "MaxElementArea", [("area", True)]),
    "quadrangle_parameters": (2, "QuadrangleParameters", [("type", True)]),
    "quadrangle_preference": (2, "QuadranglePreference", []),
    "max_element_volume": (3, "MaxElementVolume", [("volume", True)]),
}
QUAD_TYPES = {"standard": "QUAD_STANDARD", "triangle_preference": "QUAD_TRIANGLE_PREF",
              "quadrangle_preference": "QUAD_QUADRANGLE_PREF", "reduced": "QUAD_REDUCED"}


def algorithm_code(var: str, m: str, algorithm: str, geom_expr) -> str:
    if algorithm not in ALGORITHMS:
        raise ValueError(f"algorithm must be one of {', '.join(ALGORITHMS)}")
    _, method, algo = ALGORITHMS[algorithm]
    args = [a for a in (f"algo={algo}" if algo else None,
                        f"geom={geom_expr}" if geom_expr else None) if a]
    return f"{var} = {m}.{method}({', '.join(args)})"


def hypothesis_code(algo_var: str, kind: str, params: dict) -> str:
    if kind not in HYPOTHESES:
        raise ValueError(f"kind must be one of {', '.join(HYPOTHESES)}")
    _, method, spec = HYPOTHESES[kind]
    params = params or {}
    vals = []
    for key, required in spec:
        if params.get(key) is None:
            if required:
                raise ValueError(f"{kind} needs {key}")
            continue
        if key == "type":
            if params[key] not in QUAD_TYPES:
                raise ValueError(f"type must be one of {', '.join(QUAD_TYPES)}")
            vals.append(f"StdMeshers.{QUAD_TYPES[params[key]]}")
        elif key == "count":
            vals.append(str(int(params[key])))
        else:
            vals.append(repr(_pos(params, key)))
    return f"{algo_var}.{method}({', '.join(vals)})"


def _preset(executor, args: dict, setup: str, dim: int, build) -> dict:
    m = args.get("mesh")
    info = mesh_of(executor, m)
    code = build(m, args)
    if info.setup is not None:
        code = RESET_CODE.format(m=m) + "\n" + code
    st = state(executor)
    run(st, code)
    info.setup, info.dim = setup, dim
    info.netgen = f"{m}_netgen" if setup in ("mesh_tetra", "mesh_surface") else None
    info.algos = {(d, None): f"{m}_a{d}" for d in (1, 2, 3) if f"{m}_a{d}" in code}
    return {"ok": True, "records": [record(setup, code, recv=m)], "mesh": m,
            "setup": setup}


_MESH = {"mesh": {"type": "string"}}


@tool("mesh_tetra", category="setup",
      description=("Mesh `mesh` with tetrahedra (NETGEN, edges, faces and volume in one "
                   "algorithm). `max_size` is the largest element edge; `fineness` "
                   "(very_coarse … very_fine) how closely curved faces are followed; "
                   "`growth_rate` how fast sizes grow away from small features; "
                   "`second_order` adds mid-edge nodes."),
      schema={"type": "object", "properties": {
          **_MESH, "max_size": {"type": "number"}, "min_size": {"type": "number"},
          "fineness": {"type": "string", "enum": list(FINENESS)},
          "growth_rate": {"type": "number"}, "second_order": {"type": "boolean"}},
          "required": ["mesh", "max_size"]})
def tool_mesh_tetra(executor, args: dict):
    return _preset(executor, args, "mesh_tetra", 3, tetra_code)


@tool("mesh_surface", category="setup",
      description=("Mesh only the faces of `mesh` with triangles (NETGEN), or mostly "
                   "quadrangles with `quads`. No volume elements."),
      schema={"type": "object", "properties": {
          **_MESH, "max_size": {"type": "number"}, "min_size": {"type": "number"},
          "quads": {"type": "boolean"}}, "required": ["mesh", "max_size"]})
def tool_mesh_surface(executor, args: dict):
    return _preset(executor, args, "mesh_surface", 2, surface_code)


@tool("mesh_hexa", category="setup",
      description=("Structured hexahedra: edges cut into `segments` (or pieces of "
                   "`segment_length`), four-sided faces into a quadrangle grid, block "
                   "solids into hexahedra. Faces that are not four-sided and solids that "
                   "are not blocks fail, named in the compute reply."),
      schema={"type": "object", "properties": {
          **_MESH, "segments": {"type": "integer"}, "segment_length": {"type": "number"}},
          "required": ["mesh"]})
def tool_mesh_hexa(executor, args: dict):
    return _preset(executor, args, "mesh_hexa", 3, hexa_code)


@tool("mesh_body_fitted", category="setup",
      description=("Hexahedra on a Cartesian grid of step `size`; cells cut by the "
                   "boundary become polyhedra. Any solid, no face mesh needed."),
      schema={"type": "object", "properties": {**_MESH, "size": {"type": "number"}},
              "required": ["mesh", "size"]})
def tool_mesh_body_fitted(executor, args: dict):
    return _preset(executor, args, "mesh_body_fitted", 3, body_fitted_code)


def _on(executor, on) -> list:
    if on is None:
        return [None]
    if isinstance(on, str):
        on = [on]
    if not isinstance(on, list) or not on:
        raise ValueError(f"on must be a list of group names, got {on!r}")
    for g in on:
        group_ref(executor, g)
    return on


@tool("set_algorithm", category="setup",
      description=("Set the algorithm for one dimension (1 edges, 2 faces, 3 volumes) of "
                   "`mesh`, or only on the named groups in `on`."),
      schema={"type": "object", "properties": {
          **_MESH, "dimension": {"type": "integer", "enum": [1, 2, 3]},
          "algorithm": {"type": "string", "enum": list(ALGORITHMS)},
          "on": {"type": "array", "items": {"type": "string"}}},
          "required": ["mesh", "dimension", "algorithm"]})
def tool_set_algorithm(executor, args: dict):
    m, algorithm = args.get("mesh"), args.get("algorithm")
    info = mesh_of(executor, m)
    if algorithm not in ALGORITHMS:
        raise ValueError(f"algorithm must be one of {', '.join(ALGORITHMS)}")
    dim = ALGORITHMS[algorithm][0]
    if args.get("dimension") != dim:
        raise ValueError(f"{algorithm} meshes dimension {dim}, not {args.get('dimension')!r}")
    st = state(executor)
    lines, refs = [], []
    for g in _on(executor, args.get("on")):
        var = f"{m}_a{dim}" if g is None else f"{m}_a{dim}_{g}"
        lines.append(algorithm_code(var, m, algorithm, group_ref(executor, g)[0] if g else None))
        info.algos[(dim, g)] = var
        if g is not None:
            info.submeshes[g] = var
            refs.append(g)
    code = "\n".join(lines)
    run(st, code)
    info.setup = info.setup or "custom"
    info.dim = max(info.dim or 0, dim) if args.get("on") is None else info.dim
    return {"ok": True, "records": [record("set_algorithm", code, recv=m, refs=refs)],
            "mesh": m}


@tool("set_hypothesis", category="setup",
      description="Set a sizing rule on `mesh`, or only on the named groups in `on`.",
      schema={"type": "object", "properties": {
          **_MESH, "kind": {"type": "string", "enum": list(HYPOTHESES)},
          "params": {"type": "object", "description": (
              "number_of_segments {count, ratio?}; local_length {length}; "
              "arithmetic_progression {start, end}; geometric_progression {start, ratio}; "
              "start_end_length {start, end}; max_element_area {area}; "
              "max_element_volume {volume}; quadrangle_parameters {type: standard|"
              "triangle_preference|quadrangle_preference|reduced}; propagation, "
              "quadratic_mesh, quadrangle_preference {}")},
          "on": {"type": "array", "items": {"type": "string"}}},
          "required": ["mesh", "kind"]})
def tool_set_hypothesis(executor, args: dict):
    m, kind = args.get("mesh"), args.get("kind")
    info = mesh_of(executor, m)
    if kind not in HYPOTHESES:
        raise ValueError(f"kind must be one of {', '.join(HYPOTHESES)}")
    dim = HYPOTHESES[kind][0]
    lines, refs = [], []
    for g in _on(executor, args.get("on")):
        var = info.algos.get((dim, g))
        if var is None:
            where = "the mesh" if g is None else g
            raise ValueError(f"{kind} needs a dimension-{dim} algorithm on {where}: call "
                             f"set_algorithm(dimension={dim}{'' if g is None else ', on=[' + repr(g) + ']'}) first")
        lines.append(hypothesis_code(var, kind, args.get("params") or {}))
        if g is not None:
            refs.append(g)
    code = "\n".join(lines)
    try:
        run(state(executor), code)
    except AttributeError:
        raise ValueError(f"the dimension-{dim} algorithm there does not take {kind}") from None
    return {"ok": True, "records": [record("set_hypothesis", code, recv=m, refs=refs)],
            "mesh": m}


def prism_code(m: str, end_expr: str, args: dict) -> str:
    n = args.get("layers")
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError(f"layers must be a whole number >= 1, got {n!r}")
    size = _pos(args, "face_size")
    return "\n".join([f"{m}_a1 = {m}.Segment().NumberOfSegments({n})",
                      f"{m}_a3 = {m}.Prism()",
                      f"{m}_a1_end = {m}.Segment(geom={end_expr})",
                      f"{m}_a1_end.LocalLength({size!r})",
                      f"{m}_a2_end = {m}.Triangle(algo=smeshBuilder.NETGEN_2D, geom={end_expr})"])


@tool("mesh_prism", category="setup",
      description=("Sweep a triangle mesh of one end face of an extruded solid through it "
                   "into prisms, in `layers` layers; `end_face` is a named group of that "
                   "face, `face_size` the element size on it. The end faces must match."),
      schema={"type": "object", "properties": {
          **_MESH, "end_face": {"type": "string"}, "layers": {"type": "integer"},
          "face_size": {"type": "number"}},
          "required": ["mesh", "end_face", "layers", "face_size"]})
def tool_mesh_prism(executor, args: dict):
    end = args.get("end_face")
    gexpr = group_ref(executor, end)[0]
    r = _preset(executor, args, "mesh_prism", 3, lambda m, a: prism_code(m, gexpr, a))
    info = mesh_of(executor, args["mesh"])
    info.submeshes[end] = f"{args['mesh']}_a1_end"
    r["records"][0]["args"] = [{"kind": "expr", "value": "", "refs": [end]}]
    return r
