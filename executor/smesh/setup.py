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


MESHERS = ("netgen", "gmsh")


def _mesher(args: dict) -> str:
    mesher = args.get("mesher") or "netgen"
    if mesher not in MESHERS:
        raise ValueError(f"mesher must be one of {', '.join(MESHERS)}")
    if mesher == "gmsh":
        given = [k for k in ("fineness", "growth_rate") if args.get(k) is not None]
        if given:
            raise ValueError(f"{' and '.join(given)} are NETGEN options; gmsh takes "
                             "max_size, min_size and second_order")
    return mesher


def _gmsh_options(p: str, args: dict) -> list:
    lines = [f"{p}.SetMaxSize({_pos(args, 'max_size')})"]
    if _pos(args, "min_size", False) is not None:
        lines.append(f"{p}.SetMinSize({args['min_size']})")
    return lines


def tetra_code(m: str, args: dict) -> str:
    if _mesher(args) == "gmsh":
        p = f"{m}_gmsh"
        lines = [f"{m}_a3 = {m}.Tetrahedron(algo=smeshBuilder.GMSH)",
                 f"{p} = {m}_a3.Parameters()", *_gmsh_options(p, args)]
    else:
        p = f"{m}_netgen"
        lines = [f"{m}_a3 = {m}.Tetrahedron(algo=smeshBuilder.NETGEN_1D2D3D)",
                 f"{p} = {m}_a3.Parameters()", *_netgen_options(p, args)]
    if args.get("second_order") is not None:
        lines.append(f"{p}.SetSecondOrder({bool(args['second_order'])})")
    return "\n".join(lines)


def surface_code(m: str, args: dict) -> str:
    if _mesher(args) == "gmsh":
        p = f"{m}_gmsh"
        lines = [f"{m}_a2 = {m}.Triangle(algo=smeshBuilder.GMSH_2D)",
                 f"{p} = {m}_a2.Parameters()", *_gmsh_options(p, args)]
        if args.get("quads") is not None:
            lines.append(f"{p}.SetRecombineAll({bool(args['quads'])})")
        return "\n".join(lines)
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
        one = f"{m}_a1.NumberOfSegments({n})"
    else:
        one = f"{m}_a1.LocalLength({_pos(args, 'segment_length')})"
    # {m}_a1 stays the algorithm: set_hypothesis adds to it later
    return "\n".join([f"{m}_a1 = {m}.Segment()", one, f"{m}_a2 = {m}.Quadrangle()",
                      f"{m}_a3 = {m}.Hexahedron()"])


def body_fitted_code(m: str, args: dict) -> str:
    s = str(_pos(args, "size"))
    return f"{m}_a3 = {m}.BodyFitted().SetGrid({s!r}, {s!r}, {s!r}, 4.0)"


# algorithm -> (dimension, the Mesh method and its algo argument)
ALGORITHMS = {
    "segment": (1, "Segment", None),
    "composite_segment": (1, "Segment", "smeshBuilder.COMPOSITE"),
    "netgen_1d2d": (2, "Triangle", "smeshBuilder.NETGEN_1D2D"),
    "netgen_2d": (2, "Triangle", "smeshBuilder.NETGEN_2D"),
    "gmsh_2d": (2, "Triangle", "smeshBuilder.GMSH_2D"),
    "quadrangle": (2, "Quadrangle", None),
    "quadrangle_medial_axis": (2, "Quadrangle", "smeshBuilder.QUAD_MA_PROJ"),
    "radial_quadrangle": (2, "Quadrangle", "smeshBuilder.RADIAL_QUAD"),
    "polygon_per_face": (2, "Polygon", None),
    "netgen_1d2d3d": (3, "Tetrahedron", "smeshBuilder.NETGEN_1D2D3D"),
    "netgen_3d": (3, "Tetrahedron", "smeshBuilder.NETGEN_3D"),
    "gmsh_1d2d3d": (3, "Tetrahedron", "smeshBuilder.GMSH"),
    "gmsh_3d": (3, "Tetrahedron", "smeshBuilder.GMSH_3D"),
    "hexahedron": (3, "Hexahedron", None),
    "prism": (3, "Prism", None),
    "radial_prism": (3, "Prism", None),
    "body_fitting": (3, "BodyFitted", None),
    "polyhedron_per_solid": (3, "Polyhedron", None),
}

# kind -> (dimension, or the dimensions it may sit on, first found wins; method;
# [(param, required)] in the method's argument order)
HYPOTHESES = {
    "number_of_segments": (1, "NumberOfSegments", [("count", True), ("ratio", False)]),
    "local_length": (1, "LocalLength", [("length", True)]),
    "max_size": (1, "MaxSize", [("length", True)]),
    "arithmetic_progression": (1, "Arithmetic1D", [("start", True), ("end", True)]),
    "geometric_progression": (1, "GeometricProgression", [("start", True), ("ratio", True)]),
    "start_end_length": (1, "StartEndLength", [("start", True), ("end", True)]),
    "fixed_points": (1, "FixedPoints1D", [("points", True), ("segments", False)]),
    "adaptive": (1, "Adaptive", [("min", True), ("max", True), ("deflection", True)]),
    "deflection": (1, "Deflection1D", [("deflection", True)]),
    "automatic_length": (1, "AutomaticLength", [("fineness", True)]),
    "propagation": (1, "Propagation", []),
    "propagation_of_distribution": (1, "PropagationOfDistribution", []),
    "quadratic_mesh": (1, "QuadraticMesh", []),
    "max_element_area": (2, "MaxElementArea", [("area", True)]),
    "quadrangle_parameters": (2, "QuadrangleParameters", [("type", True)]),
    "quadrangle_preference": (2, "QuadranglePreference", []),
    "max_element_volume": (3, "MaxElementVolume", [("volume", True)]),
    "number_of_layers": ((3, 2), "NumberOfLayers", [("count", True)]),
    "netgen_parameters": ((3, 2), "Parameters", []),
    "gmsh_parameters": ((3, 2), "Parameters", []),
}
QUAD_TYPES = {"standard": "QUAD_STANDARD", "triangle_preference": "QUAD_TRIANGLE_PREF",
              "quadrangle_preference": "QUAD_QUADRANGLE_PREF", "reduced": "QUAD_REDUCED"}
# kinds that take `reversed`: edges meshed from their other end
REVERSIBLE = frozenset({"number_of_segments", "arithmetic_progression",
                        "geometric_progression", "start_end_length", "fixed_points"})
# kind -> the algorithms (GetName() prefixes) that take it; any other is refused
# before it runs: NumberOfLayers on a plain Prism silently does nothing.
NEEDS_ALGORITHM = {"number_of_layers": ("RadialPrism_3D", "RadialQuadrangle_1D2D"),
                   "netgen_parameters": ("NETGEN",), "gmsh_parameters": ("GMSH",)}

# parameters of a NETGEN / GMSH algorithm: param -> (setter, value kind)
NETGEN_PARAMS = {"max_size": ("SetMaxSize", "pos"), "min_size": ("SetMinSize", "pos"),
                 "fineness": ("SetFineness", "fineness"),
                 "growth_rate": ("SetGrowthRate", "pos"),
                 "chordal_error": ("SetChordalError", "pos"),
                 "second_order": ("SetSecondOrder", "bool"),
                 "quads": ("SetQuadAllowed", "bool"), "optimize": ("SetOptimize", "bool")}
GMSH_2D_ALGOS = {"automatic": 0, "meshadapt": 1, "delaunay": 2, "frontal": 3,
                 "delaunay_for_quads": 4, "packing_parallelograms": 5, "quad_qs": 6}
GMSH_3D_ALGOS = {"delaunay": 0, "frontal": 1, "mmg3d": 2, "rtree": 3, "hxt": 4}
GMSH_PARAMS = {"max_size": ("SetMaxSize", "pos"), "min_size": ("SetMinSize", "pos"),
               "size_factor": ("SetSizeFactor", "pos"),
               "algorithm_2d": ("Set2DAlgo", GMSH_2D_ALGOS),
               "algorithm_3d": ("Set3DAlgo", GMSH_3D_ALGOS),
               "quads": ("SetRecombineAll", "bool"),
               "second_order": ("SetSecondOrder", "bool")}


def dims_of(kind: str) -> tuple:
    d = HYPOTHESES[kind][0]
    return d if isinstance(d, tuple) else (d,)


def _count(params: dict, key: str) -> int:
    n = params.get(key)
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError(f"{key} must be a whole number >= 1, got {n!r}")
    return n


def _fixed_points(params: dict) -> list:
    pts, segs = params.get("points"), params.get("segments")
    if (not isinstance(pts, list) or not pts or any(
            isinstance(t, bool) or not isinstance(t, (int, float)) or not 0 < t < 1
            for t in pts) or sorted(pts) != pts):
        raise ValueError(f"points must be increasing fractions of the edge length, "
                         f"strictly between 0 and 1, got {pts!r}")
    vals = [repr(list(pts))]
    if segs is not None:
        if (not isinstance(segs, list) or len(segs) != len(pts) + 1 or any(
                isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in segs)):
            raise ValueError(f"segments must be {len(pts) + 1} whole numbers >= 1 (one per "
                             f"stretch between the points), got {segs!r}")
        vals.append(repr(list(segs)))
    return vals


def parameters_code(algo_var: str, kind: str, params: dict) -> str:
    table = NETGEN_PARAMS if kind == "netgen_parameters" else GMSH_PARAMS
    unknown = sorted(set(params) - set(table))
    if unknown or not params:
        raise ValueError(f"{kind} takes some of {', '.join(table)}"
                         + (f"; not {', '.join(unknown)}" if unknown else ""))
    p = f"{algo_var}_p"
    lines = [f"{p} = {algo_var}.Parameters()"]
    for key in table:
        if params.get(key) is None:
            continue
        setter, how = table[key]
        v = params[key]
        if how == "pos":
            val = repr(_pos(params, key))
        elif how == "bool":
            if not isinstance(v, bool):
                raise ValueError(f"{key} must be true or false, got {v!r}")
            val = str(v)
        elif how == "fineness":
            if v not in FINENESS:
                raise ValueError(f"fineness must be one of {', '.join(FINENESS)}")
            val = f"smeshBuilder.{FINENESS[v]}"
        else:
            if v not in how:
                raise ValueError(f"{key} must be one of {', '.join(how)}")
            val = str(how[v])
        if key == "chordal_error":
            lines.append(f"{p}.SetChordalErrorEnabled(True)")
        lines.append(f"{p}.{setter}({val})")
    return "\n".join(lines)


def algorithm_code(var: str, m: str, algorithm: str, geom_expr) -> str:
    if algorithm not in ALGORITHMS:
        raise ValueError(f"algorithm must be one of {', '.join(ALGORITHMS)}")
    _, method, algo = ALGORITHMS[algorithm]
    args = [a for a in (f"algo={algo}" if algo else None,
                        f"geom={geom_expr}" if geom_expr else None) if a]
    return f"{var} = {m}.{method}({', '.join(args)})"


def hypothesis_code(algo_var: str, kind: str, params: dict, reversed_exprs=()) -> str:
    if kind not in HYPOTHESES:
        raise ValueError(f"kind must be one of {', '.join(HYPOTHESES)}")
    params = dict(params or {})
    params.pop("reversed", None)
    if kind in ("netgen_parameters", "gmsh_parameters"):
        return parameters_code(algo_var, kind, params)
    _, method, spec = HYPOTHESES[kind]
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
            vals.append(str(_count(params, key)))
        elif key == "points":
            vals.extend(_fixed_points(params))
        elif key == "segments":
            continue  # with points
        elif key == "fineness":
            f = params[key]
            if isinstance(f, bool) or not isinstance(f, (int, float)) or not 0 <= f <= 1:
                raise ValueError(f"fineness must be a number from 0 (coarse) to 1 (fine), "
                                 f"got {f!r}")
            vals.append(repr(f))
        else:
            vals.append(repr(_pos(params, key)))
    if kind == "adaptive" and params["min"] >= params["max"]:
        raise ValueError("adaptive needs min below max")
    if reversed_exprs:
        if kind not in REVERSIBLE:
            raise ValueError(f"{kind} takes no reversed edges; only "
                             f"{', '.join(sorted(REVERSIBLE))} do")
        vals.append("reversedEdges=" + " + ".join(f"_ids({e})" for e in reversed_exprs))
    return f"{algo_var}.{method}({', '.join(vals)})"


# preset -> dimensions of the algorithms it sets on the whole mesh, {m}_a<d>
_PRESET_DIMS = {"mesh_tetra": (3,), "mesh_surface": (2,), "mesh_hexa": (1, 2, 3),
                "mesh_body_fitted": (3,), "mesh_prism": (1, 3)}


def _preset(executor, args: dict, setup: str, dim: int, build) -> dict:
    m = args.get("mesh")
    info = mesh_of(executor, m)
    code = build(m, args)
    if info.setup is not None:
        code = RESET_CODE.format(m=m) + "\n" + code
    mesher = _mesher(args) if setup in ("mesh_tetra", "mesh_surface") else None
    netgen = f"{m}_netgen" if mesher == "netgen" else None
    # The reset removes the NETGEN local sizes with the rest: a NETGEN preset
    # sets them again, any other drops them and says so.
    kept, dropped = (info.local_sizes, []) if netgen else ({}, list(info.local_sizes))
    for g, size in kept.items():
        code += "\n" + f"{netgen}.SetLocalSizeOnShape({group_ref(executor, g)[0]}, {size!r})"
    st = state(executor)
    run(st, code)
    info.setup, info.dim, info.netgen, info.local_sizes = setup, dim, netgen, dict(kept)
    info.mesher = mesher
    info.count_rule = None
    info.algos = {(d, None): f"{m}_a{d}" for d in _PRESET_DIMS[setup]}
    r = {"ok": True, "records": [record(setup, code, recv=m, refs=list(kept))], "mesh": m,
         "setup": setup}
    if dropped:
        r["dropped_local_sizes"] = dropped
    return r


_MESH = {"mesh": {"type": "string"}}


@tool("mesh_tetra", category="setup",
      description=("Mesh `mesh` with tetrahedra (edges, faces and volume in one "
                   "algorithm; `mesher` netgen, the default, or gmsh). `max_size` is the "
                   "largest element edge; NETGEN only: `fineness` (very_coarse … "
                   "very_fine) how closely curved faces are followed, `growth_rate` how "
                   "fast sizes grow away from small features. `second_order` adds "
                   "mid-edge nodes."),
      schema={"type": "object", "properties": {
          **_MESH, "max_size": {"type": "number"}, "min_size": {"type": "number"},
          "mesher": {"type": "string", "enum": list(MESHERS)},
          "fineness": {"type": "string", "enum": list(FINENESS)},
          "growth_rate": {"type": "number"}, "second_order": {"type": "boolean"}},
          "required": ["mesh", "max_size"]})
def tool_mesh_tetra(executor, args: dict):
    return _preset(executor, args, "mesh_tetra", 3, tetra_code)


@tool("mesh_surface", category="setup",
      description=("Mesh only the faces of `mesh` with triangles (`mesher` netgen, the "
                   "default, or gmsh), or mostly quadrangles with `quads`. No volume "
                   "elements."),
      schema={"type": "object", "properties": {
          **_MESH, "max_size": {"type": "number"}, "min_size": {"type": "number"},
          "mesher": {"type": "string", "enum": list(MESHERS)},
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
    r = _preset(executor, args, "mesh_body_fitted", 3, body_fitted_code)
    mesh_of(executor, args["mesh"]).count_rule = ("grid", float(args["size"]))
    return r


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
    info.count_rule = None
    info.dim = max(info.dim or 0, dim) if args.get("on") is None else info.dim
    return {"ok": True, "records": [record("set_algorithm", code, recv=m, refs=refs)],
            "mesh": m}


def algorithm_name(st, var) -> "str | None":
    """SMESH's name of the algorithm held by `var` (NETGEN_2D3D, RadialPrism_3D, ...),
    or None when the namespace does not hold one."""
    algo = st.ns.get(var)
    return algo.GetAlgorithm().GetName() if hasattr(algo, "GetAlgorithm") else None


@tool("set_hypothesis", category="setup",
      description=("Set a sizing rule on `mesh`, or only on the named groups in `on`. "
                   "1D rules go on a segment algorithm, area/quadrangle rules on a face "
                   "algorithm, volume rules on a volume algorithm; number_of_layers on "
                   "radial_prism or radial_quadrangle; netgen_parameters / "
                   "gmsh_parameters on a NETGEN / GMSH algorithm."),
      schema={"type": "object", "properties": {
          **_MESH, "kind": {"type": "string", "enum": list(HYPOTHESES)},
          "params": {"type": "object", "description": (
              "number_of_segments {count, ratio?}; local_length {length}; max_size "
              "{length}; arithmetic_progression {start, end}; geometric_progression "
              "{start, ratio}; start_end_length {start, end}; fixed_points {points: "
              "fractions 0-1 along the edge, segments?: one count per stretch}; adaptive "
              "{min, max, deflection}; deflection {deflection}; automatic_length "
              "{fineness 0-1}; max_element_area {area}; max_element_volume {volume}; "
              "quadrangle_parameters {type: standard|triangle_preference|"
              "quadrangle_preference|reduced}; number_of_layers {count}; "
              "netgen_parameters {max_size?, min_size?, fineness?, growth_rate?, "
              "chordal_error?, second_order?, quads?, optimize?}; gmsh_parameters "
              "{max_size?, min_size?, size_factor?, algorithm_2d?: automatic|meshadapt|"
              "delaunay|frontal|delaunay_for_quads|packing_parallelograms|quad_qs, "
              "algorithm_3d?: delaunay|frontal|mmg3d|rtree|hxt, quads?, second_order?}; "
              "propagation, propagation_of_distribution, quadratic_mesh, "
              "quadrangle_preference {}. The graded 1D rules and fixed_points also take "
              "reversed: [edge groups] meshed from their other end.")},
          "on": {"type": "array", "items": {"type": "string"}}},
          "required": ["mesh", "kind"]})
def tool_set_hypothesis(executor, args: dict):
    m, kind = args.get("mesh"), args.get("kind")
    info = mesh_of(executor, m)
    if kind not in HYPOTHESES:
        raise ValueError(f"kind must be one of {', '.join(HYPOTHESES)}")
    dims = dims_of(kind)
    params = args.get("params") or {}
    if not isinstance(params, dict):
        raise ValueError(f"params must be an object, got {params!r}")
    rev = params.get("reversed") or []
    if isinstance(rev, str):
        rev = [rev]
    rev_exprs = [group_ref(executor, g)[0] for g in rev]
    st = state(executor)
    lines, refs = [], list(rev)
    for g in _on(executor, args.get("on")):
        var = next((info.algos[(d, g)] for d in dims if (d, g) in info.algos), None)
        if var is None:
            where = "the mesh" if g is None else g
            on = "" if g is None else ", on=[" + repr(g) + "]"
            raise ValueError(f"{kind} needs a dimension-{' or '.join(map(str, dims))} "
                             f"algorithm on {where}: call set_algorithm(dimension="
                             f"{dims[0]}{on}) first")
        name = algorithm_name(st, var)
        if kind in NEEDS_ALGORITHM and name is not None and not name.startswith(
                NEEDS_ALGORITHM[kind]):
            raise ValueError(f"{kind} does not apply to the {name} algorithm on "
                             f"{'the mesh' if g is None else g}")
        lines.append(hypothesis_code(var, kind, params, rev_exprs))
        if g is not None:
            refs.append(g)
    code = "\n".join(lines)
    try:
        run(st, code)
    except AttributeError:
        raise ValueError(f"the algorithm there does not take {kind}") from None
    info.count_rule = None
    return {"ok": True, "records": [record("set_hypothesis", code, recv=m, refs=refs)],
            "mesh": m}


def prism_code(m: str, end_expr: str, args: dict) -> str:
    n = args.get("layers")
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError(f"layers must be a whole number >= 1, got {n!r}")
    size = _pos(args, "face_size")
    lines = [f"{m}_a1 = {m}.Segment()",
             f"{m}_a1.NumberOfSegments({n})",
             f"{m}_a3 = {m}.Prism()",
             f"{m}_a1_end = {m}.Segment(geom={end_expr})",
             f"{m}_a1_end.LocalLength({size!r})",
             f"{m}_a2_end = {m}.Triangle(algo=smeshBuilder.NETGEN_2D, geom={end_expr})"]
    if args.get("quads"):
        lines.append(f"{m}_a2_end.SetQuadAllowed(True)")
    return "\n".join(lines)


@tool("mesh_prism", category="setup",
      description=("Sweep a triangle mesh of one end face of an extruded solid through it "
                   "into prisms, in `layers` layers; `end_face` is a named group of that "
                   "face, `face_size` the element size on it. With `quads` the end face "
                   "is mostly quadrangles, swept into hexahedra. The end faces must match."),
      schema={"type": "object", "properties": {
          **_MESH, "end_face": {"type": "string"}, "layers": {"type": "integer"},
          "face_size": {"type": "number"}, "quads": {"type": "boolean"}},
          "required": ["mesh", "end_face", "layers", "face_size"]})
def tool_mesh_prism(executor, args: dict):
    end = args.get("end_face")
    gexpr = group_ref(executor, end)[0]
    r = _preset(executor, args, "mesh_prism", 3, lambda m, a: prism_code(m, gexpr, a))
    info = mesh_of(executor, args["mesh"])
    info.submeshes[end] = f"{args['mesh']}_a1_end"
    info.algos[(1, end)] = f"{args['mesh']}_a1_end"
    info.algos[(2, end)] = f"{args['mesh']}_a2_end"
    info.count_rule = ("layers", args["layers"])
    r["records"][0]["args"] = [{"kind": "expr", "value": "", "refs": [end]}]
    return r
