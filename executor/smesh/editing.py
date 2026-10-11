import os

from executor.smesh.setup import HYPOTHESES, _on
from executor.smesh.state import group_ref, mesh_of, record, run, state, tool

_FORMATS = {"med": "ExportMED({path!r})", "unv": "ExportUNV({path!r})",
            "stl": "ExportSTL({path!r})", "cgns": "ExportCGNS({path!r})",
            "gmf": "ExportGMF({path!r})", "dat": "ExportDAT({path!r})"}
# format -> file extension, when it is not the format's name
_EXTENSIONS = {"gmf": "mesh"}
_MESH = {"type": "object", "properties": {"mesh": {"type": "string"}}, "required": ["mesh"]}


# kind -> SMESH's names (GetName() prefixes) of the hypotheses it sets, where they
# differ from the builder method that sets it
HYPOTHESIS_NAMES = {"max_size": ("MaxLength",), "adaptive": ("Adaptive1D",),
                    "propagation_of_distribution": ("PropagOfDistribution",),
                    "quadrangle_parameters": ("QuadrangleParams",),
                    "quadrangle_preference": ("QuadrangleParams", "QuadranglePreference"),
                    "netgen_parameters": ("NETGEN_Parameters", "NETGEN_SimpleParameters"),
                    "gmsh_parameters": ("GMSH_Parameters",)}


def hypothesis_names(kind: str) -> tuple:
    return HYPOTHESIS_NAMES.get(kind, (HYPOTHESES[kind][1],))


@tool("clear_mesh", category="edit",
      description="Remove the elements of `mesh`, keeping its setup.", schema=_MESH)
def tool_clear_mesh(executor, args: dict):
    m = args.get("mesh")
    mesh_of(executor, m)
    code = f"{m}.Clear()"
    run(state(executor), code)
    return {"ok": True, "records": [record("Clear", code, recv=m)], "mesh": m}


@tool("remove_hypothesis", category="edit",
      description="Remove a sizing rule set with set_hypothesis, local_size or edge_segments.",
      schema={"type": "object", "properties": {
          "mesh": {"type": "string"}, "kind": {"type": "string", "enum": list(HYPOTHESES)},
          "on": {"type": "array", "items": {"type": "string"}}},
          "required": ["mesh", "kind"]})
def tool_remove_hypothesis(executor, args: dict):
    m, kind = args.get("mesh"), args.get("kind")
    info = mesh_of(executor, m)
    if kind not in HYPOTHESES:
        raise ValueError(f"kind must be one of {', '.join(HYPOTHESES)}")
    names = hypothesis_names(kind)
    lines, refs = [], []
    for g in _on(executor, args.get("on")):
        geom = group_ref(executor, g)[0] if g else f"{m}.GetShape()"
        lines.append(f"for _h in list({m}.GetHypothesisList({geom})):\n"
                     f"    if _h.GetName().startswith({names!r}):\n"
                     f"        {m}.RemoveHypothesis(_h, {geom})")
        if g:
            refs.append(g)
    code = "\n".join(lines)
    run(state(executor), code)
    return {"ok": True, "records": [record("RemoveHypothesis", code, recv=m, refs=refs)],
            "mesh": m}


@tool("convert_to_quadratic", category="edit",
      description="Add mid-edge nodes to every element of a computed `mesh`.", schema=_MESH)
def tool_convert_to_quadratic(executor, args: dict):
    m = args.get("mesh")
    mesh_of(executor, m)
    if state(executor).ns[m].NbNodes() == 0:
        raise ValueError(f"{m} is not computed: call compute first")
    code = f"{m}.ConvertToQuadratic(False)"
    run(state(executor), code)
    return {"ok": True, "records": [record("ConvertToQuadratic", code, recv=m)], "mesh": m}


# into -> (the split, the element counts that must be there to split)
_SPLITS = {"tetrahedra": ("{m}.SplitVolumesIntoTetra({m}, smesh.Hex_5Tet)",
                          ("NbHexas", "NbPrisms", "NbPyramids")),
           "triangles": ("{m}.QuadToTriObject({m})", ("NbQuadrangles",))}


@tool("split_elements", category="edit",
      description=("Split the elements of a computed `mesh` into simplices, keeping it "
                   "conformal: hexahedra, prisms and pyramids into tetrahedra (and their "
                   "quadrangle sides into triangles), or quadrangles into triangles. For "
                   "a solver that reads only those. A later compute does not split again: "
                   "call it again after."),
      schema={"type": "object", "properties": {
          "mesh": {"type": "string"},
          "into": {"type": "string", "enum": list(_SPLITS)}},
          "required": ["mesh", "into"]})
def tool_split_elements(executor, args: dict):
    m, into = args.get("mesh"), args.get("into")
    mesh_of(executor, m)
    if into not in _SPLITS:
        raise ValueError(f"into must be one of {', '.join(_SPLITS)}")
    mesh = state(executor).ns[m]
    if mesh.NbNodes() == 0:
        raise ValueError(f"{m} is not computed: call compute first")
    code, counted = _SPLITS[into]
    if not sum(getattr(mesh, c)() for c in counted):
        raise ValueError(f"{m} has nothing to split into {into}")
    code = code.format(m=m)
    run(state(executor), code)
    return {"ok": True, "records": [record("Split", code, recv=m)], "mesh": m,
            "tetra": mesh.NbTetras(), "triangles": mesh.NbTriangles(),
            "hexa": mesh.NbHexas(), "prism": mesh.NbPrisms(),
            "quadrangles": mesh.NbQuadrangles()}


@tool("export_mesh", category="export",
      description=("Write `mesh` and its groups to a file in the run directory: med, unv, "
                   "stl (faces only), cgns, gmf (.mesh, for MMG and other remeshers) or "
                   "dat. The reply gives the path."),
      schema={"type": "object", "properties": {
          "mesh": {"type": "string"}, "format": {"type": "string", "enum": list(_FORMATS)}},
          "required": ["mesh", "format"]})
def tool_export_mesh(executor, args: dict):
    m, fmt = args.get("mesh"), args.get("format")
    mesh_of(executor, m)
    if fmt not in _FORMATS:
        raise ValueError(f"format must be one of {', '.join(_FORMATS)}")
    run_dir = getattr(executor, "run_dir", None) or os.getcwd()
    os.makedirs(run_dir, exist_ok=True)
    rel = f"{m}.{_EXTENSIONS.get(fmt, fmt)}"
    path = os.path.join(run_dir, rel)
    run(state(executor), f"{m}.{_FORMATS[fmt].format(path=path)}")
    code = f"{m}.{_FORMATS[fmt].format(path=rel)}"
    return {"ok": True, "records": [record("Export", code, recv=m)], "mesh": m, "path": path,
            "note": f"written to {path}; result.py writes {rel} in the folder it runs from"}


def _remove_mesh_group(st, m, info, name) -> None:
    """Remove a mesh group, and the combine_groups groups made from it."""
    info.groups.pop(name)
    info.combined.pop(name, None)
    st.ns[m].RemoveGroup(st.ns.pop(f"{m}_{name}"))
    for c in [c for c, deps in info.combined.items() if name in deps]:
        if c in info.groups:
            _remove_mesh_group(st, m, info, c)


def forget(executor, names: list) -> list:
    st = executor.__dict__.get("smesh_state")
    if st is None:
        return []
    smesh, study = st.ns["smesh"], st.ns["salome"].myStudy
    gone = []
    for name in names:
        if name in st.meshes:
            smesh.RemoveMesh(st.ns[name])
            st.meshes.pop(name)
            # its algorithms, parameters and mesh groups: <mesh>_...
            for var in [v for v in st.ns if v.startswith(f"{name}_")
                        and v not in st.meshes and v not in st.geo_groups]:
                st.ns.pop(var)
        elif name in st.geo_groups:
            g = st.geo_groups.pop(name)
            for m, info in st.meshes.items():
                if name in info.groups:
                    _remove_mesh_group(st, m, info, name)
            if g.owner == "geom":
                st.ns["geompy"].RemoveObject(st.ns[name])
            else:
                from executor.shaper.state import forget as shaper_forget
                shaper_forget(executor, [name])
        else:
            continue
        st.ns.pop(name, None)
        gone.append(name)
    return gone


def clear(executor) -> None:
    st = executor.__dict__.get("smesh_state")
    if st is not None:
        for name in list(st.meshes):
            st.ns["smesh"].RemoveMesh(st.ns[name])
    executor.__dict__.pop("smesh_state", None)
