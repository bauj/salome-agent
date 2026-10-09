import os

from executor.smesh.setup import HYPOTHESES, _on
from executor.smesh.state import group_ref, mesh_of, record, run, state, tool

_FORMATS = {"med": "ExportMED({path!r})", "unv": "ExportUNV({path!r})",
            "stl": "ExportSTL({path!r})", "cgns": "ExportCGNS({path!r})"}
_MESH = {"type": "object", "properties": {"mesh": {"type": "string"}}, "required": ["mesh"]}


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
    method = HYPOTHESES[kind][1]
    lines, refs = [], []
    for g in _on(executor, args.get("on")):
        geom = group_ref(executor, g)[0] if g else f"{m}.GetShape()"
        lines.append(f"for _h in list({m}.GetHypothesisList({geom})):\n"
                     f"    if _h.GetName() == {method!r}:\n"
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


@tool("export_mesh", category="export",
      description=("Write `mesh` and its groups to a file in the run directory: med, unv, "
                   "stl (faces only) or cgns. The reply gives the path."),
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
    path = os.path.join(run_dir, f"{m}.{fmt}")
    run(state(executor), f"{m}.{_FORMATS[fmt].format(path=path)}")
    rel = f"{m}.{fmt}"
    code = f"{m}.{_FORMATS[fmt].format(path=rel)}"
    return {"ok": True, "records": [record("Export", code, recv=m)], "mesh": m, "path": path}


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
        elif name in st.geo_groups:
            g = st.geo_groups.pop(name)
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
