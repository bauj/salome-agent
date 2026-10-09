from executor.smesh.state import group_ref, mesh_of, record, run, state, tool

ELEMENT_TYPES = {"vertex": "NODE", "edge": "EDGE", "face": "FACE", "solid": "VOLUME"}
_CHOICES = {"node": "NODE", "edge": "EDGE", "face": "FACE", "volume": "VOLUME"}
_SHAPE_TYPES = {"VERTEX": "vertex", "EDGE": "edge", "FACE": "face", "SOLID": "solid"}


def element_for(sub_type: str, element_type) -> str:
    if element_type is None:
        return ELEMENT_TYPES[sub_type]
    if element_type not in _CHOICES:
        raise ValueError(f"element_type must be one of {', '.join(_CHOICES)}")
    return _CHOICES[element_type]


def group_code(var, m, gexpr, name, element) -> str:
    return f"{var} = {m}.GroupOnGeom({gexpr}, {name!r}, SMESH.{element})"


def _sub_type(st, gexpr, g) -> str:
    if g.sub_type:
        return g.sub_type
    t = str(eval(gexpr, st.ns).GetMinShapeType())
    g.sub_type = _SHAPE_TYPES.get(t, "face")
    return g.sub_type


def _add(executor, m, group, element_type) -> tuple:
    st = state(executor)
    info = mesh_of(executor, m)
    gexpr, g = group_ref(executor, group)
    if group in info.groups:
        raise ValueError(f"{m} already has a group {group!r}")
    var = f"{m}_{group}"
    code = group_code(var, m, gexpr, group, element_for(_sub_type(st, gexpr, g), element_type))
    run(st, code)
    info.groups[group] = group
    return code, var


@tool("group_on_geometry", category="group",
      description=("A group of the mesh elements on the named shape group `group` (faces on "
                   "a face group, …), named like it. It follows the mesh when it is "
                   "recomputed."),
      schema={"type": "object", "properties": {
          "mesh": {"type": "string"}, "group": {"type": "string"},
          "element_type": {"type": "string", "enum": list(_CHOICES)}},
          "required": ["mesh", "group"]})
def tool_group_on_geometry(executor, args: dict):
    m, group = args.get("mesh"), args.get("group")
    code, var = _add(executor, m, group, args.get("element_type"))
    size = state(executor).ns[var].Size()
    return {"ok": True, "records": [record("GroupOnGeom", code, recv=m, refs=[group])],
            "mesh": m, "group": group, "size": size}


@tool("groups_from_geometry", category="group",
      description="One mesh group for every named group of the mesh's shape.",
      schema={"type": "object", "properties": {"mesh": {"type": "string"}},
              "required": ["mesh"]})
def tool_groups_from_geometry(executor, args: dict):
    m = args.get("mesh")
    info = mesh_of(executor, m)
    st = state(executor)
    sh = executor.__dict__.get("shaper_state")
    for name, shape in (getattr(sh, "groups", {}) if sh else {}).items():
        if shape == info.shape:
            group_ref(executor, name)
    names = [n for n, g in st.geo_groups.items()
             if g.shape == info.shape and n not in info.groups]
    if not names:
        raise ValueError(f"{info.shape} has no named groups without a mesh group yet")
    lines = [_add(executor, m, n, None)[0] for n in names]
    return {"ok": True, "records": [record("GroupOnGeom", "\n".join(lines), recv=m, refs=names)],
            "mesh": m, "groups": names}
