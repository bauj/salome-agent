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


def sub_type_of(executor, group) -> str:
    """vertex, edge, face or solid: the sub-shape type of a named group."""
    gexpr, g = group_ref(executor, group)
    return _sub_type(state(executor), gexpr, g)


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


_OPERATIONS = ("union", "intersect", "cut")


def combine_code(var, m, name, element, operation, groups, minus) -> str:
    """A group on a filter over other mesh groups, so it follows a recompute.
    A cut is written as the union of (group AND NOT minus...) terms: read left to
    right or with AND first, it means the same."""
    def crit(g, op="FT_Undefined", neg=False):
        unary = ", UnaryOp=SMESH.FT_LogicalNOT" if neg else ""
        return (f"smesh.GetCriterion(SMESH.{element}, SMESH.FT_BelongToMeshGroup, "
                f"SMESH.FT_Undefined, {g}{unary}, BinaryOp=SMESH.{op})")
    terms = []
    if operation == "cut":
        for i, g in enumerate(groups):
            seq = [g, *minus]
            for j, x in enumerate(seq):
                last = i == len(groups) - 1 and j == len(seq) - 1
                op = "FT_Undefined" if last else (
                    "FT_LogicalOR" if j == len(seq) - 1 else "FT_LogicalAND")
                terms.append(crit(x, op, neg=j > 0))
    else:
        op = "FT_LogicalOR" if operation == "union" else "FT_LogicalAND"
        terms = [crit(g, op if i < len(groups) - 1 else "FT_Undefined")
                 for i, g in enumerate(groups)]
    body = ",\n    ".join(terms)
    return (f"{var} = {m}.GroupOnFilter(SMESH.{element}, {name!r}, "
            f"smesh.GetFilterFromCriteria([\n    {body}]))")


@tool("combine_groups", category="group",
      description=("A mesh group `name` from other mesh groups of `mesh`, all of one "
                   "element type: union (elements in any of `groups`), intersect (in "
                   "all), cut (in `groups` but in none of `minus`). It follows the mesh "
                   "when it is recomputed."),
      schema={"type": "object", "properties": {
          "mesh": {"type": "string"}, "name": {"type": "string"},
          "operation": {"type": "string", "enum": list(_OPERATIONS)},
          "groups": {"type": "array", "items": {"type": "string"}},
          "minus": {"type": "array", "items": {"type": "string"}}},
          "required": ["mesh", "name", "operation", "groups"]})
def tool_combine_groups(executor, args: dict):
    m, name, operation = args.get("mesh"), args.get("name"), args.get("operation")
    info = mesh_of(executor, m)
    st = state(executor)
    groups, minus = list(args.get("groups") or []), list(args.get("minus") or [])
    if operation not in _OPERATIONS:
        raise ValueError(f"operation must be one of {', '.join(_OPERATIONS)}")
    if not isinstance(name, str) or not name.isidentifier():
        raise ValueError(f"name must be a plain identifier, got {name!r}")
    if name in info.groups:
        raise ValueError(f"{m} already has a group {name!r}")
    if operation == "cut" and not minus:
        raise ValueError("cut needs the groups to remove in `minus`")
    if operation != "cut" and minus:
        raise ValueError("only cut takes `minus`")
    if not groups or (operation != "cut" and len(groups) < 2):
        raise ValueError(f"{operation} needs {'one' if operation == 'cut' else 'two'} or "
                         "more groups")
    missing = [g for g in groups + minus if g not in info.groups]
    if missing:
        raise ValueError(f"{m} has no mesh group {', '.join(missing)}; its groups: "
                         f"{', '.join(info.groups) or 'none yet'}")
    types = {str(st.ns[f"{m}_{g}"].GetType()) for g in groups + minus}
    if len(types) != 1:
        raise ValueError(f"the groups mix element types ({', '.join(sorted(types))})")
    var = f"{m}_{name}"
    code = combine_code(var, m, name, types.pop(), operation,
                        [f"{m}_{g}" for g in groups], [f"{m}_{g}" for g in minus])
    run(st, code)
    info.groups[name] = None
    info.combined[name] = groups + minus
    return {"ok": True, "records": [record("GroupOnFilter", code, recv=m)],
            "mesh": m, "group": name, "size": st.ns[var].Size()}
