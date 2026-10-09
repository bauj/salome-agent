from executor.smesh.state import MeshInfo, new_name, record, run, shape_ref, state, tool


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
