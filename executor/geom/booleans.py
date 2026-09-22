from executor.geom.helpers import tool, _next_name, _publish


@tool("fuse", category="boolean",
      description='Boolean union of two or more shapes into one.',
      schema={"type": "object", "properties": {"shapes": {"type": "array", "items": {"type": "string"}, "description": "shapes to fuse (2 or more)"}}, "required": ["shapes"]},
      arg_types={})
def tool_fuse(executor, args: dict):
    geompy = executor.namespace["geompy"]
    shapes = [executor.geom_bindings[s] for s in args["shapes"]]
    if len(shapes) == 2:
        result = geompy.MakeFuse(shapes[0], shapes[1])
        fn = "MakeFuse"
    else:
        result = geompy.MakeFuseList(shapes)
        fn = "MakeFuseList"
    is_valid, msg = geompy.CheckShape(result, True, 1)
    if not is_valid:
        return {"ok": False, "error": f"fuse: CheckShape failed ({msg})"}
    name = _next_name(executor, "fuse")
    _publish(executor, result, name)

    if len(shapes) == 2:
        args_list = [{"kind": "ref", "value": args["shapes"][0]},
                     {"kind": "ref", "value": args["shapes"][1]}]
    else:
        args_list = [{"kind": "list", "value": [{"kind": "ref", "value": s} for s in args["shapes"]]}]

    return {
        "ok": True,
        "records": [{
            "fn": fn,
            "args": args_list,
            "out_vars": [name],
            "publish": [name],
        }],
        "sanity": {"ok": True},
    }


@tool("cut", category="boolean",
      description='Boolean subtraction: `main` minus every shape in `tools`.',
      schema={"type": "object", "properties": {"main": {"type": "string", "description": "main shape"}, "tools": {"type": "array", "items": {"type": "string"}, "description": "tool shapes (1 or more)"}}, "required": ["main", "tools"]},
      arg_types={})
def tool_cut(executor, args: dict):
    geompy = executor.namespace["geompy"]
    main = executor.geom_bindings[args["main"]]
    tools = [executor.geom_bindings[t] for t in args["tools"]]

    if len(tools) == 1:
        result = geompy.MakeCut(main, tools[0])
        fn = "MakeCut"
        args_list = [{"kind": "ref", "value": args["main"]},
                     {"kind": "ref", "value": args["tools"][0]}]
    else:
        result = geompy.MakeCutList(main, tools)
        fn = "MakeCutList"
        args_list = [{"kind": "ref", "value": args["main"]},
                     {"kind": "list", "value": [{"kind": "ref", "value": t} for t in args["tools"]]}]

    is_valid, msg = geompy.CheckShape(result, True, 1)
    if not is_valid:
        return {"ok": False, "error": f"cut: CheckShape failed ({msg})"}
    name = _next_name(executor, "cut")
    _publish(executor, result, name)
    return {
        "ok": True,
        "records": [{
            "fn": fn,
            "args": args_list,
            "out_vars": [name],
            "publish": [name],
        }],
        "sanity": {"ok": True},
    }


@tool("common", category="boolean",
      description='Boolean intersection of two or more shapes.',
      schema={"type": "object", "properties": {"shapes": {"type": "array", "items": {"type": "string"}, "description": "shapes to intersect"}}, "required": ["shapes"]},
      arg_types={})
def tool_common(executor, args: dict):
    geompy = executor.namespace["geompy"]
    shapes = [executor.geom_bindings[s] for s in args["shapes"]]
    result = geompy.MakeCommonList(shapes)
    is_valid, msg = geompy.CheckShape(result, True, 1)
    if not is_valid:
        return {"ok": False, "error": f"common: CheckShape failed ({msg})"}
    name = _next_name(executor, "common")
    _publish(executor, result, name)
    return {
        "ok": True,
        "records": [{
            "fn": "MakeCommonList",
            "args": [{"kind": "list", "value": [{"kind": "ref", "value": s} for s in args["shapes"]]}],
            "out_vars": [name],
            "publish": [name],
        }],
        "sanity": {"ok": True},
    }


@tool("partition", category="boolean",
      description='Split `shapes` by each other (and optional `tools`), keeping all resulting pieces in one compound.',
      schema={"type": "object", "properties": {"shapes": {"type": "array", "items": {"type": "string"}, "description": "shapes to partition"}, "tools": {"type": "array", "items": {"type": "string"}, "description": "optional tool shapes", "default": []}}, "required": ["shapes"]},
      arg_types={})
def tool_partition(executor, args: dict):
    geompy = executor.namespace["geompy"]
    shapes = [executor.geom_bindings[s] for s in args["shapes"]]
    tools = [executor.geom_bindings[t] for t in args.get("tools", [])]
    result = geompy.MakePartition(shapes, tools)
    is_valid, msg = geompy.CheckShape(result, True, 1)
    if not is_valid:
        return {"ok": False, "error": f"partition: CheckShape failed ({msg})"}
    name = _next_name(executor, "part")
    _publish(executor, result, name)
    tools_refs = args.get("tools", [])
    return {
        "ok": True,
        "records": [{
            "fn": "MakePartition",
            "args": [{"kind": "list", "value": [{"kind": "ref", "value": s} for s in args["shapes"]]},
                     {"kind": "list", "value": [{"kind": "ref", "value": t} for t in tools_refs]}],
            "out_vars": [name],
            "publish": [name],
        }],
        "sanity": {"ok": True},
    }
