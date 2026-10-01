from executor.geom.helpers import tool, _gui_call


@tool("set_color", category="display",
      description='Set the display colour of `shape` (r/g/b 0-255). Apply once, last, on the final shape.',
      schema={"type": "object", "properties": {"shape": {"type": "string"}, "red": {"type": "integer", "minimum": 0, "maximum": 255}, "green": {"type": "integer", "minimum": 0, "maximum": 255}, "blue": {"type": "integer", "minimum": 0, "maximum": 255}}, "required": ["shape", "red", "green", "blue"]},
      arg_types={"red": int, "green": int, "blue": int})
def tool_set_color(executor, args: dict):
    # Live call for immediate GUI update
    shape = executor.geom_bindings[args["shape"]]
    # Import SALOMEDS at call time
    try:
        from salome.kernel import SALOMEDS
    except ImportError:  # SALOME <= 9.16 layout
        import SALOMEDS
    color = SALOMEDS.Color(args["red"] / 255.0, args["green"] / 255.0, args["blue"] / 255.0)
    shape.SetColor(color)

    # Record for codegen: no out_vars, no publish, method_call emit
    return {
        "ok": True,
        "records": [{
            "fn": "SetColor",
            "args": [{"kind": "salomeds_color", "value": [args["red"], args["green"], args["blue"]]}],
            "out_vars": [],
            "publish": None,
            "emit": "method_call",
            "recv": args["shape"],
        }],
        "sanity": {"ok": True},
    }


@tool("set_transparency", category="display",
      description='Set display transparency of `shape` (0 opaque .. 1 invisible).',
      schema={"type": "object", "properties": {"shape": {"type": "string"}, "transparency": {"type": "number", "minimum": 0.0, "maximum": 1.0}}, "required": ["shape", "transparency"]},
      arg_types={"transparency": float})
def tool_set_transparency(executor, args: dict):
    shape = executor.geom_bindings[args["shape"]]
    # Best-effort live GUI update. Headless (no GUI process),
    # ImportComponentGUI itself can raise -- keep the whole thing guarded,
    # the codegen `gui_call` emit is wrapped the same way.
    _gui_call(executor, shape, "setTransparency", args["transparency"])

    # Record for codegen: no out_vars, no publish, gui_call emit
    return {
        "ok": True,
        "records": [{
            "fn": "setTransparency",
            "args": [{"kind": "literal", "value": args["transparency"]}],
            "out_vars": [],
            "publish": None,
            "emit": "gui_call",
            "recv": args["shape"],
        }],
        "sanity": {"ok": True},
    }


@tool("set_display_mode", category="display",
      description='Set display mode of `shape` (0=wireframe, 1=shading, 2=shading+edges).',
      schema={"type": "object", "properties": {"shape": {"type": "string"}, "mode": {"type": "integer", "enum": [0, 1, 2], "description": "0=wireframe, 1=shading, 2=shading+edges"}}, "required": ["shape", "mode"]},
      arg_types={"mode": int})
def tool_set_display_mode(executor, args: dict):
    shape = executor.geom_bindings[args["shape"]]
    _gui_call(executor, shape, "setDisplayMode", args["mode"])

    # Record for codegen: no out_vars, no publish, gui_call emit
    return {
        "ok": True,
        "records": [{
            "fn": "setDisplayMode",
            "args": [{"kind": "literal", "value": args["mode"]}],
            "out_vars": [],
            "publish": None,
            "emit": "gui_call",
            "recv": args["shape"],
        }],
        "sanity": {"ok": True},
    }
