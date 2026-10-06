import ast

TOPLEVEL_MODULES = {"geompy", "smesh", "model"}
BANNED_PATTERNS = ["salome_init", "salome.sg."]


def check_static(script: str, api_map: dict) -> dict | None:
    try:
        tree = ast.parse(script)
    except SyntaxError as e:
        return {"phase": "static", "message": str(e), "offending_line": e.lineno,
                "traceback": None}

    for pattern in BANNED_PATTERNS:
        if pattern in script:
            return {"phase": "static", "message": f"banned pattern: {pattern}",
                     "offending_line": None, "traceback": None}

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            obj = node.func.value
            if isinstance(obj, ast.Name) and obj.id in TOPLEVEL_MODULES:
                members = api_map.get(obj.id, {}).get("members", [])
                if node.func.attr not in members:
                    return {"phase": "static",
                             "message": f"{obj.id}.{node.func.attr} does not exist",
                             "offending_line": node.lineno, "traceback": None}
    return None
