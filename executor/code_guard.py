"""Check SHAPER and SMESH source built from tool arguments before it runs.

The tools paste the model's names into Python source; a name that broke out of
its place could run anything in the SALOME process. Generated code only uses the
run's namespace, so source is refused when it:
- imports, or uses a name containing "__";
- reads a name that is neither in the namespace, set by the code itself, nor
  one of a few builtins;
- reaches a module outside SALOME, through a name or a module attribute.

Not caught: a call to a SALOME function the namespace already offers (a SHAPER
export writing a file, say), or a module reached through a non-module object's
attribute."""
import ast
import types

_BUILTINS = frozenset({
    "abs", "all", "any", "bool", "dict", "enumerate", "float", "int", "isinstance",
    "len", "list", "max", "min", "range", "reversed", "round", "set", "sorted", "str",
    "sum", "tuple", "zip", "True", "False", "None",
})
_SALOME_TOPS = frozenset({
    "salome", "SketchAPI", "ModelAPI", "GeomAPI", "GeomAlgoAPI", "GeomDataAPI",
    "ModelHighAPI", "SALOMEDS", "GEOM", "SMESH", "StdMeshers", "NETGENPlugin",
    "smeshBuilder", "geomBuilder", "model",
})


def _salome_module(mod) -> bool:
    return mod.__name__.split(".")[0] in _SALOME_TOPS


def _bound(tree) -> set:
    """Names the code sets itself: assignments, loop and comprehension targets,
    function and lambda parameters, function names."""
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
            out.add(n.id)
        elif isinstance(n, ast.arg):
            out.add(n.arg)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.add(n.name)
    return out


def _attr_chain(node):
    """(root name, [attr, ...]) of a.b.c, or None when the root is not a name."""
    attrs = []
    while isinstance(node, ast.Attribute):
        attrs.append(node.attr)
        node = node.value
    return (node.id, attrs[::-1]) if isinstance(node, ast.Name) else None


def refusal(code: str, ns: dict, mode: str = "exec") -> str | None:
    """Why `code` may not run in `ns`, or None when it may."""
    try:
        tree = ast.parse(code, mode=mode)
    except SyntaxError as e:
        return f"generated code does not parse: {e.msg} (line {e.lineno})"
    local = _bound(tree)
    for n in ast.walk(tree):
        if isinstance(n, (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal)):
            return "generated code may not import or rebind globals"
        if isinstance(n, ast.Attribute) and "__" in n.attr:
            return f"generated code may not use the name {n.attr!r}"
        if isinstance(n, ast.Name):
            if "__" in n.id:
                return f"generated code may not use the name {n.id!r}"
            if isinstance(n.ctx, ast.Load) and n.id not in local and n.id not in _BUILTINS:
                if n.id not in ns:
                    return f"generated code uses {n.id!r}, which is not in the run"
                v = ns[n.id]
                if isinstance(v, types.ModuleType) and not _salome_module(v):
                    return f"generated code may not use the module {n.id!r}"
        if isinstance(n, ast.Attribute):
            chain = _attr_chain(n)
            if chain is None or chain[0] in local or chain[0] not in ns:
                continue
            v = ns[chain[0]]
            for a in chain[1]:
                if not isinstance(v, types.ModuleType):
                    break
                v = getattr(v, a, None)
                if isinstance(v, types.ModuleType) and not _salome_module(v):
                    return f"generated code may not reach the module {a!r}"
    return None


def check(code: str, ns: dict, mode: str = "exec") -> None:
    why = refusal(code, ns, mode)
    if why:
        raise ValueError(why)
