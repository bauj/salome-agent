"""The calc tool: a few lines of Python arithmetic, run in a separate locked-down
interpreter so the model computes numbers instead of reasoning them out."""
import ast
import subprocess
import sys

_TIMEOUT = 5.0
_MAX_OUTPUT = 4000
_MEMORY_BYTES = 512 * 1024 * 1024

CALC_DEF = {
    "name": "calc",
    "description": (
        "Run a few lines of Python for arithmetic: trig, intersections, corner "
        "points, areas, is-this-point-inside checks. `math` and its functions "
        "(sin, cos, radians, sqrt, hypot, atan2, pi, ...) are in scope; nothing "
        "else can be imported. "
        "print() what you need; the last expression's value is shown too. Nothing "
        "is built: use the numbers in your next tool calls."),
    "parameters": {"type": "object", "properties": {
        "code": {"type": "string", "description": "Python source"}},
        "required": ["code"]},
}

# Appended to the system prompt of every module that offers calc.
CALC_PROMPT = """Arithmetic -- trig, intersections, corner points, whether a point lies
inside a shape -- goes in `calc`, never in your reasoning. The moment you need a
number you would otherwise work out by hand, stop and call calc; your reasoning
so far is kept and you continue from it once the result is back.
Exact values matter only where they are given or show in the result. A length
that ends inside other material (a bar end buried in a ring, a cutter that
pokes through) can be any round value that clearly overshoots."""

# Runs in the child: stdin is the user snippet, already checked by _refusal.
_RUNNER = f'''
import ast, builtins, math, sys, traceback
try:
    import resource
    resource.setrlimit(resource.RLIMIT_AS, ({_MEMORY_BYTES}, {_MEMORY_BYTES}))
except Exception:
    pass
src = sys.stdin.read()
safe = {{n: getattr(builtins, n)
        for n in ("abs", "min", "max", "round", "range", "len", "sum", "print",
                  "enumerate", "zip", "list", "dict", "tuple", "set", "sorted",
                  "reversed", "float", "int", "bool", "str", "any", "all", "map",
                  "filter", "isinstance", "divmod", "pow")}}
# `import math` is written by reflex; only math resolves, _refusal checked names.
safe["__import__"] = lambda name, *a, **k: math
ns = {{"__builtins__": safe, "math": math}}
ns.update({{k: v for k, v in vars(math).items() if not k.startswith("_")}})
try:
    tree = ast.parse(src, "<calc>")
    last = None
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        last = ast.Expression(tree.body.pop().value)
    exec(compile(tree, "<calc>", "exec"), ns)
    if last is not None:
        value = eval(compile(last, "<calc>", "eval"), ns)
        if value is not None:
            print(repr(value))
except SyntaxError as e:
    print(f"error: SyntaxError: {{e.msg}} (line {{e.lineno}})")
except BaseException as e:
    line = next((f.lineno for f in reversed(traceback.extract_tb(e.__traceback__))
                 if f.filename == "<calc>"), None)
    where = f" (line {{line}})" if line else ""
    print(f"error: {{type(e).__name__}}: {{e}}{{where}}")
'''


def _refusal(code: str) -> str | None:
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return f"error: SyntaxError: {e.msg} (line {e.lineno})"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import) and any(a.name != "math" for a in node.names):
            return "error: only math can be imported — and it is already in scope"
        if isinstance(node, ast.ImportFrom) and (node.module != "math" or node.level):
            return "error: only math can be imported — and it is already in scope"
        name = (node.id if isinstance(node, ast.Name)
                else node.attr if isinstance(node, ast.Attribute) else None)
        if name and "__" in name:
            return f"error: names with '__' are not allowed ({name})"
    return None


def run_calc(code: str, timeout: float = _TIMEOUT) -> str:
    """The snippet's output as the tool reply: printed text plus the last
    expression's repr, or one `error: ...` line. Never raises."""
    if not code or not code.strip():
        return "error: empty snippet — pass the Python to run in `code`"
    refused = _refusal(code)
    if refused:
        return refused
    try:
        proc = subprocess.run([sys.executable, "-I", "-c", _RUNNER], input=code,
                              capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return f"error: timed out after {timeout:g} s"
    out = proc.stdout.rstrip("\n")
    if proc.returncode != 0 and not out:
        out = f"error: calc process failed: {proc.stderr.strip()[-500:]}"
    if len(out) > _MAX_OUTPUT:
        out = out[:_MAX_OUTPUT] + "\n(output cut)"
    return out or "(no output — print() what you need, or end with an expression)"
