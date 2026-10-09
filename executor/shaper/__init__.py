"""The shaper tool package. Importing it registers every tool; SHAPER itself
is imported on the first call, so geom-only runs never load it."""
from executor.geom.helpers import _pinned
from executor.shaper.state import registry, reset, clear, forget  # noqa: F401
from executor.shaper import sketch, features, primitives, queries, transforms, subshapes, swept, groups  # noqa: F401 — registers the tools
from executor.shaper import editing
from executor.shaper.editing import check_finish as check_results  # noqa: F401

TOOLS = registry.handlers()


def list_tools() -> list[dict]:
    return registry.list_tools()


def call_tool(executor, name: str, args: dict, pin: list | None = None) -> dict:
    handler = TOOLS.get(name)
    if handler is None:
        return {"ok": False, "error": f"unknown shaper tool {name!r}"}
    _pinned[:] = list(pin or [])
    try:
        args = registry.coerce_args(name, args)
        return editing.call(executor, name, args, lambda: handler(executor, args))
    except Exception as e:  # noqa: BLE001 — a tool error goes back to the model
        return {"ok": False, "error": str(e) or type(e).__name__}
    finally:
        _pinned.clear()
