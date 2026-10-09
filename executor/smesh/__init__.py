"""The SMESH tool package. Importing it registers every tool."""
from executor.geom.helpers import _pinned
from executor.smesh.state import COMPUTE_TIMEOUT_S, registry, reset  # noqa: F401
from executor.smesh import naming, setup, compute, local, groups  # noqa: F401 — registers the tools

TOOLS = registry.handlers()
_LONG = {"compute": COMPUTE_TIMEOUT_S}


def list_tools() -> list[dict]:
    return registry.list_tools()


def timeout_for(name: str) -> "float | None":
    return _LONG.get(name)


def call_tool(executor, name: str, args: dict, pin: list | None = None) -> dict:
    handler = TOOLS.get(name)
    if handler is None:
        return {"ok": False, "error": f"unknown smesh tool {name!r}"}
    _pinned[:] = list(pin or [])
    try:
        return handler(executor, registry.coerce_args(name, args))
    except Exception as e:  # noqa: BLE001 — a tool error goes back to the model
        return {"ok": False, "error": str(e) or type(e).__name__}
    finally:
        _pinned.clear()


def check_results(executor, names: list) -> dict:
    return {n: {"status": "skip"} for n in names}


def clear(executor) -> None:
    reset(executor)


def forget(executor, names: list) -> list:
    return []
