"""The SMESH tool package. Importing it registers every tool."""
from executor.geom.helpers import _pinned
from executor.smesh.state import COMPUTE_TIMEOUT_S, registry, reset  # noqa: F401
from executor.smesh import naming, setup, compute, local, groups, editing  # noqa: F401 — registers the tools
from executor.smesh.editing import clear, forget  # noqa: F401

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
    from executor.smesh.compute import measure_mesh, mesh_status
    st = executor.__dict__.get("smesh_state")
    out = {}
    for name in names:
        if st is None or name not in st.meshes:
            out[name] = {"status": "skip"} if st and name in st.geo_groups else {"status": "missing"}
        else:
            out[name] = mesh_status(measure_mesh(executor, name), st.meshes[name].dim)
    return out
