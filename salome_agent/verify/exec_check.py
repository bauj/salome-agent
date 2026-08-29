from typing import Literal

from salome_agent.executor_client import SessionDied


def check_exec(script: str, module: str, client, timeout: float) -> dict | Literal["session_died"] | None:
    try:
        run_result = client.run(script, timeout)
    except SessionDied:
        return "session_died"

    if not run_result["ok"]:
        return {"phase": "exec", "message": run_result.get("exception", ""),
                "offending_line": run_result.get("offending_line"),
                "traceback": run_result.get("traceback")}

    sanity = client.sanity(module, {})
    if not sanity["ok"] or sanity["failures"]:
        return {"phase": "exec", "message": "; ".join(sanity["failures"]),
                "offending_line": None, "traceback": None}
    return None
