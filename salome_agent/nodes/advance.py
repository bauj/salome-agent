from salome_agent.codegen import generate
from salome_agent.modules import spec_for


def advance(state: dict, result_names: list[str] | None = None) -> dict:
    idx = state["stage_idx"]
    stage = dict(state["stages"][idx])
    if stage["module"] == "geom" and stage["script"] is None:
        call_prefix = spec_for(stage["module"]).call_prefix or "geompy."
        stage["script"] = generate(stage["call_log"], result_names, call_prefix=call_prefix)
    stage["status"] = "ok"
    stages = dict(state["stages"])
    stages[idx] = stage
    return {"stages": stages, "stage_idx": idx + 1}


def skip(state: dict) -> dict:
    idx = state["stage_idx"]
    stage = dict(state["stages"][idx])
    stage["status"] = "skipped"
    stages = dict(state["stages"])
    stages[idx] = stage
    return {"stages": stages, "stage_idx": idx + 1}
