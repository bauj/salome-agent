from salome_agent.codegen import generate


def advance(state: dict, result_names: list[str] | None = None) -> dict:
    idx = state["stage_idx"]
    stage = dict(state["stages"][idx])
    if stage["module"] == "geom" and stage["script"] is None:
        stage["script"] = generate(stage["call_log"], result_names)
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
