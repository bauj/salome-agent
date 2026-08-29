def write_report(state: dict) -> str:
    lines = [f"# Report: {state['run_id']}", "", f"Request: {state['user_request']}", "", "## Stages"]
    for idx in sorted(state["stages"]):
        stage = state["stages"][idx]
        retries = stage.get("static_retries", 0) + stage.get("exec_retries", 0)
        lines.append(f"- stage {idx} ({stage['module']}): {stage['status']}, retries={retries}")
    return "\n".join(lines)
