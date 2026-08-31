def write_report(state: dict) -> str:
    lines = [f"# Report: {state['run_id']}", "",
             f"Request: {state['user_request']}", ""]
    if state.get("fatal"):
        lines += ["## Outcome", "", f"**Run aborted:** {state['fatal']}", ""]
    sess = state.get("session") or {}
    if sess.get("mode") == "gui":
        prefix = sess.get("name_prefix") or "sa_"
        lines += [f"Session: live GUI study. Objects prefixed `{prefix}` may "
                  f"remain from failed stage attempts — safe to bulk-delete.", ""]
    elif sess.get("mode") == "headless":
        lines += ["Session: isolated study.", ""]
    lines.append("## Stages")
    total = len(state["stages"])
    for idx in sorted(state["stages"]):
        stage = state["stages"][idx]
        retries = stage.get("static_retries", 0) + stage.get("exec_retries", 0)
        lines.append(f"- stage {idx + 1}/{total} ({stage['module']}): "
                     f"{stage['status']}, retries={retries}")
    return "\n".join(lines)
