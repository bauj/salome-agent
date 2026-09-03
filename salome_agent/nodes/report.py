def _usage_lines(usage: dict) -> list[str]:
    calls = usage.get("calls", 0)
    metered = usage.get("metered_calls", 0)
    pt = usage.get("prompt_tokens", 0)
    ct = usage.get("completion_tokens", 0)
    approx = usage.get("approx_prompt_tokens", 0)
    lines = ["## LLM usage", "",
             f"- calls: {calls}",
             f"- input tokens: {pt:,}",
             f"- output tokens: {ct:,}",
             f"- total tokens: {pt + ct:,}"]
    if approx:
        lines.append(f"- approx input (all calls, request chars/4): {approx:,}")
    if metered < calls:
        lines.append(f"- note: {calls - metered} call(s) returned no usage data; "
                     f"provider input/output totals cover the {metered} that did "
                     f"(approx input above covers all {calls})")
    lines.append("")
    return lines


def write_report(state: dict, usage: dict | None = None) -> str:
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
    if usage and usage.get("calls"):
        lines += _usage_lines(usage)
    lines.append("## Stages")
    total = len(state["stages"])
    for idx in sorted(state["stages"]):
        stage = state["stages"][idx]
        retries = stage.get("static_retries", 0) + stage.get("exec_retries", 0)
        lines.append(f"- stage {idx + 1}/{total} ({stage['module']}): "
                     f"{stage['status']}, retries={retries}")
    return "\n".join(lines)
