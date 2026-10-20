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
    if usage.get("llm_seconds"):
        lines.append(f"- time waiting on the endpoint: {usage['llm_seconds']:.1f}s")
    if approx:
        lines.append(f"- approx input (all calls, request chars/4): {approx:,}")
    if usage.get("incomplete_calls"):
        lines.append(f"- incomplete calls (cut, cancelled or dropped): "
                     f"{usage['incomplete_calls']}, approx output streamed before "
                     f"the stop (chars/4): {usage.get('approx_incomplete_output_tokens', 0):,}")
    if metered < calls:
        lines.append(f"- note: {calls - metered} call(s) returned no usage data; "
                     f"provider input/output totals cover the {metered} that did "
                     f"(approx input above covers all {calls})")
    lines.append("")
    return lines


def _endpoint_lines(endpoints: list) -> list[str]:
    lines = ["## LLM endpoint", ""]
    for module, url, model in endpoints:
        lines.append(f"- {module or 'shared'}: `{model}` at {url}")
    lines.append("")
    return lines


def _tok(n: int) -> str:
    return f"{n / 1000:.1f}k" if n >= 1000 else str(n)


def _stage_timing_suffix(tm: dict) -> str:
    parts = [f"{tm.get('turns', 0)} turns",
             f"llm {tm.get('llm_s', 0):.1f}s"]
    if tm.get("exec_s"):
        parts.append(f"exec {tm['exec_s']:.1f}s")
    parts.append(f"core {tm.get('core_s', 0):.1f}s")
    parts.append(f"tokens {_tok(tm.get('in_tok', 0))} in / {_tok(tm.get('out_tok', 0))} out")
    return " — " + ", ".join(parts)


def _timing_lines(state: dict) -> list[str]:
    pt = state.get("plan_timing") or {}
    stage_tms = [(idx, state["stages"][idx].get("timing") or {})
                 for idx in sorted(state["stages"])]
    stage_tms = [(i, t) for i, t in stage_tms if t]
    if not pt and not stage_tms:
        return []
    total = len(state["stages"])
    lines = ["## Timing", ""]
    if pt:
        lines.append(f"- planner: {pt.get('wall', 0):.1f}s wall, "
                     f"{pt.get('llm_s', 0):.1f}s llm")
    for idx, tm in stage_tms:
        mod = state["stages"][idx].get("module", "?")
        lines.append(f"- stage {idx + 1}/{total} ({mod}): {tm.get('wall', 0):.1f}s wall, "
                     f"{tm.get('llm_s', 0):.1f}s llm, "
                     f"{tm.get('exec_s', 0):.1f}s exec, {tm.get('core_s', 0):.1f}s core")
    wall = pt.get("wall", 0) + sum(t.get("wall", 0) for _, t in stage_tms)
    llm_s = pt.get("llm_s", 0) + sum(t.get("llm_s", 0) for _, t in stage_tms)
    exec_s = sum(t.get("exec_s", 0) for _, t in stage_tms)
    if wall:
        lines.append(f"- total: {wall:.1f}s measured wall — {llm_s:.1f}s llm "
                     f"({llm_s / wall * 100:.0f}%), {exec_s:.1f}s exec")
    lines.append("")
    return lines


def _image_lines(state: dict) -> list[str]:
    from salome_agent.nodes.vision_spec import all_conflicts, unclaimed_parts

    spec = state.get("image_spec")
    if not isinstance(spec, dict):
        return []
    lines = []
    dim = spec.get("dimensionality")
    if isinstance(dim, dict) and dim.get("value") == "2d" and dim.get("assumed"):
        lines.append('Built as a flat face: the image gave no thickness -- say "extrude N mm" '
                     "to make it solid.")
    unclaimed = unclaimed_parts(state.get("plan") or [], spec)
    if unclaimed:
        lines.append(f"Parts in the image spec no stage built: {', '.join(unclaimed)}.")
    conflicts = all_conflicts(spec)
    if conflicts:
        lines += ["Conflicts the vision check could not resolve:"] + [f"- {c}" for c in conflicts]
    return ["## Reference image", "", *lines, ""] if lines else []


def write_report(state: dict, usage: dict | None = None,
                 endpoints: list | None = None, stopped: str | None = None) -> str:
    lines = [f"# Report: {state['run_id']}", "",
             f"Request: {state['user_request']}", ""]
    if stopped:
        lines += ["## Outcome", "", f"**Stopped by you** {stopped}.", ""]
    elif state.get("fatal"):
        lines += ["## Outcome", "", f"**Run aborted:** {state['fatal']}", ""]
    sess = state.get("session") or {}
    if sess.get("mode") == "gui":
        prefix = sess.get("name_prefix") or "sa_"
        lines += [f"Session: live GUI study. Objects prefixed `{prefix}` may "
                  f"remain from failed stage attempts — safe to bulk-delete.", ""]
    elif sess.get("mode") == "headless":
        prefix = sess.get("name_prefix") or "sa_"
        lines += [f"Session: headless study, shared across runs. This run's "
                  f"objects are prefixed `{prefix}`.", ""]
    if endpoints:
        lines += _endpoint_lines(endpoints)
    if usage and usage.get("calls"):
        lines += _usage_lines(usage)
    lines += _timing_lines(state)
    lines += _image_lines(state)
    outcome = state.get("turn_outcome") or {}
    if outcome.get("summary") or outcome.get("question"):
        lines += ["## Amend", "", outcome.get("summary") or "",
                  *([f"Question for you: {outcome['question']}"] if outcome.get("question") else []),
                  ""]
    lines.append("## Stages")
    total = len(state["stages"])
    for idx in sorted(state["stages"]):
        stage = state["stages"][idx]
        suffix = ""
        tm = stage.get("timing")
        if tm:
            suffix = _stage_timing_suffix(tm)
        lines.append(f"- stage {idx + 1}/{total} ({stage['module']}): "
                     f"{stage['status']}{suffix}")
    return "\n".join(lines)
