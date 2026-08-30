import sys

from salome_agent.logs import log


def _render_plan(payload, stream):
    stream.write("\nProposed plan:\n")
    for i, s in enumerate(payload.get("plan") or [], 1):
        stream.write(f"  {i}. [{s.get('module', '?')}] {s.get('intent', '')}\n")
    stream.flush()


def prompt_plan(payload, read=input, stream=sys.stderr):
    _render_plan(payload, stream)
    while True:
        ans = read("approve plan? [Y]es / [n]o abort: ").strip().lower()
        if ans in ("", "y", "yes", "approve"):
            return {"action": "approve"}
        if ans in ("n", "no", "abort", "q"):
            return {"action": "abort"}


def prompt_retry(payload, read=input, stream=sys.stderr):
    err = payload.get("last_error") or {}
    stream.write(f"\nStage failed: {err.get('message', '(no message)')}\n")
    line = err.get("offending_line")
    if line:
        stream.write(f"  offending line: {line}\n")
    stream.flush()
    try:
        while True:
            ans = read("[h]int / [s]kip stage / [a]bort run: ").strip().lower()
            if ans.startswith("h"):
                hint = read("hint for the specialist: ").strip()
                return {"action": "hint", "hint": hint}
            if ans.startswith("s"):
                return {"action": "skip"}
            if ans.startswith("a") or ans == "q":
                return {"action": "abort"}
    except EOFError:
        return {"action": "abort"}


def resume_interactive(graph, first_result, cfg, read=input, stream=sys.stderr):
    from langgraph.types import Command

    result = first_result
    while result.get("__interrupt__"):
        payload = result["__interrupt__"][0].value
        try:
            if "plan" in payload:
                decision = prompt_plan(payload, read, stream)
            elif "last_error" in payload:
                decision = prompt_retry(payload, read, stream)
            else:
                raise RuntimeError(f"unknown interrupt payload: {payload}")
        except EOFError:
            log.warning("no input available - aborting run")
            decision = {"action": "abort"}
        result = graph.invoke(Command(resume=decision), config=cfg)
    return result
