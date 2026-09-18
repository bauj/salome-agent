import json
import os
import subprocess
import sys
import tempfile

from salome_agent.logs import log

_PLAN_KEYS = ("module", "intent", "params", "inputs", "tools")


def normalize_plan(edited):
    if not isinstance(edited, list) or not edited:
        return None
    for stage in edited:
        if not isinstance(stage, dict) or stage.get("module") not in ("geom", "smesh", "paravis"):
            return None
        stage.setdefault("intent", "")
        stage.setdefault("params", {})
        stage.setdefault("inputs", [])
        stage.setdefault("tools", [])
    return edited


def _edit_plan_in_editor(plan):
    """Round-trip the plan through $EDITOR as JSON. Returns the parsed new
    plan, or None if the editor is unavailable or the result is unusable."""
    editor = os.environ.get("EDITOR") or os.environ.get("VISUAL")
    if not editor:
        return None
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(list(plan), fh, indent=2)
        path = fh.name
    try:
        if subprocess.call([editor, path]) != 0:
            return None
        with open(path) as fh:
            edited = json.load(fh)
    except (OSError, ValueError):
        return None
    finally:
        os.unlink(path)
    return normalize_plan(edited)


def _render_plan(payload, stream):
    stream.write("\nProposed plan:\n")
    for i, s in enumerate(payload.get("plan") or [], 1):
        stream.write(f"  {i}. [{s.get('module', '?')}] {s.get('intent', '')}\n")
    stream.flush()


def prompt_plan(payload, read=input, stream=sys.stderr, edit_fn=_edit_plan_in_editor):
    plan = payload.get("plan") or []
    while True:
        _render_plan({"plan": plan}, stream)
        ans = read("approve plan? [Y]es / [e]dit / [n]o abort: ").strip().lower()
        if ans in ("", "y", "yes", "approve"):
            return {"action": "approve"}
        if ans in ("n", "no", "abort", "q"):
            return {"action": "abort"}
        if ans in ("e", "edit"):
            edited = edit_fn(plan)
            if edited is None:
                stream.write("edit cancelled or unusable (need $EDITOR set, valid JSON, "
                             "module in geom|smesh|paravis)\n")
                continue
            return {"action": "edit", "plan": edited}


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


def terminal_gate(payload, read=input, stream=sys.stderr, edit_fn=_edit_plan_in_editor):
    try:
        if "plan" in payload:
            return prompt_plan(payload, read, stream, edit_fn)
        if "last_error" in payload:
            return prompt_retry(payload, read, stream)
    except EOFError:
        log.warning("no input available - aborting run")
        return {"action": "abort"}
    raise RuntimeError(f"unknown interrupt payload: {payload}")
