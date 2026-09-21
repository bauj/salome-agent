import os
import threading
import uuid
from dataclasses import dataclass


@dataclass
class RunResult:
    run_id: str
    run_dir: str
    final_state: dict
    exit_code: int


class EventMapper:
    """Turns graph.stream(stream_mode="updates") chunks into UI events
    ({"kind", "run_id", "data"}). Tracks the current stage index itself:
    run_stage/verify updates do not carry it, advance/skip carry the next one."""

    def __init__(self, run_id: str, run_dir: str):
        self.run_id = run_id
        self.run_dir = run_dir
        self.plan: list[dict] = []
        self.idx = 0
        self._started: set[int] = set()

    def event(self, kind: str, **data) -> dict:
        return {"kind": kind, "run_id": self.run_id, "data": data}

    def events(self, node: str, update: dict | None) -> list[dict]:
        update = update or {}
        out = []
        if node in ("plan", "plan_gate") and "plan" in update:
            self.plan = update["plan"]
            self.idx = 0
            self._started.clear()
            out.append(self.event("plan", stages=self.plan))
        if update.get("fatal"):
            out.append(self.event("fatal", message=update["fatal"]))
            return out
        if node == "run_stage" and self.idx not in self._started and self.idx < len(self.plan):
            self._started.add(self.idx)
            p = self.plan[self.idx]
            out.append(self.event("stage_start", index=self.idx,
                                  module=p.get("module", "?"), intent=p.get("intent", "")))
        if node in ("run_stage", "verify_static", "verify_exec"):
            err = update.get("last_error")
            if err:
                out.append(self.event("stage_attempt", index=self.idx,
                                      phase=err.get("phase") or node, ok=False,
                                      error=err.get("message")))
            elif node != "run_stage":
                out.append(self.event("stage_attempt", index=self.idx,
                                      phase=node.split("_", 1)[1], ok=True, error=None))
        if node == "advance":
            stage = (update.get("stages") or {}).get(self.idx) or {}
            out.append(self.event("stage_done", index=self.idx,
                                  names=list(stage.get("result_names") or [])))
            self.idx = update.get("stage_idx", self.idx + 1)
        if node == "skip":
            out.append(self.event("stage_skipped", index=self.idx))
            self.idx = update.get("stage_idx", self.idx + 1)
        if node == "report" and "report" in update:
            out.append(self.event("report",
                                  result_py=os.path.join(self.run_dir, "result.py"),
                                  report_md=os.path.join(self.run_dir, "report.md"),
                                  text=update["report"]))
        if update.get("aborted"):
            out.append(self.event("aborted"))
        return out


def _session_handle(client) -> dict:
    if client is None:
        return {"mode": "none", "pid": None, "salome_port": None, "name_prefix": ""}
    return {"mode": client.mode, "pid": None, "salome_port": None,
            "name_prefix": client.name_prefix}


def run_request(request: str, config: dict, *, client, on_event, on_gate,
                cancel: threading.Event | None = None, batch: bool = False,
                verbose: int = 0, run_id: str | None = None,
                request_timeout: float | None = None) -> RunResult:
    from langgraph.types import Command

    from salome_agent import cli
    from salome_agent.graph import build_graph
    from salome_agent.logs import log

    run_id = run_id or uuid.uuid4().hex[:12]
    work_dir = os.path.expanduser(config.get("work_dir") or ".")
    run_dir = os.path.abspath(os.path.join(work_dir, run_id))
    # exec_trace.jsonl (salome_agent/cli.py) is recorded under this same
    # verbose >= 2 condition — a recorded run always has both halves or neither.
    trace_path = os.path.join(run_dir, "llm_trace.jsonl") if verbose >= 2 else None
    if trace_path:
        log.info("LLM trace: %s", trace_path)
    llm = cli.build_llm(config, request_timeout, trace_path, run_id)
    cfg = {"configurable": {"thread_id": uuid.uuid4().hex, "llm_client": llm,
                            "executor_client": client}}
    graph = build_graph(batch=batch)
    mapper = EventMapper(run_id, run_dir)
    inp = {
        "user_request": request, "config": {**config, "batch": batch},
        "session": _session_handle(client), "plan": [], "plan_approved": False,
        "stage_idx": 0, "stages": {}, "last_error": None, "history": [],
        "study_ns": "salome_agent_run", "final_script": None, "report": None,
        "fatal": None, "aborted": False, "run_id": run_id,
    }

    def cancelled() -> bool:
        return cancel is not None and cancel.is_set()

    stopped = False
    while True:
        pending = None
        for chunk in graph.stream(inp, config=cfg, stream_mode="updates"):
            for node, update in chunk.items():
                if node == "__interrupt__":
                    pending = update[0].value
                    continue
                for ev in mapper.events(node, update):
                    if ev["kind"] == "report":
                        ev["data"]["usage"] = getattr(llm, "usage", None)
                    on_event(ev)
            if cancelled():
                stopped = True
                break
        if stopped or pending is None:
            break
        decision = on_gate(pending)
        if cancelled():
            stopped = True
            break
        inp = Command(resume=decision)

    final = dict(graph.get_state(cfg).values)
    if stopped and not final.get("aborted"):
        final["aborted"] = True
        on_event(mapper.event("aborted"))
    exit_code = 1 if (final.get("fatal") or final.get("aborted")) else 0
    return RunResult(run_id=run_id, run_dir=run_dir, final_state=final, exit_code=exit_code)
