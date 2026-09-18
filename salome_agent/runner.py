import os
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
