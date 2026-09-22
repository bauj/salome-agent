import os
import uuid
from dataclasses import dataclass

from salome_agent.build_graph import BuildGraph
from salome_agent.executor_client import SessionDied, SessionPoisoned
from salome_agent.replay import ReplayFailed, rebuild_all, sync
from salome_agent.runner import EventMapper, drive, initial_state


@dataclass
class TurnResult:
    status: str
    message: str
    final_state: dict


class Conversation:
    """Everything that outlives a turn: one executor connection (the caller
    owns it and closes it), one LLM client, the build graph, the turn history
    and one run dir."""

    def __init__(self, config: dict, client, run_id: str, *, batch: bool = False,
                 verbose: int = 0, request_timeout: float | None = None, llm=None):
        from salome_agent import cli

        self.config = config
        self.client = client
        self.run_id = run_id
        self.batch = batch
        work_dir = os.path.expanduser(config.get("work_dir") or ".")
        self.run_dir = os.path.abspath(os.path.join(work_dir, run_id))
        trace = os.path.join(self.run_dir, "llm_trace.jsonl") if verbose >= 2 else None
        self.llm = llm or cli.build_llm(config, request_timeout, trace, run_id)
        self.graph = BuildGraph()
        self.turn_idx = 0
        self.history: list[dict] = []
        self.degraded = False
        self.ended = False
        self._snapshots: list[tuple] = []
        self._stray: set[str] = set()

    def _event(self, on_event, kind, **data):
        on_event({"kind": kind, "run_id": self.run_id, "data": data})

    def _write_outputs(self) -> None:
        import json

        from salome_agent.nodes.assembler import render_script

        os.makedirs(self.run_dir, exist_ok=True)
        with open(os.path.join(self.run_dir, "result.py"), "w") as f:
            f.write(render_script(self.graph, self.run_id))
        with open(os.path.join(self.run_dir, "build_graph.json"), "w") as f:
            json.dump(self.graph.to_dict(), f, indent=1)

    def turn(self, text: str, *, selection=None, on_event, on_gate, cancel=None) -> TurnResult:
        if self.ended:
            return TurnResult("ended", "the SALOME session is gone; start a new conversation", {})
        if self.degraded:
            return TurnResult("degraded", "the study no longer matches the model; run /rebuild", {})
        from salome_agent.graph import build_graph

        n = self.turn_idx
        self._event(on_event, "turn_start", turn=n, text=text)
        before = self.graph.copy()
        inp = initial_state(text, self.config, self.client, self.run_id, self.batch)
        inp.update(build_graph=self.graph.to_dict(), turn_idx=n, selection=selection,
                   history=list(self.history))
        cfg = {"configurable": {"thread_id": uuid.uuid4().hex, "llm_client": self.llm,
                                "executor_client": self.client}}
        final = drive(build_graph(batch=self.batch), inp, cfg,
                      EventMapper(self.run_id, self.run_dir), self.llm, on_event, on_gate, cancel)
        result = self._settle(final, before)
        outcome = final.get("turn_outcome") or {}
        self.history.append({"event": "turn", "turn": n, "text": text,
                             "status": result.status, "summary": outcome.get("summary")})
        self._event(on_event, "turn_done", turn=n, status=result.status, message=result.message)
        return result

    def _settle(self, final: dict, before: BuildGraph) -> TurnResult:
        after = BuildGraph.from_dict(final.get("build_graph")) if final.get("build_graph") else before
        fatal = final.get("fatal") or ""
        outcome = final.get("turn_outcome") or {}
        if final.get("degraded"):
            self.degraded = True
            self._stray |= set(after.all_out_vars())
            self.graph = before
            return TurnResult("degraded", f"{fatal} — run /rebuild", final)
        if "SALOME agent server" in fatal:
            self.ended = True
            return TurnResult("ended", fatal, final)
        if fatal or final.get("aborted") or outcome.get("exhausted"):
            message = fatal or ("turn cancelled" if final.get("aborted") else
                                "ran out of tool iterations")
            return self._rollback(after, before, message, final)
        self._snapshots.append((before, self.turn_idx, len(self.history)))
        self.graph = after
        self.turn_idx += 1
        if outcome.get("question"):
            return TurnResult("needs_input", outcome["question"], final)
        return TurnResult("ok", outcome.get("summary") or "done", final)

    def _rollback(self, after, before, message, final) -> TurnResult:
        try:
            self.graph = sync(self.client, current=after, target=before)
        except (SessionDied, SessionPoisoned) as e:
            self.ended = True
            return TurnResult("ended", f"{message}; session lost during rollback: {e}", final)
        except ReplayFailed as e:
            self.degraded = True
            self._stray |= set(after.all_out_vars())
            self.graph = before
            return TurnResult("degraded", f"{message}; rollback failed: {e} — run /rebuild", final)
        self._write_outputs()
        return TurnResult("failed", message, final)

    def undo(self) -> TurnResult:
        if self.ended or self.degraded:
            return self.turn("", on_event=lambda e: None, on_gate=None)
        if not self._snapshots:
            return TurnResult("failed", "nothing to undo", {})
        target, turn_idx, hist_len = self._snapshots.pop()
        try:
            self.graph = sync(self.client, current=self.graph, target=target)
        except ReplayFailed as e:
            self.degraded = True
            self.graph = target
            return TurnResult("degraded", f"undo failed: {e} — run /rebuild", {})
        self.turn_idx = turn_idx
        self.history = self.history[:hist_len]
        self._write_outputs()
        return TurnResult("ok", f"undone back to turn {turn_idx}", {})

    def rebuild(self) -> TurnResult:
        if self.ended:
            return TurnResult("ended", "the SALOME session is gone", {})
        try:
            self.graph = rebuild_all(self.client, self.graph, extra_names=sorted(self._stray))
        except ReplayFailed as e:
            return TurnResult("degraded", f"rebuild failed: {e}", {})
        self.degraded = False
        self._stray.clear()
        self._write_outputs()
        return TurnResult("ok", f"rebuilt {len(self.graph.nodes)} node(s)", {})
