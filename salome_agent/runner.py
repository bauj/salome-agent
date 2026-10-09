import os
import threading
import time
import uuid
from dataclasses import dataclass


@dataclass
class RunResult:
    run_id: str
    run_dir: str
    final_state: dict
    exit_code: int


class EventMapper:
    """Turns graph.stream(stream_mode=["updates", "custom"]) chunks into UI events
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

    def reasoning(self, module, call: int, channel: str, text: str) -> dict:
        """A streamed chunk of model reasoning or tool-turn prose. No plan this
        turn and not the planner means an amend (chat edit) turn."""
        if module == "planner":
            source, index = "planner", None
        elif module == "vision":
            source, index = "vision", None
        elif self.plan:
            source, index = "stage", self.idx
        else:
            source, index = "edit", None
        return self.event("reasoning", source=source, index=index, call=call,
                          channel=channel, text=text)

    def events(self, node: str, update: dict | None) -> list[dict]:
        update = update or {}
        out = []
        if node == "amend":
            for e in update.get("history") or []:
                if e.get("event") in ("graph_edit", "replay"):
                    out.append(self.event(e["event"],
                                          **{k: v for k, v in e.items() if k != "event"}))
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
        if node == "run_stage":
            err = update.get("last_error")
            if err:
                out.append(self.event("stage_attempt", index=self.idx,
                                      phase=err.get("phase") or node, ok=False,
                                      error=err.get("message")))
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


def show_results(client, build_graph: dict | None) -> None:
    """Best-effort: draw the model's result objects in the GUI viewer (a no-op
    headless). Publishing to the study alone leaves them hidden."""
    from salome_agent.build_graph import BuildGraph
    from salome_agent.logs import log

    if client is None or not build_graph:
        return
    names = BuildGraph.from_dict(build_graph).result_names()
    if not names:
        return
    try:
        client.display(names)
    except Exception as e:  # noqa: BLE001 — display must never fail a run
        log.warning("could not display the results: %s", e)


def initial_state(request: str, config: dict, client, run_id: str, batch: bool,
                  reference_images: list[str] | None = None) -> dict:
    return {
        "user_request": request, "reference_images": reference_images,
        "image_spec": None, "config": {**config, "batch": batch},
        "session": _session_handle(client), "plan": [], "plan_approved": False,
        "stage_idx": 0, "stages": {}, "last_error": None, "history": [],
        "study_ns": "salome_agent_run", "final_script": None, "report": None,
        "fatal": None, "aborted": False, "run_id": run_id,
        "build_graph": None, "turn_idx": 0, "selection": None, "turn_outcome": None,
        "replan_context": None, "degraded": None,
    }


class ReasoningRelay:
    """LLMClient.on_stream -> `reasoning` events, merged so a long reasoning
    stream sends one event per `interval` rather than one per token."""

    def __init__(self, mapper, on_event, clock=time.monotonic, interval: float = 0.5):
        self.mapper, self.on_event = mapper, on_event
        self.clock, self.interval = clock, interval
        self.call = 0
        self.module = None
        self.channel = None
        self.buf: list[str] = []
        self.last = 0.0

    def __call__(self, ev: dict) -> None:
        kind = ev["type"]
        if kind == "start":
            self.call += 1
            self.module = ev.get("module")
            self.channel, self.buf, self.last = None, [], self.clock()
        elif kind == "delta":
            if ev["channel"] != self.channel:
                self._flush()
                self.channel = ev["channel"]
            self.buf.append(ev["text"])
            if self.clock() - self.last >= self.interval:
                self._flush()
        elif kind == "end":
            self._flush()

    def _flush(self) -> None:
        if self.buf:
            self.on_event(self.mapper.reasoning(self.module, self.call, self.channel,
                                                "".join(self.buf)))
            self.buf = []
        self.last = self.clock()


def drive(app, inp, cfg, mapper, llm, on_event, on_gate, cancel) -> dict:
    from langgraph.types import Command

    from salome_agent.llm import LLMCancelled

    def cancelled() -> bool:
        return cancel is not None and cancel.is_set()

    # nodes read it too, so a long tool loop stops mid-node (task 256); the
    # client reads it per streamed line, so a long model response does (GH #47)
    cfg = {**cfg, "configurable": {**cfg["configurable"], "cancel": cancel}}
    llm.cancel = cancel
    llm.on_stream = ReasoningRelay(mapper, on_event)

    stopped = False
    try:
        while True:
            pending = None
            for mode, chunk in app.stream(inp, config=cfg, stream_mode=["updates", "custom"]):
                if mode == "custom":
                    on_event(mapper.event(chunk["kind"], **(chunk.get("data") or {})))
                    continue
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
    except LLMCancelled:
        stopped = True
    finally:
        llm.cancel = None
        llm.on_stream = None

    final = dict(app.get_state(cfg).values)
    if stopped and not final.get("aborted"):
        final["aborted"] = True
        on_event(mapper.event("aborted"))
    return final


def run_request(request: str, config: dict, *, client, on_event, on_gate,
                cancel: threading.Event | None = None, batch: bool = False,
                verbose: int = 0, run_id: str | None = None,
                request_timeout: float | None = None,
                reference_images: list[str] | None = None) -> RunResult:
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
    inp = initial_state(request, config, client, run_id, batch,
                        reference_images=reference_images)

    final = drive(graph, inp, cfg, mapper, llm, on_event, on_gate, cancel)
    if not final.get("aborted"):
        show_results(client, final.get("build_graph"))
    exit_code = 1 if (final.get("fatal") or final.get("aborted")) else 0
    return RunResult(run_id=run_id, run_dir=run_dir, final_state=final, exit_code=exit_code)
