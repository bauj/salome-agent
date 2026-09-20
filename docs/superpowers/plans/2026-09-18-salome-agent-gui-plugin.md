# salome-agent GUI Plugin Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A SALOME GUI plugin that starts (or reuses) the in-GUI agent server and opens a PyQt5 chat window in which each message drives one salome-agent run, shown as a live stage timeline with in-chat plan / retry gates.

**Architecture:** salome-agent gains a library entry point `salome_agent.runner.run_request()` that streams LangGraph node updates as UI events and hands interrupts to an `on_gate` callback; the CLI is moved onto it. A new sibling repo `salome-agent-gui` (salome-agent as a git submodule) holds the PyQt5 window (runs in its own venv, drives `run_request` on a `QThread`) and a thin `salome_plugins.py` + `sagui_launcher.py` pair that runs inside SALOME. SAT builds and installs it, mirroring the `sarag` product.

**Tech Stack:** Python ≥3.10, LangGraph 1.2 (`graph.stream(stream_mode="updates")`, `Command(resume=...)`), PyQt5 (`QTextBrowser.setMarkdown`, no QtWebEngine), pytest, pytest-qt, SAT (SALOME build tool) pyconf + bash compile script.

**Spec:** `docs/superpowers/specs/2026-09-18-salome-agent-gui-plugin-design.md`

## Global Constraints

- Python floor 3.10 (salome-agent `requires-python = ">=3.10"`); dev box runs 3.14.
- Both SALOME 9.16 and master: any KERNEL import uses the `from salome.kernel import X` → `import X` fallback (see commit 596b16e).
- No SALOME install path (e.g. the local `SALOME-...-FD44` tree) written into salome-agent or salome-agent-gui code, comments or docs — resolve from env/config.
- Provider specifics stay in config, never hardcoded.
- Commits: subject line only, no body, no trailers. Stage named files only — never `git add -A` / `git add .` (config files hold API keys).
- salome-agent work happens on a feature branch, merged to `main` with `--ff-only` once green.
- SAT_SALOME edits happen on a new branch created before the first SAT edit.
- The API key is never logged or echoed by the GUI.
- User config path: `~/.config/salome/salome-agent.json`.
- Plugin install dir inside the SAT product: `share/salome/plugins/salome_agent_gui/`.
- Env vars set by the SAT env script: `SALOME_AGENT_GUI_ROOT_DIR`, `SALOME_AGENT_GUI`.
- Env vars stripped before spawning the window (same list as sarag): `PYTHONPATH`, `QT_PLUGIN_PATH`, `QT_QPA_PLATFORM_PLUGIN_PATH`, `QT_QPA_FONTDIR`, `LD_LIBRARY_PATH`.

## Refinement vs. spec (decided while planning)

- **One executor connection per run, not per window session.** The server's `hello` handshake resets geom counters and bindings per connection and the CLI already connects once per run; holding a connection for the whole window session would also block a concurrent CLI run. The window's Connect button only probes the handshake file; `RunWorker` connects at run start and closes at run end.
- **Events are plain dicts** `{"kind": str, "run_id": str, "data": dict}` so they cross the Qt signal boundary as `pyqtSignal(dict)`.

## File map

**salome-agent (this repo)**
- Create `salome_agent/runner.py` — `RunResult`, `EventMapper`, `run_request()`.
- Modify `salome_agent/interactive.py` — extract `normalize_plan()`, add `terminal_gate()`, delete `resume_interactive()`.
- Modify `salome_agent/cli.py` — `main()` calls `run_request()`.
- Create `tests/unit/test_runner.py`; modify `tests/unit/test_interactive.py`, `tests/unit/test_cli.py`.

**salome-agent-gui (new, `../salome-agent-gui`)**
- `salome-agent/` — submodule.
- `pyproject.toml`, `pytest.ini`, `.gitignore`, `README.md`, `salome-agent-gui.sh`.
- `salome_agent_gui/__init__.py`, `config_store.py`, `timeline.py`, `run_worker.py`, `chat_view.py`, `dialogs.py`, `main_window.py`, `app.py`.
- `sagui_launcher.py`, `salome_plugins.py` (repo root; copied into the SALOME plugins dir by SAT).
- `tests/test_config_store.py`, `test_timeline.py`, `test_run_worker.py`, `test_main_window.py`, `test_sagui_launcher.py`.

**SAT_SALOME (new branch)**
- `products/salome_agent_gui.pyconf`, `products/compil_scripts/salome_agent_gui.sh`, `products/env_scripts/salome_agent_gui.py`.
- Modify `applications/SALOME-master-native.pyconf`, `applications/SALOME-9.16.0-native.pyconf`.

---

## Part 1 — salome-agent: runner + CLI

Before Task 1: `git switch main && git merge --ff-only spec/gui-plugin && git switch -c feat/gui-runner`.

### Task 1: `EventMapper` — graph updates → UI events

**Files:**
- Create: `salome_agent/runner.py`
- Test: `tests/unit/test_runner.py`

**Interfaces:**
- Consumes: LangGraph `stream_mode="updates"` chunks, shape `{node_name: update_dict_or_None}`; node names `plan`, `plan_gate`, `run_stage`, `verify_static`, `verify_exec`, `retry_gate`, `advance`, `skip`, `assemble`, `report`. `advance`/`skip` updates carry `stage_idx` (the NEXT index) and `stages[idx]` (int keys; `result_names` on the accepted stage). Failures carry `last_error = {"phase", "message", "offending_line", "traceback"}`. Gates on abort return `{"aborted": True}`; endpoint failures return `{"fatal": str}`.
- Produces: `EventMapper(run_id: str, run_dir: str)` with `.events(node: str, update: dict | None) -> list[dict]` and `.event(kind: str, **data) -> dict`. Event kinds and `data` keys:
  - `plan`: `stages` (list of stage-plan dicts)
  - `stage_start`: `index`, `module`, `intent`
  - `stage_attempt`: `index`, `phase` (`"static"`/`"exec"`/other), `ok` (bool), `error` (str | None)
  - `stage_done`: `index`, `names` (list[str])
  - `stage_skipped`: `index`
  - `report`: `result_py`, `report_md`, `text` (report markdown); `usage` added by `run_request`
  - `fatal`: `message`
  - `aborted`: no keys

- [ ] **Step 1: Write the failing tests**

`tests/unit/test_runner.py`:

```python
from salome_agent.runner import EventMapper

PLAN = [{"module": "geom", "intent": "box", "params": {}, "inputs": []},
        {"module": "smesh", "intent": "mesh it", "params": {}, "inputs": []}]


def _kinds(evs):
    return [e["kind"] for e in evs]


def test_plan_update_emits_plan_event():
    m = EventMapper("r1", "/w/r1")
    evs = m.events("plan", {"plan": PLAN, "stages": {}, "stage_idx": 0})
    assert _kinds(evs) == ["plan"]
    assert evs[0] == {"kind": "plan", "run_id": "r1", "data": {"stages": PLAN}}


def test_edited_plan_at_the_gate_re_emits_plan():
    m = EventMapper("r1", "/w/r1")
    m.events("plan", {"plan": PLAN})
    evs = m.events("plan_gate", {"plan": PLAN[:1], "stages": {}, "stage_idx": 0})
    assert _kinds(evs) == ["plan"]
    assert evs[0]["data"]["stages"] == PLAN[:1]


def test_first_run_stage_emits_stage_start_once():
    m = EventMapper("r1", "/w/r1")
    m.events("plan", {"plan": PLAN})
    first = m.events("run_stage", {"stages": {}})
    again = m.events("run_stage", {"stages": {}})
    assert _kinds(first) == ["stage_start"]
    assert first[0]["data"] == {"index": 0, "module": "geom", "intent": "box"}
    assert again == []


def test_failed_verify_emits_failed_attempt():
    m = EventMapper("r1", "/w/r1")
    m.events("plan", {"plan": PLAN})
    m.events("run_stage", {})
    err = {"phase": "exec", "message": "boom", "offending_line": 3, "traceback": None}
    evs = m.events("verify_exec", {"stages": {}, "last_error": err})
    assert evs == [{"kind": "stage_attempt", "run_id": "r1",
                    "data": {"index": 0, "phase": "exec", "ok": False, "error": "boom"}}]


def test_passing_verify_static_with_no_update_emits_ok_attempt():
    m = EventMapper("r1", "/w/r1")
    m.events("plan", {"plan": PLAN})
    evs = m.events("verify_static", None)
    assert evs[0]["data"] == {"index": 0, "phase": "static", "ok": True, "error": None}


def test_advance_emits_stage_done_and_moves_index():
    m = EventMapper("r1", "/w/r1")
    m.events("plan", {"plan": PLAN})
    m.events("run_stage", {})
    evs = m.events("advance", {"stages": {0: {"status": "ok", "result_names": ["box_1"]}},
                               "stage_idx": 1, "last_error": None})
    assert evs[0] == {"kind": "stage_done", "run_id": "r1",
                      "data": {"index": 0, "names": ["box_1"]}}
    nxt = m.events("run_stage", {})
    assert nxt[0]["data"]["index"] == 1


def test_skip_emits_stage_skipped():
    m = EventMapper("r1", "/w/r1")
    m.events("plan", {"plan": PLAN})
    evs = m.events("skip", {"stages": {0: {"status": "skipped"}}, "stage_idx": 1})
    assert _kinds(evs) == ["stage_skipped"]
    assert evs[0]["data"] == {"index": 0}


def test_fatal_and_aborted_and_report():
    m = EventMapper("r1", "/w/r1")
    assert m.events("plan", {"fatal": "LLM endpoint down"})[0] == {
        "kind": "fatal", "run_id": "r1", "data": {"message": "LLM endpoint down"}}
    assert _kinds(m.events("plan_gate", {"aborted": True})) == ["aborted"]
    rep = m.events("report", {"report": "# Report"})[0]
    assert rep["kind"] == "report"
    assert rep["data"] == {"result_py": "/w/r1/result.py", "report_md": "/w/r1/report.md",
                           "text": "# Report"}


def test_fatal_from_run_stage_is_not_also_an_attempt():
    m = EventMapper("r1", "/w/r1")
    m.events("plan", {"plan": PLAN})
    m.events("run_stage", {})
    assert _kinds(m.events("run_stage", {"fatal": "x"})) == ["fatal"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/test_runner.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'salome_agent.runner'`.

- [ ] **Step 3: Implement `EventMapper`**

`salome_agent/runner.py`:

```python
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit/test_runner.py -q`
Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add salome_agent/runner.py tests/unit/test_runner.py
git commit -m "feat: runner.EventMapper turns graph stream updates into UI events"
```

### Task 2: `run_request()` — stream loop, gates, cancel

**Files:**
- Modify: `salome_agent/runner.py`
- Test: `tests/unit/test_runner.py`

**Interfaces:**
- Consumes: `EventMapper` (Task 1); `salome_agent.cli.build_llm(config, request_timeout, trace_path, run_id)`; `salome_agent.graph.build_graph(batch)`; `ExecutorClient` attributes `.mode`, `.name_prefix` (client may be `None` → session mode `"none"`).
- Produces:
  ```python
  def run_request(request: str, config: dict, *, client, on_event, on_gate,
                  cancel: threading.Event | None = None, batch: bool = False,
                  verbose: int = 0, run_id: str | None = None,
                  request_timeout: float | None = None) -> RunResult
  ```
  `on_event(event: dict) -> None`; `on_gate(payload: dict) -> dict` where payload is `{"plan": [...]}` or `{"last_error": {...}}` and the return is one of `{"action": "approve"}`, `{"action": "edit", "plan": [...]}`, `{"action": "hint", "hint": str}`, `{"action": "skip"}`, `{"action": "abort"}`. `RunResult.exit_code` is `1` when the final state has `fatal` or `aborted`, else `0`.

- [ ] **Step 1: Write the failing tests** (append to `tests/unit/test_runner.py`)

```python
import threading
from unittest.mock import MagicMock

import requests

from salome_agent.runner import run_request

ONE_STAGE = {"stages": [{"module": "geom", "intent": "box", "params": {}, "inputs": []}]}


def _config(tmp_path):
    return {"llm": {"base_url": "http://x", "model": "m"}, "work_dir": str(tmp_path),
            "policy": {"tool_iteration_cap": 3, "on_stage_failure": "abort"}}


def _fake_llm(monkeypatch, llm):
    monkeypatch.setattr("salome_agent.cli.build_llm", lambda *a, **k: llm)


def _http_error(code):
    resp = requests.Response()
    resp.status_code = code
    return requests.HTTPError(f"{code} Server Error", response=resp)


def test_plan_gate_abort_emits_plan_then_aborted_and_writes_nothing(tmp_path, monkeypatch):
    llm = MagicMock()
    llm.chat.return_value = ONE_STAGE
    llm.usage = None
    _fake_llm(monkeypatch, llm)
    events, gates = [], []

    def on_gate(payload):
        gates.append(payload)
        return {"action": "abort"}

    res = run_request("make a box", _config(tmp_path), client=None,
                      on_event=events.append, on_gate=on_gate, run_id="r1")
    assert [e["kind"] for e in events] == ["plan", "aborted"]
    assert gates == [{"plan": ONE_STAGE["stages"]}]
    assert res.exit_code == 1
    assert res.run_id == "r1"
    assert res.run_dir == str(tmp_path / "r1")
    assert res.final_state["aborted"] is True
    assert list(tmp_path.iterdir()) == []


def test_edit_at_the_gate_re_prompts_with_the_edited_plan(tmp_path, monkeypatch):
    llm = MagicMock()
    llm.chat.return_value = ONE_STAGE
    llm.usage = None
    _fake_llm(monkeypatch, llm)
    edited = [{"module": "geom", "intent": "cylinder", "params": {}, "inputs": []}]
    answers = iter([{"action": "edit", "plan": edited}, {"action": "abort"}])
    gates = []

    def on_gate(payload):
        gates.append(payload)
        return next(answers)

    run_request("x", _config(tmp_path), client=None, on_event=lambda e: None,
                on_gate=on_gate, run_id="r1")
    assert gates[1] == {"plan": edited}


def test_endpoint_fatal_in_batch_emits_fatal_and_report(tmp_path, monkeypatch):
    llm = MagicMock()
    llm.chat.side_effect = _http_error(502)
    llm.usage = {"calls": 1}
    _fake_llm(monkeypatch, llm)
    events = []

    res = run_request("x", _config(tmp_path), client=None, on_event=events.append,
                      on_gate=lambda p: {"action": "abort"}, batch=True, run_id="r1")
    kinds = [e["kind"] for e in events]
    assert "fatal" in kinds and kinds[-1] == "report"
    assert events[-1]["data"]["usage"] == {"calls": 1}
    assert res.exit_code == 1
    assert (tmp_path / "r1" / "report.md").exists()


def test_cancel_before_the_gate_answers_ends_aborted_without_resuming(tmp_path, monkeypatch):
    llm = MagicMock()
    llm.chat.return_value = ONE_STAGE
    llm.usage = None
    _fake_llm(monkeypatch, llm)
    cancel = threading.Event()
    events = []

    def on_gate(payload):
        cancel.set()
        return {"action": "approve"}

    res = run_request("x", _config(tmp_path), client=None, on_event=events.append,
                      on_gate=on_gate, cancel=cancel, run_id="r1")
    assert [e["kind"] for e in events] == ["plan", "aborted"]
    assert res.final_state["aborted"] is True
    assert res.exit_code == 1
    assert llm.chat.call_count == 1          # planner only, no specialist turn


def test_client_supplies_the_session_handle(tmp_path, monkeypatch):
    llm = MagicMock()
    llm.chat.return_value = ONE_STAGE
    llm.usage = None
    _fake_llm(monkeypatch, llm)
    client = MagicMock(mode="gui", name_prefix="p_")
    res = run_request("x", _config(tmp_path), client=client, on_event=lambda e: None,
                      on_gate=lambda p: {"action": "abort"}, run_id="r1")
    assert res.final_state["session"]["mode"] == "gui"
    assert res.final_state["session"]["name_prefix"] == "p_"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/test_runner.py -q`
Expected: the 5 new tests FAIL with `ImportError: cannot import name 'run_request'`.

- [ ] **Step 3: Implement `run_request`** (append to `salome_agent/runner.py`; add `import threading` and `import uuid` to the imports at the top)

```python
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
    run_dir = os.path.join(work_dir, run_id)
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit/test_runner.py -q`
Expected: 14 passed.

- [ ] **Step 5: Commit**

```bash
git add salome_agent/runner.py tests/unit/test_runner.py
git commit -m "feat: runner.run_request streams a run as events with gate and cancel callbacks"
```

### Task 3: Move the CLI onto the runner

**Files:**
- Modify: `salome_agent/interactive.py` (extract `normalize_plan`, add `terminal_gate`, delete `resume_interactive`)
- Modify: `salome_agent/cli.py:60-159` (`main`)
- Test: `tests/unit/test_interactive.py`, `tests/unit/test_cli.py`

**Interfaces:**
- Consumes: `run_request`, `RunResult` (Task 2).
- Produces:
  - `salome_agent.interactive.normalize_plan(edited) -> list[dict] | None` — returns the plan with `intent`/`params`/`inputs`/`tools` defaulted, or `None` when not a non-empty list of dicts with `module` in `geom|smesh|paravis`. Used by the GUI plan editor (Task 9).
  - `salome_agent.interactive.terminal_gate(payload, read=input, stream=sys.stderr, edit_fn=_edit_plan_in_editor) -> dict`.

- [ ] **Step 1: Write the failing tests**

In `tests/unit/test_interactive.py`: delete `_FakeInterrupt`, `_FakeGraph`, `test_resume_interactive_plan_then_retry_then_done`, `test_resume_interactive_edit_then_approve`, `test_resume_interactive_no_interrupt_is_passthrough`, remove `resume_interactive` from the import line, add `normalize_plan, terminal_gate` to it, and append:

```python
def test_normalize_plan_fills_defaults():
    out = normalize_plan([{"module": "geom"}])
    assert out == [{"module": "geom", "intent": "", "params": {}, "inputs": [], "tools": []}]


def test_normalize_plan_rejects_bad_shapes():
    assert normalize_plan([]) is None
    assert normalize_plan({"module": "geom"}) is None
    assert normalize_plan([{"module": "cad"}]) is None
    assert normalize_plan(["geom"]) is None


def test_terminal_gate_routes_plan_payload():
    d = terminal_gate({"plan": []}, read=_reader(["y"]), stream=io.StringIO())
    assert d == {"action": "approve"}


def test_terminal_gate_routes_retry_payload():
    d = terminal_gate({"last_error": {"message": "boom"}}, read=_reader(["s"]),
                      stream=io.StringIO())
    assert d == {"action": "skip"}


def test_terminal_gate_eof_aborts():
    def _raise(_p=""):
        raise EOFError

    assert terminal_gate({"plan": []}, read=_raise, stream=io.StringIO()) == {"action": "abort"}


def test_terminal_gate_unknown_payload_raises():
    import pytest

    with pytest.raises(RuntimeError):
        terminal_gate({"other": 1}, read=_reader([]), stream=io.StringIO())
```

In `tests/unit/test_cli.py` append:

```python
def test_main_no_session_delegates_to_run_request(tmp_path, monkeypatch, capsys):
    import json

    from salome_agent import cli
    from salome_agent.runner import RunResult

    cfg = tmp_path / "c.json"
    cfg.write_text(json.dumps({"llm": {"base_url": "http://x", "model": "m"},
                               "salome": {"root": "/x"},
                               "policy": {"on_stage_failure": "abort"},
                               "work_dir": str(tmp_path)}))
    seen = {}

    def fake_run(request, config, **kw):
        seen.update(kw, request=request, config=config)
        return RunResult(run_id=kw["run_id"], run_dir=str(tmp_path / kw["run_id"]),
                         final_state={"report": "REPORT-TEXT"}, exit_code=0)

    monkeypatch.setattr("salome_agent.runner.run_request", fake_run)
    rc = cli.main(["make a box", "--no-session", "--batch", "--config", str(cfg),
                   "--on-failure", "continue"])
    assert rc == 0
    assert seen["request"] == "make a box"
    assert seen["client"] is None and seen["batch"] is True
    assert seen["config"]["policy"]["on_stage_failure"] == "continue"
    assert "REPORT-TEXT" in capsys.readouterr().out


def test_main_returns_1_on_abort(tmp_path, monkeypatch):
    import json

    from salome_agent import cli
    from salome_agent.runner import RunResult

    cfg = tmp_path / "c.json"
    cfg.write_text(json.dumps({"llm": {"base_url": "http://x", "model": "m"},
                               "salome": {"root": "/x"}, "policy": {}}))
    monkeypatch.setattr(
        "salome_agent.runner.run_request",
        lambda request, config, **kw: RunResult(kw["run_id"], "/x", {"aborted": True}, 1))
    assert cli.main(["x", "--no-session", "--config", str(cfg)]) == 1
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/test_interactive.py tests/unit/test_cli.py -q`
Expected: FAIL — `ImportError: cannot import name 'normalize_plan'` (interactive) and the two CLI tests fail because `main` still calls `graph.invoke`.

- [ ] **Step 3: Implement**

In `salome_agent/interactive.py`, replace the validation tail of `_edit_plan_in_editor` and add the two functions; delete `resume_interactive` entirely:

```python
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
```

and in `_edit_plan_in_editor` the lines after `os.unlink(path)` become:

```python
    return normalize_plan(edited)
```

```python
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
```

In `salome_agent/cli.py` `main()`: keep everything up to and including the session connect block, then replace from `trace_path = ...` through the `if final.get("aborted"):` block with:

```python
    from salome_agent import runner
    from salome_agent.interactive import terminal_gate

    try:
        result = runner.run_request(
            args.request, config, client=client, on_event=lambda ev: None,
            on_gate=terminal_gate, batch=args.batch, verbose=args.verbose,
            run_id=run_id, request_timeout=args.timeout)
    finally:
        if client is not None:
            try:
                client.close()
            except Exception:
                pass
    final = result.final_state

    if final.get("aborted"):
        log.info("run aborted at your request - no result.py / report.md written")
        return 1
```

Keep the rest of `main` (paths, handshake read, summary print, `return 1 if final.get("fatal") else 0`) unchanged. Remove the now-unused `from salome_agent.graph import build_graph` import from `main`. (Referencing `runner.run_request` through the module keeps the monkeypatch in the tests effective.)

- [ ] **Step 4: Run the full unit suite**

Run: `.venv/bin/python -m pytest tests/unit -q`
Expected: all pass (365 before this plan, +14 runner, +6 interactive −3 removed, +2 cli = 384).

- [ ] **Step 5: CLI smoke without a server**

Run: `echo n | .venv/bin/salome-agent "a 10mm cube" --no-session --config config.json`
Expected: planner runs, "Proposed plan:" printed, `n` aborts, exit code 1, log line "run aborted at your request". (Needs a reachable LLM endpoint in `config.json`; if none, skip and note it.)

- [ ] **Step 6: Commit and merge**

```bash
git add salome_agent/interactive.py salome_agent/cli.py tests/unit/test_interactive.py tests/unit/test_cli.py
git commit -m "refactor: CLI drives runs through runner.run_request with a terminal gate"
git switch main && git merge --ff-only feat/gui-runner && git branch -d feat/gui-runner
```

The user pushes `main` (the submodule in Part 2 must point at a pushed commit before a SAT build).

---

## Part 2 — salome-agent-gui repo

### Task 4: Repo scaffold + config store

**Files:**
- Create: `../salome-agent-gui/{.gitignore,pyproject.toml,pytest.ini,README.md}`, `salome_agent_gui/__init__.py`, `salome_agent_gui/config_store.py`, `tests/test_config_store.py`
- Submodule: `salome-agent`

**Interfaces:**
- Consumes: `salome_agent.config.load_config(path: str) -> dict` (raises `ConfigError` on missing `llm.base_url`, `llm.model`, `salome.root`).
- Produces (`salome_agent_gui.config_store`):
  - `USER_CONFIG: Path` = `~/.config/salome/salome-agent.json`
  - `KEY_FIELDS = ("base_url", "model", "api_key")`
  - `seed_candidates() -> list[Path]`
  - `ensure_user_config(path: Path | None = None, seeds: list[Path] | None = None) -> Path`
  - `read_key_fields(path: Path) -> dict[str, str]`
  - `save_key_fields(path: Path, fields: dict) -> None`
  - `save_raw(path: Path, text: str) -> None` (raises `ValueError` on bad JSON, `ConfigError` on missing keys; file untouched on error)
  - `load(path: Path) -> dict`

- [ ] **Step 1: Create the repo, submodule and dev venv**

```bash
mkdir ../salome-agent-gui && cd ../salome-agent-gui && git init -b main
git submodule add https://github.com/bauj/salome-agent.git salome-agent
git -C salome-agent fetch ../../salome-agent main && git -C salome-agent checkout FETCH_HEAD
mkdir -p salome_agent_gui tests && touch salome_agent_gui/__init__.py tests/__init__.py
```

`.gitignore`:
```
__pycache__/
*.egg-info/
.venv/
build/
```

`pyproject.toml`:
```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "salome-agent-gui"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = ["salome-agent", "PyQt5"]

[project.optional-dependencies]
dev = ["pytest", "pytest-qt"]

[project.scripts]
salome-agent-gui = "salome_agent_gui.app:main"

[tool.setuptools]
packages = ["salome_agent_gui"]

[tool.setuptools.package-data]
salome_agent_gui = ["config.example.json"]
```

`pytest.ini`:
```ini
[pytest]
testpaths = tests
qt_api = pyqt5
pythonpath = .
```

`README.md`:
```markdown
# salome-agent-gui

SALOME GUI plugin + chat window for [salome-agent](salome-agent/).
Tools > Plugins > SALOME Agent starts the in-GUI agent server and opens the chat;
each message is one agent run (plan -> stages -> result.py).

Dev: `python3 -m venv .venv && .venv/bin/pip install -e ./salome-agent -e '.[dev]'`,
then `QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest`.
Installed by SAT (product `salome_agent_gui`).
```

```bash
python3 -m venv .venv && .venv/bin/pip install -e ./salome-agent -e '.[dev]'
```

- [ ] **Step 2: Write the failing tests** — `tests/test_config_store.py`

```python
import json
import stat

import pytest

from salome_agent_gui import config_store as cs

SEED = {"llm": {"base_url": "", "api_key": "", "model": ""},
        "salome": {"root": "/path/to/your/SALOME"}, "policy": {}}


def _seed(tmp_path):
    p = tmp_path / "seed.json"
    p.write_text(json.dumps(SEED))
    return p


def test_ensure_user_config_seeds_once_with_private_mode(tmp_path):
    user = tmp_path / "cfg" / "salome-agent.json"
    assert cs.ensure_user_config(user, [tmp_path / "missing.json", _seed(tmp_path)]) == user
    assert json.loads(user.read_text()) == SEED
    assert stat.S_IMODE(user.stat().st_mode) == 0o600
    user.write_text(json.dumps({"kept": True}))
    cs.ensure_user_config(user, [_seed(tmp_path)])
    assert json.loads(user.read_text()) == {"kept": True}


def test_ensure_user_config_without_seed_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        cs.ensure_user_config(tmp_path / "u.json", [tmp_path / "nope.json"])


def test_key_fields_round_trip_leaves_other_keys(tmp_path):
    user = cs.ensure_user_config(tmp_path / "u.json", [_seed(tmp_path)])
    cs.save_key_fields(user, {"base_url": "http://h/v1", "model": "qwen", "api_key": "k"})
    assert cs.read_key_fields(user) == {"base_url": "http://h/v1", "model": "qwen", "api_key": "k"}
    assert json.loads(user.read_text())["salome"] == SEED["salome"]


def test_save_raw_rejects_bad_json_and_missing_keys(tmp_path):
    from salome_agent.config import ConfigError

    user = cs.ensure_user_config(tmp_path / "u.json", [_seed(tmp_path)])
    before = user.read_text()
    with pytest.raises(ValueError):
        cs.save_raw(user, "{not json")
    with pytest.raises(ConfigError):
        cs.save_raw(user, json.dumps({"llm": {"base_url": "x"}}))
    assert user.read_text() == before
    good = dict(SEED, llm={"base_url": "http://h", "model": "m"})
    cs.save_raw(user, json.dumps(good))
    assert cs.load(user)["llm"]["model"] == "m"


def test_seed_candidates_include_the_submodule_example():
    names = [str(p) for p in cs.seed_candidates()]
    assert any(n.endswith("salome-agent/config.example.json") for n in names)
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest tests/test_config_store.py -q`
Expected: FAIL — `ImportError: cannot import name 'config_store'`.

- [ ] **Step 4: Implement** — `salome_agent_gui/config_store.py`

```python
import json
import os
import tempfile
from pathlib import Path

USER_CONFIG = Path.home() / ".config" / "salome" / "salome-agent.json"
KEY_FIELDS = ("base_url", "model", "api_key")


def seed_candidates() -> list[Path]:
    here = Path(__file__).resolve().parent
    return [here / "config.example.json",                       # SAT install copies it here
            here.parent / "salome-agent" / "config.example.json"]  # dev checkout


def _write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)


def ensure_user_config(path: Path | None = None, seeds: list[Path] | None = None) -> Path:
    path = Path(path or USER_CONFIG)
    if path.exists():
        return path
    for seed in seeds if seeds is not None else seed_candidates():
        if Path(seed).exists():
            _write(path, json.loads(Path(seed).read_text()))
            return path
    raise FileNotFoundError(f"no config.example.json found to seed {path}")


def read_key_fields(path: Path) -> dict:
    llm = json.loads(Path(path).read_text()).get("llm", {})
    return {k: llm.get(k, "") or "" for k in KEY_FIELDS}


def save_key_fields(path: Path, fields: dict) -> None:
    data = json.loads(Path(path).read_text())
    data.setdefault("llm", {}).update({k: fields[k] for k in KEY_FIELDS if k in fields})
    _write(Path(path), data)


def load(path: Path) -> dict:
    from salome_agent.config import load_config

    return load_config(str(path))


def save_raw(path: Path, text: str) -> None:
    data = json.loads(text)
    with tempfile.TemporaryDirectory() as d:
        probe = Path(d) / "probe.json"
        probe.write_text(json.dumps(data))
        load(probe)
    _write(Path(path), data)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest tests/test_config_store.py -q`
Expected: 5 passed.

- [ ] **Step 6: Commit**

```bash
git add .gitmodules salome-agent .gitignore pyproject.toml pytest.ini README.md salome_agent_gui/__init__.py tests/__init__.py salome_agent_gui/config_store.py tests/test_config_store.py
git commit -m "feat: repo scaffold with salome-agent submodule and user config store"
```

### Task 5: Timeline model + markdown rendering

**Files:**
- Create: `salome_agent_gui/timeline.py`
- Test: `tests/test_timeline.py`

**Interfaces:**
- Consumes: event dicts from Task 1; gate payloads from Task 2.
- Produces:
  - `StageRow(index: int, module: str, intent: str, status: str = "pending", attempts: int = 0, error: str | None = None)`; statuses `pending|running|ok|failed|skipped`.
  - `RunView` dataclass: `run_id`, `stages: list[StageRow]`, `gate: dict | None`, `objects: list[str]`, `result_py`, `report_md`, `usage`, `fatal`, `aborted: bool`, `finished: bool`; method `apply(event: dict) -> None`.
  - `render_run(view: RunView) -> str` (markdown). Gate links use hrefs `gate:approve`, `gate:edit`, `gate:abort`, `gate:hint`, `gate:skip`; file links use `file://` URLs.
  - `render_user(text: str) -> str`.

- [ ] **Step 1: Write the failing tests** — `tests/test_timeline.py`

```python
from salome_agent_gui.timeline import RunView, render_run, render_user

PLAN = [{"module": "geom", "intent": "two boxes"}, {"module": "smesh", "intent": "mesh"}]


def ev(kind, **data):
    return {"kind": kind, "run_id": "r1", "data": data}


def test_plan_then_stage_lifecycle():
    v = RunView()
    v.apply(ev("plan", stages=PLAN))
    assert [(s.module, s.status) for s in v.stages] == [("geom", "pending"), ("smesh", "pending")]
    v.apply(ev("stage_start", index=0, module="geom", intent="two boxes"))
    assert v.stages[0].status == "running"
    v.apply(ev("stage_attempt", index=0, phase="exec", ok=False, error="boom"))
    assert (v.stages[0].status, v.stages[0].attempts, v.stages[0].error) == ("failed", 1, "boom")
    v.apply(ev("stage_attempt", index=0, phase="exec", ok=True, error=None))
    assert (v.stages[0].status, v.stages[0].error) == ("running", None)
    v.apply(ev("stage_done", index=0, names=["box_1", "box_2"]))
    v.apply(ev("stage_skipped", index=1))
    assert [s.status for s in v.stages] == ["ok", "skipped"]
    assert v.objects == ["box_1", "box_2"]


def test_report_finishes_and_renders_links_and_usage():
    v = RunView()
    v.apply(ev("plan", stages=PLAN[:1]))
    v.apply(ev("stage_done", index=0, names=["box_1"]))
    v.apply(ev("report", result_py="/w/r1/result.py", report_md="/w/r1/report.md",
               text="# r", usage={"calls": 3, "prompt_tokens": 1200, "note": "x"}))
    assert v.finished
    md = render_run(v)
    assert "(file:///w/r1/result.py)" in md and "(file:///w/r1/report.md)" in md
    assert "`box_1`" in md
    assert "calls=3" in md and "prompt_tokens=1200" in md and "note" not in md


def test_plan_gate_renders_three_actions_and_clears():
    v = RunView()
    v.apply(ev("plan", stages=PLAN))
    v.gate = {"plan": PLAN}
    md = render_run(v)
    for href in ("gate:approve", "gate:edit", "gate:abort"):
        assert f"({href})" in md
    v.gate = None
    assert "gate:" not in render_run(v)


def test_retry_gate_renders_error_and_actions():
    v = RunView()
    v.gate = {"last_error": {"message": "fillet failed", "offending_line": 7}}
    md = render_run(v)
    assert "fillet failed" in md and "line 7" in md
    for href in ("gate:hint", "gate:skip", "gate:abort"):
        assert f"({href})" in md


def test_fatal_and_aborted_render():
    v = RunView()
    v.apply(ev("fatal", message="LLM endpoint returned HTTP 502"))
    assert "HTTP 502" in render_run(v)
    v.apply(ev("aborted"))
    assert v.finished and "aborted" in render_run(v).lower()


def test_event_for_unknown_index_extends_rows():
    v = RunView()
    v.apply(ev("stage_start", index=2, module="geom", intent="late"))
    assert len(v.stages) == 3 and v.stages[2].intent == "late"


def test_user_text_cannot_inject_gate_links():
    md = render_user("[x](gate:abort) <b>")
    assert "](gate:abort)" not in md
    assert "<b>" not in md
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest tests/test_timeline.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'salome_agent_gui.timeline'`.

- [ ] **Step 3: Implement** — `salome_agent_gui/timeline.py`

```python
from dataclasses import dataclass, field
from pathlib import Path

ICON = {"pending": "·", "running": "⟳", "ok": "✓", "failed": "✗", "skipped": "↷"}


@dataclass
class StageRow:
    index: int
    module: str
    intent: str
    status: str = "pending"
    attempts: int = 0
    error: str | None = None


@dataclass
class RunView:
    run_id: str | None = None
    stages: list[StageRow] = field(default_factory=list)
    gate: dict | None = None
    objects: list[str] = field(default_factory=list)
    result_py: str | None = None
    report_md: str | None = None
    usage: dict | None = None
    fatal: str | None = None
    aborted: bool = False
    finished: bool = False

    def _row(self, i: int, module: str = "?", intent: str = "") -> StageRow:
        while len(self.stages) <= i:
            self.stages.append(StageRow(len(self.stages), module, intent))
        return self.stages[i]

    def apply(self, event: dict) -> None:
        kind, d = event["kind"], event.get("data") or {}
        self.run_id = event.get("run_id") or self.run_id
        if kind == "plan":
            self.stages = [StageRow(i, s.get("module", "?"), s.get("intent", ""))
                           for i, s in enumerate(d["stages"])]
        elif kind == "stage_start":
            self._row(d["index"], d.get("module", "?"), d.get("intent", "")).status = "running"
        elif kind == "stage_attempt":
            r = self._row(d["index"])
            r.attempts += 1
            r.status, r.error = ("running", None) if d.get("ok") else ("failed", d.get("error"))
        elif kind == "stage_done":
            r = self._row(d["index"])
            r.status, r.error = "ok", None
            self.objects.extend(d.get("names") or [])
        elif kind == "stage_skipped":
            self._row(d["index"]).status = "skipped"
        elif kind == "report":
            self.result_py, self.report_md = d.get("result_py"), d.get("report_md")
            self.usage = d.get("usage")
            self.finished = True
        elif kind == "fatal":
            self.fatal = d.get("message")
        elif kind == "aborted":
            self.aborted = True
            self.finished = True


def _escape(text: str) -> str:
    for ch in "\\`*_[]()<>#":
        text = text.replace(ch, "\\" + ch)
    return text


def render_user(text: str) -> str:
    return f"**You:** {_escape(text)}"


def _file_link(label: str, path: str) -> str:
    return f"[{label}]({Path(path).as_uri()})"


def render_run(view: RunView) -> str:
    lines = ["**Agent**", ""]
    for s in view.stages:
        tries = f" (attempt {s.attempts})" if s.attempts > 1 else ""
        lines.append(f"{s.index + 1}. {ICON.get(s.status, '?')} `{s.module}` "
                     f"{_escape(s.intent)}{tries}")
        if s.error and s.status == "failed":
            lines.append(f"    ↳ {_escape(s.error)}")
    lines.append("")
    if view.gate and "plan" in view.gate:
        lines.append("**Approve this plan?** [Approve](gate:approve) · "
                     "[Edit](gate:edit) · [Abort](gate:abort)")
    elif view.gate and "last_error" in view.gate:
        err = view.gate["last_error"] or {}
        where = f" (line {err['offending_line']})" if err.get("offending_line") else ""
        lines.append(f"**Stage failed{where}:** {_escape(err.get('message') or '')}")
        lines.append("")
        lines.append("[Hint](gate:hint) · [Skip stage](gate:skip) · [Abort run](gate:abort)")
    if view.fatal:
        lines.append(f"**Run failed:** {_escape(view.fatal)}")
    if view.aborted:
        lines.append("_Run aborted._")
    elif view.finished:
        objs = ", ".join(f"`{n}`" for n in view.objects) or "none"
        lines.append(f"**Done.** Objects: {objs}")
        lines.append("")
        links = [_file_link(n, p) for n, p in (("result.py", view.result_py),
                                               ("report.md", view.report_md)) if p]
        lines.append(" · ".join(links))
        if view.usage:
            nums = ", ".join(f"{k}={v}" for k, v in view.usage.items()
                             if isinstance(v, (int, float)) and not isinstance(v, bool))
            lines.append("")
            lines.append(f"_LLM usage: {nums}_")
    return "\n".join(lines)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest tests/test_timeline.py -q`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add salome_agent_gui/timeline.py tests/test_timeline.py
git commit -m "feat: run timeline model and markdown rendering with gate links"
```

### Task 6: `RunWorker` — run on a QThread with a blocking gate

**Files:**
- Create: `salome_agent_gui/run_worker.py`
- Test: `tests/test_run_worker.py`

**Interfaces:**
- Consumes: `salome_agent.runner.run_request` (Task 2), `salome_agent.server_conn.connect(run_id) -> ExecutorClient` (raises `ServerUnavailable`).
- Produces:
  - `RunWorker(request: str, config: dict, *, batch: bool = False, verbose: int = 0, run_fn=default_run, parent=None)` — `QThread` with signals `event(dict)`, `gate(dict)`, `done(object)`; attributes `run_id: str`, `cancel: threading.Event`; methods `answer_gate(decision: dict)`, `stop()`, `gate_pending() -> bool`.
  - `default_run(request, config, *, run_id, on_event, on_gate, cancel, batch, verbose) -> RunResult` — connects, runs, closes.
  - `run_fn` has exactly `default_run`'s signature.

- [ ] **Step 1: Write the failing tests** — `tests/test_run_worker.py`

```python
from salome_agent_gui.run_worker import RunWorker


def _ev(kind):
    return {"kind": kind, "run_id": "r", "data": {}}


def make_run(events=(), gate_payload=None, raises=None):
    seen = {}

    def run(request, config, *, run_id, on_event, on_gate, cancel, batch, verbose):
        seen.update(request=request, batch=batch, verbose=verbose, run_id=run_id)
        if raises:
            raise raises
        for e in events:
            on_event(e)
        if gate_payload is not None:
            seen["decision"] = on_gate(gate_payload)
        return "RESULT"

    return run, seen


def test_events_then_done(qtbot):
    run, seen = make_run([_ev("plan"), _ev("report")])
    w = RunWorker("make a box", {}, batch=True, verbose=1, run_fn=run)
    got = []
    w.event.connect(got.append)
    with qtbot.waitSignal(w.done, timeout=3000) as blk:
        w.start()
    assert blk.args == ["RESULT"]
    qtbot.waitUntil(lambda: len(got) == 2)
    assert [e["kind"] for e in got] == ["plan", "report"]
    assert seen["request"] == "make a box" and seen["batch"] is True and seen["verbose"] == 1
    assert seen["run_id"] == w.run_id


def test_gate_blocks_until_answered(qtbot):
    run, seen = make_run(gate_payload={"plan": []})
    w = RunWorker("x", {}, run_fn=run)
    w.gate.connect(lambda payload: w.answer_gate({"action": "approve"}))
    with qtbot.waitSignal(w.done, timeout=3000):
        w.start()
    assert seen["decision"] == {"action": "approve"}


def test_stop_answers_a_pending_gate_with_abort(qtbot):
    run, seen = make_run(gate_payload={"last_error": {"message": "x"}})
    w = RunWorker("x", {}, run_fn=run)
    with qtbot.waitSignal(w.gate, timeout=3000):
        w.start()
    qtbot.waitUntil(w.gate_pending)
    with qtbot.waitSignal(w.done, timeout=3000):
        w.stop()
    assert seen["decision"] == {"action": "abort"}
    assert w.cancel.is_set()


def test_exception_becomes_fatal_event(qtbot):
    run, _ = make_run(raises=RuntimeError("no server"))
    w = RunWorker("x", {}, run_fn=run)
    got = []
    w.event.connect(got.append)
    with qtbot.waitSignal(w.done, timeout=3000) as blk:
        w.start()
    assert blk.args == [None]
    qtbot.waitUntil(lambda: bool(got))
    assert got[0]["kind"] == "fatal"
    assert "no server" in got[0]["data"]["message"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest tests/test_run_worker.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'salome_agent_gui.run_worker'`.

- [ ] **Step 3: Implement** — `salome_agent_gui/run_worker.py`

```python
import threading
import uuid

from PyQt5.QtCore import QThread, pyqtSignal


def default_run(request, config, *, run_id, on_event, on_gate, cancel, batch, verbose):
    from salome_agent import server_conn
    from salome_agent.runner import run_request

    client = server_conn.connect(run_id)
    try:
        return run_request(request, config, client=client, on_event=on_event,
                           on_gate=on_gate, cancel=cancel, batch=batch,
                           verbose=verbose, run_id=run_id)
    finally:
        try:
            client.close()
        except Exception:
            pass


class RunWorker(QThread):
    event = pyqtSignal(dict)
    gate = pyqtSignal(dict)
    done = pyqtSignal(object)

    def __init__(self, request, config, *, batch=False, verbose=0, run_fn=default_run,
                 parent=None):
        super().__init__(parent)
        self.request, self.config = request, config
        self.batch, self.verbose = batch, verbose
        self.run_id = uuid.uuid4().hex[:12]
        self.cancel = threading.Event()
        self._run_fn = run_fn
        self._answered = threading.Event()
        self._decision = None
        self._pending = False

    def run(self):
        result = None
        try:
            result = self._run_fn(self.request, self.config, run_id=self.run_id,
                                  on_event=self.event.emit, on_gate=self._on_gate,
                                  cancel=self.cancel, batch=self.batch, verbose=self.verbose)
        except Exception as e:  # noqa: BLE001
            self.event.emit({"kind": "fatal", "run_id": self.run_id,
                             "data": {"message": f"{type(e).__name__}: {e}"}})
        self.done.emit(result)

    def _on_gate(self, payload):
        self._answered.clear()
        self._decision = None
        self._pending = True
        self.gate.emit(payload)
        self._answered.wait()
        self._pending = False
        return self._decision or {"action": "abort"}

    def gate_pending(self) -> bool:
        return self._pending

    def answer_gate(self, decision):
        self._decision = decision
        self._answered.set()

    def stop(self):
        self.cancel.set()
        if self._pending:
            self.answer_gate({"action": "abort"})
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest tests/test_run_worker.py -q`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add salome_agent_gui/run_worker.py tests/test_run_worker.py
git commit -m "feat: RunWorker runs an agent request on a QThread with a blocking gate"
```

### Task 7: Chat view + dialogs

**Files:**
- Create: `salome_agent_gui/chat_view.py`, `salome_agent_gui/dialogs.py`
- Test: covered through `tests/test_main_window.py` (Task 8) plus the small test below in `tests/test_chat_view.py`

**Interfaces:**
- Consumes: `RunView`, `render_run`, `render_user` (Task 5); `config_store.save_raw` (Task 4); `salome_agent.interactive.normalize_plan` (Task 3).
- Produces:
  - `ChatView(QTextBrowser)`: signal `gate_clicked(str)` (action name); methods `add_user(text: str)`, `add_run() -> RunView`, `refresh()`; `markdown() -> str` (last rendered document, for tests).
  - `dialogs.edit_plan(parent, plan: list) -> list | None`
  - `dialogs.ask_hint(parent) -> str | None`
  - `dialogs.edit_config(parent, path: Path) -> bool`

- [ ] **Step 1: Write the failing test** — `tests/test_chat_view.py`

```python
from PyQt5.QtCore import QUrl

from salome_agent_gui.chat_view import ChatView


def test_renders_user_and_run_and_routes_gate_links(qtbot):
    v = ChatView()
    qtbot.addWidget(v)
    v.add_user("make a box")
    run = v.add_run()
    run.apply({"kind": "plan", "run_id": "r", "data": {"stages": [{"module": "geom", "intent": "box"}]}})
    run.gate = {"plan": []}
    v.refresh()
    assert "make a box" in v.markdown() and "gate:approve" in v.markdown()
    with qtbot.waitSignal(v.gate_clicked) as blk:
        v.anchorClicked.emit(QUrl("gate:approve"))
    assert blk.args == ["approve"]
```

- [ ] **Step 2: Run it to verify it fails**

Run: `QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest tests/test_chat_view.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'salome_agent_gui.chat_view'`.

- [ ] **Step 3: Implement** — `salome_agent_gui/chat_view.py`

```python
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QDesktopServices
from PyQt5.QtWidgets import QTextBrowser

from salome_agent_gui.timeline import RunView, render_run, render_user


class ChatView(QTextBrowser):
    gate_clicked = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setOpenLinks(False)
        self.anchorClicked.connect(self._on_anchor)
        self._items: list = []
        self._md = ""

    def add_user(self, text: str) -> None:
        self._items.append(text)
        self.refresh()

    def add_run(self) -> RunView:
        view = RunView()
        self._items.append(view)
        self.refresh()
        return view

    def markdown(self) -> str:
        return self._md

    def refresh(self) -> None:
        parts = [render_run(i) if isinstance(i, RunView) else render_user(i) for i in self._items]
        self._md = "\n\n---\n\n".join(parts)
        self.setMarkdown(self._md)
        bar = self.verticalScrollBar()
        bar.setValue(bar.maximum())

    def _on_anchor(self, url) -> None:
        if url.scheme() == "gate":
            self.gate_clicked.emit(url.path())
        else:
            QDesktopServices.openUrl(url)
```

`salome_agent_gui/dialogs.py`:

```python
import json
from pathlib import Path

from PyQt5.QtGui import QFontDatabase
from PyQt5.QtWidgets import (QDialog, QDialogButtonBox, QInputDialog, QLabel,
                             QPlainTextEdit, QVBoxLayout)


class JsonEditDialog(QDialog):
    """Monospace JSON editor; `validate(text)` raises to keep the dialog open."""

    def __init__(self, parent, title: str, text: str, validate):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(700, 560)
        self._validate = validate
        self.edit = QPlainTextEdit(text)
        self.edit.setFont(QFontDatabase.systemFont(QFontDatabase.FixedFont))
        self.error = QLabel("")
        self.error.setStyleSheet("color: #c0392b")
        self.error.setWordWrap(True)
        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._on_save)
        buttons.rejected.connect(self.reject)
        lay = QVBoxLayout(self)
        lay.addWidget(self.edit)
        lay.addWidget(self.error)
        lay.addWidget(buttons)

    def _on_save(self):
        try:
            self._validate(self.edit.toPlainText())
        except Exception as e:  # noqa: BLE001
            self.error.setText(str(e))
            return
        self.accept()


def edit_plan(parent, plan: list) -> list | None:
    from salome_agent.interactive import normalize_plan

    result = {}

    def validate(text):
        out = normalize_plan(json.loads(text))
        if out is None:
            raise ValueError("need a non-empty JSON list of stages, "
                             "each with module geom|smesh|paravis")
        result["plan"] = out

    dlg = JsonEditDialog(parent, "Edit plan", json.dumps(plan, indent=2), validate)
    return result.get("plan") if dlg.exec_() == QDialog.Accepted else None


def ask_hint(parent) -> str | None:
    text, ok = QInputDialog.getMultiLineText(parent, "Hint", "Hint for the specialist:")
    return text.strip() if ok and text.strip() else None


def edit_config(parent, path: Path) -> bool:
    from salome_agent_gui import config_store

    dlg = JsonEditDialog(parent, f"Edit {path}", Path(path).read_text(),
                         lambda text: config_store.save_raw(path, text))
    return dlg.exec_() == QDialog.Accepted
```

- [ ] **Step 4: Run it to verify it passes**

Run: `QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest tests/test_chat_view.py -q`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add salome_agent_gui/chat_view.py salome_agent_gui/dialogs.py tests/test_chat_view.py
git commit -m "feat: chat view with gate links, plan/hint/config dialogs"
```

### Task 8: Main window + app entry point

**Files:**
- Create: `salome_agent_gui/main_window.py`, `salome_agent_gui/app.py`
- Test: `tests/test_main_window.py`

**Interfaces:**
- Consumes: Tasks 4–7; `salome_agent.server_conn.read_handshake() -> dict | None`; `salome_agent.logs.setup_logging(verbose: int)`.
- Produces:
  - `MainWindow(config_path: Path, *, run_fn=None, probe_fn=None, ask_hint=None, edit_plan=None)` — injectables default to `run_worker.default_run`, `server_conn.read_handshake`, `dialogs.ask_hint`, `dialogs.edit_plan`. Public widgets used by tests: `input`, `send_btn`, `stop_btn`, `chat`, `status`, `base_url`, `model`, `api_key`, `batch`, `on_failure`, `tools`, `verbose`. Methods `send()`, `probe()`. Attribute `worker: RunWorker | None`.
  - `app.main(argv=None) -> int` (console script `salome-agent-gui`).

- [ ] **Step 1: Write the failing tests** — `tests/test_main_window.py`

```python
import json

import pytest

from salome_agent_gui.main_window import MainWindow

CFG = {"llm": {"base_url": "http://h/v1", "model": "m", "api_key": "secret"},
       "salome": {"root": "/x"}, "policy": {"on_stage_failure": "abort",
                                            "tool_selection": "hints"}}
PLAN = [{"module": "geom", "intent": "box", "params": {}, "inputs": []}]


@pytest.fixture
def cfg_path(tmp_path):
    p = tmp_path / "salome-agent.json"
    p.write_text(json.dumps(CFG))
    return p


def gated_run(record):
    def run(request, config, *, run_id, on_event, on_gate, cancel, batch, verbose):
        record.update(request=request, config=config, batch=batch, verbose=verbose)
        on_event({"kind": "plan", "run_id": run_id, "data": {"stages": PLAN}})
        record["decision"] = on_gate({"plan": PLAN})
        if record["decision"]["action"] == "approve":
            on_event({"kind": "stage_done", "run_id": run_id, "data": {"index": 0, "names": ["box_1"]}})
            on_event({"kind": "report", "run_id": run_id,
                      "data": {"result_py": "/w/result.py", "report_md": "/w/report.md",
                               "text": "", "usage": None}})
        return None
    return run


def make(qtbot, cfg_path, run, **kw):
    w = MainWindow(cfg_path, run_fn=run, probe_fn=lambda: {"mode": "gui", "port": 4242}, **kw)
    qtbot.addWidget(w)
    return w


def test_loads_key_fields_and_masks_the_key(qtbot, cfg_path):
    w = make(qtbot, cfg_path, gated_run({}))
    assert w.base_url.text() == "http://h/v1" and w.model.text() == "m"
    assert w.api_key.text() == "secret"
    assert w.api_key.echoMode() == w.api_key.Password


def test_probe_reports_server_and_enables_send(qtbot, cfg_path):
    w = make(qtbot, cfg_path, gated_run({}))
    w.probe()
    assert "gui" in w.status.text() and "4242" in w.status.text()
    assert w.send_btn.isEnabled()


def test_send_approve_runs_to_done(qtbot, cfg_path):
    rec = {}
    w = make(qtbot, cfg_path, gated_run(rec))
    w.probe()
    w.on_failure.setCurrentText("continue")
    w.input.setText("make a box")
    w.send()
    qtbot.waitUntil(lambda: "gate:approve" in w.chat.markdown(), timeout=3000)
    assert not w.send_btn.isEnabled() and w.stop_btn.isEnabled()
    w.chat.gate_clicked.emit("approve")
    qtbot.waitUntil(lambda: w.worker is None, timeout=3000)
    assert rec["decision"] == {"action": "approve"}
    assert rec["config"]["policy"]["on_stage_failure"] == "continue"
    assert "`box_1`" in w.chat.markdown() and "gate:" not in w.chat.markdown()
    assert w.send_btn.isEnabled()


def test_edit_uses_the_injected_editor(qtbot, cfg_path):
    rec = {}
    edited = [{"module": "geom", "intent": "cyl", "params": {}, "inputs": [], "tools": []}]
    w = make(qtbot, cfg_path, gated_run(rec), edit_plan=lambda parent, plan: edited)
    w.probe()
    w.input.setText("x")
    w.send()
    qtbot.waitUntil(lambda: "gate:edit" in w.chat.markdown(), timeout=3000)
    w.chat.gate_clicked.emit("edit")
    qtbot.waitUntil(lambda: w.worker is None, timeout=3000)
    assert rec["decision"] == {"action": "edit", "plan": edited}


def test_stop_aborts_a_pending_gate(qtbot, cfg_path):
    rec = {}
    w = make(qtbot, cfg_path, gated_run(rec))
    w.probe()
    w.input.setText("x")
    w.send()
    qtbot.waitUntil(lambda: "gate:approve" in w.chat.markdown(), timeout=3000)
    w.stop_btn.click()
    qtbot.waitUntil(lambda: w.worker is None, timeout=3000)
    assert rec["decision"] == {"action": "abort"}


def test_key_field_edit_is_saved(qtbot, cfg_path):
    w = make(qtbot, cfg_path, gated_run({}))
    w.model.setText("qwen3")
    w.model.editingFinished.emit()
    assert json.loads(cfg_path.read_text())["llm"]["model"] == "qwen3"


def test_no_server_keeps_send_disabled(qtbot, cfg_path):
    w = MainWindow(cfg_path, run_fn=gated_run({}), probe_fn=lambda: None)
    qtbot.addWidget(w)
    w.probe()
    assert not w.send_btn.isEnabled()
    assert "Plugins" in w.status.text()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest tests/test_main_window.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'salome_agent_gui.main_window'`.

- [ ] **Step 3: Implement** — `salome_agent_gui/main_window.py`

```python
from pathlib import Path

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (QCheckBox, QComboBox, QFormLayout, QGroupBox, QHBoxLayout,
                             QLabel, QLineEdit, QMainWindow, QMessageBox, QPushButton,
                             QSpinBox, QSplitter, QVBoxLayout, QWidget)

from salome_agent_gui import config_store, dialogs
from salome_agent_gui.chat_view import ChatView
from salome_agent_gui.run_worker import RunWorker, default_run

HELP = ("Each message is one salome-agent run: the planner proposes stages, you approve, "
        "edit or abort; each stage is generated and verified in this SALOME session. "
        "On a failed stage give a hint, skip it or abort. Results land in the study; "
        "result.py and report.md are linked at the end.")
NO_SERVER = "No agent server. Start it from Tools > Plugins > SALOME Agent."


class HistoryLineEdit(QLineEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._history: list[str] = []
        self._pos = 0

    def remember(self, text: str) -> None:
        self._history.append(text)
        self._pos = len(self._history)

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Up and self._history:
            self._pos = max(0, self._pos - 1)
            self.setText(self._history[self._pos])
        elif e.key() == Qt.Key_Down and self._history:
            self._pos = min(len(self._history), self._pos + 1)
            self.setText(self._history[self._pos] if self._pos < len(self._history) else "")
        else:
            super().keyPressEvent(e)


class MainWindow(QMainWindow):
    def __init__(self, config_path: Path, *, run_fn=None, probe_fn=None,
                 ask_hint=None, edit_plan=None):
        super().__init__()
        from salome_agent import server_conn

        self.config_path = Path(config_path)
        self._run_fn = run_fn or default_run
        self._probe_fn = probe_fn or server_conn.read_handshake
        self._ask_hint = ask_hint or dialogs.ask_hint
        self._edit_plan = edit_plan or dialogs.edit_plan
        self.worker: RunWorker | None = None
        self._view = None
        self._connected = False
        self.setWindowTitle("SALOME Agent")
        self.resize(1200, 780)
        self._build()
        self._load_key_fields()
        self._polls = 0
        self._poll = QTimer(self)
        self._poll.setInterval(1000)
        self._poll.timeout.connect(self._poll_tick)
        self._poll.start()
        self.probe()

    # ---- layout ---------------------------------------------------------
    def _build(self):
        conn = QGroupBox("Connection")
        form = QFormLayout(conn)
        path_lbl = QLabel(str(self.config_path))
        path_lbl.setWordWrap(True)
        form.addRow("Config:", path_lbl)
        edit_cfg = QPushButton("Edit configuration…")
        edit_cfg.clicked.connect(self._edit_config)
        form.addRow("", edit_cfg)
        self.base_url, self.model, self.api_key = QLineEdit(), QLineEdit(), QLineEdit()
        self.api_key.setEchoMode(QLineEdit.Password)
        for label, w in (("LLM base URL:", self.base_url), ("Model:", self.model),
                         ("API key:", self.api_key)):
            w.editingFinished.connect(self._save_key_fields)
            form.addRow(label, w)
        self.status = QLabel("")
        self.status.setWordWrap(True)
        form.addRow("Server:", self.status)
        connect_btn = QPushButton("Connect")
        connect_btn.clicked.connect(self.probe)
        form.addRow("", connect_btn)

        opts = QGroupBox("Run options")
        of = QFormLayout(opts)
        self.batch = QCheckBox("Batch (no approval gates)")
        self.on_failure = QComboBox()
        self.on_failure.addItems(["abort", "continue"])
        self.tools = QComboBox()
        self.tools.addItems(["hints", "all"])
        self.verbose = QSpinBox()
        self.verbose.setRange(0, 2)
        of.addRow("", self.batch)
        of.addRow("On stage failure:", self.on_failure)
        of.addRow("Tool selection:", self.tools)
        of.addRow("Verbosity:", self.verbose)

        left = QWidget()
        ll = QVBoxLayout(left)
        ll.addWidget(conn)
        ll.addWidget(opts)
        ll.addStretch(1)

        self.chat = ChatView()
        self.chat.gate_clicked.connect(self._on_gate_clicked)
        split = QSplitter()
        split.addWidget(left)
        split.addWidget(self.chat)
        split.setStretchFactor(1, 1)

        self.input = HistoryLineEdit()
        self.input.setPlaceholderText("Describe what to build…")
        self.input.returnPressed.connect(self.send)
        self.send_btn, self.stop_btn = QPushButton("Send"), QPushButton("Stop")
        self.send_btn.clicked.connect(self.send)
        self.stop_btn.clicked.connect(self._stop)
        help_btn, quit_btn = QPushButton("Help"), QPushButton("Quit")
        help_btn.clicked.connect(lambda: QMessageBox.information(self, "SALOME Agent", HELP))
        quit_btn.clicked.connect(self.close)
        row = QHBoxLayout()
        row.addWidget(self.input, 1)
        for b in (self.send_btn, self.stop_btn, help_btn, quit_btn):
            row.addWidget(b)

        central = QWidget()
        cl = QVBoxLayout(central)
        cl.addWidget(split, 1)
        cl.addLayout(row)
        self.setCentralWidget(central)
        self._sync_buttons()

    # ---- config ---------------------------------------------------------
    def _load_key_fields(self):
        f = config_store.read_key_fields(self.config_path)
        self.base_url.setText(f["base_url"])
        self.model.setText(f["model"])
        self.api_key.setText(f["api_key"])

    def _save_key_fields(self):
        config_store.save_key_fields(self.config_path, {
            "base_url": self.base_url.text().strip(), "model": self.model.text().strip(),
            "api_key": self.api_key.text()})

    def _edit_config(self):
        if dialogs.edit_config(self, self.config_path):
            self._load_key_fields()

    # ---- server ---------------------------------------------------------
    def probe(self):
        hs = self._probe_fn()
        self._connected = hs is not None
        self.status.setText(f"{hs.get('mode', '?')} server on port {hs.get('port', '?')}"
                            if hs else NO_SERVER)
        self._sync_buttons()

    def _poll_tick(self):
        self._polls += 1
        if self._connected or self._polls >= 10:
            self._poll.stop()
            return
        self.probe()

    # ---- runs -----------------------------------------------------------
    def _sync_buttons(self):
        running = self.worker is not None
        self.send_btn.setEnabled(self._connected and not running)
        self.stop_btn.setEnabled(running)

    def send(self):
        text = self.input.text().strip()
        if not text or self.worker is not None or not self._connected:
            return
        self._save_key_fields()
        try:
            config = config_store.load(self.config_path)
        except Exception as e:  # noqa: BLE001
            self.status.setText(f"Config error: {e}")
            return
        config.setdefault("policy", {})
        config["policy"]["on_stage_failure"] = self.on_failure.currentText()
        config["policy"]["tool_selection"] = self.tools.currentText()
        self.input.remember(text)
        self.input.clear()
        self.chat.add_user(text)
        self._view = self.chat.add_run()
        self.worker = RunWorker(text, config, batch=self.batch.isChecked(),
                                verbose=self.verbose.value(), run_fn=self._run_fn, parent=self)
        self.worker.event.connect(self._on_event)
        self.worker.gate.connect(self._on_gate)
        self.worker.done.connect(self._on_done)
        self.worker.start()
        self._sync_buttons()

    def _on_event(self, ev: dict):
        self._view.apply(ev)
        self.chat.refresh()

    def _on_gate(self, payload: dict):
        self._view.gate = payload
        self.chat.refresh()

    def _on_gate_clicked(self, action: str):
        if self.worker is None or not self._view or not self._view.gate:
            return
        payload = self._view.gate
        if action == "edit":
            plan = self._edit_plan(self, payload.get("plan") or [])
            if plan is None:
                return
            decision = {"action": "edit", "plan": plan}
        elif action == "hint":
            hint = self._ask_hint(self)
            if not hint:
                return
            decision = {"action": "hint", "hint": hint}
        elif action in ("approve", "skip", "abort"):
            decision = {"action": action}
        else:
            return
        self._view.gate = None
        self.chat.refresh()
        self.worker.answer_gate(decision)

    def _stop(self):
        if self.worker is not None:
            self.worker.stop()

    def _on_done(self, _result):
        if self._view is not None:
            self._view.gate = None
            self._view.finished = True
            self.chat.refresh()
        self.worker.wait(2000)
        self.worker = None
        self._sync_buttons()

    def closeEvent(self, e):
        if self.worker is not None:
            self.worker.stop()
            self.worker.wait(3000)
        super().closeEvent(e)
```

`salome_agent_gui/app.py`:

```python
import sys


def main(argv=None) -> int:
    from PyQt5.QtWidgets import QApplication

    from salome_agent.logs import setup_logging
    from salome_agent_gui import config_store
    from salome_agent_gui.main_window import MainWindow

    setup_logging(0)
    path = config_store.ensure_user_config()
    app = QApplication(sys.argv if argv is None else argv)
    app.setApplicationName("SALOME Agent")
    window = MainWindow(path)
    window.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the whole GUI suite**

Run: `QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest -q`
Expected: 24 passed (5 config + 7 timeline + 4 worker + 1 chat view + 7 window).

- [ ] **Step 5: Manual launch against a headless server**

In salome-agent: `scripts/start-headless-server.sh` (background). Then `.venv/bin/salome-agent-gui`.
Expected: status "headless server on port N"; send "a 20mm cube"; plan card appears; Approve; stage row turns ✓; summary links open `result.py`. Stop the server afterwards.

- [ ] **Step 6: Commit**

```bash
git add salome_agent_gui/main_window.py salome_agent_gui/app.py tests/test_main_window.py
git commit -m "feat: main window wiring config, server probe, runs and gates"
```

### Task 9: SALOME-side plugin (launcher + `salome_plugins.py`)

**Files:**
- Create: `sagui_launcher.py`, `salome_plugins.py`, `salome-agent-gui.sh`
- Test: `tests/test_sagui_launcher.py`

**Interfaces:**
- Consumes: `agent_server.gui_plugin` module state `_SERVER` and `launch(context)` (in-GUI server; `launch` restarts when called twice, so it is only called when `_SERVER is None`).
- Produces (`sagui_launcher`):
  - `STRIP_VARS: tuple[str, ...]`
  - `clean_env(environ: Mapping[str, str]) -> dict`
  - `pid_path() -> Path`
  - `window_alive(path: Path) -> bool`
  - `ensure_server(context, gui_plugin=None) -> str` (`"reused"` | `"started"`)
  - `launch_window(launcher: str, path: Path, environ=None, popen=subprocess.Popen) -> int`
  - `run(context) -> str` — `""` on success, otherwise a message for the user.

- [ ] **Step 1: Write the failing tests** — `tests/test_sagui_launcher.py`

```python
import os
import types

import sagui_launcher as sl


def test_clean_env_strips_salome_qt_and_python_vars():
    env = {"PYTHONPATH": "a", "LD_LIBRARY_PATH": "b", "QT_PLUGIN_PATH": "c",
           "QT_QPA_PLATFORM_PLUGIN_PATH": "d", "QT_QPA_FONTDIR": "e", "HOME": "/h"}
    assert sl.clean_env(env) == {"HOME": "/h"}
    assert env["PYTHONPATH"] == "a"


def test_window_alive_reads_the_pid_file(tmp_path):
    p = tmp_path / "w.pid"
    assert not sl.window_alive(p)
    p.write_text("garbage")
    assert not sl.window_alive(p)
    p.write_text(str(os.getpid()))
    assert sl.window_alive(p)
    p.write_text("999999999")
    assert not sl.window_alive(p)


def test_ensure_server_reuses_a_running_server():
    calls = []
    gp = types.SimpleNamespace(_SERVER=object(), launch=lambda ctx: calls.append(ctx))
    assert sl.ensure_server("ctx", gp) == "reused"
    assert calls == []


def test_ensure_server_starts_when_none():
    calls = []
    gp = types.SimpleNamespace(_SERVER=None, launch=lambda ctx: calls.append(ctx))
    assert sl.ensure_server("ctx", gp) == "started"
    assert calls == ["ctx"]


def test_launch_window_spawns_detached_with_clean_env_and_writes_pid(tmp_path):
    seen = {}

    def popen(args, **kw):
        seen.update(args=args, **kw)
        return types.SimpleNamespace(pid=4321)

    pid = sl.launch_window("/opt/x/bin/salome-agent-gui.sh", tmp_path / "w.pid",
                           environ={"PYTHONPATH": "p", "HOME": "/h"}, popen=popen)
    assert pid == 4321
    assert seen["args"] == ["/opt/x/bin/salome-agent-gui.sh"]
    assert seen["start_new_session"] is True
    assert seen["env"] == {"HOME": "/h"}
    assert (tmp_path / "w.pid").read_text() == "4321"


def test_run_without_launcher_env_reports_it(monkeypatch, tmp_path):
    monkeypatch.setattr(sl, "ensure_server", lambda ctx, gui_plugin=None: "started")
    monkeypatch.setattr(sl, "pid_path", lambda: tmp_path / "w.pid")
    monkeypatch.delenv("SALOME_AGENT_GUI", raising=False)
    assert "SALOME_AGENT_GUI" in sl.run("ctx")


def test_run_with_live_window_does_not_spawn(monkeypatch, tmp_path):
    monkeypatch.setattr(sl, "ensure_server", lambda ctx, gui_plugin=None: "reused")
    p = tmp_path / "w.pid"
    p.write_text(str(os.getpid()))
    monkeypatch.setattr(sl, "pid_path", lambda: p)
    monkeypatch.setenv("SALOME_AGENT_GUI", "/x")
    monkeypatch.setattr(sl, "launch_window", lambda *a, **k: (_ for _ in ()).throw(AssertionError))
    assert "already open" in sl.run("ctx")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest tests/test_sagui_launcher.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'sagui_launcher'`.

- [ ] **Step 3: Implement**

`sagui_launcher.py` (standard library only — runs inside SALOME's Python):

```python
"""SALOME-side half of the SALOME Agent plugin: start/reuse the in-GUI agent
server and spawn the chat window (its own venv) once."""
import getpass
import os
import subprocess
import tempfile
from pathlib import Path

STRIP_VARS = ("PYTHONPATH", "QT_PLUGIN_PATH", "QT_QPA_PLATFORM_PLUGIN_PATH",
              "QT_QPA_FONTDIR", "LD_LIBRARY_PATH")


def clean_env(environ) -> dict:
    env = dict(environ)
    for var in STRIP_VARS:
        env.pop(var, None)
    return env


def pid_path() -> Path:
    return Path(tempfile.gettempdir()) / f"salome-agent-gui-{getpass.getuser()}.pid"


def window_alive(path: Path) -> bool:
    try:
        pid = int(Path(path).read_text().strip())
    except (OSError, ValueError):
        return False
    try:
        done, _ = os.waitpid(pid, os.WNOHANG)   # reap our own exited child (else a zombie looks alive)
        if done == pid:
            return False
    except ChildProcessError:
        pass
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def ensure_server(context, gui_plugin=None) -> str:
    if gui_plugin is None:
        from agent_server import gui_plugin
    if gui_plugin._SERVER is not None:
        return "reused"
    gui_plugin.launch(context)
    return "started"


def launch_window(launcher: str, path: Path, environ=None, popen=subprocess.Popen) -> int:
    proc = popen([launcher], start_new_session=True,
                 env=clean_env(os.environ if environ is None else environ))
    Path(path).write_text(str(proc.pid))
    return proc.pid


def run(context) -> str:
    ensure_server(context)
    path = pid_path()
    if window_alive(path):
        return "The SALOME Agent window is already open."
    launcher = os.environ.get("SALOME_AGENT_GUI")
    if not launcher:
        return ("SALOME_AGENT_GUI is not set; the agent server is running but the chat "
                "window cannot be started (install the salome_agent_gui SAT product).")
    launch_window(launcher, path)
    return ""
```

`salome_plugins.py`:

```python
import salome_pluginsmanager


def runSalomeAgent(context):
    # The plugins manager exec()s this file with a throwaway locals dict, so
    # every name used at click time is imported here.
    from PyQt5.QtWidgets import QMessageBox

    import sagui_launcher

    try:
        msg = sagui_launcher.run(context)
    except Exception as e:  # noqa: BLE001
        QMessageBox.warning(None, "SALOME Agent", f"Failed to start SALOME Agent:\n{e}")
        return
    if msg:
        QMessageBox.information(None, "SALOME Agent", msg)


salome_pluginsmanager.AddFunction('SALOME Agent',
                                  'Start the agent server and open the SALOME Agent chat',
                                  runSalomeAgent)
```

`salome-agent-gui.sh`:

```bash
#!/bin/bash
if [ -z "${SALOME_AGENT_GUI_ROOT_DIR}" ]; then
    export SALOME_AGENT_GUI_ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
fi
source "${SALOME_AGENT_GUI_ROOT_DIR}/bin/activate"
exec salome-agent-gui "$@"
```

```bash
chmod 755 salome-agent-gui.sh
```

- [ ] **Step 4: Run the full GUI suite**

Run: `QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest -q`
Expected: 31 passed.

- [ ] **Step 5: Commit**

```bash
git add sagui_launcher.py salome_plugins.py salome-agent-gui.sh tests/test_sagui_launcher.py
git commit -m "feat: SALOME plugin entry that starts the agent server and the chat window"
```

- [ ] **Step 6: Remote** — ask the user to create the GitHub repo (`bauj/salome-agent-gui` assumed by Task 10's pyconf) and push; confirm the URL before Task 10.

---

## Part 3 — SAT integration

### Task 10: SAT product on a new SAT_SALOME branch

**Files (in the SAT_SALOME checkout):**
- Create: `products/salome_agent_gui.pyconf`, `products/compil_scripts/salome_agent_gui.sh`, `products/env_scripts/salome_agent_gui.py`
- Modify: `applications/SALOME-master-native.pyconf`, `applications/SALOME-9.16.0-native.pyconf` (add next to `sarag : 'master'`)

**Interfaces:**
- Consumes: the GUI repo layout from Tasks 4–9 (`salome-agent/` submodule, `salome_plugins.py`, `sagui_launcher.py`, `salome-agent-gui.sh`); SAT variables `PRODUCT_INSTALL`, `SOURCE_DIR`, `PYTHONBIN`, `PYTHON_VERSION`, `VERSION`.
- Produces: installed layout `${PRODUCT_INSTALL}/{bin/salome-agent-gui.sh, lib/python${PYTHON_VERSION}/site-packages/salome_agent_gui/config.example.json, share/salome/plugins/salome_agent_gui/{salome_plugins.py,sagui_launcher.py,agent_server/,executor/}}`; env `SALOME_AGENT_GUI_ROOT_DIR`, `SALOME_AGENT_GUI`.

- [ ] **Step 1: Branch** (the checkout currently sits on `ngh/chatbot`, which carries the sarag product, with stashes — leave them alone)

```bash
cd "$SAT_SALOME"            # the SAT_SALOME checkout
git status -sb              # must be clean apart from untracked __pycache__
git switch -c feat/salome-agent-gui
```

- [ ] **Step 2: Product config** — `products/salome_agent_gui.pyconf`

```
default :
{
    name : "salome_agent_gui"
    build_source : "script"
    compil_script :  'salome_agent_gui' + $VARS.scriptExtension
    get_source : "git"
    git_info:
    {
      repositories :
      {
        github : "https://github.com/bauj/salome-agent-gui.git"
      }
      git_commands: [
                     'git submodule update --init --recursive'
                    ]
    }
    patches : []
    system_info :
    {
        rpm : []
        rpm_dev : []
        apt : []
        apt_dev : []
    }
    environ :
    {
       env_script : "salome_agent_gui.py"
    }
    depend : ['Python', 'setuptools']
    opt_depend: ['wheel', 'PyQt']
    source_dir : $APPLICATION.workdir + $VARS.sep + 'SOURCES' + $VARS.sep + $name
    build_dir : $APPLICATION.workdir + $VARS.sep + 'BUILD' + $VARS.sep + $name
    install_dir : 'base'
    properties :
    {
      incremental : "yes"
    }
}
```

- [ ] **Step 3: Build script** — `products/compil_scripts/salome_agent_gui.sh`

```bash
#!/bin/bash

echo "##########################################################################"
echo "SALOME_AGENT_GUI" $VERSION
echo "##########################################################################"

mkdir -p "${PRODUCT_INSTALL}"
${PYTHONBIN} -m venv ${PRODUCT_INSTALL}
source ${PRODUCT_INSTALL}/bin/activate
cd ${SOURCE_DIR}

${PRODUCT_INSTALL}/bin/pip3 install ./salome-agent .
if [ $? -ne 0 ]; then
    echo "FATAL: could not install salome-agent-gui"
    exit 1
fi

cp ${SOURCE_DIR}/salome-agent-gui.sh ${PRODUCT_INSTALL}/bin/salome-agent-gui.sh
if [ $? -ne 0 ]; then
    echo "FATAL: could not find salome-agent-gui.sh"
    exit 2
fi
chmod 755 ${PRODUCT_INSTALL}/bin/salome-agent-gui.sh

# Seed config for ~/.config/salome/salome-agent.json on first launch.
cp ${SOURCE_DIR}/salome-agent/config.example.json \
   ${PRODUCT_INSTALL}/lib/python${PYTHON_VERSION}/site-packages/salome_agent_gui/config.example.json
if [ $? -ne 0 ]; then
    echo "FATAL: could not copy config.example.json"
    exit 3
fi

# GUI plugin: SALOME's plugins manager scans <X>_ROOT_DIR/share/salome/plugins
# for *_plugins.py (SALOME_AGENT_GUI_ROOT_DIR is set by env_scripts/salome_agent_gui.py)
# and puts that directory on sys.path, so sagui_launcher and the in-GUI agent
# server packages (stdlib + SALOME only) are importable in SALOME's Python.
PLUGIN_DIR=${PRODUCT_INSTALL}/share/salome/plugins/salome_agent_gui
mkdir -p ${PLUGIN_DIR}
for f in salome_plugins.py sagui_launcher.py; do
    cp ${SOURCE_DIR}/${f} ${PLUGIN_DIR}/${f}
    if [ $? -ne 0 ]; then
        echo "FATAL: could not install GUI plugin file ${f}"
        exit 4
    fi
done
for d in agent_server executor; do
    rm -rf ${PLUGIN_DIR}/${d}
    cp -r ${SOURCE_DIR}/salome-agent/${d} ${PLUGIN_DIR}/${d}
    if [ $? -ne 0 ]; then
        echo "FATAL: could not install agent server package ${d}"
        exit 5
    fi
done
find ${PLUGIN_DIR} -name __pycache__ -prune -exec rm -rf {} +

echo
echo "########## END"
```

```bash
chmod 755 products/compil_scripts/salome_agent_gui.sh
```

- [ ] **Step 4: Env script** — `products/env_scripts/salome_agent_gui.py`

```python
#!/usr/bin/env python
#-*- coding:utf-8 -*-

import os.path

def set_env(env, prereq_dir, version):
  env.set('SALOME_AGENT_GUI_ROOT_DIR', prereq_dir)
  env.set('SALOME_AGENT_GUI', os.path.join(prereq_dir, 'bin', 'salome-agent-gui.sh'))

def set_nativ_env(env):
  pass
```

- [ ] **Step 5: Register in both applications**

In `applications/SALOME-master-native.pyconf` and `applications/SALOME-9.16.0-native.pyconf`, directly under the `sarag : 'master'` line, add (same indentation):

```
        salome_agent_gui : 'master'
```

Check: `grep -n "salome_agent_gui" applications/SALOME-master-native.pyconf applications/SALOME-9.16.0-native.pyconf` → one hit each.

- [ ] **Step 6: Build**

Run from the SAT tool dir: `./sat prepare SALOME-master-native -p salome_agent_gui && ./sat compile SALOME-master-native -p salome_agent_gui`
Expected: `END` banner; `share/salome/plugins/salome_agent_gui/` holds the 2 files + 2 packages; `bin/salome-agent-gui.sh` executable. Repeat for `SALOME-9.16.0-native` only if that build is usable on this box (the local 9.16 dev install is known broken — then verify 9.16 on a machine that has it and say so).

- [ ] **Step 7: Commit**

```bash
git add products/salome_agent_gui.pyconf products/compil_scripts/salome_agent_gui.sh products/env_scripts/salome_agent_gui.py applications/SALOME-master-native.pyconf applications/SALOME-9.16.0-native.pyconf
git commit -m "add salome_agent_gui product (SALOME Agent chat plugin)"
```

### Task 11: Live smoke in SALOME (manual)

- [ ] **Step 1:** Launch the master SALOME GUI built with the product (`QT_QPA_PLATFORM=xcb` launcher).
- [ ] **Step 2:** Tools > Plugins > SALOME Agent. Expected: Message dock "SALOME agent server on 127.0.0.1:N"; chat window opens; status "gui server on port N".
- [ ] **Step 3:** Click the plugin again. Expected: "already open" info box; no second window; server not restarted (same port).
- [ ] **Step 4:** Send "two 100x50x50 boxes side by side". Approve the plan. Expected: stage rows go ✓, objects appear in the Object Browser/viewer, summary shows `box_*` names; `result.py` link opens.
- [ ] **Step 5:** Send a request, click Stop at the plan gate. Expected: "_Run aborted._", nothing written.
- [ ] **Step 6:** Quit the window, click the plugin again. Expected: a fresh window opens (stale PID detected).
- [ ] **Step 7:** Record the outcome (and any gap, e.g. `LD_LIBRARY_PATH` stripping breaking the venv Python) in the session notes; file fixes as tasks.
