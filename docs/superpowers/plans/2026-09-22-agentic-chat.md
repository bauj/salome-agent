# Agentic Chat Mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** After the first result, the user keeps talking to the agent ("make the roof taller", "add a chimney", `/undo`), and each turn amends the same SALOME study and the same `result.py` instead of starting over.

**Architecture:** A persistent `BuildGraph` (nodes = geom tool calls with their named args) becomes the single source of truth; `result.py` and the study are projections of it. A follow-up turn enters the existing LangGraph at a new `amend` node (a `ToolSpecialist` with `edit_arg` / `delete_node` / `finish_turn` / `replan`), and every edit replays only the edited node plus its downstream closure, with output names pinned so nothing drifts. A `Conversation` object owns one executor connection, one LLM client and the graph for the conversation's life; the CLI gets a `--chat` REPL over it.

**Tech Stack:** Python 3.10+, LangGraph (`StateGraph`, `Command`, `MemorySaver`), the salome-agent agent_server wire protocol, pytest.

**Spec:** `docs/superpowers/specs/2026-09-21-agentic-chat-design.md` (task 194 / GH #15). Read it before starting any task.

## Global Constraints

- Work on branch `spec/agentic-chat` (already rebased onto main `6c90d0e`). Never commit to `main`.
- Commit subjects only, no body, no trailers; every subject ends with ` (#15)`. Example: `git commit -m "feat: pin output names on tool replay (#15)"`.
- Stage files by name. Never `git add -A` / `git add .` (untracked `config_*.json`, `cat-runs.sh`, `stream-probe.py`, `test.sh` must stay out).
- Unit tests: `.venv/bin/python -m pytest tests/unit -q`. All 447 existing tests must stay green after every task (update a test only where a task says so).
- The two golden replays (`tests/unit/test_golden_runs.py`, vase + enclosure) must stay byte-identical after every task. They are the proof one-shot runs did not change.
- One-shot mode (no `--chat`) must behave exactly as today: same prompts, same provider payloads, same `result.py`.
- No SALOME install paths in code, comments or docs. No provider-specific values hardcoded; knobs go in config.
- Minimal comments, matching the surrounding code's density.
- v1 amend is geom only. smesh / paravis stages are carried through as opaque script blocks.

## Deviations from the spec (decided while planning; each is load-bearing)

1. **No wire change to return `tool` / `tool_args`.** The specialist already holds the tool name and args it sent. `ToolSpecialist` records `{tool, tool_args, records}` per successful call itself (Task 5). The wire stays as it is and the recorded cassettes stay valid.
2. **Edits apply immediately inside the amend loop, not in a separate `replay` node afterwards.** If edits were deferred, a `measure` after `edit_arg`, or a new chimney placed on the roof, would see the *old* roof. Each `edit_arg` / `delete_node` call replays its affected set on the spot, and a failure rolls back and comes back to the model as a tool error. The spec's `replay` graph node is dropped; `amend` goes straight to `assemble` or `plan`.
3. **`/rebuild` = forget every graph name + replay the whole graph with pinned names.** It does *not* use `reset_and_replay`: `GuiBackend.reset()` refuses outright ("no rollback in a live GUI study"), and the headless `reset()` zeroes the name counters. The same `replay` machinery serves edit, rollback, undo and rebuild.
4. **A fresh LangGraph thread per turn.** The persistent fields (`build_graph`, `history`, `turn_idx`) are carried explicitly from `Conversation` into each turn's input. A cancelled turn leaves its thread holding pending tasks or an interrupt, and a clean thread per turn avoids resuming into that.
5. **New: `hold` on `hello`.** `wire.Server` lets a second client take over a connection that has been idle ≥ 30 s (`BUSY_IDLE_S`). That second client's `hello` calls `begin_session`, which wipes the counters and `geom_bindings` under an idle chat. A conversation connects with `hold=True`, and the server refuses to hand over a held connection (Task 3).
6. **The graph keeps stage blocks** (label + `result_names`) so `result.py` keeps its `# stage i/N: geom` headers and per-stage pruning, which the goldens pin. Reachability is computed over the *whole* graph, so an amend turn that references a shape pruned out of an earlier stage still emits it.
7. **`finish_turn(summary, question?)`**: a `question` means the turn ends `needs_input`. **`ModuleSpec.extra_terminators`** lets `replan` end the loop alongside `finish_turn`.
8. **The salome-agent-gui `ConversationWorker` conversion is a separate follow-up plan.** It lives in the other repo, and the `Conversation` API should be live-verified first. This plan covers the salome-agent engine and the CLI.

Not in this plan (226 leftovers, unrelated to chat): planner prompt built over `MODULES` / `routable()`, the geom wording in `tool_specialist.py` strings, the duplicate specialist-registration imports in `graph.py`.

## File map

| File | Status | Responsibility |
|---|---|---|
| `executor/geom/helpers.py` | modify | `_next_name` honours a pin list |
| `executor/geom/__init__.py` | modify | `call_tool(..., pin=None)` |
| `agent_server/backend.py` | modify | `forget_names()` + `HeadlessBackend.forget` |
| `agent_server/gui_backend.py` | modify | `GuiBackend.forget` (marshalled) |
| `agent_server/core.py` | modify | `call_tool` passes `pin`; new `forget` command |
| `agent_server/wire.py` | modify | held connections are never handed over |
| `salome_agent/executor_client.py` | modify | `call_tool(pin=)`, `forget()`, `hold=` |
| `salome_agent/server_conn.py` | modify | `connect(..., hold=False)` |
| `tests/replay.py` | modify | `ReplayExecutorClient.call_tool(pin=)`, `forget()` |
| `salome_agent/modules.py` | modify | `ModuleSpec.extra_terminators` |
| `salome_agent/specialists/tool_specialist.py` | modify | per-call trace (`calls`), extra terminators |
| `salome_agent/graph.py` | modify | stage carries `calls`; router; `amend` node; graph-aware upstream; commit on assemble; per-turn report |
| `salome_agent/state.py` | modify | `StageRecord.calls`; `RunState` chat fields |
| `salome_agent/build_graph.py` | **create** | pure-data build graph |
| `salome_agent/codegen.py` | modify | `roots_for()` split out; `generate(keep=)` |
| `salome_agent/nodes/assembler.py` | modify | `commit_stages()`, `render_script()`; `assemble(state)` kept as a wrapper |
| `salome_agent/replay.py` | **create** | `replay`, `apply_edit`, `apply_delete`, `sync`, `rebuild_all` |
| `salome_agent/specialists/amend.py` | **create** | amend prompt, tools, session, `run_amend()` |
| `salome_agent/nodes/report.py` | modify | amend outcome section |
| `salome_agent/runner.py` | modify | `initial_state()`, `drive()`, new event kinds |
| `salome_agent/conversation.py` | **create** | `Conversation`, `TurnResult` |
| `salome_agent/chat.py` | **create** | the `--chat` REPL |
| `salome_agent/cli.py` | modify | `--chat` flag |
| `README.md` | modify | chat mode section |

---

### Task 1: Pinned output names on tool replay

Re-invoking a tool calls `_next_name`, so a rebuilt `box_1` would come back as `box_3`. A pin list makes `_next_name` hand back the names the node already owns, and bumps the counter past each pinned name so later fresh names never collide.

**Files:**
- Modify: `executor/geom/helpers.py` (the `_counters` / `_next_name` block)
- Modify: `executor/geom/__init__.py:62-80` (`call_tool`)
- Modify: `agent_server/core.py:52-56` (`call_tool` command)
- Modify: `salome_agent/executor_client.py` (`call_tool`)
- Modify: `tests/replay.py` (`ReplayExecutorClient.call_tool`)
- Test: `tests/unit/test_pinned_names.py` (create)

**Interfaces:**
- Produces: `executor.geom.call_tool(executor, name, args, pin=None) -> dict`; `ExecutorClient.call_tool(name, args, pin=None)`; wire request `{"cmd": "call_tool", "name", "args", "pin"?}`. `pin` is omitted from the payload when `None`, so every existing recorded payload is unchanged.

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_pinned_names.py
from types import SimpleNamespace

from executor import geom
from executor.geom import helpers


def _ex(prefix=""):
    return SimpleNamespace(name_prefix=prefix)


def setup_function():
    geom.reset_counters()
    helpers._pinned.clear()


def test_unpinned_allocation_is_unchanged():
    ex = _ex()
    assert helpers._next_name(ex, "box") == "box_1"
    assert helpers._next_name(ex, "box") == "box_2"


def test_pinned_name_is_returned_and_counter_bumped_past_it():
    ex = _ex()
    helpers._pinned[:] = ["box_7"]
    assert helpers._next_name(ex, "box") == "box_7"
    assert helpers._pinned == []
    assert helpers._next_name(ex, "box") == "box_8"


def test_pin_matches_by_prefix_and_leaves_other_prefixes_queued():
    ex = _ex()
    helpers._pinned[:] = ["vec_3", "box_2"]
    assert helpers._next_name(ex, "box") == "box_2"
    assert helpers._pinned == ["vec_3"]
    assert helpers._next_name(ex, "vec") == "vec_3"


def test_pin_respects_the_session_name_prefix():
    ex = _ex("sa_r1_")
    helpers._pinned[:] = ["sa_r1_box_4"]
    assert helpers._next_name(ex, "box") == "sa_r1_box_4"


def test_pin_does_not_match_a_longer_prefix():
    ex = _ex()
    helpers._pinned[:] = ["box_two_1"]
    assert helpers._next_name(ex, "box") == "box_1"
    assert helpers._pinned == ["box_two_1"]


def test_call_tool_scopes_the_pin_to_one_call():
    seen = []

    def fake_tool(executor, args):
        seen.append(helpers._next_name(executor, "box"))
        return {"ok": False, "error": "stop"}

    geom.TOOLS["_fake_pin_tool"] = fake_tool
    try:
        geom.call_tool(_ex(), "_fake_pin_tool", {}, pin=["box_9"])
        geom.call_tool(_ex(), "_fake_pin_tool", {})
    finally:
        del geom.TOOLS["_fake_pin_tool"]
    assert seen == ["box_9", "box_10"]
    assert helpers._pinned == []


def test_core_forwards_pin(monkeypatch):
    from agent_server import core as core_mod
    from agent_server.core import Core

    got = {}

    def fake_call_tool(b, name, args, pin=None):
        got.update(name=name, args=args, pin=pin)
        return {"ok": False, "error": "x"}

    monkeypatch.setattr(core_mod.geom, "call_tool", fake_call_tool)
    backend = SimpleNamespace(poisoned=lambda: False, refresh=lambda m: None)
    Core(backend).handle({"id": 1, "cmd": "call_tool", "name": "make_box",
                          "args": {"dx": 1}, "pin": ["box_1"]})
    assert got == {"name": "make_box", "args": {"dx": 1}, "pin": ["box_1"]}


def test_replay_client_omits_pin_when_none():
    from tests.replay import ReplayExecutorClient

    c = ReplayExecutorClient({"exec": [
        {"cmd": "call_tool", "payload": {"name": "m", "args": {}}, "response": {"ok": True}},
        {"cmd": "call_tool", "payload": {"name": "m", "args": {}, "pin": ["a_1"]},
         "response": {"ok": True}}]})
    assert c.call_tool("m", {}) == {"ok": True}
    assert c.call_tool("m", {}, pin=["a_1"]) == {"ok": True}
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/unit/test_pinned_names.py -q`
Expected: FAIL — `AttributeError: module 'executor.geom.helpers' has no attribute '_pinned'`.

- [ ] **Step 3: Implement**

In `executor/geom/helpers.py`, replace the counters block:

```python
_counters: dict[str, int] = {}
# Names a replayed tool call must reuse, in allocation order (Task 1 of the
# agentic-chat plan). call_tool fills it for one call and clears it after.
_pinned: list[str] = []


def reset_counters() -> None:
    _counters.clear()


def _next_name(executor, prefix: str) -> str:
    stem = f"{executor.name_prefix}{prefix}_"
    for i, name in enumerate(_pinned):
        suffix = name[len(stem):]
        if name.startswith(stem) and suffix.isdigit():
            del _pinned[i]
            _counters[prefix] = max(_counters.get(prefix, 0), int(suffix))
            return name
    _counters[prefix] = _counters.get(prefix, 0) + 1
    return f"{stem}{_counters[prefix]}"
```

In `executor/geom/__init__.py`, add `_pinned` to the `helpers` import list and change `call_tool`:

```python
def call_tool(executor, name: str, args: dict, pin: list | None = None) -> dict:
    handler = TOOLS.get(name)
    if handler is None:
        return {"ok": False, "error": f"unknown tool {name!r}"}
    _pinned[:] = list(pin or [])
    try:
        args = _coerce_args(name, args)
        result = handler(executor, args)
        if result.get("ok"):
            _attach_shape_info(executor, result)
            try:
                note = _delta_note(executor, name, args, result)
            except Exception:
                note = None
            if note:
                result["delta"] = note
        return result
    except Exception as e:
        return {"ok": False, "error": str(e)}
    finally:
        _pinned.clear()
```

In `agent_server/core.py`, the `call_tool` branch:

```python
        if cmd == "call_tool":
            r = geom.call_tool(b, req["name"], req["args"], pin=req.get("pin"))
```

In `salome_agent/executor_client.py`:

```python
    def call_tool(self, name, args, pin=None):
        if pin is None:
            return self._call("call_tool", name=name, args=args)
        return self._call("call_tool", name=name, args=args, pin=pin)
```

In `tests/replay.py`, `ReplayExecutorClient`:

```python
    def call_tool(self, name, args, pin=None):
        payload = {"name": name, "args": args}
        if pin is not None:
            payload["pin"] = pin
        return self._next("call_tool", payload)
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/unit -q`
Expected: all pass, including both golden replays.

- [ ] **Step 5: Commit**

```bash
git add executor/geom/helpers.py executor/geom/__init__.py agent_server/core.py salome_agent/executor_client.py tests/replay.py tests/unit/test_pinned_names.py
git commit -m "feat: pin output names on tool replay (#15)"
```

---

### Task 2: `forget(names)` — drop objects from the study and the bindings

Selective replay has to delete before it rebuilds. `forget` removes every study object with that exact name (GEOM component) and drops the binding. Internal, unpublished vars (`vec_1`, `pt_2`) exist only as bindings, so for those it is a pure `geom_bindings.pop`. Unknown names are not an error.

**Files:**
- Modify: `agent_server/backend.py` (module-level `forget_names`, `HeadlessBackend.forget`, `Backend` protocol)
- Modify: `agent_server/gui_backend.py` (`GuiBackend.forget`)
- Modify: `agent_server/core.py` (new `forget` command)
- Modify: `salome_agent/executor_client.py`, `tests/replay.py` (`forget`)
- Test: `tests/unit/test_forget.py` (create); add `forget` to `FakeBackend` in `tests/unit/test_agent_server_core.py`

**Interfaces:**
- Produces: `agent_server.backend.forget_names(namespace, geom_bindings, names) -> {"ok": True, "removed": [str]}`; `Backend.forget(names) -> dict`; wire `{"cmd": "forget", "names": [...]}` → `{"ok": True, "removed": [...]}`; `ExecutorClient.forget(names) -> dict`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_forget.py
from types import SimpleNamespace

from agent_server.backend import HeadlessBackend, forget_names
from agent_server.core import Core


class FakeStudy:
    def __init__(self, names):
        self.objs = {n: SimpleNamespace(name=n) for n in names}
        self.removed = []

    def FindObjectByName(self, name, component):
        assert component == "GEOM"
        return [self.objs[name]] if name in self.objs else []

    def NewBuilder(self):
        study = self

        class B:
            def RemoveObjectWithChildren(self, so):
                study.removed.append(so.name)
                del study.objs[so.name]
        return B()


def _ns(study):
    return {"salome": SimpleNamespace(myStudy=study), "geompy": object(),
            "smesh": object(), "find_object": lambda n: n}


def test_forget_removes_study_objects_and_bindings():
    study = FakeStudy(["box_1", "fuse_1"])
    bindings = {"box_1": 1, "vec_1": 2, "fuse_1": 3}
    r = forget_names(_ns(study), bindings, ["box_1", "vec_1"])
    assert r == {"ok": True, "removed": ["box_1"]}
    assert study.removed == ["box_1"]
    assert bindings == {"fuse_1": 3}


def test_forget_unknown_name_is_not_an_error():
    r = forget_names(_ns(FakeStudy([])), {}, ["nope_1"])
    assert r == {"ok": True, "removed": []}


def test_headless_backend_forget_uses_its_namespace():
    study = FakeStudy(["box_1"])
    b = HeadlessBackend(build_namespace=lambda live: _ns(study))
    b.geom_bindings["box_1"] = object()
    assert b.forget(["box_1"])["removed"] == ["box_1"]
    assert "box_1" not in b.geom_bindings


def test_core_forget_command_refreshes_geom():
    refreshed = []
    backend = SimpleNamespace(poisoned=lambda: False,
                              forget=lambda names: {"ok": True, "removed": list(names)},
                              refresh=refreshed.append)
    r = Core(backend).handle({"id": 4, "cmd": "forget", "names": ["box_1"]})
    assert r == {"id": 4, "ok": True, "removed": ["box_1"]}
    assert refreshed == ["GEOM"]
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/unit/test_forget.py -q`
Expected: FAIL — `ImportError: cannot import name 'forget_names'`.

- [ ] **Step 3: Implement**

`agent_server/backend.py`, after `_run_with_timeout`:

```python
def forget_names(namespace: dict, geom_bindings: dict, names: list) -> dict:
    study = namespace["salome"].myStudy
    builder = study.NewBuilder()
    removed = []
    for name in names:
        geom_bindings.pop(name, None)
        for sobj in study.FindObjectByName(name, "GEOM"):
            builder.RemoveObjectWithChildren(sobj)
            removed.append(name)
    return {"ok": True, "removed": removed}
```

Add `def forget(self, names: list) -> dict: ...` to the `Backend` protocol, and to `HeadlessBackend`:

```python
    def forget(self, names: list) -> dict:
        return forget_names(self.namespace, self.geom_bindings, names)
```

`agent_server/gui_backend.py`, import `forget_names` from `agent_server.backend` and add to `GuiBackend`:

```python
    def forget(self, names):
        return self._invoker.run_on_main(
            lambda: forget_names(self.namespace, self.geom_bindings, names), timeout=30)
```

`agent_server/core.py`, before `if cmd == "shutdown":`:

```python
        if cmd == "forget":
            r = b.forget(req["names"])
            b.refresh("GEOM")
            return {"id": rid, **r}
```

`salome_agent/executor_client.py`:

```python
    def forget(self, names):
        return self._call("forget", names=names)
```

`tests/replay.py`, `ReplayExecutorClient`:

```python
    def forget(self, names):
        return self._next("forget", {"names": names})
```

In `tests/unit/test_agent_server_core.py`, add to `FakeBackend`:

```python
    def forget(self, names):
        return {"ok": True, "removed": []}
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/unit -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add agent_server/backend.py agent_server/gui_backend.py agent_server/core.py salome_agent/executor_client.py tests/replay.py tests/unit/test_forget.py tests/unit/test_agent_server_core.py
git commit -m "feat: forget command drops objects from the study and bindings (#15)"
```

---

### Task 3: Held connections are never handed over

A conversation idles for minutes between turns. Today, once a connection has been idle ≥ `BUSY_IDLE_S` (30 s), `_claim` lets a second client in, and that client's `hello` wipes the counters and bindings. A connection whose `hello` carries `"hold": true` is not handed over while its socket is open. A dead client still frees it: EOF ends `_handle_conn`, whose `finally` clears both flags.

**Files:**
- Modify: `agent_server/wire.py` (`__init__`, `_claim`, `_handle_conn`)
- Modify: `salome_agent/executor_client.py` (`__init__`)
- Modify: `salome_agent/server_conn.py` (`connect`)
- Test: `tests/unit/test_wire.py` (append), `tests/unit/test_server_conn.py` (append)

**Interfaces:**
- Produces: `ExecutorClient(host, port, token, run_id, record_path=None, hold=False)`; `server_conn.connect(run_id, record_path=None, hold=False)`; hello payload gains `"hold": true` only when `hold=True`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_wire.py`:

```python
def test_held_connection_is_not_handed_over_when_idle(server):
    srv, port = server
    s1, f1 = _client(port)
    f1.write(b'{"cmd":"hello","token":"secret","id":1,"hold":true}\n'); f1.flush()
    f1.readline()
    srv._last_activity -= Server.BUSY_IDLE_S + 1
    s2, f2 = _client(port)
    f2.write(b'{"cmd":"hello","token":"secret","id":1}\n'); f2.flush()
    assert json.loads(f2.readline())["error"] == "busy"
    s1.close(); s2.close()


def test_unheld_idle_connection_is_still_handed_over(server):
    srv, port = server
    s1, f1 = _client(port)
    f1.write(b'{"cmd":"hello","token":"secret","id":1}\n'); f1.flush()
    f1.readline()
    srv._last_activity -= Server.BUSY_IDLE_S + 1
    s2, f2 = _client(port)
    f2.write(b'{"cmd":"hello","token":"secret","id":1}\n'); f2.flush()
    assert json.loads(f2.readline())["ready"] is True
    s1.close(); s2.close()


def test_hold_is_released_when_the_holder_disconnects(server):
    _, port = server
    s1, f1 = _client(port)
    f1.write(b'{"cmd":"hello","token":"secret","id":1,"hold":true}\n'); f1.flush()
    f1.readline()
    f1.close(); s1.close()
    time.sleep(0.3)
    s2, f2 = _client(port)
    f2.write(b'{"cmd":"hello","token":"secret","id":1}\n'); f2.flush()
    assert json.loads(f2.readline())["ready"] is True
    s2.close()
```

Append to `tests/unit/test_server_conn.py` (follow that file's existing monkeypatch style for `read_handshake` / `ExecutorClient`):

```python
def test_connect_passes_hold(monkeypatch):
    from salome_agent import server_conn

    got = {}
    monkeypatch.setattr(server_conn, "read_handshake",
                        lambda: {"host": "h", "port": 1, "token": "t"})

    class FakeClient:
        def __init__(self, host, port, token, run_id, record_path=None, hold=False):
            got["hold"] = hold

    monkeypatch.setattr(server_conn, "ExecutorClient", FakeClient)
    server_conn.connect("r1", hold=True)
    assert got["hold"] is True
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/unit/test_wire.py tests/unit/test_server_conn.py -q`
Expected: `test_held_connection_is_not_handed_over_when_idle` fails (client 2 gets `ready`), `test_connect_passes_hold` fails with `TypeError`.

- [ ] **Step 3: Implement**

`agent_server/wire.py`: in `__init__` add `self._held = False`. Replace `_claim`:

```python
    def _claim(self, hold: bool) -> bool:
        with self._conn_lock:
            if self._active and (self._held or
                                 (time.monotonic() - self._last_activity) < self.BUSY_IDLE_S):
                return False
            self._active = True
            self._held = hold
            self._last_activity = time.monotonic()
            return True
```

In `_handle_conn`: `if not self._claim(bool(req.get("hold"))):`, and in the `finally`:

```python
            if claimed:
                with self._conn_lock:
                    self._active = False
                    self._held = False
```

`salome_agent/executor_client.py` `__init__`: add a `hold: bool = False` parameter, and build the hello as

```python
        hello = {"id": 0, "cmd": "hello", "token": token, "run_id": run_id}
        if hold:
            hello["hold"] = True
        self._f.write((json.dumps(hello) + "\n").encode())
```

`salome_agent/server_conn.py`:

```python
def connect(run_id: str, record_path: str | None = None, hold: bool = False) -> ExecutorClient:
```

and pass `record_path=record_path, hold=hold` to `ExecutorClient`.

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/unit -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add agent_server/wire.py salome_agent/executor_client.py salome_agent/server_conn.py tests/unit/test_wire.py tests/unit/test_server_conn.py
git commit -m "feat: a held agent-server connection is never handed over while idle (#15)"
```

---

### Task 4: Live check of pin + forget against real SALOME

The unit tests fake the study. `FindObjectByName` / `RemoveObjectWithChildren` and name reuse after a forget have to be seen working in a real SALOME before anything is built on them.

**Files:**
- Test: `tests/salome/test_pin_forget_live.py` (create)

**Interfaces:**
- Consumes: the `agent_client` fixture from `tests/salome/conftest.py` (an `ExecutorClient` on a headless server, name prefix `""`).

- [ ] **Step 1: Write the test**

```python
# tests/salome/test_pin_forget_live.py
import pytest

from ._salome_root import SALOME_ROOT

pytestmark = [
    pytest.mark.salome,
    pytest.mark.skipif(SALOME_ROOT is None,
                       reason="set SALOME_ROOT or salome.root in config.json"),
]


def test_forget_then_pinned_rebuild_reuses_the_name(agent_client):
    r = agent_client.call_tool("make_box", {"dx": 10, "dy": 10, "dz": 10})
    name = r["records"][0]["out_vars"][0]

    f = agent_client.forget([name])
    assert f["ok"] and f["removed"] == [name]
    assert agent_client.forget([name])["removed"] == []      # gone from the study

    r2 = agent_client.call_tool("make_box", {"dx": 10, "dy": 10, "dz": 30}, pin=[name])
    assert r2["ok"], r2.get("error")
    assert r2["records"][0]["out_vars"] == [name]
    assert r2["shape_info"]["size"][2] == pytest.approx(30)
    assert agent_client.forget([name])["removed"] == [name]  # exactly one study object

    r3 = agent_client.call_tool("make_box", {"dx": 1, "dy": 1, "dz": 1})
    assert r3["records"][0]["out_vars"][0] != name


def test_pinned_multi_record_call_reuses_the_internal_vector_and_the_output(agent_client):
    box = agent_client.call_tool("make_box", {"dx": 10, "dy": 20, "dz": 5})
    box = box["records"][0]["out_vars"][0]
    r = agent_client.call_tool("rotate", {"shape": box, "axis": "OZ", "angle_deg": 30})
    assert r["ok"], r.get("error")
    out_vars = [v for rec in r["records"] for v in rec["out_vars"]]
    assert len(out_vars) == 2            # the internal "OZ" vector, then the rotated copy

    f = agent_client.forget(out_vars)
    assert out_vars[0] not in f["removed"]   # the vector was never published
    assert out_vars[1] in f["removed"]

    r2 = agent_client.call_tool("rotate", {"shape": box, "axis": "OZ", "angle_deg": 60},
                                pin=out_vars)
    assert r2["ok"], r2.get("error")
    assert [v for rec in r2["records"] for v in rec["out_vars"]] == out_vars
```

- [ ] **Step 2: Run it**

Start nothing by hand. The fixture starts a headless server and refuses if one is already running. Run:
`.venv/bin/python -m pytest tests/salome/test_pin_forget_live.py -m salome -q`
Expected: PASS. If `FindObjectByName` or `RemoveObjectWithChildren` raises, stop and report the exact error. Task 9 and later tasks depend on this.

If no SALOME is available where this task runs, mark the task **blocked on the user**, commit the test, and list it in the hand-off.

- [ ] **Step 3: Commit**

```bash
git add tests/salome/test_pin_forget_live.py
git commit -m "test: live check of pinned replay and forget (#15)"
```

---

### Task 5: ToolSpecialist records a per-call trace and supports extra terminators

A build node needs `{tool, tool_args, records}` per successful call. Only `records` survive today (flattened into `call_log`). The amend loop also needs a second terminator (`replan`).

**Files:**
- Modify: `salome_agent/modules.py` (`ModuleSpec.extra_terminators`)
- Modify: `salome_agent/specialists/tool_specialist.py`
- Test: `tests/unit/test_tool_specialist.py` (update line 33, append tests)

**Interfaces:**
- Produces: `ToolSpecialist.run(..., seed_call_log=None, seed_calls=None)`. Every result that carries `call_log` also carries `calls: list[{"tool": str, "tool_args": dict, "records": list[dict]}]`, one entry per ok call with non-empty `records`, seed first. `ModuleSpec.extra_terminators: tuple[Terminator, ...] = ()`.

- [ ] **Step 1: Write the failing tests**

In `tests/unit/test_tool_specialist.py`, change line 33 to
`assert result == {"out": {"x": 1}, "call_log": [], "calls": []}`, then append:

```python
def test_calls_trace_pairs_tool_and_args_with_records():
    llm = MagicMock()
    llm.chat.side_effect = [_resp(("work", {"a": 1}), ("work", {"a": 2})), _resp(("done", {}))]
    rec = {"fn": "F", "args": [], "out_vars": ["v"], "publish": ["v"]}

    def execute(client, name, args):
        return {"ok": True, "records": [rec] if args["a"] == 1 else []}

    result = ToolSpecialist(_spec(execute=execute)).run(
        {"module": "toy"}, llm, MagicMock(), iteration_cap=3)
    assert result["calls"] == [{"tool": "work", "tool_args": {"a": 1}, "records": [rec]}]


def test_seed_calls_come_first():
    llm = MagicMock()
    llm.chat.side_effect = [_resp(("done", {}))]
    seed = [{"tool": "old", "tool_args": {}, "records": []}]
    result = ToolSpecialist(_spec()).run({"module": "toy"}, llm, MagicMock(),
                                         iteration_cap=3, seed_calls=seed)
    assert result["calls"] == seed


def test_exhausted_result_carries_calls_when_it_carries_call_log():
    llm = MagicMock()
    llm.chat.side_effect = [_resp(("work", {"a": 1}))] * 2
    spec = _spec(execute=lambda c, n, a: {"ok": True, "records": [{"fn": "F", "args": [],
                                                                      "out_vars": [], "publish": None}]},
                 on_exhausted=lambda call_log, last_error: {"call_log": call_log})
    result = ToolSpecialist(spec).run({"module": "toy"}, llm, MagicMock(), iteration_cap=2)
    assert len(result["calls"]) == 2


def test_extra_terminator_ends_the_loop_with_its_own_handle():
    from salome_agent.modules import Terminator

    llm = MagicMock()
    llm.chat.side_effect = [_resp(("bail", {"why": "big"}))]
    bail = Terminator(name="bail", schema={"type": "object", "properties": {}},
                      description="b", handle=lambda args, call_log: {"bailed": args["why"]})
    result = ToolSpecialist(_spec(extra_terminators=(bail,))).run(
        {"module": "toy"}, llm, MagicMock(), iteration_cap=3)
    assert result == {"bailed": "big"}
    names = [t["name"] for t in llm.chat.call_args.kwargs["tools"]]
    assert names == ["work", "done", "bail"]
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/unit/test_tool_specialist.py -q`
Expected: FAIL (`KeyError: 'calls'`, `TypeError ... seed_calls`, `... extra_terminators`).

- [ ] **Step 3: Implement**

`salome_agent/modules.py`, in `ModuleSpec` after `catalogue`:

```python
    # Tools that also end the loop, each with its own handle (amend's replan).
    extra_terminators: tuple = ()
```

`salome_agent/specialists/tool_specialist.py`:

1. `run(...)` and `_run(...)` gain `seed_calls=None` after `seed_call_log`, passed through.
2. After `call_log = ...` add:

```python
        calls: list[dict] = list(seed_calls) if seed_calls else []
        terms = {spec.terminator.name: spec.terminator,
                 **{t.name: t for t in spec.extra_terminators}}
        extra_defs = [{"name": t.name, "parameters": t.schema, "description": t.description}
                      for t in spec.extra_terminators]
```

3. The per-turn `tools = ...` line becomes

```python
            tools = ([all_defs[n] for n in sorted(offered) if n in all_defs]
                     + [finish_def] + extra_defs
                     + ([load_def] if load_def is not None else []))
```

4. `if name == spec.terminator.name:` becomes `if name in terms:` with `finish_call = (call_id, args, terms[name])` (the skipped-message text stays byte-identical). The new outer `calls` accumulator (step 2) collides with the existing per-turn local `calls = response["tool_calls"]` — rename that local (and every use in the turn's loop: the debug log, the `not calls` check, the `id` backfill loop, `assistant_tool_turn(calls, ...)`, the `for k, c in enumerate(calls)` loop and `calls[k + 1:]`) to `tool_calls`.
5. In the ok branch after `call_log.extend(...)`:

```python
                    if result.get("records"):
                        calls.append({"tool": name, "tool_args": args,
                                      "records": list(result["records"])})
```

6. The finish block:

```python
            if finish_call is not None:
                call_id, args, term = finish_call
                if (term is spec.terminator and spec.measure_tools
                        and verify_cadence in ("final", "strict")
                        and risky_since_measure > 0 and not nudged_final):
                    ...unchanged nudge body...
                    continue
                return _with_calls(term.handle(args, call_log), calls)
```

7. The last line: `return _with_calls(spec.on_exhausted(call_log, last_error), calls)`, plus the module-level helper

```python
def _with_calls(result: dict, calls: list) -> dict:
    if "call_log" in result:
        return {**result, "calls": calls}
    return result
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/unit -q`
Expected: all pass. The goldens and `test_tool_payload.py` prove geom's request surface is unchanged, because `extra_defs` is empty for geom.

- [ ] **Step 5: Commit**

```bash
git add salome_agent/modules.py salome_agent/specialists/tool_specialist.py tests/unit/test_tool_specialist.py
git commit -m "feat: specialist loop records a per-call trace and takes extra terminators (#15)"
```

---

### Task 6: Stages carry their call trace; retries seed it

**Files:**
- Modify: `salome_agent/state.py` (`StageRecord.calls`)
- Modify: `salome_agent/graph.py` (`retry_context`, `_run_stage_node`)
- Modify: `salome_agent/specialists/geom.py:160-174` (`run_geom_specialist` passes `seed_calls`)
- Test: `tests/unit/test_graph.py` (update the fake `run` signatures — there are five, not two, all matching `seed_call_log=None):` — append tests)

**Interfaces:**
- Consumes: Task 5 `calls`.
- Produces: an accepted geom stage record has `stage["calls"]` (list, same shape as Task 5). `retry_context(...)["calls"]` is the failed attempt's trace.

- [ ] **Step 1: Write the failing tests**

In every fake `run` definition in `tests/unit/test_graph.py` (five of them, each monkeypatched onto `ToolSpecialist.run` and reached through `graph._run_stage_node`, which now always passes `seed_calls=` — a fake missing the kwarg raises `TypeError`), add `seed_calls=None` after `seed_call_log=None`. Append:

```python
def test_retry_context_carries_the_failed_attempts_calls():
    calls = [{"tool": "make_box", "tool_args": {"dx": 1}, "records": []}]
    state = {"stages": {0: {"status": "failed", "call_log": [], "calls": calls}},
             "last_error": None, "history": []}
    assert retry_context(state, 0, client=None)["calls"] == calls
```

```python
def _one_geom_stage(status="pending", **stage):
    return {"stage_idx": 0,
            "plan": [{"module": "geom", "intent": "box", "params": {}, "inputs": []}],
            "stages": {0: {"module": "geom", "status": status, **stage}},
            "config": {"policy": {"tool_iteration_cap": 5, "on_stage_failure": "abort"}},
            "last_error": None, "history": []}


def test_run_stage_keeps_the_call_trace(monkeypatch):
    from salome_agent import graph
    from salome_agent.specialists.tool_specialist import ToolSpecialist

    calls = [{"tool": "make_box", "tool_args": {"dx": 1}, "records": []}]

    def fake_run(self, stage_plan, llm, client, iteration_cap, *, blocks=(),
                 tool_selection="hints", verify_cadence="final", retry=None,
                 seed_call_log=None, seed_calls=None):
        return {"call_log": [], "calls": calls, "result_names": ["box_1"]}

    monkeypatch.setattr(ToolSpecialist, "run", fake_run)
    cmd = graph._run_stage_node(_one_geom_stage(), {"configurable": {
        "llm_client": MagicMock(), "executor_client": None}})
    assert cmd.update["stages"][0]["calls"] == calls


def test_retry_seeds_the_failed_attempts_calls(monkeypatch):
    from salome_agent import graph
    from salome_agent.specialists.tool_specialist import ToolSpecialist

    old = [{"tool": "make_box", "tool_args": {"dx": 1}, "records": []}]
    seen = {}

    def fake_run(self, stage_plan, llm, client, iteration_cap, *, blocks=(),
                 tool_selection="hints", verify_cadence="final", retry=None,
                 seed_call_log=None, seed_calls=None):
        seen["seed_calls"] = seed_calls
        return {"call_log": [], "calls": old, "result_names": ["box_1"]}

    monkeypatch.setattr(ToolSpecialist, "run", fake_run)
    graph._run_stage_node(_one_geom_stage("failed", call_log=[], calls=old),
                          {"configurable": {"llm_client": MagicMock(), "executor_client": None}})
    assert seen["seed_calls"] == old
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/unit/test_graph.py -q`
Expected: the two new tests fail.

- [ ] **Step 3: Implement**

`salome_agent/state.py`, in `StageRecord` after `call_log`:

```python
    # Task 5's per-call trace: [{tool, tool_args, records}] — what a build
    # node is made from. None for script modules.
    calls: Optional[list[dict]]
```

`salome_agent/graph.py`:
- `retry_context` return dict gains `"calls": stage.get("calls") or []`.
- In `_run_stage_node`, the geom branch sets `seed_calls = (retry or {}).get("calls")` next to `seed`. The non-geom branch sets `seed_calls = None`. Pass `seed_calls=seed_calls` to `.run(...)`.
- Where `stage["call_log"] = result.get("call_log")` is set, add `stage["calls"] = result.get("calls")`.

`salome_agent/specialists/geom.py` `run_geom_specialist`: add a `seed_calls: list[dict] | None = None` parameter and pass it through.

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/unit -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add salome_agent/state.py salome_agent/graph.py salome_agent/specialists/geom.py tests/unit/test_graph.py
git commit -m "feat: geom stages keep their call trace across retries (#15)"
```

---

### Task 7: `build_graph.py` — the pure-data build graph

No SALOME, no LLM. It gets the heaviest unit coverage in this plan.

**Files:**
- Create: `salome_agent/build_graph.py`
- Test: `tests/unit/test_build_graph.py` (create)

**Interfaces:**
- Produces:
  - `class GraphError(ValueError)`
  - `@dataclass BuildNode(id: str, turn: int, block: str, module: str, tool: str, tool_args: dict, records: list, deps: list)` with properties `out_vars -> list[str]`, `published -> list[str]`
  - `@dataclass Block(id: str, turn: int, label: str, module: str, result_names: list, script: str | None = None)`
  - `class BuildGraph` with `nodes: dict[str, BuildNode]`, `blocks: list[Block]`, `requests: list[str]` and methods:
    - `add_block(turn, label, module, result_names, script=None) -> str` (ids `b1`, `b2`, …)
    - `append(block, tool, tool_args, records, turn, module="geom") -> str` (ids `n1`, `n2`, …, never reused)
    - `owner(var) -> str | None`
    - `edit_arg(node_id, param, value) -> None`
    - `delete(node_id) -> list[BuildNode]`
    - `affected(node_id) -> list[str]` (the node plus its forward closure, ascending id = topological)
    - `set_records(node_id, records) -> None`
    - `block_nodes(block_id) -> list[BuildNode]`
    - `result_names() -> list[str]` (geom blocks: `result_names` if any, else every published name of the block's nodes)
    - `all_out_vars() -> list[str]`
    - `copy() -> BuildGraph`, `to_dict() -> dict`, `BuildGraph.from_dict(d: dict | None) -> BuildGraph`
    - `render() -> str`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_build_graph.py
import pytest

from salome_agent.build_graph import BuildGraph, GraphError


def _rec(fn, out, refs=(), publish=True, **extra):
    r = {"fn": fn, "args": [{"kind": "ref", "value": x} for x in refs]
         + [{"kind": "literal", "value": 1}],
         "out_vars": list(out), "publish": list(out) if publish else None}
    r.update(extra)
    return r


def _house():
    g = BuildGraph()
    b = g.add_block(0, "stage 1/1: geom", "geom", ["fuse_1"])
    g.append(b, "make_box", {"dx": 200, "dy": 120, "dz": 80}, [_rec("MakeBoxDXDYDZ", ["box_1"])], 0)
    g.append(b, "make_box", {"dx": 200, "dy": 120, "dz": 40},
             [_rec("MakeVectorDXDYDZ", ["vec_1"], publish=False),
              _rec("MakeTranslation", ["box_2"], refs=["vec_1"])], 0)
    g.append(b, "fuse", {"shapes": ["box_1", "box_2"]},
             [_rec("MakeFuse", ["fuse_1"], refs=["box_1", "box_2"])], 0)
    g.append(b, "set_color", {"shape": "fuse_1", "red": 255, "green": 0, "blue": 0},
             [{"fn": "SetColor", "args": [], "out_vars": [], "publish": None,
               "emit": "method_call", "recv": "fuse_1"}], 0)
    return g, b


def test_append_assigns_ids_and_deps_from_refs_and_recv():
    g, _ = _house()
    assert list(g.nodes) == ["n1", "n2", "n3", "n4"]
    assert g.nodes["n2"].deps == []          # vec_1 is its own record, not a dep
    assert g.nodes["n3"].deps == ["n1", "n2"]
    assert g.nodes["n4"].deps == ["n3"]      # method_call recv counts


def test_out_vars_and_published():
    g, _ = _house()
    assert g.nodes["n2"].out_vars == ["vec_1", "box_2"]
    assert g.nodes["n2"].published == ["box_2"]


def test_affected_is_forward_closure_in_id_order():
    g, _ = _house()
    assert g.affected("n1") == ["n1", "n3", "n4"]
    assert g.affected("n4") == ["n4"]


def test_edit_arg_replaces_a_literal():
    g, _ = _house()
    g.edit_arg("n1", "dz", 120)
    assert g.nodes["n1"].tool_args["dz"] == 120


def test_edit_arg_may_add_an_absent_literal_param():
    g, _ = _house()
    g.edit_arg("n1", "note", 3)
    assert g.nodes["n1"].tool_args["note"] == 3


def test_edit_arg_refuses_to_rewrite_a_ref():
    g, _ = _house()
    with pytest.raises(GraphError, match="delete_node"):
        g.edit_arg("n3", "shapes", ["box_1"])
    with pytest.raises(GraphError):
        g.edit_arg("n4", "shape", "box_1")


def test_edit_arg_refuses_a_literal_becoming_a_ref():
    g, _ = _house()
    with pytest.raises(GraphError):
        g.edit_arg("n1", "dz", "box_2")


def test_dead_node_error_names_the_live_ones():
    g, _ = _house()
    g.delete("n4")
    with pytest.raises(GraphError, match=r"n4 .*n1, n2, n3"):
        g.edit_arg("n4", "red", 0)


def test_delete_removes_downstream_and_prunes_result_names():
    g, b = _house()
    removed = g.delete("n2")
    assert [n.id for n in removed] == ["n2", "n3", "n4"]
    assert list(g.nodes) == ["n1"]
    assert g.blocks[0].result_names == []


def test_delete_drops_an_emptied_geom_block_but_keeps_script_blocks():
    g, b = _house()
    g.add_block(0, "stage 2/2: smesh", "smesh", [], script="mesh = 1")
    g.delete("n1"); g.delete("n2")
    assert [bl.module for bl in g.blocks] == ["smesh"]


def test_ids_are_never_reused():
    g, b = _house()
    g.delete("n4")
    assert g.append(b, "make_box", {"dx": 1, "dy": 1, "dz": 1},
                    [_rec("MakeBoxDXDYDZ", ["box_3"])], 1) == "n5"


def test_result_names_falls_back_to_published():
    g, b = _house()
    b2 = g.add_block(1, "turn 2: amend", "geom", [])
    g.append(b2, "make_box", {"dx": 1, "dy": 1, "dz": 1}, [_rec("MakeBoxDXDYDZ", ["box_3"])], 1)
    assert g.result_names() == ["fuse_1", "box_3"]


def test_set_records_recomputes_deps():
    g, _ = _house()
    g.set_records("n3", [_rec("MakeFuse", ["fuse_1"], refs=["box_1"])])
    assert g.nodes["n3"].deps == ["n1"]


def test_round_trip_and_copy_are_independent():
    g, _ = _house()
    g.requests.append("a house")
    g2 = BuildGraph.from_dict(g.to_dict())
    assert g2.to_dict() == g.to_dict()
    c = g.copy()
    c.edit_arg("n1", "dz", 1)
    assert g.nodes["n1"].tool_args["dz"] == 80
    assert BuildGraph.from_dict(None).to_dict() == BuildGraph().to_dict()


def test_render_is_tool_level():
    g, _ = _house()
    text = g.render()
    assert "# stage 1/1: geom" in text
    assert 'n1  box_1 = make_box(dx=200, dy=120, dz=80)' in text
    assert 'n3  fuse_1 = fuse(shapes=["box_1", "box_2"])  <- n1, n2' in text
    assert 'n4  set_color(shape="fuse_1", red=255, green=0, blue=0)  <- n3' in text
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/unit/test_build_graph.py -q`
Expected: FAIL — `ModuleNotFoundError: salome_agent.build_graph`.

- [ ] **Step 3: Implement**

```python
# salome_agent/build_graph.py
import copy
import json
from dataclasses import asdict, dataclass, field

from salome_agent.codegen import _refs_in


class GraphError(ValueError):
    pass


def _num(node_id: str) -> int:
    return int(node_id[1:])


@dataclass
class BuildNode:
    id: str
    turn: int
    block: str
    module: str
    tool: str
    tool_args: dict
    records: list
    deps: list = field(default_factory=list)

    @property
    def out_vars(self) -> list[str]:
        return [v for r in self.records for v in r.get("out_vars", [])]

    @property
    def published(self) -> list[str]:
        return [n for r in self.records for n in (r.get("publish") or [])]


@dataclass
class Block:
    id: str
    turn: int
    label: str
    module: str
    result_names: list
    script: str | None = None


def _record_refs(records: list) -> set[str]:
    refs: set[str] = set()
    for r in records:
        for arg in r.get("args", []):
            refs |= _refs_in(arg)
        if not r.get("out_vars") and r.get("recv"):
            refs.add(r["recv"])
    return refs


def _mentions(value, names: set[str]) -> bool:
    if isinstance(value, str):
        return value in names
    if isinstance(value, (list, tuple)):
        return any(_mentions(v, names) for v in value)
    return False


class BuildGraph:
    """The conversation's accumulated tool calls. result.py and the study are
    both projections of it."""

    def __init__(self):
        self.nodes: dict[str, BuildNode] = {}
        self.blocks: list[Block] = []
        self.requests: list[str] = []
        self._next_node = 1
        self._next_block = 1

    def add_block(self, turn: int, label: str, module: str, result_names: list,
                  script: str | None = None) -> str:
        bid = f"b{self._next_block}"
        self._next_block += 1
        self.blocks.append(Block(bid, turn, label, module, list(result_names), script))
        return bid

    def owner(self, var: str) -> str | None:
        for node in self.nodes.values():
            if var in node.out_vars:
                return node.id
        return None

    def _deps(self, node_id: str, records: list) -> list[str]:
        own = {v for r in records for v in r.get("out_vars", [])}
        deps = {self.owner(ref) for ref in _record_refs(records) - own}
        deps.discard(None)
        deps.discard(node_id)
        return sorted(deps, key=_num)

    def append(self, block: str, tool: str, tool_args: dict, records: list, turn: int,
               module: str = "geom") -> str:
        nid = f"n{self._next_node}"
        self._next_node += 1
        self.nodes[nid] = BuildNode(nid, turn, block, module, tool, dict(tool_args),
                                    list(records), self._deps(nid, records))
        return nid

    def _live(self, node_id: str) -> BuildNode:
        node = self.nodes.get(node_id)
        if node is None:
            live = ", ".join(self.nodes) or "(none)"
            raise GraphError(f"{node_id} does not exist (deleted or never built); "
                             f"current nodes are {live}")
        return node

    def edit_arg(self, node_id: str, param: str, value) -> None:
        node = self._live(node_id)
        names = set(self.all_out_vars())
        if _mentions(node.tool_args.get(param), names) or _mentions(value, names):
            raise GraphError(
                f"{node_id}.{param} names a shape; edit_arg only changes literal values. "
                f"To change which shapes {node_id} uses, delete_node it and build it again.")
        node.tool_args[param] = value

    def affected(self, node_id: str) -> list[str]:
        self._live(node_id)
        hit = {node_id}
        for nid in sorted(self.nodes, key=_num):
            if hit & set(self.nodes[nid].deps):
                hit.add(nid)
        return sorted(hit, key=_num)

    def delete(self, node_id: str) -> list[BuildNode]:
        ids = self.affected(node_id)
        removed = [self.nodes.pop(nid) for nid in ids]
        gone = {v for n in removed for v in n.out_vars}
        for block in self.blocks:
            block.result_names = [n for n in block.result_names if n not in gone]
        self.blocks = [b for b in self.blocks
                       if b.script is not None or self.block_nodes(b.id)]
        return removed

    def set_records(self, node_id: str, records: list) -> None:
        node = self._live(node_id)
        node.records = list(records)
        node.deps = self._deps(node_id, records)

    def block_nodes(self, block_id: str) -> list[BuildNode]:
        return [n for nid, n in sorted(self.nodes.items(), key=lambda kv: _num(kv[0]))
                if n.block == block_id]

    def result_names(self) -> list[str]:
        out: list[str] = []
        for b in self.blocks:
            if b.script is not None:
                continue
            out += b.result_names or [p for n in self.block_nodes(b.id) for p in n.published]
        return out

    def all_out_vars(self) -> list[str]:
        return [v for nid in sorted(self.nodes, key=_num) for v in self.nodes[nid].out_vars]

    def copy(self) -> "BuildGraph":
        return copy.deepcopy(self)

    def to_dict(self) -> dict:
        return {"version": 1, "requests": list(self.requests),
                "next_node": self._next_node, "next_block": self._next_block,
                "blocks": [asdict(b) for b in self.blocks],
                "nodes": [asdict(self.nodes[nid]) for nid in sorted(self.nodes, key=_num)]}

    @classmethod
    def from_dict(cls, d: dict | None) -> "BuildGraph":
        g = cls()
        if not d:
            return g
        g.requests = list(d.get("requests") or [])
        g._next_node = d.get("next_node", 1)
        g._next_block = d.get("next_block", 1)
        g.blocks = [Block(**b) for b in d.get("blocks") or []]
        g.nodes = {n["id"]: BuildNode(**n) for n in d.get("nodes") or []}
        return g

    def render(self) -> str:
        lines = []
        for b in self.blocks:
            lines.append(f"# {b.label}")
            if b.script is not None:
                lines.append("    (script stage — not editable)")
                continue
            for n in self.block_nodes(b.id):
                args = ", ".join(f"{k}={json.dumps(v)}" for k, v in n.tool_args.items())
                lhs = f"{', '.join(n.published)} = " if n.published else ""
                deps = f"  <- {', '.join(n.deps)}" if n.deps else ""
                lines.append(f"{n.id:<4}{lhs}{n.tool}({args}){deps}")
        return "\n".join(lines)
```

Note for `test_render_is_tool_level`: `json.dumps(200)` is `200`, and `json.dumps(["box_1","box_2"])` is `["box_1", "box_2"]`. The test fixture uses ints on purpose, so render prints them unchanged.

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/unit -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add salome_agent/build_graph.py tests/unit/test_build_graph.py
git commit -m "feat: build graph of tool-call nodes with edit, delete and forward closure (#15)"
```

---

### Task 8: `result.py` becomes a projection of the build graph

**Files:**
- Modify: `salome_agent/codegen.py` (`roots_for`, `generate(keep=)`)
- Modify: `salome_agent/nodes/assembler.py` (`commit_stages`, `render_script`, `assemble` wrapper)
- Modify: `salome_agent/graph.py` (`_assemble_node`)
- Test: `tests/unit/test_assembler.py` (append), `tests/unit/test_codegen.py` (append)

**Interfaces:**
- Consumes: Task 6 `stage["calls"]`, Task 7 `BuildGraph`.
- Produces:
  - `codegen.roots_for(call_log, result_names) -> list[str]` (the issue-#7 fallback, moved out of `generate`)
  - `codegen.generate(call_log, result_names, call_prefix="geompy.", keep: set | None = None) -> str`. With `keep`, that set replaces the local reachability computation.
  - `assembler.commit_stages(graph, state) -> BuildGraph` (mutates and returns `graph`)
  - `assembler.render_script(graph, run_id) -> str`
  - `assembler.assemble(state) -> str` (unchanged signature; = `render_script(commit_stages(BuildGraph.from_dict(state.get("build_graph")), state), state["run_id"])`)
  - `_assemble_node` update gains `"build_graph": graph.to_dict()`.

**Block labels.** Turn 0: `stage {idx+1}/{total}: {module}` (as today). Later turns: `turn {t+1} stage {idx+1}/{total}: {module}`. A geom stage with `calls is None` (older callers, unit tests) becomes a script block holding `stage["script"]`, exactly today's behaviour.

**Header.** `# generated by salome-agent, run {run_id}`, `# request: {requests[0]}`, then one `# turn {i+1}: {requests[i]}` line per later request. A single-turn run's header is byte-identical to today.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_codegen.py`:

```python
def test_generate_with_keep_uses_that_set():
    from salome_agent.codegen import generate
    log = [{"fn": "MakeBoxDXDYDZ", "args": [{"kind": "literal", "value": 1}] * 3,
            "out_vars": ["box_1"], "publish": ["box_1"]},
           {"fn": "MakeBoxDXDYDZ", "args": [{"kind": "literal", "value": 2}] * 3,
            "out_vars": ["box_2"], "publish": ["box_2"]}]
    out = generate(log, ["box_2"], keep={"box_1", "box_2"})
    assert "box_1 = " in out and "box_2 = " in out


def test_roots_for_falls_back_to_published():
    from salome_agent.codegen import roots_for
    log = [{"fn": "F", "args": [], "out_vars": ["a"], "publish": ["a"]}]
    assert roots_for(log, ["nope"]) == ["a"]
    assert roots_for(log, ["a"]) == ["a"]
```

Append to `tests/unit/test_assembler.py`:

```python
def _box(name, dz=1.0):
    return {"fn": "MakeBoxDXDYDZ", "args": [{"kind": "literal", "value": 1.0},
                                            {"kind": "literal", "value": 1.0},
                                            {"kind": "literal", "value": dz}],
            "out_vars": [name], "publish": [name]}


def _fuse(out, a, b):
    return {"fn": "MakeFuse", "args": [{"kind": "ref", "value": a}, {"kind": "ref", "value": b}],
            "out_vars": [out], "publish": [out]}


def _state_with_calls():
    s1 = [{"tool": "make_box", "tool_args": {}, "records": [_box("box_1")]},
          {"tool": "make_box", "tool_args": {}, "records": [_box("box_2")]}]
    s2 = [{"tool": "fuse", "tool_args": {}, "records": [_fuse("fuse_1", "box_1", "box_2")]}]
    return {"run_id": "r1", "user_request": "two boxes fused",
            "stages": {0: {"module": "geom", "status": "ok", "script": "IGNORED",
                           "calls": s1, "result_names": ["box_1"]},
                       1: {"module": "geom", "status": "ok", "script": "IGNORED",
                           "calls": s2, "result_names": ["fuse_1"]}}}


def test_graph_projection_keeps_stage_headers_and_crosses_stage_refs():
    out = assemble(_state_with_calls())
    assert "# stage 1/2: geom\nbox_1 = geompy.MakeBoxDXDYDZ(1.0, 1.0, 1.0)" in out
    # box_2 is not in stage 1's result_names but stage 2 needs it: whole-graph reachability
    assert "box_2 = geompy.MakeBoxDXDYDZ(1.0, 1.0, 1.0)" in out
    assert "# stage 2/2: geom\nfuse_1 = geompy.MakeFuse(box_1, box_2)" in out
    assert "IGNORED" not in out


def test_commit_stages_labels_later_turns_and_records_requests():
    from salome_agent.build_graph import BuildGraph
    from salome_agent.nodes.assembler import commit_stages, render_script
    g = commit_stages(BuildGraph(), _state_with_calls())
    st = _state_with_calls()
    st.update(turn_idx=1, user_request="add a third", stages={
        0: {"module": "geom", "status": "ok", "script": None, "result_names": ["box_3"],
            "calls": [{"tool": "make_box", "tool_args": {}, "records": [_box("box_3")]}]}})
    commit_stages(g, st)
    assert [b.label for b in g.blocks][-1] == "turn 2 stage 1/1: geom"
    out = render_script(g, "r1")
    assert out.startswith("# generated by salome-agent, run r1\n# request: two boxes fused\n"
                          "# turn 2: add a third\n")


def test_script_block_for_geom_stage_without_calls_is_todays_output():
    state = {"run_id": "r1", "user_request": "x",
             "stages": {0: {"module": "geom", "status": "ok", "script": "a = 1"}}}
    assert "# stage 1/1: geom\na = 1" in assemble(state)
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/unit/test_codegen.py tests/unit/test_assembler.py -q`
Expected: the new tests fail.

- [ ] **Step 3: Implement**

`salome_agent/codegen.py`: move the fallback block at the top of `generate` into

```python
def roots_for(call_log: list[dict], result_names: list[str]) -> list[str]:
    # A weak model can call finish_stage with names that match no shape it
    # actually built (invented names, or an empty list). Pruning to those would
    # emit nothing and silently drop the whole stage from result.py (issue #7),
    # so fall back to keeping every published shape when nothing resolves.
    built = {v for call in call_log for v in call.get("out_vars", [])}
    # A display/method_call record (set_color, set_transparency, ...) has no
    # out_vars — it mutates an existing object named by `recv` — but naming
    # that object in finish_stage is a legitimate match, not an invented name.
    built |= {call["recv"] for call in call_log if not call.get("out_vars") and call.get("recv")}
    if set(result_names) & built:
        return list(result_names)
    from salome_agent.logs import log

    if call_log:
        log.warning("codegen: result_names %s match no built shape — keeping "
                    "every published shape for this stage", result_names)
    return [n for call in call_log for n in (call.get("publish") or [])]
```

and start `generate` with

```python
def generate(call_log: list[dict], result_names: list[str],
             call_prefix: str = "geompy.", keep: set | None = None) -> str:
    if keep is None:
        reachable = _reachable(call_log, roots_for(call_log, result_names))
    else:
        reachable = keep
    reachable_calls = _reachable_with_display(call_log, reachable)
```

The rest of `generate` is unchanged. Note that `generate` still logs the fallback warning exactly as before, via `roots_for`.

`salome_agent/nodes/assembler.py`: keep every constant and the preamble logic byte-for-byte. Replace `assemble` with:

```python
def commit_stages(graph, state: dict):
    turn = state.get("turn_idx") or 0
    if len(graph.requests) <= turn:
        graph.requests.append(state["user_request"])
    total = len(state["stages"])
    for idx in sorted(state["stages"]):
        stage = state["stages"][idx]
        if stage.get("status") != "ok":
            continue
        where = f"stage {idx + 1}/{total}" if turn == 0 else f"turn {turn + 1} stage {idx + 1}/{total}"
        label = f"{where}: {stage['module']}"
        calls = stage.get("calls")
        if stage["module"] == "geom" and calls is not None:
            bid = graph.add_block(turn, label, "geom", stage.get("result_names") or [])
            for c in calls:
                graph.append(bid, c["tool"], c["tool_args"], c["records"], turn)
        elif stage.get("script"):
            graph.add_block(turn, label, stage["module"], [], script=stage["script"])
    return graph


def _geom_bodies(graph) -> dict:
    from salome_agent.codegen import _reachable, generate, roots_for
    from salome_agent.modules import spec_for

    per_block, roots = {}, []
    for b in graph.blocks:
        if b.script is None:
            records = [r for n in graph.block_nodes(b.id) for r in n.records]
            names = b.result_names or [p for n in graph.block_nodes(b.id) for p in n.published]
            per_block[b.id] = (records, names)
            roots += roots_for(records, names)
    everything = [r for records, _ in per_block.values() for r in records]
    keep = _reachable(everything, roots)
    prefix = spec_for("geom").call_prefix or "geompy."
    return {bid: generate(records, names, call_prefix=prefix, keep=keep)
            for bid, (records, names) in per_block.items()}


def render_script(graph, run_id: str) -> str:
    from salome_agent.modules import spec_for

    requests = graph.requests or [""]
    header = f"# generated by salome-agent, run {run_id}\n# request: {requests[0]}\n"
    for i, r in enumerate(requests[1:], start=2):
        header += f"# turn {i}: {r}\n"

    geom_bodies = _geom_bodies(graph)
    bodies = []
    modules = set()
    for b in graph.blocks:
        text = b.script if b.script is not None else geom_bodies.get(b.id)
        if text:
            bodies.append(f"# {b.label}\n{text}")
            modules.add(b.module)
    body = "\n\n".join(bodies)

    # ---- everything below this line is the old assemble() tail, unchanged ----
    preamble = _KERNEL_INIT
    ...
    return header + "\n" + preamble + "\n" + body + "\n"


def assemble(state: dict) -> str:
    from salome_agent.build_graph import BuildGraph

    graph = commit_stages(BuildGraph.from_dict(state.get("build_graph")), state)
    return render_script(graph, state["run_id"])
```

(`...` above means: paste the existing preamble lines verbatim, from `preamble = _KERNEL_INIT` through the `study_ns` check.)

`salome_agent/graph.py` `_assemble_node`:

```python
def _assemble_node(state: dict):
    from langgraph.types import Command
    from salome_agent.build_graph import BuildGraph
    from salome_agent.nodes.assembler import commit_stages, render_script

    log.info("assembling result.py")
    graph = commit_stages(BuildGraph.from_dict(state.get("build_graph")), state)
    return Command(update={"final_script": render_script(graph, state["run_id"]),
                           "build_graph": graph.to_dict()}, goto="report")
```

Add `build_graph: Optional[dict]` to `RunState` in `salome_agent/state.py` now (Task 11 adds the rest).

- [ ] **Step 4: Run tests — goldens are the gate**

Run: `.venv/bin/python -m pytest tests/unit -q`
Expected: all pass. **If `test_golden_runs.py` fails, do not touch the golden files.** Diff the output and fix the projection until it is byte-identical.

- [ ] **Step 5: Commit**

```bash
git add salome_agent/codegen.py salome_agent/nodes/assembler.py salome_agent/graph.py salome_agent/state.py tests/unit/test_codegen.py tests/unit/test_assembler.py
git commit -m "feat: result.py is a projection of the build graph (#15)"
```

---

### Task 9: `replay.py` — selective replay, rollback, sync and rebuild

**Files:**
- Create: `salome_agent/replay.py`
- Test: `tests/unit/test_replay.py` (create)

**Interfaces:**
- Consumes: Task 1 `client.call_tool(name, args, pin=)`, Task 2 `client.forget(names)`, Task 7 `BuildGraph`.
- Produces:
  - `class ReplayFailed(RuntimeError)` with `.node_id`, `.error`
  - `class RollbackFailed(RuntimeError)` with `.graph` (the latest graph the study may partially match)
  - `replay(client, graph, ids) -> None` (mutates `graph` records in place)
  - `apply_edit(client, graph, node_id, param, value) -> tuple[BuildGraph, list[str]]` (new graph, rebuilt ids)
  - `apply_delete(client, graph, node_id) -> tuple[BuildGraph, list[str]]` (new graph, removed ids)
  - `sync(client, current, target) -> BuildGraph` (makes the study match `target`, returns a fresh copy of it)
  - `rebuild_all(client, graph, extra_names=()) -> BuildGraph`
  - `GraphError` from `build_graph` propagates unchanged out of `apply_edit` / `apply_delete`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_replay.py
import pytest

from salome_agent.build_graph import BuildGraph, GraphError
from salome_agent.replay import (ReplayFailed, RollbackFailed, apply_delete, apply_edit,
                                 rebuild_all, sync)


class FakeClient:
    """Builds boxes (dz is the only thing that matters) and fuses; `fail_on`
    makes a tool call with those exact args fail."""

    def __init__(self):
        self.study = {}
        self.log = []
        self.fail_on = []

    def forget(self, names):
        self.log.append(("forget", sorted(names)))
        for n in names:
            self.study.pop(n, None)
        return {"ok": True, "removed": []}

    def call_tool(self, name, args, pin=None):
        self.log.append(("call", name, dict(args), list(pin or [])))
        if (name, args) in self.fail_on:
            return {"ok": False, "error": "boom"}
        out = pin[0]
        self.study[out] = (name, dict(args))
        refs = args.get("shapes", [])
        return {"ok": True, "records": [{"fn": name, "args": [{"kind": "ref", "value": r}
                                                              for r in refs],
                                         "out_vars": [out], "publish": [out]}]}


def _graph(client):
    g = BuildGraph()
    b = g.add_block(0, "stage 1/1: geom", "geom", ["fuse_1"])
    for name, args, out in [("make_box", {"dz": 80}, "box_1"),
                            ("make_box", {"dz": 40}, "box_2"),
                            ("fuse", {"shapes": ["box_1", "box_2"]}, "fuse_1")]:
        r = client.call_tool(name, args, pin=[out])
        g.append(b, name, args, r["records"], 0)
    client.log.clear()
    return g


def test_apply_edit_forgets_then_rebuilds_the_closure_with_pinned_names():
    c = FakeClient()
    g = _graph(c)
    new, ids = apply_edit(c, g, "n2", "dz", 120)
    assert ids == ["n2", "n3"]
    assert c.log == [("forget", ["box_2", "fuse_1"]),
                     ("call", "make_box", {"dz": 120}, ["box_2"]),
                     ("call", "fuse", {"shapes": ["box_1", "box_2"]}, ["fuse_1"])]
    assert c.study["box_2"] == ("make_box", {"dz": 120})
    assert g.nodes["n2"].tool_args["dz"] == 40      # input graph untouched
    assert new.nodes["n2"].tool_args["dz"] == 120


def test_apply_edit_failure_restores_the_pre_edit_subtree():
    c = FakeClient()
    g = _graph(c)
    c.fail_on = [("make_box", {"dz": -1})]
    with pytest.raises(ReplayFailed) as e:
        apply_edit(c, g, "n2", "dz", -1)
    assert e.value.node_id == "n2"
    assert c.study["box_2"] == ("make_box", {"dz": 40})
    assert c.study["fuse_1"][0] == "fuse"


def test_apply_edit_rollback_failure_is_rollbackfailed():
    c = FakeClient()
    g = _graph(c)
    c.fail_on = [("make_box", {"dz": -1}), ("make_box", {"dz": 40})]
    with pytest.raises(RollbackFailed):
        apply_edit(c, g, "n2", "dz", -1)


def test_apply_edit_bad_node_is_grapherror_and_touches_nothing():
    c = FakeClient()
    g = _graph(c)
    with pytest.raises(GraphError):
        apply_edit(c, g, "n9", "dz", 1)
    assert c.log == []


def test_apply_delete_forgets_the_closure():
    c = FakeClient()
    g = _graph(c)
    new, removed = apply_delete(c, g, "n1")
    assert removed == ["n1", "n3"]
    assert c.log == [("forget", ["box_1", "fuse_1"])]
    assert list(new.nodes) == ["n2"]


def test_sync_reverts_an_edit_an_append_and_a_delete():
    c = FakeClient()
    before = _graph(c)
    after, _ = apply_edit(c, before, "n1", "dz", 5)
    after, _ = apply_delete(c, after, "n2")          # also removes n3
    b = after.blocks[0].id if after.blocks else after.add_block(1, "turn 2: amend", "geom", [])
    r = c.call_tool("make_box", {"dz": 9}, pin=["box_3"])
    after.append(b, "make_box", {"dz": 9}, r["records"], 1)
    restored = sync(c, current=after, target=before)
    assert set(c.study) == {"box_1", "box_2", "fuse_1"}
    assert c.study["box_1"] == ("make_box", {"dz": 80})
    assert restored.to_dict() == before.to_dict()


def test_rebuild_all_forgets_everything_and_replays_in_order():
    c = FakeClient()
    g = _graph(c)
    rebuild_all(c, g, extra_names=["stray_1"])
    assert c.log[0] == ("forget", ["box_1", "box_2", "fuse_1", "stray_1"])
    assert [e[1] for e in c.log[1:]] == ["make_box", "make_box", "fuse"]
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/unit/test_replay.py -q`
Expected: FAIL — `ModuleNotFoundError: salome_agent.replay`.

- [ ] **Step 3: Implement**

```python
# salome_agent/replay.py
from salome_agent.build_graph import BuildGraph, _num


class ReplayFailed(RuntimeError):
    def __init__(self, node_id: str, error: str):
        super().__init__(f"rebuilding {node_id} failed: {error}")
        self.node_id = node_id
        self.error = error


class RollbackFailed(RuntimeError):
    def __init__(self, message: str, graph: BuildGraph):
        super().__init__(message)
        self.graph = graph


def _forget(client, names) -> None:
    names = list(dict.fromkeys(names))
    if not names:
        return
    r = client.forget(names)
    if not r.get("ok"):
        raise ReplayFailed("forget", r.get("error") or "forget failed")


def _vars(graph: BuildGraph, ids) -> list[str]:
    return [v for nid in ids if nid in graph.nodes for v in graph.nodes[nid].out_vars]


def replay(client, graph: BuildGraph, ids) -> None:
    for nid in sorted(ids, key=_num):
        node = graph.nodes[nid]
        r = client.call_tool(node.tool, node.tool_args, pin=node.out_vars)
        if not r.get("ok"):
            raise ReplayFailed(nid, r.get("error") or "unknown error")
        graph.set_records(nid, r.get("records") or [])


def apply_edit(client, graph: BuildGraph, node_id: str, param: str, value):
    new = graph.copy()
    new.edit_arg(node_id, param, value)
    ids = new.affected(node_id)
    _forget(client, _vars(graph, ids))
    try:
        replay(client, new, ids)
    except ReplayFailed:
        restored = graph.copy()
        try:
            _forget(client, _vars(new, ids) + _vars(graph, ids))
            replay(client, restored, ids)
        except ReplayFailed as e2:
            raise RollbackFailed(f"could not restore after a failed edit: {e2}", new) from e2
        raise
    return new, ids


def apply_delete(client, graph: BuildGraph, node_id: str):
    new = graph.copy()
    removed = new.delete(node_id)
    _forget(client, [v for n in removed for v in n.out_vars])
    return new, [n.id for n in removed]


def sync(client, current: BuildGraph, target: BuildGraph) -> BuildGraph:
    cur, tgt = current.nodes, target.nodes
    changed = [nid for nid, n in tgt.items()
               if nid not in cur or cur[nid].tool_args != n.tool_args]
    rebuild: set[str] = set()
    for nid in changed:
        rebuild |= set(target.affected(nid))
    drop = [nid for nid in cur if nid not in tgt] + [nid for nid in rebuild if nid in cur]
    _forget(client, _vars(current, drop))
    out = target.copy()
    replay(client, out, rebuild)
    return out


def rebuild_all(client, graph: BuildGraph, extra_names=()) -> BuildGraph:
    _forget(client, sorted(set(graph.all_out_vars()) | set(extra_names)))
    out = graph.copy()
    replay(client, out, list(out.nodes))
    return out
```

`test_rebuild_all_forgets_everything_and_replays_in_order` expects the forget list sorted, which is why `rebuild_all` sorts.

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/unit -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add salome_agent/replay.py tests/unit/test_replay.py
git commit -m "feat: selective replay with rollback, sync and full rebuild (#15)"
```

---

### Task 10: The amend specialist

**Files:**
- Create: `salome_agent/specialists/amend.py`
- Test: `tests/unit/test_amend.py` (create)

**Interfaces:**
- Consumes: Task 5 (`ToolSpecialist`, `extra_terminators`), Task 7 (`BuildGraph`, `GraphError`), Task 9 (`apply_edit`, `apply_delete`, `ReplayFailed`; `RollbackFailed` propagates).
- Produces:
  - `class AmendSession(graph, client, turn)` with `.graph`, `.events: list[dict]`, `.execute(client, name, args) -> dict`
  - `run_amend(text, graph, llm, client, turn, policy, *, history=(), selection=None) -> tuple[AmendSession, dict]`. The outcome dict is one of `{"summary": str, "question": str | None}`, `{"replan": str}`, `{"exhausted": True, "last_error": ...}`, `{"fatal": str}`.
  - Event dicts appended to `session.events`:
    - `{"event": "graph_edit", "op": "append"|"edit"|"delete", "node": id, "param": str|None, "old": any, "new": any}`
    - `{"event": "replay", "nodes": [ids]}`
  - LLM calls are tagged `module="amend"` (so `llm.module_extra_body["amend"]` applies).

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_amend.py
from unittest.mock import MagicMock

from salome_agent.build_graph import BuildGraph
from salome_agent.specialists.amend import AmendSession, render_transcript, run_amend


class FakeClient:
    def __init__(self):
        self.calls = []

    def list_tools(self):
        return {"tools": [{"name": n, "parameters": {}, "description": n, "category": c}
                          for n, c in [("make_box", "primitive"), ("measure", "query"),
                                       ("fuse", "boolean"), ("translate", "transform"),
                                       ("align", "transform"), ("cut", "boolean"),
                                       ("check_shape", "query"), ("get_subshapes", "query"),
                                       ("list_study_objects", "query")]]}

    def forget(self, names):
        self.calls.append(("forget", names))
        return {"ok": True, "removed": []}

    def call_tool(self, name, args, pin=None):
        self.calls.append((name, args, pin))
        if name == "measure":
            return {"ok": True, "measurements": {n: {"bbox": [0] * 6, "size": [1, 1, 1],
                                                     "centre": [0, 0, 0]} for n in args["names"]}}
        out = (pin or [f"{name}_new"])[0]
        return {"ok": True, "records": [{"fn": "F", "args": [], "out_vars": [out],
                                         "publish": [out]}]}


def _graph():
    g = BuildGraph()
    g.requests.append("a box")
    b = g.add_block(0, "stage 1/1: geom", "geom", ["box_1"])
    g.append(b, "make_box", {"dx": 1, "dy": 1, "dz": 1},
             [{"fn": "F", "args": [], "out_vars": ["box_1"], "publish": ["box_1"]}], 0)
    return g


def _resp(*calls):
    return {"tool_calls": [{"name": n, "args": a, "id": None} for n, a in calls],
            "content": None, "raw_message": None}


POLICY = {"tool_iteration_cap": 5, "tool_selection": "hints", "verify_cadence": "none"}


def test_edit_arg_replays_and_records_events():
    llm = MagicMock()
    llm.chat.side_effect = [_resp(("edit_arg", {"node": "n1", "param": "dz", "value": "3"})),
                            _resp(("finish_turn", {"summary": "taller"}))]
    client = FakeClient()
    session, outcome = run_amend("make it taller", _graph(), llm, client, 1, POLICY)
    assert outcome == {"summary": "taller", "question": None}
    assert session.graph.nodes["n1"].tool_args["dz"] == 3
    assert ("make_box", {"dx": 1, "dy": 1, "dz": 3}, ["box_1"]) in client.calls
    assert session.events[0] == {"event": "graph_edit", "op": "edit", "node": "n1",
                                 "param": "dz", "old": 1, "new": 3}
    assert session.events[1] == {"event": "replay", "nodes": ["n1"]}


def test_prompt_carries_graph_snapshot_turn_text_and_amend_tools():
    llm = MagicMock()
    llm.chat.side_effect = [_resp(("finish_turn", {"summary": "nothing"}))]
    run_amend("make it red", _graph(), llm, FakeClient(), 1, POLICY,
              history=[{"event": "turn", "turn": 0, "text": "a box", "status": "ok",
                        "summary": "built"}])
    kw = llm.chat.call_args.kwargs
    user = kw["messages"][1]["content"]
    assert "n1  box_1 = make_box(dx=1, dy=1, dz=1)" in user
    assert "box_1 — bbox" in user
    assert "turn 1 (ok): a box -> built" in user
    assert "make it red" in user
    names = {t["name"] for t in kw["tools"]}
    assert {"edit_arg", "delete_node", "finish_turn", "replan", "load_tools"} <= names
    assert kw["module"] == "amend"


def test_selection_block_renders_only_when_given():
    llm = MagicMock()
    llm.chat.side_effect = [_resp(("finish_turn", {"summary": "x"}))] * 2
    run_amend("t", _graph(), llm, FakeClient(), 1, POLICY)
    assert "currently has these selected" not in llm.chat.call_args.kwargs["messages"][1]["content"]
    run_amend("t", _graph(), llm, FakeClient(), 1, POLICY,
              selection=[{"object": "box_1", "subshape": {"type": "FACE", "index": 3}}])
    assert "currently has these selected" in llm.chat.call_args.kwargs["messages"][1]["content"]


def test_append_goes_into_one_amend_block_for_the_turn():
    llm = MagicMock()
    llm.chat.side_effect = [_resp(("make_box", {"dx": 2, "dy": 2, "dz": 2})),
                            _resp(("make_box", {"dx": 3, "dy": 3, "dz": 3})),
                            _resp(("finish_turn", {"summary": "added"}))]
    session, _ = run_amend("add two", _graph(), llm, FakeClient(), 1, POLICY)
    labels = [b.label for b in session.graph.blocks]
    assert labels == ["stage 1/1: geom", "turn 2: amend"]
    assert [n.block for n in session.graph.nodes.values()] == ["b1", "b2", "b2"]


def test_bad_node_is_a_tool_error_with_the_remedy():
    session = AmendSession(_graph(), FakeClient(), 1)
    r = session.execute(None, "edit_arg", {"node": "n7", "param": "dz", "value": "1"})
    assert r["ok"] is False and "current nodes are n1" in r["error"]


def test_replan_and_question_outcomes():
    llm = MagicMock()
    llm.chat.side_effect = [_resp(("replan", {"reason": "whole new wing"}))]
    _, outcome = run_amend("add a wing", _graph(), llm, FakeClient(), 1, POLICY)
    assert outcome == {"replan": "whole new wing"}
    llm.chat.side_effect = [_resp(("finish_turn", {"summary": "", "question": "which box?"}))]
    _, outcome = run_amend("move the box", _graph(), llm, FakeClient(), 1, POLICY)
    assert outcome["question"] == "which box?"


def test_render_transcript_keeps_the_last_ten_turns():
    hist = [{"event": "turn", "turn": i, "text": f"t{i}", "status": "ok", "summary": "s"}
            for i in range(15)] + [{"event": "retry_hint", "hint": "x"}]
    text = render_transcript(hist)
    assert "t4" not in text and "t5" in text and "t14" in text
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/unit/test_amend.py -q`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

```python
# salome_agent/specialists/amend.py
import json

from salome_agent.build_graph import GraphError
from salome_agent.modules import ModuleSpec, Terminator
from salome_agent.replay import ReplayFailed, apply_delete, apply_edit
from salome_agent.specialists.geom import _compact_result, _geom_tools_for
from salome_agent.specialists.tool_selection import CORE_TOOLS
from salome_agent.specialists.tool_specialist import ToolSpecialist

AMEND_SYSTEM_PROMPT = """You amend a CAD model the user is building with you in
a live SALOME study. The model is a build graph: each node is one tool call,
shown as `id  name = tool(args)  <- the nodes it uses`. Study names are the
handles; node ids are what edit_arg and delete_node take.

To change something that exists, call edit_arg on the node that made it —
"make the roof taller" is an edit of the roof node's height argument, not a
new shape. The node and everything built on it is rebuilt immediately with
the same names, and you get the rebuilt shapes' measurements back. edit_arg
only changes literal values. To change which shapes a node uses, delete_node
it and build the replacement with the geom tools.

To add something, call the geom tools as usual; they are added to the graph.
Measure before you place anything relative to an existing shape. Lengths are
millimetres; angles are degrees.

Do only what the user asked. If the request could mean two different nodes,
change nothing and call finish_turn with your question in `question`. If the
request is too big to express as edits plus a few new shapes (a whole new
assembly, a different object), call replan before changing anything.

Each turn: state your reasoning in at most 2-3 sentences, then call one or
more tools. When the change is done, call finish_turn(summary) with one
sentence saying what changed.

You are given only a subset of the geom tools. If you need one that is not in
your list, call load_tools with its name or category."""

_LITERAL_HELP = ('the new value as a JSON literal, e.g. 120, 7.5, [0, 0, 1] or "OZ"')

EDIT_ARG_DEF = {
    "name": "edit_arg", "category": "amend",
    "description": ("Change one literal argument of an existing node; the node and "
                    "everything built on it are rebuilt with the same names."),
    "parameters": {"type": "object", "properties": {
        "node": {"type": "string", "description": "node id, e.g. n2"},
        "param": {"type": "string", "description": "argument name as shown in the build graph"},
        "value": {"type": "string", "description": _LITERAL_HELP}},
        "required": ["node", "param", "value"]}}

DELETE_NODE_DEF = {
    "name": "delete_node", "category": "amend",
    "description": "Remove a node and every node built on it, from the model and the study.",
    "parameters": {"type": "object", "properties": {
        "node": {"type": "string", "description": "node id, e.g. n4"}},
        "required": ["node"]}}

FINISH_TURN = Terminator(
    name="finish_turn",
    schema={"type": "object", "properties": {
        "summary": {"type": "string", "description": "one sentence: what changed"},
        "question": {"type": "string",
                     "description": "only if you changed nothing because the request was "
                                    "ambiguous: your question to the user"}},
        "required": ["summary"]},
    description="End this turn.",
    handle=lambda args, call_log: {"summary": args.get("summary") or "",
                                   "question": args.get("question") or None})

REPLAN = Terminator(
    name="replan",
    schema={"type": "object", "properties": {"reason": {"type": "string"}},
            "required": ["reason"]},
    description=("Hand this request to the planner because it is too big for edits. "
                 "Call it before changing anything."),
    handle=lambda args, call_log: {"replan": args.get("reason") or ""})


def _literal(value):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value


class AmendSession:
    def __init__(self, graph, client, turn: int):
        self.graph = graph
        self.client = client
        self.turn = turn
        self.events: list[dict] = []
        self._block = None

    def _amend_block(self) -> str:
        if self._block is None:
            self._block = self.graph.add_block(self.turn, f"turn {self.turn + 1}: amend",
                                               "geom", [])
        return self._block

    def _measure(self, ids) -> dict | None:
        names = [p for nid in ids if nid in self.graph.nodes
                 for p in self.graph.nodes[nid].published]
        if not names:
            return None
        m = self.client.call_tool("measure", {"names": names})
        return m.get("measurements") if m.get("ok") else None

    def execute(self, client, name: str, args: dict) -> dict:
        try:
            if name == "edit_arg":
                node, param = args["node"], args["param"]
                value = _literal(args["value"])
                old = self.graph.nodes[node].tool_args.get(param) if node in self.graph.nodes else None
                self.graph, ids = apply_edit(self.client, self.graph, node, param, value)
                self.events += [{"event": "graph_edit", "op": "edit", "node": node,
                                 "param": param, "old": old, "new": value},
                                {"event": "replay", "nodes": ids}]
                return {"ok": True, "edited": node, "rebuilt": ids,
                        "measurements": self._measure(ids)}
            if name == "delete_node":
                self.graph, ids = apply_delete(self.client, self.graph, args["node"])
                self.events.append({"event": "graph_edit", "op": "delete", "node": args["node"],
                                    "param": None, "old": ids, "new": None})
                return {"ok": True, "deleted": ids}
        except GraphError as e:
            return {"ok": False, "error": str(e)}
        except ReplayFailed as e:
            return {"ok": False, "error": f"{e} — the model was restored to before this edit"}
        r = self.client.call_tool(name, args)
        if r.get("ok") and r.get("records"):
            nid = self.graph.append(self._amend_block(), name, args, r["records"], self.turn)
            self.events.append({"event": "graph_edit", "op": "append", "node": nid,
                                "param": None, "old": None, "new": name})
            r = {**r, "node": nid}
        return r


def _render(result: dict) -> str:
    out = _compact_result(result)
    for k in ("node", "edited", "rebuilt", "deleted", "measurements"):
        if result.get(k) is not None:
            out[k] = result[k]
    return str(out)


def render_transcript(history) -> str:
    turns = [e for e in history if e.get("event") == "turn"][-10:]
    return "\n".join(f"turn {e['turn'] + 1} ({e.get('status')}): {e.get('text')} -> "
                     f"{e.get('summary') or ''}" for e in turns)


def _snapshot(client, graph) -> str | None:
    names = graph.result_names()
    if not names or client is None:
        return None
    m = client.call_tool("measure", {"names": names})
    if not m.get("ok"):
        return None
    lines = [f"  {n} — bbox {d['bbox']}, size {d['size']}, centre {d['centre']}"
             for n, d in (m.get("measurements") or {}).items() if d]
    return "Current study, measured now:\n" + "\n".join(lines) if lines else None


def _spec(session: AmendSession) -> ModuleSpec:
    def tools_for(client):
        defs = _geom_tools_for(client)
        defs["edit_arg"] = EDIT_ARG_DEF
        defs["delete_node"] = DELETE_NODE_DEF
        return defs

    return ModuleSpec(
        name="amend",
        prompt=lambda cadence: AMEND_SYSTEM_PROMPT,
        terminator=FINISH_TURN,
        extra_terminators=(REPLAN,),
        execute=session.execute,
        render_result=_render,
        tools_for=tools_for,
        no_tool_nudge="You did not call a tool. Call a tool now — finish_turn when done.",
        on_exhausted=lambda call_log, last_error: {"exhausted": True, "last_error": last_error},
        core_tools=frozenset(CORE_TOOLS),
        categories=("primitive", "sketch", "swept", "boolean",
                    "transform", "feature", "display", "query"),
        risky_categories=frozenset({"boolean", "transform"}),
        measure_tools=frozenset({"measure", "check_shape"}),
    )


def run_amend(text: str, graph, llm, client, turn: int, policy: dict, *,
              history=(), selection=None):
    session = AmendSession(graph, client, turn)
    blocks = ["Build graph:\n" + graph.render()]
    snap = _snapshot(client, graph)
    if snap:
        blocks.append(snap)
    transcript = render_transcript(history)
    if transcript:
        blocks.append("Conversation so far:\n" + transcript)
    if selection:
        blocks.append("The user currently has these selected in the viewer:\n"
                      + "\n".join(json.dumps(s) for s in selection))
    blocks.append(f"The user says: {text}")
    stage_plan = {"module": "amend", "intent": text, "tools": ["edit_arg", "delete_node"]}
    outcome = ToolSpecialist(_spec(session)).run(
        stage_plan, llm, client, policy["tool_iteration_cap"], blocks=tuple(blocks),
        tool_selection=policy.get("tool_selection", "hints"),
        verify_cadence=policy.get("verify_cadence", "final"))
    return session, outcome
```

`session.graph` is reassigned by `apply_edit` / `apply_delete`, so callers must read `session.graph` after the run, never the graph they passed in.

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/unit -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add salome_agent/specialists/amend.py tests/unit/test_amend.py
git commit -m "feat: amend specialist with edit_arg, delete_node, finish_turn and replan (#15)"
```

---

### Task 11: Wire amend into the LangGraph

**Files:**
- Modify: `salome_agent/state.py` (`RunState` chat fields)
- Modify: `salome_agent/graph.py` (`route_start`, `_amend_node`, `_plan_node` replan context, `_run_stage_node` graph-aware upstream, `_report_node`, `build_graph`)
- Modify: `salome_agent/nodes/report.py` (amend section)
- Test: `tests/unit/test_graph_amend.py` (create)

**Interfaces:**
- Consumes: Task 8 (`build_graph` in state, `_assemble_node`), Task 10 (`run_amend`), Task 9 (`RollbackFailed`).
- Produces `RunState` keys (all optional; absent in one-shot runs):
  - `build_graph: Optional[dict]`
  - `turn_idx: int`
  - `selection: Optional[list]`
  - `turn_outcome: Optional[dict]`
  - `replan_context: Optional[str]`
  - `degraded: Optional[bool]`
- Also:
  - `route_start(state) -> "plan" | "amend"` (`amend` iff `turn_idx > 0`).
  - `_amend_node` → `Command(goto="assemble" | "plan")`, update carries `build_graph`, `turn_outcome`, `history` (events), and on replan `replan_context`.
  - Files written in the run dir: `build_graph.json` every run; `report.md` appended with a `# Turn N` header when `turn_idx > 0`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_graph_amend.py
import json
import os
import uuid

from salome_agent.build_graph import BuildGraph
from salome_agent.graph import build_graph, route_start

POLICY = {"tool_iteration_cap": 5, "static_retries": 1, "exec_retries": 1,
          "exec_timeout_s": 120, "tool_selection": "hints", "verify_cadence": "none",
          "plan_detail": "balanced", "on_stage_failure": "abort"}


def _state(tmp_path, graph, text, turn=1):
    return {"run_id": "r1", "user_request": text,
            "config": {"batch": True, "work_dir": str(tmp_path), "policy": POLICY},
            "plan": [], "plan_approved": True, "stage_idx": 0, "plan_timing": None,
            "session": {"mode": "none", "pid": None, "salome_port": None, "name_prefix": ""},
            "study_ns": "salome_agent_run", "stages": {}, "last_error": None,
            "history": [], "final_script": None, "report": None, "fatal": None,
            "aborted": None, "build_graph": graph.to_dict(), "turn_idx": turn,
            "selection": None, "turn_outcome": None, "replan_context": None,
            "degraded": None}


def _graph():
    g = BuildGraph()
    g.requests.append("a box")
    b = g.add_block(0, "stage 1/1: geom", "geom", ["box_1"])
    g.append(b, "make_box", {"dx": 1, "dy": 1, "dz": 1},
             [{"fn": "MakeBoxDXDYDZ", "args": [{"kind": "literal", "value": 1.0}] * 3,
               "out_vars": ["box_1"], "publish": ["box_1"]}], 0)
    return g


def test_route_start():
    assert route_start({"turn_idx": 0}) == "plan"
    assert route_start({}) == "plan"
    assert route_start({"turn_idx": 2}) == "amend"


def test_amend_turn_rewrites_result_and_graph_json(tmp_path, monkeypatch):
    from salome_agent.specialists import amend as amend_mod

    def fake_run_amend(text, graph, llm, client, turn, policy, *, history=(), selection=None):
        session = amend_mod.AmendSession(graph, client, turn)
        session.graph = graph.copy()
        session.graph.nodes["n1"].tool_args["dz"] = 3
        session.graph.nodes["n1"].records[0]["args"][2] = {"kind": "literal", "value": 3.0}
        session.events = [{"event": "replay", "nodes": ["n1"]}]
        return session, {"summary": "taller", "question": None}

    monkeypatch.setattr(amend_mod, "run_amend", fake_run_amend)
    app = build_graph(batch=True)
    cfg = {"configurable": {"thread_id": uuid.uuid4().hex, "llm_client": None,
                            "executor_client": None}}
    out = app.invoke(_state(tmp_path, _graph(), "make it taller"), config=cfg)
    assert out["turn_outcome"] == {"summary": "taller", "question": None}
    assert "MakeBoxDXDYDZ(1.0, 1.0, 3.0)" in out["final_script"]
    assert "# turn 2: make it taller" in out["final_script"]
    run_dir = tmp_path / "r1"
    assert json.loads((run_dir / "build_graph.json").read_text())["nodes"][0]["tool_args"]["dz"] == 3
    assert "# Turn 2" in (run_dir / "report.md").read_text()


def test_replan_goes_to_the_planner_with_context(tmp_path, monkeypatch):
    from salome_agent.specialists import amend as amend_mod
    from salome_agent.nodes import planner as planner_mod

    monkeypatch.setattr(amend_mod, "run_amend",
                        lambda *a, **k: (amend_mod.AmendSession(a[1], a[3], a[4]),
                                         {"replan": "new wing"}))
    seen = {}

    def fake_plan(request, llm, detail="balanced"):
        seen["request"] = request
        raise RuntimeError("stop after planning")   # _plan_node turns this into a clean fatal

    monkeypatch.setattr(planner_mod, "plan", fake_plan)
    app = build_graph(batch=True)
    cfg = {"configurable": {"thread_id": uuid.uuid4().hex, "llm_client": None,
                            "executor_client": None}}
    app.invoke(_state(tmp_path, _graph(), "add a wing"), config=cfg)
    assert seen["request"].startswith("add a wing")
    assert "box_1" in seen["request"]
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/unit/test_graph_amend.py -q`
Expected: FAIL — `ImportError: cannot import name 'route_start'`.

- [ ] **Step 3: Implement**

`salome_agent/state.py`, append to `RunState`:

```python
    # agentic chat (task 194) — absent in one-shot runs
    build_graph: Optional[dict]
    turn_idx: int
    selection: Optional[list]
    turn_outcome: Optional[dict]
    replan_context: Optional[str]
    degraded: Optional[bool]
```

`salome_agent/graph.py`:

```python
def route_start(state: dict) -> str:
    return "amend" if (state.get("turn_idx") or 0) > 0 else "plan"


def _graph_upstream(state: dict) -> list[str]:
    from salome_agent.build_graph import BuildGraph

    if not state.get("build_graph"):
        return []
    return BuildGraph.from_dict(state["build_graph"]).result_names()


def _amend_node(state: dict, config: "RunnableConfig"):
    from langgraph.types import Command
    from salome_agent.build_graph import BuildGraph
    from salome_agent.replay import RollbackFailed
    from salome_agent.specialists import amend

    llm = config["configurable"]["llm_client"]
    client = config["configurable"].get("executor_client")
    graph = BuildGraph.from_dict(state.get("build_graph"))
    log.info("amend turn %d: %s", (state.get("turn_idx") or 0) + 1, state["user_request"])
    try:
        session, outcome = amend.run_amend(
            state["user_request"], graph, llm, client, state.get("turn_idx") or 0,
            state["config"]["policy"], history=state.get("history") or [],
            selection=state.get("selection"))
    except RollbackFailed as e:
        log.error("  %s", e)
        return Command(update={"fatal": f"rollback failed: {e}", "degraded": True,
                               "build_graph": e.graph.to_dict()}, goto="assemble")
    except Exception as e:  # noqa: BLE001
        msg = _llm_endpoint_fatal(e)
        if msg is None:
            raise
        log.error("  %s", msg)
        return Command(update={"fatal": msg}, goto="assemble")
    update = {"build_graph": session.graph.to_dict(), "turn_outcome": outcome,
              "history": session.events}
    if "fatal" in outcome:
        update["fatal"] = outcome["fatal"]
        return Command(update=update, goto="assemble")
    if "replan" in outcome:
        names = ", ".join(session.graph.result_names()) or "(nothing yet)"
        update["replan_context"] = (
            f"The study already contains: {names}. Build on those objects by name; "
            f"do not rebuild them. ({outcome['replan']})")
        log.info("  amend -> replan: %s", outcome["replan"])
        return Command(update=update, goto="plan")
    return Command(update=update, goto="assemble")
```

In `_plan_node`, build the request as

```python
    request = state["user_request"]
    if state.get("replan_context"):
        request = f"{request}\n\n{state['replan_context']}"
```

and call `plan(request, llm, detail)`. The existing function-local `from salome_agent.nodes.planner import blank_stages, plan` already runs at call time, so the test's monkeypatch of `planner.plan` reaches it. Leave that import as it is.

In `_run_stage_node`: `upstream = _graph_upstream(state) + select_upstream(stage_plan, state["stages"], idx)`.

`_report_node`: after writing `result.py`:

```python
    import json

    turn = state.get("turn_idx") or 0
    if state.get("build_graph") is not None:
        with open(os.path.join(run_dir, "build_graph.json"), "w") as f:
            json.dump(state["build_graph"], f, indent=1)
    mode = "a" if turn > 0 else "w"
    with open(os.path.join(run_dir, "report.md"), mode) as f:
        f.write(f"\n\n# Turn {turn + 1}\n\n{report_text}" if turn > 0 else report_text)
```

(This replaces the existing `report.md` write.)

`build_graph()`: `g.add_node("amend", _amend_node)` and replace `g.add_edge(START, "plan")` with
`g.add_conditional_edges(START, route_start, {"plan": "plan", "amend": "amend"})`.

`salome_agent/nodes/report.py` `write_report`: before `lines.append("## Stages")`:

```python
    outcome = state.get("turn_outcome") or {}
    if outcome.get("summary") or outcome.get("question"):
        lines += ["## Amend", "", outcome.get("summary") or "",
                  *([f"Question for you: {outcome['question']}"] if outcome.get("question") else []),
                  ""]
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/unit -q`
Expected: all pass. The goldens still pass because their state has no `turn_idx`, so they route to `plan`.

- [ ] **Step 5: Commit**

```bash
git add salome_agent/state.py salome_agent/graph.py salome_agent/nodes/report.py tests/unit/test_graph_amend.py
git commit -m "feat: follow-up turns enter the graph at an amend node (#15)"
```

---

### Task 12: `Conversation` — the engine that outlives a turn

**Files:**
- Modify: `salome_agent/runner.py` (`initial_state`, `drive`, `EventMapper` amend events; `run_request` rebuilt on them)
- Create: `salome_agent/conversation.py`
- Test: `tests/unit/test_conversation.py` (create), `tests/unit/test_runner.py` (append)

**Interfaces:**
- Consumes: Tasks 8–11.
- Produces:
  - `runner.initial_state(request, config, client, run_id, batch) -> dict`
  - `runner.drive(app, inp, cfg, mapper, llm, on_event, on_gate, cancel) -> dict` (final state; sets `aborted` and emits `aborted` if cancelled)
  - `EventMapper` emits `graph_edit` / `replay` from `amend` node history events
  - `conversation.TurnResult(status: str, message: str, final_state: dict)` with `status ∈ {"ok", "failed", "needs_input", "degraded", "ended"}`
  - `conversation.Conversation(config, client, run_id, *, batch=False, verbose=0, request_timeout=None, llm=None)` with `.graph`, `.turn_idx`, `.history`, `.run_dir`, `.degraded`, `.ended`, and methods:
    - `.turn(text, *, selection=None, on_event, on_gate, cancel=None) -> TurnResult`
    - `.undo() -> TurnResult`
    - `.rebuild() -> TurnResult`
  - Events emitted by `Conversation`:
    - `turn_start {turn, text}`
    - `turn_done {turn, status, message}`

**Turn rules** (from the spec's error table):

| final state | status | graph |
|---|---|---|
| no `fatal` / `aborted`, outcome has `question` | `needs_input` | committed |
| no `fatal` / `aborted`, outcome `exhausted` | `failed` | rolled back |
| no `fatal` / `aborted`, otherwise | `ok` | committed |
| `degraded` | `degraded` | kept at the pre-turn graph; stray names remembered for `/rebuild` |
| `fatal` mentions `"SALOME agent server"`, or rollback raises `SessionDied` / `SessionPoisoned` | `ended` | frozen |
| any other `fatal` / `aborted` | `failed` | rolled back via `sync(current=final graph, target=pre-turn graph)` |

On a rollback that raises `ReplayFailed`, the status becomes `degraded`. Only a committed turn increments `turn_idx` and pushes an undo snapshot `(graph, turn_idx, len(history))`. Every turn, whatever its status, appends `{"event": "turn", "turn", "text", "status", "summary"}` to `history`. After any rollback, undo or rebuild, `result.py` and `build_graph.json` are rewritten from the graph.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_runner.py`:

```python
def test_amend_history_events_become_ui_events():
    m = EventMapper("r1", "/w/r1")
    evs = m.events("amend", {"history": [
        {"event": "graph_edit", "op": "edit", "node": "n1", "param": "dz", "old": 1, "new": 2},
        {"event": "replay", "nodes": ["n1", "n3"]},
        {"event": "retry_hint", "hint": "x"}]})
    assert [e["kind"] for e in evs] == ["graph_edit", "replay"]
    assert evs[1]["data"] == {"nodes": ["n1", "n3"]}
```

Create `tests/unit/test_conversation.py`:

```python
from unittest.mock import MagicMock

import pytest

from salome_agent.build_graph import BuildGraph
from salome_agent import conversation as conv_mod
from salome_agent.conversation import Conversation

CONFIG = {"work_dir": None, "llm": {}, "policy": {
    "tool_iteration_cap": 5, "static_retries": 1, "exec_retries": 1, "exec_timeout_s": 120,
    "tool_selection": "hints", "verify_cadence": "none", "plan_detail": "balanced",
    "on_stage_failure": "abort"}}


def _graph_with(n_boxes):
    g = BuildGraph()
    g.requests.append("boxes")
    b = g.add_block(0, "stage 1/1: geom", "geom", [])
    for i in range(1, n_boxes + 1):
        g.append(b, "make_box", {"dz": i}, [{"fn": "F", "args": [], "out_vars": [f"box_{i}"],
                                            "publish": [f"box_{i}"]}], 0)
    return g


@pytest.fixture
def conv(tmp_path, monkeypatch):
    cfg = {**CONFIG, "work_dir": str(tmp_path)}
    c = Conversation(cfg, client=MagicMock(), run_id="r1", batch=True, llm=MagicMock(usage={}))
    return c


def _fake_drive(monkeypatch, final_update):
    def fake(app, inp, cfg, mapper, llm, on_event, on_gate, cancel):
        return {**inp, **final_update}
    monkeypatch.setattr(conv_mod, "drive", fake)


def test_ok_turn_commits_graph_and_snapshot(conv, monkeypatch):
    g = _graph_with(1)
    _fake_drive(monkeypatch, {"build_graph": g.to_dict(), "final_script": "x"})
    events = []
    r = conv.turn("one box", on_event=events.append, on_gate=None)
    assert r.status == "ok"
    assert conv.turn_idx == 1 and list(conv.graph.nodes) == ["n1"]
    assert [e["kind"] for e in events] == ["turn_start", "turn_done"]
    assert conv.history[-1]["event"] == "turn" and conv.history[-1]["status"] == "ok"


def test_failed_turn_rolls_back_via_sync(conv, monkeypatch):
    conv.graph = _graph_with(1); conv.turn_idx = 1
    bad = _graph_with(2)
    _fake_drive(monkeypatch, {"build_graph": bad.to_dict(), "fatal": "LLM endpoint unreachable"})
    seen = {}
    monkeypatch.setattr(conv_mod, "sync",
                        lambda client, current, target: seen.setdefault("t", target).copy())
    r = conv.turn("more", on_event=lambda e: None, on_gate=None)
    assert r.status == "failed"
    assert list(seen["t"].nodes) == ["n1"]
    assert conv.turn_idx == 1 and list(conv.graph.nodes) == ["n1"]


def test_needs_input_commits_without_failing(conv, monkeypatch):
    conv.graph = _graph_with(1); conv.turn_idx = 1
    _fake_drive(monkeypatch, {"build_graph": conv.graph.to_dict(),
                              "turn_outcome": {"summary": "", "question": "which one?"}})
    r = conv.turn("move it", on_event=lambda e: None, on_gate=None)
    assert r.status == "needs_input" and "which one?" in r.message


def test_session_loss_ends_the_conversation(conv, monkeypatch):
    _fake_drive(monkeypatch, {"fatal": "SALOME agent server unavailable: socket closed"})
    r = conv.turn("x", on_event=lambda e: None, on_gate=None)
    assert r.status == "ended" and conv.ended
    assert conv.turn("y", on_event=lambda e: None, on_gate=None).status == "ended"


def test_degraded_blocks_turns_until_rebuild(conv, monkeypatch):
    conv.graph = _graph_with(1); conv.turn_idx = 1
    _fake_drive(monkeypatch, {"build_graph": _graph_with(2).to_dict(), "degraded": True,
                              "fatal": "rollback failed: x"})
    assert conv.turn("x", on_event=lambda e: None, on_gate=None).status == "degraded"
    assert conv.turn("y", on_event=lambda e: None, on_gate=None).status == "degraded"
    calls = {}

    def fake_rebuild_all(client, graph, extra_names=()):
        calls["extra"] = list(extra_names)
        return graph.copy()

    monkeypatch.setattr(conv_mod, "rebuild_all", fake_rebuild_all)
    assert conv.rebuild().status == "ok"
    assert "box_2" in calls["extra"] and not conv.degraded


def test_undo_restores_the_previous_snapshot(conv, monkeypatch):
    _fake_drive(monkeypatch, {"build_graph": _graph_with(1).to_dict()})
    conv.turn("one", on_event=lambda e: None, on_gate=None)
    _fake_drive(monkeypatch, {"build_graph": _graph_with(2).to_dict()})
    conv.turn("two", on_event=lambda e: None, on_gate=None)
    monkeypatch.setattr(conv_mod, "sync", lambda client, current, target: target.copy())
    assert conv.undo().status == "ok"
    assert list(conv.graph.nodes) == ["n1"] and conv.turn_idx == 1
    assert conv.undo().status == "ok" and conv.turn_idx == 0
    assert conv.undo().status == "failed"


def test_turn_input_carries_graph_turn_and_selection(conv, monkeypatch):
    seen = {}

    def fake(app, inp, cfg, mapper, llm, on_event, on_gate, cancel):
        seen.update(inp)
        return {**inp, "build_graph": inp["build_graph"]}

    monkeypatch.setattr(conv_mod, "drive", fake)
    conv.graph = _graph_with(1); conv.turn_idx = 1
    sel = [{"object": "box_1", "subshape": None}]
    conv.turn("x", selection=sel, on_event=lambda e: None, on_gate=None)
    assert seen["turn_idx"] == 1 and seen["selection"] == sel
    assert seen["build_graph"]["nodes"][0]["id"] == "n1"
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/unit/test_conversation.py tests/unit/test_runner.py -q`
Expected: FAIL (`ModuleNotFoundError: salome_agent.conversation`, new mapper test fails).

- [ ] **Step 3: Implement**

`salome_agent/runner.py`:

- In `EventMapper.events`, right after `out = []`:

```python
        if node == "amend":
            for e in update.get("history") or []:
                if e.get("event") in ("graph_edit", "replay"):
                    out.append(self.event(e["event"],
                                          **{k: v for k, v in e.items() if k != "event"}))
```

- Extract the input dict and the stream loop:

```python
def initial_state(request: str, config: dict, client, run_id: str, batch: bool) -> dict:
    return {
        "user_request": request, "config": {**config, "batch": batch},
        "session": _session_handle(client), "plan": [], "plan_approved": False,
        "stage_idx": 0, "stages": {}, "last_error": None, "history": [],
        "study_ns": "salome_agent_run", "final_script": None, "report": None,
        "fatal": None, "aborted": False, "run_id": run_id,
        "build_graph": None, "turn_idx": 0, "selection": None, "turn_outcome": None,
        "replan_context": None, "degraded": None,
    }


def drive(app, inp, cfg, mapper, llm, on_event, on_gate, cancel) -> dict:
    from langgraph.types import Command

    def cancelled() -> bool:
        return cancel is not None and cancel.is_set()

    stopped = False
    while True:
        pending = None
        for chunk in app.stream(inp, config=cfg, stream_mode="updates"):
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

    final = dict(app.get_state(cfg).values)
    if stopped and not final.get("aborted"):
        final["aborted"] = True
        on_event(mapper.event("aborted"))
    return final
```

- `run_request` keeps its signature and becomes: build `llm`, `cfg`, `graph`, `mapper` as today; `inp = initial_state(request, config, client, run_id, batch)`; `final = drive(graph, inp, cfg, mapper, llm, on_event, on_gate, cancel)`; the same `exit_code` and `RunResult`. The existing `test_runner.py` tests must pass unchanged.

`salome_agent/conversation.py`:

```python
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
```

`undo()` on an ended or degraded conversation simply reuses `turn()`'s early returns. That is intentional: one message, one place.

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/unit -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add salome_agent/runner.py salome_agent/conversation.py tests/unit/test_runner.py tests/unit/test_conversation.py
git commit -m "feat: Conversation engine with rollback, undo and rebuild (#15)"
```

---

### Task 13: CLI `--chat` REPL

**Files:**
- Create: `salome_agent/chat.py`
- Modify: `salome_agent/cli.py` (`--chat` flag, `main`)
- Modify: `README.md` (chat mode section)
- Test: `tests/unit/test_chat.py` (create), `tests/unit/test_cli.py` (append)

**Interfaces:**
- Consumes: Task 12 `Conversation`, `TurnResult`; Task 3 `server_conn.connect(..., hold=True)`.
- Produces:
  - `chat.repl(conv, first_request, *, read=input, stream=sys.stderr, gate=terminal_gate, run_turn=None) -> int`
  - `chat.run_turn(conv, text, gate, stream) -> TurnResult`, which installs a SIGINT handler for the turn: first Ctrl-C sets `cancel`, second raises `KeyboardInterrupt`.
  - `repl` exit codes: `0` on `/quit` or EOF, `1` if the conversation ended.

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_chat.py
import io
from types import SimpleNamespace

from salome_agent.chat import repl
from salome_agent.conversation import TurnResult


class FakeConv:
    def __init__(self):
        self.turns = []
        self.run_dir = "/w/r1"
        self.graph = SimpleNamespace(render=lambda: "n1  box_1 = make_box(dx=1)")
        self.ended = False
        self.undone = 0
        self.rebuilt = 0

    def undo(self):
        self.undone += 1
        return TurnResult("ok", "undone", {})

    def rebuild(self):
        self.rebuilt += 1
        return TurnResult("ok", "rebuilt 1 node(s)", {})


def _run(lines, conv=None):
    conv = conv or FakeConv()
    feed = iter(lines)

    def read(prompt):
        try:
            return next(feed)
        except StopIteration:
            raise EOFError

    def run_turn(c, text, gate, stream):
        c.turns.append(text)
        return TurnResult("ok", f"did {text}", {})

    out = io.StringIO()
    code = repl(conv, "a house", read=read, stream=out, run_turn=run_turn)
    return conv, out.getvalue(), code


def test_first_request_then_bare_text_turns():
    conv, out, code = _run(["make the roof taller", "/quit"])
    assert conv.turns == ["a house", "make the roof taller"]
    assert code == 0 and "did make the roof taller" in out


def test_commands():
    conv, out, _ = _run(["/show", "/undo", "/rebuild", "/help", "/nope", ""])
    assert "n1  box_1 = make_box(dx=1)" in out
    assert conv.undone == 1 and conv.rebuilt == 1
    assert "/undo" in out                      # help text
    assert "unknown command /nope" in out
    assert conv.turns == ["a house"]           # blank line is not a turn


def test_eof_exits_cleanly():
    _, _, code = _run([])
    assert code == 0


def test_result_prints_path(tmp_path):
    conv = FakeConv()
    conv.run_dir = str(tmp_path)
    (tmp_path / "result.py").write_text("print('hi')\n")
    _, out, _ = _run(["/result"], conv)
    assert str(tmp_path / "result.py") in out and "print('hi')" in out


def test_ended_conversation_exits_1():
    conv = FakeConv()

    def run_turn(c, text, gate, stream):
        c.ended = True
        return TurnResult("ended", "server gone", {})

    code = repl(conv, "a house", read=lambda p: "/quit", stream=io.StringIO(),
                run_turn=run_turn)
    assert code == 1
```

Append to `tests/unit/test_cli.py`:

```python
def test_chat_with_batch_is_an_error(capsys):
    from salome_agent.cli import main
    assert main(["a box", "--chat", "--batch"]) == 2


def test_chat_with_no_session_is_an_error(capsys):
    from salome_agent.cli import main
    assert main(["a box", "--chat", "--no-session"]) == 2
```
- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/unit/test_chat.py tests/unit/test_cli.py -q`
Expected: FAIL.

- [ ] **Step 3: Implement**

```python
# salome_agent/chat.py
import os
import signal
import sys
import threading

from salome_agent.interactive import terminal_gate

_HELP = """commands:
  /undo      revert the last turn
  /show      print the build graph
  /result    print result.py
  /rebuild   rebuild the whole study from the model (after a failed rollback)
  /quit      leave (Ctrl-D too)
anything else is a request to the agent. Ctrl-C cancels the running turn."""


def _print_event(stream):
    def on_event(ev):
        d = ev["data"]
        if ev["kind"] == "graph_edit":
            if d["op"] == "edit":
                stream.write(f"  edit {d['node']}.{d['param']}: {d['old']} -> {d['new']}\n")
            elif d["op"] == "delete":
                stream.write(f"  deleted {', '.join(d['old'])}\n")
            else:
                stream.write(f"  added {d['node']} ({d['new']})\n")
        elif ev["kind"] == "replay":
            stream.write(f"  rebuilding {len(d['nodes'])} object(s): {', '.join(d['nodes'])}\n")
        stream.flush()
    return on_event


def run_turn(conv, text, gate, stream):
    cancel = threading.Event()

    def on_sigint(signum, frame):
        if cancel.is_set():
            raise KeyboardInterrupt
        cancel.set()
        stream.write("\n(cancelling this turn — Ctrl-C again to quit)\n")

    prev = signal.signal(signal.SIGINT, on_sigint)
    try:
        return conv.turn(text, on_event=_print_event(stream), on_gate=gate, cancel=cancel)
    finally:
        signal.signal(signal.SIGINT, prev)


def _say(stream, result):
    stream.write(f"[{result.status}] {result.message}\n")
    stream.flush()


def repl(conv, first_request, *, read=input, stream=sys.stderr, gate=terminal_gate,
         run_turn=run_turn) -> int:
    try:
        import readline  # noqa: F401 — history and line editing for input()
    except ImportError:
        pass
    _say(stream, run_turn(conv, first_request, gate, stream))
    stream.write(_HELP + "\n")
    while not conv.ended:
        try:
            line = read("chat> ").strip()
        except (EOFError, KeyboardInterrupt):
            stream.write("\n")
            return 0
        if not line:
            continue
        if not line.startswith("/"):
            _say(stream, run_turn(conv, line, gate, stream))
            continue
        cmd = line.split()[0]
        if cmd == "/quit":
            return 0
        if cmd == "/help":
            stream.write(_HELP + "\n")
        elif cmd == "/show":
            stream.write(conv.graph.render() + "\n")
        elif cmd == "/undo":
            _say(stream, conv.undo())
        elif cmd == "/rebuild":
            _say(stream, conv.rebuild())
        elif cmd == "/result":
            path = os.path.join(conv.run_dir, "result.py")
            stream.write(f"{path}\n")
            if os.path.exists(path):
                with open(path) as f:
                    stream.write(f.read())
        else:
            stream.write(f"unknown command {cmd} — /help\n")
        stream.flush()
    return 1
```

`salome_agent/cli.py`:
- `parse_args`: `parser.add_argument("--chat", action="store_true", help="after the first result, keep talking to the agent in a REPL (needs a SALOME session)")`.
- `main`, right after `setup_logging`:

```python
    if args.chat and args.batch:
        log.error("--chat is interactive; it cannot be combined with --batch")
        return 2
    if args.chat and args.no_session:
        log.error("--chat needs a SALOME session; drop --no-session")
        return 2
```

- In `main`, pass `hold=args.chat` to `server_conn.connect(...)`.
- In `main`, after the client is connected, branch on `args.chat`:

```python
    if args.chat:
        from salome_agent import chat
        from salome_agent.conversation import Conversation

        try:
            conv = Conversation(config, client, run_id, verbose=args.verbose,
                                request_timeout=args.timeout)
            return chat.repl(conv, args.request)
        finally:
            try:
                client.close()
            except Exception:
                pass
```

  The existing one-shot path stays below, untouched.

`README.md`: add a "Chat mode" section after the CLI usage section. Show `salome-agent "a small house" --chat`, the command list from `_HELP`, and one sentence each on: one connection per conversation, edits rebuild downstream only with names unchanged, `/undo`, `/rebuild`, and "`--chat` cannot be combined with `--batch` or `--no-session`".

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/unit -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add salome_agent/chat.py salome_agent/cli.py README.md tests/unit/test_chat.py tests/unit/test_cli.py
git commit -m "feat: --chat REPL over the conversation engine (#15)"
```

---

### Task 14: Live multi-turn verification (the real proof)

Mocked LLMs cannot catch message-shape, prompt-thinness or schema problems. This task is run **by the user** against the real LLM and real SALOME, after a **full** SALOME / agent-server restart (the server caches the tools module in `sys.modules`).

**Files:** none (report results back into the task).

- [ ] **Step 1: Restart the headless server** (`scripts/start-headless-server.sh`), confirming no stale server is running first.

- [ ] **Step 2: Run the scenario**

`salome-agent "a small house: 200x120x80mm box base with a 200x120x40mm prism roof on top" --chat -vv`

Then, at `chat>`:
1. `make the roof taller` — expect `edit nX.dz: 40 -> …` and `rebuilding …` lines.
2. `/show` — note every object name.
3. `add a 20x20x60mm chimney on the roof`
4. `/undo`
5. `/result`, then `/quit`.

- [ ] **Step 3: Check, and write each result into task 194**

1. After step 1: the roof's bbox z-extent grew (look at the `measurements` in the `-vv` trace). The base's bbox is unchanged. **No name changed**: compare `/show` output before and after, and the study object names.
2. After step 3: the chimney node is in its own `turn 3: amend` block, and its bbox sits on top of the roof.
3. After `/undo`: the chimney is gone from the study and from `/show`, and the taller roof is still there.
4. `salome shell -- python <run_dir>/result.py` reproduces the study (same object names, same bboxes via a quick `measure` or the GUI).
5. `build_graph.json` and `report.md` in the run dir carry every turn.
6. Ctrl-C during a turn returns to `chat>`, and the model is unchanged (compare `/show`).

Repeat steps 2–3 once in the SALOME GUI with the plugin server (`--chat` from a terminal while the GUI server is running). There, also check that the old roof really disappears from the viewer after an edit. `forget` removes the study object, but a lingering presentation in the viewer is a known risk to watch for.

- [ ] **Step 4: Anything that fails becomes a new task, not a silent fix.** Report the run ids and the failing assertion.

---

## Follow-up plan (not in this one)

salome-agent-gui: `RunWorker` becomes a long-lived `ConversationWorker` (one `QThread` owning the `Conversation`, turns from a queue, `done` → `turn_done`, undo and rebuild actions, rebuild enabled only when `degraded`). Write that plan after Task 14 passes. It lands before task 250 (dock widget) and task 249 (transcript polish), per the spec.
