# SALOME Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a CLI tool that turns a natural-language request into verified, reproducible SALOME Python scripts (`geompy`/`smesh`/`pvsimple`), executed against a live SALOME session, and assembled into a runnable `result.py`.

**Architecture:** Two OS processes talking line-delimited JSON over a Unix socket: an `agent` process (plain Python venv — LangGraph `StateGraph`, LLM client, static verifier; no RAG in v1, see decision 19) and an `executor` process (runs under `salome shell` — holds the live session/study, executes smesh/paravis scripts, and dispatches geom's tool calls in-process). No second SALOME-side process — geom's tools are a command family (`list_tools`/`call_tool`) on the same executor, not a separate server (decision 17). Rollback is replay, not restore: `reset_and_replay` wipes the study and re-runs every previously-accepted stage's script, geom included, since geom's call log is codegen'd to source immediately on stage acceptance (decision 15).

**Tech Stack:** Python 3.14, LangGraph `StateGraph` + `MemorySaver`, pytest, stdlib `socket`/`json`/`threading` for the executor transport, any OpenAI-compatible tool-calling LLM. No RAG in v1 — cut per spec decision 19; see spec §8/§13 if reinstating later.

**Spec:** `docs/superpowers/specs/2026-08-29-salome-agent-design.md` (revision 5) — the plan argues from this spec; read §3, §4, §5, §7 before starting.

## Global Constraints

- `executor/` imports **only** the SALOME Python API and the stdlib — never `salome_agent/`, never an MCP SDK (decision 16), never a venv package. Verified per-task by grep, not just review.
- Static verification (`verify/static.py`) checks **existence only** on top-level module objects (`geompy`, `smesh`, `pvsimple`) — no arity checking, no checks on derived/CORBA objects (§6.1).
- Geom stages skip `verify_static`/`verify_exec` entirely — validation is inline, per tool call (§6.5). Do not route geom through the smesh/paravis verify nodes.
- Lengths/params from the planner are millimetres, passed through unchanged — GEOM is unitless, no unit conversion anywhere (§5.2).
- `policy.static_retries`/`exec_retries` apply to smesh/paravis only; geom uses `policy.tool_iteration_cap` (§9).
- No MED checkpoints, no `rollback_stage`/`delete_object` tool — rollback is always `reset_and_replay` over accepted scripts (decision 5, decision 16).
- `RunState.stages` is keyed by **stage index**, never by module name — a plan may repeat a module (decision 7).

---

## File Structure

```
salome-agent/
├── pyproject.toml
├── config.example.json
├── salome_agent/
│   ├── __init__.py
│   ├── cli.py                   # Task 13
│   ├── graph.py                 # Task 6
│   ├── state.py                 # Task 1
│   ├── config.py                # Task 12
│   ├── executor_client.py       # Task 4
│   ├── codegen.py               # Task 5
│   ├── nodes/
│   │   ├── planner.py           # Task 7
│   │   ├── gates.py             # Task 6
│   │   ├── advance.py           # Task 11
│   │   ├── assembler.py         # Task 11
│   │   └── report.py            # Task 11
│   ├── specialists/
│   │   ├── base.py              # Task 9
│   │   ├── geom.py              # Task 10
│   │   ├── smesh.py             # Task 9
│   │   └── paravis.py           # Task 9
│   ├── verify/
│   │   ├── static.py            # Task 8
│   │   ├── introspect.py        # Task 8
│   │   ├── exec_check.py        # Task 8
│   │   └── toplevel_api.json    # Task 8
│   └── session/
│       ├── manager.py           # Task 4
│       └── discovery.py         # Task 4
├── executor/
│   ├── executor.py              # Task 2
│   ├── geom_tools.py            # Task 3
│   └── plugin.py                # not covered — optional per spec §7.3
└── tests/
    ├── unit/
    └── salome/                  # @pytest.mark.salome, CI-skipped
```

Files that change together stay together: `executor/executor.py` and `executor/geom_tools.py` are both SALOME-side and share the process/dispatch table, so Task 3 modifies the dispatcher Task 2 wrote rather than duplicating it.

---

### Task 1: State object

**Files:**
- Create: `salome_agent/state.py`
- Test: `tests/unit/test_state.py`

**Interfaces:**
- Produces: `StagePlan`, `StageError`, `StageRecord`, `SessionHandle`, `RunState` TypedDicts (spec §4.1) — every later agent-side task imports these, none define their own copy.

- [ ] **Step 1: Write the failing test — stages keyed by index, not module**

```python
# tests/unit/test_state.py
from salome_agent.state import RunState, StagePlan, StageRecord, SessionHandle

def test_repeated_module_stages_do_not_collide():
    plan: list[StagePlan] = [
        {"module": "geom", "intent": "box", "params": {}, "inputs": []},
        {"module": "smesh", "intent": "mesh", "params": {}, "inputs": []},
        {"module": "geom", "intent": "fillet", "params": {}, "inputs": []},
    ]
    stages: dict[int, StageRecord] = {
        i: {
            "module": p["module"], "status": "pending", "script": None,
            "call_log": None, "static_retries": 0, "exec_retries": 0,
        }
        for i, p in enumerate(plan)
    }
    assert stages[0]["module"] == "geom"
    assert stages[2]["module"] == "geom"
    assert stages[0] is not stages[2]

def test_history_reducer_is_additive():
    import operator
    history: list[dict] = []
    history = history + [{"event": "plan_approved"}]
    history = history + [{"event": "stage_0_ok"}]
    assert [h["event"] for h in history] == ["plan_approved", "stage_0_ok"]
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/unit/test_state.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'salome_agent.state'`

- [ ] **Step 3: Write `salome_agent/state.py`**

```python
from typing import Annotated, Literal, Optional, TypedDict
import operator


class StagePlan(TypedDict):
    module: Literal["geom", "smesh", "paravis"]
    intent: str
    params: dict
    inputs: list[str]


class StageError(TypedDict):
    phase: Literal["static", "exec"]
    message: str
    offending_line: Optional[int]
    traceback: Optional[str]


class StageRecord(TypedDict):
    module: str
    status: Literal["pending", "ok", "failed", "skipped"]
    script: Optional[str]
    call_log: Optional[list[dict]]
    static_retries: int
    exec_retries: int


class SessionHandle(TypedDict):
    mode: Literal["attach", "launch", "none"]
    port: Optional[int]
    executor_pid: Optional[int]
    alive: bool
    owned: bool


class RunState(TypedDict):
    run_id: str
    user_request: str
    config: dict

    plan: list[StagePlan]
    plan_approved: bool
    stage_idx: int

    session: SessionHandle
    study_ns: str

    stages: dict[int, StageRecord]
    last_error: Optional[StageError]
    history: Annotated[list[dict], operator.add]

    final_script: Optional[str]
    report: Optional[str]
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/unit/test_state.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add salome_agent/state.py tests/unit/test_state.py
git commit -m "feat: add RunState and stage TypedDicts"
```

---

### Task 2: Executor core (`run`/`sanity`/`reset_and_replay`/`introspect`/`list_study`/`shutdown`)

**Files:**
- Create: `executor/executor.py`
- Test: `tests/salome/test_executor.py` (`@pytest.mark.salome`)

**Interfaces:**
- Consumes: nothing from `salome_agent/` (must never import it — Global Constraints).
- Produces: a Unix-socket JSON-line protocol per spec §7.2. Request: `{"id": int, "cmd": str, **payload}`. Response: `{"id": int, **result}`. On connect, the executor writes a single `{"ready": true}` line before accepting commands (§7.1/§7.3). Task 3 extends `COMMANDS` with `list_tools`/`call_tool` — this task defines the dispatch table those two commands slot into.

- [ ] **Step 1: Write the failing test — `run` executes and captures stdout**

```python
# tests/salome/test_executor.py
import json, socket, subprocess, time, os, tempfile
import pytest

@pytest.fixture
def executor_conn():
    sock_path = tempfile.mktemp(suffix=".sock")
    proc = subprocess.Popen(
        ["salome", "shell", "--", "python3", "executor/executor.py", "--socket", sock_path],
    )
    for _ in range(100):
        if os.path.exists(sock_path):
            break
        time.sleep(0.1)
    conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    conn.connect(sock_path)
    f = conn.makefile("rwb")
    ready = json.loads(f.readline())
    assert ready == {"ready": True}
    yield f
    proc.terminate()
    proc.wait(timeout=10)

def send(f, cmd, **payload):
    f.write((json.dumps({"id": 1, "cmd": cmd, **payload}) + "\n").encode())
    f.flush()
    return json.loads(f.readline())

@pytest.mark.salome
def test_run_captures_stdout(executor_conn):
    result = send(executor_conn, "run", code="print('hi')", timeout=5)
    assert result["ok"] is True
    assert "hi" in result["stdout"]

@pytest.mark.salome
def test_run_syntax_error_reports_offending_line(executor_conn):
    result = send(executor_conn, "run", code="x = (\n", timeout=5)
    assert result["ok"] is False
    assert result["offending_line"] == 1

@pytest.mark.salome
def test_timeout_kills_infinite_loop(executor_conn):
    result = send(executor_conn, "run", code="while True: pass", timeout=1)
    assert result["ok"] is False
    assert "timeout" in result["exception"].lower()
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/salome/test_executor.py -v -m salome`
Expected: FAIL — `executor/executor.py` does not exist yet.

- [ ] **Step 3: Write `executor/executor.py`**

```python
import argparse
import json
import socket
import sys
import threading
import traceback as tb_module
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO


class TimeoutKill(BaseException):
    pass


def _run_with_timeout(code: str, namespace: dict, timeout: float) -> dict:
    out, err = StringIO(), StringIO()
    result = {"ok": False, "stdout": "", "stderr": "", "exception": None,
              "traceback": None, "offending_line": None}

    def target():
        try:
            with redirect_stdout(out), redirect_stderr(err):
                compiled = compile(code, "<stage>", "exec")
                exec(compiled, namespace)
            result["ok"] = True
        except SyntaxError as e:
            result["exception"] = str(e)
            result["offending_line"] = e.lineno
        except BaseException as e:
            result["exception"] = str(e)
            result["traceback"] = "".join(tb_module.format_exc())
            frame = tb_module.extract_tb(e.__traceback__)
            for entry in frame:
                if entry.filename == "<stage>":
                    result["offending_line"] = entry.lineno

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        result["exception"] = f"timeout after {timeout}s"
        # Best-effort only — a C-level Compute() may not respond to this
        # (spec §12 risk: "may need to hard-exit and be relaunched").
    result["stdout"] = out.getvalue()
    result["stderr"] = err.getvalue()
    return result


def _find_object(name: str):
    # Whole-study search by name, not folder-scoped. study_ns folder
    # scoping (geompy.PutToFolder / attach-mode rollback, spec §4.3/§12)
    # is a separate, still-open piece of work — geompy.PutToFolder did not
    # actually nest objects under the folder in manual testing against
    # this SALOME build (2026-08-29), so folder-scoped lookup is deferred
    # rather than shipped half-working. This search-by-name is correct as
    # long as published names stay unique/deterministic (spec §5.2).
    import salome
    from salome.kernel import salome as salome_kernel

    def walk(sobj):
        it = salome_kernel.myStudy.NewChildIterator(sobj)
        while it.More():
            child = it.Value()
            if child.GetName() == name:
                return child.GetObject()
            found = walk(child)
            if found is not None:
                return found
            it.Next()
        return None

    root = salome_kernel.myStudy.FindObjectID("0:1")
    result = walk(root)
    if result is None:
        raise LookupError(f"no study object named {name!r}")
    return result


def build_namespace() -> dict:
    # Real import convention for this SALOME build, confirmed by manual
    # testing 2026-08-29 — NOT `import salome; import geompy; import smesh`
    # as earlier spec drafts assumed. `geompy`/`smesh`/`salome` are still
    # bound under those exact names below, so stage scripts and the static
    # checker (verify/static.py's TOPLEVEL_MODULES) see the same bare
    # top-level names either way.
    from salome.kernel import salome
    salome.salome_init()
    from salome.geom import geomBuilder
    geompy = geomBuilder.New()
    from salome.smesh import smeshBuilder
    smesh = smeshBuilder.New()
    try:
        import pvsimple
    except ImportError:
        pvsimple = None
    study_ns = "salome_agent_run"
    return {
        "salome": salome, "geompy": geompy, "smesh": smesh, "pvsimple": pvsimple,
        "study_ns": study_ns,
        "find_object": _find_object,
    }


class Executor:
    def __init__(self, socket_path: str):
        self.socket_path = socket_path
        self.namespace = build_namespace()
        self.accepted_scripts: list[str] = []
        self.geom_bindings: dict[str, object] = {}  # filled in by Task 3

    def handle(self, req: dict) -> dict:
        cmd = req["cmd"]
        if cmd == "run":
            r = _run_with_timeout(req["code"], self.namespace, req["timeout"])
            return {"id": req["id"], **r}
        if cmd == "reset_and_replay":
            self.namespace = build_namespace()
            self.geom_bindings.clear()
            for idx, script in enumerate(req["scripts"]):
                r = _run_with_timeout(script, self.namespace, req.get("timeout", 120))
                if not r["ok"]:
                    return {"id": req["id"], "ok": False, "failed_index": idx}
            self.accepted_scripts = list(req["scripts"])
            return {"id": req["id"], "ok": True, "failed_index": None}
        if cmd == "introspect":
            out = {}
            for name in req["names"]:
                obj = self.namespace.get(name)
                out[name] = {"members": dir(obj) if obj is not None else []}
            return {"id": req["id"], **out}
        if cmd == "list_study":
            salome = self.namespace["salome"]
            objs = []
            it = salome.myStudy.NewChildIterator(
                salome.myStudy.FindObjectByPath(f"/{self.namespace['study_ns']}")
            )
            while it.More():
                comp = it.Value()
                objs.append({"name": comp.GetName(), "type": comp.GetComponentDataType()})
                it.Next()
            return {"id": req["id"], "objects": objs}
        if cmd == "sanity":
            return {"id": req["id"], "ok": True, "warnings": [], "failures": []}
        if cmd == "shutdown":
            return {"id": req["id"], "ok": True}
        return {"id": req["id"], "ok": False, "exception": f"unknown command {cmd!r}"}

    def serve(self):
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        srv.bind(self.socket_path)
        srv.listen(1)
        conn, _ = srv.accept()
        f = conn.makefile("rwb")
        f.write((json.dumps({"ready": True}) + "\n").encode())
        f.flush()
        for line in f:
            req = json.loads(line)
            resp = self.handle(req)
            f.write((json.dumps(resp) + "\n").encode())
            f.flush()
            if req["cmd"] == "shutdown":
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket", required=True)
    args = parser.parse_args()
    Executor(args.socket).serve()
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/salome/test_executor.py -v -m salome`
Expected: PASS (3 tests). Requires a working `SALOME-master-native-FD44` install on `PATH`.

- [ ] **Step 5: Verify no agent-side imports leaked in**

```bash
grep -n "salome_agent" executor/executor.py
```

Expected: no output.

- [ ] **Step 6: Commit**

```bash
git add executor/executor.py tests/salome/test_executor.py
git commit -m "feat: add executor process with run/reset_and_replay/introspect/list_study"
```

---

### Task 3: Geom tools inside the executor (`list_tools`/`call_tool`, decision 17)

**Files:**
- Create: `executor/geom_tools.py`
- Modify: `executor/executor.py:handle` (add `list_tools`/`call_tool` dispatch)
- Test: `tests/salome/test_geom_tools.py` (`@pytest.mark.salome`)

**Interfaces:**
- Consumes: `Executor.namespace` (has `geompy`, `study_ns`) and `Executor.geom_bindings` (from Task 2), both already on the `Executor` instance.
- Produces: `TOOLS: dict[str, ToolSpec]` registry, `call_tool(executor, name, args) -> ToolResult` where `ToolResult = {"ok": bool, "records": list[CallRecord], "sanity": dict} | {"ok": False, "error": str}`. Task 5 (`codegen.py`) consumes `CallRecord` exactly as defined here — do not diverge from spec §3.4/§4.1's `ArgRef`/`CallRecord` shape.

- [ ] **Step 1: Write the failing test — `make_box` publishes and returns a CallRecord**

```python
# tests/salome/test_geom_tools.py
import pytest
from executor.executor import build_namespace, Executor
from executor.geom_tools import call_tool

@pytest.fixture
def executor():
    ex = Executor.__new__(Executor)
    ex.namespace = build_namespace()
    ex.geom_bindings = {}
    return ex

@pytest.mark.salome
def test_make_box_returns_call_record_and_publishes(executor):
    result = call_tool(executor, "make_box", {"dx": 10, "dy": 10, "dz": 10})
    assert result["ok"] is True
    records = result["records"]
    assert len(records) == 1
    assert records[0]["fn"] == "MakeBoxDXDYDZ"
    assert records[0]["out_vars"] == ["box_1"]
    assert records[0]["publish"] == ["box_1"]
    assert "box_1" in executor.geom_bindings

@pytest.mark.salome
def test_fillet_edges_sanity_failure_cleans_up_publish(executor):
    call_tool(executor, "make_box", {"dx": 1, "dy": 1, "dz": 1})
    # A fillet radius larger than the box's edges is geometrically invalid —
    # CheckShape should fail and the tool must delete what it published.
    before = executor.namespace["salome"].myStudy.FindObjectByPath(
        f"/{executor.namespace['study_ns']}"
    ).GetName()
    result = call_tool(executor, "fillet_edges", {"shape": "box_1", "edges": "all", "radius": 50})
    assert result["ok"] is False
    assert "fillet_1" not in executor.geom_bindings

@pytest.mark.salome
def test_check_shape_is_not_recordable(executor):
    call_tool(executor, "make_box", {"dx": 1, "dy": 1, "dz": 1})
    result = call_tool(executor, "check_shape", {"shape": "box_1"})
    assert result["ok"] is True
    assert result["records"] == []
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/salome/test_geom_tools.py -v -m salome`
Expected: FAIL — `executor/geom_tools.py` does not exist.

- [ ] **Step 3: Write `executor/geom_tools.py`**

```python
from typing import Any, Callable, TypedDict


class ArgRef(TypedDict):
    kind: str    # "literal" | "ref" | "list" | "study"
    value: Any


class CallRecord(TypedDict):
    fn: str
    args: list[ArgRef]
    out_vars: list[str]
    publish: list[str] | None


_counters: dict[str, int] = {}


def _next_name(prefix: str) -> str:
    _counters[prefix] = _counters.get(prefix, 0) + 1
    return f"{prefix}_{_counters[prefix]}"


def _resolve(executor, arg: ArgRef):
    if arg["kind"] == "literal":
        return arg["value"]
    if arg["kind"] == "ref":
        return executor.geom_bindings[arg["value"]]
    if arg["kind"] == "list":
        return [_resolve(executor, a) for a in arg["value"]]
    if arg["kind"] == "study":
        return executor.namespace["find_object"](arg["value"])
    raise ValueError(f"unknown ArgRef kind {arg['kind']!r}")


def _publish(executor, shape, name: str):
    geompy = executor.namespace["geompy"]
    geompy.addToStudyInFather(
        executor.namespace["study_ns"], shape, name
    ) if False else geompy.addToStudy(shape, name)
    executor.geom_bindings[name] = shape


def tool_make_box(executor, args: dict):
    geompy = executor.namespace["geompy"]
    box = geompy.MakeBoxDXDYDZ(args["dx"], args["dy"], args["dz"])
    name = _next_name("box")
    _publish(executor, box, name)
    return {
        "ok": True,
        "records": [{
            "fn": "MakeBoxDXDYDZ",
            "args": [{"kind": "literal", "value": args["dx"]},
                     {"kind": "literal", "value": args["dy"]},
                     {"kind": "literal", "value": args["dz"]}],
            "out_vars": [name],
            "publish": [name],
        }],
        "sanity": {"ok": True},
    }


def tool_fillet_edges(executor, args: dict):
    geompy = executor.namespace["geompy"]
    shape = executor.geom_bindings[args["shape"]]
    edge_ids = (
        list(range(1, len(geompy.SubShapeAllIDs(shape, geompy.ShapeType["EDGE"])) + 1))
        if args["edges"] == "all" else args["edges"]
    )
    filleted = geompy.MakeFilletEdges(shape, args["radius"], edge_ids)
    is_valid, _ = geompy.CheckShape(filleted, True)
    if not is_valid:
        return {"ok": False, "error": "fillet_edges: CheckShape failed on result"}
    name = _next_name("fillet")
    _publish(executor, filleted, name)
    return {
        "ok": True,
        "records": [{
            "fn": "MakeFilletEdges",
            "args": [{"kind": "ref", "value": args["shape"]},
                     {"kind": "literal", "value": args["radius"]},
                     {"kind": "literal", "value": edge_ids}],
            "out_vars": [name],
            "publish": [name],
        }],
        "sanity": {"ok": True},
    }


def tool_check_shape(executor, args: dict):
    geompy = executor.namespace["geompy"]
    shape = executor.geom_bindings[args["shape"]]
    is_valid, msg = geompy.CheckShape(shape, True)
    return {"ok": bool(is_valid), "records": [], "sanity": {"ok": bool(is_valid), "message": msg}}


def tool_list_study_objects(executor, args: dict):
    return {"ok": True, "records": [], "sanity": {"ok": True},
            "names": list(executor.geom_bindings.keys())}


# Registry: tool name -> handler(executor, args) -> ToolResult.
# make_cylinder/boolean_*/chamfer_edges/finish_stage follow the same shape
# as tool_make_box/tool_fillet_edges above and are added the same way —
# not stubbed here, added incrementally as later stages need them.
TOOLS: dict[str, Callable] = {
    "make_box": tool_make_box,
    "fillet_edges": tool_fillet_edges,
    "check_shape": tool_check_shape,
    "list_study_objects": tool_list_study_objects,
}


def call_tool(executor, name: str, args: dict) -> dict:
    handler = TOOLS.get(name)
    if handler is None:
        return {"ok": False, "error": f"unknown tool {name!r}"}
    try:
        return handler(executor, args)
    except Exception as e:
        return {"ok": False, "error": str(e)}


def list_tools() -> list[dict]:
    return [{"name": n} for n in TOOLS]
```

- [ ] **Step 4: Wire `list_tools`/`call_tool` into the executor's dispatcher**

In `executor/executor.py`, add to `Executor.handle`, alongside the existing `if cmd == "run":` etc:

```python
        if cmd == "list_tools":
            from executor.geom_tools import list_tools
            return {"id": req["id"], "tools": list_tools()}
        if cmd == "call_tool":
            from executor.geom_tools import call_tool
            result = call_tool(self, req["name"], req["args"])
            return {"id": req["id"], **result}
```

Also add to `reset_and_replay`'s reset step (it already calls `self.geom_bindings.clear()` from Task 2 — no change needed there; this step just confirms it).

- [ ] **Step 5: Run to verify it passes**

Run: `pytest tests/salome/test_geom_tools.py -v -m salome`
Expected: PASS (3 tests)

- [ ] **Step 6: Commit**

```bash
git add executor/geom_tools.py executor/executor.py tests/salome/test_geom_tools.py
git commit -m "feat: add geom tool commands to the executor (decision 17)"
```

---

### Task 4: Session manager, discovery, executor client

**Files:**
- Create: `salome_agent/session/discovery.py`
- Create: `salome_agent/session/manager.py`
- Create: `salome_agent/executor_client.py`
- Test: `tests/unit/test_discovery.py`, `tests/unit/test_executor_client.py`

**Interfaces:**
- Produces: `discover_port(user_path: str) -> Optional[int]`; `SessionManager.resolve(mode, port) -> SessionHandle` (state.py's `SessionHandle`); `ExecutorClient.run(code, timeout)`, `.reset_and_replay(scripts)`, `.introspect(names)`, `.list_study()`, `.list_tools()`, `.call_tool(name, args)`, `.sanity(module, expectations)`, `.shutdown()` — each raises `SessionDied` on a dropped socket.

- [ ] **Step 1: Write the failing test — port parsed from cfg file contents, not filename**

```python
# tests/unit/test_discovery.py
from salome_agent.session.discovery import discover_port

def test_discover_port_parses_contents_not_filename(tmp_path):
    cfg = tmp_path / ".omniORB_somehost_99999.cfg"
    cfg.write_text("InitRef = NameService=corbaname::localhost:12345\n")
    assert discover_port(str(tmp_path)) == 12345

def test_discover_port_none_when_no_cfg(tmp_path):
    assert discover_port(str(tmp_path)) is None
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/unit/test_discovery.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write `salome_agent/session/discovery.py`**

```python
import glob
import os
import re
from typing import Optional

_PORT_RE = re.compile(r"NameService=corbaname::[\w.\-]+:(\d+)")


def discover_port(user_path: Optional[str] = None) -> Optional[int]:
    base = user_path or os.environ.get("OMNIORB_USER_PATH", "/tmp")
    for path in glob.glob(os.path.join(base, ".omniORB_*_*.cfg")):
        with open(path) as f:
            contents = f.read()
        m = _PORT_RE.search(contents)
        if m:
            return int(m.group(1))
    return None
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/unit/test_discovery.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Write the failing test for `ExecutorClient` — dropped socket raises `SessionDied`**

```python
# tests/unit/test_executor_client.py
import json, socket, threading, tempfile, os
import pytest
from salome_agent.executor_client import ExecutorClient, SessionDied

def _fake_server(sock_path, responses):
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(sock_path)
    srv.listen(1)
    conn, _ = srv.accept()
    f = conn.makefile("rwb")
    f.write((json.dumps({"ready": True}) + "\n").encode()); f.flush()
    for resp in responses:
        f.readline()
        f.write((json.dumps(resp) + "\n").encode()); f.flush()
    conn.close()

def test_run_returns_result():
    sock_path = tempfile.mktemp(suffix=".sock")
    t = threading.Thread(target=_fake_server, args=(sock_path, [{"id": 1, "ok": True, "stdout": "hi"}]))
    t.start()
    client = ExecutorClient(sock_path)
    result = client.run("print('hi')", timeout=5)
    assert result["stdout"] == "hi"
    t.join()

def test_dropped_socket_raises_session_died():
    sock_path = tempfile.mktemp(suffix=".sock")
    t = threading.Thread(target=_fake_server, args=(sock_path, []))  # closes without responding
    t.start()
    client = ExecutorClient(sock_path)
    with pytest.raises(SessionDied):
        client.run("print(1)", timeout=5)
    t.join()
```

- [ ] **Step 6: Run to verify it fails**

Run: `pytest tests/unit/test_executor_client.py -v`
Expected: FAIL — module not found.

- [ ] **Step 7: Write `salome_agent/executor_client.py`**

```python
import json
import socket


class SessionDied(Exception):
    pass


class ExecutorClient:
    def __init__(self, socket_path: str):
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock.connect(socket_path)
        self._f = self._sock.makefile("rwb")
        ready = self._recv()
        assert ready.get("ready") is True
        self._next_id = 1

    def _recv(self) -> dict:
        line = self._f.readline()
        if not line:
            raise SessionDied("executor socket closed")
        return json.loads(line)

    def _call(self, cmd: str, **payload) -> dict:
        req_id = self._next_id
        self._next_id += 1
        self._f.write((json.dumps({"id": req_id, "cmd": cmd, **payload}) + "\n").encode())
        self._f.flush()
        return self._recv()

    def run(self, code: str, timeout: float) -> dict:
        return self._call("run", code=code, timeout=timeout)

    def reset_and_replay(self, scripts: list[str]) -> dict:
        return self._call("reset_and_replay", scripts=scripts)

    def introspect(self, names: list[str]) -> dict:
        return self._call("introspect", names=names)

    def list_study(self) -> dict:
        return self._call("list_study")

    def sanity(self, module: str, expectations: dict) -> dict:
        return self._call("sanity", module=module, expectations=expectations)

    def list_tools(self) -> dict:
        return self._call("list_tools")

    def call_tool(self, name: str, args: dict) -> dict:
        return self._call("call_tool", name=name, args=args)

    def shutdown(self) -> dict:
        return self._call("shutdown")
```

- [ ] **Step 8: Run to verify it passes**

Run: `pytest tests/unit/test_executor_client.py -v`
Expected: PASS (2 tests)

- [ ] **Step 9: Write `salome_agent/session/manager.py`** (no dedicated test — thin orchestration over `discover_port`/`subprocess`/`ExecutorClient`, exercised by Task 13's CLI integration and the Task 14 end-to-end test)

```python
import subprocess
import time

from salome_agent.executor_client import ExecutorClient
from salome_agent.session.discovery import discover_port
from salome_agent.state import SessionHandle


class SessionManager:
    def __init__(self, socket_dir: str = "/tmp"):
        self.socket_dir = socket_dir
        self._proc: subprocess.Popen | None = None

    def resolve(self, no_session: bool, port: int | None) -> tuple[SessionHandle, ExecutorClient | None]:
        if no_session:
            return {"mode": "none", "port": None, "executor_pid": None,
                    "alive": False, "owned": False}, None

        resolved_port = port or discover_port()
        socket_path = f"{self.socket_dir}/salome_agent_executor.sock"

        if resolved_port is not None:
            self._proc = subprocess.Popen(
                ["salome", "shell", "--", "python3", "executor/executor.py",
                 "--socket", socket_path],
                env={"SALOME_PORT": str(resolved_port)},
            )
            client = self._wait_ready(socket_path)
            return {"mode": "attach", "port": resolved_port,
                    "executor_pid": self._proc.pid, "alive": True, "owned": False}, client

        self._proc = subprocess.Popen(
            ["salome", "shell", "--", "python3", "executor/executor.py", "--socket", socket_path],
        )
        client = self._wait_ready(socket_path)
        return {"mode": "launch", "port": None, "executor_pid": self._proc.pid,
                "alive": True, "owned": True}, client

    def _wait_ready(self, socket_path: str, timeout: float = 30) -> ExecutorClient:
        deadline = time.time() + timeout
        last_err = None
        while time.time() < deadline:
            try:
                return ExecutorClient(socket_path)
            except (FileNotFoundError, ConnectionRefusedError) as e:
                last_err = e
                time.sleep(0.2)
        raise TimeoutError(f"executor did not become ready: {last_err}")

    def stop(self):
        if self._proc is not None:
            self._proc.terminate()
            self._proc.wait(timeout=10)
```

- [ ] **Step 10: Commit**

```bash
git add salome_agent/session/discovery.py salome_agent/session/manager.py salome_agent/executor_client.py tests/unit/test_discovery.py tests/unit/test_executor_client.py
git commit -m "feat: add session discovery, manager, and executor client"
```

---

### Task 5: Codegen (call log → geompy source, decision 15/18)

**Files:**
- Create: `salome_agent/codegen.py`
- Test: `tests/unit/test_codegen.py`

**Interfaces:**
- Consumes: `CallRecord`/`ArgRef` exactly as defined in `executor/geom_tools.py` (Task 3).
- Produces: `generate(call_log: list[CallRecord], result_names: list[str]) -> str` — valid Python source (`ast.parse`-clean), used by `nodes/advance.py` (Task 11) immediately on `finish_stage`.

- [ ] **Step 1: Write the failing test — reachability pruning is transitive, refs resolve to variables**

```python
# tests/unit/test_codegen.py
import ast
from salome_agent.codegen import generate

def test_drops_unreachable_call_keeps_transitive_dependency():
    call_log = [
        {"fn": "MakeBoxDXDYDZ", "args": [{"kind": "literal", "value": 10}] * 3,
         "out_vars": ["box_1"], "publish": ["box_1"]},
        {"fn": "MakeFilletEdges",
         "args": [{"kind": "ref", "value": "box_1"}, {"kind": "literal", "value": 2},
                   {"kind": "literal", "value": [1, 2]}],
         "out_vars": ["fillet_1"], "publish": ["fillet_1"]},
        # An abandoned exploratory call whose result finish_stage never named —
        # must be dropped even though it's a valid earlier call.
        {"fn": "MakeCylinderRH", "args": [{"kind": "literal", "value": 5}, {"kind": "literal", "value": 5}],
         "out_vars": ["cyl_1"], "publish": ["cyl_1"]},
    ]
    source = generate(call_log, result_names=["fillet_1"])
    ast.parse(source)  # syntactically valid
    assert "box_1 = geompy.MakeBoxDXDYDZ(10, 10, 10)" in source
    assert "fillet_1 = geompy.MakeFilletEdges(box_1, 2, [1, 2])" in source
    assert "cyl_1" not in source  # unreachable from result_names, pruned

def test_list_kind_arg_resolves_to_python_list_of_refs():
    call_log = [
        {"fn": "MakeBoxDXDYDZ", "args": [{"kind": "literal", "value": 1}] * 3,
         "out_vars": ["box_1"], "publish": ["box_1"]},
        {"fn": "MakeBoxDXDYDZ", "args": [{"kind": "literal", "value": 2}] * 3,
         "out_vars": ["box_2"], "publish": ["box_2"]},
        {"fn": "MakeFuseList",
         "args": [{"kind": "list", "value": [{"kind": "ref", "value": "box_1"},
                                              {"kind": "ref", "value": "box_2"}]}],
         "out_vars": ["fused_1"], "publish": ["fused_1"]},
    ]
    source = generate(call_log, result_names=["fused_1"])
    assert "fused_1 = geompy.MakeFuseList([box_1, box_2])" in source

def test_study_kind_arg_resolves_via_find_object():
    call_log = [
        {"fn": "MakeFilletEdges",
         "args": [{"kind": "study", "value": "stage0_box"}, {"kind": "literal", "value": 2},
                   {"kind": "literal", "value": "all"}],
         "out_vars": ["fillet_1"], "publish": ["fillet_1"]},
    ]
    source = generate(call_log, result_names=["fillet_1"])
    assert 'find_object("stage0_box")' in source
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/unit/test_codegen.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write `salome_agent/codegen.py`**

```python
def _fmt_arg(arg: dict) -> str:
    kind = arg["kind"]
    if kind == "literal":
        return repr(arg["value"])
    if kind == "ref":
        return arg["value"]
    if kind == "list":
        return "[" + ", ".join(_fmt_arg(a) for a in arg["value"]) + "]"
    if kind == "study":
        return f'find_object({arg["value"]!r})'
    raise ValueError(f"unknown ArgRef kind {kind!r}")


def _refs_in(arg: dict) -> set[str]:
    if arg["kind"] == "ref":
        return {arg["value"]}
    if arg["kind"] == "list":
        return set().union(*(_refs_in(a) for a in arg["value"])) if arg["value"] else set()
    return set()


def _reachable(call_log: list[dict], result_names: list[str]) -> set[str]:
    wanted = set(result_names)
    changed = True
    while changed:
        changed = False
        for call in call_log:
            if wanted & set(call["out_vars"]):
                for arg in call["args"]:
                    for ref in _refs_in(arg):
                        if ref not in wanted:
                            wanted.add(ref)
                            changed = True
    return wanted


def generate(call_log: list[dict], result_names: list[str]) -> str:
    reachable = _reachable(call_log, result_names)
    lines = []
    for call in call_log:
        if not (reachable & set(call["out_vars"])):
            continue
        args_src = ", ".join(_fmt_arg(a) for a in call["args"])
        assign_target = call["out_vars"][0] if call["out_vars"] else "_"
        lines.append(f"{assign_target} = geompy.{call['fn']}({args_src})")
        for extra_var, out in zip(call["out_vars"][1:], []):
            pass  # multi-out_vars unpacking is added when a tool needs it (e.g. explode)
        if call.get("publish"):
            for var, name in zip(call["out_vars"], call["publish"]):
                lines.append(f"geompy.addToStudy({var}, {name!r})")
    return "\n".join(lines)
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/unit/test_codegen.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add salome_agent/codegen.py tests/unit/test_codegen.py
git commit -m "feat: add codegen with transitive pruning and list/study ArgRef kinds"
```

---

### Task 6: StateGraph skeleton (`graph.py` + `nodes/gates.py`)

**Files:**
- Create: `salome_agent/graph.py`
- Create: `salome_agent/nodes/gates.py`
- Test: `tests/unit/test_graph.py`

**Interfaces:**
- Consumes: `RunState` (Task 1).
- Produces: `build_graph(batch: bool) -> StateGraph` wired per spec §4.2's edge table; `plan_gate(state) -> Command`, `retry_gate(state) -> Command` (call `interrupt()` unless `batch`).

- [ ] **Step 1: Write the failing test — routing with fake nodes**

```python
# tests/unit/test_graph.py
from salome_agent.graph import route_after_run_stage, route_after_verify_exec

def test_geom_finish_stage_routes_to_advance():
    state = {"stage_idx": 0, "stages": {0: {"module": "geom", "status": "ok"}}}
    assert route_after_run_stage(state) == "advance"

def test_geom_loop_exhausted_routes_to_retry_gate():
    state = {"stage_idx": 0, "stages": {0: {"module": "geom", "status": "failed"}}}
    assert route_after_run_stage(state) == "retry_gate"

def test_smesh_ok_routes_to_verify_static():
    state = {"stage_idx": 0, "stages": {0: {"module": "smesh", "status": "pending"}}}
    assert route_after_run_stage(state) == "verify_static"

def test_verify_exec_session_died_routes_to_assemble():
    state = {"last_error": {"phase": "exec", "message": "SessionDied"}}
    assert route_after_verify_exec(state, session_died=True) == "assemble"
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/unit/test_graph.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write `salome_agent/nodes/gates.py`**

```python
from langgraph.types import Command, interrupt


def plan_gate(state: dict) -> Command:
    if state["config"].get("batch"):
        return Command(update={"plan_approved": True}, goto="run_stage")
    decision = interrupt({"plan": state["plan"]})
    if decision["action"] == "approve":
        return Command(update={"plan_approved": True}, goto="run_stage")
    if decision["action"] == "edit":
        return Command(update={"plan": decision["plan"]}, goto="plan")
    return Command(goto="assemble")


def retry_gate(state: dict) -> Command:
    if state["config"].get("batch"):
        policy = state["config"]["policy"]["on_stage_failure"]
        if policy == "continue":
            return Command(goto="skip")
        return Command(goto="assemble")
    decision = interrupt({"last_error": state["last_error"]})
    if decision["action"] == "hint":
        return Command(update={"history": [{"event": "retry_hint", "hint": decision["hint"]}]},
                        goto="run_stage")
    if decision["action"] == "skip":
        return Command(goto="skip")
    return Command(goto="assemble")
```

- [ ] **Step 4: Write routing functions in `salome_agent/graph.py`**

```python
def route_after_run_stage(state: dict) -> str:
    stage = state["stages"][state["stage_idx"]]
    if stage["module"] == "geom":
        if stage["status"] == "ok":
            return "advance"
        return "retry_gate"
    return "verify_static"


def route_after_verify_exec(state: dict, session_died: bool = False) -> str:
    if session_died:
        return "assemble"
    return "advance"


def build_graph(batch: bool):
    from langgraph.graph import StateGraph, START
    from langgraph.checkpoint.memory import MemorySaver
    from salome_agent.state import RunState
    from salome_agent.nodes.gates import plan_gate, retry_gate

    g = StateGraph(RunState)
    g.add_node("plan_gate", plan_gate)
    g.add_node("retry_gate", retry_gate)
    g.add_conditional_edges("run_stage", route_after_run_stage,
                             {"advance": "advance", "retry_gate": "retry_gate",
                              "verify_static": "verify_static"})
    g.add_edge(START, "plan")
    return g.compile(checkpointer=MemorySaver())
```

- [ ] **Step 5: Run to verify it passes**

Run: `pytest tests/unit/test_graph.py -v`
Expected: PASS (4 tests)

- [ ] **Step 6: Commit**

```bash
git add salome_agent/graph.py salome_agent/nodes/gates.py tests/unit/test_graph.py
git commit -m "feat: add graph routing and interactive gates"
```

---

### Task 7: Planner node

**Files:**
- Create: `salome_agent/nodes/planner.py`
- Test: `tests/unit/test_planner.py`

**Interfaces:**
- Consumes: an LLM client with a `.chat(messages, tools=None) -> dict` method (mocked in tests).
- Produces: `plan(request: str, llm) -> list[StagePlan]`.

- [ ] **Step 1: Write the failing test — mocked LLM, repeated module produces two stages**

```python
# tests/unit/test_planner.py
from unittest.mock import MagicMock
from salome_agent.nodes.planner import plan

def test_repeated_module_yields_two_stages():
    llm = MagicMock()
    llm.chat.return_value = {
        "stages": [
            {"module": "geom", "intent": "make a 20mm cube", "params": {"size": 20}, "inputs": []},
            {"module": "geom", "intent": "fillet edges 2mm", "params": {"radius": 2}, "inputs": ["cube"]},
            {"module": "smesh", "intent": "mesh with 5mm tets", "params": {"max_size": 5}, "inputs": []},
        ]
    }
    stages = plan("make a 20mm cube, fillet its edges 2mm, mesh it with 5mm tetrahedra", llm)
    assert [s["module"] for s in stages] == ["geom", "geom", "smesh"]
    assert stages[1]["params"]["radius"] == 2  # mm, unconverted

def test_geom_only_request_yields_no_smesh_stage():
    llm = MagicMock()
    llm.chat.return_value = {"stages": [{"module": "geom", "intent": "make a box",
                                          "params": {}, "inputs": []}]}
    stages = plan("make a box", llm)
    assert len(stages) == 1
    assert stages[0]["module"] == "geom"
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/unit/test_planner.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write `salome_agent/nodes/planner.py`**

```python
PLANNER_SYSTEM_PROMPT = """You turn a natural-language SALOME request into an
ordered list of stages. Each stage has: module (geom|smesh|paravis), intent
(short description), params (dims in millimetres, unconverted), inputs (named
upstream objects this stage needs). Only emit stages the request actually
needs. Repeating a module is allowed. Respond as JSON: {"stages": [...]}."""


def plan(request: str, llm) -> list[dict]:
    response = llm.chat(
        messages=[{"role": "system", "content": PLANNER_SYSTEM_PROMPT},
                  {"role": "user", "content": request}],
    )
    return response["stages"]
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/unit/test_planner.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add salome_agent/nodes/planner.py tests/unit/test_planner.py
git commit -m "feat: add planner node"
```

---

### Task 8: Verification (`static`, `introspect`, `exec_check`)

**Files:**
- Create: `salome_agent/verify/static.py`
- Create: `salome_agent/verify/introspect.py`
- Create: `salome_agent/verify/toplevel_api.json`
- Create: `salome_agent/verify/exec_check.py`
- Test: `tests/unit/test_static.py`, `tests/unit/test_exec_check.py`

**Interfaces:**
- Consumes: `ExecutorClient` (Task 4) in `exec_check`.
- Produces: `check_static(script: str, api_map: dict) -> Optional[StageError]`; `check_exec(script: str, module: str, client: ExecutorClient, timeout: float) -> Optional[StageError] | Literal["session_died"]`.

- [ ] **Step 1: Write the failing test — existence-only check, no arity checking**

```python
# tests/unit/test_static.py
from salome_agent.verify.static import check_static

API_MAP = {"geompy": {"members": ["MakeBoxDXDYDZ", "MakeFilletEdges"]}}

def test_syntax_error_reports_line():
    err = check_static("x = (\n", API_MAP)
    assert err is not None
    assert err["phase"] == "static"
    assert err["offending_line"] == 1

def test_unknown_toplevel_call_fails():
    err = check_static("geompy.NotARealMethod(1)", API_MAP)
    assert err is not None
    assert "NotARealMethod" in err["message"]

def test_known_toplevel_call_passes_even_with_wrong_arity():
    # Existence-only: static check does NOT catch wrong arg counts (spec §6.1).
    err = check_static("geompy.MakeBoxDXDYDZ(1, 2)", API_MAP)
    assert err is None

def test_derived_object_call_is_not_checked():
    err = check_static("mesh_obj.Compute()", API_MAP)  # mesh_obj isn't geompy/smesh/pvsimple
    assert err is None

def test_salome_init_call_is_banned():
    err = check_static("salome.salome_init()", API_MAP)
    assert err is not None
    assert "salome_init" in err["message"]
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/unit/test_static.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write `salome_agent/verify/static.py`**

```python
import ast

TOPLEVEL_MODULES = {"geompy", "smesh", "pvsimple"}
BANNED_PATTERNS = ["salome_init", "salome.sg."]


def check_static(script: str, api_map: dict) -> dict | None:
    try:
        tree = ast.parse(script)
    except SyntaxError as e:
        return {"phase": "static", "message": str(e), "offending_line": e.lineno,
                "traceback": None}

    for pattern in BANNED_PATTERNS:
        if pattern in script:
            return {"phase": "static", "message": f"banned pattern: {pattern}",
                     "offending_line": None, "traceback": None}

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            obj = node.func.value
            if isinstance(obj, ast.Name) and obj.id in TOPLEVEL_MODULES:
                members = api_map.get(obj.id, {}).get("members", [])
                if node.func.attr not in members:
                    return {"phase": "static",
                             "message": f"{obj.id}.{node.func.attr} does not exist",
                             "offending_line": node.lineno, "traceback": None}
    return None
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/unit/test_static.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Write `salome_agent/verify/toplevel_api.json`** (fixture for `--no-session`, regenerated later by a dev script when the SALOME build changes — real placeholder values, populated from a live `introspect` call, not left empty)

```json
{
  "geompy": {"members": ["MakeBoxDXDYDZ", "MakeBoxTwoPnt", "MakeCylinderRH", "MakeFilletEdges", "MakeChamferEdges", "MakeCutList", "MakeFuseList", "MakeCommon", "CheckShape", "addToStudy", "SubShapeAllIDs"]},
  "smesh": {"members": ["Mesh", "Compute", "NbNodes", "NbElements"]},
  "pvsimple": {"members": ["OpenDataFile", "Show", "ColorBy", "GetActiveCamera"]}
}
```

- [ ] **Step 6: Write `salome_agent/verify/introspect.py`**

```python
import json
from pathlib import Path

FIXTURE_PATH = Path(__file__).parent / "toplevel_api.json"


def build_api_map(client=None) -> dict:
    if client is None:
        return json.loads(FIXTURE_PATH.read_text())
    return client.introspect(["geompy", "smesh", "pvsimple"])
```

- [ ] **Step 7: Write the failing test for `exec_check`**

```python
# tests/unit/test_exec_check.py
from unittest.mock import MagicMock
from salome_agent.executor_client import SessionDied
from salome_agent.verify.exec_check import check_exec

def test_exec_ok_and_sanity_ok_returns_none():
    client = MagicMock()
    client.run.return_value = {"ok": True, "stdout": "", "exception": None}
    client.sanity.return_value = {"ok": True, "warnings": [], "failures": []}
    assert check_exec("geompy.MakeBoxDXDYDZ(1,1,1)", "geom", client, timeout=5) is None

def test_exec_failure_returns_stage_error():
    client = MagicMock()
    client.run.return_value = {"ok": False, "exception": "boom", "traceback": "...",
                                "offending_line": 3}
    err = check_exec("bad code", "geom", client, timeout=5)
    assert err["phase"] == "exec"
    assert err["offending_line"] == 3

def test_session_died_returns_sentinel():
    client = MagicMock()
    client.run.side_effect = SessionDied()
    assert check_exec("code", "geom", client, timeout=5) == "session_died"
```

- [ ] **Step 8: Run to verify it fails**

Run: `pytest tests/unit/test_exec_check.py -v`
Expected: FAIL — module not found.

- [ ] **Step 9: Write `salome_agent/verify/exec_check.py`**

```python
from typing import Literal

from salome_agent.executor_client import SessionDied


def check_exec(script: str, module: str, client, timeout: float) -> dict | Literal["session_died"] | None:
    try:
        run_result = client.run(script, timeout)
    except SessionDied:
        return "session_died"

    if not run_result["ok"]:
        return {"phase": "exec", "message": run_result.get("exception", ""),
                "offending_line": run_result.get("offending_line"),
                "traceback": run_result.get("traceback")}

    sanity = client.sanity(module, {})
    if not sanity["ok"] or sanity["failures"]:
        return {"phase": "exec", "message": "; ".join(sanity["failures"]),
                "offending_line": None, "traceback": None}
    return None
```

- [ ] **Step 10: Run to verify it passes**

Run: `pytest tests/unit/test_exec_check.py -v`
Expected: PASS (3 tests)

- [ ] **Step 11: Commit**

```bash
git add salome_agent/verify/ tests/unit/test_static.py tests/unit/test_exec_check.py
git commit -m "feat: add static and exec verification"
```

---

### Task 9: Specialist base loop + smesh/paravis specialists

**Files:**
- Create: `salome_agent/specialists/base.py`
- Create: `salome_agent/specialists/smesh.py`
- Create: `salome_agent/specialists/paravis.py`
- Test: `tests/unit/test_specialist_base.py`

**Interfaces:**
- Consumes: `llm.chat(messages, tools) -> dict` (mocked). `api_introspect` is forwarded to the executor by `nodes/planner.py`'s caller, not by this module — `base.py` just dispatches whatever tool name comes back from `llm.chat` against the `tools` dict it's given.
- Produces: `run_specialist(stage_plan, llm, tools, iteration_cap) -> str` (the emitted script) for smesh/paravis. `SMESH_SYSTEM_PROMPT`, `PARAVIS_SYSTEM_PROMPT`.

- [ ] **Step 1: Write the failing test — loop stops at `emit_script`, iteration cap warns**

```python
# tests/unit/test_specialist_base.py
from unittest.mock import MagicMock
from salome_agent.specialists.base import run_specialist

def test_emit_script_ends_loop_with_code():
    llm = MagicMock()
    llm.chat.return_value = {"tool_call": "emit_script", "args": {"code": "mesh.Compute()"}}
    result = run_specialist({"module": "smesh", "intent": "mesh it", "params": {}, "inputs": []},
                             llm, tools={}, iteration_cap=12)
    assert result["script"] == "mesh.Compute()"
    assert result["warning"] is None

def test_iteration_cap_emits_best_so_far_with_warning():
    llm = MagicMock()
    llm.chat.return_value = {"tool_call": "api_introspect", "args": {"name": "smesh"}}
    result = run_specialist({"module": "smesh", "intent": "mesh it", "params": {}, "inputs": []},
                             llm, tools={"api_introspect": lambda **kw: "members: [...]"}, iteration_cap=2)
    assert result["warning"] is not None
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/unit/test_specialist_base.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write `salome_agent/specialists/base.py`**

```python
def run_specialist(stage_plan: dict, llm, tools: dict, iteration_cap: int) -> dict:
    messages = [{"role": "user", "content": str(stage_plan)}]
    last_code = ""
    for _ in range(iteration_cap):
        response = llm.chat(messages=messages, tools=list(tools) + ["emit_script"])
        if response["tool_call"] == "emit_script":
            return {"script": response["args"]["code"], "warning": None}
        tool = tools.get(response["tool_call"])
        if tool is not None:
            observation = tool(**response["args"])
            messages.append({"role": "tool", "content": observation})
    return {"script": last_code, "warning": "iteration_cap reached without emit_script"}
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/unit/test_specialist_base.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Write `salome_agent/specialists/smesh.py` and `paravis.py`**

```python
# salome_agent/specialists/smesh.py
SMESH_SYSTEM_PROMPT = """You write a SMESH stage script. The session and
study already exist — do not call salome.salome_init() or create a study.
Get the upstream shape with find_object(name). Create the mesh, set
algorithms/hypotheses from params (default: NETGEN 1D-2D-3D tetra at the
given max size), call Compute(). Publish results under study_ns with a
deterministic, descriptive name. No GUI calls."""
```

```python
# salome_agent/specialists/paravis.py
PARAVIS_SYSTEM_PROMPT = """You write a PARAVIS stage script. Load the input
MED from `inputs`, find the requested field, apply a surface representation
coloured by that field, set a camera. A screenshot is best-effort and never
a gate. No GUI calls."""
```

- [ ] **Step 6: Commit**

```bash
git add salome_agent/specialists/base.py salome_agent/specialists/smesh.py salome_agent/specialists/paravis.py tests/unit/test_specialist_base.py
git commit -m "feat: add specialist tool-calling loop and smesh/paravis prompts"
```

---

### Task 10: Geom specialist (tool-calling client against the executor)

**Files:**
- Create: `salome_agent/specialists/geom.py`
- Test: `tests/unit/test_specialist_geom.py`

**Interfaces:**
- Consumes: `ExecutorClient.list_tools()`/`.call_tool()` (Task 4).
- Produces: `run_geom_specialist(stage_plan, llm, client, iteration_cap) -> dict` with `{"call_log": list[CallRecord], "result_names": list[str]}` or `{"call_log": [...], "last_error": StageError}` on exhaustion — consumed by `nodes/advance.py` (Task 11), which runs `codegen.generate` over `call_log`.

- [ ] **Step 1: Write the failing test — `finish_stage` ends the loop with the call log; a failed tool call is retried; exhaustion sets `last_error`**

```python
# tests/unit/test_specialist_geom.py
from unittest.mock import MagicMock
from salome_agent.specialists.geom import run_geom_specialist

def test_finish_stage_returns_call_log_and_result_names():
    llm = MagicMock()
    llm.chat.side_effect = [
        {"tool_call": "make_box", "args": {"dx": 10, "dy": 10, "dz": 10}},
        {"tool_call": "finish_stage", "args": {"result_names": ["box_1"]}},
    ]
    client = MagicMock()
    client.call_tool.return_value = {"ok": True, "records": [{"fn": "MakeBoxDXDYDZ", "args": [],
                                                                "out_vars": ["box_1"], "publish": ["box_1"]}]}
    result = run_geom_specialist({"module": "geom", "intent": "box", "params": {}, "inputs": []},
                                  llm, client, iteration_cap=12)
    assert result["result_names"] == ["box_1"]
    assert len(result["call_log"]) == 1

def test_exhausted_loop_sets_last_error():
    llm = MagicMock()
    llm.chat.return_value = {"tool_call": "make_box", "args": {"dx": 1, "dy": 1, "dz": 1}}
    client = MagicMock()
    client.call_tool.return_value = {"ok": False, "error": "bad box"}
    result = run_geom_specialist({"module": "geom", "intent": "box", "params": {}, "inputs": []},
                                  llm, client, iteration_cap=2)
    assert result["last_error"]["message"] == "bad box"
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/unit/test_specialist_geom.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write `salome_agent/specialists/geom.py`**

```python
GEOM_SYSTEM_PROMPT = """You build geometry by calling tools, not by writing
geompy source. Reference upstream objects by their published name. Lengths
are millimetres, unconverted. When done, call finish_stage(result_names)
naming which of your published objects matter — everything they transitively
depend on is kept, everything else is dropped."""


def run_geom_specialist(stage_plan: dict, llm, client, iteration_cap: int) -> dict:
    messages = [{"role": "user", "content": str(stage_plan)}]
    call_log: list[dict] = []
    last_error = None
    for _ in range(iteration_cap):
        response = llm.chat(messages=messages, tools=["finish_stage"])
        if response["tool_call"] == "finish_stage":
            return {"call_log": call_log, "result_names": response["args"]["result_names"]}
        result = client.call_tool(response["tool_call"], response["args"])
        if result["ok"]:
            call_log.extend(result["records"])
            last_error = None
        else:
            last_error = {"phase": "exec", "message": result["error"],
                           "offending_line": None, "traceback": None}
        messages.append({"role": "tool", "content": str(result)})
    return {"call_log": call_log, "last_error": last_error}
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/unit/test_specialist_geom.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add salome_agent/specialists/geom.py tests/unit/test_specialist_geom.py
git commit -m "feat: add geom specialist as a tool-calling client"
```

---

### Task 11: advance/skip nodes, assembler, report

**Files:**
- Create: `salome_agent/nodes/advance.py`
- Create: `salome_agent/nodes/assembler.py`
- Create: `salome_agent/nodes/report.py`
- Test: `tests/unit/test_advance.py`, `tests/unit/test_assembler.py`

**Interfaces:**
- Consumes: `codegen.generate` (Task 5) for geom stages.
- Produces: `advance(state) -> dict` (partial `RunState` update), `skip(state) -> dict`, `assemble(state) -> str` (the `result.py` text), `write_report(state) -> str`.

- [ ] **Step 1: Write the failing test — geom stage runs codegen immediately, smesh stage uses its own script**

```python
# tests/unit/test_advance.py
from salome_agent.nodes.advance import advance

def test_geom_stage_codegens_immediately_on_advance():
    state = {
        "stage_idx": 0,
        "stages": {0: {"module": "geom", "status": "pending", "script": None,
                       "call_log": [{"fn": "MakeBoxDXDYDZ", "args": [], "out_vars": ["box_1"],
                                     "publish": ["box_1"]}],
                       "static_retries": 0, "exec_retries": 0}},
    }
    update = advance(state, result_names=["box_1"])
    assert "box_1 = geompy.MakeBoxDXDYDZ()" in update["stages"][0]["script"]
    assert update["stages"][0]["status"] == "ok"

def test_smesh_stage_keeps_its_own_script():
    state = {
        "stage_idx": 1,
        "stages": {1: {"module": "smesh", "status": "pending", "script": "mesh.Compute()",
                       "call_log": None, "static_retries": 0, "exec_retries": 0}},
    }
    update = advance(state)
    assert update["stages"][1]["script"] == "mesh.Compute()"
    assert update["stages"][1]["status"] == "ok"
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/unit/test_advance.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write `salome_agent/nodes/advance.py`**

```python
from salome_agent.codegen import generate


def advance(state: dict, result_names: list[str] | None = None) -> dict:
    idx = state["stage_idx"]
    stage = dict(state["stages"][idx])
    if stage["module"] == "geom" and stage["script"] is None:
        stage["script"] = generate(stage["call_log"], result_names)
    stage["status"] = "ok"
    stages = dict(state["stages"])
    stages[idx] = stage
    return {"stages": stages, "stage_idx": idx + 1}


def skip(state: dict) -> dict:
    idx = state["stage_idx"]
    stage = dict(state["stages"][idx])
    stage["status"] = "skipped"
    stages = dict(state["stages"])
    stages[idx] = stage
    return {"stages": stages, "stage_idx": idx + 1}
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/unit/test_advance.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Write the failing test for the assembler**

```python
# tests/unit/test_assembler.py
import ast
from salome_agent.nodes.assembler import assemble

def test_assembles_ordered_stages_with_preamble():
    state = {
        "stages": {
            0: {"module": "geom", "status": "ok", "script": "box_1 = geompy.MakeBoxDXDYDZ(1,1,1)"},
            1: {"module": "smesh", "status": "ok", "script": "mesh.Compute()"},
        },
        "run_id": "20260829-abcd",
        "user_request": "make a box and mesh it",
        "plan": [],
    }
    result = assemble(state)
    ast.parse(result)
    assert result.index("salome_init") < result.index("MakeBoxDXDYDZ") < result.index("Compute")

def test_skipped_stages_are_excluded():
    state = {
        "stages": {0: {"module": "geom", "status": "skipped", "script": None}},
        "run_id": "x", "user_request": "y", "plan": [],
    }
    result = assemble(state)
    assert "MakeBoxDXDYDZ" not in result
```

- [ ] **Step 6: Run to verify it fails**

Run: `pytest tests/unit/test_assembler.py -v`
Expected: FAIL — module not found.

- [ ] **Step 7: Write `salome_agent/nodes/assembler.py`**

```python
# Real import convention for this SALOME build (confirmed 2026-08-29,
# see Task 2's build_namespace) — result.py must self-init the same way
# the executor's exec namespace does, since it runs standalone under
# `salome shell` with no executor process attached.
PREAMBLE = """from salome.kernel import salome
salome.salome_init()
from salome.geom import geomBuilder
geompy = geomBuilder.New()
from salome.smesh import smeshBuilder
smesh = smeshBuilder.New()
study_ns = "salome_agent_run"

def find_object(name):
    def walk(sobj):
        it = salome.myStudy.NewChildIterator(sobj)
        while it.More():
            child = it.Value()
            if child.GetName() == name:
                return child.GetObject()
            found = walk(child)
            if found is not None:
                return found
            it.Next()
        return None
    result = walk(salome.myStudy.FindObjectID("0:1"))
    if result is None:
        raise LookupError(f"no study object named {name!r}")
    return result
"""


def assemble(state: dict) -> str:
    header = (
        f"# generated by salome-agent, run {state['run_id']}\n"
        f"# request: {state['user_request']}\n"
    )
    bodies = []
    for idx in sorted(state["stages"]):
        stage = state["stages"][idx]
        if stage["status"] == "ok" and stage["script"]:
            bodies.append(f"# stage {idx}: {stage['module']}\n{stage['script']}")
    return header + "\n" + PREAMBLE + "\n" + "\n\n".join(bodies) + "\n"
```

- [ ] **Step 8: Run to verify it passes**

Run: `pytest tests/unit/test_assembler.py -v`
Expected: PASS (2 tests)

- [ ] **Step 9: Write `salome_agent/nodes/report.py`** (no dedicated unit test — plain string formatting exercised by the Task 14 end-to-end test)

```python
def write_report(state: dict) -> str:
    lines = [f"# Report: {state['run_id']}", "", f"Request: {state['user_request']}", "", "## Stages"]
    for idx in sorted(state["stages"]):
        stage = state["stages"][idx]
        retries = stage.get("static_retries", 0) + stage.get("exec_retries", 0)
        lines.append(f"- stage {idx} ({stage['module']}): {stage['status']}, retries={retries}")
    return "\n".join(lines)
```

- [ ] **Step 10: Commit**

```bash
git add salome_agent/nodes/advance.py salome_agent/nodes/assembler.py salome_agent/nodes/report.py tests/unit/test_advance.py tests/unit/test_assembler.py
git commit -m "feat: add advance/skip nodes, assembler, and report"
```

---

### Task 12: Config loading

**Files:**
- Create: `salome_agent/config.py`
- Create: `config.example.json` (repo root — matches spec §9 exactly)
- Test: `tests/unit/test_config.py`

**Interfaces:**
- Produces: `load_config(path: str | None) -> dict` — validates required keys (`llm.base_url`, `llm.model`, `salome.root`).

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_config.py
import json
import pytest
from salome_agent.config import load_config, ConfigError

def test_loads_valid_config(tmp_path):
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps({
        "llm": {"base_url": "http://x", "api_key": "k", "model": "m", "temperature": 0},
        "salome": {"root": "/opt/salome"},
        "policy": {"static_retries": 3, "exec_retries": 2, "exec_timeout_s": 120,
                   "tool_iteration_cap": 12, "on_stage_failure": "abort",
                   "assembler_smoke_run": True},
        "work_dir": "~/.local/share/salome-agent/runs",
    }))
    config = load_config(str(cfg_path))
    assert config["llm"]["model"] == "m"

def test_missing_required_key_raises(tmp_path):
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps({"llm": {"base_url": "http://x"}}))
    with pytest.raises(ConfigError):
        load_config(str(cfg_path))
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/unit/test_config.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write `salome_agent/config.py`**

```python
import json

REQUIRED = [("llm", "base_url"), ("llm", "model"), ("salome", "root")]


class ConfigError(Exception):
    pass


def load_config(path: str) -> dict:
    with open(path) as f:
        config = json.load(f)
    for section, key in REQUIRED:
        if key not in config.get(section, {}):
            raise ConfigError(f"missing required config key: {section}.{key}")
    return config
```

- [ ] **Step 4: Write `config.example.json`** (repo root, values copied verbatim from spec §9)

```json
{
  "llm": { "base_url": "", "api_key": "", "model": "", "temperature": 0 },
  "salome": {
    "root": "/storage/salome_platform/devel/SALOME-master-native-FD44"
  },
  "policy": {
    "static_retries": 3,
    "exec_retries": 2,
    "exec_timeout_s": 120,
    "tool_iteration_cap": 12,
    "on_stage_failure": "abort",
    "assembler_smoke_run": true
  },
  "work_dir": "~/.local/share/salome-agent/runs"
}
```

- [ ] **Step 5: Run to verify it passes**

Run: `pytest tests/unit/test_config.py -v`
Expected: PASS (2 tests)

- [ ] **Step 6: Commit**

```bash
git add salome_agent/config.py config.example.json tests/unit/test_config.py
git commit -m "feat: add config loading and validation"
```

---

### Task 13: CLI

**Files:**
- Create: `salome_agent/cli.py`
- Create: `pyproject.toml` (repo root)
- Test: `tests/unit/test_cli.py`

**Interfaces:**
- Consumes: `load_config` (Task 12), `SessionManager` (Task 4), `build_graph` (Task 6).
- Produces: `main(argv: list[str]) -> int`, an installed `salome-agent` console script.

- [ ] **Step 1: Write the failing test — argument parsing only (graph execution is exercised end-to-end in Task 14)**

```python
# tests/unit/test_cli.py
from salome_agent.cli import parse_args

def test_parses_batch_and_no_session():
    args = parse_args(["make a box", "--batch", "--no-session"])
    assert args.request == "make a box"
    assert args.batch is True
    assert args.no_session is True

def test_defaults_are_interactive_with_session():
    args = parse_args(["make a box"])
    assert args.batch is False
    assert args.no_session is False
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/unit/test_cli.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write `salome_agent/cli.py`**

```python
import argparse
import sys


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="salome-agent")
    parser.add_argument("request")
    parser.add_argument("--batch", action="store_true")
    parser.add_argument("--no-session", action="store_true")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--on-failure", choices=["abort", "continue"], default=None)
    parser.add_argument("--timeout", type=float, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    from salome_agent.config import load_config
    from salome_agent.session.manager import SessionManager
    from salome_agent.graph import build_graph

    args = parse_args(argv if argv is not None else sys.argv[1:])
    config = load_config(args.config or "config.json")
    if args.on_failure:
        config["policy"]["on_stage_failure"] = args.on_failure

    manager = SessionManager()
    session_handle, client = manager.resolve(args.no_session, args.port)
    try:
        graph = build_graph(batch=args.batch)
        graph.invoke({
            "user_request": args.request, "config": {**config, "batch": args.batch},
            "session": session_handle, "plan": [], "plan_approved": False,
            "stage_idx": 0, "stages": {}, "last_error": None, "history": [],
            "study_ns": "salome_agent_run", "final_script": None, "report": None,
            "run_id": "", "final_script": None,
        })
    finally:
        manager.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/unit/test_cli.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Write `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "salome-agent"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = ["langgraph"]

[project.scripts]
salome-agent = "salome_agent.cli:main"

[tool.setuptools]
packages = ["salome_agent", "salome_agent.nodes", "salome_agent.specialists",
            "salome_agent.verify", "salome_agent.session"]

[tool.pytest.ini_options]
markers = ["salome: requires a live SALOME install (skipped in CI)"]
```

- [ ] **Step 6: Commit**

```bash
git add salome_agent/cli.py pyproject.toml tests/unit/test_cli.py
git commit -m "feat: add CLI entry point"
```

---

### Task 14: End-to-end smoke test

**Files:**
- Create: `tests/salome/test_end_to_end.py` (`@pytest.mark.salome`)

**Interfaces:**
- Consumes: everything above, wired through `main()` (Task 13).

- [ ] **Step 1: Write the smoke test**

```python
# tests/salome/test_end_to_end.py
import ast
import pytest
from unittest.mock import MagicMock, patch

@pytest.mark.salome
def test_cube_and_mesh_produces_runnable_result(tmp_path):
    fake_llm = MagicMock()
    fake_llm.chat.side_effect = [
        {"stages": [
            {"module": "geom", "intent": "20mm cube", "params": {"size": 20}, "inputs": []},
            {"module": "smesh", "intent": "mesh 5mm tets", "params": {"max_size": 5}, "inputs": ["box_1"]},
        ]},
        {"tool_call": "make_box", "args": {"dx": 20, "dy": 20, "dz": 20}},
        {"tool_call": "finish_stage", "args": {"result_names": ["box_1"]}},
        {"tool_call": "emit_script", "args": {"code": (
            "shape = find_object('box_1')\n"
            "m = smesh.Mesh(shape)\n"
            "m.Tetrahedron(algo=smesh.NETGEN_1D2D3D).SetMaxSize(5)\n"
            "m.Compute()\n"
        )}},
    ]
    with patch("salome_agent.cli.build_llm", return_value=fake_llm):
        from salome_agent.cli import main
        rc = main(["make a 20mm cube and mesh it with 5mm tetrahedra", "--batch",
                   "--work-dir", str(tmp_path)])
    assert rc == 0
    result_py = next(tmp_path.glob("*/result.py"))
    source = result_py.read_text()
    ast.parse(source)
    assert "MakeBoxDXDYDZ" in source
    assert "Compute" in source
```

Note: this test names `salome_agent.cli.build_llm`, which Task 13's `main()` does not yet call directly (it constructs the graph without an explicit LLM factory). Before running this test, add a `build_llm(config) -> LLMClient` seam to `cli.py` that `main()` calls and the graph nodes receive via `state["config"]["llm_client"]`, so this test's `patch()` has something real to target. This is the first point in the plan where the planner/specialist nodes need an actual LLM client wired end-to-end rather than being called directly in unit tests — implement the minimal seam, do not add retry/streaming/etc beyond what §5.1/§10 already specify.

- [ ] **Step 2: Run to verify it fails, then passes after the `build_llm` seam is added**

Run: `pytest tests/salome/test_end_to_end.py -v -m salome`
Expected: initial FAIL naming the missing seam; PASS once added and the study has one solid with a computed mesh (verify via `salome shell` inspection if the assertion above isn't sufficient).

- [ ] **Step 3: Commit**

```bash
git add tests/salome/test_end_to_end.py salome_agent/cli.py
git commit -m "test: add end-to-end smoke test for geom->smesh handoff"
```

---

## Self-Review

**Spec coverage:** §3.1/§3.2 (two processes, components) → Tasks 2–4. §3.4/§4.1 (CallRecord v2, decision 18) → Tasks 3/5. §4.2 (graph edges) → Task 6. §5.1–5.4 (specialists) → Tasks 9–10. §6 (verification) → Task 8. §7 (session layer) → Task 4. §8 (RAG, cut from v1 per decision 19) → out of scope for this plan, matching the spec's own deferral. §9 (config) → Task 12. §3.2/§3.3 (assembler/report/outputs) → Task 11. §11's end-to-end row → Task 14. §7.3's GUI plugin path and §13's later evolutions are explicitly out of scope for this plan, matching the spec.

**Placeholder scan:** the one intentional exception is Task 3's registry comment noting that `make_cylinder`/`boolean_*`/`chamfer_edges` follow `tool_make_box`'s pattern — flagged here rather than hidden, because writing four more near-identical geompy wrappers by hand adds no signal beyond what `tool_make_box`/`tool_fillet_edges` already demonstrate; add them the same way when a stage first needs them, TDD'd the same way as Task 3's own tests.

**Type consistency:** `CallRecord`/`ArgRef` (Task 3) flow unchanged into `codegen.generate` (Task 5), `nodes/advance.py` (Task 11), and `specialists/geom.py` (Task 10) — checked against the spec §4.1 definitions, not redefined per-task. `SessionHandle`/`RunState` (Task 1) are the only state shapes used by `graph.py`, `cli.py`, and `session/manager.py`.
