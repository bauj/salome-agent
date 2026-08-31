# GUI Plugin Server Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the CORBA attach path to a running SALOME with a server started
inside the GUI process by a Tools ▸ Plugins entry; the agent connects to it over
loopback TCP.

**Architecture:** A new `agent_server/` package holds a transport-agnostic `Core`
(command dispatch) sitting behind a `Backend` protocol with two implementations —
`GuiBackend` (marshals every SALOME call onto the Qt main thread via a
cross-thread Qt signal) and `HeadlessBackend` (direct calls, isolated study).
`wire.py` is a loopback-TCP + token listener. A GUI plugin (`launch()`) starts
the server on a daemon thread; `python -m agent_server` starts the identical
server headless for CI. Discovery is a handshake file `~/.salome_agent/server.json`.
The agent side loses `salome_agent/session/` entirely and gains `server_conn.py`.

**Tech Stack:** Python 3.10+, stdlib `socket` / `threading` / `hmac` / `secrets`,
LangGraph (unchanged), PySide2 (only inside the SALOME GUI interpreter — bound via
`SalomePyQt.usePySide()`), pytest.

**Spec:** `docs/superpowers/specs/2026-08-31-gui-plugin-server-design.md` — read it
alongside this plan.

## Global Constraints

- **This build is PySide2, not PyQt5.** Bind Qt symbols through the
  `SalomePyQt.usePySide()` branch, exactly as `salome_pluginsmanager.py:91` does.
  Never `import PyQt5` unconditionally.
- **`agent_server/core.py` and `agent_server/backend.py` MUST import with no
  PySide2 / SalomePyQt / SALOME present** — they are unit-tested in the plain
  dev venv. Only `gui_backend.py`, `gui_plugin.py`, `namespace.py` may import
  SALOME/Qt.
- **`executor/` is not a declared package.** `import executor.geom_tools`
  resolves only via repo-root-on-`sys.path`. `agent_server/core.py` imports it
  at **module top level** (before anything `salome_init`-adjacent — `salome_init`
  mutates `sys.path`).
- **Wire protocol unchanged:** one JSON object per line, `\n`-delimited, request
  carries `id`, response echoes it. New command: `hello`.
- **Name prefix (GUI mode only):** every study name the geom side publishes is
  `sa_<run_id>_`-prefixed, where `run_id` is the agent's 12-hex run id
  (`cli.py:51`). Headless prefix is `""`.
- **Timeout is terminal.** A stage timeout poisons the server (both backends);
  every subsequent command including `hello` returns
  `{"ok": false, "fatal": true, "error": ...}`.
- **TDD, frequent commits.** Every task ends with a green test run and a commit.
- Branch: `feat/2-gui-plugin-server` (already checked out).
- Run unit tests with `python -m pytest tests/unit -q` (from repo root, venv
  active). SALOME-marked tests: `python -m pytest tests/salome -q` under
  `scripts/start-headless-server.sh`-style env, skipped without `SALOME_ROOT`.

---

## File Structure

**New — `agent_server/` package:**
| File | Responsibility |
|---|---|
| `agent_server/__init__.py` | Empty. |
| `agent_server/handshake.py` | `PROTO`, `path()`, `write()`, `read()`, `remove()` for `~/.salome_agent/server.json`. Pure stdlib. |
| `agent_server/namespace.py` | `build_namespace(live: bool) -> dict`. Imports SALOME. |
| `agent_server/backend.py` | `Backend` protocol + `HeadlessBackend`. No Qt, no SALOME import at module level (takes `build_namespace` injected). |
| `agent_server/core.py` | `Core` — command dispatch + poison gate. Imports `executor.geom_tools` at top level. No Qt. |
| `agent_server/wire.py` | `serve(core, token, on_bound)` — loopback TCP listener, `hmac` token, framing, single-client, incumbent liveness. |
| `agent_server/gui_backend.py` | `MainThreadInvoker` (QObject) + `GuiBackend`. Qt via `usePySide()` branch. |
| `agent_server/gui_plugin.py` | `launch(context)` — plugin entry, module-level `_INVOKER`/`_SERVER_THREAD`/`_CORE`, restart, `aboutToQuit`. |
| `agent_server/__main__.py` | `main(argv)` — headless entry, `--once` default. |

**New — elsewhere:**
| File | Responsibility |
|---|---|
| `salome_agent/server_conn.py` | `ServerUnavailable`, `read_handshake()`, `connect(run_id)`. |
| `plugin/salome_agent_plugin.py` | `AddFunction('Launch SALOME agent server', ..., launch)`. |
| `scripts/install-plugin.sh` | Idempotent plugin installer. |
| `scripts/start-headless-server.sh` | `.venv`-stripping wrapper around `salome shell -- python -m agent_server --once`. |

**Modified:**
| File | Change |
|---|---|
| `pyproject.toml` | `packages`: drop `salome_agent.session`, add `agent_server`. |
| `executor/geom_tools.py` | `_next_name(executor, prefix)` honours `executor.name_prefix`; `reset_counters()`. |
| `salome_agent/executor_client.py` | `AF_INET` + `hello`; `SessionPoisoned`. |
| `salome_agent/verify/exec_check.py` | `"session_poisoned"` sentinel. |
| `salome_agent/specialists/geom.py` | catch `SessionDied`/`SessionPoisoned` → `{"fatal": ...}`. |
| `salome_agent/graph.py` | `_verify_exec_node` + `_run_stage_node` route `fatal` → `assemble`. |
| `salome_agent/state.py` | `SessionHandle` new shape; `RunState.fatal`. |
| `salome_agent/nodes/report.py` | render `fatal` + mode line. |
| `salome_agent/cli.py` | drop `--live-gui`/`--port`; `server_conn.connect`; exit 1 on `fatal`. |
| `tests/salome/conftest.py`, `test_executor.py`, `test_geom_tools.py`, `test_end_to_end.py`, `test_specialists_live.py` | launch headless server instead of `executor/executor.py`. |
| `README.md` | "GUI plugin setup" section; drop `--port`/`--live-gui`. |
| `docs/superpowers/specs/2026-08-29-salome-agent-design.md` | one-line "superseded" note near the executor description. |

**Deleted:** `salome_agent/session/` (whole package), `executor/executor.py`,
`tests/unit/test_session_manager.py`, `tests/unit/test_discovery.py`,
`tests/unit/test_executor_live.py`, `tests/salome/test_live_gui.py`.

---

## Task 1: Phase 0 spike — Qt marshal Gate A (manual, throwaway)

**Files:** none committed. Throwaway script under the scratchpad dir.

**Interfaces:** none. This task's only output is a go / no-go recorded in the
commit message of Task 2 and a note appended to the spec.

**Why:** the whole design rests on a cross-thread Qt signal firing from a bare
`threading.Thread` inside a plugin function. `QTimer.singleShot` provably does
not; `_Signal(object)` + `Qt.QueuedConnection` should. Confirm before building.

- [ ] **Step 1: Write the throwaway plugin**

Create `~/.config/salome/Plugins/salome_plugins.py` (or append to the user's
existing one) — ask the user to do this if the file is theirs:

```python
def _spike(context):
    import threading, time, gc
    import SalomePyQt
    if SalomePyQt.usePySide():
        from PySide2.QtCore import QObject, Signal, Qt
    else:
        from PyQt5.QtCore import QObject, pyqtSignal as Signal, Qt

    class Inv(QObject):
        sig = Signal(object)
        def __init__(self):
            super().__init__()
            self.sig.connect(self._run, Qt.QueuedConnection)
        def _run(self, job):
            job["thread"] = threading.current_thread().name
            job["done"].set()

    inv = Inv()
    globals()["_spike_inv"] = inv          # hold a ref
    gc.collect()

    def worker():
        job = {"done": threading.Event()}
        inv.sig.emit(job)
        ok = job["done"].wait(2)
        print(f"[SPIKE] fired={ok} ran_on={job.get('thread')} "
              f"main={threading.main_thread().name}")

    threading.Thread(target=worker, daemon=True).start()

import salome_pluginsmanager
salome_pluginsmanager.AddFunction('SPIKE marshal', 'spike', _spike)
```

- [ ] **Step 2: Run it**

Launch `"$SALOME_ROOT"/salome` (single-process, GUI). Tools ▸ Plugins ▸ SPIKE
marshal. Read the terminal.

Expected on success: `[SPIKE] fired=True ran_on=MainThread main=MainThread`.

- [ ] **Step 3: Gate B + C in the same session**

Extend `_spike`'s `_run` to also do:
```python
from salome.geom import geomBuilder
geompy = geomBuilder.New()
b = geompy.MakeBoxDXDYDZ(11, 22, 33)
geompy.addToStudy(b, "SPIKE_BOX")
import salome
salome.sg.updateObjBrowser()
```
Expected: `SPIKE_BOX` appears in the Object Browser **and** the 3D viewer with
no crash and no manual refresh.

- [ ] **Step 4: Record the verdict**

- Gate A pass → proceed with the `_Signal` design.
- Gate A fail → **stop**, escalate to the user: fallback is a `QThread`-based
  server (see spec Phase 0). Do not continue the plan as written.
- Note `SalomePyQt.usePySide()`'s value, and Gate B/C results, in a
  `## Phase 0 result` block appended to the spec file.

- [ ] **Step 5: Commit the spec note**

```bash
git add docs/superpowers/specs/2026-08-31-gui-plugin-server-design.md
git commit -m "docs: Phase 0 spike result (Qt marshal gate)"
```

Remove the throwaway plugin file / lines.

---

## Task 2: Package skeleton + `pyproject.toml`

**Files:**
- Create: `agent_server/__init__.py` (empty)
- Modify: `pyproject.toml:18-19`
- Test: `tests/unit/test_agent_server_import.py`

**Interfaces:**
- Produces: the importable `agent_server` package.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_agent_server_import.py
import importlib
import sys


def test_core_and_backend_import_without_qt(monkeypatch):
    # Simulate the dev venv: no PySide2, no SalomePyQt.
    for mod in list(sys.modules):
        if mod.startswith(("PySide2", "SalomePyQt", "agent_server")):
            monkeypatch.delitem(sys.modules, mod, raising=False)
    monkeypatch.setitem(sys.modules, "PySide2", None)
    monkeypatch.setitem(sys.modules, "SalomePyQt", None)

    importlib.import_module("agent_server.core")
    importlib.import_module("agent_server.backend")
    importlib.import_module("agent_server.handshake")
```

- [ ] **Step 2: Run it — expect fail**

Run: `python -m pytest tests/unit/test_agent_server_import.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'agent_server'` (and the
submodules do not exist yet).

- [ ] **Step 3: Create the package + fix pyproject**

`agent_server/__init__.py`: empty file.

`pyproject.toml` lines 18-19 → :
```toml
[tool.setuptools]
packages = ["salome_agent", "salome_agent.nodes", "salome_agent.specialists",
            "salome_agent.verify", "agent_server"]
```

(The test still fails until Tasks 3-5 create `core`/`backend`/`handshake` — that
is expected; this step only unblocks the package import. Mark the test
`@pytest.mark.xfail(reason="core/backend/handshake land in tasks 3-5", strict=False)`
for now and remove the marker in Task 5.)

- [ ] **Step 4: Run — expect xfail**

Run: `python -m pytest tests/unit/test_agent_server_import.py -q`
Expected: XFAIL.

- [ ] **Step 5: Commit**

```bash
git add agent_server/__init__.py pyproject.toml tests/unit/test_agent_server_import.py
git commit -m "chore: agent_server package skeleton; drop salome_agent.session from packages"
```

---

## Task 3: `agent_server/handshake.py`

**Files:**
- Create: `agent_server/handshake.py`
- Test: `tests/unit/test_handshake.py`

**Interfaces:**
- Produces:
  - `PROTO: int = 1`
  - `path() -> pathlib.Path` — `~/.salome_agent/server.json` (respects `$HOME`).
  - `write(*, port: int, token: str, mode: str, log_path: str, salome_port: int | None = None) -> None`
  - `read() -> dict | None` — `None` on missing file, bad JSON, or `proto != PROTO`.
  - `remove() -> None` — unlink only if the file's `pid == os.getpid()`; silent if absent.

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_handshake.py
import json
import os

import pytest

from agent_server import handshake


@pytest.fixture(autouse=True)
def _home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(os.path, "expanduser",
                        lambda p: p.replace("~", str(tmp_path), 1))
    yield tmp_path


def test_write_then_read_roundtrip():
    handshake.write(port=54123, token="abc", mode="gui",
                    log_path="/x/server.log")
    got = handshake.read()
    assert got["port"] == 54123
    assert got["token"] == "abc"
    assert got["mode"] == "gui"
    assert got["proto"] == handshake.PROTO
    assert got["pid"] == os.getpid()
    assert "pid_start" in got


def test_read_missing_returns_none():
    assert handshake.read() is None


def test_read_proto_mismatch_returns_none(_home):
    p = handshake.path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"proto": 999, "port": 1}))
    assert handshake.read() is None


def test_remove_only_unlinks_own_pid(_home):
    p = handshake.path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"proto": handshake.PROTO, "pid": os.getpid() + 999999}))
    handshake.remove()
    assert p.exists()                       # not ours -> left alone
    p.write_text(json.dumps({"proto": handshake.PROTO, "pid": os.getpid()}))
    handshake.remove()
    assert not p.exists()


def test_dir_and_file_permissions():
    handshake.write(port=1, token="t", mode="headless", log_path="/x")
    assert oct(handshake.path().parent.stat().st_mode)[-3:] == "700"
    assert oct(handshake.path().stat().st_mode)[-3:] == "600"
```

- [ ] **Step 2: Run — expect fail**

Run: `python -m pytest tests/unit/test_handshake.py -q`
Expected: FAIL — `ModuleNotFoundError: agent_server.handshake`.

- [ ] **Step 3: Implement**

```python
# agent_server/handshake.py
import json
import os
import time
from pathlib import Path

PROTO = 1


def path() -> Path:
    return Path(os.path.expanduser("~/.salome_agent/server.json"))


def _pid_start(pid: int) -> int | None:
    try:
        with open(f"/proc/{pid}/stat") as f:
            return int(f.read().rsplit(")", 1)[1].split()[19])
    except (OSError, IndexError, ValueError):
        return None


def write(*, port: int, token: str, mode: str, log_path: str,
          salome_port: int | None = None) -> None:
    p = path()
    p.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(p.parent, 0o700)
    pid = os.getpid()
    data = {
        "proto": PROTO, "host": "127.0.0.1", "port": port, "token": token,
        "pid": pid, "pid_start": _pid_start(pid), "mode": mode,
        "salome_port": salome_port, "log_path": log_path,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.chmod(tmp, 0o600)
    os.replace(tmp, p)


def read() -> dict | None:
    try:
        data = json.loads(path().read_text())
    except (OSError, ValueError):
        return None
    if data.get("proto") != PROTO:
        return None
    return data


def remove() -> None:
    try:
        data = json.loads(path().read_text())
    except (OSError, ValueError):
        return
    if data.get("pid") == os.getpid():
        path().unlink(missing_ok=True)
```

- [ ] **Step 4: Run — expect pass**

Run: `python -m pytest tests/unit/test_handshake.py -q`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add agent_server/handshake.py tests/unit/test_handshake.py
git commit -m "feat: agent_server.handshake — server.json read/write/remove"
```

---

## Task 4: `executor/geom_tools.py` — name-prefix support

**Files:**
- Modify: `executor/geom_tools.py:16-21` (`_counters`, `_next_name`), and the two
  `_next_name("box")` / `_next_name("fillet")` call sites (lines 45, 78).
- Test: `tests/unit/test_geom_tools_naming.py`

**Interfaces:**
- Consumes: `executor` objects now carry `.name_prefix: str` (added on the
  `Backend` in Task 5).
- Produces:
  - `geom_tools.reset_counters() -> None`
  - `geom_tools._next_name(executor, prefix: str) -> str` — returns
    `f"{executor.name_prefix}{prefix}_{n}"`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_geom_tools_naming.py
from types import SimpleNamespace

from executor import geom_tools


def test_next_name_applies_prefix_and_increments():
    geom_tools.reset_counters()
    ex = SimpleNamespace(name_prefix="sa_deadbeef_")
    assert geom_tools._next_name(ex, "box") == "sa_deadbeef_box_1"
    assert geom_tools._next_name(ex, "box") == "sa_deadbeef_box_2"
    assert geom_tools._next_name(ex, "fillet") == "sa_deadbeef_fillet_1"


def test_empty_prefix_matches_legacy_names():
    geom_tools.reset_counters()
    ex = SimpleNamespace(name_prefix="")
    assert geom_tools._next_name(ex, "box") == "box_1"


def test_reset_counters_is_independent_per_session():
    ex = SimpleNamespace(name_prefix="")
    geom_tools.reset_counters()
    geom_tools._next_name(ex, "box")
    geom_tools.reset_counters()
    assert geom_tools._next_name(ex, "box") == "box_1"
```

- [ ] **Step 2: Run — expect fail**

Run: `python -m pytest tests/unit/test_geom_tools_naming.py -q`
Expected: FAIL — `_next_name()` takes 1 positional arg; no `reset_counters`.

- [ ] **Step 3: Implement**

`executor/geom_tools.py` lines 16-21 →:
```python
_counters: dict[str, int] = {}


def reset_counters() -> None:
    _counters.clear()


def _next_name(executor, prefix: str) -> str:
    _counters[prefix] = _counters.get(prefix, 0) + 1
    return f"{executor.name_prefix}{prefix}_{_counters[prefix]}"
```

Line 45: `name = _next_name("box")` → `name = _next_name(executor, "box")`
Line 78: `name = _next_name("fillet")` → `name = _next_name(executor, "fillet")`

- [ ] **Step 4: Run — expect pass; also run the existing geom_tools + codegen unit tests**

Run: `python -m pytest tests/unit/test_geom_tools_naming.py tests/unit/test_codegen.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add executor/geom_tools.py tests/unit/test_geom_tools_naming.py
git commit -m "feat: geom_tools._next_name honours executor.name_prefix; reset_counters()"
```

---

## Task 5: `agent_server/backend.py` — `Backend` protocol + `HeadlessBackend`

**Files:**
- Create: `agent_server/backend.py`
- Test: `tests/unit/test_backend.py`
- Modify: `tests/unit/test_agent_server_import.py` (drop the xfail marker)

**Interfaces:**
- Consumes: `agent_server.handshake` (no); `executor.geom_tools.reset_counters`.
- Produces:
  - `class Backend(typing.Protocol)` with:
    - attrs `name_prefix: str`, `namespace: dict`, `geom_bindings: dict`,
      `accepted_scripts: list[str]`
    - `begin_session(run_id: str | None) -> None`
    - `end_session() -> None`
    - `run_code(code: str, timeout: float) -> dict` — the `_run_with_timeout`
      result dict (`ok`/`stdout`/`stderr`/`exception`/`traceback`/`offending_line`)
    - `reset(scripts: list[str]) -> dict` — `{"ok": bool, "failed_index": int | None}`
    - `refresh(module: str | None) -> None`
    - `find_object(name: str)` — delegates to `namespace["find_object"]`
    - `poisoned() -> bool`
  - `class HeadlessBackend` — `__init__(self, build_namespace=namespace.build_namespace)`.
    `name_prefix = ""`. Keeps the `_run_with_timeout` watchdog. A `timeout after Ns`
    result sets `self._poisoned = True`.
  - `_run_with_timeout(code, ns, timeout) -> dict` — **moved verbatim** from
    `executor/executor.py:25-55` into this module.

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_backend.py
import pytest

from agent_server.backend import HeadlessBackend


def _fake_ns():
    return {"geompy": object(), "smesh": object(), "find_object": lambda n: n}


def _mk(counter=[0]):
    counter[0] += 1
    return {**_fake_ns(), "tag": counter[0]}


def test_begin_session_headless_has_empty_prefix_and_resets_counters():
    from executor import geom_tools
    b = HeadlessBackend(build_namespace=lambda live: _fake_ns())
    geom_tools._counters["box"] = 7
    b.begin_session("abc123")
    assert b.name_prefix == ""
    assert geom_tools._counters == {}


def test_run_code_ok():
    b = HeadlessBackend(build_namespace=lambda live: _fake_ns())
    b.begin_session(None)
    r = b.run_code("x = 1 + 1", timeout=5)
    assert r["ok"] is True


def test_run_code_exception_reports_offending_line():
    b = HeadlessBackend(build_namespace=lambda live: _fake_ns())
    b.begin_session(None)
    r = b.run_code("raise ValueError('boom')", timeout=5)
    assert r["ok"] is False
    assert r["offending_line"] == 1


def test_timeout_poisons_the_backend():
    b = HeadlessBackend(build_namespace=lambda live: _fake_ns())
    b.begin_session(None)
    assert b.poisoned() is False
    r = b.run_code("while True: pass", timeout=1)
    assert r["ok"] is False and "timeout" in r["exception"].lower()
    assert b.poisoned() is True


def test_reset_rebuilds_namespace_and_replays():
    made = []
    b = HeadlessBackend(build_namespace=lambda live: made.append(1) or _fake_ns())
    b.begin_session(None)
    made.clear()
    r = b.reset(["a = 1", "b = a + 1"])
    assert r == {"ok": True, "failed_index": None}
    assert made == [1]                       # rebuilt once


def test_reset_reports_failed_index():
    b = HeadlessBackend(build_namespace=lambda live: _fake_ns())
    b.begin_session(None)
    r = b.reset(["a = 1", "a = undefined_name"])
    assert r == {"ok": False, "failed_index": 1}
```

- [ ] **Step 2: Run — expect fail**

Run: `python -m pytest tests/unit/test_backend.py -q`
Expected: FAIL — `ModuleNotFoundError: agent_server.backend`.

- [ ] **Step 3: Implement**

```python
# agent_server/backend.py
import sys
import threading
import traceback as tb_module
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from typing import Protocol

from executor import geom_tools


def _run_with_timeout(code: str, namespace: dict, timeout: float) -> dict:
    # Moved verbatim from executor/executor.py. The watchdog thread is a
    # best-effort bound only — a C-level Compute() may not respond. In the
    # persistent-server world a timeout must also poison the backend
    # (HeadlessBackend.run_code below) since the abandoned thread keeps
    # mutating `namespace`.
    out, err = StringIO(), StringIO()
    result = {"ok": False, "stdout": "", "stderr": "", "exception": None,
              "traceback": None, "offending_line": None}

    def target():
        try:
            with redirect_stdout(out), redirect_stderr(err):
                exec(compile(code, "<stage>", "exec"), namespace)
            result["ok"] = True
        except SyntaxError as e:
            result["exception"] = str(e)
            result["offending_line"] = e.lineno
        except BaseException as e:  # noqa: BLE001
            result["exception"] = str(e)
            result["traceback"] = "".join(tb_module.format_exc())
            for entry in tb_module.extract_tb(e.__traceback__):
                if entry.filename == "<stage>":
                    result["offending_line"] = entry.lineno

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        result["exception"] = f"timeout after {timeout}s"
    result["stdout"] = out.getvalue()
    result["stderr"] = err.getvalue()
    return result


class Backend(Protocol):
    name_prefix: str
    namespace: dict
    geom_bindings: dict
    accepted_scripts: list

    def begin_session(self, run_id: str | None) -> None: ...
    def end_session(self) -> None: ...
    def run_code(self, code: str, timeout: float) -> dict: ...
    def reset(self, scripts: list) -> dict: ...
    def refresh(self, module: str | None) -> None: ...
    def find_object(self, name: str): ...
    def poisoned(self) -> bool: ...


class HeadlessBackend:
    mode = "headless"

    def __init__(self, build_namespace=None):
        if build_namespace is None:
            from agent_server.namespace import build_namespace as bn
            build_namespace = bn
        self._build = build_namespace
        self.name_prefix = ""
        self.namespace = self._build(False)
        self.geom_bindings: dict = {}
        self.accepted_scripts: list = []
        self._poisoned = False

    def begin_session(self, run_id: str | None) -> None:
        geom_tools.reset_counters()
        self.geom_bindings.clear()
        self.accepted_scripts = []

    def end_session(self) -> None:
        pass

    def run_code(self, code: str, timeout: float) -> dict:
        r = _run_with_timeout(code, self.namespace, timeout)
        if not r["ok"] and (r["exception"] or "").startswith("timeout after"):
            self._poisoned = True
        return r

    def reset(self, scripts: list) -> dict:
        self.namespace = self._build(False)
        geom_tools.reset_counters()
        self.geom_bindings.clear()
        for idx, s in enumerate(scripts):
            r = _run_with_timeout(s, self.namespace, 120)
            if not r["ok"]:
                return {"ok": False, "failed_index": idx}
        self.accepted_scripts = list(scripts)
        return {"ok": True, "failed_index": None}

    def refresh(self, module: str | None) -> None:
        pass

    def find_object(self, name: str):
        return self.namespace["find_object"](name)

    def poisoned(self) -> bool:
        return self._poisoned
```

- [ ] **Step 4: Run — expect pass; drop the xfail from Task 2's test**

In `tests/unit/test_agent_server_import.py` remove any `xfail` marker (Task 2
step 3). Then:

Run: `python -m pytest tests/unit/test_backend.py tests/unit/test_agent_server_import.py -q`
Expected: `test_backend.py` PASS (7); `test_agent_server_import.py` still fails on
`agent_server.core` — that is Task 6. Leave a `@pytest.mark.xfail(reason="core lands in task 6")`
on it for now.

- [ ] **Step 5: Commit**

```bash
git add agent_server/backend.py tests/unit/test_backend.py tests/unit/test_agent_server_import.py
git commit -m "feat: agent_server.backend — Backend protocol + HeadlessBackend"
```

---

## Task 6: `agent_server/core.py` — command dispatch + poison gate

**Files:**
- Create: `agent_server/core.py`
- Test: `tests/unit/test_agent_server_core.py`
- Modify: `tests/unit/test_agent_server_import.py` (drop xfail)

**Interfaces:**
- Consumes: `Backend` (Task 5), `executor.geom_tools` (`list_tools`, `call_tool`).
- Produces:
  - `class Core` — `__init__(self, backend: Backend)`.
  - `Core.handle(req: dict) -> dict` — dispatches on `req["cmd"]`, always echoes
    `req.get("id")` into the response.
    - `hello`: `backend.begin_session(req.get("run_id"))`; returns
      `{"id", "ready": True, "proto": handshake.PROTO, "mode": backend.mode,
      "name_prefix": backend.name_prefix}`.
    - `run`: `{"id", **backend.run_code(req["code"], req["timeout"])}`; on
      `ok` calls `backend.refresh(req.get("module"))`.
    - `reset_and_replay`: `{"id", **backend.reset(req["scripts"])}`.
    - `introspect`: `{"id", <name>: {"members": dir(...)}}`.
    - `list_study`: walks `backend.namespace["salome"].myStudy` (moved from
      `executor.py:169-178`).
    - `sanity`: `{"id", "ok": True, "warnings": [], "failures": []}`.
    - `list_tools`: `{"id", "tools": geom_tools.list_tools()}`.
    - `call_tool`: `geom_tools.call_tool(backend, req["name"], req["args"])`; on
      `ok` calls `backend.refresh("GEOM")`. (`geom_tools` uses `backend.namespace`,
      `backend.geom_bindings`, `backend.name_prefix`.)
    - `shutdown`: `backend.end_session()`; `{"id", "ok": True}`.
    - unknown: `{"id", "ok": False, "exception": "unknown command ..."}`.
  - **Poison gate:** at the top of `handle`, if `backend.poisoned()`, return
    `{"id", "ok": False, "fatal": True, "error": "server poisoned by a prior timeout; restart the agent server"}`
    for **every** command including `hello`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_agent_server_core.py
from agent_server.core import Core


class FakeBackend:
    mode = "headless"

    def __init__(self):
        self.name_prefix = ""
        self.namespace = {"salome": None}
        self.geom_bindings = {}
        self.accepted_scripts = []
        self._poisoned = False
        self.began_with = None
        self.refreshed = []

    def begin_session(self, run_id):
        self.began_with = run_id
        self.name_prefix = f"sa_{run_id}_" if run_id else ""

    def end_session(self):
        pass

    def run_code(self, code, timeout):
        return {"ok": True, "stdout": "", "exception": None}

    def reset(self, scripts):
        return {"ok": True, "failed_index": None}

    def refresh(self, module):
        self.refreshed.append(module)

    def find_object(self, name):
        return name

    def poisoned(self):
        return self._poisoned


def test_hello_begins_session_and_returns_prefix():
    b = FakeBackend()
    resp = Core(b).handle({"id": 1, "cmd": "hello", "run_id": "deadbeef1234"})
    assert resp["ready"] is True
    assert resp["mode"] == "headless"
    assert resp["name_prefix"] == "sa_deadbeef1234_"
    assert b.began_with == "deadbeef1234"
    assert resp["id"] == 1


def test_run_ok_triggers_refresh():
    b = FakeBackend()
    Core(b).handle({"id": 2, "cmd": "run", "code": "x=1", "timeout": 5,
                    "module": "GEOM"})
    assert b.refreshed == ["GEOM"]


def test_poison_gate_refuses_every_command_including_hello():
    b = FakeBackend()
    b._poisoned = True
    core = Core(b)
    for cmd in ("hello", "run", "list_tools", "sanity"):
        resp = core.handle({"id": 9, "cmd": cmd})
        assert resp["fatal"] is True
        assert resp["ok"] is False


def test_unknown_command():
    resp = Core(FakeBackend()).handle({"id": 3, "cmd": "frobnicate"})
    assert resp["ok"] is False
    assert "unknown" in resp["exception"]
```

- [ ] **Step 2: Run — expect fail**

Run: `python -m pytest tests/unit/test_agent_server_core.py -q`
Expected: FAIL — `ModuleNotFoundError: agent_server.core`.

- [ ] **Step 3: Implement**

```python
# agent_server/core.py
import sys

from executor import geom_tools  # top-level — salome_init mutates sys.path later
from agent_server import handshake

_FATAL = {"ok": False, "fatal": True,
          "error": "server poisoned by a prior timeout; restart the agent server"}


class Core:
    def __init__(self, backend):
        self.backend = backend

    def handle(self, req: dict) -> dict:
        rid = req.get("id")
        if self.backend.poisoned():
            return {"id": rid, **_FATAL}
        cmd = req.get("cmd")
        b = self.backend

        if cmd == "hello":
            b.begin_session(req.get("run_id"))
            return {"id": rid, "ready": True, "proto": handshake.PROTO,
                    "mode": b.mode, "name_prefix": b.name_prefix}
        if cmd == "run":
            r = b.run_code(req["code"], req["timeout"])
            if r["ok"]:
                b.refresh(req.get("module"))
            return {"id": rid, **r}
        if cmd == "reset_and_replay":
            return {"id": rid, **b.reset(req["scripts"])}
        if cmd == "introspect":
            out = {}
            for name in req["names"]:
                obj = b.namespace.get(name)
                out[name] = {"members": dir(obj) if obj is not None else []}
            return {"id": rid, **out}
        if cmd == "list_study":
            salome = b.namespace["salome"]
            objs = []
            root = salome.myStudy.FindObjectID("0:1")
            it = salome.myStudy.NewChildIterator(root)
            while it.More():
                objs.append({"name": it.Value().GetName(), "type": ""})
                it.Next()
            return {"id": rid, "objects": objs}
        if cmd == "sanity":
            return {"id": rid, "ok": True, "warnings": [], "failures": []}
        if cmd == "list_tools":
            return {"id": rid, "tools": geom_tools.list_tools()}
        if cmd == "call_tool":
            r = geom_tools.call_tool(b, req["name"], req["args"])
            if r.get("ok"):
                b.refresh("GEOM")
            return {"id": rid, **r}
        if cmd == "shutdown":
            b.end_session()
            return {"id": rid, "ok": True}
        return {"id": rid, "ok": False, "exception": f"unknown command {cmd!r}"}
```

- [ ] **Step 4: Run — expect pass; drop the xfail; run full agent_server unit set**

Remove the `xfail` marker from `tests/unit/test_agent_server_import.py`.

Run: `python -m pytest tests/unit/test_agent_server_core.py tests/unit/test_agent_server_import.py tests/unit/test_backend.py tests/unit/test_handshake.py -q`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add agent_server/core.py tests/unit/test_agent_server_core.py tests/unit/test_agent_server_import.py
git commit -m "feat: agent_server.core — command dispatch + poison gate"
```

---

## Task 7: `agent_server/wire.py` — loopback TCP listener

**Files:**
- Create: `agent_server/wire.py`
- Test: `tests/unit/test_wire.py`

**Interfaces:**
- Consumes: a `core` object with `.handle(req) -> dict`.
- Produces:
  - `serve(core, token: str, on_bound, *, poll: float = 0.5) -> None` — blocking.
    Binds `127.0.0.1:0`, calls `on_bound(port: int)` once bound, then loops
    accepting **one client at a time**. Returns when a client sends
    `{"cmd": "shutdown"}` **and** the backend is not persistent... actually: for
    the GUI case `serve` runs until the thread is asked to stop; for headless
    `--once` the caller stops it after one session. So: `serve` loops forever;
    stopping is done by closing the listen socket from another thread. Provide
    `Server` class instead:
  - `class Server` — `__init__(self, core, token)`; `.bind() -> int` (returns
    port); `.serve_forever()`; `.stop()`. `on_bound` folded into `bind`'s
    return.
  - Pre-`hello` read timeout: 5s on a freshly accepted socket before the
    `hello` line arrives; on timeout close and keep listening.
  - Token check: `hmac.compare_digest(req["token"], self.token)`; mismatch →
    close the connection (no response).
  - Incumbent liveness: `Server` tracks `self._last_activity` (updated each time
    a request is read). A new connection arriving while one is active: if
    `time.monotonic() - self._last_activity < 30`, reply
    `{"ready": False, "error": "busy"}` and close; else treat the incumbent as
    dead, drop it, serve the newcomer.
  - Framing: `conn.makefile("rwb")`, `readline()` / `write(json+"\n")` / `flush()`,
    same as today.

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_wire.py
import json
import socket
import threading
import time

import pytest

from agent_server.wire import Server


class FakeCore:
    def handle(self, req):
        return {"id": req.get("id"), "ok": True, "echo": req.get("cmd")}


def _client(port):
    s = socket.create_connection(("127.0.0.1", port), timeout=2)
    return s, s.makefile("rwb")


@pytest.fixture
def server():
    srv = Server(FakeCore(), token="secret")
    port = srv.bind()
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield srv, port
    srv.stop()
    t.join(timeout=2)


def test_hello_with_good_token_then_a_command(server):
    _, port = server
    s, f = _client(port)
    f.write(b'{"cmd":"hello","token":"secret","id":1}\n'); f.flush()
    assert json.loads(f.readline())["ready"] is True
    f.write(b'{"cmd":"run","token":"secret","id":2}\n'); f.flush()
    assert json.loads(f.readline())["echo"] == "run"
    s.close()


def test_bad_token_drops_connection(server):
    _, port = server
    s, f = _client(port)
    f.write(b'{"cmd":"hello","token":"WRONG","id":1}\n'); f.flush()
    assert f.readline() == b""            # closed, no response
    s.close()


def test_pre_hello_timeout_frees_the_listener(server):
    srv, port = server
    srv.PRE_HELLO_TIMEOUT = 0.3
    s, _ = _client(port)                  # connect, send nothing
    time.sleep(0.6)
    s2, f2 = _client(port)                # a second client must still get in
    f2.write(b'{"cmd":"hello","token":"secret","id":1}\n'); f2.flush()
    assert json.loads(f2.readline())["ready"] is True
    s.close(); s2.close()


def test_second_live_client_gets_busy(server):
    _, port = server
    s1, f1 = _client(port)
    f1.write(b'{"cmd":"hello","token":"secret","id":1}\n'); f1.flush()
    f1.readline()
    s2, f2 = _client(port)
    f2.write(b'{"cmd":"hello","token":"secret","id":1}\n'); f2.flush()
    assert json.loads(f2.readline())["error"] == "busy"
    s1.close(); s2.close()
```

- [ ] **Step 2: Run — expect fail**

Run: `python -m pytest tests/unit/test_wire.py -q`
Expected: FAIL — `ModuleNotFoundError: agent_server.wire`.

- [ ] **Step 3: Implement**

```python
# agent_server/wire.py
import hmac
import json
import socket
import threading
import time


class Server:
    PRE_HELLO_TIMEOUT = 5.0
    BUSY_IDLE_S = 30.0

    def __init__(self, core, token: str):
        self.core = core
        self.token = token
        self._srv: socket.socket | None = None
        self._stop = threading.Event()
        self._conn_lock = threading.Lock()
        self._active = False
        self._last_activity = 0.0

    def bind(self) -> int:
        self._srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._srv.bind(("127.0.0.1", 0))
        self._srv.listen(8)
        self._srv.settimeout(0.5)
        return self._srv.getsockname()[1]

    def serve_forever(self) -> None:
        while not self._stop.is_set():
            try:
                conn, _ = self._srv.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            self._handle_conn(conn)
        if self._srv is not None:
            self._srv.close()

    def stop(self) -> None:
        self._stop.set()

    def _reject_busy(self, conn) -> bool:
        with self._conn_lock:
            if self._active and (time.monotonic() - self._last_activity) < self.BUSY_IDLE_S:
                return True
            self._active = True
            self._last_activity = time.monotonic()
            return False

    def _handle_conn(self, conn) -> None:
        conn.settimeout(self.PRE_HELLO_TIMEOUT)
        f = conn.makefile("rwb")
        try:
            first = f.readline()
            if not first:
                return
            req = json.loads(first)
            if not hmac.compare_digest(str(req.get("token", "")), self.token):
                return                       # drop, no response
            if req.get("cmd") == "hello" and self._reject_busy(conn):
                f.write(b'{"ready": false, "error": "busy"}\n'); f.flush()
                return
            conn.settimeout(None)
            resp = self.core.handle(req)
            f.write((json.dumps(resp) + "\n").encode()); f.flush()
            for line in f:
                self._last_activity = time.monotonic()
                resp = self.core.handle(json.loads(line))
                f.write((json.dumps(resp) + "\n").encode()); f.flush()
                if json.loads(line).get("cmd") == "shutdown":
                    break
        except (socket.timeout, ValueError, OSError):
            pass
        finally:
            with self._conn_lock:
                self._active = False
            try:
                conn.close()
            except OSError:
                pass
```

Note: `_reject_busy` sets `_active=True` on the accepted path; the `finally`
clears it. The `test_second_live_client_gets_busy` case works because client 1's
`hello` sets `_active` and never closes before client 2 connects.

- [ ] **Step 4: Run — expect pass**

Run: `python -m pytest tests/unit/test_wire.py -q`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add agent_server/wire.py tests/unit/test_wire.py
git commit -m "feat: agent_server.wire — loopback TCP + token listener"
```

---

## Task 8: `agent_server/namespace.py`

**Files:**
- Create: `agent_server/namespace.py`
- Test: covered by `tests/salome/` (Task 15) — no unit test (needs SALOME).

**Interfaces:**
- Produces: `build_namespace(live: bool) -> dict` with keys `salome`, `geompy`,
  `smesh`, `pvsimple`, `study_ns`, `find_object` (same contract as
  `executor/executor.py:87-116`).

- [ ] **Step 1: Implement (no unit test — SALOME-only)**

```python
# agent_server/namespace.py
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _find_object_factory(salome_kernel):
    def find_object(name: str):
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
        result = walk(salome_kernel.myStudy.FindObjectID("0:1"))
        if result is None:
            raise LookupError(f"no study object named {name!r}")
        return result
    return find_object


def build_namespace(live: bool) -> dict:
    if not live:
        from salome.kernel import KernelBasis
        KernelBasis.setSSLMode(True)
    from salome.kernel import salome
    salome.salome_init()   # guarded — no-op if the GUI already ran it (live=True)
    from salome.geom import geomBuilder
    geompy = geomBuilder.New()
    from salome.smesh import smeshBuilder
    smesh = smeshBuilder.New()
    try:
        import pvsimple
    except ImportError:
        pvsimple = None
    return {
        "salome": salome, "geompy": geompy, "smesh": smesh, "pvsimple": pvsimple,
        "study_ns": "salome_agent_run",
        "find_object": _find_object_factory(salome),
    }
```

- [ ] **Step 2: Sanity — import compiles**

Run: `python -c "import ast; ast.parse(open('agent_server/namespace.py').read())"`
Expected: no output (parses).

- [ ] **Step 3: Commit**

```bash
git add agent_server/namespace.py
git commit -m "feat: agent_server.namespace — build_namespace(live)"
```

---

## Task 9: `agent_server/gui_backend.py` — Qt marshal + `GuiBackend`

**Files:**
- Create: `agent_server/gui_backend.py`
- Test: `tests/salome/test_gui_marshal.py` (marked `salome` — needs PySide2)

**Interfaces:**
- Consumes: `agent_server.backend` (`_run_with_timeout`), `agent_server.namespace`.
- Produces:
  - `class MainThreadInvoker(QObject)` — `.run_on_main(fn, timeout) -> Any`.
    Raises `TimeoutError` on a bounded-wait expiry (and **leaves `_busy`
    latched**). Raises `RuntimeError` if a call is already in flight.
  - `class GuiBackend` — `mode = "gui"`. `__init__(self, invoker, build_namespace=None)`.
    `begin_session(run_id)` → `name_prefix = f"sa_{run_id}_"` (or `""` if no
    run_id), `reset_counters()`, rebind namespace via `run_on_main`.
    `run_code(code, timeout)` marshals `exec` via `run_on_main`; on `TimeoutError`
    sets `self._poisoned = True` and returns a `{"ok": False, "exception":
    "timeout after Ns"}` dict. `reset(scripts)` → `{"ok": False, "error": "no
    rollback in a live GUI study"}`. `refresh(module)` → marshal
    `salome.sg.updateObjBrowser()` (best-effort, swallow exceptions).
    `find_object(name)` → marshal: try `namespace["find_object"](name_prefix + name)`,
    on `LookupError` retry `namespace["find_object"](name)`.

- [ ] **Step 1: Write the failing tests** (`tests/salome/test_gui_marshal.py`)

```python
import gc
import threading

import pytest

pytestmark = pytest.mark.salome

try:
    import SalomePyQt  # noqa
    _HAVE_QT = True
except ImportError:
    _HAVE_QT = False


@pytest.fixture
def qapp():
    if not _HAVE_QT:
        pytest.skip("run under `salome shell -- pytest`")
    from PySide2.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app


def test_marshal_fires_from_bare_thread_onto_main(qapp):
    from agent_server.gui_backend import MainThreadInvoker
    inv = MainThreadInvoker()
    gc.collect()
    box = {}

    def worker():
        box["ran_on"] = inv.run_on_main(
            lambda: threading.current_thread().name, timeout=2)
        box["done"] = True

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    deadline = 3
    from PySide2.QtCore import QCoreApplication
    import time
    start = time.time()
    while not box.get("done") and time.time() - start < deadline:
        QCoreApplication.processEvents()
        time.sleep(0.01)
    assert box.get("ran_on") == threading.main_thread().name


def test_slot_exception_surfaces_without_abort(qapp):
    from agent_server.gui_backend import MainThreadInvoker
    inv = MainThreadInvoker()

    def worker(box):
        try:
            inv.run_on_main(lambda: (_ for _ in ()).throw(ValueError("boom")),
                            timeout=2)
        except ValueError as e:
            box["err"] = str(e)

    box = {}
    t = threading.Thread(target=worker, args=(box,), daemon=True)
    t.start()
    from PySide2.QtCore import QCoreApplication
    import time
    start = time.time()
    while "err" not in box and time.time() - start < 3:
        QCoreApplication.processEvents(); time.sleep(0.01)
    assert box["err"] == "boom"


def test_timeout_raises_and_latches_busy(qapp):
    from agent_server.gui_backend import MainThreadInvoker
    inv = MainThreadInvoker()
    # nothing pumps the event loop -> the emit never runs -> bounded wait fires
    with pytest.raises(TimeoutError):
        inv.run_on_main(lambda: 1, timeout=0.2)
    with pytest.raises(RuntimeError):
        inv.run_on_main(lambda: 1, timeout=0.2)   # still latched
```

- [ ] **Step 2: Run — expect fail / skip**

Run (under salome): `"$SALOME_ROOT"/salome shell -- python -m pytest tests/salome/test_gui_marshal.py -q`
Expected: FAIL — `ModuleNotFoundError: agent_server.gui_backend`. (Plain
`python -m pytest` → SKIP.)

- [ ] **Step 3: Implement**

```python
# agent_server/gui_backend.py
import threading

import SalomePyQt

if SalomePyQt.usePySide():
    from PySide2.QtCore import QObject, Signal as _Signal, Qt
else:
    from PyQt5.QtCore import QObject, pyqtSignal as _Signal, Qt

from agent_server.backend import _run_with_timeout
from executor import geom_tools


class _Job:
    __slots__ = ("fn", "box", "ev")

    def __init__(self, fn):
        self.fn = fn
        self.box: dict = {}
        self.ev = threading.Event()


class MainThreadInvoker(QObject):
    _invoke = _Signal(object)

    def __init__(self):
        super().__init__()
        self._lock = threading.Lock()
        self._busy = False
        self._invoke.connect(self._on_invoke, Qt.QueuedConnection)

    def _on_invoke(self, job):
        try:
            job.box["r"] = job.fn()
        except BaseException as e:            # an escape -> qFatal() -> GUI dies
            job.box["e"] = e
        finally:
            job.ev.set()

    def run_on_main(self, fn, timeout):
        with self._lock:
            if self._busy:
                raise RuntimeError("a marshalled call is already in flight")
            self._busy = True
        job = _Job(fn)
        timed_out = False
        try:
            self._invoke.emit(job)
            if not job.ev.wait(timeout):
                timed_out = True
                raise TimeoutError(f"main-thread call exceeded {timeout}s")
            if "e" in job.box:
                raise job.box["e"]
            return job.box["r"]
        finally:
            if not timed_out:
                with self._lock:
                    self._busy = False


class GuiBackend:
    mode = "gui"

    def __init__(self, invoker: MainThreadInvoker, build_namespace=None):
        if build_namespace is None:
            from agent_server.namespace import build_namespace as bn
            build_namespace = bn
        self._invoker = invoker
        self._build = build_namespace
        self.name_prefix = ""
        self.namespace = invoker.run_on_main(lambda: self._build(True), timeout=120)
        self.geom_bindings: dict = {}
        self.accepted_scripts: list = []
        self._poisoned = False

    def begin_session(self, run_id):
        self.name_prefix = f"sa_{run_id}_" if run_id else ""
        geom_tools.reset_counters()
        self.geom_bindings.clear()
        self.accepted_scripts = []

    def end_session(self):
        pass

    def run_code(self, code, timeout):
        try:
            self._invoker.run_on_main(
                lambda: exec(compile(code, "<stage>", "exec"), self.namespace),
                timeout,
            )
            return {"ok": True, "stdout": "", "stderr": "", "exception": None,
                    "traceback": None, "offending_line": None}
        except TimeoutError:
            self._poisoned = True
            return {"ok": False, "stdout": "", "stderr": "",
                    "exception": f"timeout after {timeout}s",
                    "traceback": None, "offending_line": None}
        except BaseException as e:            # noqa: BLE001
            import traceback
            return {"ok": False, "stdout": "", "stderr": "", "exception": str(e),
                    "traceback": traceback.format_exc(), "offending_line": None}

    def reset(self, scripts):
        return {"ok": False, "error": "no rollback in a live GUI study"}

    def refresh(self, module):
        def _do():
            self.namespace["salome"].sg.updateObjBrowser()
        try:
            self._invoker.run_on_main(_do, timeout=10)
        except Exception:
            pass

    def find_object(self, name):
        def _do():
            fo = self.namespace["find_object"]
            try:
                return fo(self.name_prefix + name)
            except LookupError:
                return fo(name)
        return self._invoker.run_on_main(_do, timeout=30)

    def poisoned(self):
        return self._poisoned
```

- [ ] **Step 4: Run under salome — expect pass**

Run: `"$SALOME_ROOT"/salome shell -- python -m pytest tests/salome/test_gui_marshal.py -q`
Expected: PASS (3). If Gate A failed in Task 1, this is where it is
formally blocked — escalate.

- [ ] **Step 5: Commit**

```bash
git add agent_server/gui_backend.py tests/salome/test_gui_marshal.py
git commit -m "feat: agent_server.gui_backend — Qt main-thread marshal + GuiBackend"
```

---

## Task 10: `agent_server/__main__.py` — headless entry

**Files:**
- Create: `agent_server/__main__.py`
- Test: `tests/unit/test_agent_server_main.py` (arg parsing only)

**Interfaces:**
- Consumes: `Core`, `HeadlessBackend`, `Server` (Task 7), `handshake`.
- Produces:
  - `parse_args(argv) -> argparse.Namespace` — `--once` (default `True`),
    `--serve-forever` (sets `once=False`).
  - `main(argv=None) -> int` — generates a token, `Server(Core(HeadlessBackend()), token)`,
    `port = srv.bind()`, `handshake.write(port=port, token=token, mode="headless",
    log_path=<~/.salome_agent/server.log>)`, redirect stdout/stderr to that log,
    `srv.serve_forever()` (which returns after one client disconnects when
    `--once`), `handshake.remove()`, return 0 (1 if the backend ended poisoned).

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_agent_server_main.py
from agent_server.__main__ import parse_args


def test_once_is_the_default():
    assert parse_args([]).once is True


def test_serve_forever_flag():
    assert parse_args(["--serve-forever"]).once is False
```

- [ ] **Step 2: Run — expect fail**

Run: `python -m pytest tests/unit/test_agent_server_main.py -q`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement**

```python
# agent_server/__main__.py
import argparse
import os
import secrets
import sys

from agent_server import handshake
from agent_server.backend import HeadlessBackend
from agent_server.core import Core
from agent_server.wire import Server


def parse_args(argv):
    p = argparse.ArgumentParser(prog="agent_server")
    p.add_argument("--serve-forever", dest="once", action="store_false",
                   default=True,
                   help="keep serving after the first client disconnects "
                        "(accepts the abandoned-watchdog hazard)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    log_path = os.path.expanduser("~/.salome_agent/server.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logf = open(log_path, "w")
    os.dup2(logf.fileno(), sys.stdout.fileno())
    os.dup2(logf.fileno(), sys.stderr.fileno())

    backend = HeadlessBackend()
    token = secrets.token_hex(16)
    srv = Server(Core(backend), token)
    port = srv.bind()
    handshake.write(port=port, token=token, mode="headless", log_path=log_path)
    try:
        srv.serve_forever_once() if args.once else srv.serve_forever()
    finally:
        srv.stop()
        handshake.remove()
    return 1 if backend.poisoned() else 0


if __name__ == "__main__":
    sys.exit(main())
```

Add `serve_forever_once()` to `agent_server/wire.py::Server` — identical to
`serve_forever` but `return`s after `_handle_conn` completes once. Add a
one-line unit test for it in `tests/unit/test_wire.py` (connect, `hello`,
`shutdown`, assert the thread ends).

- [ ] **Step 4: Run — expect pass**

Run: `python -m pytest tests/unit/test_agent_server_main.py tests/unit/test_wire.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add agent_server/__main__.py agent_server/wire.py tests/unit/test_agent_server_main.py tests/unit/test_wire.py
git commit -m "feat: agent_server.__main__ — headless entry (--once default)"
```

---

## Task 11: `agent_server/gui_plugin.py` + `plugin/salome_agent_plugin.py`

**Files:**
- Create: `agent_server/gui_plugin.py`, `plugin/salome_agent_plugin.py`
- Test: `tests/unit/test_gui_plugin.py` (with fakes — no real Qt)

**Interfaces:**
- Consumes: `Core`, `GuiBackend`, `MainThreadInvoker`, `Server`, `handshake`.
- Produces:
  - `launch(context) -> None` — the `AddFunction` callback. Idempotent-restart:
    stop any prior `_SERVER` / thread, `handshake.remove()`, then start fresh.
    Module globals `_INVOKER`, `_SERVER`, `_SERVER_THREAD`, `_CORE` hold refs
    (else PySide2 GC's the `QObject`). Posts a non-modal status message via
    `context`'s `sg` / `SalomePyQt` (`sgPyQt.message(...)`), success shows
    `127.0.0.1:<port>`, failure shows the exception string.
  - `stop() -> None` — used by `aboutToQuit`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_gui_plugin.py
import sys
import types


def _install_fakes(monkeypatch):
    # Fake the Qt/SALOME-only deps so gui_plugin imports in the venv.
    fake_gb = types.ModuleType("agent_server.gui_backend")

    class FakeInvoker:
        def __init__(self): pass

    class FakeGuiBackend:
        mode = "gui"
        def __init__(self, invoker, build_namespace=None): pass

    fake_gb.MainThreadInvoker = FakeInvoker
    fake_gb.GuiBackend = FakeGuiBackend
    monkeypatch.setitem(sys.modules, "agent_server.gui_backend", fake_gb)


def test_launch_starts_server_and_writes_handshake(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    _install_fakes(monkeypatch)
    started = {}

    import agent_server.wire as wire

    class FakeServer:
        def __init__(self, core, token): started["token"] = token
        def bind(self): return 55555
        def serve_forever(self): started["served"] = True
        def stop(self): started["stopped"] = True

    monkeypatch.setattr(wire, "Server", FakeServer)

    from agent_server import gui_plugin, handshake
    gui_plugin.launch(context=types.SimpleNamespace(sg=None))
    hs = handshake.read()
    assert hs["port"] == 55555
    assert hs["mode"] == "gui"
    gui_plugin.stop()
    assert started["stopped"] is True
```

- [ ] **Step 2: Run — expect fail**

Run: `python -m pytest tests/unit/test_gui_plugin.py -q`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement**

```python
# agent_server/gui_plugin.py
import secrets
import threading

from agent_server import handshake
from agent_server import wire as _wire
from agent_server.core import Core

_INVOKER = None
_SERVER = None
_SERVER_THREAD = None
_CORE = None


def _status(context, text):
    try:
        import SalomePyQt
        SalomePyQt.SalomePyQt().message(text, False)
    except Exception:
        print(f"[agent-server] {text}")


def stop() -> None:
    global _SERVER, _SERVER_THREAD
    if _SERVER is not None:
        _SERVER.stop()
    if _SERVER_THREAD is not None:
        _SERVER_THREAD.join(timeout=3)
    _SERVER = _SERVER_THREAD = None
    handshake.remove()


def launch(context) -> None:
    global _INVOKER, _SERVER, _SERVER_THREAD, _CORE
    from agent_server.gui_backend import GuiBackend, MainThreadInvoker

    stop()                                    # 2nd click == restart
    try:
        if _INVOKER is None:
            _INVOKER = MainThreadInvoker()
        backend = GuiBackend(_INVOKER)
        token = secrets.token_hex(16)
        _CORE = Core(backend)
        _SERVER = _wire.Server(_CORE, token)
        port = _SERVER.bind()
        _SERVER_THREAD = threading.Thread(target=_SERVER.serve_forever, daemon=True)
        _SERVER_THREAD.start()
        import os
        handshake.write(port=port, token=token, mode="gui",
                        log_path=os.path.expanduser("~/.salome_agent/server.log"))
        _register_aboutToQuit()
        _status(context, f"SALOME agent server on 127.0.0.1:{port}")
    except Exception as e:                     # noqa: BLE001
        _status(context, f"SALOME agent server failed to start: {e}")
        raise


def _register_aboutToQuit():
    try:
        from PySide2.QtWidgets import QApplication
    except ImportError:
        from PyQt5.QtWidgets import QApplication
    app = QApplication.instance()
    if app is not None:
        app.aboutToQuit.connect(stop)
```

`plugin/salome_agent_plugin.py`:
```python
import salome_pluginsmanager
from agent_server.gui_plugin import launch

salome_pluginsmanager.AddFunction(
    'Launch SALOME agent server',
    'Start (or restart) the SALOME-agent server in this GUI process',
    launch,
)
```

- [ ] **Step 4: Run — expect pass**

Run: `python -m pytest tests/unit/test_gui_plugin.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add agent_server/gui_plugin.py plugin/salome_agent_plugin.py tests/unit/test_gui_plugin.py
git commit -m "feat: agent_server.gui_plugin + the plugin registration file"
```

---

## Task 12: `salome_agent/executor_client.py` + `server_conn.py`

**Files:**
- Modify: `salome_agent/executor_client.py` (whole file)
- Create: `salome_agent/server_conn.py`
- Modify: `tests/unit/test_executor_client.py` (fake server → `AF_INET` + `hello`)
- Test: `tests/unit/test_server_conn.py`

**Interfaces:**
- Produces:
  - `executor_client.SessionDied` (kept), `executor_client.SessionPoisoned` (new).
  - `ExecutorClient.__init__(self, host: str, port: int, token: str, run_id: str)`
    — TCP connect, send `{"cmd": "hello", "token": token, "run_id": run_id, "id": 0}`,
    read the `ready` reply; raises `SessionPoisoned` if the reply has
    `"fatal": true`; raises `ConnectionError`-subclass on `busy`.
  - `_call` raises `SessionPoisoned` when a response carries `"fatal": true`.
  - All the existing methods (`run`, `reset_and_replay`, `introspect`,
    `list_study`, `sanity`, `list_tools`, `call_tool`, `shutdown`) unchanged in
    signature.
  - `server_conn.ServerUnavailable(Exception)`.
  - `server_conn.read_handshake() -> dict | None` — wraps `agent_server`?
    No — the agent side must not import `agent_server`. Re-read the JSON directly:
    `~/.salome_agent/server.json`, return `None` on missing / bad JSON /
    `proto != 1`.
  - `server_conn.connect(run_id: str) -> ExecutorClient` — `read_handshake()`,
    TCP `ExecutorClient(...)`, one retry after re-reading the handshake on a
    dropped connect, raise `ServerUnavailable` (absent / busy / poisoned) with a
    message telling the user to start it from Tools ▸ Plugins or
    `scripts/start-headless-server.sh`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_server_conn.py
import json
import socket
import threading

import pytest

from salome_agent import server_conn
from salome_agent.executor_client import SessionPoisoned


def _fake_server(port_box, replies, expect_hello=True):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port_box.append(srv.getsockname()[1])
    conn, _ = srv.accept()
    f = conn.makefile("rwb")
    if expect_hello:
        f.readline()
    for r in replies:
        f.write((json.dumps(r) + "\n").encode()); f.flush()
        f.readline()
    conn.close()
    srv.close()


@pytest.fixture
def _home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    d = tmp_path / ".salome_agent"
    d.mkdir()
    return d


def _write_hs(_home, port, token="t"):
    (_home / "server.json").write_text(json.dumps(
        {"proto": 1, "host": "127.0.0.1", "port": port, "token": token,
         "pid": 1, "mode": "gui"}))


def test_connect_absent_handshake_raises(_home):
    with pytest.raises(server_conn.ServerUnavailable):
        server_conn.connect("run1")


def test_connect_happy_path(_home):
    box = []
    t = threading.Thread(target=_fake_server,
                         args=(box, [{"ready": True, "proto": 1, "mode": "gui",
                                      "name_prefix": "sa_run1_"}]))
    t.start()
    while not box:
        pass
    _write_hs(_home, box[0])
    client = server_conn.connect("run1")
    assert client is not None
    t.join()


def test_connect_poisoned_raises(_home):
    box = []
    t = threading.Thread(target=_fake_server,
                         args=(box, [{"ready": False, "fatal": True,
                                      "error": "poisoned"}]))
    t.start()
    while not box:
        pass
    _write_hs(_home, box[0])
    with pytest.raises((server_conn.ServerUnavailable, SessionPoisoned)):
        server_conn.connect("run1")
    t.join()
```

- [ ] **Step 2: Run — expect fail**

Run: `python -m pytest tests/unit/test_server_conn.py -q`
Expected: FAIL — `salome_agent.server_conn` missing.

- [ ] **Step 3: Implement `executor_client.py`**

```python
# salome_agent/executor_client.py
import json
import socket


class SessionDied(Exception):
    pass


class SessionPoisoned(Exception):
    pass


class ServerBusy(Exception):
    pass


class ExecutorClient:
    def __init__(self, host: str, port: int, token: str, run_id: str):
        self._sock = socket.create_connection((host, port), timeout=10)
        self._f = self._sock.makefile("rwb")
        self._next_id = 1
        self._f.write((json.dumps(
            {"id": 0, "cmd": "hello", "token": token, "run_id": run_id}) + "\n"
        ).encode())
        self._f.flush()
        ready = self._recv()
        if ready.get("fatal"):
            raise SessionPoisoned(ready.get("error", "poisoned"))
        if ready.get("error") == "busy":
            raise ServerBusy("another agent run is in progress")
        assert ready.get("ready") is True
        self.mode = ready.get("mode")
        self.name_prefix = ready.get("name_prefix", "")

    def _recv(self) -> dict:
        line = self._f.readline()
        if not line:
            raise SessionDied("server socket closed")
        return json.loads(line)

    def _call(self, cmd: str, **payload) -> dict:
        req_id = self._next_id
        self._next_id += 1
        try:
            self._f.write((json.dumps({"id": req_id, "cmd": cmd, **payload}) + "\n").encode())
            self._f.flush()
        except (BrokenPipeError, ConnectionResetError, OSError) as e:
            raise SessionDied(f"server socket write failed: {e}") from e
        resp = self._recv()
        if resp.get("fatal"):
            raise SessionPoisoned(resp.get("error", "poisoned"))
        return resp

    def run(self, code, timeout, module=None):
        return self._call("run", code=code, timeout=timeout, module=module)

    def reset_and_replay(self, scripts):
        return self._call("reset_and_replay", scripts=scripts)

    def introspect(self, names):
        return self._call("introspect", names=names)

    def list_study(self):
        return self._call("list_study")

    def sanity(self, module, expectations):
        return self._call("sanity", module=module, expectations=expectations)

    def list_tools(self):
        return self._call("list_tools")

    def call_tool(self, name, args):
        return self._call("call_tool", name=name, args=args)

    def shutdown(self):
        return self._call("shutdown")
```

`salome_agent/server_conn.py`:
```python
import json
import os

from salome_agent.executor_client import (ExecutorClient, ServerBusy,
                                          SessionDied, SessionPoisoned)

_HELP = ("Start it from Tools ▸ Plugins ▸ Launch SALOME agent server, "
         "or run scripts/start-headless-server.sh")


class ServerUnavailable(Exception):
    pass


def read_handshake() -> dict | None:
    try:
        data = json.loads(
            open(os.path.expanduser("~/.salome_agent/server.json")).read())
    except (OSError, ValueError):
        return None
    if data.get("proto") != 1:
        return None
    return data


def connect(run_id: str) -> ExecutorClient:
    for attempt in (1, 2):
        hs = read_handshake()
        if hs is None:
            raise ServerUnavailable(f"No SALOME agent server running. {_HELP}")
        try:
            return ExecutorClient(hs["host"], hs["port"], hs["token"], run_id)
        except ServerBusy as e:
            raise ServerUnavailable(f"{e}. {_HELP}") from e
        except SessionPoisoned as e:
            raise ServerUnavailable(
                f"The SALOME agent server is poisoned ({e}); restart it. {_HELP}") from e
        except (ConnectionError, SessionDied, OSError, AssertionError):
            if attempt == 2:
                raise ServerUnavailable(
                    f"Could not connect to the SALOME agent server. {_HELP}")
            # fall through: re-read handshake (token may have rotated) and retry
    raise ServerUnavailable(f"Could not connect to the SALOME agent server. {_HELP}")
```

- [ ] **Step 4: Rework `tests/unit/test_executor_client.py`**

Replace the `AF_UNIX` fake server with an `AF_INET` one that first reads the
`hello` line and replies `{"ready": true, "proto": 1, "mode": "headless"}`, then
serves the scripted responses. Keep both existing test intents
(`test_run_returns_result`, `test_dropped_socket_raises_session_died`) and add
`test_fatal_reply_raises_session_poisoned`.

- [ ] **Step 5: Run — expect pass**

Run: `python -m pytest tests/unit/test_server_conn.py tests/unit/test_executor_client.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add salome_agent/executor_client.py salome_agent/server_conn.py tests/unit/test_server_conn.py tests/unit/test_executor_client.py
git commit -m "feat: TCP ExecutorClient with hello handshake; server_conn.connect; SessionPoisoned"
```

---

## Task 13: fatal path — `exec_check.py`, `graph.py`, `specialists/geom.py`, `state.py`

**Files:**
- Modify: `salome_agent/verify/exec_check.py`, `salome_agent/graph.py:139-165` and
  `:57-111`, `salome_agent/specialists/geom.py:18-60`, `salome_agent/state.py`
- Test: `tests/unit/test_exec_check.py`, `tests/unit/test_graph.py`,
  `tests/unit/test_specialist_geom.py`, `tests/unit/test_state.py`

**Interfaces:**
- Consumes: `executor_client.SessionPoisoned`, `SessionDied`.
- Produces:
  - `check_exec(...) -> dict | Literal["session_died", "session_poisoned"] | None`.
  - `RunState` gains `fatal: Optional[str]`.
  - `run_geom_specialist(...)` returns `{"fatal": "<msg>"}` on `SessionDied` /
    `SessionPoisoned` (in addition to the existing `{"call_log", "result_names"}`
    / `{"call_log", "last_error"}` shapes).
  - `_verify_exec_node`: `"session_poisoned"` → `Command(update={"fatal": msg}, goto="assemble")`.
  - `_run_stage_node`: if the geom result dict has `"fatal"`, →
    `Command(update={"fatal": result["fatal"]}, goto="assemble")`.

- [ ] **Step 1: Write the failing tests**

```python
# add to tests/unit/test_exec_check.py
from salome_agent.executor_client import SessionPoisoned

def test_session_poisoned_returns_sentinel():
    client = MagicMock()
    client.run.side_effect = SessionPoisoned("boom")
    assert check_exec("code", "geom", client, timeout=5) == "session_poisoned"
```

```python
# add to tests/unit/test_specialist_geom.py
from salome_agent.executor_client import SessionPoisoned

def test_geom_specialist_returns_fatal_on_session_poisoned():
    from salome_agent.specialists.geom import run_geom_specialist
    llm = MagicMock()
    client = MagicMock()
    client.list_tools.side_effect = SessionPoisoned("dead")
    out = run_geom_specialist({"module": "geom", "intent": "x"}, llm, client, 5)
    assert "fatal" in out
```

```python
# add to tests/unit/test_graph.py  (needs a light node-level test)
def test_verify_exec_session_poisoned_sets_fatal(monkeypatch):
    import salome_agent.graph as g
    from langgraph.types import Command
    monkeypatch.setattr(g, "check_exec", lambda *a, **k: "session_poisoned",
                        raising=False)
    # call the node with a minimal state/config; assert Command routes to
    # assemble with update={"fatal": ...}
    # (use the existing fake-config pattern from test_graph.py's neighbours)
```

```python
# add to tests/unit/test_state.py
def test_runstate_has_fatal_key():
    from salome_agent.state import RunState
    assert "fatal" in RunState.__annotations__
```

- [ ] **Step 2: Run — expect fail**

Run: `python -m pytest tests/unit/test_exec_check.py tests/unit/test_specialist_geom.py tests/unit/test_state.py -q`
Expected: FAIL.

- [ ] **Step 3: Implement**

`verify/exec_check.py` — import `SessionPoisoned`, wrap the `client.run` call:
```python
from salome_agent.executor_client import SessionDied, SessionPoisoned
...
    try:
        run_result = client.run(script, timeout, module=module.upper())
    except SessionPoisoned:
        return "session_poisoned"
    except SessionDied:
        return "session_died"
```
Update the return type hint to include `"session_poisoned"`.

`graph.py::_verify_exec_node` — after the `session_died` branch:
```python
    if result == "session_poisoned":
        log.error("  SALOME agent server poisoned by a timeout - aborting run")
        return Command(update={"fatal": "SALOME agent server poisoned by a stage timeout"},
                       goto="assemble")
```

`graph.py::_run_stage_node` — in the `if module == "geom":` branch, right after
`result = run_geom_specialist(...)`:
```python
        if "fatal" in result:
            return Command(update={"fatal": result["fatal"]}, goto="assemble")
```

`specialists/geom.py` — wrap the body:
```python
from salome_agent.executor_client import SessionDied, SessionPoisoned
...
def run_geom_specialist(stage_plan, llm, client, iteration_cap):
    try:
        return _run(stage_plan, llm, client, iteration_cap)   # existing body
    except (SessionDied, SessionPoisoned) as e:
        return {"fatal": f"SALOME agent server unavailable: {e}"}
```
(Rename the current function body to `_run`.)

`state.py` — add `fatal: Optional[str]` to `RunState`.

- [ ] **Step 4: Run — expect pass; run the whole unit suite**

Run: `python -m pytest tests/unit -q`
Expected: PASS (minus the yet-to-be-deleted `test_session_manager.py` /
`test_discovery.py` / `test_executor_live.py`, which still pass for now, and the
`test_cli.py` changes in Task 14).

- [ ] **Step 5: Commit**

```bash
git add salome_agent/verify/exec_check.py salome_agent/graph.py salome_agent/specialists/geom.py salome_agent/state.py tests/unit/
git commit -m "feat: fatal-run path — SessionPoisoned routes stage -> assemble, RunState.fatal"
```

---

## Task 14: `state.py` `SessionHandle`, `report.py`, `cli.py` rewire

**Files:**
- Modify: `salome_agent/state.py:28-34`, `salome_agent/nodes/report.py`,
  `salome_agent/cli.py`
- Test: `tests/unit/test_state.py`, `tests/unit/test_cli.py`, and a new
  `tests/unit/test_report.py`

**Interfaces:**
- Produces:
  - `SessionHandle = {mode: Literal["gui","headless","none"], pid: Optional[int],
    salome_port: Optional[int], name_prefix: str}`.
  - `write_report(state)` renders a `## Outcome` line when `state.get("fatal")`,
    and a session-mode line ("live GUI study — objects prefixed `sa_<run_id>_`
    may remain from failed attempts" for `gui`, "isolated study" for `headless`).
  - `cli.main`: no `--live-gui`, no `--port`; connects via
    `server_conn.connect(run_id)`; on `ServerUnavailable` logs the message and
    returns 1; builds `session_handle` from `client.mode` / `client.name_prefix`;
    prints `log_path` from `server_conn.read_handshake()`; `return 1 if
    final.get("fatal") else 0` (kept: the `__interrupt__` → return 1 path).

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_report.py
from salome_agent.nodes.report import write_report


def _state(**over):
    base = {"run_id": "r1", "user_request": "make a box", "stages": {},
            "session": {"mode": "gui", "name_prefix": "sa_r1_"}, "fatal": None}
    base.update(over)
    return base


def test_report_notes_gui_debris():
    out = write_report(_state())
    assert "sa_" in out and ("GUI" in out or "live" in out.lower())


def test_report_renders_fatal():
    out = write_report(_state(fatal="server poisoned by a timeout"))
    assert "poisoned" in out
```

```python
# tests/unit/test_cli.py  — replace the --live-gui / --port assertions
def test_parse_args_has_no_live_gui_or_port():
    from salome_agent.cli import parse_args
    ns = parse_args(["make a box"])
    assert not hasattr(ns, "live_gui")
    assert not hasattr(ns, "port")
    assert ns.no_session is False
```

```python
# tests/unit/test_state.py  — replace SessionHandle assertions
def test_session_handle_shape():
    from salome_agent.state import SessionHandle
    keys = SessionHandle.__annotations__
    assert set(keys) == {"mode", "pid", "salome_port", "name_prefix"}
```

- [ ] **Step 2: Run — expect fail**

Run: `python -m pytest tests/unit/test_report.py tests/unit/test_cli.py tests/unit/test_state.py -q`
Expected: FAIL.

- [ ] **Step 3: Implement**

`state.py`:
```python
class SessionHandle(TypedDict):
    mode: Literal["gui", "headless", "none"]
    pid: Optional[int]
    salome_port: Optional[int]
    name_prefix: str
```

`nodes/report.py`:
```python
def write_report(state: dict) -> str:
    lines = [f"# Report: {state['run_id']}", "",
             f"Request: {state['user_request']}", ""]
    if state.get("fatal"):
        lines += ["## Outcome", "", f"**Run aborted:** {state['fatal']}", ""]
    sess = state.get("session") or {}
    if sess.get("mode") == "gui":
        lines += [f"Session: live GUI study. Objects prefixed "
                  f"`{sess.get('name_prefix', 'sa_')}` may remain from failed "
                  f"stage attempts — safe to bulk-delete.", ""]
    elif sess.get("mode") == "headless":
        lines += ["Session: isolated study.", ""]
    lines.append("## Stages")
    total = len(state["stages"])
    for idx in sorted(state["stages"]):
        stage = state["stages"][idx]
        retries = stage.get("static_retries", 0) + stage.get("exec_retries", 0)
        lines.append(f"- stage {idx + 1}/{total} ({stage['module']}): "
                     f"{stage['status']}, retries={retries}")
    return "\n".join(lines)
```

`cli.py` — the substantive rewrite:
- `parse_args`: drop lines 9-14 (`session_mode` group, `--live-gui`, `--port`).
  Keep `--no-session` as a plain `add_argument("--no-session", action="store_true")`.
- `main`: replace the `SessionManager` block (lines 39, 55, 66-90, and the
  `finally: manager.stop()` at 104-105) with:
  ```python
  from salome_agent import server_conn

  ...
  run_id = uuid.uuid4().hex[:12]
  ...
  if args.no_session:
      log.info("session: --no-session (planner + specialists + static checks only)")
      client = None
      session_handle = {"mode": "none", "pid": None, "salome_port": None,
                        "name_prefix": ""}
  else:
      try:
          client = server_conn.connect(run_id)
      except server_conn.ServerUnavailable as e:
          log.error("%s", e)
          return 1
      session_handle = {"mode": client.mode, "pid": None,
                        "salome_port": None, "name_prefix": client.name_prefix}
      log.info("session: connected to the %s agent server", client.mode)
  ...
  try:
      graph = build_graph(batch=args.batch)
      final = graph.invoke({... "fatal": None, ...}, config=cfg)
      if not args.batch:
          final = resume_interactive(graph, final, cfg)
  finally:
      if client is not None:
          try:
              client.shutdown()
          except Exception:
              pass
  ...
  hs = server_conn.read_handshake() or {}
  ...
  print(f"  server log    : {hs.get('log_path', '(n/a)')}")
  ...
  if final.get("__interrupt__"):
      return 1
  return 1 if final.get("fatal") else 0
  ```
- Add `"fatal": None` to the `graph.invoke` initial state dict.
- Drop the `executor_log` variable and the "executor log" print line.

- [ ] **Step 4: Run — expect pass**

Run: `python -m pytest tests/unit/test_report.py tests/unit/test_cli.py tests/unit/test_state.py tests/unit/test_interactive.py -q`
Expected: PASS. (`test_interactive.py` exercises `resume_interactive` — confirm
it still passes; it does not touch `SessionManager`.)

- [ ] **Step 5: Commit**

```bash
git add salome_agent/state.py salome_agent/nodes/report.py salome_agent/cli.py tests/unit/
git commit -m "feat: cli connects via server_conn; SessionHandle reshape; report renders fatal + mode"
```

---

## Task 15: delete `session/` + `executor/executor.py`; retarget SALOME test fixtures

**Files:**
- Delete: `salome_agent/session/` (all 3 files), `executor/executor.py`,
  `tests/unit/test_session_manager.py`, `tests/unit/test_discovery.py`,
  `tests/unit/test_executor_live.py`, `tests/salome/test_live_gui.py`
- Modify: `tests/salome/conftest.py`, `tests/salome/test_executor.py`,
  `tests/salome/test_geom_tools.py`, `tests/salome/test_end_to_end.py`,
  `tests/salome/test_specialists_live.py`
- Create: `scripts/install-plugin.sh`, `scripts/start-headless-server.sh`

**Interfaces:**
- Produces:
  - `scripts/start-headless-server.sh` — strips `.venv` from `PATH`, unsets
    `VIRTUAL_ENV`, `exec "$SALOME_ROOT"/salome shell -- python -m agent_server "$@"`.
  - `tests/salome/conftest.py::agent_server` fixture — starts
    `python -m agent_server --serve-forever` under `salome shell` (serve-forever
    so a whole test module can reuse it), waits for `~/.salome_agent/server.json`,
    yields an `ExecutorClient(host, port, token, run_id="test")`, then
    `client.shutdown()` + `proc.terminate()`.

- [ ] **Step 1: Write the new fixture first (failing)**

Rewrite `tests/salome/conftest.py`'s `executor_conn` fixture as `agent_server`:

```python
@pytest.fixture
def agent_server():
    if SALOME_ROOT is None:
        pytest.skip("set SALOME_ROOT or salome.root in config.json")
    import json as _json
    hs = os.path.expanduser("~/.salome_agent/server.json")
    if os.path.exists(hs):
        os.unlink(hs)
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    proc = subprocess.Popen(
        [f"{SALOME_ROOT}/salome", "shell", "--", "python3", "-m", "agent_server",
         "--serve-forever"],
        cwd=repo_root, env=_salome_subprocess_env(),
    )
    for _ in range(300):
        if os.path.exists(hs):
            break
        time.sleep(0.2)
    else:
        proc.terminate()
        raise TimeoutError("agent_server never wrote its handshake")
    data = _json.loads(open(hs).read())
    from salome_agent.executor_client import ExecutorClient
    client = ExecutorClient(data["host"], data["port"], data["token"], "test")
    yield client
    try:
        client.shutdown()
    except Exception:
        pass
    proc.terminate()
    proc.wait(timeout=15)
```

Update `tests/salome/test_executor.py` + `test_geom_tools.py` to take
`agent_server` instead of `executor_conn` and call `client.run(...)` /
`client.call_tool(...)` instead of the local `send()` helper. `test_end_to_end.py`
and `test_specialists_live.py`: they call `cli.main()` — set up the same
`--serve-forever` server in a fixture and let `cli` discover it via the
handshake (no code change to `cli` needed since it reads
`~/.salome_agent/server.json`).

- [ ] **Step 2: Delete the dead files**

```bash
git rm -r salome_agent/session
git rm executor/executor.py
git rm tests/unit/test_session_manager.py tests/unit/test_discovery.py tests/unit/test_executor_live.py
git rm tests/salome/test_live_gui.py
```

- [ ] **Step 3: Write the scripts**

`scripts/start-headless-server.sh`:
```bash
#!/usr/bin/env bash
set -euo pipefail
: "${SALOME_ROOT:?set SALOME_ROOT to your SALOME install root}"
# The agent's own venv python lacks psutil + SALOME's sys.path; strip it.
export PATH="$(printf '%s' "$PATH" | tr ':' '\n' | grep -v '\.venv' | paste -sd:)"
unset VIRTUAL_ENV
cd "$(dirname "$0")/.."
exec "$SALOME_ROOT/salome" shell -- python3 -m agent_server "$@"
```

`scripts/install-plugin.sh`:
```bash
#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PLUGDIR="${HOME}/.config/salome/Plugins"
mkdir -p "$PLUGDIR"
PLUGFILE="$PLUGDIR/salome_plugins.py"
LINE='import salome_agent_plugin  # noqa  — SALOME-agent server launcher'
touch "$PLUGFILE"
grep -qF 'salome_agent_plugin' "$PLUGFILE" || echo "$LINE" >> "$PLUGFILE"
echo "Plugin registered in $PLUGFILE"
echo
echo "Add this to the environment that launches SALOME:"
echo "  export SALOME_PLUGINS_PATH=\"$REPO_ROOT/plugin:$REPO_ROOT\""
```

`chmod +x scripts/*.sh`.

- [ ] **Step 4: Run the full unit suite + shell lint**

Run: `python -m pytest tests/unit -q`
Expected: PASS, no collection errors (all references to `session` / `executor.executor`
are gone).

Run: `bash -n scripts/install-plugin.sh scripts/start-headless-server.sh`
Expected: no output.

- [ ] **Step 5: Run the SALOME suite (if `SALOME_ROOT` set)**

Run: `SALOME_ROOT=... python -m pytest tests/salome -q`
Expected: the headless end-to-end "make a box" passes; `test_gui_marshal.py`
skips (not under `salome shell`).

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "refactor: delete salome_agent/session + executor/executor.py; retarget SALOME fixtures to agent_server; add scripts"
```

---

## Task 16: docs — README + base-spec note

**Files:**
- Modify: `README.md`, `docs/superpowers/specs/2026-08-29-salome-agent-design.md`

- [ ] **Step 1: README**

Replace the `--live-gui` table row and its "Live GUI" prose section with a
**GUI plugin setup** section:

```markdown
## GUI plugin setup

1. `scripts/install-plugin.sh` — registers the plugin and prints the
   `SALOME_PLUGINS_PATH` you need.
2. `export SALOME_PLUGINS_PATH="<repo>/plugin:<repo>"` in the shell that
   launches SALOME.
3. Launch `"$SALOME_ROOT"/salome` (single-process is fine), then
   **Tools ▸ Plugins ▸ Launch SALOME agent server**. A status message shows
   `127.0.0.1:<port>`.
4. Run the agent normally — it finds the server via
   `~/.salome_agent/server.json`.

Objects a run publishes appear in the Object Browser and 3D viewer, prefixed
`sa_<run_id>_`. A failed stage attempt can leave debris under that prefix —
`report.md` says so; bulk-delete by prefix.

**Headless (CI / batch):** `scripts/start-headless-server.sh` (needs
`SALOME_ROOT`) — isolated study, no GUI. Do **not** run `python -m agent_server`
directly from an activated venv; the wrapper strips `.venv` from `PATH`.

**Demo-only / stability:** the live-GUI path can leave the SALOME session
unstable after a run (see task 181). A stage timeout **poisons** the server —
the run aborts and you restart it from the same menu (or, if the GUI is frozen,
kill it).
```

Drop the `--port` and `--live-gui` rows from the flags table. Note `--no-session`
is planner + specialists + static checks only.

- [ ] **Step 2: Base spec note**

In `docs/superpowers/specs/2026-08-29-salome-agent-design.md`, near the executor
description (~line 139), add:

```markdown
> **Superseded (2026-08-31):** the `salome shell -- executor.py` subprocess and
> the `--live-gui` `salome remote` attach are replaced by the in-GUI plugin
> server — see `2026-08-31-gui-plugin-server-design.md`.
```

- [ ] **Step 3: Commit**

```bash
git add README.md docs/superpowers/specs/2026-08-29-salome-agent-design.md
git commit -m "docs: README GUI plugin setup; supersede note on the old executor path"
```

---

## Task 17: full-suite green + branch wrap

- [ ] **Step 1: Unit suite**

Run: `python -m pytest tests/unit -q`
Expected: all green, no skips beyond the pre-existing ones, no collection errors.

- [ ] **Step 2: grep for stragglers**

Run:
```bash
grep -rn "SessionManager\|discover_port\|--live-gui\|live_gui\|executor/executor\|from executor.executor\|salome_agent.session" salome_agent/ tests/ README.md
```
Expected: no hits (comments in `tests/salome/conftest.py` about `_salome_subprocess_env`
are fine).

- [ ] **Step 3: SALOME suite (if available)**

Run: `SALOME_ROOT=<path> python -m pytest tests/salome -q`
and `"$SALOME_ROOT"/salome shell -- python -m pytest tests/salome/test_gui_marshal.py -q`
Expected: headless end-to-end + geom_tools pass; marshal tests pass under
`salome shell`.

- [ ] **Step 4: Manual GUI checklist** (record results in the PR / commit)

- `install-plugin.sh`, set `SALOME_PLUGINS_PATH`, launch `./salome`.
- Tools ▸ Plugins ▸ Launch SALOME agent server → status shows a port; handshake
  file exists.
- `salome-agent "make a 20mm cube"` → cube appears in the GUI as
  `sa_<run_id>_box_1`; `result.py` runs standalone.
- `salome-agent "make a 20mm cube and mesh it with 5mm tets"` → both stages,
  mesh visible.
- Second menu click while idle → server restarts (new port in the handshake).
- Kill the agent mid-run, start a new run → not permanently "busy".

- [ ] **Step 5: Update project memory + push**

Update the remember-da-project session note and `salome-agent-project.md` memory
with the outcome. Then hand back to the user for review before merging to `main`
(per `feedback-work-by-branch` — merge only when verified).

---

## Self-Review

**Spec coverage:**
- Backend split (spec §Architecture) → Tasks 5, 9. ✓
- `core.py` + poison gate → Task 6. ✓
- `wire.py` (pre-hello timeout, hmac, busy/liveness) → Task 7. ✓
- Handshake (proto, pid_start, log_path, unlink-own-pid) → Task 3. ✓
- Qt marshal via `_Signal` + `QueuedConnection`, `_busy` latch, bounded wait →
  Task 9. ✓
- PySide2 binding (Global Constraint) → Tasks 9, 11. ✓
- Name prefix at `_next_name`, `_counters` reset per session → Tasks 4, 5, 9. ✓
- GUI `find_object` prefixed-then-fallback → Task 9. ✓
- Fatal path (SessionPoisoned, RunState.fatal, graph → assemble, cli exit 1) →
  Tasks 12, 13, 14. ✓
- `--once` default headless + timeout poisons headless → Tasks 5, 10. ✓
- Deletions (`session/`, `executor.py`, dead tests) → Task 15. ✓
- `pyproject.toml` packages → Task 2. ✓
- Scripts (`install-plugin.sh`, `start-headless-server.sh`) → Task 15. ✓
- `report.py` mode line + `SessionHandle` reshape → Task 14. ✓
- README + supersede note → Task 16. ✓
- Phase 0 gates → Task 1 (+ formalized in Task 9). ✓
- `reset_and_replay` kept as headless-real / GUI-refusal dead surface → Tasks 5, 9. ✓

**Gaps accepted (spec "out of scope"):** graph-level rollback wiring; the
`--no-session` geom-stage crash; real MCP protocol; auto-launch.

**Placeholder scan:** no "TBD"/"handle errors appropriately"/"similar to Task N";
every code step has real code.

**Type consistency:** `ExecutorClient(host, port, token, run_id)` used
identically in Tasks 12 (def), 15 (fixture); `Backend` attrs (`name_prefix`,
`namespace`, `geom_bindings`, `poisoned()`) consistent Tasks 5/6/9;
`check_exec` sentinel `"session_poisoned"` consistent Tasks 13; `state["fatal"]`
consistent Tasks 13/14; `handshake.write(port=, token=, mode=, log_path=)`
kwargs consistent Tasks 3/10/11.
