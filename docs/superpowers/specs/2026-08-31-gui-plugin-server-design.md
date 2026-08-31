# GUI plugin server — design

**Date:** 2026-08-31
**Status:** approved (brainstorm), revised twice after Opus design reviews, pending implementation plan
**Issue:** GitHub #2 — "avoid corba session connection"
**Branch:** `feat/2-gui-plugin-server`

## Problem

Today the agent reaches a running SALOME in two ways, both fragile:

- **Isolated runs** (`salome shell -- executor/executor.py`): a fresh SSL-mode
  session spawned per run. No CORBA attach to a live GUI, but a full
  `salome_init()` every time and no way to see results in a GUI.
- **`--live-gui`**: `salome remote -p <port>` attaches a fresh executor process
  to a running `salome withsession` GUI, calling `KernelBasis.setSSLMode(False)`
  + `salome_init()` so `geomBuilder.New()` returns the GUI's own engine. This
  path depends on: discovering the omniORB port from `/tmp/.omniORB_*` config
  files, the `salome remote` launcher, a CORBA attach handshake, and
  `sess.emitMessageOneWay(...)` on `/Kernel/Session` for GUI refresh. It is
  known to destabilise the GUI session (task 181) and rests on CORBA internals
  SALOME may not maintain long term (issue #2).

Issue #2's suggestion: a SALOME GUI plugin (like the built-in "SALOME shell
session" under Tools ▸ Plugins) that starts a server **inside the GUI's own
Python process**. The agent connects to that server instead of attaching over
CORBA. Because the server runs in the already-initialized GUI interpreter,
`geompy` / the study / `sg` are all live and correct with no attach ceremony
and no naming-service discovery.

## What this removes — and what it does not

**Removed:** the out-of-process attach (`salome remote`, `setSSLMode(False)`,
a second `salome_init()` in a fresh process), the naming-service /
`/tmp/.omniORB_*` port discovery, and the `/Kernel/Session` `emitMessageOneWay`
refresh path. These are the fragile parts.

**Not removed:** CORBA itself. `geomBuilder.New()` returns a CORBA client to a
GEOM servant even in single-process `./salome` — it is merely *colocated*
in-process behind a fake naming service rather than reached remotely through
omniNames. Issue #2's broader concern ("CORBA may not be maintained") is only
*partly* addressed by this work: the attach and discovery fragility goes away;
`geompy` remains a CORBA client. A full move off CORBA is a separate, larger
piece of work and is out of scope here.

## Decisions

1. The in-GUI plugin server becomes the **only** executor transport. The
   `salome shell -- executor.py` subprocess spawn and the `salome remote`
   attach are both removed.
2. Primary target launch mode: **single-process `./salome`** (default launcher,
   fake naming service, colocated GEOM engine). GUI refresh uses `sg` directly.
3. A **headless variant** runs the *same code, a different backend* under
   `salome shell -- python -m agent_server` for CI / batch / no-display use.
   The plugin is just the GUI way to start it.
4. Startup is **manual**; discovery is via a **handshake file**
   (`~/.salome_agent/server.json`). No auto-launching of SALOME (session-10
   lesson: auto-launching `salome withsession` is unreliable).
5. Transport: **TCP on `127.0.0.1`**, ephemeral port, with a **random token**
   the agent presents on connect.
6. The in-GUI server **stays up** for the GUI's lifetime. A second menu click
   **restarts** it (drops any client, regenerates the token, rewrites the
   handshake). Closing the GUI stops it.
7. In the GUI backend, **all** SALOME calls are **marshalled onto the Qt main
   thread** via a cross-thread Qt signal (not `QTimer.singleShot`), one call in
   flight at a time (a `_busy` lock), with a bounded wait. Qt symbols are bound
   through `SalomePyQt.usePySide()` — **this build is PySide2, not PyQt5**.
8. **There is no cancellation.** Nothing can interrupt Python already running on
   the Qt main thread. A stage timeout **poisons the server** (both backends —
   see decision 9). The agent surfaces a fatal run error and exits non-zero.
   Timeouts are never retryable.
9. **Poison is server-scoped and permanent until restart.** Once a `run_on_main`
   call (GUI) or an abandoned watchdog thread (headless) times out, the server
   refuses *every* subsequent command **including `hello`** with a fatal error.
   Recovery: GUI — 2nd menu click if the main thread is responsive, else kill
   the GUI (unsaved study lost); headless — the process exits (`--once` is the
   default for `scripts/start-headless-server.sh`) or must be killed.
10. **GUI mode does not roll back.** `reset_and_replay` is headless-only. Failed
    stage attempts leave debris in the user's study, prefixed `sa_<run_id>_`;
    the run report says so. (`reset_and_replay` has zero callers today anyway.)
11. Real MCP-protocol compliance (HTTP+SSE / stdio, MCP SDK) is a later
    evolution, not this work. "MCP" in issue #2 is honoured as an interface
    shape, not a protocol dependency — consistent with decision 17/18 that
    collapsed `geom_mcp` into the executor to keep heavy deps out of SALOME's
    interpreter.

## Architecture

### The Backend split

`core.py` dispatches commands and **never inspects a `live` flag**. All
divergence lives behind a `Backend` protocol:

```python
class Backend(Protocol):
    name_prefix: str                       # "" headless, "sa_<run_id>_" GUI
    def run_code(self, code: str, timeout: float) -> dict: ...
    def refresh(self, module: str | None) -> None: ...      # no-op headless
    def reset(self, scripts: list[str]) -> dict: ...        # GUI: refuse
    def find_object(self, name: str): ...
    def build_namespace(self) -> dict: ...
    def poisoned(self) -> bool: ...
```

- `HeadlessBackend` — `setSSLMode(True)` + `salome_init()` once at process
  start; `run_code` keeps the `_run_with_timeout` watchdog thread **but** a
  timeout sets its own poison latch (the abandoned thread is why — see risks);
  `reset` = full `build_namespace()` rebuild + replay; `refresh` no-op;
  `find_object` = whole-study search; `name_prefix = ""`.
- `GuiBackend` — owns the `MainThreadInvoker`; `run_code` marshals `exec` onto
  the Qt main thread with a single bounded `ev.wait(timeout)` (no inner
  watchdog thread — it could not act on the result anyway); a `TimeoutError`
  sets the poison latch; `reset` returns
  `{"ok": false, "error": "no rollback in a live GUI study"}`; `refresh` =
  `salome.sg.updateObjBrowser()` (already on the main thread inside a marshalled
  call); `find_object` = **try `name_prefix + name` first, then fall back to a
  whole-study search** (so nothing breaks, but the agent *can* still bind a
  user's pre-existing `Box_1` — documented); `name_prefix = "sa_<run_id>_"`.

`core` holds one `Backend`, checks `backend.poisoned()` before dispatching any
command (returns the fatal error if set), and routes `run` / `reset` /
`introspect` / `list_study` / `sanity` / `call_tool` through it.

### New package: `agent_server/`

| File | Responsibility | Imports |
|---|---|---|
| `agent_server/core.py` | Command dispatch + poison gate. Holds a `Backend`. **Must import with no PySide2 / SalomePyQt present** (unit-tested) — it references only the `Backend` protocol. | `executor.geom_tools`, stdlib |
| `agent_server/backend.py` | `Backend` protocol + `HeadlessBackend`. No Qt imports. | `agent_server.namespace`, `executor.geom_tools` |
| `agent_server/gui_backend.py` | `GuiBackend` + `MainThreadInvoker`. Qt symbols via the `usePySide()` branch. | `SalomePyQt`, PySide2/PyQt5, `agent_server.backend` |
| `agent_server/namespace.py` | `build_namespace(live)`. `live=True`: `salome.salome_init()` **defensively** (guarded — `salome/kernel/salome/__init__.py:156` early-returns when `lcc` is set; `salome_pluginsmanager.py:256` already calls it during `importPlugins()`), no `setSSLMode`, then bind. `live=False`: `setSSLMode(True)` + `salome_init()` then bind. Carries the `sys.path` lesson (below). | SALOME API |
| `agent_server/wire.py` | TCP listener on `127.0.0.1:0`. Pre-`hello` socket read timeout (5s). Token check with `hmac.compare_digest`. Line-delimited JSON (`\n`, one object per line, `id` echoed). One client at a time. Incumbent-liveness via a `last_activity` timestamp the serving loop updates (no peek race); a stale incumbent is reclaimed, a live one → new client gets `{"ready": false, "error": "busy"}`. | `core`, stdlib `socket` |
| `agent_server/handshake.py` | Write / read / remove `~/.salome_agent/server.json`. Dir `0700`, file `0600`. Fields incl. `proto`, `pid`, `pid_start`, `log_path`. Unlink only if the file's `pid` is our own. | stdlib |
| `agent_server/gui_plugin.py` | `launch(context)` — the plugin entry. Holds **module-level globals** `_INVOKER`, `_SERVER_THREAD`, `_CORE` (else PySide2 GC's the `QObject` and silently kills the signal connection). Restart-on-2nd-click. Non-modal status message via `sgPyQt.message(...)` (a modal `QMessageBox` is itself a nested-event-loop source). `QApplication.aboutToQuit` → stop + unlink handshake. | `core`, `wire`, `handshake`, `gui_backend` |
| `agent_server/__main__.py` | Headless entry. Builds `core` with `HeadlessBackend`. `--once` **default on** (one session then exit — bounds the abandoned-watchdog hazard); `--serve-forever` to opt out. | `core`, `wire`, `handshake`, `backend` |

**`sys.path` ordering (from `executor/executor.py:11-22`):** `salome_init()`'s
import chain mutates `sys.path` (KERNEL's `extend_path`), dropping an earlier
`sys.path.insert` and breaking a lazy `from executor.geom_tools import ...`. So
`core.py` imports `executor.geom_tools` at module top level, and
`agent_server/__main__.py` puts the repo root on `sys.path` **before** anything
`salome_init`-adjacent. The GUI interpreter needs the repo root on its path via
`SALOME_PLUGINS_PATH` / `PYTHONPATH`. Note `executor/` is not a declared
package — `import executor.geom_tools` resolves only via repo-root-on-path.

### The Qt marshal (GUI backend)

```python
import SalomePyQt
if SalomePyQt.usePySide():
    from PySide2.QtCore import QObject, Signal as _Signal, Qt
else:
    from PyQt5.QtCore import QObject, pyqtSignal as _Signal, Qt


class MainThreadInvoker(QObject):
    _invoke = _Signal(object)                 # object = a _Job

    def __init__(self):
        super().__init__()
        self._lock = threading.Lock()
        self._busy = False
        self._invoke.connect(self._on_invoke, Qt.QueuedConnection)

    def _on_invoke(self, job):                # runs on the Qt main thread
        try:
            job.box["r"] = job.fn()
        except BaseException as e:            # an escape here -> qFatal() -> GUI dies
            job.box["e"] = e
        finally:
            job.ev.set()

    def run_on_main(self, fn, timeout):       # called from the server thread
        with self._lock:
            if self._busy:
                raise RuntimeError("a marshalled call is already in flight")
            self._busy = True
        job = _Job(fn)
        timed_out = False
        try:
            self._invoke.emit(job)            # cross-thread emit — thread-safe
            if not job.ev.wait(timeout):
                timed_out = True
                raise TimeoutError(f"main-thread call exceeded {timeout}s")
            if "e" in job.box:
                raise job.box["e"]
            return job.box["r"]
        finally:
            if not timed_out:                 # leave _busy LATCHED after a timeout
                with self._lock:
                    self._busy = False
```

- Cross-thread **signal emit** is thread-safe; `QTimer.singleShot` from a bare
  `threading.Thread` is not (no Qt event dispatcher in that thread). Phase 0
  Gate A tests this one fact before any other code is written.
- `_busy` stays **latched** after a timeout, so the invoker itself refuses
  further work even if a `core` path forgets to check the poison latch.
- `GuiBackend.run_code` wraps `run_on_main`; on `TimeoutError` it sets the
  server poison latch. `core` checks `backend.poisoned()` (which also refuses
  `hello`) before every dispatch. Because `wire.py` is single-client and the
  serving thread is the one that raised, no new command is read until the
  handler returns — combined with the server-scoped latch there is no window
  for a second run to slip past onto a still-busy main thread.

### Plugin registration

- Repo ships `plugin/salome_agent_plugin.py`:
  ```python
  import salome_pluginsmanager
  from agent_server.gui_plugin import launch

  salome_pluginsmanager.AddFunction(
      'Launch SALOME agent server',
      'Start (or restart) the SALOME-agent server in this GUI process',
      launch,
  )
  ```
- `~/.config/salome/Plugins/` loads one `salome_plugins.py` per dir and may
  hold the user's own, so the user adds one line to theirs:
  ```python
  import salome_agent_plugin  # noqa  — SALOME-agent server launcher
  ```
  with `SALOME_PLUGINS_PATH` containing the repo's `plugin/` dir **and** the
  repo root.
- `scripts/install-plugin.sh` — idempotent: create the dir, append the import
  line if absent, print the exact `export SALOME_PLUGINS_PATH=`. Resolves
  `$SALOME_ROOT` / `salome.root` from `config.json`; no path baked into any
  shipped file.

### Files touched / deleted (complete — grep-verified)

**Deleted:**
- `salome_agent/session/` — whole package.
- `executor/executor.py` — guts move to `agent_server/`.
- `executor/executor_client`... (n/a — see modified).
- `tests/unit/test_session_manager.py`, `tests/unit/test_discovery.py`,
  `tests/unit/test_executor_live.py`.
- `tests/salome/test_live_gui.py`.
- `reset_and_replay` is **kept** as `Backend.reset` (headless real, GUI refusal)
  — deliberately preserved surface for future graph-level rollback (out of
  scope here); `executor_client.reset_and_replay` stays. Stated, not silent.

**Kept unchanged:** `executor/geom_tools.py` (imported by `core.py`).

**Modified:**
- `pyproject.toml` — `packages`: drop `"salome_agent.session"`, add
  `"agent_server"`. (`executor` stays undeclared — resolved via path.)
- `salome_agent/executor_client.py` — `AF_UNIX` → `AF_INET`; send
  `{"cmd": "hello", "token": ..., "run_id": ...}` as the first line; new
  `SessionPoisoned(Exception)` raised on a reply with `"fatal": true`; re-read
  handshake + retry once on a dropped connect (token may have rotated); bounded
  read timeout. `SessionDied` detection unchanged.
- `salome_agent/server_conn.py` — **new.** `read_handshake() -> dict | None`
  (missing file / `proto` mismatch → `None`). `connect(run_id) -> ExecutorClient`
  — TCP connect (a failed connect is the staleness signal; `pid` fields are
  advisory), `hello` with `run_id`, wait `ready`, single token-rotation retry.
  Raises `ServerUnavailable` (busy / absent / poisoned) with an actionable
  message.
- `salome_agent/cli.py` — remove `--live-gui`, `--port`; keep `--no-session` as
  pure-static. Default: `server_conn.connect(run_id)`. Pass `run_id` so the
  server's `name_prefix` (`sa_<run_id>_`, 12 hex — matches `cli.py:51`) and the
  study, `report.md` and `result.py` all name the same thing. Print `log_path`
  from the handshake in the closing summary. `return 1 if final.get("fatal")`.
- `salome_agent/graph.py` — **out of "unchanged".** `_verify_exec_node` and the
  geom-specialist call site in `_run_stage_node` catch `SessionPoisoned` /
  `SessionDied` and `return Command(update={"fatal": ...}, goto="assemble")`
  (the `session_died → goto="assemble"` path at `:150` is the existing model to
  extend). `verify/exec_check.py` gains a `"session_poisoned"` sentinel
  alongside `"session_died"`.
- `salome_agent/specialists/geom.py` — wrap the `client.list_tools()` /
  `client.call_tool()` loop so `SessionDied` / `SessionPoisoned` propagate as a
  fatal, not a burned 12-iteration cap.
- `salome_agent/state.py` — `SessionHandle` becomes
  `{mode: Literal["gui","headless","none"], pid: Optional[int],
  salome_port: Optional[int], name_prefix: str}`. `RunState` gains
  `fatal: Optional[str]`.
- `salome_agent/nodes/report.py` — **new work**: render `state["fatal"]` if
  set; add a mode line ("live GUI study — failed-attempt objects prefixed
  `sa_<run_id>_` may remain" vs "isolated study").
- `salome_agent/config.py` — `("salome","root")` **stays required** (needed by
  `install-plugin.sh` and `tests/salome`). The earlier claim that the agent
  needs no `salome_root` is dropped.
- `tests/unit/test_cli.py` — drop `args.live_gui` / `--port` / mutual-exclusion.
- `tests/unit/test_executor_client.py` — fake server `AF_INET` + `hello`.
- `tests/salome/test_end_to_end.py`, `conftest.py`,
  `test_specialists_live.py`, `test_executor.py`, `test_geom_tools.py` — start
  `scripts/start-headless-server.sh` (or `python -m agent_server --once`) in a
  fixture instead of launching `executor/executor.py` by path.
- `README.md` — replace the `--live-gui` section with "GUI plugin setup";
  document the headless launcher and the `deactivate` caveat; drop `--port`.
- `docs/superpowers/specs/2026-08-29-salome-agent-design.md:139` — one-line
  "superseded by 2026-08-31-gui-plugin-server-design.md" note.

**New:** `agent_server/` (8 modules), `plugin/salome_agent_plugin.py`,
`scripts/install-plugin.sh`, `scripts/start-headless-server.sh` (strips `.venv`
from `PATH`, unsets `VIRTUAL_ENV`, `exec "$SALOME_ROOT"/salome shell -- python
-m agent_server --once "$@"` — carries the deleted `_salome_subprocess_env`
knowledge), `salome_agent/server_conn.py`, `tests/unit/test_agent_server_core.py`,
`test_backend.py`, `test_wire.py`, `test_handshake.py`, `test_server_conn.py`,
`tests/salome/test_gui_marshal.py`.

### Unchanged (logic)

`nodes/*` except `report.py`, `specialists/smesh.py` / `paravis.py`,
`codegen.py`, `assembler.py`, `nodes/gates.py`, `nodes/planner.py`. They only
ever saw `ExecutorClient` and the JSON vocabulary — both preserved. (`graph.py`,
`verify/exec_check.py`, `state.py`, `specialists/geom.py` **are** changed — see
above.)

## Handshake file

`~/.salome_agent/server.json`, written on start, rewritten on restart, removed
on clean stop and on `QApplication.aboutToQuit`:

```json
{
  "proto": 1,
  "host": "127.0.0.1",
  "port": 54123,
  "token": "<32 hex>",
  "pid": 88214,
  "pid_start": 41288213,
  "mode": "gui",
  "salome_port": null,
  "log_path": "/home/user/.salome_agent/server.log",
  "started_at": "2026-08-31T18:00:00Z"
}
```

- `proto`: `connect()` refuses a mismatch (the long-lived server can outlive an
  agent `git pull`).
- `pid` / `pid_start` (`/proc/<pid>/stat` field 22): advisory diagnostics.
  **The TCP connect is the real staleness signal.**
- `log_path`: `~/.salome_agent/server.log`, truncated on each server start. The
  server writes a `--- run <run_id> ---` banner on each `hello` (one file spans
  a whole GUI session). Replaces the per-run `executor.log`.

**Connect sequence (`server_conn.connect(run_id)`):**
1. `read_handshake()` → dict or `None` (missing / `proto` mismatch).
2. TCP connect with a short timeout. Failed connect ⇒ "no server", whatever
   `pid` says.
3. Send `{"cmd": "hello", "token": ..., "run_id": ...}`.
4. `{"ready": true, "proto": 1, "mode": "gui", "name_prefix": "sa_<run_id>_"}`
   — or connection dropped (bad token) — or `{"ready": false, "error": "busy"}`
   — or `{"ready": false, "error": "poisoned", "fatal": true}`.
5. On bad-token / drop: re-read handshake once, retry step 2 once.
6. On any remaining failure raise `ServerUnavailable`:
   *"No usable SALOME agent server (absent / busy / poisoned). Start or restart
   it from Tools ▸ Plugins ▸ Launch SALOME agent server, or run
   `scripts/start-headless-server.sh`."*

## Run / session semantics

The server persists across agent runs (GUI) or one session (`--once` headless).
Each `salome-agent "..."` is one **session**, opened by `hello`, closed by
`shutdown`.

### GUI backend — live study, no rollback, name-prefixed

- On `hello`: `name_prefix = "sa_<run_id>_"` from the agent's `run_id`; fresh
  namespace binding; `geom_tools._counters` **reset** (module global — else
  `result.py` differs run to run).
- Prefixing is applied at **`geom_tools._next_name`** (the single choke point
  used for both `addToStudy` and `geom_bindings` and `codegen`'s variable +
  study name — injecting at publish time would desync them). `sa_<12 hex>_` is a
  valid identifier prefix, so it also prefixes `result.py`'s variable names,
  self-consistently.
- **smesh / paravis specialists write their own names** into raw scripts and
  SMESH publishes them unprefixed — the server never sees them. `GuiBackend.
  find_object` therefore tries `name_prefix + name` first, then a whole-study
  search, so:
  - geom→smesh handoff (gotcha-26) still works — `select_upstream` passes the
    already-prefixed geom `result_names` into the smesh prompt, the LLM writes
    the prefixed literal, `find_object` hits on the first try.
  - smesh→paravis handoff still works — the unprefixed mesh name misses the
    prefixed lookup and is found by the fallback search.
  - Cost: in the fallback path the agent could bind a user's pre-existing
    identically-named object. Rare; documented in the README caveat.
- `reset` (`reset_and_replay`): refused. Acceptable — zero callers today;
  wiring graph-level rollback is out of scope.
- **Debris:** a failed stage attempt that published before failing leaves
  `sa_<run_id>_`-prefixed objects in the user's study; a retry may add more. The
  final `result.py` is still correct (built from the accepted call log, not the
  study). `report.md` says so and names the prefix for bulk cleanup. No worse
  than `README.md:246-248` today.
- Timeout → server poisoned (decision 8/9): all further commands incl. `hello`
  return `{"ok": false, "fatal": true, "error": "session poisoned by a
  timeout"}`. Agent → `fatal` in `RunState` → `assemble` → `report` → `cli`
  exits 1. **Recovery** needs the runaway main-thread job to finish on its own;
  if it never does, the only recovery is killing the GUI (unsaved study lost).
- `shutdown`: drop namespace + prefix; leave objects in the study; listener
  keeps running.

### HeadlessBackend — isolated study

- `build_namespace(live=False)`: `setSSLMode(True)` + `salome_init()` once at
  process start. `name_prefix = ""`. `find_object` = whole-study search.
- `run_code` keeps the `_run_with_timeout` watchdog thread — **but** the
  abandoned-thread-keeps-mutating-namespace hazard (safe before only because
  `SessionManager.stop()` SIGTERM'd the process every run) is bounded by
  `--once` default: the process exits after one session. A timeout still sets
  the poison latch and the process exits non-zero. `--serve-forever` opts out
  and accepts the hazard.
- `reset` = full `build_namespace()` rebuild + replay + `_counters` clear.
- `shutdown` ends the session; `--once` → exit 0.

## Testing

| Layer | Approach |
|---|---|
| `agent_server/core.py` | Real unit, **no PySide2 importable**. Fake `Backend`. Assert: dispatch; poison gate refuses all commands incl. `hello`; `run_id` → `hello` reply. |
| `agent_server/backend.py` (`HeadlessBackend`) | Unit with a fake namespace. `_counters` reset; `reset` rebuild; timeout → poison + would-exit. |
| `agent_server/gui_backend.py` marshal | **`tests/salome/test_gui_marshal.py`** under `salome shell -- pytest` (PySide2 only lives in the SALOME install; a venv PyQt would test a different Qt). **Test #0:** emit from a bare `threading.Thread`, assert it runs on the main thread within 1s, incl. after `gc.collect()`. Then: slot exception surfaces on the caller without abort; bounded-wait `TimeoutError` + `_busy` stays latched; concurrent call rejected. |
| `agent_server/wire.py` | Unit. Loopback TCP, fake `core`. Pre-`hello` timeout; `hmac` reject; framing / `id` echo; incumbent live → `busy`; incumbent stale (old `last_activity`) → reclaimed. |
| `agent_server/handshake.py` | Unit. Tmp `$HOME`. Write / read / `proto` mismatch / `pid_start` mismatch / unlink-only-own-pid / run banner. |
| `server_conn.py` + `executor_client.py` | Unit. Fake server. `hello` + `run_id`; token-rotation retry; `ServerUnavailable` (absent / busy / poisoned); `SessionDied`; `SessionPoisoned`; `proto` mismatch. |
| `graph.py` fatal routing | Unit (existing fake-graph style). `SessionPoisoned` from `verify_exec` and from the geom specialist → `fatal` set → `assemble` → `cli` returns 1. |
| headless integration | `tests/salome/` — `python -m agent_server --once` in a fixture, real geompy, "make a box" end to end, `result.py` runs standalone. Skips without `SALOME_ROOT`. |
| GUI integration | Manual checklist. Gated by Phase 0. |

Net test delta: `test_session_manager.py` + `test_discovery.py` +
`test_executor_live.py` deleted; `test_cli.py` + `test_executor_client.py`
modified; `test_agent_server_core` / `test_backend` / `test_wire` /
`test_handshake` / `test_server_conn` / `test_gui_marshal` added.

## Phasing

### Phase 0 — spike, three independent gates

- **Gate A (architecture gate).** From a bare `threading.Thread` started inside
  a plugin function, get a no-op callable onto the GUI main thread within 1s via
  `_Signal(object)` + `Qt.QueuedConnection`, with the `QObject` held only by a
  module global and a `gc.collect()` in between. Also record
  `SalomePyQt.usePySide()` for this build. **Fallback:** a `QThread`-based
  server — `salome_pluginsmanager`'s handler is an ordinary main-thread
  callback and `QThread.start()` returns immediately, so the plugin still
  returns promptly, and a `QThread` guarantees the affinity `QueuedConnection`
  needs. (The `QThread` object also needs a module global — S4 applies.)
- **Gate B (confirm, ~5 min).** `salome.sg.updateObjBrowser()` from inside a
  marshalled job refreshes the Object Browser. Verified from source that
  `SALOMEGUI_Swig::updateObjBrowser` → `ProcessVoidEvent` short-circuits to a
  direct inline `Execute()` on the session thread (`SALOME_Event.h:215`) — no
  posting, no nested dispatch — so this should just work; the `emitMessageOneWay`
  fallback is almost certainly unneeded.
- **Gate C.** A shape published by the marshalled code appears in the GUI's own
  3D viewer + Object Browser (colocated engine is the GUI's, not a private one).
  **Fallback:** target `withsession` instead of single-process.
- Plus: the Tools ▸ Plugins entry appears on single-process `./salome`.

### Phase 1 — build

One branch. `agent_server/` package, plugin file, both scripts, agent-side
rewiring (incl. the `graph.py` / `state.py` / `report.py` / `specialists/geom.py`
fatal path), and all deletions land together. No deprecation window.

## Risks

| Risk | Mitigation |
|---|---|
| `_Signal` marshal doesn't fire from a bare thread | Phase 0 Gate A first; fallback `QThread`-based server. |
| Wrong Qt binding (PySide2 vs PyQt5) | Bind via `SalomePyQt.usePySide()`, same branch SALOME uses; Gate A records it. |
| Unhandled exception in the marshal slot → `qFatal()` kills the GUI | Airtight `except BaseException` in `_on_invoke`, `ev.set()` in `finally`, unit-tested. |
| `MainThreadInvoker` GC'd → dead signal connection → phantom timeouts | Module-level `_INVOKER` / `_SERVER_THREAD` / `_CORE` in `gui_plugin.py`; Gate A tests it with `gc.collect()`. |
| Stage timeout: uncancellable main-thread work, retry queued behind it | Server poisoned on first `TimeoutError` — every command incl. `hello` refused; `_busy` stays latched; agent treats it as fatal, exits 1. |
| Persistent **headless** server: abandoned watchdog thread mutates the next run's namespace | `--once` default — process exits after one session; timeout also poisons + non-zero exit. `--serve-forever` opts out explicitly. |
| Poison recovery unreachable if the main thread is hung | Documented: recovery needs the runaway job to end on its own, else kill the GUI (unsaved study lost). Reconsider `exec_timeout_s` default for GUI mode. |
| Nested event loop (modal dialog / `processEvents()`) dispatches a queued job mid-interaction | `_busy` lock — one marshalled call ever; plugin uses non-modal `sgPyQt.message`, not `QMessageBox`. |
| GUI `find_object` fallback binds a user's identically-named object | Prefixed lookup tried first; documented in the README caveat; names are `sa_<run_id>_`-prefixed on the geom side. |
| Study debris from failed attempts | `report.md` names the `sa_<run_id>_` prefix for bulk cleanup; no worse than today. |
| Wedged server → "busy" lockout | Pre-`hello` timeout; `last_activity`-based incumbent reclaim; 2nd menu click = restart. |
| Handshake pid reuse | TCP connect is the staleness signal; `proto` + `pid_start` guards; unlink only own pid. |
| `agent_server` not importable in the GUI interpreter | `install-plugin.sh` prints the exact `SALOME_PLUGINS_PATH`; Phase 0 confirms; `sys.path` ordering rule stated. |
| Headless launcher picks the venv python | `scripts/start-headless-server.sh` strips `.venv` / `VIRTUAL_ENV`; README says use it. |
| "zero CORBA" overclaim | Reworded — `geompy` stays a CORBA client; only attach + discovery removed. |

## Out of scope

- A full move off CORBA (`geompy` remains a CORBA client).
- Real MCP-protocol server.
- Wiring `reset_and_replay` into the graph's retry loop (the command is kept as
  deliberately-preserved dead surface).
- Remote / cross-machine; multiple concurrent clients; auto-launching SALOME;
  automated GUI integration tests.
- Fixing the pre-existing `--no-session` + geom-stage crash
  (`specialists/geom.py:27` `client.list_tools()` on `None`, then
  `verify/exec_check.py` `client.run` on `None`) — recorded as a known gap;
  `plan_gate` should reject geom stages under `--no-session`, still
  unimplemented.

## Phase 0 result (2026-08-31)

Ran on `SALOME-master-native-FD44`, single-process `QT_QPA_PLATFORM=xcb ./salome`,
throwaway plugin clicked from Tools ▸ Plugins.

- **Gate A — PASS.** `SalomePyQt.usePySide()` is `True` (PySide2). A `Signal(object)`
  on a `QObject` created in the plugin function, connected `Qt.QueuedConnection`,
  fires from a bare `threading.Thread`: `fired=True ran_on=MainThread`. Also verified
  with the `QObject` held only by a module global + `gc.collect()` before the emit.
  No `QThread` fallback needed.
- **Gate B — PASS, with a correction.** `salome.sg` does **not** exist on this build.
  Both `context.sg.updateObjBrowser()` (the plugin `context`) and
  `libSALOME_Swig.SALOMEGUI_Swig().updateObjBrowser()` work. `gui_backend.py` now uses
  the `libSALOME_Swig` form (no `context` needed on the server thread). No
  `emitMessageOneWay` / `register_module_in_study` required — a plain
  `updateObjBrowser()` after `addToStudy` is enough in single-process mode.
- **Gate C — PASS.** The box published by the marshalled code appeared in the GUI's
  own Object Browser and 3D viewer immediately — the colocated GEOM engine is the
  GUI's, no attach.

Automated proxy: `tests/salome/test_gui_marshal.py` (3 tests) passes under
`salome shell -- pytest`.
