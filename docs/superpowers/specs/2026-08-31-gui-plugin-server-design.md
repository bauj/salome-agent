# GUI plugin server — design

**Date:** 2026-08-31
**Status:** approved (brainstorm), revised after Opus design review, pending implementation plan
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
`geompy` remains a CORBA client. A full move off CORBA (a real in-process API,
or a GEOM-side plugin exposing a non-CORBA surface) is a separate, larger piece
of work and is out of scope here.

## Decisions (from the brainstorm, amended by the review)

1. The in-GUI plugin server becomes the **only** executor transport. The
   `salome shell -- executor.py` subprocess spawn and the `salome remote`
   attach are both removed.
2. Primary target launch mode: **single-process `./salome`** (default launcher,
   fake naming service, colocated GEOM engine). GUI refresh uses `sg` directly,
   not `/Kernel/Session` messages.
3. A **headless variant** of the identical server runs under
   `salome shell -- python -m agent_server` for CI / batch / no-display use.
   The plugin is just the GUI way to start the same server.
4. Startup is **manual**; discovery is via a **handshake file**
   (`~/.salome_agent/server.json`). No auto-launching of SALOME (session-10
   lesson: auto-launching `salome withsession` is unreliable).
5. Transport: **TCP on `127.0.0.1`**, ephemeral port, with a **random token**
   the agent presents on connect.
6. The in-GUI server **stays up** for the GUI's lifetime and accepts one agent
   run after another. An agent `shutdown` ends that run's session; the listener
   keeps running. A second menu click **restarts** the server (drops any client,
   regenerates the token, rewrites the handshake). Closing the GUI stops it.
7. In the GUI server, **all** SALOME calls (geompy, study, publish, refresh) are
   **marshalled onto the Qt main thread** via a cross-thread Qt signal (not
   `QTimer.singleShot`), one call in flight at a time, with a bounded wait.
8. **There is no cancellation.** Nothing can interrupt Python already running on
   the Qt main thread. A stage timeout is therefore **terminal in GUI mode**:
   the session is poisoned, further `run`s are refused, the agent surfaces a run
   failure and exits. Timeouts are not retryable in GUI mode.
9. **GUI mode does not roll back.** `reset_and_replay` is headless-only. Failed
   stage attempts leave debris in the user's study; the run report says so.
   (Rationale below — `reset_and_replay` currently has zero callers anyway.)
10. Real MCP-protocol compliance (HTTP+SSE / stdio, MCP SDK) is a later
    evolution, not this work. "MCP" in issue #2 is honoured as an interface
    shape (a server the agent calls), not a protocol dependency — consistent
    with decision 17/18 that collapsed `geom_mcp` into the executor to keep
    heavy deps out of SALOME's interpreter.

## Architecture

### New package: `agent_server/`

| File | Responsibility | Depends on |
|---|---|---|
| `agent_server/core.py` | Command dispatch — `run`, `reset_and_replay` (headless only), `introspect`, `list_study`, `sanity`, `list_tools`, `call_tool`, `shutdown`. Ported from `executor/executor.py`'s `Executor.handle`. Transport- and thread-agnostic: every SALOME call routes through an injected `run_on_main(fn)` callable. **Must import cleanly with no PyQt / SalomePyQt present** (unit-tested). | `executor.geom_tools`, stdlib |
| `agent_server/wire.py` | TCP listener on `127.0.0.1:0`. Pre-`hello` socket read timeout (5s). Token check with `hmac.compare_digest`. Line-delimited JSON framing (`\n`, one object per line, `id` echoed). One client at a time; a second connection gets `{"ready": false, "error": "busy"}` **only after** the incumbent is probed live, else the incumbent is reclaimed. | `core`, stdlib `socket` |
| `agent_server/handshake.py` | Write / read / remove `~/.salome_agent/server.json`. Dir `0700`, file `0600`. Carries `proto`, `pid`, process start-time, `log_path`. Unlink only if the file's `pid` is our own. | stdlib |
| `agent_server/marshal.py` | The Qt main-thread marshal: a `QObject` subclass created on the main thread inside `launch()`, with a `pyqtSignal(object)` connected `Qt.QueuedConnection`; `run_on_main(fn)` emits and blocks on a bounded `threading.Event`. A `_busy` flag rejects a second in-flight call (guards nested-event-loop re-entrancy). Airtight `except BaseException` in the slot — an escaped exception calls `qFatal()` and kills the GUI. | `SalomePyQt` / PyQt |
| `agent_server/gui_plugin.py` | `launch(context)` — the plugin entry point. Builds `core` with `marshal.run_on_main` and a `live=True` namespace, starts `wire.serve` on a `threading.Thread(daemon=True)`, writes the handshake, posts a `QMessageBox` (host:port on success, the error on failure). Registers `aboutToQuit` cleanup. `_emit_refresh` via `sg` (runs inside an already-marshalled call). | `core`, `wire`, `handshake`, `marshal` |
| `agent_server/__main__.py` | Headless entry. `run_on_main = lambda fn: fn()` (single-threaded process). `--once` optional. `_emit_refresh` is a no-op. | `core`, `wire`, `handshake`, `namespace` |
| `agent_server/namespace.py` | `build_namespace(live: bool)`. `live=True`: call `salome.salome_init()` **defensively** (it is internally guarded and a no-op when already initialized — this is not the sticky-globals hazard, which is about *re-init with different settings*), **no** `setSSLMode`, then bind `geompy` / `smesh` / `pvsimple` / `study_ns` / `find_object`. `live=False`: `setSSLMode(True)` + `salome_init()` then bind. Ported from `executor.py`, keeping the `sys.path` lesson (below). | SALOME API |

**`sys.path` ordering (carried from `executor/executor.py:11-22`):**
`salome_init()`'s import chain mutates `sys.path` (KERNEL's `extend_path`),
silently dropping an earlier `sys.path.insert` and breaking a lazy
`from executor.geom_tools import ...`. So `core.py` must import
`executor.geom_tools` at module top level, and `agent_server/__main__.py` must
put the repo root on `sys.path` **before** calling anything `salome_init`-
adjacent. The GUI interpreter needs the repo root on its path too (via
`SALOME_PLUGINS_PATH` / `PYTHONPATH`).

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
- SALOME's plugin manager loads exactly one `salome_plugins.py` per search
  directory, and `~/.config/salome/Plugins/` may already hold the user's own.
  So the user adds one line to their
  `~/.config/salome/Plugins/salome_plugins.py`:
  ```python
  import salome_agent_plugin  # noqa  — SALOME-agent server launcher
  ```
  with `SALOME_PLUGINS_PATH` (or `PYTHONPATH`) containing the repo's `plugin/`
  dir **and** the repo root (so `agent_server` imports inside the GUI
  interpreter).
- `scripts/install-plugin.sh` — idempotent: create `~/.config/salome/Plugins/`,
  append the import line if absent, print the exact `export SALOME_PLUGINS_PATH=`
  the user needs. Resolves `$SALOME_ROOT` / `salome.root` from `config.json`;
  no install path baked into any shipped file.

### Files touched / deleted (complete)

**Deleted:**
- `salome_agent/session/` — whole package (`manager.py`, `discovery.py`,
  `__init__.py`).
- `executor/executor.py` — guts move to `agent_server/`.
- `tests/unit/test_session_manager.py`
- `tests/unit/test_discovery.py`
- `tests/unit/test_executor_live.py` (imports `executor.executor.Executor`,
  tests `_emit_refresh`).
- `tests/salome/test_live_gui.py` (whole file — the `--live-gui` path).

**Kept, unchanged:** `executor/geom_tools.py` (imported by `agent_server/core.py`).

**Modified:**
- `salome_agent/executor_client.py` — `AF_UNIX` → `AF_INET`; send `hello` +
  token as the first line; re-read the handshake and retry once on a failed
  connect (token may have rotated); bounded socket read timeout. `SessionDied`
  detection (`BrokenPipeError` / `ConnectionResetError` / EOF) unchanged.
- `salome_agent/cli.py` — remove `--live-gui`, remove `--port`, remove the
  `--no-session` executor-spawn path (keep `--no-session` as pure-static).
  Default path: `server_conn.connect()`. On `ServerUnavailable`, exit 1 with the
  actionable message. Print `log_path` from the handshake in the closing
  summary. No `SessionManager.stop()` — send `shutdown`, close the socket.
- `salome_agent/state.py` — `SessionHandle` becomes
  `{mode: Literal["gui", "headless", "none"], pid: Optional[int],
  salome_port: Optional[int]}`. Drop `owned`, `executor_pid`, `port`.
- `salome_agent/nodes/report.py` — **new small work**: read `state["session"]["mode"]`
  and add one header line ("live GUI study — failed-attempt objects may remain"
  vs "isolated study"). The claim that this branching already exists was wrong.
- `salome_agent/config.py` — `("salome", "root")` **stays required**: it is
  needed by `scripts/install-plugin.sh` and the `tests/salome` suite. The spec
  no longer claims the agent needs no `salome_root`; it needs it for install
  and tests, not for the connection.
- `tests/unit/test_cli.py` — drop `args.live_gui` / `--port` / mutual-exclusion
  assertions.
- `tests/unit/test_executor_client.py` — fake server becomes `AF_INET` + `hello`.
- `tests/salome/conftest.py`, `tests/salome/test_specialists_live.py`,
  `tests/salome/test_executor.py`, `tests/salome/test_geom_tools.py` — launch
  `python -m agent_server` (headless) instead of `executor/executor.py` by path;
  drop the duplicated `_salome_subprocess_env` comment or point it at
  `scripts/start-headless-server.sh`.
- `README.md` — replace the `--live-gui` section with "GUI plugin setup";
  document the headless one-liner and the `deactivate`/wrapper-script caveat;
  drop the `--port` row.

**New:**
- `agent_server/` package (7 modules above).
- `plugin/salome_agent_plugin.py`
- `scripts/install-plugin.sh`
- `scripts/start-headless-server.sh` — strips `.venv` from `PATH`, unsets
  `VIRTUAL_ENV`, then `exec "$SALOME_ROOT"/salome shell -- python -m agent_server "$@"`.
  (This is where the deleted `_salome_subprocess_env` knowledge now lives.)
- `salome_agent/server_conn.py`
- `tests/unit/test_agent_server_core.py`, `test_wire.py`, `test_handshake.py`,
  `test_server_conn.py`.

### Unchanged (logic)

`graph.py`, `nodes/*` except `report.py`, `specialists/*`, `codegen.py`,
`assembler.py`, `verify/*`. They only ever saw `ExecutorClient` and the JSON
command vocabulary — both preserved.

## Handshake file

`~/.salome_agent/server.json`, written on server start, rewritten on restart
(2nd menu click), removed on clean stop and on `QApplication.aboutToQuit`:

```json
{
  "proto": 1,
  "host": "127.0.0.1",
  "port": 54123,
  "token": "<32 hex chars>",
  "pid": 88214,
  "pid_start": 41288213,
  "mode": "gui",
  "salome_port": null,
  "log_path": "/home/user/.salome_agent/server.log",
  "started_at": "2026-08-31T18:00:00Z"
}
```

- `proto`: protocol version. `connect()` refuses a mismatch with an actionable
  message (the long-lived GUI server can outlive an agent `git pull`).
- `pid` / `pid_start`: `pid_start` is `/proc/<pid>/stat` field 22, to defeat
  pid reuse. But see below — **the TCP connect is the real staleness signal.**
- `mode`: `"gui"` or `"headless"`.
- `salome_port`: omniORB port if launched via `withsession`, else `null`.
  Informational only.
- `log_path`: `~/.salome_agent/server.log` (truncated on each server start).
  Replaces the deleted `executor.log`. SALOME-side stderr / vtk / paraview
  chatter goes here.

**Agent connect sequence (`server_conn.connect()`):**

1. `read_handshake()` → dict, or `None` if the file is missing / `proto`
   mismatch.
2. TCP connect to `host:port` with a short timeout. **A failed connect is
   treated as "no server" regardless of what `pid` says** — TCP is ground
   truth; the pid fields are advisory diagnostics only.
3. Send `{"cmd": "hello", "token": "<token>"}` as the first line.
4. Server replies `{"ready": true, "proto": 1, "mode": "gui"}`, or drops the
   connection (bad token), or `{"ready": false, "error": "busy"}`.
5. On bad-token or a dropped connection, **re-read the handshake once** (the
   token may have rotated via a restart) and retry step 2 once.
6. On any remaining failure, raise `ServerUnavailable`:
   *"No SALOME agent server running. Start it from Tools ▸ Plugins ▸ Launch
   SALOME agent server, or run `scripts/start-headless-server.sh`."*

**"busy" handling.** `wire.py` reports `busy` only after probing the incumbent
connection (a zero-byte send / `recv` with `MSG_PEEK`); a dead incumbent is
reclaimed and the new client proceeds. A live incumbent → the new agent gets
`ServerUnavailable` with *"another agent run is in progress"*. The manual escape
hatch is the 2nd menu click (restart). `cli.py` does **not** auto-retry a busy
server.

## Threading model (GUI mode)

`launch(context)` runs on the Qt main thread and must return promptly.

**`launch(context)`:**
1. If a server thread from a prior click is alive → stop it (close its socket,
   join briefly), so this click is a clean **restart**.
2. Create the `marshal.MainThreadInvoker` `QObject` **on this (main) thread**.
3. Build `core` with `run_on_main = invoker.run_on_main` and
   `namespace = build_namespace(live=True)`.
4. Start `wire.serve(core, on_bound=write_handshake, log_path=...)` on a
   `threading.Thread(daemon=True)`. Wait (bounded) for the bind so the
   `QMessageBox` can show the real port.
5. `QMessageBox` — `127.0.0.1:<port>` on success, the exception on failure.
6. Connect `QApplication.instance().aboutToQuit` → stop thread + remove
   handshake.

**`marshal.MainThreadInvoker`:**
```python
class MainThreadInvoker(QObject):
    _invoke = pyqtSignal(object)          # object = a zero-arg callable

    def __init__(self):
        super().__init__()
        self._busy = False
        self._invoke.connect(self._on_invoke, Qt.QueuedConnection)

    def _on_invoke(self, job):            # runs on the main thread
        box, ev = job.box, job.ev
        try:
            box["r"] = job.fn()
        except BaseException as e:        # an escape here → qFatal() → GUI dies
            box["e"] = e
        finally:
            ev.set()

    def run_on_main(self, fn, timeout):   # called from the server thread
        if self._busy:
            raise RuntimeError("a marshalled call is already in flight")
        self._busy = True
        try:
            job = _Job(fn)
            self._invoke.emit(job)        # cross-thread signal emit — thread-safe
            if not job.ev.wait(timeout):
                raise TimeoutError(f"main-thread call exceeded {timeout}s")
            if "e" in job.box:
                raise job.box["e"]
            return job.box["r"]
        finally:
            self._busy = False
```

- Cross-thread **signal emit** is documented thread-safe; `QTimer.singleShot`
  from a bare `threading.Thread` is **not** — it needs a Qt event dispatcher in
  the calling thread, which a `threading.Thread` does not have. This is the
  fix for review finding C1.
- `_busy` rejects a second concurrent marshal — guards the nested-event-loop
  case (a modal dialog or an internal `processEvents()` inside a GEOM op runs a
  nested loop that would otherwise dispatch a queued job while the user is
  mid-interaction). One in flight, ever.
- `job.ev.wait(timeout)` is **bounded**. On expiry `run_on_main` raises
  `TimeoutError` — but the job may still be running on the main thread. See
  the timeout rule below.

**Timeout is terminal in GUI mode (review finding C3).** There is no way to
kill work already on the Qt main thread — the old `SessionManager.stop()`
SIGTERM'd the whole executor process; that mechanism is gone. So:

- The per-stage watchdog thread inside `_run_with_timeout` is **dropped in GUI
  mode**. The single bound is `run_on_main`'s `ev.wait(policy.exec_timeout_s)`.
- On a `TimeoutError` from `run_on_main`, `core` marks the session
  **poisoned**: every subsequent command returns
  `{"ok": false, "error": "session poisoned by a prior timeout; restart the
  agent server"}`. The still-running job is left to finish or hang on the main
  thread — nothing the agent does can help it, and queueing a retry behind it
  (the C3 failure) is exactly what the poison flag prevents.
- The agent treats a poisoned-session error as a **fatal run error** (not a
  retryable exec failure): it stops the graph, writes the report noting the
  incomplete state, exits 1.
- Risk-table entry updated accordingly. `policy.exec_timeout_s` defaults to
  120 — up to two minutes of frozen GUI on the pathological stage, once, then
  the run ends.

**Headless mode** keeps the watchdog thread and `reset_and_replay` exactly as
`executor.py` has them today (`run_on_main` is the identity function).

**`_emit_refresh`:** runs inside an already-marshalled `fn`, i.e. on the main
thread → call `salome.sg.updateObjBrowser()` directly. `sg` is real inside the
GUI interpreter (a no-op stub only *outside* it). Note: `sg` methods
self-marshal (post a `SALOME_Event` and block) — calling one from inside an
already-marshalled slot is a nested dispatch; the Phase 0 spike checks it
rather than assuming. No `emitMessageOneWay`, no `/Kernel/Session` resolve.

## Run / session semantics

The server persists across agent runs. Each `salome-agent "..."` invocation is
one **session**, opened by `hello`, closed by `shutdown`.

### GUI mode — live study, no rollback

- On `hello`: start a fresh session — new `namespace` (bind against the running
  process; `salome_init()` is called defensively and no-ops if already done),
  a fresh **name prefix** `sa_<run_id>_` (8 hex), and `geom_tools._counters`
  **reset** (it is a module global — must be cleared per session or `result.py`
  differs run to run).
- **All published study names are prefixed** with `sa_<run_id>_`
  (`_next_name` and the specialists' explicit names both go through the prefix).
  `find_object` in GUI mode searches **only** names under that prefix — so the
  agent can never bind the user's pre-existing `Box_1`. `codegen` emits the
  prefixed names; `result.py` is self-consistent and deterministic within a run.
- `reset_and_replay`: **not supported in GUI mode.** Returns
  `{"ok": false, "error": "no rollback in a live GUI study"}`. This is
  acceptable because `reset_and_replay` currently has **zero callers** —
  `verify_static` (`graph.py:133`) and `verify_exec` (`graph.py:162`) both jump
  straight back to `run_stage` with no rollback. Wiring real rollback into the
  graph is explicitly **out of scope** for this spec (it would be a `graph.py`
  change with its own tests).
- **Consequence, stated honestly:** a failed stage attempt that published
  objects before failing leaves them in the user's study, prefixed
  `sa_<run_id>_`. A retry re-runs the full stage script and may publish a second
  set. The final `result.py` is still correct (it is generated from the
  accepted call log, not the study), but the live study accumulates debris from
  failed attempts. `report.md` says so, and names the prefix so the user can
  bulk-delete. This matches the pre-existing behaviour documented in
  `README.md:246-248`; the plugin does not make it worse.
- On `shutdown`: drop the session namespace + prefix. Leave published objects in
  the study (the accepted ones are the user's result; the debris is theirs to
  clear). Listener keeps running.
- On a poisoned session (timeout): as above — terminal.

### Headless mode — isolated study

- `build_namespace(live=False)` runs `setSSLMode(True)` + `salome_init()` once
  at `python -m agent_server` startup.
- No name prefixing (the study is private and empty).
- `reset_and_replay` = full `build_namespace()` rebuild + replay, exactly as
  `executor.py` does today. `_counters` cleared on rebuild.
- `shutdown` ends the session; the process stays alive for the next one unless
  started `--once`.

### Agent side

`report.py` gains the mode-aware header line (new work — see Files Modified).
Nothing else in the graph branches on mode.

## Testing

| Layer | Approach |
|---|---|
| `agent_server/core.py` | Unit, no SALOME, **no PyQt importable**. Fake `run_on_main` + fake namespace. Assert: command dispatch; `_counters` reset on `hello`; name-prefix application and prefix-scoped `find_object`; `reset_and_replay` refused in GUI mode / works in headless; poisoned-session latch after a `TimeoutError`. |
| `agent_server/marshal.py` | Unit with a real `QApplication` (`pytest-qt` or a minimal app). **Test #0:** emit from a bare `threading.Thread`, assert the callable runs on the main thread within 1s. Then: exception in the job surfaces on the caller thread without aborting; bounded-wait `TimeoutError`; `_busy` rejects a concurrent call. |
| `agent_server/wire.py` | Unit. Real loopback TCP, fake `core`. Assert: pre-`hello` read timeout; `hmac` token reject; framing / `id` echo; second connection while incumbent live → `busy`; second connection while incumbent dead → reclaimed. |
| `agent_server/handshake.py` | Unit. Tmp `$HOME`. Write / read / `proto` mismatch / unlink-only-own-pid / `pid_start` mismatch. |
| `salome_agent/server_conn.py` + `executor_client.py` | Unit. Fake server on loopback. Assert: `hello` handshake; token-rotation re-read + single retry; `ServerUnavailable` messages; `SessionDied`; `proto` mismatch. |
| headless integration | `tests/salome/` — `scripts/start-headless-server.sh` under CI, real geompy, "make a box" end to end, assert the assembled `result.py` runs standalone. Skips without `SALOME_ROOT`. |
| GUI integration | Manual, documented checklist (a GUI click cannot be automated). Gated by Phase 0. |

Explicit uncovered-before, covered-now: the GUI-mode timeout → poison path; a
second connection mid-marshal; handshake removal on GUI close; pid-reuse
staleness; `proto` mismatch; `core` importing without PyQt.

The "69 unit tests untouched" claim from the first draft was wrong. Net test
delta: `test_session_manager.py` + `test_discovery.py` + `test_executor_live.py`
deleted; `test_cli.py` + `test_executor_client.py` modified; 4 new
`test_agent_server_*` / `test_server_conn` files added.

## Phasing

### Phase 0 — spike, three independent gates (do first)

Only **Gate A** is a genuine architecture gate; B and C have their own
fallbacks.

- **Gate A — the marshal fires at all.** From a bare `threading.Thread` started
  inside a plugin function, get a no-op callable to run on the GUI main thread
  within 1s via the `pyqtSignal` + `Qt.QueuedConnection` mechanism. If this
  fails, the whole in-GUI-server approach needs rethinking (a `QThread`-based
  server, or a different IPC). **Test this before writing any other code.**
- **Gate B — `sg` refresh works.** `salome.sg.updateObjBrowser()` called from
  inside a marshalled job refreshes the Object Browser with a newly published
  shape, no nested-dispatch crash. Fallback: `emitMessageOneWay` (accepts one
  bit of CORBA back for refresh only).
- **Gate C — the colocated engine is the GUI's.** A shape published by the
  marshalled code appears in the GUI's own 3D viewer and Object Browser (not a
  private disconnected engine). Fallback: revisit `withsession` as the target
  mode instead of single-process.
- Plus: register the plugin, confirm the Tools ▸ Plugins entry appears on
  single-process `./salome`.

### Phase 1 — build

One branch, `feat/2-gui-plugin-server`. `agent_server/` package, plugin file,
both scripts, agent-side rewiring, and all deletions land together. No
deprecation window — a prototype with a single user.

## Risks

| Risk | Mitigation |
|---|---|
| The `pyqtSignal` marshal doesn't fire from a bare thread | Phase 0 Gate A tests it before anything else; fallback is a `QThread`-based server. |
| Unhandled exception in the marshal slot → `qFatal()` kills the GUI | Airtight `except BaseException` in `_on_invoke`; `ev.set()` in `finally`; unit-tested. |
| Stage timeout in GUI mode: uncancellable work left on the main thread, retry queued behind it | Session poisoned on first `TimeoutError`; all further commands refused; agent treats it as fatal, not retryable. Watchdog thread dropped in GUI mode. |
| Nested event loop (modal dialog / `processEvents()`) dispatches a queued job mid-interaction | `_busy` flag — one marshalled call in flight, ever; a second raises. |
| GUI mode leaves failed-attempt debris in the user's study | Documented in `report.md` with the `sa_<run_id>_` prefix for bulk cleanup; no worse than today's documented behaviour. |
| Wedged / stranded server → permanent "busy" lockout | Pre-`hello` read timeout; incumbent liveness probe before reporting busy; 2nd menu click = restart (drops client, rotates token). |
| Handshake pid reuse → agent connects to a random port | TCP connect is the staleness signal; pid fields advisory only; `proto` guard. |
| Two GUIs, two servers, one handshake file | Last-write-wins; unlink only own pid; single-user dev tool, documented unsupported. |
| `agent_server` not importable inside the GUI interpreter | `install-plugin.sh` prints the exact `SALOME_PLUGINS_PATH`; Phase 0 confirms; `sys.path` ordering rule in the spec. |
| Headless one-liner picks the venv python (missing `psutil` / SALOME path) | `scripts/start-headless-server.sh` strips `.venv` from `PATH`, unsets `VIRTUAL_ENV`; README says run it, not bare `python`. |
| Debuggability regression from losing `executor.log` | Server writes `~/.salome_agent/server.log` (truncate on start); `log_path` in the handshake; `cli.py` prints it. |
| "zero CORBA" overclaim | Reworded — `geompy` stays a CORBA client; only the attach + discovery are removed. |

## Out of scope

- A full move off CORBA (`geompy` remains a CORBA client).
- Real MCP-protocol server (HTTP+SSE / stdio, MCP SDK).
- Wiring rollback / `reset_and_replay` into the graph's retry loop.
- Remote / cross-machine connections.
- Multiple concurrent agent clients.
- Auto-launching SALOME from the agent.
- Automated GUI integration tests.
- Fixing the pre-existing `--no-session` + geom-stage `AttributeError`
  (`graph.py:145` calls `client.run` with `client is None`) — recorded here as
  a known gap; `plan_gate` should reject geom stages under `--no-session`, still
  unimplemented.
