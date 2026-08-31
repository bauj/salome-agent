# GUI plugin server — design

**Date:** 2026-08-31
**Status:** approved (brainstorm), pending implementation plan
**Issue:** GitHub #2 — "avoid corba session connection"
**Branch:** `feat/2-gui-plugin-server`

## Problem

Today the agent reaches a running SALOME in two ways, both fragile:

- **Isolated runs** (`salome shell -- executor/executor.py`): a fresh SSL-mode
  session spawned per run. No CORBA to a live GUI, but a full `salome_init()`
  every time and no way to see results in a GUI.
- **`--live-gui`**: `salome remote -p <port>` attaches a fresh executor process to
  a running `salome withsession` GUI, calling `KernelBasis.setSSLMode(False)` +
  `salome_init()` so `geomBuilder.New()` returns the GUI's own engine. This path
  depends on: discovering the omniORB port from `/tmp/.omniORB_*` config files,
  the `salome remote` launcher, a CORBA attach handshake, and
  `sess.emitMessageOneWay(...)` on `/Kernel/Session` for GUI refresh. It is
  known to destabilise the GUI session (issue: task 181) and rests on CORBA
  internals that SALOME may not maintain long term (issue #2).

Issue #2's suggestion: a SALOME GUI plugin (like the built-in "SALOME shell
session" under Tools ▸ Plugins) that starts a server **inside the GUI's own
Python process**. The agent connects to that server instead of attaching over
CORBA. Because the server runs in the already-initialized GUI interpreter,
`geompy` / the study / `sg` are all live and correct with zero attach ceremony.

## Decisions (from the brainstorm)

1. The in-GUI plugin server becomes the **only** executor transport. The
   `salome shell -- executor.py` subprocess spawn and the `salome remote`
   attach are both removed.
2. Primary target launch mode: **single-process `./salome`** (default launcher,
   fake naming service, in-process GEOM engine). No CORBA anywhere on the agent
   path. GUI refresh uses `sg` directly, not `/Kernel/Session` messages.
3. A **headless variant** of the identical server runs under
   `salome shell -- python -m agent_server` for CI / batch / no-display use.
   The plugin is just the GUI way to start the same server.
4. Startup is **manual**; discovery is via a **handshake file**
   (`~/.salome_agent/server.json`). No auto-launching of SALOME (the
   session-10 lesson: auto-launching `salome withsession` is unreliable).
5. Transport: **TCP on `127.0.0.1`**, ephemeral port, with a **random token**
   the agent must present on connect.
6. The in-GUI server **stays up** for the GUI's lifetime and accepts one agent
   run after another. An agent `shutdown` ends that run's session; the listener
   keeps running. A second menu click, or closing the GUI, stops it.
7. In the GUI server, **all** SALOME calls (geompy, study, publish, refresh) are
   **marshalled onto the Qt main thread**. The server's own thread does socket
   I/O only.
8. Real MCP-protocol compliance (HTTP+SSE / stdio, MCP SDK) is explicitly a
   later evolution, not this work. "MCP" in issue #2 is honoured as an
   interface shape (a server the agent calls), not a protocol dependency —
   consistent with decision 17/18 that collapsed `geom_mcp` into the executor
   to keep heavy deps out of SALOME's interpreter.

## Architecture

### New package: `agent_server/`

| File | Responsibility | Depends on |
|---|---|---|
| `agent_server/core.py` | Command dispatch — `run`, `reset_and_replay`, `introspect`, `list_study`, `sanity`, `list_tools`, `call_tool`, `shutdown`. Moved verbatim from `executor/executor.py`'s `Executor.handle`. Transport- and thread-agnostic: every SALOME call routes through an injected `run_on_main(fn)` callable. Owns per-session namespace + (GUI mode) `published_entries` tracking. | `executor.geom_tools`, stdlib |
| `agent_server/wire.py` | TCP listener on `127.0.0.1:0`. Token check on the `hello` line. Line-delimited JSON framing (`\n`-delimited, one object per line, `id` echoed). Sequential single-client accept loop; fail-fast when a client is already connected. | `core`, stdlib `socket` |
| `agent_server/handshake.py` | Write / read / remove `~/.salome_agent/server.json`. Dir mode `0700`, file `0600`. Stale detection via `pid`. | stdlib |
| `agent_server/gui_plugin.py` | `run_on_main` for the GUI: `QTimer.singleShot(0, slot)` + `threading.Event`. `_emit_refresh` via `sg` on the main thread. `launch(context)` — the plugin entry point. | `core`, `wire`, `handshake`, `SalomePyQt` |
| `agent_server/__main__.py` | Headless entry. `run_on_main` = direct call (process is single-threaded). `build_namespace` does `setSSLMode(True)` + `salome_init()` once at startup. `--once` optional. | `core`, `wire`, `handshake` |
| `agent_server/namespace.py` | `build_namespace(live: bool)` — extracted from `executor.py`. GUI (`live=True`): bind `geompy`/`smesh`/`pvsimple`/`study_ns`/`find_object` against the already-initialized process, **no** `setSSLMode`/`salome_init`. Headless (`live=False`): `setSSLMode(True)` + `salome_init()` then bind. | SALOME API |

### Plugin registration

- Repo ships `plugin/salome_agent_plugin.py`:
  ```python
  import salome_pluginsmanager
  from agent_server.gui_plugin import launch

  salome_pluginsmanager.AddFunction(
      'Launch SALOME agent server',
      'Start the SALOME-agent server in this GUI process',
      launch,
  )
  ```
- SALOME's plugin manager loads exactly one `salome_plugins.py` per search
  directory, and `~/.config/salome/Plugins/` may already hold the user's own.
  So the user adds one line to their `~/.config/salome/Plugins/salome_plugins.py`:
  ```python
  import salome_agent_plugin  # noqa  — SALOME-agent server launcher
  ```
  with `SALOME_PLUGINS_PATH` (or `PYTHONPATH`) containing the repo's `plugin/`
  dir **and** the repo root (so `agent_server` imports inside the GUI
  interpreter).
- `scripts/install-plugin.sh` — idempotent: create `~/.config/salome/Plugins/`,
  append the import line if absent, print the exact `export SALOME_PLUGINS_PATH=`
  the user needs. No magic.

### Deleted

- `salome_agent/session/manager.py`, `salome_agent/session/discovery.py`, the
  whole `salome_agent/session/` package.
- `executor/executor.py` (guts move to `agent_server/`). `executor/geom_tools.py`
  stays — imported by `agent_server/core.py`.
- `cli.py` `--live-gui` flag and its `SessionManager._resolve_live` branch.
- `salome remote` usage, `discover_port()`, `/tmp/.omniORB_*` parsing,
  `KernelBasis.setSSLMode(False)` attach path.
- `tests/unit/test_session_manager.py`.

### Unchanged

`graph.py`, `nodes/*`, `specialists/*`, `codegen.py`, `assembler.py`,
`verify/*`. They only ever saw `ExecutorClient` and the JSON command
vocabulary — both preserved byte-for-byte.

## Handshake file

`~/.salome_agent/server.json`, written on server start, removed on clean stop:

```json
{
  "host": "127.0.0.1",
  "port": 54123,
  "token": "<32 hex chars>",
  "pid": 88214,
  "mode": "gui",
  "salome_port": null,
  "started_at": "2026-08-31T18:00:00Z"
}
```

- `mode`: `"gui"` or `"headless"`.
- `salome_port`: omniORB port if launched via `withsession`, else `null`.
  Informational only — the agent does not connect to it.

**Agent connect sequence:**

1. `read_handshake()` → dict, or `None` if the file is missing.
2. If `pid` is not alive → treat as stale, same error as missing.
3. TCP connect to `host:port`.
4. Send `{"cmd": "hello", "token": "<token>"}` as the first line.
5. Server replies `{"ready": true, "mode": "gui"}` or drops the connection
   (bad token) or replies `{"ready": false, "error": "server busy"}` (a client
   is already connected).
6. On any failure the agent raises `ServerUnavailable` with:
   *"No SALOME agent server running. Start it from Tools ▸ Plugins ▸ Launch
   SALOME agent server, or run `salome shell -- python -m agent_server`."*

## Wire protocol

Unchanged from the current executor:

- One JSON object per line, `\n`-delimited.
- Request carries `id`; response echoes it.
- Commands: `hello` (new), `run`, `reset_and_replay`, `introspect`,
  `list_study`, `sanity`, `list_tools`, `call_tool`, `shutdown`.
- `SessionDied` detection in `ExecutorClient` (`BrokenPipeError`,
  `ConnectionResetError`, EOF-on-read) unchanged.

**Concurrency:** `wire.py` handles one client to EOF, then loops back to
`accept()`. A second connection while one is active gets an immediate
`{"ready": false, "error": "server busy"}` and is closed. Retry logic belongs
to the agent's graph, not the socket.

**Token:** `secrets.token_hex(16)`, regenerated per server start. Without it any
local process could drive arbitrary Python in the GUI process.

## Threading model (GUI mode)

The plugin `launch(context)` function runs on the Qt main thread and must
return promptly or the GUI freezes.

**`launch(context)`:**

1. Build `core.Server` with `run_on_main` = the marshal helper below and
   `namespace = build_namespace(live=True)`.
2. Start `wire.serve(core, on_bound=write_handshake)` on a
   `threading.Thread(daemon=True)`.
3. Return immediately. The GUI event loop keeps running.

**`run_on_main(fn)` — called from the daemon (server) thread:**

```python
box = {}
ev = threading.Event()

def slot():
    try:
        box["r"] = fn()
    except BaseException as e:      # noqa: BLE001 — re-raised on the caller thread
        box["e"] = e
    finally:
        ev.set()

QTimer.singleShot(0, slot)          # queued onto the Qt event loop
ev.wait()                           # server thread parks here
if "e" in box:
    raise box["e"]
return box["r"]
```

Result: socket I/O, JSON parse, and timeout bookkeeping run on the daemon
thread; every geompy / study / `sg` / publish call runs on the Qt main thread,
one at a time, with the server thread blocked. SALOME is only ever touched from
the thread the GUI itself uses.

**Cost:** the GUI is unresponsive for the duration of a stage `exec` (seconds
by design). `_run_with_timeout` still runs its watchdog on the daemon side and
can report `timeout after Ns` to the agent, though — as today with a C-level
`Compute()` — it cannot kill work already running on the main thread.

**`_emit_refresh`:** runs inside the marshalled `fn`, already on the main
thread → call `sg.updateObjBrowser()` (and, for display,
`salome.sg.updateObjBrowser()` / `ImportComponentGUI` as needed) directly. No
`emitMessageOneWay`, no `/Kernel/Session` resolve, no CORBA.

**Headless mode:** `run_on_main = lambda fn: fn()` (process is single-threaded).
`_emit_refresh` is a no-op.

## Run / session semantics

The server persists across agent runs. Each `salome-agent "..."` invocation is
one **session** on that server, opened by `hello` and closed by `shutdown`.

### GUI mode — live study

- On `hello`: start a fresh session — new `namespace` (bind against the running
  process, no re-init), empty `published_entries` list.
- `run` / `call_tool`: after a successful step, append any newly published study
  entries to `published_entries`.
- `reset_and_replay` (stage retry / rollback): the user's study must not be
  wiped. Instead — remove the objects in `published_entries` (`geompy` remove /
  study remove), clear the list, rebuild the namespace binding, then replay the
  accepted scripts, re-tracking as it goes.
  - **Fallback:** if remove-by-entry proves unreliable (the earlier
    `PutToFolder` / remove gotcha on this build), GUI mode returns
    `{"ok": false, "error": "cannot roll back in a live GUI study; restart the
    run"}` for `reset_and_replay`. The agent surfaces that as a run failure.
    Acceptable degradation for a live-GUI convenience mode.
- On `shutdown`: drop the session namespace + `published_entries`. Leave the
  published objects in the study — they are the user's result. Listener keeps
  running.

### Headless mode — isolated study

- `build_namespace(live=False)` runs `setSSLMode(True)` + `salome_init()` once
  at `python -m agent_server` startup.
- `reset_and_replay` = full `build_namespace()` rebuild (exactly as
  `executor.py` does today). No entry tracking.
- `shutdown` ends the session; the process stays alive for the next one, unless
  started with `--once`.

### Agent side

- The graph already branches live-vs-isolated for a few log/report lines. The
  `mode` value from the handshake file feeds that branch.

## Agent-side changes

### New: `salome_agent/server_conn.py`

- `read_handshake() -> dict | None` — missing file or dead `pid` → `None`.
- `connect() -> ExecutorClient` — TCP connect, send `hello` + token, wait for
  `ready`. Raises `ServerUnavailable` (with the actionable message) on any
  failure. No process spawning, no port discovery, no `salome_root`.

### `salome_agent/executor_client.py`

- Framing unchanged. Swap `AF_UNIX` connect for `AF_INET` + the `hello`
  handshake line before the first real request.
- `SessionDied` detection unchanged.

### `salome_agent/cli.py`

- Remove `--live-gui`. Remove the `--no-session` executor-spawn path; keep
  `--no-session` as pure-static (no server connect at all).
- Default path: `server_conn.connect()`. On `ServerUnavailable`, exit 1 with the
  actionable message.
- `mode` from the handshake → the one log line and the `report.md` header
  ("live GUI study" vs "isolated study").
- No `SessionManager.stop()`. The agent sends `shutdown` (ends the session,
  server lives on) and closes the socket.

### `salome_agent/state.py`

- `SessionHandle` shrinks to `mode`, `pid`, `salome_port`. Drop `owned`,
  `executor_pid`, and the launch bookkeeping fields.

## Packaging / install

- `plugin/salome_agent_plugin.py` — the `AddFunction` registration (above).
- `scripts/install-plugin.sh` — idempotent installer, prints the
  `SALOME_PLUGINS_PATH` export.
- README gains a "GUI plugin setup" section: run the installer, set
  `SALOME_PLUGINS_PATH`, launch `./salome`, Tools ▸ Plugins ▸ **Launch SALOME
  agent server**, then run the agent normally. Plus the headless one-liner
  (`salome shell -- python -m agent_server`) for CI / batch.
- No SALOME install path baked into any shipped file (per the repo convention):
  the installer resolves `$SALOME_ROOT` / `salome.root` from `config.json`, docs
  use `/path/to/your/SALOME`.

## Testing

| Layer | Approach |
|---|---|
| `agent_server/core.py` | Unit, no SALOME. Inject a fake `run_on_main` + fake namespace; assert command dispatch, `published_entries` tracking, GUI-mode reset logic and its fallback. |
| `agent_server/wire.py` | Unit. Real loopback TCP, fake `core`. Assert token reject, framing, sequential single-client, `server busy` fail-fast. |
| `agent_server/handshake.py` | Unit. Tmp `$HOME`. Write / read / stale-pid / remove. |
| `salome_agent/server_conn.py` + `executor_client.py` | Unit. Fake server on loopback. Assert `hello` handshake, `ServerUnavailable`, `SessionDied`. |
| headless integration | `tests/salome/` — `python -m agent_server` under `salome shell`, real geompy, "make a box" end to end, assert the assembled `result.py` runs standalone. Skips without `SALOME_ROOT`. |
| GUI integration | Manual, documented checklist (a GUI click cannot be automated — same position as `--live-gui` today). Gated by the Phase 0 spike. |

Existing 69 unit tests: `test_session_manager.py` is deleted; the rest run
untouched.

## Phasing

### Phase 0 — spike (gates everything else)

- Plugin registers; the entry appears under Tools ▸ Plugins on single-process
  `./salome`.
- Daemon-thread socket server + `QTimer.singleShot` marshal: drive it with 50+
  external `MakeBoxDXDYDZ` + `addToStudy` calls. Assert: no crash, objects
  appear in the Object Browser and 3D viewer, GUI stays responsive between
  calls.
- `sg.updateObjBrowser()` called directly on the main thread refreshes the
  browser (no `emitMessageOneWay`).
- **Verdict:** proceed as designed, or fall back (`emitMessageOneWay` for
  refresh / rethink the threading split).

### Phase 1 — build

One branch, `feat/2-gui-plugin-server`. `agent_server/` package, plugin file,
installer, agent-side rewiring, and the deletion of `session/` + `--live-gui` +
`executor.py` all land together. No deprecation window — this is a prototype
with a single user.

## Risks

| Risk | Mitigation |
|---|---|
| Main-thread marshal wedges the GUI on a long `Compute()` | Documented; daemon-side timeout still reports to the agent; stages are short by design. |
| GUI-mode `reset_and_replay` remove-by-entry unreliable (the `PutToFolder` / remove gotcha on this build) | Fallback: refuse rollback, tell the agent to restart the run. |
| `agent_server` not importable inside the GUI interpreter | `install-plugin.sh` prints the exact `SALOME_PLUGINS_PATH`; the Phase 0 spike confirms. |
| Two SALOME GUIs → two servers → one handshake file | Last-write-wins + pid check. Single-user dev tool; documented as unsupported. |
| Single-process `./salome` behaves differently for `/Kernel/Session` | Not relied on — refresh goes through `sg`, not Session messages. Spike confirms. |

## Out of scope

- Real MCP-protocol server (HTTP+SSE / stdio, MCP SDK) — a later evolution.
- Remote / cross-machine connections.
- Multiple concurrent agent clients.
- Auto-launching SALOME from the agent.
- Automated GUI integration tests.
