# salome-agent GUI plugin — design

Date: 2026-09-18 (session 38). Status: approved in brainstorming, spec pending review.

## Goal

A SALOME GUI plugin, modelled on the `sarag` assistant plugin, that from one menu click:

1. starts (or reuses) the in-GUI agent server inside the running SALOME process, and
2. opens a chat window in which each message drives one full salome-agent run
   (plan → stages → verify → `result.py`), with the plan gate and the failed-stage
   gate rendered as in-chat buttons and the run shown as a live stage timeline.

Targets both SALOME 9.16 and master; version differences are absorbed by
import fallbacks (see 596b16e for the KERNEL layout).

## Non-goals (v1)

- Conversational follow-ups that refine the previous result — task 194.
- GUI selection as agent context (object / sub-shape picking) — task 242, after 194.
- Streamed visible reasoning, per-tool-call events inside a stage.
- smesh tool layer (smesh stays free-form script generation).
- A full typed form for every `llm.*` / `policy.*` knob.

## Decisions

| # | Decision | Rationale |
|---|---|---|
| D1 | Chat = one run per message. | Matches what the CLI does today; 194 layers conversation on later without blocking the plugin. |
| D2 | Chat window runs in a **separate process** (own venv), not inside SALOME's Python. | No dependency clash (LangGraph, httpx) with SALOME's embedded Python; an agent hang cannot freeze or crash SALOME. Same model as sarag. |
| D3 | The window **imports `salome_agent` as a library** and drives the graph in-process (approach A). | Smallest change here; timeline events come straight from `graph.stream`; CLI and GUI share one code path. Rejected: per-message CLI subprocess with a JSON-lines protocol (extra wire contract, ~1–2 s startup per message; its only win, hard-kill, is covered by `llm.timeout`); long-lived agent daemon (only pays off with 194 state). |
| D4 | Plugin code lives in a **new repo** with salome-agent as a git submodule, installed by **SAT**. | Mirrors sarag (`raglib` submodule, SAT product). |
| D5 | Config UI = key fields + raw JSON edit. | Covers the common edits without tracking every knob. |
| D6 | Run display = live stage timeline. | Gives plan/stage/verify visibility without dumping logs. |

## Architecture

```
SALOME GUI process                              chat window process (venv)
┌──────────────────────────────────┐            ┌─────────────────────────────────┐
│ salome_plugins.py  (menu entry)  │  spawns    │ salome_agent_gui (PyQt5)        │
│   ├─ agent_server.gui_plugin     │ ─────────► │   MainWindow                    │
│   │    .launch()  (in-GUI server)│            │   RunWorker(QThread)            │
│   └─ start window (PID guard)    │            │     └─ salome_agent.runner      │
│                                  │  TCP +     │          .run_request()         │
│ GuiBackend / MainThreadInvoker   │ ◄───────── │     └─ server_conn.connect()    │
│ handshake file (port, token)     │ handshake  │                                 │
└──────────────────────────────────┘            └─────────────────────────────────┘
```

### Repo A — `salome-agent-gui` (new; name final at repo creation)

```
salome-agent-gui/
  salome-agent/                 # git submodule (this repo)
  salome_plugins.py             # SALOME plugins-manager entry
  salome_agent_gui/
    __init__.py
    app.py                      # QApplication entry point (console script)
    main_window.py              # layout: connection panel, run options, chat, input
    chat_view.py                # QTextBrowser-based chat + timeline rendering
    run_worker.py               # QThread wrapping runner.run_request
    config_store.py             # user config load/save/seed
    dialogs.py                  # raw-JSON config editor, plan JSON editor
  salome-agent-gui.sh           # launcher: activate venv, run console script
  pyproject.toml                # deps: PyQt5 + ./salome-agent (path dependency)
  tests/
```

**`salome_plugins.py`** — registers "SALOME Agent" via
`salome_pluginsmanager.AddFunction`. On click:

1. Start the in-GUI server with `agent_server.gui_plugin.launch(context)`,
   unless a live server owned by this process already exists (reuse, do not
   restart — `launch()` today restarts on every call, so the plugin checks
   `handshake.live_owner()` / its own module state first).
2. Launch the chat window: `subprocess.Popen([$SALOME_AGENT_GUI], start_new_session=True, env=clean_env)`
   where `clean_env` strips `PYTHONPATH`, `QT_PLUGIN_PATH`,
   `QT_QPA_PLATFORM_PLUGIN_PATH`, `QT_QPA_FONTDIR`, `LD_LIBRARY_PATH` (same
   list as sarag) so the venv's PyQt5 does not load SALOME's Qt.
3. Singleton guard: PID file in the temp dir keyed by user; a second click
   while the window is alive shows an info box and does nothing else.

All names used at click time are imported inside the function: the plugins
manager `exec`s the file with a discarded locals dict (documented sarag gotcha).

**Server code inside SALOME's Python.** `agent_server/` and `executor/` depend on
the standard library, SALOME modules and `typing_extensions` (already used
in-GUI today by the existing plugin, so present in SALOME's Python). The SAT
build copies both packages next to `salome_plugins.py`; the plugins manager puts
that directory on `sys.path`. The chat window's venv never needs them.

### Repo B — `salome-agent` (this repo) changes

1. **`salome_agent/runner.py`** — new. Library entry point extracted from `cli.main`:

   ```python
   def run_request(request: str, config: dict, *, client, on_event, on_gate,
                   cancel: threading.Event | None = None,
                   verbose: int = 0) -> RunResult
   ```

   - Builds the LLM (`build_llm`) and graph (`build_graph(batch=config["batch"])`),
     initial state as `cli.main` does today.
   - Iterates `graph.stream(input, config=cfg, stream_mode="updates")`; maps each
     node update to `on_event(Event)`.
   - On an interrupt: `decision = on_gate(payload)` (blocking), then resumes with
     `graph.stream(Command(resume=decision), ...)`. Decision dicts keep the shape
     `interactive.py` produces today (`approve` / `edit`+`plan` / `abort` /
     `hint`+`hint` / `skip`).
   - `cancel` is checked between stream steps; when set the run ends `aborted`.
     A pending gate is answered with `abort` by the caller.
   - Returns `RunResult{run_id, run_dir, final_state, exit_code}`.

2. **Event model** (`Event = {kind, run_id, data}`):

   | kind | emitted after node | data |
   |---|---|---|
   | `plan` | `plan` | stages list |
   | `stage_start` | `run_stage` (first attempt of an index) | index, module, intent |
   | `stage_attempt` | `verify_static` / `verify_exec` | index, phase, ok, error |
   | `stage_done` | `advance` | index, published names |
   | `stage_skipped` | `skip` | index |
   | `report` | `report` | `result.py` / `report.md` paths, token usage, timing |
   | `fatal` | any (state `fatal` set) or runner exception | message |
   | `aborted` | gate abort or cancel | — |

   Mapping lives in one function (`_events_from_update`) so node changes touch one place.

3. **CLI refactor** — `cli.main` becomes argparse → `load_config` →
   `server_conn.connect` → `run_request(on_event=log-only, on_gate=interactive prompts)`
   → the same summary print. `interactive.prompt_plan` / `prompt_retry` become
   the CLI's `on_gate`; `resume_interactive` is removed. No behaviour change.

4. **User config path** — `load_config` accepts an explicit path (already does);
   the GUI passes `~/.config/salome/salome-agent.json`. No change to the CLI default.

### SAT integration (SAT_SALOME repo, on a new branch created before any edit)

- `products/salome_agent_gui.pyconf` — `get_source: git`,
  `git_commands: ['git submodule update --init --recursive']`,
  `build_source: script`, `depend: ['Python', 'setuptools']`,
  `opt_depend: ['PyQt']`, `install_dir: 'base'`, env script below.
- `products/compil_scripts/salome_agent_gui.sh`:
  1. `${PYTHONBIN} -m venv ${PRODUCT_INSTALL}`; `pip install ./salome-agent .`
  2. copy `salome-agent-gui.sh` to `${PRODUCT_INSTALL}/bin/`
  3. copy `salome_plugins.py`, `salome-agent/agent_server/`, `salome-agent/executor/`
     to `${PRODUCT_INSTALL}/share/salome/plugins/salome_agent_gui/`
     (plugins manager scans `<X>_ROOT_DIR/share/salome/plugins` and adds the dir to `sys.path`)
  4. copy `salome-agent/config.example.json` next to the package as the seed config
- `products/env_scripts/salome_agent_gui.py` — sets `SALOME_AGENT_GUI_ROOT_DIR`
  and `SALOME_AGENT_GUI` (launcher path).
- Add the product to the application pyconfs for 9.16 and master.

## Chat window

**Left panel — Connection:** config file path; LLM base URL, model, API key
(masked) bound to `llm.base_url` / `llm.model` / `llm.api_key`; "Edit
configuration…" opens the full JSON in a text dialog, validated through
`load_config` before saving; server status line (mode, port, connected);
Connect button.

**Left panel — Run options:** batch (skip gates), `policy.on_stage_failure`,
`policy.tool_selection`, verbosity (`-vv` equivalent writes `llm_trace.jsonl`).

**Right panel — Chat:** `QTextBrowser` with markdown rendering
(`setMarkdown`, no QtWebEngine dependency). Per message:

- user bubble;
- one assistant message per run, updated in place from events:
  - plan card — Approve / Edit (plan JSON dialog) / Abort;
  - stage rows — status (running / ok / failed / skipped), module, intent,
    attempt count;
  - failed-stage card — error message + offending line, Hint (text input) / Skip / Abort;
  - summary — created objects, clickable `result.py` / `report.md` (opened with
    the desktop handler), token usage.

**Bottom:** input line with history (Up/Down), Send, Stop, Help, Quit.
Send is disabled while a run is active or the client is disconnected.

**Threading:** `RunWorker(QThread)` owns a run. `on_event` emits a Qt signal
to the UI thread. `on_gate` emits a gate signal and blocks on a
`threading.Event` until the UI thread posts the decision. Stop sets `cancel`
and, if a gate is pending, answers it with `abort`. One `ExecutorClient` per
window session, reconnected via `server_conn.connect` on `SessionDied` /
connection loss.

## Config

- User config: `~/.config/salome/salome-agent.json`. First launch seeds it from
  the shipped `config.example.json`.
- Key fields write back into `llm.*`; everything else via the raw editor.
- API key stored in that file in clear, as today; never logged or echoed.

## Error handling

| Situation | Behaviour |
|---|---|
| Handshake absent at Connect | poll ~10 s (plugin just started the server), then status: "Start it from Tools > Plugins > SALOME Agent". |
| Server busy / poisoned | `ServerUnavailable` text in the status line; Send stays disabled. |
| Server restarted (token rotated) | `server_conn.connect` re-reads the handshake (existing retry). |
| LLM outage / 429 / `StreamBudgetExceeded` | existing labels surface as a `fatal` event → error card; window stays usable. |
| Runner exception | caught in `RunWorker`, emitted as `fatal`. |
| Window already open | PID guard → info box. |

## Testing

- **salome-agent (unit):** `runner` with a fake LLM / fake graph — event order,
  gate round-trip (approve, edit, hint, skip, abort), cancel between steps,
  fatal path. Existing CLI tests pin the refactor.
- **salome-agent-gui (unit):** `pytest-qt` (`qtbot`), `QT_QPA_PLATFORM=offscreen`:
  `RunWorker` signal flow with a stub runner, gate cards post the right
  decision, Stop answers a pending gate, config store seed/load/save.
- **Manual live smoke:** SAT build → SALOME GUI → Tools > Plugins > SALOME Agent
  → "two boxes side by side" → approve plan → objects appear in the viewer,
  `result.py` link opens. Run on 9.16 and master.

## Follow-ups (filed or to file)

- Task 194 — conversational refinement in the same window.
- Task 242 — selection as agent context (`get_selection` server command,
  `libGEOM_Swig` local selection).
- Per-tool-call events (hook via `configurable`) and streamed reasoning in the timeline.
