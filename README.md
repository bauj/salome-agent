# salome-agent

Turn a natural-language request into a **verified, reproducible SALOME Python script**.

You describe what you want ("make a 10 mm cube, then tetrahedral-mesh it"). The agent
plans the stages, drives per-module specialist agents (geom / smesh / paravis) against a
live SALOME session, verifies each stage actually runs, and assembles a standalone
`result.py` that reproduces the whole thing with no agent or executor involved.

Built for SALOME module development / automation — not an end-user chatbot or a solver
pipeline.

---

## How it works

Two processes:

| Process   | Runs where | Does |
|-----------|-----------|------|
| **agent**    | your Python venv | planning, LLM calls, static verification, script assembly (LangGraph `StateGraph`) |
| **server**   | inside a SALOME process — a GUI plugin thread, or headless `python -m agent_server` | executes stage scripts, geom tool calls, introspection, sanity checks, timeout kills |

The agent connects to the server over loopback TCP (token-authenticated,
line-delimited JSON); it discovers a running server via `~/.salome_agent/server.json`.

Flow: `request → plan → [per stage: specialist → verify static → verify exec → advance] → assemble result.py → report`

- **geom** stages run as tool calls (`make_box`, `get_subshapes`, `fillet_edges`, ...) against the server; the script is generated from the call log.
- **smesh / paravis** stages are raw scripts, statically checked (top-level API existence) then executed under a timeout with per-module sanity checks.
- Stage retry re-runs the full stage script (no MED checkpoints — GEOM can't export MED). GUI mode does not roll back published objects; see [Known limitations](#known-limitations).

---

## Requirements

- **Python 3.10+**
- A working **SALOME install** (developed against a recent `master` native build;
  point `salome.root` at its top directory).
- An **OpenAI-compatible, tool-calling LLM endpoint** (base URL + API key + model name).

---

## Install

```bash
cd /path/to/salome-agent   # the repo root
python3 -m venv .venv
source .venv/bin/activate
pip install -e .          # or -e ".[dev]" to also get pytest
```

This installs `langgraph` and `requests` and puts a `salome-agent` entry point in
`.venv/bin`.

> The command is `salome-agent` (hyphen), not `salome_agent`.
> If `pip install -e .` was skipped, there is no entry point — run it as
> `python -m salome_agent.cli ...` from the repo root instead.

---

## Configure

Copy the template and fill in your LLM credentials:

```bash
cp config.example.json config.json
```

```json
{
  "llm": {
    "base_url": "http://your-endpoint:port/v1",
    "api_key": "sk-...",
    "model": "auto",
    "ssl_cert_file": "",
    "temperature": 0,
    "timeout": 120,
    "json_retries": 2
  },
  "salome": {
    "root": "/path/to/your/SALOME"
  },
  "policy": {
    "static_retries": 3,
    "exec_retries": 2,
    "exec_timeout_s": 120,
    "tool_iteration_cap": 12,
    "tool_selection": "hints",
    "verify_cadence": "final",
    "on_stage_failure": "abort",
    "assembler_smoke_run": true
  },
  "work_dir": "~/.local/share/salome-agent/runs"
}
```

Required keys: `llm.base_url`, `llm.model`, `salome.root`. Missing any of them is a hard error.

`llm.ssl_cert_file` is optional: a path to a CA bundle for verifying an HTTPS endpoint with a private CA. Empty/omitted verifies against the system trust store.

`llm.extra_headers` is optional: an object of extra HTTP headers sent verbatim on every LLM request. In a string value, `{run_id}` is replaced with the current run's id — useful for gateways that want a stable per-conversation token, e.g. `{"x-opencode-session": "{run_id}"}`.

`llm.json_retries` (default 2): when the planner LLM returns something that will not parse as JSON, its answer plus the exact parser error are fed back and it is re-asked this many times before the run fails.

`policy.verify_cadence` controls how hard the geom specialist is pushed to `measure`-check where its shapes landed:
- `none` — no self-check.
- `final` (default) — read the `shape_info` that rides on every result for free; call `measure` explicitly only when something looks off and once on the result shape(s) before `finish_stage`. If a move or boolean is still unmeasured at `finish_stage`, the loop asks for a measure once before accepting.
- `strict` — measure after every move or boolean. If the model does three in a row without one, the loop forces a measure before it can continue.

`config.json` is gitignored — keep your credentials there, not in `config.example.json`.

---

## Usage

Run **from the repo root** (`config.json` is resolved relative to the current directory).

The agent talks to a **SALOME agent server** — the executor loop running inside a
SALOME process. Start it first (once per SALOME session):

- **GUI:** launch `"$SALOME_ROOT"/salome`, then **Tools ▸ Plugins ▸ Launch SALOME
  agent server** — see [GUI plugin setup](#gui-plugin-setup) for the one-time install.
- **Headless (CI / batch):** `scripts/start-headless-server.sh` (needs `SALOME_ROOT`)
  — isolated study, no GUI. Runs in the foreground; ready when
  `~/.salome_agent/server.json` appears (~10-20s), logs to
  `~/.salome_agent/server.log`, stop with Ctrl-C. One server at a time.

Then:

```bash
cd /path/to/salome-agent   # the repo root
source .venv/bin/activate

salome-agent "make a 10mm cube" --batch
```

The agent will:

1. Connect to the running server via `~/.salome_agent/server.json`. If none is
   running it exits with a message telling you to start one.
2. Plan the stages, run each specialist, verify, and assemble — progress is logged to
   stderr as it goes (`planning...`, `stage 1/2 [geom] ...`, `static check: ok`,
   `exec check: ok`, ...). Add `-v` for per-LLM-call and per-retry detail, `-vv`
   to also dump every LLM prompt, response and reasoning to the terminal and
   `<run_id>/llm_trace.jsonl`.
3. Write output to `<work_dir>/<run_id>/`:
   - `result.py` — the standalone reproducible script
   - `report.md` — per-stage status and retry counts (also printed at the end)

Raw SALOME/paraview noise (`vtkSocket ... Connection refused` etc.) goes to
`~/.salome_agent/server.log`, not your terminal.

Run the result on its own, with no agent involved:

```bash
"$SALOME_ROOT"/salome shell -- \
  python3 ~/.local/share/salome-agent/runs/<run_id>/result.py
```

### Options

| Flag | Meaning |
|------|---------|
| `--batch` | Non-interactive. Auto-approve the plan; on stage failure follow `policy.on_stage_failure` (`abort` or `continue`). Without it, the run is interactive — see below. |
| `--no-session` | Planner + specialists + static checks only. No connection to a SALOME agent server, no exec verification. smesh/paravis only (geom needs a session). |
| `--work-dir DIR` | Override `config.work_dir`. |
| `--config PATH` | Use a different config file (default: `./config.json`). |
| `--on-failure {abort,continue}` | Override `policy.on_stage_failure`. |
| `--tools {hints,all}` | Geom tool retrieval. `hints` (default): the geom specialist is offered a small per-stage tool subset (planner hint + a core set) and widens it on demand via `load_tools`. `all`: offer the full geom registry on every call (pre-retrieval behaviour). Overrides `policy.tool_selection`. |
| `--timeout SECONDS` | Override per-stage exec timeout. |
| `-v`, `--verbose` | Per-LLM-call and per-retry detail on top of the milestone log. Repeat (`-vv`) to dump every LLM prompt, response and reasoning to the terminal and `<run_id>/llm_trace.jsonl`. |

### GUI plugin setup

One-time: `scripts/install-plugin.sh` — appends a self-bootstrapping block to
`~/.config/salome/Plugins/salome_plugins.py` that puts the repo on `sys.path` and
registers the menu entry with a direct `AddFunction` call (an `import`-only
indirection does not reach the plugin manager). No env vars needed.

Each session: launch `"$SALOME_ROOT"/salome` (single-process is fine), then
**Tools ▸ Plugins ▸ Launch SALOME agent server**. A status message shows
`127.0.0.1:<port>`. Run the agent normally — it finds the server via
`~/.salome_agent/server.json`.

Objects a run publishes appear in the Object Browser and 3D viewer, prefixed
`sa_<run_id>_`. A failed stage attempt can leave debris under that prefix —
`report.md` names it; bulk-delete by prefix.

A second menu click **restarts** the server (new token, new port). A stage timeout
**poisons** the server: the run aborts (exit 1) and you restart it from the same menu
— or, if the GUI is frozen by a runaway stage, kill it.

> Auto-launching SALOME from the agent was tried and reverted (task 182). Repeated
> external attach/detach against a long-lived GUI has crashed the session before
> (task 181) — save your work; treat live-GUI use as demo/interactive, not unattended
> batch. Headless (`scripts/start-headless-server.sh`) is the unattended path.

**Headless:** do **not** run `python -m agent_server` from an activated venv — the
wrapper script strips `.venv` from `PATH` first (the venv python lacks `psutil` and
SALOME's `sys.path`).

### Interactive mode

Without `--batch`, the run pauses for you at two points:

- **Plan approval** — after planning, the stage list is printed and you `[Y]es` /
  `[e]dit` (opens the plan as JSON in `$EDITOR`; add/drop/reorder stages, then it
  re-prompts) / `[n]o abort`.
- **Stage failure** — if a stage exhausts its retries, you get `[h]int` (give the specialist
  a text hint and retry the stage), `[s]kip stage`, or `[a]bort run`.

Everything up to the pause is preserved (LangGraph checkpointer); your decision resumes the
same run. If stdin is not a terminal (piped, no input), an unanswered prompt aborts the run —
use `--batch` for unattended runs.

Aborting at either gate stops immediately with exit 1 and writes nothing — no
`result.py`, no `report.md`, no run directory.

---

## Development

```bash
# fast unit tests (no SALOME needed)
pytest tests/unit

# live tests — require a working SALOME install, ~15-20s startup each
pytest -m salome
```

Tests marked `salome` are skipped unless a real SALOME install is present.
They locate it via `$SALOME_ROOT`, else `salome.root` in `config.json`; with
neither set the `salome` tests skip.

### Static-check API map (`--no-session`)

A normal run verifies each stage script against the **live** SALOME session
(`introspect()` on the running modules — authoritative). `--no-session` has no
session, so its static check falls back to `salome_agent/verify/toplevel_api.json`
— a snapshot of the valid `geompy` / `smesh` / `pvsimple` top-level members. It
only catches calls to names that do not exist on the module; it is a coarse
check, and `--no-session` skips exec verification entirely.

That fixture has no source but a live session, so regenerate it whenever the
SALOME build changes:

```bash
python3 scripts/regen-toplevel-api.py
```

It reuses a running agent server (`~/.salome_agent/server.json`, GUI plugin or
headless) if one is up, otherwise spawns a throwaway headless server (needs
`$SALOME_ROOT` or `salome.root` in `config.json`) and tears it down after. A
module whose live introspect comes back empty keeps its existing member list.
`tests/salome/test_toplevel_api_fixture.py` fails if the committed fixture has
drifted from a live introspect.

### Layout

```
salome_agent/
  cli.py            entry point
  graph.py          the LangGraph StateGraph (all nodes + routing)
  config.py         config loading + validation
  llm.py            minimal OpenAI-compatible tool-calling client
  state.py          RunState / stage TypedDicts
  nodes/            planner, gates (plan/retry), advance, assembler, report
  specialists/      geom (tool-calling), smesh, paravis (raw script), base
  verify/           static (top-level API check), exec_check, introspect
  server_conn.py    finds + connects to the SALOME agent server
  executor_client.py  TCP client for the server (hello handshake, framing)
  codegen.py        geom call-log -> Python source
agent_server/       the server that runs inside SALOME
  core.py           command dispatch + poison gate
  backend.py        Backend protocol + HeadlessBackend
  gui_backend.py    GuiBackend + Qt main-thread marshal
  wire.py           loopback TCP + token listener
  handshake.py      ~/.salome_agent/server.json
  gui_plugin.py     the Tools > Plugins entry point (launch/restart)
  namespace.py      builds the exec namespace (geompy/smesh/... ; defers pvsimple)
  __main__.py       headless entry (`python -m agent_server`)
executor/
  geom_tools.py     geom tool implementations + per-tool JSON schemas
plugin/             salome_agent_plugin.py — standalone manual-registration variant
                    (install-plugin.sh inlines the same call; not imported by it)
scripts/            install-plugin.sh, start-headless-server.sh, regen-toplevel-api.py
docs/superpowers/   design specs + implementation plans
```

---

## Known limitations

- **The GUI plugin server is new and lightly tested.** External driving of a live
  GUI has crashed the session before (task 181). GUI mode does not roll back — a
  failed stage attempt leaves `sa_<run_id>_`-prefixed debris in the study. A stage
  timeout poisons the server (run aborts, exit 1); recovery may need killing the GUI.
  Demos only; save your work first. Headless mode is the unattended path.
- `--no-session` mode only does a coarse "does this name exist" static check
  against a snapshot API map and no exec verification — see *Static-check API map*
  under Development. A real run always verifies against the live session.
- geom tool registry (~45 tools: primitives, sketch, swept, booleans, transforms,
  fillet/chamfer, display, query, subshape selection). Subshape ids come from
  `get_subshapes` (query — returns ids/centres, emits no `CallRecord`) or
  `explode` (publishes each subshape); `get_subshape` builds one shape from an id
  list. Add tools using the existing handler + `ARG_TYPES` + `TOOL_SCHEMAS` +
  `CallRecord` pattern in `executor/geom_tools.py`.
- `reset_and_replay` (rollback) is wired only for the headless backend and has no
  graph-level caller yet — retries re-run the full stage script.
- RAG / documentation retrieval was cut from v1.
