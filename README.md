# salome-agent

Turn a natural-language request into a **verified, reproducible SALOME Python script**.

You describe what you want ("make a 10 mm cube, then tetrahedral-mesh it"). The agent
plans the stages, drives per-module specialist agents (geom / smesh) against a
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
- **smesh** stages are raw scripts, statically checked (top-level API existence) then executed under a timeout with per-module sanity checks.
- Stage retry re-runs the full stage script (no MED checkpoints — GEOM can't export MED). GUI mode does not roll back published objects; see [Known limitations](#known-limitations).

---

## Requirements

- **Python 3.10+**
- A working **SALOME install** (developed against a recent `master` native build).
  The live tests find it via `$SALOME_ROOT` or `salome.root` in `config.json`.
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
    "json_retries": 2,
    "vision_fix_retries": 2,
    "loop_retries": 3,
    "max_retries": 2,
    "reasoning_budget": 20000,
    "carry_reasoning": true,
    "extra_headers": {},
    "extra_body": {},
    "module_extra_body": {},
    "modules": {}
  },
  "salome": {
    "root": "/path/to/your/SALOME"
  },
  "policy": {
    "static_retries": 3,
    "exec_retries": 2,
    "exec_timeout_s": 120,
    "tool_iteration_cap": 12,
    "plan_detail": "balanced",
    "tool_selection": "hints",
    "verify_cadence": "final",
    "on_stage_failure": "abort"
  },
  "work_dir": "~/.local/share/salome-agent/runs"
}
```

Required keys: `llm.base_url` and `llm.model`. Missing either is a hard error. An unknown key in the `llm` block (a typo such as `extra-body`) is logged as a warning and ignored.

#### `llm` — the model endpoint

Any OpenAI-compatible `/chat/completions` endpoint. Requests are streamed (SSE).

| Key | Default | Meaning |
|-----|---------|---------|
| `base_url` | — (required) | Endpoint base URL, up to and including `/v1`. |
| `model` | — (required) | Model name sent in every request. |
| `api_key` | `""` | Sent as `Authorization: Bearer <key>`. Leave empty for an endpoint without auth. |
| `ssl_cert_file` | `""` | Path to a CA bundle for an HTTPS endpoint with a private CA. `~` is expanded; a missing file fails at startup. Empty verifies against the system trust store. |
| `temperature` | `0` | Sampling temperature. |
| `timeout` | `120` | Seconds. Total wall-clock budget for one streamed LLM call; a generation still running at the limit (a runaway reasoning loop) is cut and not retried. The gap between two streamed chunks is capped separately at 60 s. `--timeout` overrides it. |
| `json_retries` | `2` | When the planner's answer does not parse as JSON, the answer and the exact parser error are fed back and the planner is re-asked this many times before the run fails. |
| `vision_fix_retries` | `2` | When the image-analysis agent's spec fails the mechanical checks (an outline that does not close, an arc end off its radius, a hole outside the outline, a dangling reference), the errors are fed back and it is re-asked this many times; errors still left go to the planner and geom as `conflicts`. |
| `max_retries` | `2` | Retries for transient failures — connection drop, truncated stream, HTTP 429 (honouring `Retry-After`), fast 502/503/504 — with exponential backoff. Raise it for a flaky free-tier endpoint. |
| `reasoning_budget` | `20000` | Specialist (tool-calling) calls only: approximate reasoning tokens (characters / 4) a model may spend before its first tool call. Past it the call is cut and re-asked once with the tail of its own reasoning and a "continue from it and act" nudge; a second cut in a row ends the stage. If the cut reasoning ends in a loop, it is handled as a loop (see `loop_retries`). `0` turns it off. |
| `loop_retries` | `3` | Every call (planner, specialists, vision): when the streamed reasoning or answer starts repeating itself (the same sentences over and over, as small models do), the call is cut at once. A planner or specialist call is then re-asked with its text from before the loop and one repeated sentence, never the loop itself, up to this many cuts in a row; past that the stage ends (the plan fails). A vision call that loops ends the run. |
| `carry_reasoning` | `true` | Specialist (tool-calling) calls only: each turn's reasoning (`reasoning_content`) stays in the stage's history, so after a tool result (a `calc`, say) the model resumes its own thinking instead of starting over. Costs input tokens for the rest of the stage. Set `false` for a provider that rejects `reasoning_content` in requests. |
| `extra_headers` | `{}` | Extra HTTP headers sent verbatim on every request. In a string value `{run_id}` is replaced with the run's id — for gateways that want a stable per-conversation token, e.g. `{"x-opencode-session": "{run_id}"}`. |
| `extra_body` | `{}` | Extra top-level fields merged into every request's JSON body, next to `model`/`messages`/`temperature`. Use this, not `extra_headers`, for provider body params such as `{"reasoning_effort": "high"}`; whether the provider honours a field is up to the provider. |
| `module_extra_body` | `{}` | `{"planner" \| "geom" \| "smesh": {...}}` — a body sent *instead of* `extra_body` (full replacement, not a merge) on that module's calls. Lets an expensive knob sit on the planner only, e.g. `{"planner": {"reasoning_effort": "high"}}`, while the many specialist calls run without it. |
| `modules` | `{}` | `{"vision" \| "planner" \| "geom" \| "smesh": {...}}` — sends that module's calls to their own endpoint and/or model. Each entry may set `base_url`, `api_key`, `model`, `ssl_cert_file`, `temperature`, `timeout`, `extra_headers`, `extra_body`; a key left out keeps the shared `llm` value — except with a `base_url` of its own, which never inherits the shared `api_key`, `ssl_cert_file`, `extra_headers` or `extra_body` (another host gets no credentials meant for the main one). Its `extra_body`, when set, wins over `module_extra_body`. E.g. a stronger drawing reader for images only: `{"vision": {"base_url": "https://generativelanguage.googleapis.com/v1beta/openai", "api_key": "...", "model": "gemini-3-pro-preview"}}`. |

#### `policy` — how a run behaves

The first five keys have no built-in default — keep them in your config.

| Key | Default | Meaning |
|-----|---------|---------|
| `static_retries` | — | Per stage: how many times a script that fails the static check is sent back to its specialist before the stage counts as failed. |
| `exec_retries` | — | Per stage: how many times a script that fails when run under SALOME is sent back to its specialist. |
| `exec_timeout_s` | — | Seconds allowed for one stage's script to run under SALOME in the exec check. |
| `tool_iteration_cap` | — | Maximum LLM turns (each turn may carry several tool calls) a tool-calling specialist gets for one stage. |
| `on_stage_failure` | — | `abort` \| `continue`. What `--batch` does when a stage still fails after its retries: stop the run, or skip the stage and go on. Interactive runs ask instead. `--on-failure` overrides it. |
| `plan_detail` | `balanced` | `coarse` \| `balanced` \| `fine`. How finely the planner splits a request into stages — `coarse` favours fewer, bigger stages; `fine` favours splitting a part or assembly into more stages. |
| `tool_selection` | `hints` | `hints` \| `all`. `hints`: the geom specialist is offered a small per-stage tool subset (planner hint + a core set) and widens it on demand via `load_tools`. `all`: the full geom registry on every call. `--tools` overrides it. |
| `verify_cadence` | `final` | How hard the geom specialist is pushed to `measure`-check where its shapes landed — see below. |

`policy.verify_cadence`:
- `none` — no self-check.
- `final` (default) — read the `shape_info` that rides on every result for free; call `measure` explicitly only when something looks off and once on the result shape(s) before `finish_stage`. If a move or boolean is still unmeasured at `finish_stage`, the loop asks for a measure once before accepting.
- `strict` — measure after every move or boolean. If the model does three in a row without one, the loop forces a measure before it can continue.

#### Other keys

| Key | Default | Meaning |
|-----|---------|---------|
| `work_dir` | `.` | Where each run's folder (`<run_id>/` with `result.py`, `report.md`, logs) is written. `~` is expanded. `--work-dir` overrides it. |
| `salome.root` | — | Path to your SALOME install. The agent itself does not use it (it talks to an already-running agent server); the live tests and `scripts/regen-toplevel-api.py` read it when `$SALOME_ROOT` is not set. |

The salome-agent-gui plugin stores its own settings (`gui.theme`, `gui.batch`, …) in a `gui` section of the same file; the agent ignores that section.

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
  `~/.salome_agent/server.log`, stop with Ctrl-C. One server at a time. It
  stays up across CLI runs — the CLI drops its connection but never shuts the
  server down.

Then:

```bash
cd /path/to/salome-agent   # the repo root
source .venv/bin/activate

salome-agent "make a 10mm cube" --batch
```

To point the agent at a picture (or several) of the part you want, add `--images`
(comma-separated paths, no spaces around the commas needed):

```bash
salome-agent "make this bracket" --images ./bracket-sketch.png --batch

# several views/plans of the same part
salome-agent "make this bracket" --images ./front.png,./side.png,./top.bmp --batch
```

Each image is base64-encoded (PNG/JPEG/WEBP/GIF/BMP, recognised by content; at most 10
images and 20 MB in all; needs a vision-capable LLM behind `llm.base_url`) and first handed to a dedicated **vision agent**. It extracts what is
visible — the parts and their shapes, dimensions written on the drawing, proportions,
symmetry, and what is ambiguous — as a structured description, which is logged and passed
to the **planner** and to the **geom specialist** in place of the raw pictures — neither
sees the images themselves. The planner uses it to shape the stage breakdown, the
specialist for proportions and placement. Multiple
images are treated as different views/plans of the *same* part, reconciled into one 3D
description — not as separate parts. Explicit numeric dimensions in the text request
win, then dimensions read from the image; values the vision agent had to estimate are
marked `~`. smesh stages don't see the images — it only feeds CAD construction.

The vision call is tagged `module="vision"`, so `llm.modules.vision` can send it to a
different model or endpoint (one that reads drawings better) and `llm.module_extra_body` can
give it its own request body (e.g. a different `reasoning_effort`) like the other modules.

The agent will:

1. Connect to the running server via `~/.salome_agent/server.json`. If none is
   running it exits with a message telling you to start one.
2. Plan the stages, run each specialist, verify, and assemble — progress is logged to
   stderr as it goes (`planning...`, `stage 1/2 [geom] ...`, `static check: ok`,
   `exec check: ok`, ...). Add `-v` for per-LLM-call and per-retry detail, `-vv`
   to also dump every LLM prompt, response and reasoning to the terminal and
   `<run_id>/llm_trace.jsonl` (a call cut off, cancelled or dropped still gets a line,
   marked `incomplete`, with what it streamed). `-vv` also writes
   `<run_id>/llm_live.txt` as each call streams: `tail -f` it to watch the model's
   reasoning while a call is in flight.
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
| `--images PATH,PATH,...` | Comma-separated reference image(s) (PNG/JPEG/WEBP/GIF/BMP) the generated CAD should match — e.g. several views/plans of the same part; described by a vision agent whose text description feeds the planner and the geom specialist (they never see the images). |
| `--batch` | Non-interactive. Auto-approve the plan; on stage failure follow `policy.on_stage_failure` (`abort` or `continue`). Without it, the run is interactive — see below. |
| `--no-session` | Planner + specialists + static checks only. No connection to a SALOME agent server, no exec verification. smesh only (geom needs a session). |
| `--work-dir DIR` | Override `config.work_dir`. |
| `--config PATH` | Use a different config file (default: `./config.json`). |
| `--on-failure {abort,continue}` | Override `policy.on_stage_failure`. |
| `--tools {hints,all}` | Geom tool retrieval. `hints` (default): the geom specialist is offered a small per-stage tool subset (planner hint + a core set) and widens it on demand via `load_tools`. `all`: offer the full geom registry on every call (pre-retrieval behaviour). Overrides `policy.tool_selection`. |
| `--timeout SECONDS` | Override the per-round-trip LLM budget (`llm.timeout` in config). Total wall-clock for one streamed call and, capped at 60s, the socket read-gap; a generation still running at the limit (a runaway reasoning loop) is cut and not retried. |
| `-v`, `--verbose` | Per-LLM-call and per-retry detail on top of the milestone log. Repeat (`-vv`) to dump every LLM prompt, response and reasoning to the terminal and `<run_id>/llm_trace.jsonl`, and stream each call's reasoning to `<run_id>/llm_live.txt`. |
| `--chat` | After the first result, keep talking to the agent in a REPL — see [Chat mode](#chat-mode). Cannot be combined with `--batch` or `--no-session`. |

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
— or, if the GUI is frozen by a runaway stage, kill it. The headless server (which
otherwise stays up across runs) **exits** on a poisoning timeout — restart it.

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

### Chat mode

```bash
salome-agent "a small house" --chat
```

`--chat` keeps the session open after the first result, so you can keep talking:

```
chat> make the roof taller
chat> add a chimney
chat> /undo
```

Commands:

| Command | Meaning |
|---------|---------|
| `/undo` | revert the last turn |
| `/redo` | re-apply the last undone turn (cleared by any new turn) |
| `/show` | print the build graph |
| `/result` | print `result.py` |
| `/rebuild` | rebuild the whole study from the model (after a failed rollback) |
| `/quit` | leave (Ctrl-D too) |

One executor connection and one LLM session serve the whole conversation. An edit
replays only the affected object and everything downstream of it, reusing the same
object names, rather than starting the study over. `/undo` reverts the last turn;
`/rebuild` forgets every name and replays the whole graph from scratch, for when a
rollback itself fails. Ctrl-C cancels the turn in progress and returns to `chat>`.
`--chat` is interactive and needs a live session, so it cannot be combined with
`--batch` or `--no-session`.

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
— a snapshot of the valid `geompy` / `smesh` top-level members. It
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
  llm.py            minimal OpenAI-compatible tool-calling client (+ with_images() for multimodal messages)
  state.py          RunState / stage TypedDicts
  nodes/            planner, gates (plan/retry), advance, assembler, report
  specialists/      geom (tool-calling), smesh (raw script)
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
  namespace.py      builds the exec namespace (geompy/smesh/...)
  __main__.py       headless entry (`python -m agent_server`)
executor/
  registry.py       ToolRegistry — tool decorator, schemas, arg-type coercion
  geom/             geom tool implementations + per-tool JSON schemas
plugin/             salome_agent_plugin.py — standalone manual-registration variant
                    (install-plugin.sh inlines the same call; not imported by it)
scripts/            install-plugin.sh, start-headless-server.sh, regen-toplevel-api.py
docs/superpowers/   design specs + implementation plans
```

---

## Known limitations

- **The GUI plugin server is newer and less tested than the headless path.** GUI
  mode does not roll back — a failed stage attempt leaves `sa_<run_id>_`-prefixed
  debris in the study. A stage timeout poisons the server (run aborts, exit 1);
  recovery may need killing the GUI. Save your work first. Headless mode is the
  unattended path.
- `--no-session` mode only does a coarse "does this name exist" static check
  against a snapshot API map and no exec verification — see *Static-check API map*
  under Development. A real run always verifies against the live session.
- geom tool registry (~45 tools: primitives, sketch, swept, booleans, transforms,
  fillet/chamfer, display, query, subshape selection). `align` does relative
  placement (move a part so a chosen face/centre meets another's) so the model
  states the spatial relationship instead of computing the translation itself.
  Subshape ids come from
  `get_subshapes` (query — returns ids/centres, emits no `CallRecord`) or
  `explode` (publishes each subshape); `get_subshape` builds one shape from an id
  list. Add a tool with `@tool(name, category=..., description=..., schema=...,
  arg_types=...)` in `executor/geom/<category>.py` (see an existing tool);
  registration order = import order in `executor/geom/__init__.py` and is
  pinned by `tests/fixtures/golden/list_tools.json`.
- `reset_and_replay` (rollback) is wired only for the headless backend and has no
  graph-level caller yet — retries re-run the full stage script.
- RAG / documentation retrieval was cut from v1.
