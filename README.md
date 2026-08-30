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
| **executor** | inside SALOME, launched via `salome shell -- executor.py` | executes stage scripts, geom tool calls, introspection, sanity checks, timeout kills |

They talk over a Unix socket (line-delimited JSON).

Flow: `request → plan → [per stage: specialist → verify static → verify exec → advance] → assemble result.py → report`

- **geom** stages run as tool calls (`make_box`, `fillet_edges`, ...) against the executor; the script is generated from the call log.
- **smesh / paravis** stages are raw scripts, statically checked (top-level API existence) then executed under a timeout with per-module sanity checks.
- Rollback on retry = replay the accepted stages (no MED checkpoints — GEOM can't export MED).

---

## Requirements

- **Python 3.10+**
- A working **SALOME install**. Developed against `SALOME-master-native-FD44` at
  `/storage/salome_platform/devel/SALOME-master-native-FD44`.
  The `SALOME-9.16.0-native-FD44` dev install is broken — do not use it.
- An **OpenAI-compatible, tool-calling LLM endpoint** (base URL + API key + model name).

---

## Install

```bash
cd /storage/Projects/salome-agent
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
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
    "temperature": 0
  },
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

Required keys: `llm.base_url`, `llm.model`, `salome.root`. Missing any of them is a hard error.

`config.json` is gitignored — keep your credentials there, not in `config.example.json`.

---

## Usage

Run **from the repo root** (the executor is launched via the relative path
`executor/executor.py`, and `config.json` is resolved relative to the current directory).

```bash
cd /storage/Projects/salome-agent
source .venv/bin/activate

salome-agent "make a 10mm cube" --batch
```

The agent will:

1. Start an executor inside SALOME via `salome shell`. This is **always an isolated,
   standalone study** — even if you have a SALOME GUI running, the executor does not
   touch it and objects do not appear in that GUI. (`--port` only picks which omniORB
   port to hand `salome shell`; it does not change the isolation.) Driving a live GUI
   session is a separate, not-yet-built mode.
2. Plan the stages, run each specialist, verify, and assemble.
3. Write output to `<work_dir>/<run_id>/`:
   - `result.py` — the standalone reproducible script
   - `report.md` — per-stage status and retry counts

Run the result on its own, with no agent involved:

```bash
/storage/salome_platform/devel/SALOME-master-native-FD44/salome shell -- \
  python3 ~/.local/share/salome-agent/runs/<run_id>/result.py
```

### Options

| Flag | Meaning |
|------|---------|
| `--batch` | Non-interactive. Auto-approve the plan; on stage failure follow `policy.on_stage_failure` (`abort` or `continue`). **Currently the only fully working mode — see below.** |
| `--no-session` | Planner + specialists + static checks only. No live SALOME session, no exec verification. smesh/paravis only (geom needs a live session). |
| `--port N` | Hand `salome shell` the omniORB port `N` instead of auto-discovering it. Still an isolated study — not live-session injection. |
| `--work-dir DIR` | Override `config.work_dir`. |
| `--config PATH` | Use a different config file (default: `./config.json`). |
| `--on-failure {abort,continue}` | Override `policy.on_stage_failure`. |
| `--timeout SECONDS` | Override per-stage exec timeout. |

### Interactive mode is not wired into the CLI yet

Without `--batch`, the graph pauses at the plan-approval gate (and at the retry-help gate on
stage failure) using a LangGraph `interrupt()`. The CLI does **not** yet handle those
interrupts — it will plan and then stop without executing the stages. **Use `--batch` for
now.** Wiring the interactive resume loop into `cli.py` is a known TODO.

---

## Development

```bash
# fast unit tests (no SALOME needed)
pytest tests/unit

# live tests — require a working SALOME install, ~15-20s startup each
pytest -m salome
```

Tests marked `salome` are skipped unless a real SALOME install is present.

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
  session/          manager (launch/attach SALOME), discovery (omniORB port)
  codegen.py        geom call-log -> Python source
executor/
  executor.py       the in-SALOME process (socket server)
  geom_tools.py     geom tool implementations + per-tool JSON schemas
docs/superpowers/   design spec (rev 6) + implementation plan
```

---

## Known limitations

- Interactive (non-`--batch`) mode isn't hooked up in the CLI (see above).
- Only proven end to end for a single geom stage against a real LLM. Multi-stage
  (geom → smesh) against a real LLM is untested.
- `verify/toplevel_api.json` (the `--no-session` static-check fixture) still holds
  placeholder API member lists — not regenerated from a live `introspect()`.
- geom tool registry covers `make_box`, `fillet_edges`, `check_shape`,
  `list_study_objects` only. Add more (`make_cylinder`, `boolean_*`, `chamfer_edges`)
  using the existing `TOOL_SCHEMAS` pattern in `executor/geom_tools.py`.
- RAG / documentation retrieval was cut from v1.
