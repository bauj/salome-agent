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

1. Start an executor inside SALOME via `salome shell`. By default this is an **isolated,
   standalone study** — even if you have a SALOME GUI running, the executor does not
   touch it and objects do not appear in that GUI. (`--port` only picks which omniORB
   port to hand `salome shell`; it does not change the isolation.) To instead publish
   into an open GUI as the run progresses, use `--live-gui` (see below).
2. Plan the stages, run each specialist, verify, and assemble — progress is logged to
   stderr as it goes (`planning...`, `stage 1/2 [geom] ...`, `static check: ok`,
   `exec check: ok`, ...). Add `-v` / `--verbose` for per-LLM-call and per-retry detail.
3. Write output to `<work_dir>/<run_id>/`:
   - `result.py` — the standalone reproducible script
   - `report.md` — per-stage status and retry counts (also printed at the end)
   - `executor.log` — raw SALOME/paraview startup noise from the executor subprocess
     (this is where all that `vtkSocket ... Connection refused` chatter goes now,
     instead of your terminal)

Run the result on its own, with no agent involved:

```bash
/storage/salome_platform/devel/SALOME-master-native-FD44/salome shell -- \
  python3 ~/.local/share/salome-agent/runs/<run_id>/result.py
```

### Options

| Flag | Meaning |
|------|---------|
| `--batch` | Non-interactive. Auto-approve the plan; on stage failure follow `policy.on_stage_failure` (`abort` or `continue`). Without it, the run is interactive — see below. |
| `--no-session` | Planner + specialists + static checks only. No live SALOME session, no exec verification. smesh/paravis only (geom needs a live session). Mutually exclusive with `--live-gui`. |
| `--live-gui` | Publish into a live SALOME GUI as stages run — objects appear in the Object Browser and 3D viewer. Requires a `salome withsession` GUI already running (see below). Mutually exclusive with `--no-session`. |
| `--port N` | Hand the executor the omniORB port `N` instead of auto-discovering it (applies to both the default isolated mode and `--live-gui`). |
| `--work-dir DIR` | Override `config.work_dir`. |
| `--config PATH` | Use a different config file (default: `./config.json`). |
| `--on-failure {abort,continue}` | Override `policy.on_stage_failure`. |
| `--timeout SECONDS` | Override per-stage exec timeout. |
| `-v`, `--verbose` | Per-LLM-call and per-retry detail on top of the milestone log. |

### Live GUI mode (`--live-gui`)

Publishes each stage's objects into a live SALOME GUI, so they show up in the Object
Browser and 3D viewer as the run progresses (instead of an isolated study you never
see).

**Prerequisite — launch the GUI yourself, in a real session:**

```bash
# on Wayland, prefix QT_QPA_PLATFORM=xcb
/storage/salome_platform/devel/SALOME-master-native-FD44/salome withsession
```

A plain `salome` GUI is single-process with an in-process naming service that nothing
can attach to — only `withsession` (multiple servers, real omniNames) works. Then:

```bash
salome-agent "make a 20mm cube and mesh it with 5mm tets" --live-gui --batch
```

The agent discovers the running port (`--port N` overrides), attaches the executor via
`salome remote`, and calls `KernelBasis.setSSLMode(False)` so it shares the GUI's own
engine. After a geom stage, the GEOM module is loaded and connected to the study so
its tree and shapes are visible; the Object Browser is refreshed after every publish.

> Auto-launching the GUI from the agent was tried and reverted — `salome withsession`
> started non-interactively from the agent failed to come up reliably (task 182).

**Caveats — this mode can crash your SALOME session:**

- Hovering the mouse over a *freshly externally-published* Object Browser entry has
  crashed the session. Click to select is fine; don't hover brand-new entries mid-run.
- Repeated attach/detach cycles (several `--live-gui` runs, or the
  `tests/salome/test_live_gui.py` test, against one long-lived GUI) have also brought
  the session down once, with no core dump and no identified root cause (task 181).

Save your work before a `--live-gui` run, and treat the mode as demo/interactive
only — not for unattended batches.

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

- **`--live-gui` is new and lightly tested** — the attach path and per-publish refresh
  work (verified live for geom and geom → smesh), but attaching external clients to a
  live GUI has crashed the session at least once (no root cause yet), and rollback-retry
  (`reset_and_replay`) re-runs scripts against the live study, which can leave duplicate
  objects. Demos only; save your work first.
- `verify/toplevel_api.json` (the `--no-session` static-check fixture) still holds
  placeholder API member lists — not regenerated from a live `introspect()`.
- geom tool registry covers `make_box`, `fillet_edges`, `check_shape`,
  `list_study_objects` only. Add more (`make_cylinder`, `boolean_*`, `chamfer_edges`)
  using the existing `TOOL_SCHEMAS` pattern in `executor/geom_tools.py`.
- RAG / documentation retrieval was cut from v1.
