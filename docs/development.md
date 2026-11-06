# Development guide

How [salome-agent](../README.md) works inside, and how to test and extend it. To use the agent,
the README is enough.

- [Architecture](#architecture)
- [Images](#images)
- [Logs and traces](#logs-and-traces)
- [The agent server](#the-agent-server)
- [Tests](#tests)
- [SALOME API map](#salome-api-map)
- [Layout](#layout)
- [Adding a tool](#adding-a-tool)
- [SMESH element limit](#smesh-element-limit)

## Architecture

Two processes:

| Process   | Runs where | Does |
|-----------|-----------|------|
| **agent**    | your Python venv | planning, LLM calls, stage checks, script assembly (LangGraph `StateGraph`) |
| **server**   | inside a SALOME process — a GUI plugin thread, or headless `python -m agent_server` | executes SHAPER / GEOM / SMESH tool calls, introspection, timeout kills |

The agent connects to the server over loopback TCP (token-authenticated,
line-delimited JSON); it discovers a running server via `~/.local/share/salome-agent/server.json`.

Flow: `request → [vision, when images are given] → plan → [per stage: specialist tool loop → finish_stage check → advance] → assemble result.py → report`

- **shaper** stages — the default CAD module — run as tool calls (`create_sketch`, `sketch_add`, `extrude`, `fillet_edges`, ...) against the server. Each call executes the same source line that goes into `result.py`, so the live document and the script cannot drift.
- **geom** stages run as tool calls (`make_box`, `get_subshapes`, `fillet_edges`, ...) against the server; the script is generated from the call log. The planner picks GEOM only when the request names it ("using GEOM").
- **smesh** stages run as tool calls (`create_mesh`, `mesh_tetra`, `name_subshapes`, `local_size`, `compute`, ...) on a shape an earlier stage built, SHAPER or GEOM. Each call executes the line it records into `result.py`; `compute` answers with element counts, the mesh volume next to the shape's, quality and every part that failed, and `finish_stage` is refused until the mesh holds up. One exception: a mesh whose volume is more than 1% off the shape's is accepted when the user asked for the size that causes it; the model then passes `request_conflict` with the user's own words for that value (refused if those words are not in the request), and the miss is shown to the user as a warning. A defect (not computed, meshing errors, no elements of the stage's dimension, an empty group) is always refused.
- A failed stage stops at the stage-failure gate (or, with `--batch`, follows `on_stage_failure`). A retry continues from what the failed attempt built and is told what that is. GUI mode does not roll back published objects; see [Known limitations](../README.md#known-limitations).

## Images

Each image is base64-encoded (PNG/JPEG/WEBP/GIF/BMP, recognised by content; at most 10
images and 20 MB in all; needs a vision-capable LLM behind `llm.base_url`) and first handed to a dedicated **vision agent**. It extracts what is
visible — the parts and their shapes, dimensions written on the drawing, proportions,
symmetry, and what is ambiguous — as a structured description, which is logged and passed
to the **planner** and to the **SHAPER / GEOM specialist** in place of the raw pictures — neither
sees the images themselves. The planner uses it to shape the stage breakdown, the
specialist for proportions and placement. Multiple
images are treated as different views/plans of the *same* part, reconciled into one 3D
description — not as separate parts. Explicit numeric dimensions in the text request
win, then dimensions read from the image; values the vision agent had to estimate are
marked `~`. SMESH stages don't see the images — it only feeds CAD construction.

The vision call is tagged `module="vision"`, so `llm.modules.vision` can send it to a
different model or endpoint (one that reads drawings better) and `llm.module_extra_body` can
give it its own request body (e.g. a different `reasoning_effort`) like the other modules.

## Logs and traces

Progress goes to stderr (`planning...`, `stage 1/2 [shaper] ...`, `stage 1/2 accepted`, ...).
`-v` adds per-LLM-call detail. `-vv` also dumps every LLM prompt, response and reasoning to the
terminal and `<run_id>/llm_trace.jsonl` (a call cut off, cancelled or dropped still gets a line,
marked `incomplete`, with what it streamed), and writes `<run_id>/llm_live.txt` as each call
streams: `tail -f` it to watch the model's reasoning while a call is in flight, and the tool calls
it ends with. Raw SALOME output goes to `~/.local/share/salome-agent/server.log`.

Aborting at the plan or stage-failure prompt of the CLI stops immediately with exit 1 and writes
nothing. Everything up to a prompt is preserved by the LangGraph checkpointer, and your decision
resumes the same run. If stdin is not a terminal, an unanswered prompt aborts the run.

## The agent server

The agent finds a running server through `~/.local/share/salome-agent/server.json`. One server at
a time.

**In the SALOME GUI.** The plugin menu entry comes with salome-agent-gui. CLI runs find the same
server. Auto-launching SALOME from the agent was tried and reverted: repeated external
attach/detach against a long-lived GUI has crashed the session before.

If an older setup ran `scripts/install-plugin.sh` (removed), delete its
`# --- salome-agent server launcher ---` block from
`~/.config/salome/Plugins/salome_plugins.py`: it puts a git checkout first on
`sys.path`, so SALOME runs that checkout's server instead of the installed one.

**Headless.** `scripts/start-headless-server.sh` runs an isolated study with no GUI, in the
foreground. It logs to `~/.local/share/salome-agent/server.log` and stays up across CLI runs: the
CLI drops its connection but never shuts the server down.

Ctrl-C, SIGTERM and closing the terminal all stop it cleanly and remove `server.json`. This needs
its own handlers: GEOM loads OpenCASCADE's `OSD::SetSignal`, whose SIGINT handler only sets a flag,
so `agent_server/__main__.py` installs Python handlers after the backend starts.

**Headless:** do **not** run `python -m agent_server` from an activated venv — the
wrapper script strips `.venv` from `PATH` first (the venv python lacks `psutil` and
SALOME's `sys.path`). The server refuses the raw `run` command (any code sent with
the token) unless started with `--allow-run`; only the live tests pass it.

**Timeouts.** A stage timeout poisons the server: the run aborts (exit 1). In the GUI, the next
menu click restarts it; if the GUI is frozen by a runaway stage, kill it. The headless server
exits on a poisoning timeout; restart it.

## Tests

```bash
# fast unit tests (no SALOME needed), in parallel (pytest-xdist)
pytest tests/unit -n auto

# live tests — require a working SALOME install, ~15-20s startup each;
# keep these serial (no -n): they share one SALOME server
pytest -m salome

# tests that call a real LLM (spend tokens) — deselected by default
pytest -m llm
```

Tests marked `salome` are skipped unless a real SALOME install is present.
They locate it via `$SALOME_ROOT`, else `salome.root` in `config.json`; with
neither set the `salome` tests skip. Tests marked `llm` call the endpoint in
`config.json`; `pytest.ini` deselects them (`addopts = -m "not llm"`) unless
you pass `-m llm`.

Test layout:

- `tests/unit/` — no SALOME, no network.
- `tests/salome/` — live tests against a headless agent server the fixture
  starts (it refuses to run if another server owns `~/.local/share/salome-agent/server.json`).
- `tests/replay.py` + `tests/fixtures/cassettes/` — a recorded run (LLM replies
  and server responses) replayed strictly in order, no network or SALOME;
  `test_golden_runs.py` checks the replay still produces
  `tests/fixtures/golden/<name>.py`. Record a new cassette from a run directory
  with `scripts/make-cassette.py <run_dir> <name>`.
- `tests/fixtures/golden/planner_prompt.txt`, `list_tools.json` — snapshots of
  the planner prompt and the GEOM tool order; a change shows up as a diff.
- `tests/fixtures/repetition/*.txt` — real reasoning streamed by models
  (`loop_*` repeat themselves, `ok_*` do not), test data for the loop detector
  in `salome_agent/repetition.py`.

## SALOME API map

`salome_agent/verify/toplevel_api.json` is a snapshot of the top-level members
of `geompy`, `smesh` and SHAPER's `model`: a reference for what the installed
SALOME offers.

That fixture has no source but a live session, so regenerate it whenever the
SALOME build changes:

```bash
python3 scripts/regen-toplevel-api.py
```

It reuses a running agent server (`~/.local/share/salome-agent/server.json`, GUI plugin or
headless) if one is up, otherwise spawns a throwaway headless server (needs
`$SALOME_ROOT` or `salome.root` in `config.json`) and tears it down after. A
module whose live introspect comes back empty keeps its existing member list.
`tests/salome/test_toplevel_api_fixture.py` fails if the committed fixture has
drifted from a live introspect.

## Layout

```
salome_agent/
  cli.py            entry point
  graph.py          the LangGraph StateGraph (all nodes + routing)
  runner.py         runs one request through the graph, maps its events
  config.py         config loading + validation
  llm.py            minimal OpenAI-compatible tool-calling client (+ with_images() for multimodal messages)
  repetition.py     catches a model repeating itself while it streams
  calc.py           the calc tool (arithmetic in a locked-down subprocess)
  modules.py        ModuleSpec registry (shaper, geom, smesh)
  state.py          RunState / stage TypedDicts
  interactive.py    plan approval / stage-failure prompts
  chat.py, conversation.py  --chat REPL and the multi-turn conversation
  build_graph.py    the object graph a conversation edits
  replay.py         replays part of the build graph after an edit or /undo
  logs.py           logging setup (-v / -vv)
  nodes/            planner, vision, gates (plan/retry), advance, assembler, report
  specialists/      tool_specialist (shared tool loop), shaper + geom (tool-calling),
                    smesh (tool-calling), amend (chat edits), tool_selection
  verify/           toplevel_api.json (SALOME API snapshot) and its module list
  server_conn.py    finds + connects to the SALOME agent server
  executor_client.py  TCP client for the server (hello handshake, framing)
  codegen.py        call log -> Python source
agent_server/       the server that runs inside SALOME
  core.py           command dispatch + poison gate
  backend.py        Backend protocol + HeadlessBackend
  gui_backend.py    GuiBackend + Qt main-thread marshal
  wire.py           loopback TCP + token listener
  handshake.py      ~/.local/share/salome-agent/server.json
  gui_plugin.py     the Tools > Plugins entry point (launch/restart)
  namespace.py      builds the exec namespace (geompy/smesh/...)
  __main__.py       headless entry (`python -m agent_server`)
executor/
  registry.py       ToolRegistry — tool decorator, schemas, arg-type coercion
  code_guard.py     checks SHAPER / SMESH source built from tool arguments before it runs
  shaper/           SHAPER tool implementations (sketch, features, primitives, ...)
  geom/             geom tool implementations + per-tool JSON schemas
  smesh/            SMESH tool implementations (setup, naming, sizes, groups, compute, export)
scripts/            start-headless-server.sh, regen-toplevel-api.py, make-cassette.py,
                    dump-view.sh (+ dump_view.py, run inside the SALOME GUI)
```

## Adding a tool

Tool registries: SHAPER 55 tools, GEOM 52 (primitives, sketch, swept,
  booleans, transforms, fillet/chamfer, display, query, subshape selection),
  SMESH 25 (setup, naming, local sizes, groups, compute and checks, export).
  The SHAPER stage prompt lists every SHAPER tool in one line each; a call to
  one not loaded yet is refused and the tool loaded. `align` does relative
  placement (move a part so a chosen face/centre meets another's) so the model
  states the spatial relationship instead of computing the translation itself.
  Subshape ids come from
  `get_subshapes` (query — returns ids/centres, emits no `CallRecord`) or
  `explode` (publishes each subshape); `get_subshape` builds one shape from an id
  list. Add a tool with `@tool(name, category=..., description=..., schema=...,
  arg_types=...)` in `executor/geom/<category>.py` (see an existing tool);
  registration order = import order in `executor/geom/__init__.py` and is
  pinned by `tests/fixtures/golden/list_tools.json`. SHAPER tools live in
  `executor/shaper/<category>.py` the same way.

## SMESH element limit

SMESH `compute` refuses a mesh predicted above 1 000 000 elements, but SMESH's
  `Evaluate()` cannot predict every setup. `mesh_body_fitted` and `mesh_prism`
  are counted by their own rule (grid cells, layers × end-face elements).
  Hand-built `set_algorithm` setups that `Evaluate()` leaves uncounted (Prism,
  quadrangle_medial_axis, ...) compute without the limit, with a note in the
  reply; `polyhedron_per_solid` is predicted far too low and slips through.
  NETGEN is predicted 2–6× too high, so a NETGEN mesh can be refused well below
  the limit.
