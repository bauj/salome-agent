# Generic specialist loop, ModuleSpec and tool registry — design

Task 226 / GH #40. Written 2026-09-21.

## Purpose

`specialists/geom.py:_run` is 351 lines and owns every hard-won behaviour the agent
has: per-iteration tool subsets and `load_tools`, batch execution with per-id tool
messages, terminator-mid-batch handling, `assistant_tool_turn` continuity,
transcript compaction, the repeat-failed-call guard, verify-cadence nudges, the
no-tool-call nudge, and fatal handling for `SessionDied` / `SessionPoisoned`.

`specialists/base.py` — smesh and paravis — is 76 lines and has none of it.

Every one of those behaviours was found live against a real LLM and paid for with a
debugging session. Adding a second tool-calling module (SHAPER, smesh-as-tools, or
194's amend specialist) by copying `geom.py` would mean maintaining two copies of the
subtlest code in the repo. This task makes geom *configuration* rather than code, so
the next module is a `ModuleSpec` and not a fork.

**This is a prerequisite for task 194** (agentic chat). See
`2026-09-21-agentic-chat-design.md`.

## Decisions taken

1. **All three parts ship on one branch** — generic loop, `ModuleSpec`, and the tool
   registry plus package split. One coherent refactor, one review.
2. **One loop, not two.** `base.py` is deleted. A single `ToolSpecialist` covers both
   the tool-calling and the `emit_script` modules, with `ModuleSpec` declaring the
   terminator and how tools execute.
3. **`ModuleSpec` is a frozen dataclass**, not an abstract base class. Most of what
   differs between modules is data; subclassing to hold data invites overriding the
   loop, which is the thing this task exists to prevent.
4. **Strangler order, geom first.** Introduce the new structures alongside the old,
   migrate geom and prove it byte-identical, then absorb smesh/paravis, then the
   registry and package split last.

## Standing constraint

**geom is the module being made to work. smesh and paravis are absorbed
structurally, not improved.** Behaviour-equivalent before and after, including prompt
wording. smesh gets real attention later, shaper and paravis possibly after that.

This constraint decides several design details below — most visibly `render_retry`
being a spec field so each module keeps its exact retry wording.

## Corrections to the filed task

Both found by reading the code:

- **`StagePlan.inputs` is not dead.** `PARAVIS_SYSTEM_PROMPT` tells the model to take
  the MED file path from it (`specialists/paravis.py:29`). What *is* dead is the
  scoping branch in `select_upstream` (`graph.py:89-96`): `planner.py:27` instructs
  the planner to put plain words in `inputs`, never object names, so
  `len(declared) == len(inputs)` essentially never holds and the function always
  returns everything. Delete the branch, keep the field.
- **"Empty catalogue = not routable" needs a caveat.** `_PLANNER_BASE` already carries
  a static one-liner for geom, smesh and paravis. All three must populate `catalogue`
  or the planner loses its routing text. Empty is for 194's amend module, which is
  never a stage.

## The `ModuleSpec` contract

`salome_agent/modules.py`, plus a `MODULES` registry keyed by name.

```python
@dataclass(frozen=True)
class ModuleSpec:
    name: str                                  # "geom"
    prompt: Callable[[str], str]               # verify_cadence -> system prompt
    terminator: Terminator
    execute: Callable[[Any, str, dict], str]   # client, tool name, args -> observation
    tools_for: Callable[[Any], dict]           # client -> {name: tool def}

    core_tools: frozenset[str] = frozenset()
    categories: frozenset[str] = frozenset()
    risky_categories: frozenset[str] = frozenset()
    measure_tool: str | None = None

    render_upstream: Callable[..., str | None] | None = None
    render_retry: Callable[[dict], str | None] | None = None

    call_prefix: str | None = None             # "geompy." — codegen hardcodes this today
    preamble: str = ""                         # assembler init block
    catalogue: str = ""                        # planner one-liner
```

### The terminator

This is what makes "one loop" honest rather than an `if/else` in disguise:

```python
@dataclass(frozen=True)
class Terminator:
    name: str          # "finish_stage" | "emit_script"
    schema: dict
    description: str
    handle: Callable[[dict, list], dict]   # args, call_log -> stage result
```

geom's `handle` returns `{result_names, call_log}`; the script modules' returns
`{script}`. That one callable absorbs the entire difference between the two loops, so
`ToolSpecialist` never asks what mode it is in.

`execute` absorbs the other difference. geom's is `client.call_tool`; the script
modules' is a lookup into the `api_introspect` / `list_study` closures `graph.py`
builds today.

### Inert by configuration, not by branching

Everything geom-only — tool subsets, `load_tools`, verify-cadence nudges, the
placement-risky counter — is switched off by empty configuration rather than by a
branch. `core_tools=frozenset()` and `risky_categories=frozenset()` mean smesh offers
whatever `tools_for` returns and is never nudged.

That is what "behaviour-equivalent, not improved" looks like in practice: smesh and
paravis run through the general loop with every geom-specific lever configured off.

### Two fields worth calling out

- **`call_prefix`** exists because `codegen.py` hardcodes `geompy.` for
  `emit="geompy_call"`. It must become per-module before any second tool-calling
  module can emit code.
- **`catalogue` empty means not routable.** The planner only sees modules with
  non-empty catalogue text. Forward-compat check: 194's amend module is a `ModuleSpec`
  on the same `ToolSpecialist`, with an empty catalogue, so the planner never offers
  "amend" as a stage.

## `ToolSpecialist`

`salome_agent/specialists/tool_specialist.py`.

```python
class ToolSpecialist:
    def __init__(self, spec: ModuleSpec): ...

    def run(self, stage_plan, llm, client, iteration_cap, *,
            blocks=(), tool_selection="hints", verify_cadence="final",
            retry=None, seed_call_log=None) -> dict
```

Returns whatever `spec.terminator.handle` produced, or `{"fatal": ...}` on
`SessionDied` / `SessionPoisoned`, or `{"last_error": ...}` — the same three shapes
`run_geom_specialist` returns today, so `graph.py`'s result handling is untouched.

**The loop body is a move, not a rewrite.** Any behaviour difference before and after
is a bug, not an improvement. That is what makes the golden-file check meaningful.

Three things generalise as they move:

- The terminator is `spec.terminator.name`, not the literal `"finish_stage"`, and its
  result goes through `spec.terminator.handle`.
- Tool execution is `spec.execute(client, name, args)`, not `client.call_tool`.
- `_PLACEMENT_RISKY` (`geom.py:87`) and the measure-tool name come from the spec.
  `list_tools()` entries already carry `"category"` client-side
  (`tool_selection.py:44` reads it), so this ports without new plumbing.

### Prompt blocks move out of the loop

Today `geom.py:_run` builds the user message by concatenating `str(stage_plan)`, the
upstream-shapes block and the retry block — and the upstream block's formatting
(`bbox`, `size`, `centre`) is geom-specific knowledge sitting inside what should be a
generic loop.

So the caller passes `blocks`, a list of pre-rendered strings, which the loop joins
after `str(stage_plan)`. The payoff is forward-looking: 194's amend specialist adds
its build-graph and selection blocks by passing more strings, with no edit to the loop.

The retry block stays inside, because `retry` and `seed_call_log` are coupled — the
seed exists so a reused-by-name shape stays traceable for codegen (task 224 / GH #38)
— and splitting them across the boundary would invite passing one without the other.

**`specialists/base.py` and `specialists/geom.py` both disappear as loop code.** What
remains of geom is its `ModuleSpec`.

## Tool registry and package split

44 tools, each currently maintained across five parallel dicts: `TOOLS` (1437),
`TOOL_CATEGORY` (1487), `ARG_TYPES` (1539), `TOOL_SCHEMAS` (1586),
`TOOL_DESCRIPTIONS` (2020). Five edit sites per tool, and only `category` is
guard-tested.

### The decorator

```python
@tool("make_box", category="primitive",
      description="Create an axis-aligned box ...",
      schema={...}, arg_types={...})
def tool_make_box(executor, args): ...
```

The registry object lives in `executor/registry.py`, **not** under `geom/` — shaper
and smesh will register into their own instances of the same class. `list_tools()`,
`call_tool()` and `_coerce_args()` read from it.

**`list_tools()` wire output stays byte-identical**, asserted both as parsed structure
and as serialized bytes: a field-order change is invisible to a dict comparison and
very visible to a prefix cache.

**The guard tests get better for free.** A registry makes the invariant expressible —
every registered tool has a schema, a description, `arg_types`, and a category from
the known set. One test replaces five opportunities to forget an edit site.

### The split

```
executor/registry.py          @tool + ToolRegistry  (shared with shaper/smesh later)
executor/geom/
  __init__.py                 imports the submodules for registration, exposes the API
  helpers.py                  _next_name, _publish, _resolve, _parse_coords,
                              _resolve_axis / _resolve_plane / _resolve_point, _gui_call
  primitives.py  sketch.py  swept.py  booleans.py
  transforms.py  features.py  display.py  queries.py
```

`executor/geom_tools.py` stays as a re-export shim during the migration commits, so the
~1800-line test file is untouched while the split lands — an unchanged suite passing is
the strongest available evidence that a mechanical move changed nothing. The final
commit rewrites the five production imports plus the test imports and deletes the shim.

Production consumers are only five sites: `agent_server/core.py` (`list_tools`,
`call_tool`), `agent_server/backend.py` and `agent_server/gui_backend.py`
(`reset_counters`), and `salome_agent/nodes/planner.py` (`TOOL_DESCRIPTIONS`).

### The caching gotcha

`agent_server` caches the tools module in `sys.modules`. After the package split a
running server keeps serving the old registry, and **the failure is silent** — stale
tool definitions, no error.

**A full SALOME restart is mandatory before any live verification of this branch.**
This belongs in the plan as a step, not as a remark.

## Integration points

### `graph.py`

The `if module == "geom": ... else:` at line 231 becomes `MODULES[module]`. The node
keeps everything around it — timing and usage accounting, `retry_context`, the three
result shapes, the `_llm_endpoint_fatal` wrapper.

The two prompt blocks are the subtle part. Geom's upstream block is measured (`bbox`,
`size`, `centre`, gathered by calling measure); the script modules' is a bare name list
from `select_upstream`. Both the gathering and the wording differ, so the spec owns
them via `render_upstream` and `render_retry`.

These are chunky callables, and that is the lesser evil: leaving `graph.py` knowing
what a bbox is would defeat the point. `render_retry` is a spec field specifically
because geom's `_retry_block` and the script path's "Your previous attempt failed at
execution with..." are worded differently — unifying them would change smesh's prompt,
which the standing constraint forbids.

### `codegen.py`

`geompy.` is hardcoded for `emit="geompy_call"`; it becomes `spec.call_prefix`, passed
per stage. `method_call` and `gui_call` emit through `recv` and are unaffected.

### `assembler.py`

Only the module-driven preamble parts move to `spec.preamble` — `_GEOM_INIT` behind
`"geom" in modules`, `_SMESH_INIT` behind `{"smesh","paravis"} & modules`.

The body-sniffing heuristics stay exactly as they are: `"SALOMEDS.Color(" in body`,
`"gg." in body`, `"find_object(" in body`, `"study_ns" in body`. Those key on generated
content, not on module; turning them into spec fields would be a behaviour change
dressed as a refactor.

### `planner.py`

`_geom_catalogue()` becomes a build over `MODULES` where `catalogue` is non-empty,
geom's still generated from its registry's descriptions. With all three populated with
their current text the assembled prompt is byte-identical — that is the assertion, not
the hope.

### `StagePlan`

Gains `tools: list[str]`, emitted by the planner since task 193 and never typed.
`interactive.py:_PLAN_KEYS` already lists it.

`inputs` stays (see Corrections). Removing it would mean editing the planner prompt,
which changes the plan JSON, which invalidates every golden baseline this branch
depends on. Deleting the dead scoping branch in `select_upstream` is the part that is
safe here.

## Verification

### Step 0: a deterministic replay harness

There is no golden or replay harness in `tests/` today. "Same `result.py`" is
meaningless against a live LLM that answers differently every run, so this comes first.

A fake LLM client replays a recorded `llm_trace.jsonl`, returning each recorded
response in order instead of calling out. Baseline `result.py` files for the vase and
enclosure prompts are captured **from `main`**, committed as fixtures, and every
subsequent commit on the branch is checked against them.

Without this, the "no behaviour change" claim is unfalsifiable. It is also reusable far
beyond 226: it gives the repo a cheap way to regression-test prompt and loop changes,
which tasks 191, 199, 215 and 225 all had to verify live.

### Commit sequence, each one green

1. `modules.py` — `ModuleSpec`, `Terminator`, `MODULES`, geom's spec. Not yet wired.
2. `tool_specialist.py` — the loop moved out of `geom.py:_run`, geom routed through it,
   `base.py` untouched. Check: vase and enclosure `result.py` byte-identical.
3. smesh and paravis absorbed — specs, `emit_script` terminator, closure `execute`.
   `base.py` deleted, `graph.py`'s fork gone. Check: both system prompts
   byte-identical.
4. `codegen` `call_prefix`, `assembler` preamble, planner catalogue from `MODULES`.
   Check: assembled planner system prompt byte-identical.
5. `executor/registry.py` + `@tool`; geom_tools registers through it, still one file.
   Check: `list_tools()` identical as parsed structure and as serialized bytes.
6. The package split. `geom_tools.py` becomes a shim; the test file is untouched and
   passing is the evidence.
7. Cleanup — five production imports plus test imports rewritten, shim deleted,
   `StagePlan.tools` typed, dead `select_upstream` branch removed.

The registry split sits at 5-6 rather than earlier on purpose: it is the step with the
`sys.modules` gotcha, and by then everything above it is already proven.

### Live verification, after a full SALOME restart

- Vase and enclosure run live and produce working geometry. Compare bboxes within
  tolerance, not bytes — a live LLM will not reproduce a trace.
- A smesh stage still meshes; a paravis stage still loads a MED. Smoke tests for "I did
  not break the modules I was told not to improve", nothing more.
- `result.py` from each still runs standalone under `salome shell`.

## Non-goals

- No behaviour change in geom.
- No quality work on smesh or paravis.
- No `llm.py` changes.
- No `agent_server` wire changes. `forget()`, pinned-name replay and `tool` /
  `tool_args` on records belong to **194's** plan, not this one.
- `StagePlan.inputs` is not removed.
