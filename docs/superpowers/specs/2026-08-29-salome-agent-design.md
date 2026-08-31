# SALOME Agent — Design Spec

**Date:** 2026-08-29
**Status:** Approved for planning
**Author:** hamma.juba@gmail.com (with Claude)
**Revision:** 6 (incorporates three rounds of Opus design review. Round 1:
two-process executor, checkpoint/restore removed, scope cuts, RAG
integration corrected. Round 2 — the GEOM MCP pivot, plus fixes for 5
blockers the pivot introduced: shared session ownership, incremental
codegen, transactional tool rollback, a lightweight custom transport
instead of the reference MCP SDK, and a codegen-friendly call-record
schema. Round 3 found the revision-4 fixes for shared session and geom
rollback/state did not actually close, plus new `CallRecord` schema gaps —
root-caused to `geom_mcp` being a separate OS process in the first place.
Revision 5 collapses `geom_mcp` into the executor as a command family
(decision 17) and fixes the schema gaps (decision 18). Revision 6 cuts
RAG (`rag_and_chatbot` integration) from v1 entirely (decision 19) — it
was never load-bearing for the core plan → specialist → verify → retry →
assemble loop, and pulled in the heaviest dependency set in the project
(torch, chromadb, sentence-transformers, deepagents, gradio) for a
first-attempt-quality improvement the retry loop already covers. Deferred
to a later evolution (§13). See §3.4, §5.4, §6.5, §7, §8, §13.)

## 1. Purpose

A command-line developer/automation tool that turns a natural-language
request ("make a 20 mm cube, fillet its edges 2 mm, mesh it with 5 mm
tetrahedra") into verified, reproducible SALOME Python scripts, executed
against a live SALOME session.

The tool is for the author's own use during SALOME module development. It
is not an end-user assistant, a chatbot, or a simulation pipeline with a
solver. Those are possible later evolutions; the design leaves room for
them but they are out of scope for v1.

### Goals

- One prompt in → an ordered plan → per-stage scripts → one assembled,
  runnable `result.py`.
- Every smesh/paravis stage script is verified twice: statically (parses,
  uses real top-level API) and by execution (runs clean within a time
  limit, produces a sane result). Geom stages are validated inline, per
  tool call, instead (§6.5) — there is no separate script to verify.
- Interactive by default: the run pauses to show the plan and to ask for
  help when a stage cannot be fixed automatically. `--batch` runs
  unattended.
- Execution happens in a real SALOME session; objects are published in
  the shared study so they are visible in an attached GUI (on the user's
  refresh — see §7.4 for the visibility limitation).
- A `--no-session` mode runs planner + specialists + static checks with
  no SALOME, for fast iteration on the LLM half — smesh/paravis stages
  only; a geom stage needs a live executor session (§7.1).

### Non-goals (v1)

- No RAG / documentation retrieval (`rag_and_chatbot` integration).
  Specialists rely on `api_introspect` and the LLM's own training
  knowledge; a failed first attempt goes through the existing retry loop.
  Deferred to v2 (decision 19, §13).
- No solver (code_aster or other). PARAVIS displays a field that already
  exists in a MED or a trivial computed field.
- No SHAPER. Geometry is GEOM (`geompy`) only.
- No web UI, no daemon.
- No parametric re-execution / design-of-experiments.
- No crash recovery. If SALOME dies, the run aborts with the error and
  the user reruns.
- No live GUI auto-refresh / auto-display from the agent (see §7.4).

## 2. Key decisions

| # | Decision |
|---|----------|
| 1 | Target: developer/automation CLI, headless-capable, reproducible scripts as the primary output. |
| 2 | Architecture: orchestrator + specialists as a LangGraph `StateGraph`. Specialist nodes internally run a small tool-calling agent. |
| 3 | Verification: staged — static check (cheap, top-level API only), then execution check (runs in the session under a timeout, sanity-checks the result). Failure feeds back to the specialist with bounded retries. |
| 4 | **Two processes.** The *agent* runs in a normal venv (LLM, RAG, LangGraph). A thin *executor* runs inside a SALOME session and does all `exec()`, introspection, and study I/O. They speak line-delimited JSON over a local socket. |
| 5 | **Rollback = replay, not restore.** No MED checkpoints. To roll back before a retry, the executor resets to a fresh study and replays the already-accepted stage scripts. Those scripts are the source of truth. |
| 6 | Framework: LangGraph `StateGraph` — cyclic flow, typed state, hard gates, `interrupt()` for the two human gates. |
| 7 | Dynamic plan: the planner emits only the stages the request needs (geom-only, geom+mesh, mesh-only on an existing MED, paravis-only, and repeats of a module are allowed). |
| 8 | Session resolution: attach to a running session if one is found or requested; otherwise the agent launches one (`salome shell` with the executor script). `--batch` never launches a GUI. |
| 9 | Interaction: interactive by default (plan gate + retry-help gate). `--batch` disables both. |
| 10 | LLM: provider-agnostic. Any OpenAI-compatible, tool-calling chat model, configured by endpoint + key + model in `config.json`. |
| 11 | *(Deferred to v2 — see decision 19, revision 6.)* ~~RAG: `rag_and_chatbot` is made importable...~~ Originally: `rag_and_chatbot` made importable and installed `-e`; `rag_search` called `DocumentationChatbot.ask()` and returned `sources[].full_content`, not the generated `answer`, reranker disabled. |
| 12 | SALOME build: a recent `master` native build, path in `config.json` (`salome.root`). GUI launch, when a human wants it, is `QT_QPA_PLATFORM=xcb ./salome` — done by the user, not the agent. |
| 13 | **GEOM as intent-level tools, not raw scripts (v1 pivot).** *(Process placement superseded by decision 17 — see revision 5.)* The `geom` stage no longer runs through the raw-script path. Intent-level, primitive+op-tier tools over `geompy` (one tool per primitive/operation family) each validate params and sanity-check their own result inline. The geom specialist becomes a tool-calling client. `smesh` and `paravis` are unaffected in v1 — they keep the raw-script + executor + static/exec verify path from decision 3/4; their own tool surfaces are a later evolution (§13). "MCP" here names the *shape* of the interface (intent-level tools, a client loop), not an interoperability requirement — reusability from other MCP clients was explicitly not a goal. See §3.4, §5.4, §6.5. |
| 14 | *(Superseded by decision 17 — see revision 5.)* ~~One shared session, held by two attached processes.~~ Originally: `session/manager.py` launches (or attaches to) exactly one SALOME session per run, with `executor` and a separate `geom_mcp/server.py` both attached to it. Round-3 review found this split was never actually reconciled (launch ordering, geom-only plans starting no executor, `reset_and_replay` not resetting `geom_mcp`'s own state) — decision 17 removes the split instead of continuing to patch it. |
| 15 | **Geom codegen runs incrementally, not at assemble time.** The moment a geom stage's `finish_stage` succeeds and `advance` fires, its call log is translated to geompy source immediately, filling `StageRecord.script` — the same field smesh/paravis populate on their own success. This is required for `reset_and_replay` (decision 5) to work: a later stage's retry replays *all* prior accepted stage scripts, geom included, and that only works if the script already exists. Confirmed still correct by the round-3 review. See §3.4, §4.3. |
| 16 | **Geom rollback = transactional tools + existing replay, not a new mechanism.** A tool that fails its own inline sanity check deletes what it just published before returning the error — no orphan survives past its own call, so no cleanup step is needed for retries *within* the specialist's loop. A whole-stage retry (graph-level `retry_gate --hint`) reuses the *existing* `reset_and_replay` (decision 5): the generated geompy source is plain code, executed by the executor's `run()`, exactly like a smesh/paravis script. No `rollback_stage` or `delete_object` tool in v1. Decision 17 also makes this work for the case decision 16 didn't originally cover — a whole-stage retry now resets geom's in-memory name→ref bindings too, for free, because they live in the same process `reset_and_replay` already resets. See §3.4, §4.3. |
| 17 | **Collapse `geom_mcp` into the executor as a command family (revision 5, round-3 fix).** No separate `geom_mcp/server.py` process. The executor's existing protocol (§7.2) gains `list_tools` / `call_tool(name, args)` alongside `run`/`sanity`/`reset_and_replay`; geom's tool dispatch (validation + `geompy` call + inline `CheckShape` + transactional cleanup) becomes a module the executor imports and calls in its own process, using the study/session it already has. This eliminates round-3 blockers 1–3 outright (launch ordering, geom-only plans not starting an executor, stale cross-process state on reset) instead of patching each one, and makes decision 14's shared-session goal moot rather than fixed. Trade-off accepted deliberately: a `geompy` segfault now also kills the executor (and any in-flight smesh/paravis exec in the same run), where a separate process would have contained it to `geom_mcp` alone — accepted because `CheckShape`-style checks already catch most bad geometry before a crash, and eliminating three live blockers outweighs the reduced blast radius on the rarer hard-crash case. Everything else about the geom pivot (intent-level tools, tool-calling specialist, inline validation, no separate MCP SDK) is unchanged — only the process boundary is removed. See §3.4, §7. |
| 18 | **CallRecord schema v2 (revision 5, round-3 fix for blockers 4–7).** Four gaps found in the decision-16 schema: (a) no way to represent a list-of-shapes argument (`MakeCutList`/`MakeFuseList`/`MakePartition`) — `ArgRef` gains a `"list"` kind; (b) one tool call can need more than one `geompy` line (e.g. a two-point box needs two `MakeVertex` calls first) — a tool call now returns `list[CallRecord]`, not one; (c) no way to reference a *prior stage's* published object, only a same-stage `out_var` — `ArgRef` gains a `"study"` kind resolved via `find_object(name)` at codegen time; (d) §3.4's pruning rule was stated as direct membership in `finish_stage`'s result set, which is wrong and non-transitive (would drop calls that a kept call still depends on) — corrected to match §5.4's already-correct wording: keep everything *reachable* from `result_names` by walking `ref`/`list` args backward. See §3.4. |
| 19 | **Cut RAG from v1 (revision 6).** No `rag_search` tool, no `rag_and_chatbot` dependency, no `salome_agent/rag.py`. The core loop (planner → specialist → verify/exec → bounded retry → assemble) does not need it: specialists lean on `api_introspect` (authoritative — forwarded to the live executor) and the LLM's own training knowledge for API calls, and a wrong first attempt is already caught and retried by the existing static/exec verify or geom's inline tool validation. Also removes the heaviest dependency set in the project (torch, chromadb, sentence-transformers, deepagents, gradio) and the `rag_and_chatbot` packaging work from the v1 critical path. Reinstate later if retry counts in practice show the LLM is missing obscure API calls RAG would have surfaced — see §13. |

## 3. Architecture

### 3.1 Two processes

```
┌──────────────────────────────┐        line-delimited JSON        ┌───────────────────────────────┐
│  agent process (venv)        │  ◄─────── local socket ────────►  │  executor (inside SALOME)     │
│  - CLI                       │                                   │  - started as:                │
│  - LangGraph StateGraph      │                                   │    salome shell -- executor.py│
│  - planner + specialists     │                                   │    (or attached: connects to  │
│  - LLM client                │                                   │     an existing session)      │
│  - static verifier           │                                   │  - exec(stage script)         │
│                              │                                   │  - introspection (dir/sig)    │
│  - assembler / report        │                                   │  - study reset + replay       │
│                              │                                   │  - sanity checks              │
│                              │                                   │  - timeout + kill of exec     │
└──────────────────────────────┘                                   └───────────────────────────────┘
```

Rationale (from the review):

- SALOME's `PYTHONPATH` front-loads its own numpy / VTK / PySide2. Loading
  a modern LLM/orchestration stack (LangGraph, an OpenAI-compatible
  client) into that same interpreter risks version conflicts. Keep them
  apart. (RAG's own dependency weight — LangChain, ChromaDB, embedding
  models — was the sharpest version of this argument; RAG is cut from v1
  as of revision 6/decision 19, but the split holds regardless, for this
  and the two reasons below.)
- A segfaulting `geompy`/`smesh` call, or a runaway `Compute()`, kills the
  executor only. The agent detects the dropped socket and reports.
- `salome.sg` (real GUI bindings) only exists inside the session
  interpreter. The executor is the only component that could ever drive
  display; the agent never tries.

The executor is deliberately dumb: it holds no plan, no retry logic, no
LLM. It receives commands (`introspect`, `run`, `reset_and_replay`,
`sanity`, `list_study`, `list_tools`, `call_tool`, `shutdown`) and returns
structured results. `list_tools`/`call_tool` are geom's tool-calling
surface (§3.4, decision 17) — mechanical dispatch + validation, not
planning, so the executor stays "dumb" in the same sense as `run`.

### 3.2 Components

| Component | Process | Responsibility |
|-----------|---------|----------------|
| **CLI** (`salome_agent/cli.py`) | agent | Parse prompt + flags, resolve the session and start/attach the one executor (unless `--no-session` — §7.1), build and invoke the graph, stream progress, write outputs, resume the graph after an interrupt. |
| **Executor** (`executor/executor.py`) | SALOME | Socket server. Commands: `introspect`, `run(code, timeout)`, `reset_and_replay(scripts)`, `sanity(module, expectations)`, `list_study`, `list_tools`, `call_tool(name, args)`, `shutdown`. Runs `exec()` in a prepared namespace; enforces the timeout with a watchdog thread that raises/kills; captures stdout, stderr, exception, traceback with the line mapped into the submitted script. Also dispatches geom's tool calls (`executor/geom_tools.py`, decision 17) in the same process/namespace, and holds geom's in-memory name→ref bindings — `reset_and_replay` clears both the study and these bindings in one call. |
| **Executor client** (`salome_agent/executor_client.py`) | agent | Thin RPC wrapper. Detects a dropped connection → `SessionDied`. |
| **Planner node** (`nodes/planner.py`) | agent | Request → ordered `list[StagePlan]`. Each stage: module, intent, params, declared inputs. Only the needed stages; repeats allowed. |
| **Plan gate** (`nodes/gates.py`) | agent | Interactive: show plan; approve / edit / abort. `--batch`: auto-approve. |
| **Specialist nodes** (`specialists/geom.py`, `smesh.py`, `paravis.py`) | agent | Tool-calling agent per module; own system prompt + tools. `smesh`/`paravis` produce stage script text and do not execute (§5.1). `geom` calls the executor's `list_tools`/`call_tool` commands, which execute for real and return a call log; `codegen.py` derives the script from it on success (§3.4, §5.4). |
| **verify_static node** (`verify/static.py`) | agent | `ast.parse`; check calls against the **top-level** introspection map; lint for semantic banned patterns. Failure → `StageError(phase="static")` → back to specialist. |
| **verify_exec node** (`verify/exec_check.py`) | agent | Ask the executor to `run` the script under the timeout, then `sanity`. Failure → `StageError(phase="exec")` → back to specialist (bounded). `SessionDied` → abort run. |
| **Retry-help gate** (`nodes/gates.py`) | agent | Retries exhausted. Interactive: show last error + failed script; hint / skip / abort. `--batch`: apply `policy.on_stage_failure`. |
| **advance node** (`nodes/advance.py`) | agent | Passing stage only: record the accepted script (for geom, run `codegen.py` over the call log first — decision 15), mark `ok`, advance the cursor. (No checkpoint, no display.) |
| **skip node** (`nodes/advance.py`) | agent | Mark the stage `skipped`, advance the cursor. Separate from `advance`. |
| **Assembler node** (`nodes/assembler.py`) | agent | Emit `result.py`: a preamble (`import salome; salome.salome_init()`, module imports, the study-lookup helper) followed by the accepted stage bodies in order, with a header (generated-by, timestamp, prompt, plan). Optionally `ast.parse` the result and, in **launch mode only**, do one headless smoke run in a fresh study (skipped in attach mode — the run's session already holds this run's objects, so replaying `result.py` there would double-publish under the deterministic names of §5.2, and in attach mode it would also pollute the human's study). |
| **Report node** (`nodes/report.py`) | agent | `report.md`: prompt, final plan + params, per-stage status + retry counts, sanity warnings, skipped stages. |
| **Session manager** (`session/manager.py`) | agent | Resolve: attach (given `--port`, or discovered) vs launch (`salome shell -- executor.py`). Start/stop the one executor process it owns. |
| **Session discovery** (`session/discovery.py`) | agent | Scan `OMNIORB_USER_PATH` (defaults to `/tmp`) for `.omniORB_*_*.cfg`; parse `InitRef = NameService=corbaname::host:PORT` from the file **contents** (the filename carries the PID, not the port). |
| **Geom tools** (`executor/geom_tools.py`) | SALOME (inside the executor process, decision 17) | Primitive+op tools over `geompy`, dispatched by the executor's `list_tools`/`call_tool` commands. Transactional on inline sanity failure. See §3.4, §5.4. |

### 3.3 Outputs

Per run, under `<work_dir>/<run_id>/`:

```
result.py              # assembled, runnable end to end (with preamble)
stage_00_geom.py       # per-stage scripts, prefixed by stage index
stage_01_smesh.py
stage_02_paravis.py
report.md
state.json             # final RunState, for debugging
```

`run_id` = timestamp + short random suffix. Filenames are keyed by stage
index, so a plan that repeats a module does not collide.

### 3.4 GEOM tool commands (v1 pivot for the `geom` module only; decision 17 collapses the process split)

```
┌──────────────────────────────┐   line-delimited JSON,            ┌───────────────────────────────┐
│  agent process (venv)        │◄──── Unix socket (§7.2) ─────────►│  executor.py (SALOME)         │
│  - geom specialist = tool-   │                                   │  - one process, one session,  │
│    calling client, calls     │                                   │    started once by             │
│    list_tools/call_tool      │                                   │    session/manager.py (§7.1)  │
│  - smesh/paravis specialists │                                   │  - run/sanity/reset_and_replay│
│    call run/sanity as before │                                   │    (smesh/paravis, §3.1)       │
│                              │                                   │  - list_tools/call_tool, over  │
│                              │                                   │    executor/geom_tools.py:     │
│                              │                                   │    make_box, make_cylinder,    │
│                              │                                   │    boolean_cut/fuse/common,    │
│                              │                                   │    fillet_edges, chamfer_edges │
│                              │                                   │    (edge/face selection args   │
│                              │                                   │    on every tool that needs     │
│                              │                                   │    one — see tool set below),   │
│                              │                                   │    check_shape,                │
│                              │                                   │    list_study_objects,         │
│                              │                                   │    finish_stage                │
│                              │                                   │  - holds geom's name→ref        │
│                              │                                   │    bindings in-process;         │
│                              │                                   │    reset_and_replay clears      │
│                              │                                   │    the study AND these          │
│                              │                                   │    bindings in one call         │
└──────────────────────────────┘                                   └───────────────────────────────┘
```

One process, one session, one socket. There is no second SALOME-side
process and no second session to reconcile — `find_object()` in a smesh
stage sees a shape a prior geom stage published because they were always
the same in-process study; no cross-process story is needed.

**Why not a separate process (superseded decisions 14/16 — see decision
17).** An earlier revision ran geom's tools in a standalone `geom_mcp/`
process attached to the same session as the executor. A round-3 review
found that split was never actually reconciled — launch ordering was
unspecified, a geom-only plan started no executor at all (so
`reset_and_replay` and the assembler smoke run had nothing to run
against), and `reset_and_replay` reset the study through the executor
while `geom_mcp`'s own in-memory ref bindings went stale. Folding geom's
tools into the executor removes the second process instead of continuing
to patch the split. The one thing given up: a `geompy` segfault now takes
down the executor (and any in-flight smesh/paravis work in the same run)
rather than being contained to a separate `geom_mcp` process — accepted
because `CheckShape`-style inline checks already catch most bad geometry
before a hard crash.

**Tools validate, not scripts.** Each tool takes typed params, calls the
matching `geompy` operation, runs its own sanity check inline (e.g.
`geompy.CheckShape` after any shape-producing op), publishes the result
under `study_ns`, and returns one or more **codegen-friendly call
records** (schema below) plus the sanity outcome. It does **not** return
geompy source text.

**Transactional tools (decision 16).** If a tool's inline sanity check
fails *after* it already published an object (e.g. `fillet_edges` runs but
the result fails `CheckShape`), the tool deletes what it just published
before returning the structured error. A retry inside the specialist's
loop never collides with a deterministic name left over from a failed
attempt, and no separate cleanup tool is needed for this case.

**Call-record schema v2 (decision 18 — codegen-friendly, not just
introspection-friendly; fixes round-3 blockers 4–7).** Each tool call
returns a **list** of records (most tools emit one; a few, like a
two-point box, need a short sequence of `geompy` lines):

```python
class ArgRef(TypedDict):
    kind: Literal["literal", "ref", "list", "study"]
    # "literal" — value is used as-is
    # "ref"     — value is an out_var name from an earlier call THIS STAGE
    # "list"    — value is a list[ArgRef], for list-of-shape args
    #             (MakeCutList, MakeFuseList, MakePartition)
    # "study"   — value is a study object name PUBLISHED BY A PRIOR STAGE;
    #             codegen resolves it via find_object(name), same helper
    #             smesh/paravis stages already use (§5.2)
    value: Any

class CallRecord(TypedDict):
    fn: str                            # exact geompy function, e.g. "MakeBoxDXDYDZ"
                                        # (the tool resolves its own dispatch —
                                        # e.g. make_box -> MakeBoxDXDYDZ vs
                                        # MakeBoxTwoPnt by which params were
                                        # given — codegen never re-derives this)
    args: list[ArgRef]                 # positional, in call order
    out_vars: list[str]                # 0+ result variable names this call
                                        # produces (0 for check_shape /
                                        # list_study_objects; 2+ for an
                                        # explode/SubShapeAll-style tool)
    publish: Optional[list[str]]       # study names, if this call published
                                        # anything (None if not; aligned
                                        # index-for-index with the published
                                        # subset of out_vars)
```

A tool call returns `list[CallRecord]` — e.g. a two-point box tool returns
two `MakeVertex` records (one `out_var` each) followed by one
`MakeBoxTwoPnt` record whose args are `kind="ref"` to those two vertices.
Non-producing tools (`check_shape`, `list_study_objects`) return `[]` —
not recordable; the specialist sees their result directly for its own
reasoning, but they never enter the call log or `result.py`. A tool call
referencing a shape a *prior stage* published (e.g. a second geom stage
continuing work on stage 0's result) uses `kind="study"`, not `kind="ref"`
— `ref` only ever resolves within the current stage's own call log. This
keeps `codegen.py` a dumb formatter — one `ArgRef`/`out_vars` → source-line
pass with zero per-op knowledge — instead of re-implementing each tool's
own geompy dispatch.

**Pruning is a transitive closure, not direct membership (decision 18,
fixes blocker 6).** `finish_stage(result_names)` does not keep only the
calls whose `out_vars` literally appear in `result_names` — that would
silently drop a call a kept call still depends on and produce undefined
names in `result.py`. Codegen keeps every call **reachable** from
`result_names`, walking backward through `ref`/`list` args transitively
(§5.4 already stated this correctly; this is the same rule, restated here
so §3.4 and §5.4 agree).

**Codegen runs incrementally (decision 15), not at assembly.** The moment
`finish_stage` succeeds, the agent runs `codegen.py` over that stage's call
log right there, filling `StageRecord.script` immediately (dropping any
call records whose `out_vars` aren't in `finish_stage`'s declared result set
— unreferenced exploratory calls never reach `result.py`). This is what
`reset_and_replay` needs (§4.3): a later stage's retry replays *all* prior
accepted stage scripts, geom included, which only works if the script
exists before that retry happens. The assembler (§3.2) is otherwise
unchanged — it just concatenates already-generated stage scripts in order,
uniformly across geom/smesh/paravis.

## 4. State and graph

### 4.1 State object

LangGraph `StateGraph`, `TypedDict` state, with an in-memory checkpointer
(`MemorySaver`) so `interrupt()` can suspend and the CLI can resume with
`Command(resume=...)`. `state.json` is written once at the end for
debugging; it is not a second live persistence mechanism.

```python
class StagePlan(TypedDict):
    module: Literal["geom", "smesh", "paravis"]
    intent: str
    params: dict            # dims (mm, as-is, no conversion), sizes, field names, ...
    inputs: list[str]       # named upstream study objects, or MED paths, this stage needs

class StageError(TypedDict):
    phase: Literal["static", "exec"]
    message: str
    offending_line: Optional[int]   # line within the stage script
    traceback: Optional[str]

class StageRecord(TypedDict):
    module: str
    status: Literal["pending", "ok", "failed", "skipped"]
    script: Optional[str]           # accepted script, once ok. For geom
                                     # stages, codegen'd from call_log
                                     # IMMEDIATELY on finish_stage (§3.4
                                     # decision 15) — not deferred to
                                     # assemble, so reset_and_replay (§4.3)
                                     # can use it for a later stage's retry.
    call_log: Optional[list[dict]]  # geom stages only: accumulated
                                     # CallRecord entries (§3.4), source
                                     # for codegen; last_error (below) is
                                     # populated from the last failed call
                                     # if the specialist's loop exhausts
                                     # without finish_stage.
    static_retries: int              # smesh/paravis only; geom stages skip
                                     # verify_static entirely (§6.5)
    exec_retries: int                # smesh/paravis only; geom stages skip
                                     # verify_exec entirely (§6.5). Geom's
                                     # own tool-call retries are bounded by
                                     # policy.tool_iteration_cap instead
                                     # (§5.4), counted inside the
                                     # specialist's loop, not here.

class SessionHandle(TypedDict):
    mode: Literal["attach", "launch", "none"]
    port: Optional[int]
    executor_pid: Optional[int]     # only when we launched it
    alive: bool
    owned: bool

class RunState(TypedDict):
    run_id: str
    user_request: str
    config: dict

    plan: list[StagePlan]
    plan_approved: bool
    stage_idx: int

    session: SessionHandle
    study_ns: str                   # study folder for this run's objects

    stages: dict[int, StageRecord]  # keyed by stage index, NOT module
    last_error: Optional[StageError]
    history: Annotated[list[dict], operator.add]   # append-only event log

    final_script: Optional[str]
    report: Optional[str]
```

All per-stage bookkeeping lives in `stages[stage_idx]`. `history` uses an
additive reducer; every other field is last-write-wins and only one node
writes it per step.

### 4.2 Graph

```
START -> plan -> plan_gate
plan_gate --approved--> run_stage
plan_gate --edited-----> plan
plan_gate --abort------> assemble

run_stage --module=smesh|paravis--> verify_static
verify_static --fail, static_retries left--> run_stage
verify_static --fail, exhausted------------> retry_gate
verify_static --ok------------------------> verify_exec

verify_exec --fail, exec_retries left--> run_stage
verify_exec --fail, exhausted----------> retry_gate
verify_exec --ok----------------------> advance
verify_exec --SessionDied-------------> assemble

run_stage --module=geom, finish_stage reached--> advance
run_stage --module=geom, loop exhausted without finish_stage--> retry_gate
run_stage --module=geom, executor connection dropped--> assemble  # SessionDied, §4.3

retry_gate --hint-------> run_stage
retry_gate --skip-------> skip
retry_gate --abort------> assemble

advance --more stages--> run_stage
advance --done---------> assemble
skip ------------------> advance_cursor --more--> run_stage
skip ------------------> advance_cursor --done--> assemble

assemble -> report -> END
```

`run_stage` reads `plan[stage_idx].module` and calls the matching
specialist.

### 4.3 Retry policy

- **Static failures**: `policy.static_retries` (default 3), counted in
  `stages[i].static_retries`. Exhaustion → `retry_gate`.
- **Exec failures**: `policy.exec_retries` (default 2), counted in
  `stages[i].exec_retries`. On each retry the specialist receives
  `last_error` and the previous failed script.
- **Rollback before any retry**: before re-running a stage, the agent
  sends `reset_and_replay` with the accepted scripts of all prior `ok`
  stages. The executor resets to a fresh study and replays them, so the
  retry starts from a known-clean state. **In attach mode `reset` is
  forbidden** — instead the executor only ever *adds* objects under
  `study_ns`, and rollback deletes that folder's contents and replays.
  The user's own study objects are never touched. This applies uniformly
  whether the prior accepted stages were smesh/paravis or geom — geom's
  script exists at this point because codegen already ran on `advance`
  (decision 15), so it replays exactly like any other stage script.
- **Geom's own tool-call retries stay inside the specialist's loop**
  (§5.4) and never trigger `reset_and_replay` — each tool is
  transactional (decision 16) so a failed call's own side effects are
  already cleaned up before the loop tries again. `reset_and_replay` only
  fires for a graph-level `retry_gate --hint` re-entry, i.e. after the
  geom loop has fully exhausted `policy.tool_iteration_cap`.
- **Exhausted → `retry_gate`.** Interactive: hint / skip / abort.
  `--batch`: `policy.on_stage_failure` — `abort` (default) assembles what
  exists and stops; `continue` marks the stage `failed` and proceeds. For
  a geom stage, `last_error` is populated from the last failed tool call
  in the exhausted loop, alongside the partial `call_log`, so the gate has
  something to show.
- **`SessionDied`** (dropped executor socket), at any point → jump to
  `assemble` with whatever stages passed, and note the crash in the
  report. This covers geom tool-call drops too — decision 17 means there
  is only ever the one executor connection to drop.

### 4.4 Interrupts

`plan_gate` and `retry_gate` call `interrupt()`. `--batch` takes the
default branch without pausing.

## 5. Specialists

Same shape for `geom`, `smesh`, `paravis`; differences are the system
prompt and the sanity expectations.

### 5.1 Agent loop (`specialists/base.py`)

```
input:
  - StagePlan (intent + params + inputs)
  - names/types of study objects produced by prior stages
  - last_error + previous failed script, if this is a retry
tools:
  - api_introspect(name) -> str
        Forwarded to the executor: dir() / signature / docstring on the
        live object — the specialist's only source of ground truth on the
        API beyond its own training knowledge (RAG cut from v1, decision
        19). In --no-session mode this returns the cached top-level map
        only.
  - list_study() -> list[str]
        Forwarded to the executor: names + types of objects under study_ns.
  - emit_script(code: str) -> FINAL
        Ends the loop; code is the stage script body.
output: the code string
```

Tool-iteration cap: `policy.tool_iteration_cap` (default 12). On cap, emit
the best script so far and record a warning.

### 5.2 System prompt conventions (all modules)

- Use the standard Python API: `geompy`, `smesh`, `pvsimple`.
- The session and study already exist. Do **not** call
  `salome.salome_init()` and do not create a study. (The assembler adds
  init to `result.py`; the executor namespace already has it.)
- Publish every result with a deterministic, descriptive name under the
  `study_ns` folder.
- No GUI calls (`salome.sg.*`, viewer calls) — they are inert outside the
  GUI interpreter anyway.
- Parameters come from the StagePlan. Lengths are in millimetres, passed
  through unchanged (GEOM is unitless; no conversion). If a required
  value is missing, use a documented default and emit a comment naming
  the assumption.
- Reference upstream objects by their published name via the
  `find_object(name)` helper (defined in the executor namespace and in
  the `result.py` preamble), not by raw study path.

### 5.3 Module specifics

- **geom**: build with `geompy`; publish with `addToStudy` /
  `addToStudyInFather`.
- **smesh**: get the upstream shape with `find_object`; create the mesh,
  set algorithms/hypotheses from params (default: NETGEN 1D-2D-3D tetra
  at the given max size), call `Compute()`.
- **paravis**: load the input MED from `inputs`, find the requested
  field, apply a surface representation coloured by that field, set a
  camera. A screenshot is best-effort (see §7.4) and never a gate.

### 5.4 GEOM specialist as a tool-calling client (v1 pivot)

Same loop shape as §5.1, different tool set — calls go to the executor's
`list_tools`/`call_tool` commands (§3.4, decision 17), not `run`:

```
tools:
  - <geom tools>                      # make_box, make_cylinder, boolean_*,
                                       # fillet_edges, chamfer_edges (both
                                       # take an edge/face selection: a list
                                       # of subshape indices, or "all" —
                                       # without this, fillet/chamfer can't
                                       # target anything specific),
                                       # check_shape, list_study_objects —
                                       # each returns list[CallRecord]
                                       # (§3.4, decision 18) + sanity
                                       # outcome, or a structured error on
                                       # failure. Publishing is implicit in
                                       # every shape-producing tool (no
                                       # standalone publish tool — one
                                       # publish path, one codegen case).
  - finish_stage(result_names) -> FINAL   # ends the loop; result_names
                                       # names which out_vars matter —
                                       # codegen keeps every call reachable
                                       # from them, transitively (§3.4,
                                       # decision 18)
```

`api_introspect` and `emit_script` (§5.1) do not apply to geom — the tool
surface is the introspectable API, and there is no script to emit. A
failed tool call is retried inside this loop like any tool error
(transactional cleanup per decision 16), bounded by
`policy.tool_iteration_cap`. If the cap is hit without `finish_stage`, the
stage goes to `retry_gate` with `last_error` set from the last failed call
(interactive: hint/skip/abort; `--batch`: `policy.on_stage_failure`) same
as an exhausted smesh/paravis stage. A dropped executor connection during
the loop is treated as `SessionDied` (§4.3) — the run jumps to `assemble`.

System-prompt conventions from §5.2 (mm units, no conversion; deterministic
names under `study_ns`; reference upstream objects via `find_object`) still
apply — they now describe how the specialist should choose tool arguments,
not how it should write geompy source.

## 6. Verification

### 6.1 verify_static (`verify/static.py`) — agent process, no session needed

1. `ast.parse` — syntax errors → `StageError(offending_line)`.
2. AST walk — for attribute calls **on the top-level module objects
   only** (`geompy.X`, `smesh.X`, `pvsimple.X`), fail if `X` is not in
   that module's introspection entry. Calls on derived objects (mesh
   objects, hypotheses, `GEOM_Object` CORBA stubs — which expose no
   signatures) are **not** checked here; the exec check is their gate.
   No arity checking.
3. Semantic lint: `salome.salome_init(`, `salome.sg.` / viewer calls,
   creation of a new study. These are correctness issues, not security.
   (No `exec`/`eval`/`subprocess`/path lint — the user runs
   model-written code by design and is the only user.)

### 6.2 Introspection map (`verify/introspect.py` + executor)

Requested from the executor once at startup: `dir()` and, where
available, `inspect.signature` on `geompy`, `smesh`, `pvsimple`. Held in
memory for the run. No disk cache, no Doxygen/HTML signature fallback
(cut — parser project for marginal gain). In `--no-session` mode the map
is loaded from a small checked-in fixture (`verify/toplevel_api.json`),
regenerated by a dev script when the SALOME build changes.

### 6.3 verify_exec (`verify/exec_check.py` + executor `run`/`sanity`) — smesh/paravis only (§6.5)

1. Executor `run(code, timeout=policy.exec_timeout_s)` — executes in a
   namespace pre-bound with `salome`, `geompy`, `smesh`, `pvsimple`,
   `study_ns`, `find_object`. A watchdog thread enforces the timeout and
   kills the running call; timeout → `StageError(message="timeout")`.
   stdout/stderr/exception captured; traceback line mapped to the stage
   script.
2. Exception or timeout → `StageError(phase="exec")` → back to the
   specialist (bounded).
3. Executor `sanity(module, expectations)` (geom no longer routes here —
   its `CheckShape`-equivalent runs inline per tool call, §3.4):
   - **smesh**: `Compute()` returned `True`; `NbNodes() > 0` and
     `NbElements() > 0`. Elements over an aspect-ratio threshold → a
     recorded **warning**, not a failure.
   - **paravis**: the requested field name is present on the loaded MED.
     (Screenshot presence is not checked.)
4. Sanity failure (not warning) → back to the specialist.

### 6.4 exec timeout

`policy.exec_timeout_s` (default 120). Prevents an LLM-chosen tiny mesh
size from running for hours or exhausting memory. The kill is only
reliable because exec lives in the separate executor process.

### 6.5 GEOM stages skip verify_static / verify_exec entirely

`verify_static` (§6.1) and `verify_exec` (§6.3) apply to smesh and paravis
stages only. For geom, validation happens inline in each MCP tool call
(§3.4): param checks before the `geompy` call, `CheckShape`-style sanity
after it. There is no separate static-API-existence pass (the MCP tool
surface **is** the API — you can't call a tool that doesn't exist) and no
separate exec-and-sanity pass (each tool call already ran for real inside
the session and already sanity-checked its own result). `run_stage` routes
a geom stage straight to `advance` on `finish_stage`, or to `retry_gate` on
loop exhaustion (§5.4, §4.2) — the `static`/`exec` phases of `StageError`
(§4.1) do not occur for geom; a geom stage failure is always
`phase="exec"`-shaped (a failed tool call) surfaced and retried inside the
specialist's own loop, never as a graph-level static/exec cycle.

## 7. SALOME session layer

### 7.1 Resolution (`session/manager.py`)

`session/manager.py` and `session/discovery.py` (agent process) resolve
the session and start (or attach) exactly one executor process — this is
the one place resolution happens; nothing downstream launches or resets a
session of its own. Decision 17 removed the second SALOME-side process, so
there is only ever one process to start per run.

1. `--no-session` → `SessionHandle(mode="none")`. No executor at all.
   `verify_exec`, geom tool calls, and the assembler smoke run are all
   skipped; the run stops after producing per-stage scripts and
   `result.py` for smesh/paravis stages only — a geom stage in the plan
   cannot run in `--no-session` mode in v1 (there is no
   introspection-fixture equivalent for tool calls the way §6.2 covers raw
   API existence). If a geom stage is in the plan under `--no-session`,
   `plan_gate` rejects the plan up front with this message, rather than
   failing later at `run_stage`. Documented limitation, not a bug: for
   fast planner/LLM iteration on a geom-only request, iterate on the
   prompt/plan instead.
2. `--port N`, or exactly one session discovered (`session/discovery.py`)
   → **attach**. The executor is started with `salome shell` pointed at
   that port (or, if a GUI is running, the user starts the executor plugin
   — see §7.3) and connects to the existing study. `owned` for the
   *process* may be true, but the *session* is not owned: `reset` is
   forbidden, rollback is folder-scoped (§4.3).
3. Otherwise → **launch**: `salome shell -- executor/executor.py --socket
   PATH`. `owned = True`. The session and the executor are killed on exit.

`--batch` uses only options 1 or 3; it never attaches to a human's
session and never starts a GUI.

### 7.2 Executor protocol (`executor/executor.py`)

Line-delimited JSON over a Unix domain socket. One process, one protocol —
`list_tools`/`call_tool` are geom's tool-calling surface (§3.4, decision
17), not a separate server.

| Command | Payload | Result |
|---------|---------|--------|
| `introspect` | `{names: [...]}` | `{name: {members: [...], signatures: {...}}}` |
| `run` | `{code, timeout}` | `{ok, stdout, stderr, exception, traceback, offending_line}` |
| `reset_and_replay` | `{scripts: [...]}` | `{ok, failed_index?}` — fresh study (or folder-scoped in attach mode) then replay; also clears geom's in-memory name→ref bindings (decision 17) |
| `sanity` | `{module, expectations}` | `{ok, warnings: [...], failures: [...]}` |
| `list_study` | `{}` | `{objects: [{name, type}]}` |
| `list_tools` | `{}` | `{tools: [{name, params_schema}]}` — geom's tool set (§3.4) |
| `call_tool` | `{name, args}` | `{ok, records: list[CallRecord], sanity}` or a structured error (§3.4, decision 18) |
| `shutdown` | `{}` | closes the socket; exits if the session is owned |

The executor maps tracebacks back to the submitted script's line numbers
by executing with a known `<stage>` filename and offset.

### 7.3 Running the executor inside a session

- **Launched session** (`salome shell -- executor.py`): trivial, the
  executor is the main script.
- **Attached headless session**: `salome shell` connects to the running
  port; the executor script runs there.
- **Attached GUI session**: a small SALOME Python plugin
  (`executor/plugin.py`) that a running GUI can load to start the socket
  server on a background thread. Optional for v1 — the common attach case
  is headless. Documented, not required.

### 7.4 GUI visibility — known limitation

Objects the executor publishes go into the shared CORBA study
(`SALOMEDS`), so an attached GUI **can** show them — but only after the
user refreshes the Object Browser, and 3D display is manual. Automatic
`updateObjBrowser` / `createAndDisplayGO` / `FitAll` from the executor
work **only** if the executor runs as the GUI plugin (§7.3), because
`salome.sg` is a no-op stub in any other interpreter. v1 does not promise
live auto-display. If §7.3's plugin path is implemented, the executor
calls display after each `ok` stage; otherwise the user refreshes.
PARAVIS screenshots via in-process `pvsimple` render in a private
ParaView, not the session's PARAVIS view — best-effort artefact only.

## 8. RAG integration — cut from v1 (decision 19)

`rag_and_chatbot` integration (`salome_agent/rag.py`, a `rag_search` tool,
per-module `DocumentationChatbot` instances) is deferred to a later
evolution — see §13. It is not load-bearing for the core loop: specialists
use `api_introspect` (authoritative, forwarded to the live executor) and
the LLM's own training knowledge, and a bad first attempt is already
caught by the existing static/exec verify (smesh/paravis) or inline tool
validation (geom) and retried. Reinstate if practice shows the LLM is
missing obscure API calls RAG would have surfaced.

If and when it is reinstated, re-derive the integration notes (package
layout, `ask()` vs a would-be `retrieve()`, corpus coverage, import side
effects) from `rag_and_chatbot`'s own code rather than assuming this spec
still has them — this section was trimmed to a summary when RAG was cut,
not preserved verbatim. One correction worth carrying forward regardless:
an
earlier version of this section claimed
`rag_and_chatbot/chatbot/config_smesh_chatbot.json`'s API key was a leaked
credential needing rotation before depending on that repo. Confirmed
2026-08-29 via `git ls-files`/`git check-ignore` that `chatbot/config_*.json`
is gitignored and was never tracked — not a leak, no rotation needed.

## 9. Configuration

`config.json` (see `config.example.json`):

```json
{
  "llm": { "base_url": "", "api_key": "", "model": "", "temperature": 0 },
  "salome": {
    "root": "/path/to/your/SALOME"
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

`policy.static_retries`/`exec_retries` apply to smesh/paravis stages only
(§4.1, §6.5); geom stages use `tool_iteration_cap` instead.

CLI flags: `--batch`, `--no-session`, `--port N`, `--work-dir PATH`,
`--config PATH`, `--on-failure {abort,continue}`, `--timeout SECONDS`.

## 10. Project layout

```
salome-agent/
├── pyproject.toml
├── config.example.json
├── salome_agent/
│   ├── __init__.py
│   ├── cli.py
│   ├── graph.py
│   ├── state.py
│   ├── config.py
│   ├── executor_client.py
│   ├── nodes/
│   │   ├── planner.py
│   │   ├── gates.py
│   │   ├── advance.py
│   │   ├── assembler.py
│   │   └── report.py
│   ├── specialists/
│   │   ├── base.py
│   │   ├── geom.py             # tool-calling client (§5.4), calls
│   │   │                       # list_tools/call_tool on the executor
│   │   ├── smesh.py
│   │   └── paravis.py
│   ├── codegen.py               # geom call-log -> geompy source, run on
│   │                            # advance (decision 15), not at assemble
│   ├── verify/
│   │   ├── static.py            # smesh/paravis only
│   │   ├── introspect.py        # smesh/paravis only
│   │   ├── exec_check.py        # smesh/paravis only
│   │   └── toplevel_api.json
│   └── session/
│       ├── manager.py           # resolves the ONE session for the run,
│       │                        # starts/attaches the ONE executor
│       │                        # process (decision 17)
│       └── discovery.py
├── executor/
│   ├── executor.py            # runs under `salome shell` — no agent deps.
│   │                           # Dispatches run/sanity/reset_and_replay
│   │                           # (smesh/paravis) AND list_tools/call_tool
│   │                           # (geom, decision 17) in one process.
│   ├── geom_tools.py           # make_box, make_cylinder, boolean_*,
│   │                            # fillet_edges, chamfer_edges (edge/face
│   │                            # selection args), check_shape,
│   │                            # list_study_objects, finish_stage — each
│   │                            # returns list[CallRecord] (§3.4, decision
│   │                            # 18). Imported by executor.py only.
│   └── plugin.py              # optional GUI-plugin entry point
├── tests/
└── docs/superpowers/specs/
```

The `executor/` tree imports only the SALOME Python API and the stdlib —
no MCP SDK (decision 16), no agent deps. It must never import
`salome_agent/`
or any other venv package.

## 11. Testing strategy

TDD per component. Unit layers mock the LLM and the executor client; real
SALOME only in `@pytest.mark.salome` tests, which CI skips.

| Layer | Approach |
|-------|----------|
| `verify/static` | Pure unit — good/bad script strings, assert pass/fail + line. Uses the fixture top-level map. |
| `verify/exec_check` | Unit with a fake executor client returning canned `run`/`sanity` results — assert routing (retry, exhaust → gate, SessionDied → assemble). |
| `verify/introspect` | Unit — parse a canned `introspect` response into the map; fixture-map loader. |
| planner | Unit, mocked LLM — prompt → assert stages, modules, params (mm, no conversion), inputs, and that a repeated module yields two stages. |
| graph edges / retry | Unit, fake nodes — static-fail loop + cap, exec-fail retries then gate, skip path (skip node, not advance), abort, advance-to-done, SessionDied short-circuit; geom-specific: `finish_stage`→advance, loop-exhausted→retry_gate, dropped executor connection during a geom stage→assemble. |
| state | Unit — `stages` keyed by index; two same-module stages do not collide; `history` reducer accumulates. |
| `session/discovery` | Unit — fixture `/tmp/.omniORB_*` files; assert port parsed from *contents*. |
| `executor` | Integration `@pytest.mark.salome` — start `salome shell -- executor.py`, send `run` for a known-good smesh/paravis script, assert `ok` + shape via `sanity`; send an infinite loop, assert timeout kill; send a syntax error, assert `offending_line`. |
| `executor/geom_tools` | Integration `@pytest.mark.salome` — call each tool against a live session via the executor's `call_tool`; assert `list[CallRecord]` shape (single- and multi-record tools), inline sanity (e.g. `check_shape` after `make_box`), transactional cleanup (a sanity-failing `fillet_edges` leaves no orphan object), and that a bad-param call returns a structured error without crashing the executor. |
| `codegen` | Unit — canned `CallRecord` lists (including `ref`, `list`, and `study` args, multi-`out_vars` calls, and a multi-record tool call) -> assert generated geompy source is syntactically valid (`ast.parse`), resolves refs to the right variables (including cross-stage `find_object` calls for `kind="study"`), and keeps exactly the transitive closure reachable from `finish_stage`'s result names (§3.4, decision 18) — not just direct membership. Plus one `@pytest.mark.salome` **replay-equivalence** test: run generated source in a fresh study, compare `CheckShape` + bounding box + subshape counts against the original live objects (§12), including a case where a prior stage was geom (exercises `reset_and_replay` restoring an equivalent study across a geom→smesh boundary). |
| `session/manager` | Unit, fake process handles — assert manager waits on the executor's ready-handshake before proceeding; launch mode starts the one executor against the resolved session/port; `--no-session` + a geom stage in the plan is rejected at `plan_gate`, not left to fail at `run_stage`. |
| assembler | Unit — accepted stage bodies → `result.py` has a preamble, `ast.parse` clean, stages in order. |
| end-to-end | `@pytest.mark.salome` smoke — `"make a 20 mm cube and mesh it with 5 mm tetrahedra"` → exercises the geom→smesh handoff (executor's `find_object` sees the geom stage's published shape — trivially true post decision 17, same process/study) → `result.py` runs clean under `salome shell` and the study has one solid mesh. |

## 12. Risks and open items

- **Executor packaging under `salome shell`**: the executor must run with
  SALOME's interpreter and see no agent deps. The socket path is passed
  as an argument; the agent waits for a "ready" line before sending
  commands. Startup races are the main integration risk.
- **Attach-mode folder-scoped rollback**: deleting and replaying only the
  `study_ns` folder assumes stage scripts publish *everything* under it.
  For smesh/paravis the static lint flags an `addToStudy` call whose
  parent is not within `study_ns` in attach mode; geom has no static lint
  (§6.5), so its `study_ns` containment is instead enforced tool-side —
  every geom tool that publishes hard-codes `study_ns` as the parent
  folder, so there is no code path for a geom call to publish outside it.
- **Timeout granularity**: a watchdog that interrupts a C-level
  `Compute()` may not return cleanly; the executor may need to hard-exit
  and be relaunched (owned sessions only). Acceptable — the run aborts.
- **`inspect.signature` coverage** on SWIG/CORBA bindings is thin; static
  checking is existence-only by design, so this only affects how helpful
  `api_introspect` output is.
- **GEOM / PARAVIS corpora** absent — v1 runs without them.
- **GUI live display** is not delivered unless the §7.3 plugin path is
  built; the spec treats that as optional.
- **Codegen fidelity**: even with the `CallRecord` schema (§3.4) making
  `codegen.py` a dumb formatter, a per-op equivalence gap (an `ArgRef`
  resolved wrong, an `out_vars` ordering mismatch) would only surface as a
  `result.py` that diverges from what the live tool calls actually built.
  `ast.parse`-clean is not enough evidence; needs a per-call-type unit test
  plus a replay-equivalence check (run generated source in a fresh study,
  compare `CheckShape` + bounding box + subshape counts against the
  original live objects) — see §11.
- **Geom crash blast radius (decision 17 trade-off)**: a `geompy` segfault
  now kills the executor process, taking down any in-flight smesh/paravis
  exec in the same run along with it, where a separate `geom_mcp` process
  would have contained the crash. Accepted deliberately (§3.4) — revisit
  if hard crashes turn out more common in practice than expected.

## 13. Later evolutions (not v1)

- **RAG integration (decision 19), reinstated** — `rag_and_chatbot` as a
  `rag_search` tool on each specialist, if retry counts in practice show
  the LLM is missing obscure API calls that documentation retrieval would
  have surfaced. Prerequisite work either way: `DocumentationChatbot.retrieve()`
  upstream (retrieval without generation, so `rag_search` doesn't pay for
  a second, unwanted LLM call per lookup).
- GUI executor plugin for true live display and PARAVIS-view screenshots.
- Solver specialist (code_aster) between smesh and paravis.
- SHAPER specialist alongside GEOM.
- Smesh and paravis tool-command families, mirroring geom's (§3.4,
  decision 17), on the same executor process.
- High-level parametric-part composite tools on top of geom's
  primitive+op tier (deferred at pivot time — see task 170 discussion).
- Crash recovery: relaunch an owned session and replay accepted stages.
- End-user assistant mode: act on a developer's open study
  conversationally.
