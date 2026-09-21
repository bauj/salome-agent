# Agentic chat mode — design

Task 194 / GH #15. Written 2026-09-21.

## Purpose

Today a run is one-shot: prompt → plan → stages → `result.py` → exit. This design
makes the agent conversational. After the first result, the user keeps talking to it
("make the roof taller", "the window is on the wrong wall", "now add a chimney"), and
each turn amends the same study and the same `result.py` instead of starting over.

This matters beyond convenience. salome-agent will eventually run embedded inside
SALOME, where no host agent (Claude Code, Codex, opencode) exists to supply a
conversational loop. The agent has to bring its own.

## Decisions taken

Four decisions were made before the design and everything below follows from them.

1. **A follow-up turn replays only downstream.** Objects the change cannot affect are
   left alone; the changed node and its dependents are rebuilt. Not a full wipe-and-
   replay (destroys the user's own work, costs `O(whole model)` per turn), and not an
   in-place mutation (study drifts from `result.py`).
2. **v1 ships both surfaces**: a CLI REPL and the converted salome-agent-gui chat
   window, over one shared engine.
3. **Deixis is a contract, not an implementation.** The turn interface carries a
   structured `selection` field; v1 always passes `None`. Task 242 fills it.
4. **The amend path is primary, with `replan` as an escape hatch** — approach C. The
   amend loop handles the common case in one LLM hop and calls a `replan` tool when a
   request is too big to express as local edits, which hands off to the existing
   planner path.

## Prerequisite: task 226

The amend specialist needs the same machinery `specialists/geom.py` already has —
iteration cap, the `failed_calls` byte-identical-repeat guard from task 215,
`load_tools` retrieval, continuity fields, tool-result compaction. Task 226 extracts
that into a module-agnostic `ToolSpecialist` + `ModuleSpec`.

226 lands first, on its own branch and its own spec. Writing `amend.py` as a
near-duplicate would mean several hundred lines of the subtlest code in the repo
existing twice. 226 was waiting for a second consumer to justify it; this is that
consumer.

Two points of contact:

- 226's `ModuleSpec` has a "terminator tool" slot. Amend's `finish_turn` uses it.
- 226's `@tool` registry is where `tool` / `tool_args` capture belongs (see
  "Executor additions"), but 226's non-goals exclude `agent_server` / wire changes.
  **Those changes belong to 194's plan, not 226's.**

## Constraints from the existing code

Verified, and load-bearing:

- `begin_session(run_id)` calls `geom_tools.reset_counters()` and sets
  `name_prefix = f"sa_{run_id}_"`. It runs on every `hello`, i.e. every
  `ExecutorClient` connect. **A conversation must hold one connection for its whole
  life** — reconnecting mid-conversation resets the counters and turn 2's `box_1`
  collides with turn 1's.
- `reset_and_replay(scripts)` is all-or-nothing: it rebuilds the namespace, resets the
  counters and re-runs every script. It cannot serve selective replay. It remains the
  recovery path of last resort.
- There is no delete tool in `geom_tools`.
- One tool call returns a `records` **list** (`_resolve_axis` / `_resolve_plane` emit
  extra records), and `geom.py` does `call_log.extend(result["records"])`.
- Records store the *geompy* call (`MakeBoxDXDYDZ`, positional args). The tool-level
  named args (`dx`, `dy`, `dz`) are discarded at the tool boundary.

That last pair is why a build node is a tool call rather than a record: "make the roof
taller" must edit `dz`, not "positional arg 3 of `MakePrismVecH`".

## The build graph

New module `salome_agent/build_graph.py`, owning the conversation's accumulated call
trace. It is the single source of truth; `result.py` and the SALOME study are both
projections of it, which is what structurally prevents them from drifting.

```
BuildNode:
  id: str                 # "n7", assigned on append, never reused
  turn: int               # which turn created it
  module: str             # "geom" in v1
  tool: str               # the tool the model called
  tool_args: dict         # the named args it sent — what edit_arg addresses
  records: list[dict]     # the emitted call_log records, unchanged shape
  deps: list[str]         # node ids, from kind:"ref" args in records
```

`deps` is computed at append time by resolving each `ref` arg's variable name through a
`var -> node id` index the graph maintains. This is the same relation
`codegen._reachable()` walks today, stored forwards as well as backwards.

Operations:

- `append(tool, tool_args, records, turn)` — a specialist's tool call becomes a node.
- `edit_arg(node_id, param, value)` — replace one named literal in `tool_args`.
- `delete(node_id)` — remove it and, transitively, everything downstream.
- `affected(node_id) -> list[str]` — the node plus its forward closure, topologically
  ordered. This is the replay set.

`edit_arg` and `delete` refuse a node that no longer exists, so a stale reference from
a confused model is a clean error rather than a corrupt graph.

**`edit_arg` accepts a literal replacing a literal and refuses to rewrite a `ref`.**
Changing what a node depends on is a topology change, and topology changes go through
`delete` + rebuild. This keeps the op set to three genuinely distinct operations
instead of letting `edit_arg` become a general graph rewriter.

Persistence: the graph serializes to `build_graph.json` in the run dir, so a
conversation survives a crash and is inspectable when a run goes wrong — the same
affordance `llm_trace.jsonl` gives today.

### Impact on codegen and assembler

`codegen.generate()` takes a node list instead of a flat call_log and emits from each
node's `records`; its reachability prune is unchanged.
`assembler.assemble()` stops iterating `stages` and iterates the graph in topological
order. `result.py` becomes a projection of the graph rather than a concatenation of
per-stage scripts.

Both are verified by golden-file tests (see Testing).

## Executor additions

Three wire changes, all small and all load-bearing:

1. **`call_tool` returns the tool name and args** alongside its records, so a node can
   be built. Without it there is nothing for `edit_arg` to address.
2. **`forget(names)`** — remove published objects from the study and drop their
   `geom_bindings`. Selective replay must delete before it rebuilds, and nothing today
   can do that at object granularity.
3. **Pinned output names on replay.** Re-invoking `tool_make_box` calls `_next_name`,
   so a rebuilt `roof_1` returns as `roof_4`. Study names, `result.py` and the user's
   mental model would all drift on every edit. `call_tool` needs a replay mode binding
   outputs to the names the node already owns instead of allocating new ones.

Replay re-invokes the *tool*, not the geompy call, so the arg normalisation and
validation from task 202 apply on replay exactly as they did on the first build.

## The conversation engine

`salome_agent/conversation.py` owns everything that outlives a turn.

```
Conversation(config, client, llm, run_id)
  .turn(text, *, selection=None, on_event, on_gate, cancel) -> TurnResult
  .undo()
  .graph, .turns, .run_dir
```

It holds one `ExecutorClient` (per the counter constraint), one `LLMClient` (so
`llm.usage` accumulates and the gateway prefix cache stays warm across turns), the
`BuildGraph`, the turn history and one run dir. **The caller owns the client's lifetime
and closes it when the conversation ends** — the CLI REPL for the REPL's life,
`ConversationWorker` for the window's.

### One graph, entry-routed

Rather than a second `StateGraph` for amend turns, `START` gets a router:

```
START -> (build graph empty?) -> describe_image / plan      [first turn]
                              -> amend                      [follow-up turn]

amend -> replay          (ops produced)
      -> plan            (replan called)
```

Everything downstream — `plan_gate`, `run_stage`, `verify_static`, `verify_exec`,
`retry_gate`, `advance`, `skip`, `assemble`, `report` — is reused unchanged. Approach
C's escape hatch costs one edge.

New nodes: `amend`, `replay`.

### Turns share a thread

`MemorySaver` is already in `build_graph()`. The conversation keeps one `thread_id` for
its life; each turn is an `invoke` with a delta — new `user_request`, `turn_idx += 1`,
and a reset of the per-turn fields (`plan`, `stages`, `stage_idx`, `last_error`,
`fatal`, `aborted`).

`history` is already `Annotated[list, operator.add]`, so it accumulates across turns for
free and becomes the transcript the amend specialist reads.

`RunState` gains `build_graph`, `turn_idx`, `selection` and `ops`.

### Run dir

Per conversation, not per turn. `result.py` and `build_graph.json` are rewritten each
turn (both are projections); `report.md` gains a section per turn; `llm_trace.jsonl`
appends as it already does.

### Events

`EventMapper` gains four kinds, so a surface can render a turn without knowing how
amend works:

- `turn_start`
- `graph_edit` — which node, which param, old and new value
- `replay` — how many nodes, which ones
- `turn_done`

## The amend specialist

`salome_agent/specialists/amend.py`, a tool-calling loop on 226's `ToolSpecialist`.

Prompt carries four things: the rendered build graph, the conversation transcript, a
measured snapshot of the current study, and the user's turn text.

The graph renders at tool level — compact, and in the vocabulary the user's words land
in:

```
n1  base_1  = make_box(dx=200, dy=120, dz=80)
n2  roof_1  = make_prism(base=n1, dx=0, dy=0, dz=40)      <- n1
n3  house_1 = fuse(shapes=[base_1, roof_1])               <- n1, n2
```

Study names are the handles; node ids are the addresses the tools take.

### Tool surface

Four new tools on top of the existing geom registry:

| Tool | Purpose |
|------|---------|
| `edit_arg(node, param, value)` | replace one named literal |
| `delete_node(node)` | remove it and its dependents |
| `replan(reason)` | escape hatch to the planner path |
| `finish_turn(summary)` | the `finish_stage` analogue |

Appends use the ordinary geom tools, retrieved through `load_tools` as today. The query
family from tasks 185/192 (`measure`, `shape_info`, `basic_properties`) is reachable
the same way — that is how the model grounds "taller" in the actual current height
instead of guessing, and it costs nothing new.

### Ambiguity

If "the roof" matches two nodes, the specialist says so and the turn ends with
`needs_input`. Resolving ambiguity by highlighting candidates in the viewer needs the
selection channel running in both directions; that belongs with task 242.

### Token cost

The graph is re-rendered every turn and grows with the model, so the amend prompt is
where this design adds per-turn input cost. Because the graph changes each turn, the
prefix cache from task 225 breaks at that boundary.

Rendering tool-level args rather than geompy detail keeps it small for a while.
Summarising older turns is the lever if it stops being small — the same lever tasks
191/225 already pull.

## Replay and undo

Replay of an affected set:

1. Compute the forward closure in topological order.
2. `forget()` the study objects those nodes published.
3. Re-invoke each node's tool with its current `tool_args`, outputs pinned to the
   node's existing names.
4. Append the fresh records back onto the node.

### Undo

**Non-linear undo is free.** "Undo the fillet but keep the chimney" is `delete_node` on
an older node, which the amend specialist already has. No undo stack, no special
casing.

**Linear `/undo`** is the convenience on top: snapshot the graph before each turn;
undo restores the snapshot and replays the difference between the two graphs. The graph
is small JSON, so the snapshot is cheap and the diff gives the affected set directly.
Redo is the same mechanism in the other direction, if it turns out to be wanted.

### No confirmation gate on large replays

The `replay` event carries the node count so a surface can show "rebuilding 14
objects". If that proves painful, `policy.replay_confirm_over` is a one-line addition.
Building the gate before feeling the pain is the wrong order.

## Surfaces

### CLI

A `--chat` flag, not a subcommand. `salome-agent "make a house" --chat` runs the first
turn exactly as today, then drops into a REPL instead of exiting. One-shot stays the
default, so `--batch` runs, scripts and the existing tests are untouched. `--chat` with
`--batch` is an error.

```
chat> make the roof taller
chat> /undo
chat> /show          render the build graph
chat> /result        path + print result.py
chat> /rebuild       full reset_and_replay recovery
chat> /quit
```

Bare text is a turn; `/`-prefixed is a command. `readline` gives history. The existing
`terminal_gate` works unchanged for plan gates on a `replan` turn.

**Ctrl-C cancels the turn, not the REPL.** A SIGINT handler around each turn sets the
existing `cancel` event; the turn unwinds through the rollback path and the prompt
returns. Ctrl-C at an idle prompt, or twice, exits.

### salome-agent-gui

`RunWorker` becomes a long-lived `ConversationWorker`. Today it is a `QThread` per
request that calls `run_request` and exits; the socket must stay on one thread for the
conversation's life, so the new worker is a single thread owning the `Conversation` and
consuming turns from a queue.

The signal surface barely moves — `event`, `gate`, and `done` becomes `turn_done` —
because `EventMapper` is already the boundary and `chat_view` / `timeline` already
consume its events. The four new event kinds give `timeline` something to show during a
replay. `chat_view` needs no change.

Cancel reuses the existing per-turn `threading.Event`. Closing the window ends the
conversation and closes the client.

The window needs the same two recovery affordances the REPL has: an undo action and a
rebuild action, the latter enabled only when the conversation is `degraded`.

**Ordering against the GUI polish tasks:** land the conversion on the current floating
window. Task 250 (chat as a dock widget) is a container change, not a content change,
and doing it after avoids a collision. Task 249's transcript polish is also better done
after — the transcript gains turn boundaries here, so polishing it first would be
polishing the wrong thing.

### The selection slot

`Conversation.turn(text, selection=...)` takes a list of structured references:

```json
{"object": "house_1", "subshape": {"type": "FACE", "index": 3}}
```

It threads through `RunState.selection` into the amend prompt as a "the user currently
has these selected" block. In v1 both surfaces pass `None` and the block never renders.

Task 242 fills it from `salome.sg.getAllSelected()` and resolves entries to subshape
indices via the task-185 query family. Because the shape is fixed now, 242 becomes an
additive change rather than a breaking one.

## Error handling

**The invariant: the graph is never committed unless the study matches it.**

Ops are applied to a copy of the graph. The copy commits only if replay succeeded. If
replay fails, the engine rebuilds the affected subtree from the pre-edit graph to
restore the study.

| Failure | What survives |
|---------|---------------|
| LLM endpoint fatal | Turn fails, ops never committed, **conversation survives** — retype and continue |
| Bad node reference | Error returned to the model as a tool error inside the iteration cap |
| Replay failure | Rollback to the pre-edit subtree |
| Rollback failure | Conversation marked `degraded`; only `/rebuild` gets out |
| `SessionDied` / `SessionPoisoned` | **Conversation ends.** Client is gone, study frozen |

A failed turn must not kill a conversation. That is the difference between this and
today's one-shot run.

`SessionPoisoned` is the exception, and the honest one: this is the live-wedge gotcha
from session 24 (`MakeFillet` hanging in OCC, `wire._active` stuck). Reporting it and
ending the conversation is correct — a reconnect would reset the counters anyway.

Bad-node-reference messages should carry the remedy, not just the complaint:
`"n4 was deleted in turn 3; current nodes are ..."`. That is the shape task 218 wants.

`TurnResult` carries `ok | failed | needs_input | degraded`.

## Testing

`build_graph.py` is pure data — no SALOME, no LLM — so it takes the heaviest unit
coverage for the least money: append / edit / delete, forward closure and topological
order, serialize round-trip, refusing a dead node, refusing to rewrite a `ref`.

Above that:

- Conversation lifecycle: plumbing tests with a fake client and scripted LLM.
- Amend loop: scripted tool calls, asserting the ops produced and the graph rendering
  the model actually sees.
- `codegen` / `assembler` from a graph: **golden-file tests against the existing vase
  and enclosure baselines — same `result.py`, byte for byte.** Same discipline as 226's
  non-goals, and it is what proves the graph rewrite did not change geometry.

None of that counts as done. The standing lesson in this repo, repeated from session 7
through 17, is that a mocked LLM cannot catch message-shape, prompt-thinness or
schema-completeness problems.

**The proof is a live multi-turn run:**

1. Build a house.
2. "Make the roof taller" — assert the roof's bbox z grew, the base is untouched, **and
   the names did not drift**.
3. "Add a chimney".
4. `/undo`.
5. Re-run `result.py` standalone under `salome shell`; confirm it reproduces the study.

The name-drift assertion is explicit because it is the exact failure the pinned-output-
names work exists to prevent, and it is invisible to every unit test.

## Out of scope for v1

- **Manual study edits between turns.** If the user edits the study by hand in SALOME,
  the graph will not know. Detecting it needs a study diff every turn. Noted, not paid
  for.
- Viewer-highlight disambiguation (differentiator 6) — needs 242.
- Vision self-check per turn (differentiator 3) — needs task 209/GH #30. PR #44 lands
  the input half and `llm.with_images()` is reusable, but `describe_images` is a
  one-shot pre-planner node, not a per-turn self-check loop.
- smesh / paravis amend. v1 is geom only.
- Redo.

## Work order

1. **226** — generic specialist loop, `ModuleSpec`, tool registry. Own branch, own spec.
2. Executor additions — `forget(names)`, pinned-name replay, `tool` / `tool_args` on
   records.
3. `build_graph.py` + codegen / assembler rewrite, golden-file verified.
4. `conversation.py` + entry-routed graph.
5. `amend.py` + the four tools.
6. Replay and undo.
7. CLI `--chat`, then the salome-agent-gui worker conversion.
