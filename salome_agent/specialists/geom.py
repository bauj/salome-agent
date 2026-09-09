import json

from salome_agent.codegen import _refs_in
from salome_agent.executor_client import SessionDied, SessionPoisoned
from salome_agent.llm import assistant_tool_turn, continuity_fields
from salome_agent.logs import log
from salome_agent.specialists.tool_selection import initial_offered, resolve_load

_GEOM_BASE = """You build geometry by calling tools, not by writing
geompy source. Each tool carries its own description — read it, especially
for the default placement and orientation of primitives. Reference any
shape, point, vector or plane by the name it was published under. For an
axis you may pass "OX", "OY" or "OZ" instead of building a vector. Lengths
are millimetres, unconverted; angles are in degrees.

When done, call finish_stage(result_names) naming which of your published
objects matter — everything they transitively depend on is kept, everything
else is dropped.

Do only what the request asks. Do not invent colours, transparency or
display modes that were not requested.

If a tool result carries a `delta` note, it flagged something numerically
wrong with that boolean or transform (a cut that removed nothing, a fuse of
parts that never touched, a rigid move that changed volume) — fix the cause
before you continue. If a call comes back saying an identical call already
failed, do not retry it unchanged: change the arguments or gather
information first.

Each turn: state your reasoning in at most 2-3 sentences, then call one or
more tools. Do not reply without calling a tool. You may batch several
INDEPENDENT calls in one turn (e.g. the four walls of a frame, or four
translates of shapes that already exist). Keep DEPENDENT steps in separate
turns -- if one call needs the name of a shape another call is about to
create, it must wait for the next turn.
{verify}
To place one part relative to another -- on top of, butted against a face,
flush with an edge, centred on it -- use `align` rather than measuring both
shapes and working out the translation yourself. `align` reads both bounding
boxes and moves the part for you.

Every transform (translate, rotate, ...) produces a NEW shape that does not
inherit the source's colour. So apply display properties (set_color,
set_transparency, set_display_mode) LAST, once, on the final shape — after
every primitive, boolean and transform.

To act on specific edges or faces (e.g. "fillet the top four edges"), first
call get_subshapes to get their integer ids -- by where = "all", a centre
axis (sorted_by / take / count), size (by_size / count), or a plane
(on_plane / state) -- then pass that id list straight to fillet_edges,
chamfer_edges or get_subshape. fillet_edges / chamfer_edges also accept the
same where predicate directly instead of an id list. get_subshapes only
reports ids and centres;
it changes no geometry. Use explode when you instead want every subshape
published as its own named object.

You are given only a subset of the geom tools, chosen for this stage. If you
need a tool that is not in your list, call load_tools with specific tool
`names` or whole `categories` (primitive, sketch, swept, boolean, transform,
feature, display, query) to add them, then call the tool you wanted."""

# policy.verify_cadence — how hard the specialist is pushed to self-check
# shape placement. shape_info rides on every ok result for free; the question
# is how often to spend an explicit `measure` round-trip.
_VERIFY_HINT = {
    "none": "",
    "final": (
        "\nEvery successful tool result reports the shape's bbox, size and "
        "centre (mm) in `shape_info` — glance at it each time. Call `measure` "
        "explicitly only when a position looks wrong, and on your result "
        "shape(s) once before finish_stage. Call `measure` on any upstream "
        "object whose position you need but did not build yourself.\n"
    ),
    "strict": (
        "\nEvery successful tool result reports the shape's bbox, size and "
        "centre (mm) in `shape_info`. After each translate, rotate, fuse or "
        "cut, read that `shape_info` (or call `measure`) and confirm the "
        "shape's centre and extent match what you intended before continuing "
        "— if a part is in the wrong place, correct it now, not at the end. "
        "Call `measure` on any upstream object whose position you need but "
        "did not build yourself.\n"
    ),
}

# Categories whose output can land in the wrong place — the ones a measure
# self-check is actually about.
_PLACEMENT_RISKY = {"boolean", "transform"}
_MEASURE_TOOLS = {"measure", "check_shape"}
# strict: nudge once every N risky ops with no intervening measure.
_STRICT_NUDGE_EVERY = 3


def _geom_prompt(verify_cadence: str = "final") -> str:
    return _GEOM_BASE.format(verify=_VERIFY_HINT.get(verify_cadence, _VERIFY_HINT["final"]))


GEOM_SYSTEM_PROMPT = _geom_prompt("final")

FINISH_STAGE_SCHEMA = {
    "type": "object",
    "properties": {"result_names": {"type": "array", "items": {"type": "string"}}},
    "required": ["result_names"],
}
FINISH_STAGE_DESCRIPTION = (
    "Finish this stage. `result_names` = the published names that matter; "
    "their dependencies are kept, everything else is dropped."
)

LOAD_TOOLS_SCHEMA = {
    "type": "object",
    "properties": {
        "names": {"type": "array", "items": {"type": "string"},
                  "description": "specific geom tool names to add"},
        "categories": {"type": "array", "items": {
            "type": "string",
            "enum": ["primitive", "sketch", "swept", "boolean",
                     "transform", "feature", "display", "query"]},
            "description": "whole tool categories to add"},
    },
}
LOAD_TOOLS_DESCRIPTION = (
    "Add geom tools to your available set for the rest of this stage. Pass "
    "tool `names`, a list of `categories`, or both. Use it whenever the tool "
    "you need is not in your current list."
)


# Boolean ops consume their operands into the result — an operand kept in
# result_names alongside the boolean's output is a superseded intermediate.
# Transforms (translate/rotate/align/...) are copy semantics: the source
# stays independently valid, so it is NOT dropped.
_BOOLEAN_FNS = {"MakeFuse", "MakeFuseList", "MakeCut", "MakeCutList",
                "MakeCommon", "MakeCommonList", "MakeBoolean", "MakePartition"}


def _leaf_result_names(call_log: list[dict], names: list[str]) -> list[str]:
    """Drop any name that a boolean op folded into another kept result. An
    over-split stage that finish_stage's a pre-boolean operand would otherwise
    hand the superseded shape to the next stage, which then builds on the
    wrong object (GH #9)."""
    names = list(dict.fromkeys(names))
    if len(names) < 2:
        return names
    kept = set(names)
    consumed: set[str] = set()
    for call in call_log:
        if call["fn"] in _BOOLEAN_FNS and kept & set(call["out_vars"]):
            for arg in call["args"]:
                consumed |= _refs_in(arg)
    leaves = [n for n in names if n not in consumed]
    return leaves or names


def _call_key(name: str, args: dict) -> tuple:
    return (name, json.dumps(args, sort_keys=True, default=str))


def _compact_result(result: dict) -> dict:
    """A prior tool result, stripped to what the model reads back on later
    turns. The verbose `records` blob (geompy dispatch detail — only codegen
    needs it, and codegen reads it straight from `call_log`) is dropped;
    linear-growth transcript cost was ~66% of the input on a real run."""
    if not result.get("ok"):
        return {"ok": False, "error": result.get("error")}
    keep = {"ok": True}
    for k in ("result_shape", "shape_info", "delta", "measurements", "names", "sanity",
              "ids", "centres", "distance", "properties"):
        if k in result:
            keep[k] = result[k]
    published = [n for rec in result.get("records", []) for n in (rec.get("publish") or [])]
    if published:
        keep["published"] = published
    return keep


def run_geom_specialist(stage_plan: dict, llm, client, iteration_cap: int,
                        upstream: dict | None = None,
                        tool_selection: str = "hints",
                        verify_cadence: str = "final",
                        retry: dict | None = None,
                        seed_call_log: list[dict] | None = None) -> dict:
    try:
        return _run(stage_plan, llm, client, iteration_cap, upstream,
                    tool_selection, verify_cadence, retry, seed_call_log)
    except (SessionDied, SessionPoisoned) as e:
        return {"fatal": f"SALOME agent server unavailable: {e}"}


def _retry_block(retry: dict) -> str:
    lines = []
    err = retry.get("last_error")
    if err:
        lines.append(f"Your previous attempt at this stage failed: {err.get('message')}")
    if retry.get("hint"):
        lines.append(f"User hint for this retry: {retry['hint']}")
    built = retry.get("built") or []
    if built:
        lines.append("Shapes your previous attempt already built and published — reuse "
                     "them if they are still correct, or rebuild only the ones the error "
                     "was actually in:")
        built_info = retry.get("built_info") or {}
        for name in built:
            info = built_info.get(name)
            if info:
                lines.append(f"  {name} — bbox {info['bbox']}, size {info['size']}, "
                             f"centre {info['centre']}")
            else:
                lines.append(f"  {name}")
    return "\n".join(lines)


def _run(stage_plan: dict, llm, client, iteration_cap: int,
         upstream: dict | None = None, tool_selection: str = "hints",
         verify_cadence: str = "final", retry: dict | None = None,
         seed_call_log: list[dict] | None = None) -> dict:
    content = str(stage_plan)
    if upstream:
        lines = []
        for name, info in upstream.items():
            if info:
                lines.append(f"  {name} — bbox {info['bbox']}, size {info['size']}, "
                             f"centre {info['centre']}")
            else:
                lines.append(f"  {name} — (geometry unavailable)")
        content += ("\n\nObjects built by an EARLIER stage, given only for reference "
                    "(e.g. to measure or align against) — reference them by these exact "
                    "names, but do not modify, colour or finish_stage them unless this "
                    "stage's own task says to:\n" + "\n".join(lines))
    if retry:
        content += "\n\n" + _retry_block(retry)
    messages = [{"role": "system", "content": _geom_prompt(verify_cadence)},
                {"role": "user", "content": content}]
    # A retry seeds call_log with the failed attempt's records so a shape the
    # model reuses by name (rather than rebuilding) stays traceable by
    # codegen's reachability pass instead of silently dropping out of
    # result.py (GH #38).
    call_log: list[dict] = list(seed_call_log) if seed_call_log else []
    last_error = None
    # Stage-local record of calls that already failed: hash((tool, args)) ->
    # error. A byte-identical repeat is answered here without a round-trip to
    # the server, and the model is told it is repeating itself (seen live:
    # weak models loop 4x on the same failing call, inventing a new wrong
    # diagnosis each time — task 215). Reset per stage.
    failed_calls: dict[tuple, str] = {}
    # policy.verify_cadence bookkeeping: count placement-risky ops (boolean /
    # transform) since the last measure/check_shape. "final" nudges once at
    # finish_stage if any are outstanding; "strict" also nudges mid-loop every
    # _STRICT_NUDGE_EVERY of them.
    risky_since_measure = 0
    nudged_final = False
    # Per-stage tool retrieval: offer CORE_TOOLS + the planner's hints, not the
    # whole registry (which was ~66% of a run's input tokens). The model widens
    # its set on demand via load_tools. "all" restores the send-everything
    # behaviour. finish_stage/load_tools are always appended, never in `offered`.
    all_defs = {t["name"]: t for t in client.list_tools()["tools"]}
    finish_def = {"name": "finish_stage", "parameters": FINISH_STAGE_SCHEMA,
                  "description": FINISH_STAGE_DESCRIPTION}
    load_def = {"name": "load_tools", "parameters": LOAD_TOOLS_SCHEMA,
                "description": LOAD_TOOLS_DESCRIPTION}
    if tool_selection == "all":
        offered = set(all_defs)
    else:
        offered = initial_offered(stage_plan, set(all_defs))
    log.info("  geom stage: offered %d/%d tools (%s)",
             len(offered), len(all_defs), ", ".join(sorted(offered)))
    for i in range(iteration_cap):
        log.debug("    geom iter %d/%d: asking LLM...", i + 1, iteration_cap)
        tools = ([all_defs[n] for n in sorted(offered) if n in all_defs]
                 + [finish_def, load_def])
        # Tool results are stored already compacted (see below), so `messages`
        # is sent verbatim: each turn's request extends the previous one byte
        # for byte, which keeps a gateway's prefix/KV cache alive. `load_tools`
        # widening `tools` mid-stage is the one remaining prefix break.
        # A shallow copy per call: the message dicts are never mutated after
        # append, so the copy is byte-identical content, and each request is a
        # verbatim extension of the last.
        response = llm.chat(messages=list(messages), tools=tools)
        if response.get("content"):
            log.debug("    LLM reasoning: %s", response["content"])
        calls = response["tool_calls"]
        log.debug("    LLM -> %s", [(c["name"], c["args"]) for c in calls])
        if not calls:
            # tool_choice="auto" — the model replied with prose and no call.
            # Keep its message, nudge it, and let the loop try again.
            messages.append({"role": "assistant", "content": response["content"] or "(no reply)",
                             **continuity_fields(response.get("raw_message"))})
            messages.append({"role": "user", "content": "You did not call a tool. "
                             "Call at least one tool now — finish_stage once the geometry is done."})
            continue
        # The assistant's own tool-call turn must be echoed back before any
        # tool result — without it the model has no memory of what it just
        # did (each turn looks like a fresh start) and, seen live against a
        # real LLM, it just calls make_box forever instead of ever finishing.
        # The model may batch several independent calls in one turn; every
        # one gets executed and answered below, in order.
        for j, c in enumerate(calls):
            c.setdefault("id", None)
            c["id"] = c["id"] or f"call_{i}_{j}"
        messages.append(assistant_tool_turn(calls, response.get("raw_message")))
        batch_error = None
        finish_call = None
        for k, c in enumerate(calls):
            name, args, call_id = c["name"], c["args"], c["id"]
            if name == "finish_stage":
                finish_call = (call_id, args)
                # Anything the model batched after finish_stage still needs a
                # tool response for its id, or the next request is malformed.
                for skipped in calls[k + 1:]:
                    messages.append({"role": "tool", "tool_call_id": skipped["id"],
                                     "content": "skipped — finish_stage was called"})
                break
            if name == "load_tools":
                added = resolve_load(args.get("names"), args.get("categories"), all_defs)
                new = added - offered
                offered |= added
                obs = (f"added: {', '.join(sorted(new)) or '(nothing new)'} "
                       f"— {len(offered)} tools available")
                log.info("  load_tools(names=%s, categories=%s) -> +%d, %d available",
                         args.get("names"), args.get("categories"), len(new), len(offered))
                messages.append({"role": "tool", "tool_call_id": call_id, "content": obs})
                continue
            key = _call_key(name, args)
            if key in failed_calls:
                log.info("  repeat of a call that already failed (%s) — answered "
                         "locally, no round-trip", name)
                result = {"ok": False, "repeat": True, "error": (
                    f"identical call already failed with: {failed_calls[key]}. "
                    "Do not repeat it — change the arguments or the approach, or "
                    "gather information first (measure / get_subshapes / check_shape).")}
                batch_error = {"phase": "exec", "message": failed_calls[key],
                               "offending_line": None, "traceback": None}
                messages.append({"role": "tool", "tool_call_id": call_id,
                                 "content": str(_compact_result(result))})
                continue
            result = client.call_tool(name, args)
            if result["ok"]:
                call_log.extend(result["records"])
                log.debug("    tool %s ok", name)
                if name in _MEASURE_TOOLS:
                    risky_since_measure = 0
                elif all_defs.get(name, {}).get("category") in _PLACEMENT_RISKY:
                    risky_since_measure += 1
            else:
                failed_calls[key] = result["error"]
                batch_error = {"phase": "exec", "message": result["error"],
                               "offending_line": None, "traceback": None}
                log.debug("    tool %s failed: %s", name, result["error"])
            messages.append({"role": "tool", "tool_call_id": call_id,
                             "content": str(_compact_result(result))})
        # A later ok call in the same batch must not mask an earlier failure —
        # settle last_error once per turn from the batch outcome.
        last_error = batch_error

        if finish_call is not None:
            call_id, args = finish_call
            if (verify_cadence in ("final", "strict")
                    and risky_since_measure > 0 and not nudged_final):
                nudged_final = True
                log.info("  verify_cadence=%s: nudging a measure before finish_stage",
                         verify_cadence)
                messages.append({"role": "tool", "tool_call_id": call_id, "content":
                    "You have moved or combined shapes since your last measure. "
                    "Call `measure` on your result shape(s) and confirm their "
                    "placement, then call finish_stage again."})
                continue
            result_names = args["result_names"]
            # A schema hint doesn't guarantee compliance — seen live, a model
            # returned this as a JSON-encoded string instead of an actual
            # array, which codegen's set(result_names) would then silently
            # iterate as individual characters.
            if isinstance(result_names, str):
                result_names = json.loads(result_names)
            result_names = _leaf_result_names(call_log, result_names)
            return {"call_log": call_log, "result_names": result_names}

        if verify_cadence == "strict" and risky_since_measure >= _STRICT_NUDGE_EVERY:
            log.info("  verify_cadence=strict: nudging a measure (%d risky ops)",
                     risky_since_measure)
            messages.append({"role": "user", "content":
                f"You have made {risky_since_measure} shapes since your last "
                "measure. Call `measure` on your recent shapes and confirm "
                "their placement before continuing."})
            risky_since_measure = 0
    return {"call_log": call_log, "last_error": last_error}
