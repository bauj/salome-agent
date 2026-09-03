import json

from salome_agent.executor_client import SessionDied, SessionPoisoned
from salome_agent.logs import log
from salome_agent.specialists.tool_selection import initial_offered, resolve_load

GEOM_SYSTEM_PROMPT = """You build geometry by calling tools, not by writing
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

Each turn: briefly state your reasoning in the message, then call exactly
one tool. Do not reply without calling a tool.

Every successful tool result reports the shape's bbox, size and centre (mm)
in `shape_info`. After each translate, rotate, fuse or cut, read that
`shape_info` (or call `measure`) and confirm the shape's centre and extent
match what you intended before continuing — if a part is in the wrong
place, correct it now, not at the end. Call `measure` on any upstream
object whose position you need but did not build yourself.

Every transform (translate, rotate, ...) produces a NEW shape that does not
inherit the source's colour. So apply display properties (set_color,
set_transparency, set_display_mode) LAST, once, on the final shape — after
every primitive, boolean and transform.

You are given only a subset of the geom tools, chosen for this stage. If you
need a tool that is not in your list, call load_tools with specific tool
`names` or whole `categories` (primitive, sketch, swept, boolean, transform,
feature, display, query) to add them, then call the tool you wanted."""

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


def _compact_result(result: dict) -> dict:
    """A prior tool result, stripped to what the model reads back on later
    turns. The verbose `records` blob (geompy dispatch detail — only codegen
    needs it, and codegen reads it straight from `call_log`) is dropped;
    linear-growth transcript cost was ~66% of the input on a real run."""
    if not result.get("ok"):
        return {"ok": False, "error": result.get("error")}
    keep = {"ok": True}
    for k in ("result_shape", "shape_info", "measurements", "names", "sanity"):
        if k in result:
            keep[k] = result[k]
    published = [n for rec in result.get("records", []) for n in (rec.get("publish") or [])]
    if published:
        keep["published"] = published
    return keep


def run_geom_specialist(stage_plan: dict, llm, client, iteration_cap: int,
                        upstream: dict | None = None,
                        tool_selection: str = "hints") -> dict:
    try:
        return _run(stage_plan, llm, client, iteration_cap, upstream, tool_selection)
    except (SessionDied, SessionPoisoned) as e:
        return {"fatal": f"SALOME agent server unavailable: {e}"}


def _run(stage_plan: dict, llm, client, iteration_cap: int,
         upstream: dict | None = None, tool_selection: str = "hints") -> dict:
    content = str(stage_plan)
    if upstream:
        lines = []
        for name, info in upstream.items():
            if info:
                lines.append(f"  {name} — bbox {info['bbox']}, size {info['size']}, "
                             f"centre {info['centre']}")
            else:
                lines.append(f"  {name} — (geometry unavailable)")
        content += ("\n\nObjects already in the study — reference them by these "
                    "exact names:\n" + "\n".join(lines))
    messages = [{"role": "system", "content": GEOM_SYSTEM_PROMPT},
                {"role": "user", "content": content}]
    call_log: list[dict] = []
    tool_results: dict[int, dict] = {}  # message index -> full result dict
    last_error = None
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
        # Compact every prior tool result; only the newest keeps full detail.
        latest = max(tool_results) if tool_results else -1
        send = [{**m, "content": str(_compact_result(tool_results[idx]))}
                if idx in tool_results and idx != latest else m
                for idx, m in enumerate(messages)]
        response = llm.chat(messages=send, tools=tools)
        if response.get("content"):
            log.debug("    LLM reasoning: %s", response["content"])
        log.debug("    LLM -> %s(%s)", response["tool_call"], response.get("args"))
        if response["tool_call"] is None:
            # tool_choice="auto" — the model replied with prose and no call.
            # Keep its message, nudge it, and let the loop try again.
            messages.append({"role": "assistant", "content": response["content"] or "(no reply)"})
            messages.append({"role": "user", "content": "You did not call a tool. "
                             "Call exactly one tool now — finish_stage once the geometry is done."})
            continue
        # The assistant's own tool-call turn must be echoed back before the
        # tool result — without it the model has no memory of what it just
        # did (each turn looks like a fresh start) and, seen live against a
        # real LLM, it just calls make_box forever instead of ever finishing.
        call_id = response.get("tool_call_id") or "call_0"
        messages.append({"role": "assistant", "content": None, "tool_calls": [
            {"id": call_id, "type": "function",
             "function": {"name": response["tool_call"], "arguments": json.dumps(response["args"])}},
        ]})
        if response["tool_call"] == "load_tools":
            added = resolve_load(response["args"].get("names"),
                                 response["args"].get("categories"), all_defs)
            new = added - offered
            offered |= added
            obs = (f"added: {', '.join(sorted(new)) or '(nothing new)'} "
                   f"— {len(offered)} tools available")
            log.info("  load_tools(names=%s, categories=%s) -> +%d, %d available",
                     response["args"].get("names"), response["args"].get("categories"),
                     len(new), len(offered))
            messages.append({"role": "tool", "tool_call_id": call_id, "content": obs})
            continue
        if response["tool_call"] == "finish_stage":
            result_names = response["args"]["result_names"]
            # A schema hint doesn't guarantee compliance — seen live, a
            # model returned this as a JSON-encoded string instead of an
            # actual array, which codegen's set(result_names) would then
            # silently iterate as individual characters.
            if isinstance(result_names, str):
                result_names = json.loads(result_names)
            return {"call_log": call_log, "result_names": result_names}
        result = client.call_tool(response["tool_call"], response["args"])
        if result["ok"]:
            call_log.extend(result["records"])
            last_error = None
            log.debug("    tool %s ok", response["tool_call"])
        else:
            last_error = {"phase": "exec", "message": result["error"],
                           "offending_line": None, "traceback": None}
            log.debug("    tool %s failed: %s", response["tool_call"], result["error"])
        tool_results[len(messages)] = result
        messages.append({"role": "tool", "tool_call_id": call_id, "content": str(result)})
    return {"call_log": call_log, "last_error": last_error}
