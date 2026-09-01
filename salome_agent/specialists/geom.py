import json

from salome_agent.executor_client import SessionDied, SessionPoisoned
from salome_agent.logs import log

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
every primitive, boolean and transform."""

FINISH_STAGE_SCHEMA = {
    "type": "object",
    "properties": {"result_names": {"type": "array", "items": {"type": "string"}}},
    "required": ["result_names"],
}
FINISH_STAGE_DESCRIPTION = (
    "Finish this stage. `result_names` = the published names that matter; "
    "their dependencies are kept, everything else is dropped."
)


def run_geom_specialist(stage_plan: dict, llm, client, iteration_cap: int,
                        upstream: dict | None = None) -> dict:
    try:
        return _run(stage_plan, llm, client, iteration_cap, upstream)
    except (SessionDied, SessionPoisoned) as e:
        return {"fatal": f"SALOME agent server unavailable: {e}"}


def _run(stage_plan: dict, llm, client, iteration_cap: int,
         upstream: dict | None = None) -> dict:
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
    last_error = None
    # The LLM must see the actual geom tool registry, not just finish_stage —
    # offering only finish_stage leaves it no way to build anything (real bug
    # found running against a live LLM: it had nothing to call and the
    # response came back with no tool_calls at all).
    tools = client.list_tools()["tools"] + [
        {"name": "finish_stage", "parameters": FINISH_STAGE_SCHEMA,
         "description": FINISH_STAGE_DESCRIPTION}]
    for i in range(iteration_cap):
        log.debug("    geom iter %d/%d: asking LLM...", i + 1, iteration_cap)
        response = llm.chat(messages=messages, tools=tools)
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
        messages.append({"role": "tool", "tool_call_id": call_id, "content": str(result)})
    return {"call_log": call_log, "last_error": last_error}
