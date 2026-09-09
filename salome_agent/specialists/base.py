import json

from salome_agent.llm import assistant_tool_turn, continuity_fields
from salome_agent.logs import log

TOOL_SCHEMAS = {
    "api_introspect": {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
    },
    "list_study": {"type": "object", "properties": {}},
    "emit_script": {
        "type": "object",
        "properties": {"code": {"type": "string"}},
        "required": ["code"],
    },
}

TOOL_DESCRIPTIONS = {
    "api_introspect": "Report the members (methods/attributes) of a SALOME API object by name.",
    "list_study": "List the names of every object currently published in the study.",
    "emit_script": "Submit the finished Python script for this stage as `code`. Call once, when done.",
}


def run_specialist(stage_plan: dict, llm, tools: dict, iteration_cap: int,
                    system_prompt: str | None = None,
                    upstream_names: list[str] | None = None,
                    last_error: dict | None = None) -> dict:
    content = str(stage_plan)
    if upstream_names:
        # The planner's stage_plan["inputs"] are names it invented from the
        # request ("cube"); the geom stage actually published something else
        # ("box_1"). Give the specialist the real study names to find_object.
        content += ("\n\nObjects already in the study — call find_object() with "
                    f"exactly these names: {upstream_names}")
    if last_error:
        content += ("\n\nYour previous attempt failed at execution with:\n"
                    f"{last_error.get('message')}\nReturn a corrected script.")
    messages = [{"role": "user", "content": content}]
    if system_prompt:
        messages.insert(0, {"role": "system", "content": system_prompt})
    tool_defs = [{"name": n,
                  "parameters": TOOL_SCHEMAS.get(n, {"type": "object", "properties": {}}),
                  "description": TOOL_DESCRIPTIONS.get(n, "")}
                 for n in list(tools) + ["emit_script"]]
    last_code = ""
    for i in range(iteration_cap):
        log.debug("    specialist iter %d/%d: asking LLM...", i + 1, iteration_cap)
        response = llm.chat(messages=messages, tools=tool_defs,
                            module=stage_plan.get("module"))
        if response.get("content"):
            log.debug("    LLM reasoning: %s", response["content"])
        log.debug("    LLM -> %s", response["tool_call"])
        if response["tool_call"] is None:
            # tool_choice="auto" — prose, no call. Keep it, nudge, retry.
            messages.append({"role": "assistant", "content": response["content"] or "(no reply)",
                             **continuity_fields(response.get("raw_message"))})
            messages.append({"role": "user", "content": "You did not call a tool. "
                             "Call exactly one tool now — emit_script once the script is ready."})
            continue
        # Echo the assistant's own tool-call turn back before the tool
        # result — see salome_agent/specialists/geom.py for why (same bug,
        # same fix, found live against a real LLM).
        call_id = response.get("tool_call_id") or "call_0"
        messages.append(assistant_tool_turn(
            [{"name": response["tool_call"], "args": response["args"], "id": call_id}],
            response.get("raw_message")))
        if response["tool_call"] == "emit_script":
            return {"script": response["args"]["code"], "warning": None}
        tool = tools.get(response["tool_call"])
        if tool is not None:
            observation = tool(**response["args"])
            messages.append({"role": "tool", "tool_call_id": call_id, "content": observation})
    return {"script": last_code, "warning": "iteration_cap reached without emit_script"}
