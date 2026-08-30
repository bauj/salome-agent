import json

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


def run_specialist(stage_plan: dict, llm, tools: dict, iteration_cap: int,
                    system_prompt: str | None = None) -> dict:
    messages = [{"role": "user", "content": str(stage_plan)}]
    if system_prompt:
        messages.insert(0, {"role": "system", "content": system_prompt})
    tool_defs = [{"name": n, "parameters": TOOL_SCHEMAS.get(n, {"type": "object", "properties": {}})}
                 for n in list(tools) + ["emit_script"]]
    last_code = ""
    for _ in range(iteration_cap):
        response = llm.chat(messages=messages, tools=tool_defs)
        # Echo the assistant's own tool-call turn back before the tool
        # result — see salome_agent/specialists/geom.py for why (same bug,
        # same fix, found live against a real LLM).
        call_id = response.get("tool_call_id") or "call_0"
        messages.append({"role": "assistant", "content": None, "tool_calls": [
            {"id": call_id, "type": "function",
             "function": {"name": response["tool_call"], "arguments": json.dumps(response["args"])}},
        ]})
        if response["tool_call"] == "emit_script":
            return {"script": response["args"]["code"], "warning": None}
        tool = tools.get(response["tool_call"])
        if tool is not None:
            observation = tool(**response["args"])
            messages.append({"role": "tool", "tool_call_id": call_id, "content": observation})
    return {"script": last_code, "warning": "iteration_cap reached without emit_script"}
