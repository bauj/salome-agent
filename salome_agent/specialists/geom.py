import json

GEOM_SYSTEM_PROMPT = """You build geometry by calling tools, not by writing
geompy source. Reference upstream objects by their published name. Lengths
are millimetres, unconverted. When done, call finish_stage(result_names)
naming which of your published objects matter — everything they transitively
depend on is kept, everything else is dropped."""

FINISH_STAGE_SCHEMA = {
    "type": "object",
    "properties": {"result_names": {"type": "array", "items": {"type": "string"}}},
    "required": ["result_names"],
}


def run_geom_specialist(stage_plan: dict, llm, client, iteration_cap: int) -> dict:
    messages = [{"role": "system", "content": GEOM_SYSTEM_PROMPT},
                {"role": "user", "content": str(stage_plan)}]
    call_log: list[dict] = []
    last_error = None
    # The LLM must see the actual geom tool registry, not just finish_stage —
    # offering only finish_stage leaves it no way to build anything (real bug
    # found running against a live LLM: it had nothing to call and the
    # response came back with no tool_calls at all).
    tools = client.list_tools()["tools"] + [{"name": "finish_stage", "parameters": FINISH_STAGE_SCHEMA}]
    for _ in range(iteration_cap):
        response = llm.chat(messages=messages, tools=tools)
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
        else:
            last_error = {"phase": "exec", "message": result["error"],
                           "offending_line": None, "traceback": None}
        messages.append({"role": "tool", "tool_call_id": call_id, "content": str(result)})
    return {"call_log": call_log, "last_error": last_error}
