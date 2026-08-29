def run_specialist(stage_plan: dict, llm, tools: dict, iteration_cap: int) -> dict:
    messages = [{"role": "user", "content": str(stage_plan)}]
    last_code = ""
    for _ in range(iteration_cap):
        response = llm.chat(messages=messages, tools=list(tools) + ["emit_script"])
        if response["tool_call"] == "emit_script":
            return {"script": response["args"]["code"], "warning": None}
        tool = tools.get(response["tool_call"])
        if tool is not None:
            observation = tool(**response["args"])
            messages.append({"role": "tool", "content": observation})
    return {"script": last_code, "warning": "iteration_cap reached without emit_script"}
