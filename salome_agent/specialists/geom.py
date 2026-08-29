GEOM_SYSTEM_PROMPT = """You build geometry by calling tools, not by writing
geompy source. Reference upstream objects by their published name. Lengths
are millimetres, unconverted. When done, call finish_stage(result_names)
naming which of your published objects matter — everything they transitively
depend on is kept, everything else is dropped."""


def run_geom_specialist(stage_plan: dict, llm, client, iteration_cap: int) -> dict:
    messages = [{"role": "user", "content": str(stage_plan)}]
    call_log: list[dict] = []
    last_error = None
    for _ in range(iteration_cap):
        response = llm.chat(messages=messages, tools=["finish_stage"])
        if response["tool_call"] == "finish_stage":
            return {"call_log": call_log, "result_names": response["args"]["result_names"]}
        result = client.call_tool(response["tool_call"], response["args"])
        if result["ok"]:
            call_log.extend(result["records"])
            last_error = None
        else:
            last_error = {"phase": "exec", "message": result["error"],
                           "offending_line": None, "traceback": None}
        messages.append({"role": "tool", "content": str(result)})
    return {"call_log": call_log, "last_error": last_error}
