from salome_agent.modules import ModuleSpec, Terminator, register
from salome_agent.specialists.paravis import PARAVIS_SYSTEM_PROMPT
from salome_agent.specialists.smesh import SMESH_SYSTEM_PROMPT

API_TOOL_SCHEMAS = {
    "api_introspect": {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
    },
    "list_study": {"type": "object", "properties": {}},
}

API_TOOL_DESCRIPTIONS = {
    "api_introspect": "Report the members (methods/attributes) of a SALOME API object by name.",
    "list_study": "List the names of every object currently published in the study.",
}

EMIT_SCRIPT_SCHEMA = {
    "type": "object",
    "properties": {"code": {"type": "string"}},
    "required": ["code"],
}
EMIT_SCRIPT_DESCRIPTION = (
    "Submit the finished Python script for this stage as `code`. Call once, when done.")

_SCRIPT_PREAMBLE = """from salome.smesh import smeshBuilder
smesh = smeshBuilder.New()
"""


def _tools_for(client) -> dict:
    if client is None:
        return {}
    return {n: {"name": n, "parameters": API_TOOL_SCHEMAS[n],
                "description": API_TOOL_DESCRIPTIONS[n]}
            for n in ("api_introspect", "list_study")}


def _execute(client, name: str, args: dict) -> dict:
    if name == "api_introspect":
        return {"ok": True, "observation": str(client.introspect([args["name"]]))}
    if name == "list_study":
        return {"ok": True, "observation": str(client.list_study())}
    return {"ok": True, "observation": f"unknown tool {name!r}"}


def _upstream(client, names) -> "str | None":
    # The planner's stage_plan["inputs"] are names it invented from the
    # request ("cube"); the geom stage actually published something else
    # ("box_1"). Give the specialist the real study names to find_object.
    if not names:
        return None
    return ("Objects already in the study — call find_object() with "
            f"exactly these names: {list(names)}")


def _retry(retry: dict) -> "str | None":
    err = (retry or {}).get("last_error")
    if not err:
        return None
    return ("Your previous attempt failed at execution with:\n"
            f"{err.get('message')}\nReturn a corrected script.")


def _script_spec(name: str, system_prompt: str, catalogue: str) -> ModuleSpec:
    return ModuleSpec(
        name=name,
        prompt=lambda cadence, _p=system_prompt: _p,
        terminator=Terminator(name="emit_script", schema=EMIT_SCRIPT_SCHEMA,
                              description=EMIT_SCRIPT_DESCRIPTION,
                              handle=lambda args, call_log: {"script": args["code"],
                                                             "warning": None}),
        execute=_execute,
        render_result=lambda result: result["observation"],
        tools_for=_tools_for,
        no_tool_nudge=("You did not call a tool. Call exactly one tool now — "
                       "emit_script once the script is ready."),
        on_exhausted=lambda call_log, last_error: {
            "script": "", "warning": "iteration_cap reached without emit_script"},
        render_upstream=_upstream,
        render_retry=_retry,
        preamble=_SCRIPT_PREAMBLE,
        catalogue=catalogue,
    )


SMESH_SPEC = register(_script_spec("smesh", SMESH_SYSTEM_PROMPT, catalogue=""))
PARAVIS_SPEC = register(_script_spec("paravis", PARAVIS_SYSTEM_PROMPT, catalogue=""))
