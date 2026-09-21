"""Pins exactly what geom's ToolSpecialist SENDS the model in `tools`. Neither
test_specialist_geom.py (asserts on results, not requests) nor
test_golden_runs.py (ReplayLLM ignores the request payload) can see a
prompt-surface regression like an enum silently changing order — this is the
guard that closes that hole."""

from unittest.mock import MagicMock

from salome_agent.specialists.geom import FINISH_STAGE_SCHEMA, GEOM_SPEC
from salome_agent.specialists.tool_specialist import ToolSpecialist

# Literal, not re-derived from GEOM_SPEC — a guard that recomputes its own
# expectation from the same source it is guarding proves nothing.
_EXPECTED_TOOL_ORDER = [
    "align", "check_shape", "cut", "fuse", "get_subshapes",
    "list_study_objects", "make_box", "measure", "translate",
    "finish_stage", "load_tools",
]

_EXPECTED_LOAD_TOOLS_PARAMETERS = {
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


def _defs():
    # exactly CORE_TOOLS, so `offered` is deterministic with no planner hints.
    names = ["align", "check_shape", "cut", "fuse", "get_subshapes",
             "list_study_objects", "make_box", "measure", "translate"]
    return [{"name": n, "parameters": {"type": "object", "properties": {}},
             "description": "", "category": "query"} for n in names]


def test_geom_tool_payload_is_pinned():
    llm = MagicMock()
    llm.chat.return_value = {"tool_calls": [{"name": "finish_stage",
                                             "args": {"result_names": []}, "id": None}],
                             "content": None, "raw_message": None}
    client = MagicMock()
    client.list_tools.return_value = {"tools": _defs()}

    ToolSpecialist(GEOM_SPEC).run({"module": "geom", "intent": "x", "params": {}, "inputs": []},
                                  llm, client, iteration_cap=1)

    tools = llm.chat.call_args.kwargs["tools"]
    assert [t["name"] for t in tools] == _EXPECTED_TOOL_ORDER

    load_tools = next(t for t in tools if t["name"] == "load_tools")
    assert load_tools["parameters"] == _EXPECTED_LOAD_TOOLS_PARAMETERS

    finish_stage = next(t for t in tools if t["name"] == "finish_stage")
    assert finish_stage["parameters"] == FINISH_STAGE_SCHEMA
    assert finish_stage["parameters"] == {
        "type": "object",
        "properties": {"result_names": {"type": "array", "items": {"type": "string"}}},
        "required": ["result_names"],
    }
