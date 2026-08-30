from unittest.mock import MagicMock

from salome_agent.specialists.base import run_specialist


def test_emit_script_ends_loop_with_code():
    llm = MagicMock()
    llm.chat.return_value = {"tool_call": "emit_script", "args": {"code": "mesh.Compute()"}}
    result = run_specialist({"module": "smesh", "intent": "mesh it", "params": {}, "inputs": []},
                             llm, tools={}, iteration_cap=12)
    assert result["script"] == "mesh.Compute()"
    assert result["warning"] is None


def test_upstream_names_and_last_error_are_injected_into_the_prompt():
    llm = MagicMock()
    llm.chat.return_value = {"tool_call": "emit_script", "args": {"code": "x"}}
    run_specialist({"module": "smesh", "intent": "mesh it", "params": {}, "inputs": ["cube"]},
                   llm, tools={}, iteration_cap=12,
                   upstream_names=["box_1"], last_error={"message": "no study object named 'cube'"})
    sent = llm.chat.call_args.kwargs["messages"][0]["content"]
    assert "box_1" in sent
    assert "no study object named 'cube'" in sent


def test_iteration_cap_emits_best_so_far_with_warning():
    llm = MagicMock()
    llm.chat.return_value = {"tool_call": "api_introspect", "args": {"name": "smesh"}}
    result = run_specialist({"module": "smesh", "intent": "mesh it", "params": {}, "inputs": []},
                             llm, tools={"api_introspect": lambda **kw: "members: [...]"}, iteration_cap=2)
    assert result["warning"] is not None
