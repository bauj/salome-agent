from unittest.mock import MagicMock

from salome_agent.specialists.base import run_specialist


def test_emit_script_ends_loop_with_code():
    llm = MagicMock()
    llm.chat.return_value = {"tool_call": "emit_script", "args": {"code": "mesh.Compute()"}}
    result = run_specialist({"module": "smesh", "intent": "mesh it", "params": {}, "inputs": []},
                             llm, tools={}, iteration_cap=12)
    assert result["script"] == "mesh.Compute()"
    assert result["warning"] is None


def test_iteration_cap_emits_best_so_far_with_warning():
    llm = MagicMock()
    llm.chat.return_value = {"tool_call": "api_introspect", "args": {"name": "smesh"}}
    result = run_specialist({"module": "smesh", "intent": "mesh it", "params": {}, "inputs": []},
                             llm, tools={"api_introspect": lambda **kw: "members: [...]"}, iteration_cap=2)
    assert result["warning"] is not None
