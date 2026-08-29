from unittest.mock import MagicMock

from salome_agent.specialists.geom import run_geom_specialist


def test_finish_stage_returns_call_log_and_result_names():
    llm = MagicMock()
    llm.chat.side_effect = [
        {"tool_call": "make_box", "args": {"dx": 10, "dy": 10, "dz": 10}},
        {"tool_call": "finish_stage", "args": {"result_names": ["box_1"]}},
    ]
    client = MagicMock()
    client.call_tool.return_value = {"ok": True, "records": [{"fn": "MakeBoxDXDYDZ", "args": [],
                                                                "out_vars": ["box_1"], "publish": ["box_1"]}]}
    result = run_geom_specialist({"module": "geom", "intent": "box", "params": {}, "inputs": []},
                                  llm, client, iteration_cap=12)
    assert result["result_names"] == ["box_1"]
    assert len(result["call_log"]) == 1


def test_exhausted_loop_sets_last_error():
    llm = MagicMock()
    llm.chat.return_value = {"tool_call": "make_box", "args": {"dx": 1, "dy": 1, "dz": 1}}
    client = MagicMock()
    client.call_tool.return_value = {"ok": False, "error": "bad box"}
    result = run_geom_specialist({"module": "geom", "intent": "box", "params": {}, "inputs": []},
                                  llm, client, iteration_cap=2)
    assert result["last_error"]["message"] == "bad box"
