from unittest.mock import MagicMock

from salome_agent.specialists.geom import run_geom_specialist


def test_finish_stage_returns_call_log_and_result_names():
    llm = MagicMock()
    llm.chat.side_effect = [
        {"tool_call": "make_box", "args": {"dx": 10, "dy": 10, "dz": 10}},
        {"tool_call": "finish_stage", "args": {"result_names": ["box_1"]}},
    ]
    client = MagicMock()
    client.list_tools.return_value = {"tools": [{"name": "make_box"}]}
    client.call_tool.return_value = {"ok": True, "records": [{"fn": "MakeBoxDXDYDZ", "args": [],
                                                                "out_vars": ["box_1"], "publish": ["box_1"]}]}
    result = run_geom_specialist({"module": "geom", "intent": "box", "params": {}, "inputs": []},
                                  llm, client, iteration_cap=12)
    assert result["result_names"] == ["box_1"]
    assert len(result["call_log"]) == 1


def test_no_tool_call_is_nudged_then_loop_continues():
    llm = MagicMock()
    llm.chat.side_effect = [
        {"tool_call": None, "content": "let me think"},
        {"tool_call": "finish_stage", "args": {"result_names": ["box_1"]}},
    ]
    client = MagicMock()
    client.list_tools.return_value = {"tools": [{"name": "make_box"}]}
    result = run_geom_specialist({"module": "geom", "intent": "box", "params": {}, "inputs": []},
                                  llm, client, iteration_cap=5)
    assert result["result_names"] == ["box_1"]
    second = llm.chat.call_args_list[1].kwargs["messages"]
    assert any(isinstance(m.get("content"), str) and "did not call a tool" in m["content"]
               for m in second)


def test_upstream_shape_info_is_injected_into_the_opening_prompt():
    llm = MagicMock()
    llm.chat.return_value = {"tool_call": "finish_stage", "args": {"result_names": ["x"]}}
    client = MagicMock()
    client.list_tools.return_value = {"tools": [{"name": "make_box"}]}
    run_geom_specialist(
        {"module": "geom", "intent": "walls", "params": {}, "inputs": []},
        llm, client, iteration_cap=3,
        upstream={"fuse_1": {"bbox": [0, 100, 0, 100, 0, 20],
                             "size": [100, 100, 20], "centre": [50, 50, 10]}},
    )
    sent = llm.chat.call_args.kwargs["messages"][1]["content"]
    assert "fuse_1" in sent
    assert "centre [50, 50, 10]" in sent


def test_no_upstream_leaves_the_prompt_unchanged():
    llm = MagicMock()
    llm.chat.return_value = {"tool_call": "finish_stage", "args": {"result_names": ["x"]}}
    client = MagicMock()
    client.list_tools.return_value = {"tools": [{"name": "make_box"}]}
    run_geom_specialist({"module": "geom", "intent": "box", "params": {}, "inputs": []},
                        llm, client, iteration_cap=3)
    sent = llm.chat.call_args.kwargs["messages"][1]["content"]
    assert "already in the study" not in sent


def test_system_prompt_mandates_a_placement_self_check():
    from salome_agent.specialists.geom import GEOM_SYSTEM_PROMPT
    p = GEOM_SYSTEM_PROMPT.lower()
    assert "measure" in p
    assert "fuse" in p and "translate" in p


def test_exhausted_loop_sets_last_error():
    llm = MagicMock()
    llm.chat.return_value = {"tool_call": "make_box", "args": {"dx": 1, "dy": 1, "dz": 1}}
    client = MagicMock()
    client.list_tools.return_value = {"tools": [{"name": "make_box"}]}
    client.call_tool.return_value = {"ok": False, "error": "bad box"}
    result = run_geom_specialist({"module": "geom", "intent": "box", "params": {}, "inputs": []},
                                  llm, client, iteration_cap=2)
    assert result["last_error"]["message"] == "bad box"


def test_geom_specialist_returns_fatal_on_session_poisoned():
    from salome_agent.executor_client import SessionPoisoned
    llm = MagicMock()
    client = MagicMock()
    client.list_tools.side_effect = SessionPoisoned("dead")
    out = run_geom_specialist({"module": "geom", "intent": "x", "params": {}, "inputs": []},
                              llm, client, iteration_cap=5)
    assert "fatal" in out
