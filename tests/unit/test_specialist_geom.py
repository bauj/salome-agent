from unittest.mock import MagicMock

from salome_agent.specialists.geom import run_geom_specialist


def test_finish_stage_returns_call_log_and_result_names():
    llm = MagicMock()
    llm.chat.side_effect = [
        {"tool_call": "make_box", "args": {"dx": 10, "dy": 10, "dz": 10}},
        {"tool_call": "finish_stage", "args": {"result_names": ["box_1"]}},
    ]
    client = MagicMock()
    client.list_tools.return_value = {"tools": [{"name": "make_box", "category": "primitive"}]}
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
    client.list_tools.return_value = {"tools": [{"name": "make_box", "category": "primitive"}]}
    result = run_geom_specialist({"module": "geom", "intent": "box", "params": {}, "inputs": []},
                                  llm, client, iteration_cap=5)
    assert result["result_names"] == ["box_1"]
    second = llm.chat.call_args_list[1].kwargs["messages"]
    assert any(isinstance(m.get("content"), str) and "did not call a tool" in m["content"]
               for m in second)


def test_prior_tool_results_are_compacted_latest_kept_full():
    llm = MagicMock()
    llm.chat.side_effect = [
        {"tool_call": "make_box", "args": {"dx": 10, "dy": 10, "dz": 10}},
        {"tool_call": "make_box", "args": {"dx": 5, "dy": 5, "dz": 5}},
        {"tool_call": "finish_stage", "args": {"result_names": ["box_2"]}},
    ]
    client = MagicMock()
    client.list_tools.return_value = {"tools": [{"name": "make_box", "category": "primitive"}]}
    client.call_tool.side_effect = [
        {"ok": True,
         "records": [{"fn": "MakeBoxDXDYDZ", "args": [{"kind": "literal", "value": 10}],
                      "out_vars": ["box_1"], "publish": ["box_1"]}],
         "sanity": {"ok": True},
         "result_shape": "box_1",
         "shape_info": {"bbox": [0, 10, 0, 10, 0, 10], "size": [10, 10, 10], "centre": [5, 5, 5]}},
        {"ok": True,
         "records": [{"fn": "MakeBoxDXDYDZ", "args": [{"kind": "literal", "value": 5}],
                      "out_vars": ["box_2"], "publish": ["box_2"]}],
         "sanity": {"ok": True},
         "result_shape": "box_2",
         "shape_info": {"bbox": [0, 5, 0, 5, 0, 5], "size": [5, 5, 5], "centre": [2.5, 2.5, 2.5]}},
    ]
    run_geom_specialist({"module": "geom", "intent": "two boxes", "params": {}, "inputs": []},
                        llm, client, iteration_cap=6)
    # 3rd chat call: turn-1 tool result compacted, turn-2 (latest) full.
    sent = llm.chat.call_args_list[2].kwargs["messages"]
    tool_msgs = [m["content"] for m in sent if m.get("role") == "tool"]
    assert len(tool_msgs) == 2
    assert "MakeBoxDXDYDZ" not in tool_msgs[0]      # verbose records dropped
    assert "box_1" in tool_msgs[0]                  # published name still there
    assert "'centre': [5, 5, 5]" in tool_msgs[0]    # shape_info kept
    assert "MakeBoxDXDYDZ" in tool_msgs[1]          # latest turn untouched


def test_upstream_shape_info_is_injected_into_the_opening_prompt():
    llm = MagicMock()
    llm.chat.return_value = {"tool_call": "finish_stage", "args": {"result_names": ["x"]}}
    client = MagicMock()
    client.list_tools.return_value = {"tools": [{"name": "make_box", "category": "primitive"}]}
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
    client.list_tools.return_value = {"tools": [{"name": "make_box", "category": "primitive"}]}
    run_geom_specialist({"module": "geom", "intent": "box", "params": {}, "inputs": []},
                        llm, client, iteration_cap=3)
    sent = llm.chat.call_args.kwargs["messages"][1]["content"]
    assert "already in the study" not in sent


def test_system_prompt_mandates_a_placement_self_check():
    from salome_agent.specialists.geom import GEOM_SYSTEM_PROMPT
    p = GEOM_SYSTEM_PROMPT.lower()
    assert "measure" in p
    assert "fuse" in p and "translate" in p


def test_system_prompt_caps_reasoning_output():
    from salome_agent.specialists.geom import GEOM_SYSTEM_PROMPT
    assert "at most 2-3 sentences" in GEOM_SYSTEM_PROMPT


def test_exhausted_loop_sets_last_error():
    llm = MagicMock()
    llm.chat.return_value = {"tool_call": "make_box", "args": {"dx": 1, "dy": 1, "dz": 1}}
    client = MagicMock()
    client.list_tools.return_value = {"tools": [{"name": "make_box", "category": "primitive"}]}
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


def _defs():
    cats = {
        "make_box": "primitive", "make_cylinder": "primitive",
        "translate": "transform", "set_color": "display",
        "fuse": "boolean", "cut": "boolean", "measure": "query",
        "check_shape": "query", "list_study_objects": "query",
        "make_thick_solid": "swept",
    }
    return [{"name": n, "parameters": {"type": "object", "properties": {}},
             "description": "", "category": c} for n, c in cats.items()]


def _tool_names(chat_call):
    return {t["name"] for t in chat_call.kwargs["tools"]}


def test_hinted_stage_offers_core_plus_hints_plus_meta_tools_only():
    llm = MagicMock()
    llm.chat.return_value = {"tool_call": "finish_stage", "args": {"result_names": ["x"]}}
    client = MagicMock()
    client.list_tools.return_value = {"tools": _defs()}
    run_geom_specialist(
        {"module": "geom", "intent": "red box", "params": {}, "inputs": [],
         "tools": ["make_box", "set_color"]},
        llm, client, iteration_cap=3,
    )
    offered = _tool_names(llm.chat.call_args)
    assert offered == {
        "measure", "check_shape", "list_study_objects", "make_box", "fuse",
        "cut", "translate",
        "set_color",
        "finish_stage", "load_tools",
    }
    assert "make_thick_solid" not in offered


def test_load_tools_widens_the_next_call():
    llm = MagicMock()
    llm.chat.side_effect = [
        {"tool_call": "load_tools", "args": {"categories": ["swept"]}},
        {"tool_call": "finish_stage", "args": {"result_names": ["x"]}},
    ]
    client = MagicMock()
    client.list_tools.return_value = {"tools": _defs()}
    run_geom_specialist({"module": "geom", "intent": "hollow", "params": {}, "inputs": []},
                        llm, client, iteration_cap=5)
    first = _tool_names(llm.chat.call_args_list[0])
    second = _tool_names(llm.chat.call_args_list[1])
    assert "make_thick_solid" not in first
    assert "make_thick_solid" in second
    msgs = llm.chat.call_args_list[1].kwargs["messages"]
    assert any(m.get("role") == "tool" and "make_thick_solid" in str(m.get("content"))
               for m in msgs)


def test_tool_selection_all_offers_the_full_registry():
    llm = MagicMock()
    llm.chat.return_value = {"tool_call": "finish_stage", "args": {"result_names": ["x"]}}
    client = MagicMock()
    client.list_tools.return_value = {"tools": _defs()}
    run_geom_specialist({"module": "geom", "intent": "x", "params": {}, "inputs": [],
                         "tools": ["make_box"]},
                        llm, client, iteration_cap=3, tool_selection="all")
    offered = _tool_names(llm.chat.call_args)
    assert "make_thick_solid" in offered
    assert "make_cylinder" in offered
