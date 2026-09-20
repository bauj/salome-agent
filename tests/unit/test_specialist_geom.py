from unittest.mock import MagicMock

from salome_agent.specialists.geom import run_geom_specialist


def _resp(*calls, content=None, raw_message=None):
    """Build a chat() return value. Each call is (name, args_dict); pass none
    for a prose-only turn."""
    return {"tool_calls": [{"name": n, "args": a, "id": None} for n, a in calls],
            "content": content, "raw_message": raw_message}


def test_finish_stage_returns_call_log_and_result_names():
    llm = MagicMock()
    llm.chat.side_effect = [
        _resp(("make_box", {"dx": 10, "dy": 10, "dz": 10})),
        _resp(("finish_stage", {"result_names": ["box_1"]})),
    ]
    client = MagicMock()
    client.list_tools.return_value = {"tools": [{"name": "make_box", "category": "primitive"}]}
    client.call_tool.return_value = {"ok": True, "records": [{"fn": "MakeBoxDXDYDZ", "args": [],
                                                                "out_vars": ["box_1"], "publish": ["box_1"]}]}
    result = run_geom_specialist({"module": "geom", "intent": "box", "params": {}, "inputs": []},
                                  llm, client, iteration_cap=12)
    assert result["result_names"] == ["box_1"]
    assert len(result["call_log"]) == 1


def test_finish_stage_drops_a_boolean_operand_kept_alongside_its_result():
    # model makes box_1 + box_2, fuses them to fuse_1, then names all three —
    # the next stage must only see fuse_1, not the folded-in operands (GH #9).
    llm = MagicMock()
    llm.chat.side_effect = [
        _resp(("make_box", {"dx": 10, "dy": 10, "dz": 10})),
        _resp(("make_box", {"dx": 5, "dy": 5, "dz": 5})),
        _resp(("fuse", {"shapes": ["box_1", "box_2"]})),
        _resp(("finish_stage", {"result_names": ["box_1", "box_2", "fuse_1"]})),
    ]
    client = MagicMock()
    client.list_tools.return_value = {"tools": [
        {"name": "make_box", "category": "primitive"},
        {"name": "fuse", "category": "boolean"}]}
    client.call_tool.side_effect = [
        {"ok": True, "records": [{"fn": "MakeBoxDXDYDZ", "args": [],
                                  "out_vars": ["box_1"], "publish": ["box_1"]}]},
        {"ok": True, "records": [{"fn": "MakeBoxDXDYDZ", "args": [],
                                  "out_vars": ["box_2"], "publish": ["box_2"]}]},
        {"ok": True, "records": [{"fn": "MakeFuseList",
                                  "args": [{"kind": "list", "value": [
                                      {"kind": "ref", "value": "box_1"},
                                      {"kind": "ref", "value": "box_2"}]}],
                                  "out_vars": ["fuse_1"], "publish": ["fuse_1"]}]},
    ]
    result = run_geom_specialist({"module": "geom", "intent": "fused", "params": {}, "inputs": []},
                                 llm, client, iteration_cap=12, verify_cadence="none")
    assert result["result_names"] == ["fuse_1"]


def test_finish_stage_keeps_a_transform_source_that_is_also_a_deliverable():
    # box_1 is the left cube AND translate's source for the right cube. Copy
    # semantics -> box_1 is still a real deliverable, must not be dropped.
    llm = MagicMock()
    llm.chat.side_effect = [
        _resp(("make_box", {"dx": 10, "dy": 10, "dz": 10})),
        _resp(("translate", {"shape": "box_1", "dx": 50, "dy": 0, "dz": 0})),
        _resp(("finish_stage", {"result_names": ["box_1", "tr_1"]})),
    ]
    client = MagicMock()
    client.list_tools.return_value = {"tools": [
        {"name": "make_box", "category": "primitive"},
        {"name": "translate", "category": "transform"}]}
    client.call_tool.side_effect = [
        {"ok": True, "records": [{"fn": "MakeBoxDXDYDZ", "args": [],
                                  "out_vars": ["box_1"], "publish": ["box_1"]}]},
        {"ok": True, "records": [{"fn": "MakeTranslation",
                                  "args": [{"kind": "ref", "value": "box_1"},
                                           {"kind": "literal", "value": 50}],
                                  "out_vars": ["tr_1"], "publish": ["tr_1"]}]},
    ]
    result = run_geom_specialist({"module": "geom", "intent": "two boxes", "params": {}, "inputs": []},
                                 llm, client, iteration_cap=12, verify_cadence="none")
    assert result["result_names"] == ["box_1", "tr_1"]


def test_finish_stage_keeps_independent_parts():
    llm = MagicMock()
    llm.chat.side_effect = [
        _resp(("make_box", {"dx": 10, "dy": 10, "dz": 10})),
        _resp(("make_box", {"dx": 5, "dy": 5, "dz": 5})),
        _resp(("finish_stage", {"result_names": ["box_1", "box_2"]})),
    ]
    client = MagicMock()
    client.list_tools.return_value = {"tools": [{"name": "make_box", "category": "primitive"}]}
    client.call_tool.side_effect = [
        {"ok": True, "records": [{"fn": "MakeBoxDXDYDZ", "args": [],
                                  "out_vars": ["box_1"], "publish": ["box_1"]}]},
        {"ok": True, "records": [{"fn": "MakeBoxDXDYDZ", "args": [],
                                  "out_vars": ["box_2"], "publish": ["box_2"]}]},
    ]
    result = run_geom_specialist({"module": "geom", "intent": "two boxes", "params": {}, "inputs": []},
                                 llm, client, iteration_cap=12, verify_cadence="none")
    assert result["result_names"] == ["box_1", "box_2"]


def test_no_tool_call_is_nudged_then_loop_continues():
    llm = MagicMock()
    llm.chat.side_effect = [
        _resp(content="let me think"),
        _resp(("finish_stage", {"result_names": ["box_1"]})),
    ]
    client = MagicMock()
    client.list_tools.return_value = {"tools": [{"name": "make_box", "category": "primitive"}]}
    result = run_geom_specialist({"module": "geom", "intent": "box", "params": {}, "inputs": []},
                                  llm, client, iteration_cap=5)
    assert result["result_names"] == ["box_1"]
    second = llm.chat.call_args_list[1].kwargs["messages"]
    assert any(isinstance(m.get("content"), str) and "did not call a tool" in m["content"]
               for m in second)


def test_all_tool_results_are_compacted_including_the_newest():
    llm = MagicMock()
    llm.chat.side_effect = [
        _resp(("make_box", {"dx": 10, "dy": 10, "dz": 10})),
        _resp(("make_box", {"dx": 5, "dy": 5, "dz": 5})),
        _resp(("finish_stage", {"result_names": ["box_2"]})),
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
    # 3rd chat call: every prior tool result is compacted, newest included.
    sent = llm.chat.call_args_list[2].kwargs["messages"]
    tool_msgs = [m["content"] for m in sent if m.get("role") == "tool"]
    assert len(tool_msgs) == 2
    for tm in tool_msgs:
        assert "MakeBoxDXDYDZ" not in tm            # verbose records dropped
    assert "box_1" in tool_msgs[0]                  # published name still there
    assert "'centre': [5, 5, 5]" in tool_msgs[0]    # shape_info kept
    assert "box_2" in tool_msgs[1]
    assert "'centre': [2.5, 2.5, 2.5]" in tool_msgs[1]


def test_message_prefix_is_byte_stable_across_turns():
    # The transcript sent each turn must extend the previous one verbatim (no
    # per-turn mutation of earlier messages) so a gateway's prefix/KV cache
    # survives. Only the appended tail and a widening tool list may differ.
    llm = MagicMock()
    llm.chat.side_effect = [
        _resp(("make_box", {"dx": 10, "dy": 10, "dz": 10})),
        _resp(("make_box", {"dx": 5, "dy": 5, "dz": 5})),
        _resp(("finish_stage", {"result_names": ["box_2"]})),
    ]
    client = MagicMock()
    client.list_tools.return_value = {"tools": [{"name": "make_box", "category": "primitive"}]}
    client.call_tool.side_effect = [
        {"ok": True,
         "records": [{"fn": "MakeBoxDXDYDZ", "args": [], "out_vars": ["box_1"],
                      "publish": ["box_1"]}],
         "sanity": {"ok": True}, "result_shape": "box_1",
         "shape_info": {"bbox": [0, 10, 0, 10, 0, 10], "size": [10, 10, 10],
                        "centre": [5, 5, 5]}},
        {"ok": True,
         "records": [{"fn": "MakeBoxDXDYDZ", "args": [], "out_vars": ["box_2"],
                      "publish": ["box_2"]}],
         "sanity": {"ok": True}, "result_shape": "box_2",
         "shape_info": {"bbox": [0, 5, 0, 5, 0, 5], "size": [5, 5, 5],
                        "centre": [2.5, 2.5, 2.5]}},
    ]
    run_geom_specialist({"module": "geom", "intent": "two boxes", "params": {}, "inputs": []},
                        llm, client, iteration_cap=6, verify_cadence="none")
    sent = [c.kwargs["messages"] for c in llm.chat.call_args_list]
    for prev, cur in zip(sent, sent[1:]):
        assert cur[:len(prev)] == prev


def test_upstream_shape_info_is_injected_into_the_opening_prompt():
    llm = MagicMock()
    llm.chat.return_value = _resp(("finish_stage", {"result_names": ["x"]}))
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
    llm.chat.return_value = _resp(("finish_stage", {"result_names": ["x"]}))
    client = MagicMock()
    client.list_tools.return_value = {"tools": [{"name": "make_box", "category": "primitive"}]}
    run_geom_specialist({"module": "geom", "intent": "box", "params": {}, "inputs": []},
                        llm, client, iteration_cap=3)
    sent = llm.chat.call_args.kwargs["messages"][1]["content"]
    assert "already in the study" not in sent


def test_retry_context_is_injected_into_the_opening_prompt():
    llm = MagicMock()
    llm.chat.return_value = _resp(("finish_stage", {"result_names": ["box_1"]}))
    client = MagicMock()
    client.list_tools.return_value = {"tools": [{"name": "make_box", "category": "primitive"}]}
    run_geom_specialist(
        {"module": "geom", "intent": "box", "params": {}, "inputs": []},
        llm, client, iteration_cap=3,
        retry={"last_error": {"message": "MakeBoxDXDYDZ failed"},
               "hint": "make it smaller", "built": ["box_1"],
               "built_info": {"box_1": {"bbox": [0, 10, 0, 10, 0, 10],
                                        "size": [10, 10, 10], "centre": [5, 5, 5]}},
               "call_log": []},
    )
    sent = llm.chat.call_args.kwargs["messages"][1]["content"]
    assert "MakeBoxDXDYDZ failed" in sent
    assert "make it smaller" in sent
    assert "box_1" in sent
    assert "centre [5, 5, 5]" in sent


def test_no_retry_leaves_the_prompt_unchanged():
    llm = MagicMock()
    llm.chat.return_value = _resp(("finish_stage", {"result_names": ["x"]}))
    client = MagicMock()
    client.list_tools.return_value = {"tools": [{"name": "make_box", "category": "primitive"}]}
    run_geom_specialist({"module": "geom", "intent": "box", "params": {}, "inputs": []},
                        llm, client, iteration_cap=3)
    sent = llm.chat.call_args.kwargs["messages"][1]["content"]
    assert "previous attempt" not in sent


def test_seed_call_log_keeps_reused_shapes_reachable():
    # retry's opening prompt tells the model box_1 already exists; the model
    # names it in finish_stage without rebuilding — the seeded call_log record
    # must still be there so codegen can trace it (not silently dropped).
    llm = MagicMock()
    llm.chat.return_value = _resp(("finish_stage", {"result_names": ["box_1"]}))
    client = MagicMock()
    client.list_tools.return_value = {"tools": [{"name": "make_box", "category": "primitive"}]}
    seed = [{"fn": "MakeBoxDXDYDZ", "args": [], "out_vars": ["box_1"], "publish": ["box_1"]}]
    result = run_geom_specialist(
        {"module": "geom", "intent": "box", "params": {}, "inputs": []},
        llm, client, iteration_cap=3, seed_call_log=seed,
    )
    assert client.call_tool.call_count == 0
    assert result["result_names"] == ["box_1"]
    assert result["call_log"] == seed


def test_system_prompt_mandates_a_placement_self_check():
    from salome_agent.specialists.geom import GEOM_SYSTEM_PROMPT
    p = GEOM_SYSTEM_PROMPT.lower()
    assert "measure" in p
    assert "before finish_stage" in p          # default cadence = "final"


def test_verify_hint_has_all_three_cadences():
    from salome_agent.specialists.geom import _VERIFY_HINT, _geom_prompt
    assert set(_VERIFY_HINT) == {"none", "final", "strict"}
    assert _VERIFY_HINT["none"] == ""
    assert "measure" not in _geom_prompt("none")
    assert "before finish_stage" in _geom_prompt("final")
    assert "before continuing" in _geom_prompt("strict")


def test_system_prompt_caps_reasoning_output():
    from salome_agent.specialists.geom import GEOM_SYSTEM_PROMPT
    assert "at most 2-3 sentences" in GEOM_SYSTEM_PROMPT


def test_exhausted_loop_sets_last_error():
    llm = MagicMock()
    llm.chat.return_value = _resp(("make_box", {"dx": 1, "dy": 1, "dz": 1}))
    client = MagicMock()
    client.list_tools.return_value = {"tools": [{"name": "make_box", "category": "primitive"}]}
    client.call_tool.return_value = {"ok": False, "error": "bad box"}
    result = run_geom_specialist({"module": "geom", "intent": "box", "params": {}, "inputs": []},
                                  llm, client, iteration_cap=2)
    assert result["last_error"]["message"] == "bad box"


def test_batched_calls_all_run_in_one_turn():
    llm = MagicMock()
    llm.chat.side_effect = [
        _resp(("make_box", {"dx": 1, "dy": 1, "dz": 1}),
              ("make_box", {"dx": 2, "dy": 2, "dz": 2}),
              ("translate", {"shape": "box_2", "dx": 5})),
        _resp(("finish_stage", {"result_names": ["tr_1"]})),
    ]
    client = MagicMock()
    client.list_tools.return_value = {"tools": [
        {"name": "make_box", "category": "primitive"},
        {"name": "translate", "category": "transform"}]}
    client.call_tool.return_value = {"ok": True, "records": [{"fn": "X", "args": [],
                                     "out_vars": ["o"], "publish": ["o"]}]}
    run_geom_specialist({"module": "geom", "intent": "x", "params": {}, "inputs": [],
                         "tools": ["make_box", "translate"]}, llm, client, iteration_cap=5,
                        verify_cadence="none")
    assert client.call_tool.call_count == 3
    # one assistant turn carrying all three call ids, then one tool msg each
    sent = llm.chat.call_args_list[1].kwargs["messages"]
    asst = [m for m in sent if m.get("role") == "assistant" and m.get("tool_calls")]
    assert len(asst) == 1 and len(asst[0]["tool_calls"]) == 3
    ids = [tc["id"] for tc in asst[0]["tool_calls"]]
    tool_ids = [m["tool_call_id"] for m in sent if m.get("role") == "tool"]
    assert tool_ids == ids


def test_provider_continuity_field_is_echoed_into_history_reasoning_dropped():
    # Gemini 3 hangs thought_signature on the individual tool_call, not the message.
    raw = {"role": "assistant", "content": None,
           "reasoning_content": "long private thoughts",
           "tool_calls": [{"id": None,
                           "extra_content": {"google": {"thought_signature": "SIG"}},
                           "function": {"name": "make_box", "arguments": "{}"}}]}
    llm = MagicMock()
    llm.chat.side_effect = [
        _resp(("make_box", {"dx": 1, "dy": 1, "dz": 1}), raw_message=raw),
        _resp(("finish_stage", {"result_names": ["box_1"]})),
    ]
    client = MagicMock()
    client.list_tools.return_value = {"tools": [{"name": "make_box", "category": "primitive"}]}
    client.call_tool.return_value = {"ok": True, "records": [{"fn": "MakeBoxDXDYDZ", "args": [],
                                     "out_vars": ["box_1"], "publish": ["box_1"]}]}
    run_geom_specialist({"module": "geom", "intent": "box", "params": {}, "inputs": []},
                        llm, client, iteration_cap=5, verify_cadence="none")
    sent = llm.chat.call_args_list[1].kwargs["messages"]
    asst = next(m for m in sent if m.get("role") == "assistant" and m.get("tool_calls"))
    assert asst["tool_calls"][0]["extra_content"] == {"google": {"thought_signature": "SIG"}}
    assert "reasoning_content" not in asst


def test_batch_ending_in_finish_stage_runs_earlier_calls_then_finishes():
    llm = MagicMock()
    llm.chat.return_value = _resp(("make_box", {"dx": 1, "dy": 1, "dz": 1}),
                                  ("finish_stage", {"result_names": ["box_1"]}))
    client = MagicMock()
    client.list_tools.return_value = {"tools": [{"name": "make_box", "category": "primitive"}]}
    client.call_tool.return_value = {"ok": True, "records": [{"fn": "MakeBoxDXDYDZ", "args": [],
                                     "out_vars": ["box_1"], "publish": ["box_1"]}]}
    result = run_geom_specialist({"module": "geom", "intent": "box", "params": {}, "inputs": []},
                                  llm, client, iteration_cap=5)
    assert client.call_tool.call_count == 1
    assert result["result_names"] == ["box_1"]
    assert len(result["call_log"]) == 1


def test_error_mid_batch_still_runs_the_rest_and_sets_last_error():
    llm = MagicMock()
    llm.chat.return_value = _resp(("make_box", {"dx": 1, "dy": 1, "dz": 1}),
                                  ("make_box", {"dx": 2, "dy": 2, "dz": 2}),
                                  ("make_box", {"dx": 3, "dy": 3, "dz": 3}))
    client = MagicMock()
    client.list_tools.return_value = {"tools": [{"name": "make_box", "category": "primitive"}]}
    client.call_tool.side_effect = [
        {"ok": True, "records": []},
        {"ok": False, "error": "bad middle box"},
        {"ok": True, "records": []},
    ] * 3
    result = run_geom_specialist({"module": "geom", "intent": "box", "params": {}, "inputs": []},
                                  llm, client, iteration_cap=1)
    assert client.call_tool.call_count == 3
    assert result["last_error"]["message"] == "bad middle box"


def _walls_client():
    client = MagicMock()
    client.list_tools.return_value = {"tools": [
        {"name": "make_box", "category": "primitive"},
        {"name": "translate", "category": "transform"},
        {"name": "measure", "category": "query"}]}
    client.call_tool.return_value = {"ok": True, "records": [{"fn": "X", "args": [],
                                     "out_vars": ["o"], "publish": ["o"]}]}
    return client


def test_final_cadence_nudges_a_measure_before_finish_then_accepts():
    llm = MagicMock()
    llm.chat.side_effect = [
        _resp(("translate", {"shape": "box_1", "dx": 5})),
        _resp(("finish_stage", {"result_names": ["tr_1"]})),
        _resp(("measure", {"names": ["tr_1"]})),
        _resp(("finish_stage", {"result_names": ["tr_1"]})),
    ]
    result = run_geom_specialist({"module": "geom", "intent": "x", "params": {}, "inputs": [],
                                  "tools": ["make_box", "translate"]},
                                 llm, _walls_client(), iteration_cap=8, verify_cadence="final")
    assert result["result_names"] == ["tr_1"]
    # exactly one nudge, delivered as the tool response to the first finish_stage
    nudges = [m for m in llm.chat.call_args_list[-1].kwargs["messages"]
              if m.get("role") == "tool" and "since your last measure" in str(m.get("content"))]
    assert len(nudges) == 1


def test_final_cadence_does_not_nudge_when_measure_was_called():
    llm = MagicMock()
    llm.chat.side_effect = [
        _resp(("translate", {"shape": "box_1", "dx": 5})),
        _resp(("measure", {"names": ["tr_1"]})),
        _resp(("finish_stage", {"result_names": ["tr_1"]})),
    ]
    result = run_geom_specialist({"module": "geom", "intent": "x", "params": {}, "inputs": [],
                                  "tools": ["make_box", "translate"]},
                                 llm, _walls_client(), iteration_cap=8, verify_cadence="final")
    assert result["result_names"] == ["tr_1"]
    assert llm.chat.call_count == 3  # no extra nudge round


def test_final_cadence_does_not_nudge_a_primitive_only_stage():
    llm = MagicMock()
    llm.chat.side_effect = [
        _resp(("make_box", {"dx": 1, "dy": 1, "dz": 1})),
        _resp(("finish_stage", {"result_names": ["box_1"]})),
    ]
    result = run_geom_specialist({"module": "geom", "intent": "box", "params": {}, "inputs": []},
                                 llm, _walls_client(), iteration_cap=8, verify_cadence="final")
    assert result["result_names"] == ["box_1"]
    assert llm.chat.call_count == 2


def test_strict_cadence_nudges_after_three_risky_ops():
    llm = MagicMock()
    llm.chat.side_effect = [
        _resp(("translate", {"shape": "a", "dx": 1}),
              ("translate", {"shape": "b", "dx": 2}),
              ("translate", {"shape": "c", "dx": 3})),
        _resp(("measure", {"names": ["a"]})),
        _resp(("finish_stage", {"result_names": ["a"]})),
    ]
    run_geom_specialist({"module": "geom", "intent": "x", "params": {}, "inputs": [],
                         "tools": ["translate"]},
                        llm, _walls_client(), iteration_cap=8, verify_cadence="strict")
    second = llm.chat.call_args_list[1].kwargs["messages"]
    assert any(m.get("role") == "user" and "since your last measure" in str(m.get("content"))
               for m in second)


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
    llm.chat.return_value = _resp(("finish_stage", {"result_names": ["x"]}))
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
        _resp(("load_tools", {"categories": ["swept"]})),
        _resp(("finish_stage", {"result_names": ["x"]})),
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
    llm.chat.return_value = _resp(("finish_stage", {"result_names": ["x"]}))
    client = MagicMock()
    client.list_tools.return_value = {"tools": _defs()}
    run_geom_specialist({"module": "geom", "intent": "x", "params": {}, "inputs": [],
                         "tools": ["make_box"]},
                        llm, client, iteration_cap=3, tool_selection="all")
    offered = _tool_names(llm.chat.call_args)
    assert "make_thick_solid" in offered
    assert "make_cylinder" in offered


def test_compact_result_keeps_query_payload_keys():
    from salome_agent.specialists.geom import _compact_result

    full = {"ok": True, "records": [{"fn": "x", "args": [], "out_vars": [], "publish": None}],
            "sanity": {"ok": True},
            "ids": [3, 7, 11, 15], "centres": [[0, 0, 20], [10, 0, 20]],
            "distance": 4.2, "properties": {"length": 40.0, "area": 0.0, "volume": 0.0}}
    compact = _compact_result(full)
    assert compact["ids"] == [3, 7, 11, 15]
    assert compact["centres"] == [[0, 0, 20], [10, 0, 20]]
    assert compact["distance"] == 4.2
    assert compact["properties"] == {"length": 40.0, "area": 0.0, "volume": 0.0}
    assert "records" not in compact


def test_prompt_documents_the_query_then_act_pattern():
    from salome_agent.specialists.geom import _geom_prompt
    for cadence in ("none", "final", "strict"):
        p = _geom_prompt(cadence)
        assert "get_subshapes" in p
        assert "fillet_edges" in p


def test_repeated_failing_call_is_answered_locally_without_a_round_trip():
    # weak models loop on a byte-identical failing call; the second attempt is
    # answered from the stage-local failure cache, no server round-trip (task 215).
    llm = MagicMock()
    bad = ("make_box_two_pnt", {"point1": "(0,0,0)", "point2": "(1,1,1)"})
    llm.chat.side_effect = [
        _resp(bad),
        _resp(bad),
        _resp(("finish_stage", {"result_names": []})),
    ]
    client = MagicMock()
    client.list_tools.return_value = {"tools": [{"name": "make_box_two_pnt",
                                                 "category": "primitive"}]}
    client.call_tool.return_value = {"ok": False, "error": "'(0,0,0)'"}

    result = run_geom_specialist({"module": "geom", "intent": "x", "params": {}, "inputs": []},
                                 llm, client, iteration_cap=12, verify_cadence="none")

    assert client.call_tool.call_count == 1
    assert result["result_names"] == []
    # the model was handed a repeat notice on the second turn
    sent = [m for call in llm.chat.call_args_list for m in call.kwargs["messages"]]
    assert any("identical call already failed" in str(m.get("content")) for m in sent)


def test_a_different_call_after_a_failure_still_runs():
    llm = MagicMock()
    llm.chat.side_effect = [
        _resp(("make_box_two_pnt", {"point1": "(0,0,0)", "point2": "(1,1,1)"})),
        _resp(("make_box", {"dx": 10, "dy": 10, "dz": 10})),
        _resp(("finish_stage", {"result_names": ["box_1"]})),
    ]
    client = MagicMock()
    client.list_tools.return_value = {"tools": [
        {"name": "make_box_two_pnt", "category": "primitive"},
        {"name": "make_box", "category": "primitive"}]}
    client.call_tool.side_effect = [
        {"ok": False, "error": "bad point"},
        {"ok": True, "records": [{"fn": "MakeBoxDXDYDZ", "args": [],
                                  "out_vars": ["box_1"], "publish": ["box_1"]}]},
    ]
    result = run_geom_specialist({"module": "geom", "intent": "x", "params": {}, "inputs": []},
                                 llm, client, iteration_cap=12, verify_cadence="none")
    assert client.call_tool.call_count == 2
    assert result["result_names"] == ["box_1"]


def test_compact_result_keeps_a_delta_note():
    from salome_agent.specialists.geom import _compact_result
    compact = _compact_result({"ok": True, "records": [], "delta": "cut: removed nothing"})
    assert compact["delta"] == "cut: removed nothing"


def test_image_description_is_injected_as_text_never_as_images():
    llm = MagicMock()
    llm.chat.side_effect = [_resp(("finish_stage", {"result_names": []}))]
    client = MagicMock()
    client.list_tools.return_value = {"tools": []}
    run_geom_specialist({"module": "geom", "intent": "box", "params": {}, "inputs": []},
                        llm, client, iteration_cap=3,
                        image_description="Summary: an L-bracket")
    user = llm.chat.call_args.kwargs["messages"][1]["content"]
    assert isinstance(user, str)
    assert "Summary: an L-bracket" in user
