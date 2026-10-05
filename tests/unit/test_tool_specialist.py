from unittest.mock import MagicMock

from salome_agent.modules import ModuleSpec, Terminator
from salome_agent.specialists.tool_specialist import ToolSpecialist


def _resp(*calls, content=None, raw_message=None):
    return {"tool_calls": [{"name": n, "args": a, "id": None} for n, a in calls],
            "content": content, "raw_message": raw_message}


def _spec(**over):
    base = dict(
        name="toy",
        prompt=lambda cadence: "SYSTEM",
        terminator=Terminator(name="done", schema={"type": "object", "properties": {}},
                              description="finish",
                              handle=lambda args, call_log: {"out": args, "call_log": call_log}),
        execute=lambda client, name, args: client.call_tool(name, args),
        render_result=lambda r: str(r),
        tools_for=lambda client: {"work": {"name": "work", "parameters": {}, "description": "w"}},
        no_tool_nudge="Call a tool.",
        on_exhausted=lambda call_log, last_error: {"exhausted": True, "last_error": last_error},
    )
    base.update(over)
    return ModuleSpec(**base)


def test_terminator_name_and_handle_come_from_the_spec():
    llm = MagicMock()
    llm.chat.side_effect = [_resp(("done", {"x": 1}))]
    result = ToolSpecialist(_spec()).run({"module": "toy"}, llm, MagicMock(), iteration_cap=3)
    assert result == {"out": {"x": 1}, "call_log": [], "calls": []}


def test_execute_comes_from_the_spec():
    seen = []
    llm = MagicMock()
    llm.chat.side_effect = [_resp(("work", {"a": 1})), _resp(("done", {}))]

    def execute(client, name, args):
        seen.append((name, args))
        return {"ok": True, "records": [{"fn": "F", "args": [], "out_vars": ["v"], "publish": ["v"]}]}

    result = ToolSpecialist(_spec(execute=execute)).run(
        {"module": "toy"}, llm, MagicMock(), iteration_cap=3)
    assert seen == [("work", {"a": 1})]
    assert result["call_log"] == [{"fn": "F", "args": [], "out_vars": ["v"], "publish": ["v"]}]


def test_blocks_are_joined_after_the_stage_plan():
    llm = MagicMock()
    llm.chat.side_effect = [_resp(("done", {}))]
    ToolSpecialist(_spec()).run({"module": "toy"}, llm, MagicMock(), iteration_cap=3,
                                blocks=("FIRST BLOCK", "SECOND BLOCK"))
    user = llm.chat.call_args.kwargs["messages"][1]["content"]
    assert user.endswith("FIRST BLOCK\n\nSECOND BLOCK")
    assert user.startswith("{'module': 'toy'}")


def test_falsy_blocks_are_skipped():
    llm = MagicMock()
    llm.chat.side_effect = [_resp(("done", {}))]
    ToolSpecialist(_spec()).run({"module": "toy"}, llm, MagicMock(), iteration_cap=3,
                                blocks=(None, "", "ONLY"))
    user = llm.chat.call_args.kwargs["messages"][1]["content"]
    assert user.count("\n\n") == 1


def test_system_prompt_comes_from_the_spec_with_the_cadence():
    seen = []
    llm = MagicMock()
    llm.chat.side_effect = [_resp(("done", {}))]
    ToolSpecialist(_spec(prompt=lambda c: seen.append(c) or f"P:{c}")).run(
        {"module": "toy"}, llm, MagicMock(), iteration_cap=3, verify_cadence="strict")
    assert seen == ["strict"]
    assert llm.chat.call_args.kwargs["messages"][0]["content"] == "P:strict"


def test_empty_categories_means_no_load_tools_tool_is_offered():
    llm = MagicMock()
    llm.chat.side_effect = [_resp(("done", {}))]
    ToolSpecialist(_spec()).run({"module": "toy"}, llm, MagicMock(), iteration_cap=3)
    offered = [t["name"] for t in llm.chat.call_args.kwargs["tools"]]
    assert offered == ["work", "done"]


def test_empty_risky_categories_never_nudges_a_measure():
    llm = MagicMock()
    llm.chat.side_effect = [_resp(("work", {})), _resp(("done", {}))]
    client = MagicMock()
    client.call_tool.return_value = {"ok": True, "records": []}
    result = ToolSpecialist(_spec()).run({"module": "toy"}, llm, client,
                                         iteration_cap=3, verify_cadence="strict")
    assert "out" in result
    assert llm.chat.call_count == 2


def test_no_tool_call_uses_the_spec_nudge():
    llm = MagicMock()
    llm.chat.side_effect = [_resp(content="prose"), _resp(("done", {}))]
    ToolSpecialist(_spec()).run({"module": "toy"}, llm, MagicMock(), iteration_cap=3)
    second = llm.chat.call_args_list[1].kwargs["messages"]
    assert second[-1]["content"] == "Call a tool."


def test_iteration_cap_uses_the_spec_on_exhausted():
    llm = MagicMock()
    llm.chat.return_value = _resp(("work", {}))
    client = MagicMock()
    client.call_tool.return_value = {"ok": False, "error": "boom"}
    result = ToolSpecialist(_spec()).run({"module": "toy"}, llm, client, iteration_cap=2)
    assert result["exhausted"] is True
    assert result["last_error"]["message"] == "boom"


def test_session_died_becomes_a_fatal_result():
    from salome_agent.executor_client import SessionDied

    llm = MagicMock()
    llm.chat.side_effect = [_resp(("work", {}))]

    def execute(client, name, args):
        raise SessionDied("socket closed")

    result = ToolSpecialist(_spec(execute=execute)).run(
        {"module": "toy"}, llm, MagicMock(), iteration_cap=3)
    assert result["fatal"].startswith("SALOME agent server unavailable")


def test_a_session_module_with_no_server_ends_fatal_before_any_llm_call():
    llm = MagicMock()
    result = ToolSpecialist(_spec(needs_session=True)).run(
        {"module": "toy"}, llm, None, iteration_cap=3)
    assert "toy stage needs a SALOME agent server" in result["fatal"]
    llm.chat.assert_not_called()


def test_geom_needs_a_session_and_script_modules_do_not():
    from salome_agent.modules import spec_for
    import salome_agent.specialists.geom  # noqa: F401
    import salome_agent.specialists.script_modules  # noqa: F401

    assert spec_for("geom").needs_session
    assert not spec_for("smesh").needs_session


def test_calls_trace_pairs_tool_and_args_with_records():
    llm = MagicMock()
    llm.chat.side_effect = [_resp(("work", {"a": 1}), ("work", {"a": 2})), _resp(("done", {}))]
    rec = {"fn": "F", "args": [], "out_vars": ["v"], "publish": ["v"]}

    def execute(client, name, args):
        return {"ok": True, "records": [rec] if args["a"] == 1 else []}

    result = ToolSpecialist(_spec(execute=execute)).run(
        {"module": "toy"}, llm, MagicMock(), iteration_cap=3)
    assert result["calls"] == [{"tool": "work", "tool_args": {"a": 1}, "records": [rec]}]


def test_seed_calls_come_first():
    llm = MagicMock()
    llm.chat.side_effect = [_resp(("done", {}))]
    seed = [{"tool": "old", "tool_args": {}, "records": []}]
    result = ToolSpecialist(_spec()).run({"module": "toy"}, llm, MagicMock(),
                                         iteration_cap=3, seed_calls=seed)
    assert result["calls"] == seed


def test_exhausted_result_carries_calls_when_it_carries_call_log():
    llm = MagicMock()
    llm.chat.side_effect = [_resp(("work", {"a": 1}))] * 2
    spec = _spec(execute=lambda c, n, a: {"ok": True, "records": [{"fn": "F", "args": [],
                                                                      "out_vars": [], "publish": None}]},
                 on_exhausted=lambda call_log, last_error: {"call_log": call_log})
    result = ToolSpecialist(spec).run({"module": "toy"}, llm, MagicMock(), iteration_cap=2)
    assert len(result["calls"]) == 2


def test_extra_terminator_ends_the_loop_with_its_own_handle():
    from salome_agent.modules import Terminator

    llm = MagicMock()
    llm.chat.side_effect = [_resp(("bail", {"why": "big"}))]
    bail = Terminator(name="bail", schema={"type": "object", "properties": {}},
                      description="b", handle=lambda args, call_log: {"bailed": args["why"]})
    result = ToolSpecialist(_spec(extra_terminators=(bail,))).run(
        {"module": "toy"}, llm, MagicMock(), iteration_cap=3)
    assert result == {"bailed": "big"}
    names = [t["name"] for t in llm.chat.call_args.kwargs["tools"]]
    assert names == ["work", "done", "bail"]


def test_cancel_stops_the_loop_before_the_next_tool_call():
    import threading
    cancel = threading.Event()
    seen = []
    llm = MagicMock()
    llm.chat.side_effect = [_resp(("work", {"a": 1}), ("work", {"a": 2})), _resp(("done", {}))]

    def execute(client, name, args):
        seen.append(args["a"])
        cancel.set()
        return {"ok": True, "records": [{"fn": "F", "args": [], "out_vars": ["v"], "publish": ["v"]}]}

    spec = _spec(execute=execute,
                 on_exhausted=lambda call_log, last_error: {"call_log": call_log})
    result = ToolSpecialist(spec).run(
        {"module": "toy"}, llm, MagicMock(), iteration_cap=5, cancel=cancel)
    assert seen == [1]
    assert llm.chat.call_count == 1
    assert len(result["call_log"]) == 1
    assert [c["tool_args"] for c in result["calls"]] == [{"a": 1}]


def test_cancel_already_set_skips_the_llm():
    import threading
    cancel = threading.Event()
    cancel.set()
    llm = MagicMock()
    result = ToolSpecialist(_spec()).run({"module": "toy"}, llm, MagicMock(),
                                         iteration_cap=5, cancel=cancel)
    assert llm.chat.call_count == 0
    assert result["exhausted"] is True


def test_llm_cancelled_mid_call_ends_the_stage_with_its_calls():
    from salome_agent.llm import LLMCancelled
    llm = MagicMock()
    llm.chat.side_effect = [_resp(("work", {"a": 1})), LLMCancelled()]
    spec = _spec(on_exhausted=lambda call_log, last_error: {"call_log": call_log,
                                                            "exhausted": True})
    result = ToolSpecialist(spec).run({"module": "toy"}, llm, MagicMock(), iteration_cap=5)
    assert result["exhausted"] is True
    assert [c["tool_args"] for c in result["calls"]] == [{"a": 1}]


def _in_place(recv):
    return {"ok": True, "records": [{"fn": "SetColor", "args": [], "out_vars": [],
                                     "publish": None, "emit": "method_call", "recv": recv}]}


def _tool_messages(llm):
    return [m["content"] for m in llm.chat.call_args.kwargs["messages"] if m["role"] == "tool"]


def test_in_place_change_to_an_upstream_shape_is_flagged():
    llm = MagicMock()
    llm.chat.side_effect = [_resp(("work", {})), _resp(("done", {}))]
    ToolSpecialist(_spec(execute=lambda c, n, a: _in_place("beam_1"))).run(
        {"module": "toy"}, llm, MagicMock(), iteration_cap=3, upstream=("beam_1",))
    [content] = _tool_messages(llm)
    assert "beam_1 was built in an earlier stage" in content


def test_in_place_change_to_this_stages_own_shape_is_silent():
    llm = MagicMock()
    llm.chat.side_effect = [_resp(("work", {})), _resp(("done", {}))]
    ToolSpecialist(_spec(execute=lambda c, n, a: _in_place("walls_1"))).run(
        {"module": "toy"}, llm, MagicMock(), iteration_cap=3, upstream=("beam_1",))
    [content] = _tool_messages(llm)
    assert "earlier stage" not in content


def test_copy_that_reads_an_upstream_shape_is_silent():
    llm = MagicMock()
    llm.chat.side_effect = [_resp(("work", {})), _resp(("done", {}))]
    copy = {"ok": True, "records": [{"fn": "MakeFuse",
                                     "args": [{"kind": "ref", "value": "beam_1"}],
                                     "out_vars": ["fuse_1"], "publish": ["fuse_1"]}]}
    ToolSpecialist(_spec(execute=lambda c, n, a: copy)).run(
        {"module": "toy"}, llm, MagicMock(), iteration_cap=3, upstream=("beam_1",))
    [content] = _tool_messages(llm)
    assert "earlier stage" not in content


def test_reasoning_overrun_is_nudged_once_then_the_turn_is_retried():
    from salome_agent.llm import ReasoningBudgetExceeded
    llm = MagicMock()
    llm.chat.side_effect = [ReasoningBudgetExceeded(21000, 20000), _resp(("done", {"x": 1}))]
    result = ToolSpecialist(_spec()).run({"module": "toy"}, llm, MagicMock(), iteration_cap=5)
    assert result["out"] == {"x": 1}
    nudge = llm.chat.call_args.kwargs["messages"][-1]
    assert nudge["role"] == "user"
    assert "21000" in nudge["content"] and "tool call now" in nudge["content"]


def test_second_overrun_in_a_row_ends_the_stage():
    from salome_agent.llm import ReasoningBudgetExceeded
    llm = MagicMock()
    llm.chat.side_effect = [ReasoningBudgetExceeded(21000, 20000),
                            ReasoningBudgetExceeded(22000, 20000)]
    result = ToolSpecialist(_spec()).run({"module": "toy"}, llm, MagicMock(), iteration_cap=5)
    assert result["exhausted"] is True
    assert "reasoning budget" in result["last_error"]
    assert llm.chat.call_count == 2


def test_overrun_after_an_acting_turn_is_nudged_again():
    from salome_agent.llm import ReasoningBudgetExceeded
    llm = MagicMock()
    llm.chat.side_effect = [ReasoningBudgetExceeded(21000, 20000), _resp(("work", {"a": 1})),
                            ReasoningBudgetExceeded(21000, 20000), _resp(("done", {}))]
    result = ToolSpecialist(_spec()).run({"module": "toy"}, llm, MagicMock(), iteration_cap=6)
    assert "out" in result


def test_loop_is_nudged_with_the_reasoning_before_it_never_the_loop():
    from salome_agent.llm import RepetitionLoopDetected
    llm = MagicMock()
    loop = RepetitionLoopDetected("reasoning", 2800, "The ring is two circles, R50 and R41.5.",
                                  "So the bars point upward.")
    llm.chat.side_effect = [loop, _resp(("done", {"x": 1}))]
    llm.loop_retries = 3
    result = ToolSpecialist(_spec()).run({"module": "toy"}, llm, MagicMock(), iteration_cap=5)
    assert result["out"] == {"x": 1}
    nudge = llm.chat.call_args.kwargs["messages"][-1]["content"]
    assert "R50 and R41.5" in nudge
    assert "So the bars point upward." in nudge and "repeating" in nudge
    assert "tool call now" in nudge


def test_loop_then_overrun_in_a_row_ends_the_stage():
    from salome_agent.llm import ReasoningBudgetExceeded, RepetitionLoopDetected
    llm = MagicMock()
    llm.chat.side_effect = [RepetitionLoopDetected("reasoning", 2800, "x", "y y y."),
                            ReasoningBudgetExceeded(22000, 20000)]
    llm.loop_retries = 3
    result = ToolSpecialist(_spec()).run({"module": "toy"}, llm, MagicMock(), iteration_cap=5)
    assert result["exhausted"] is True
    assert llm.chat.call_count == 2


def test_loops_are_re_asked_up_to_loop_retries_then_the_stage_ends():
    from salome_agent.llm import RepetitionLoopDetected
    llm = MagicMock()
    llm.loop_retries = 2
    llm.chat.side_effect = [RepetitionLoopDetected("reasoning", 2800, "x", "y y y.")] * 3
    result = ToolSpecialist(_spec()).run({"module": "toy"}, llm, MagicMock(), iteration_cap=9)
    assert result["exhausted"] is True
    assert "looped" in result["last_error"]
    assert llm.chat.call_count == 3


def test_loops_within_loop_retries_still_reach_the_answer():
    from salome_agent.llm import RepetitionLoopDetected
    llm = MagicMock()
    llm.loop_retries = 3
    llm.chat.side_effect = ([RepetitionLoopDetected("reasoning", 2800, "x", "y y y.")] * 3
                            + [_resp(("done", {"x": 1}))])
    result = ToolSpecialist(_spec()).run({"module": "toy"}, llm, MagicMock(), iteration_cap=9)
    assert result["out"] == {"x": 1}


def test_a_new_cut_nudge_replaces_the_previous_one():
    from salome_agent.llm import RepetitionLoopDetected
    llm = MagicMock()
    llm.loop_retries = 3
    llm.chat.side_effect = [RepetitionLoopDetected("reasoning", 2800, "x", f"sample {i} here.")
                            for i in range(3)] + [_resp(("done", {"x": 1}))]
    ToolSpecialist(_spec()).run({"module": "toy"}, llm, MagicMock(), iteration_cap=9)
    sent = llm.chat.call_args.kwargs["messages"]
    nudges = [m["content"] for m in sent if "You kept saying" in str(m.get("content"))]
    assert len(nudges) == 1 and "sample 2 here." in nudges[0]


def _paraphrase_loop():
    from pathlib import Path
    return (Path(__file__).parent.parent / "fixtures" / "repetition"
            / "loop_nemotron_paraphrase.txt").read_text()


def test_a_budget_cut_that_ends_in_a_loop_gets_the_loop_nudge():
    from salome_agent.llm import ReasoningBudgetExceeded
    text = _paraphrase_loop()
    llm = MagicMock()
    llm.loop_retries = 3
    llm.chat.side_effect = [ReasoningBudgetExceeded(20002, 20000, text), _resp(("done", {}))]
    ToolSpecialist(_spec()).run({"module": "toy"}, llm, MagicMock(), iteration_cap=5)
    nudge = llm.chat.call_args.kwargs["messages"][-1]["content"]
    assert "repeating" in nudge
    assert text[-300:] not in nudge


def test_a_budget_cut_then_loops_counts_against_loop_retries():
    from salome_agent.llm import ReasoningBudgetExceeded, RepetitionLoopDetected
    llm = MagicMock()
    llm.loop_retries = 3
    llm.chat.side_effect = [ReasoningBudgetExceeded(20002, 20000, _paraphrase_loop()),
                            RepetitionLoopDetected("reasoning", 2100, "x", "y y y."),
                            _resp(("done", {"x": 1}))]
    result = ToolSpecialist(_spec()).run({"module": "toy"}, llm, MagicMock(), iteration_cap=5)
    assert result["out"] == {"x": 1}


def test_prune_drops_records_owned_by_a_discarded_receiver():
    from salome_agent.specialists.tool_specialist import prune_records
    sketch = {"fn": "addSketch", "emit": "stmt", "code": "s = ...", "args": [],
              "out_vars": ["sketch_1"], "publish": ["sketch_1"]}
    line = {"fn": "addLine", "emit": "stmt", "code": "l = ...", "args": [],
            "out_vars": ["sketch_1_L1"], "publish": None, "recv": "sketch_1"}
    other = {"fn": "addSketch", "emit": "stmt", "code": "t = ...", "args": [],
             "out_vars": ["sketch_2"], "publish": ["sketch_2"]}
    calls = [{"tool": "create_sketch", "tool_args": {}, "records": [sketch]},
             {"tool": "sketch_add", "tool_args": {}, "records": [line]},
             {"tool": "create_sketch", "tool_args": {}, "records": [other]}]
    log, kept, removed = prune_records([sketch, line, other], calls, {"sketch_1"})
    assert log == [other]
    assert removed == [0, 1]


def test_gate_refuses_finish_on_every_call_until_it_clears():
    seen = []

    def gate(args, call_log, infos):
        seen.append(dict(infos))
        return None if infos.get("s", {}).get("open") == [] else "close s first"

    results = iter([{"ok": True, "records": [], "result_shape": "s",
                     "shape_info": {"open": ["x"]}},
                    {"ok": True, "records": [], "result_shape": "s",
                     "shape_info": {"open": []}}])
    client = MagicMock()
    client.call_tool.side_effect = lambda name, args: next(results)
    llm = MagicMock()
    llm.chat.side_effect = [_resp(("work", {})), _resp(("done", {})), _resp(("done", {})),
                            _resp(("work", {})), _resp(("done", {}))]
    result = ToolSpecialist(_spec(gate=gate)).run({"module": "toy"}, llm, client,
                                                  iteration_cap=6)
    assert result["out"] == {}
    assert len(seen) == 3
    refusals = [m["content"] for m in llm.chat.call_args.kwargs["messages"]
                if m["role"] == "tool" and m["content"] == "close s first"]
    assert len(refusals) == 2


def test_calc_is_offered_and_answered_locally_when_the_spec_asks():
    llm = MagicMock()
    client = MagicMock()
    llm.chat.side_effect = [_resp(("calc", {"code": "2 * 21"})), _resp(("done", {}))]
    ToolSpecialist(_spec(calc=True)).run({"module": "toy"}, llm, client, iteration_cap=3)
    first_tools = [t["name"] for t in llm.chat.call_args_list[0].kwargs["tools"]]
    assert "calc" in first_tools
    assert _tool_messages(llm) == ["42"]
    client.call_tool.assert_not_called()


def test_calc_not_offered_by_default():
    llm = MagicMock()
    llm.chat.side_effect = [_resp(("done", {}))]
    ToolSpecialist(_spec()).run({"module": "toy"}, llm, MagicMock(), iteration_cap=3)
    assert "calc" not in [t["name"] for t in llm.chat.call_args.kwargs["tools"]]


def test_turn_reasoning_is_kept_in_history_when_returned():
    llm = MagicMock()
    first = _resp(("work", {"a": 1}))
    first["reasoning"] = "corners first"
    llm.chat.side_effect = [first, _resp(("done", {}))]
    ToolSpecialist(_spec()).run({"module": "toy"}, llm, MagicMock(), iteration_cap=3)
    turn = [m for m in llm.chat.call_args.kwargs["messages"] if m["role"] == "assistant"][0]
    assert turn["reasoning_content"] == "corners first"


def test_overrun_nudge_carries_the_cut_reasoning_tail_and_names_calc():
    from salome_agent.llm import ReasoningBudgetExceeded
    llm = MagicMock()
    llm.chat.side_effect = [ReasoningBudgetExceeded(21000, 20000, "x" * 50_000 + "THE END"),
                            _resp(("done", {}))]
    ToolSpecialist(_spec(calc=True)).run({"module": "toy"}, llm, MagicMock(), iteration_cap=5)
    nudge = llm.chat.call_args.kwargs["messages"][-1]["content"]
    assert "THE END" in nudge and "earlier part omitted" in nudge
    assert "calc" in nudge
    assert len(nudge) < 42_000
