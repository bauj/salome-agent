from unittest.mock import MagicMock

import pytest
import requests
from langgraph.types import Command

from salome_agent.graph import (build_graph, retry_context, route_after_run_stage,
                                route_after_verify_exec, select_upstream)


def test_select_upstream_passes_all_published_names_by_default():
    stages = {0: {"status": "ok", "result_names": ["box_1", "box_2"]}}
    assert select_upstream({"inputs": []}, stages, 1) == ["box_1", "box_2"]


def test_retry_context_is_none_on_a_first_attempt():
    state = {"stages": {0: {"status": "pending"}}, "history": [], "last_error": None}
    assert retry_context(state, 0, client=None) is None


def test_retry_context_gathers_error_hint_and_built_shapes():
    state = {
        "stages": {0: {"status": "failed",
                       "call_log": [{"fn": "MakeBoxDXDYDZ", "args": [],
                                    "out_vars": ["box_1"], "publish": ["box_1"]}]}},
        "history": [{"event": "retry_hint", "hint": "move it further right"}],
        "last_error": {"phase": "exec", "message": "bad box", "offending_line": None,
                       "traceback": None},
    }
    client = MagicMock()
    client.call_tool.return_value = {"ok": True, "measurements": {
        "box_1": {"bbox": [0, 10, 0, 10, 0, 10], "size": [10, 10, 10], "centre": [5, 5, 5]}}}
    ctx = retry_context(state, 0, client=client)
    assert ctx["last_error"]["message"] == "bad box"
    assert ctx["hint"] == "move it further right"
    assert ctx["built"] == ["box_1"]
    assert ctx["built_info"]["box_1"]["centre"] == [5, 5, 5]
    assert ctx["call_log"] == state["stages"][0]["call_log"]
    client.call_tool.assert_called_once_with("measure", {"names": ["box_1"]})


def test_retry_context_without_a_hint_leaves_hint_none():
    state = {
        "stages": {0: {"status": "failed", "call_log": []}},
        "history": [],
        "last_error": {"phase": "exec", "message": "bad box", "offending_line": None,
                       "traceback": None},
    }
    assert retry_context(state, 0, client=None)["hint"] is None


def test_retry_context_measure_failure_is_best_effort():
    state = {
        "stages": {0: {"status": "failed",
                       "call_log": [{"fn": "X", "args": [], "out_vars": ["a"], "publish": ["a"]}]}},
        "history": [], "last_error": None,
    }
    client = MagicMock()
    client.call_tool.side_effect = RuntimeError("session busy")
    ctx = retry_context(state, 0, client=client)
    assert ctx["built"] == ["a"]
    assert ctx["built_info"] is None


def test_select_upstream_falls_back_when_inputs_do_not_resolve():
    stages = {0: {"status": "ok", "result_names": ["box_1"]}}
    assert select_upstream({"inputs": ["cube"]}, stages, 1) == ["box_1"]


def test_select_upstream_ignores_a_partial_input_match():
    # planner named one real object and invented another -> hand over
    # everything live, not just the one that happened to resolve (GH #9).
    stages = {0: {"status": "ok", "result_names": ["box_1", "cyl_1"]}}
    assert select_upstream({"inputs": ["cyl_1", "beam_union"]}, stages, 1) == ["box_1", "cyl_1"]


def test_select_upstream_ignores_non_ok_and_later_stages():
    stages = {0: {"status": "skipped", "result_names": ["x"]},
              1: {"status": "ok", "result_names": ["box_1"]},
              2: {"status": "ok", "result_names": ["later"]}}
    assert select_upstream({"inputs": []}, stages, 2) == ["box_1"]


def test_geom_finish_stage_routes_to_advance():
    state = {"stage_idx": 0, "stages": {0: {"module": "geom", "status": "ok"}}}
    assert route_after_run_stage(state) == "advance"


def test_geom_loop_exhausted_routes_to_retry_gate():
    state = {"stage_idx": 0, "stages": {0: {"module": "geom", "status": "failed"}}}
    assert route_after_run_stage(state) == "retry_gate"


def test_smesh_ok_routes_to_verify_static():
    state = {"stage_idx": 0, "stages": {0: {"module": "smesh", "status": "pending"}}}
    assert route_after_run_stage(state) == "verify_static"


def test_verify_exec_session_died_routes_to_assemble():
    state = {"last_error": {"phase": "exec", "message": "SessionDied"}}
    assert route_after_verify_exec(state, session_died=True) == "assemble"


def _initial_state(work_dir):
    return {
        "user_request": "make a cube", "run_id": "testrun",
        "config": {"batch": False, "work_dir": str(work_dir),
                   "policy": {"tool_iteration_cap": 3}},
        "session": {"mode": "none", "pid": None, "salome_port": None, "name_prefix": ""},
        "plan": [], "plan_approved": False, "stage_idx": 0, "stages": {},
        "last_error": None, "history": [], "study_ns": "salome_agent_run",
        "final_script": None, "report": None, "fatal": None, "aborted": False,
    }


def test_abort_at_plan_gate_writes_no_output_files(tmp_path):
    llm = MagicMock()
    llm.chat.return_value = {
        "stages": [{"module": "geom", "intent": "box", "params": {}, "inputs": []}]}
    graph = build_graph(batch=False)
    cfg = {"configurable": {"thread_id": "t1", "llm_client": llm, "executor_client": None}}

    first = graph.invoke(_initial_state(tmp_path), config=cfg)
    assert first.get("__interrupt__")

    final = graph.invoke(Command(resume={"action": "abort"}), config=cfg)
    assert final.get("aborted") is True
    assert not final.get("report")
    assert not final.get("final_script")
    assert list(tmp_path.iterdir()) == []


def test_geom_stage_measures_upstream_and_forwards_it(monkeypatch):
    from salome_agent import graph
    from salome_agent.specialists.tool_specialist import ToolSpecialist

    captured = {}

    def fake_run(self, stage_plan, llm, client, iteration_cap, *, blocks=(),
                tool_selection="hints", verify_cadence="final", retry=None,
                seed_call_log=None, seed_calls=None, cancel=None, upstream=()):
        captured["blocks"] = blocks
        captured["tool_selection"] = tool_selection
        captured["upstream"] = upstream
        return {"call_log": [], "result_names": ["walls_1"]}

    monkeypatch.setattr(ToolSpecialist, "run", fake_run)

    client = MagicMock()
    client.call_tool.return_value = {
        "ok": True,
        "measurements": {"fuse_1": {"bbox": [0, 100, 0, 100, 0, 20],
                                    "size": [100, 100, 20], "centre": [50, 50, 10]}},
    }
    state = {
        "stage_idx": 1,
        "plan": [
            {"module": "geom", "intent": "beams", "params": {}, "inputs": []},
            {"module": "geom", "intent": "walls", "params": {}, "inputs": []},
        ],
        "stages": {
            0: {"module": "geom", "status": "ok", "result_names": ["fuse_1"]},
            1: {"module": "geom", "status": "pending"},
        },
        "config": {"policy": {"tool_iteration_cap": 5, "on_stage_failure": "abort"}},
        "last_error": None,
    }
    config = {"configurable": {"llm_client": MagicMock(), "executor_client": client}}

    graph._run_stage_node(state, config)

    client.call_tool.assert_called_once_with("measure", {"names": ["fuse_1"]})
    assert len(captured["blocks"]) == 1
    assert "fuse_1" in captured["blocks"][0]
    assert "[50, 50, 10]" in captured["blocks"][0]
    assert list(captured["upstream"]) == ["fuse_1"]


def _http_error(code):
    resp = requests.Response()
    resp.status_code = code
    return requests.HTTPError(f"{code} Server Error", response=resp)


def test_plan_node_turns_endpoint_5xx_into_a_clean_fatal(monkeypatch):
    from salome_agent import graph

    llm = MagicMock()
    llm.chat.side_effect = _http_error(503)
    state = {"user_request": "x", "config": {"policy": {}}}
    cmd = graph._plan_node(state, {"configurable": {"llm_client": llm}})
    assert cmd.goto == "assemble"
    assert "503" in cmd.update["fatal"]
    assert cmd.update["fatal"].startswith("LLM endpoint")


def test_endpoint_fatal_message_distinguishes_429_and_4xx():
    from salome_agent.graph import _llm_endpoint_fatal

    assert "rate-limited" in _llm_endpoint_fatal(_http_error(429))
    assert "429" in _llm_endpoint_fatal(_http_error(429))
    assert "rejected the request" in _llm_endpoint_fatal(_http_error(404))
    assert "backend is" in _llm_endpoint_fatal(_http_error(502))


def test_plan_node_still_raises_a_non_endpoint_error(monkeypatch):
    from salome_agent import graph

    llm = MagicMock()
    llm.chat.side_effect = KeyError("stages")
    state = {"user_request": "x", "config": {"policy": {}}}
    with pytest.raises(KeyError):
        graph._plan_node(state, {"configurable": {"llm_client": llm}})


def test_run_stage_node_turns_endpoint_error_into_a_clean_fatal(monkeypatch):
    from salome_agent import graph
    from salome_agent.specialists.tool_specialist import ToolSpecialist

    def boom(self, *a, **k):
        raise requests.ConnectionError("no route to host")

    monkeypatch.setattr(ToolSpecialist, "run", boom)
    state = {
        "stage_idx": 0,
        "plan": [{"module": "geom", "intent": "box", "params": {}, "inputs": []}],
        "stages": {0: {"module": "geom", "status": "pending"}},
        "config": {"policy": {"tool_iteration_cap": 5}},
        "last_error": None,
    }
    config = {"configurable": {"llm_client": MagicMock(), "executor_client": None}}
    cmd = graph._run_stage_node(state, config)
    assert cmd.goto == "assemble"
    assert "fatal" in cmd.update


def test_unknown_planner_module_becomes_a_fatal_not_a_crash(tmp_path):
    from salome_agent import graph

    llm = MagicMock()
    llm.chat.return_value = {
        "stages": [{"module": "mesh", "intent": "mesh it", "params": {}, "inputs": []}]}
    llm.usage = None
    built = build_graph(batch=True)
    cfg = {"configurable": {"thread_id": "t1", "llm_client": llm, "executor_client": None}}
    state = _initial_state(tmp_path)
    state["config"]["batch"] = True

    final = built.invoke(state, config=cfg)
    assert final["fatal"] == "planner produced unknown module 'mesh'"
    assert final["report"]


def test_endpoint_fatal_run_produces_report_and_exit_code(tmp_path, monkeypatch):
    llm = MagicMock()
    llm.chat.side_effect = _http_error(502)
    llm.usage = None
    graph = build_graph(batch=True)
    cfg = {"configurable": {"thread_id": "t1", "llm_client": llm, "executor_client": None}}

    final = graph.invoke(_initial_state(tmp_path), config=cfg)
    assert final.get("fatal")
    assert "502" in final["fatal"]
    assert "Run aborted" in final["report"]
    assert (tmp_path / "testrun" / "report.md").exists()


def test_first_geom_stage_does_not_call_measure(monkeypatch):
    from salome_agent import graph
    from salome_agent.specialists.tool_specialist import ToolSpecialist

    captured = {}

    def fake_run(self, stage_plan, llm, client, iteration_cap, *, blocks=(),
                tool_selection="hints", verify_cadence="final", retry=None,
                seed_call_log=None, seed_calls=None, cancel=None, upstream=()):
        captured["tool_selection"] = tool_selection
        return {"call_log": [], "result_names": ["box_1"]}

    monkeypatch.setattr(ToolSpecialist, "run", fake_run)
    client = MagicMock()
    state = {
        "stage_idx": 0,
        "plan": [{"module": "geom", "intent": "box", "params": {}, "inputs": []}],
        "stages": {0: {"module": "geom", "status": "pending"}},
        "config": {"policy": {"tool_iteration_cap": 5, "on_stage_failure": "abort",
                              "tool_selection": "all"}},
        "last_error": None,
    }
    config = {"configurable": {"llm_client": MagicMock(), "executor_client": client}}
    graph._run_stage_node(state, config)
    client.call_tool.assert_not_called()
    assert captured["tool_selection"] == "all"


class _FakeLLM:
    """An llm client with a real usage dict a fake specialist can advance."""
    def __init__(self):
        self.usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
                      "llm_seconds": 0.0}

    def burn(self, calls, in_tok, out_tok, seconds):
        self.usage["calls"] += calls
        self.usage["prompt_tokens"] += in_tok
        self.usage["completion_tokens"] += out_tok
        self.usage["llm_seconds"] += seconds


def _geom_state(status_stage=None):
    return {
        "stage_idx": 0,
        "plan": [{"module": "geom", "intent": "box", "params": {}, "inputs": []}],
        "stages": {0: status_stage or {"module": "geom", "status": "pending"}},
        "config": {"policy": {"tool_iteration_cap": 5, "on_stage_failure": "abort"}},
        "last_error": None, "history": [],
    }


def test_run_stage_node_records_per_stage_timing(monkeypatch):
    from salome_agent import graph
    from salome_agent.specialists.tool_specialist import ToolSpecialist

    llm = _FakeLLM()

    def fake_run(self, stage_plan, llm, client, iteration_cap, *, blocks=(),
                tool_selection="hints", verify_cadence="final", retry=None,
                seed_call_log=None, seed_calls=None, cancel=None, upstream=()):
        llm.burn(calls=4, in_tok=12000, out_tok=1500, seconds=20.0)
        return {"call_log": [], "result_names": ["box_1"]}

    monkeypatch.setattr(ToolSpecialist, "run", fake_run)
    state = _geom_state()
    cmd = graph._run_stage_node(state, {"configurable": {"llm_client": llm,
                                                         "executor_client": None}})
    tm = cmd.update["stages"][0]["timing"]
    assert tm["turns"] == 4 and tm["in_tok"] == 12000 and tm["out_tok"] == 1500
    assert tm["llm_s"] == 20.0
    assert tm["core_s"] == round(tm["wall"] - 20.0, 2)


def test_run_stage_node_timing_accumulates_across_a_retry(monkeypatch):
    from salome_agent import graph
    from salome_agent.specialists.tool_specialist import ToolSpecialist

    llm = _FakeLLM()

    def fake_run(self, stage_plan, llm, client, iteration_cap, *, blocks=(),
                tool_selection="hints", verify_cadence="final", retry=None,
                seed_call_log=None, seed_calls=None, cancel=None, upstream=()):
        llm.burn(calls=3, in_tok=5000, out_tok=600, seconds=10.0)
        return {"call_log": [], "result_names": ["box_1"]}

    monkeypatch.setattr(ToolSpecialist, "run", fake_run)
    prior = {"module": "geom", "status": "failed",
             "timing": {"wall": 12.0, "llm_s": 10.0, "core_s": 2.0, "turns": 3,
                        "in_tok": 5000, "out_tok": 600}}
    state = _geom_state(status_stage=prior)
    cmd = graph._run_stage_node(state, {"configurable": {"llm_client": llm,
                                                         "executor_client": None}})
    tm = cmd.update["stages"][0]["timing"]
    assert tm["turns"] == 6 and tm["in_tok"] == 10000
    assert tm["llm_s"] == 20.0


def test_run_stage_node_skips_timing_without_a_real_usage_dict(monkeypatch):
    from salome_agent import graph
    from salome_agent.specialists.tool_specialist import ToolSpecialist

    def fake_run(self, stage_plan, llm, client, iteration_cap, *, blocks=(),
                tool_selection="hints", verify_cadence="final", retry=None,
                seed_call_log=None, seed_calls=None, cancel=None, upstream=()):
        return {"call_log": [], "result_names": ["box_1"]}

    monkeypatch.setattr(ToolSpecialist, "run", fake_run)
    state = _geom_state()
    cmd = graph._run_stage_node(state, {"configurable": {"llm_client": MagicMock(),
                                                         "executor_client": None}})
    assert "timing" not in cmd.update["stages"][0]


def test_describe_image_node_skips_straight_to_plan_without_images():
    from salome_agent import graph

    llm = MagicMock()
    cmd = graph._describe_image_node({"user_request": "x"}, {"configurable": {"llm_client": llm}})
    assert cmd.goto == "plan" and not cmd.update
    llm.chat.assert_not_called()


def test_describe_image_node_stores_the_description_for_the_planner():
    from salome_agent import graph

    llm = MagicMock()
    llm.chat.return_value = {"summary": "an L-bracket"}
    state = {"user_request": "x", "reference_images": ["data:image/png;base64,AAAA"]}
    cmd = graph._describe_image_node(state, {"configurable": {"llm_client": llm}})
    assert cmd.goto == "plan"
    assert "an L-bracket" in cmd.update["image_description"]


def test_describe_image_node_hands_a_follow_up_turn_to_amend():
    from salome_agent import graph

    llm = MagicMock()
    llm.chat.return_value = {"summary": "a handle"}
    state = {"user_request": "x", "turn_idx": 1,
             "reference_images": ["data:image/png;base64,AAAA"]}
    cmd = graph._describe_image_node(state, {"configurable": {"llm_client": llm}})
    assert cmd.goto == "amend" and "a handle" in cmd.update["image_description"]
    cmd = graph._describe_image_node({"user_request": "x", "turn_idx": 1},
                                     {"configurable": {"llm_client": llm}})
    assert cmd.goto == "amend"


def test_describe_image_node_turns_endpoint_error_into_a_clean_fatal():
    from salome_agent import graph

    llm = MagicMock()
    llm.chat.side_effect = _http_error(503)
    state = {"user_request": "x", "reference_images": ["data:image/png;base64,AAAA"]}
    cmd = graph._describe_image_node(state, {"configurable": {"llm_client": llm}})
    assert cmd.goto == "assemble" and "503" in cmd.update["fatal"]


def test_plan_node_hands_the_description_not_the_images_to_the_planner(monkeypatch):
    from salome_agent import graph
    from salome_agent.nodes import planner

    seen = {}

    def fake_plan(request, llm, detail="balanced", images=None, image_description=None):
        seen.update(images=images, image_description=image_description)
        return [{"module": "geom", "intent": "box", "params": {}, "inputs": []}]

    monkeypatch.setattr(planner, "plan", fake_plan)
    state = {"user_request": "x", "config": {"policy": {}},
             "reference_images": ["data:image/png;base64,AAAA"],
             "image_description": "Summary: a box"}
    graph._plan_node(state, {"configurable": {"llm_client": MagicMock()}})
    assert seen == {"images": None, "image_description": "Summary: a box"}


def test_select_upstream_always_returns_every_available_name():
    from salome_agent.graph import select_upstream

    stages = {0: {"status": "ok", "result_names": ["box_1", "cyl_1"]}}
    # `inputs` holds plain words by design (planner.py), never object names —
    # a full match used to scope down and never fires in practice.
    assert select_upstream({"inputs": ["the box"]}, stages, 1) == ["box_1", "cyl_1"]
    assert select_upstream({"inputs": ["box_1"]}, stages, 1) == ["box_1", "cyl_1"]
    assert select_upstream({"inputs": []}, stages, 1) == ["box_1", "cyl_1"]


def test_retry_context_carries_the_failed_attempts_calls():
    calls = [{"tool": "make_box", "tool_args": {"dx": 1}, "records": []}]
    state = {"stages": {0: {"status": "failed", "call_log": [], "calls": calls}},
             "last_error": None, "history": []}
    assert retry_context(state, 0, client=None)["calls"] == calls


def _one_geom_stage(status="pending", **stage):
    return {"stage_idx": 0,
            "plan": [{"module": "geom", "intent": "box", "params": {}, "inputs": []}],
            "stages": {0: {"module": "geom", "status": status, **stage}},
            "config": {"policy": {"tool_iteration_cap": 5, "on_stage_failure": "abort"}},
            "last_error": None, "history": []}


def test_run_stage_keeps_the_call_trace(monkeypatch):
    from salome_agent import graph
    from salome_agent.specialists.tool_specialist import ToolSpecialist

    calls = [{"tool": "make_box", "tool_args": {"dx": 1}, "records": []}]

    def fake_run(self, stage_plan, llm, client, iteration_cap, *, blocks=(),
                 tool_selection="hints", verify_cadence="final", retry=None,
                 seed_call_log=None, seed_calls=None, cancel=None, upstream=()):
        return {"call_log": [], "calls": calls, "result_names": ["box_1"]}

    monkeypatch.setattr(ToolSpecialist, "run", fake_run)
    cmd = graph._run_stage_node(_one_geom_stage(), {"configurable": {
        "llm_client": MagicMock(), "executor_client": None}})
    assert cmd.update["stages"][0]["calls"] == calls


def test_retry_seeds_the_failed_attempts_calls(monkeypatch):
    from salome_agent import graph
    from salome_agent.specialists.tool_specialist import ToolSpecialist

    old = [{"tool": "make_box", "tool_args": {"dx": 1}, "records": []}]
    seen = {}

    def fake_run(self, stage_plan, llm, client, iteration_cap, *, blocks=(),
                 tool_selection="hints", verify_cadence="final", retry=None,
                 seed_call_log=None, seed_calls=None, cancel=None, upstream=()):
        seen["seed_calls"] = seed_calls
        return {"call_log": [], "calls": old, "result_names": ["box_1"]}

    monkeypatch.setattr(ToolSpecialist, "run", fake_run)
    graph._run_stage_node(_one_geom_stage("failed", call_log=[], calls=old),
                          {"configurable": {"llm_client": MagicMock(), "executor_client": None}})
    assert seen["seed_calls"] == old
