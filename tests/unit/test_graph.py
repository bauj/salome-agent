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


def test_retry_context_measures_a_shaper_stage_in_shaper():
    state = {"stages": {0: {"status": "failed", "module": "shaper",
                            "call_log": [{"out_vars": ["ext_1"], "publish": ["ext_1"]}]}},
             "history": [], "last_error": {"message": "open sketch"}}
    client = MagicMock()
    client.call_tool.return_value = {"ok": True, "measurements": {"ext_1": {"faces": 6}}}
    assert retry_context(state, 0, client=client)["built_info"] == {"ext_1": {"faces": 6}}
    client.call_tool.assert_called_once_with("measure", {"names": ["ext_1"]}, module="shaper")


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


def test_outdated_server_ends_the_run_with_its_own_message():
    from salome_agent.executor_client import ServerOutdated
    from salome_agent.graph import _llm_endpoint_fatal

    msg = _llm_endpoint_fatal(ServerOutdated("the agent server has no shaper tools"))
    assert msg == "the agent server has no shaper tools"


def test_endpoint_fatal_4xx_names_base_url_and_quotes_the_body():
    from salome_agent.graph import _llm_endpoint_fatal

    err = _http_error(400)
    err.response._content = b'{"error": {"message":\n  "missing thought_signature"}}'
    msg = _llm_endpoint_fatal(err)
    assert "base_url" in msg
    assert 'endpoint said: {"error": {"message": "missing thought_signature"}}' in msg
    err.response._content = b"x" * 1000
    assert _llm_endpoint_fatal(err).endswith("x" * 300 + "…")
    assert "endpoint said" not in _llm_endpoint_fatal(_http_error(404))


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


def test_describe_image_node_stores_the_spec_for_the_planner():
    from salome_agent import graph
    from tests.unit.test_vision_spec import box_spec

    llm = MagicMock()
    llm.chat.return_value = box_spec()
    state = {"user_request": "x", "reference_images": ["data:image/png;base64,AAAA"]}
    cmd = graph._describe_image_node(state, {"configurable": {"llm_client": llm}})
    assert cmd.goto == "plan"
    assert cmd.update["image_spec"] == box_spec()
    assert cmd.update["reference_images"] is None


def test_describe_image_node_reads_vision_fix_retries_from_config():
    from salome_agent import graph

    llm = MagicMock()
    llm.chat.return_value = {"summary": "a box"}
    state = {"user_request": "x", "reference_images": ["x"],
             "config": {"llm": {"vision_fix_retries": 0}}}
    cmd = graph._describe_image_node(state, {"configurable": {"llm_client": llm}})
    assert llm.chat.call_count == 1
    assert "parts must be a non-empty list" in cmd.update["image_spec"]["conflicts"]


def test_describe_image_node_hands_a_follow_up_turn_to_amend():
    from salome_agent import graph
    from tests.unit.test_vision_spec import box_spec

    llm = MagicMock()
    llm.chat.return_value = box_spec()
    state = {"user_request": "x", "turn_idx": 1,
             "reference_images": ["data:image/png;base64,AAAA"]}
    cmd = graph._describe_image_node(state, {"configurable": {"llm_client": llm}})
    assert cmd.goto == "amend" and cmd.update["image_spec"]["summary"] == "a block with a hole"
    cmd = graph._describe_image_node({"user_request": "x", "turn_idx": 1},
                                     {"configurable": {"llm_client": llm}})
    assert cmd.goto == "amend"


def test_plan_node_hands_the_spec_not_the_images_to_the_planner(monkeypatch):
    from salome_agent import graph
    from salome_agent.nodes import planner

    seen = {}

    def fake_plan(request, llm, detail="balanced", image_spec=None, **kw):
        seen.update(image_spec=image_spec)
        return [{"module": "geom", "intent": "box", "params": {}, "inputs": []}]

    monkeypatch.setattr(planner, "plan", fake_plan)
    state = {"user_request": "x", "config": {"policy": {}},
             "reference_images": ["data:image/png;base64,AAAA"],
             "image_spec": {"summary": "a box", "parts": []}}
    graph._plan_node(state, {"configurable": {"llm_client": MagicMock()}})
    assert seen == {"image_spec": {"summary": "a box", "parts": []}}


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


def _capture_stage_blocks(monkeypatch, plan, idx, image_spec):
    from salome_agent import graph
    from salome_agent.specialists import tool_specialist

    seen = {}

    def fake_run(self, stage_plan, llm, client, cap, blocks=(), **kw):
        seen["blocks"] = blocks
        return {"status": "ok", "call_log": [], "calls": [], "result_names": []}

    monkeypatch.setattr(tool_specialist.ToolSpecialist, "run", fake_run)
    monkeypatch.setattr(graph, "retry_context", lambda state, idx, client: None)
    state = {"plan": plan, "stage_idx": idx,
             "stages": {i: {"module": p["module"], "status": "pending"}
                        for i, p in enumerate(plan)},
             "config": {"policy": {"tool_iteration_cap": 5}}, "image_spec": image_spec,
             "build_graph": None}
    try:
        graph._run_stage_node(state, {"configurable": {"llm_client": MagicMock(),
                                                       "executor_client": None}})
    except Exception:  # noqa: BLE001 — only the blocks handed to the specialist matter
        pass
    return "\n".join(seen.get("blocks") or ())


def test_geom_stage_gets_only_its_parts(monkeypatch):
    from tests.unit.test_vision_spec import box_spec
    plan = [{"module": "geom", "intent": "a", "params": {}, "inputs": [], "parts": ["P1"]},
            {"module": "geom", "intent": "b", "params": {}, "inputs": [], "parts": ["P2"]}]
    text = _capture_stage_blocks(monkeypatch, plan, 1, box_spec())
    assert '"id": "P2"' in text and '"id": "P1"' not in text


def test_shaper_stage_gets_its_parts_too(monkeypatch):
    from tests.unit.test_vision_spec import box_spec
    plan = [{"module": "shaper", "intent": "a", "params": {}, "inputs": [], "parts": ["P1"]}]
    text = _capture_stage_blocks(monkeypatch, plan, 0, box_spec())
    assert '"id": "P1"' in text and '"id": "P2"' not in text


def test_geom_stage_gets_every_part_when_the_planner_listed_none(monkeypatch):
    from tests.unit.test_vision_spec import box_spec
    plan = [{"module": "geom", "intent": "a", "params": {}, "inputs": []}]
    text = _capture_stage_blocks(monkeypatch, plan, 0, box_spec())
    assert '"id": "P1"' in text and '"id": "P2"' in text


def test_geom_stage_without_parts_gets_no_spec_when_others_list_theirs(monkeypatch):
    from tests.unit.test_vision_spec import box_spec
    plan = [{"module": "geom", "intent": "a", "params": {}, "inputs": [], "parts": ["P1", "P2"]},
            {"module": "geom", "intent": "colour it", "params": {}, "inputs": []}]
    text = _capture_stage_blocks(monkeypatch, plan, 1, box_spec())
    assert "Reference image spec" not in text


def test_endpoint_fatal_explains_a_second_reasoning_cut():
    from salome_agent.graph import _llm_endpoint_fatal
    from salome_agent.llm import ReasoningTimeExceeded

    msg = _llm_endpoint_fatal(ReasoningTimeExceeded(300, 9000))
    assert "even after being told to decide" in msg
    assert "300s" in msg


def test_endpoint_fatal_explains_a_repetition_loop():
    from salome_agent.graph import _llm_endpoint_fatal
    from salome_agent.llm import RepetitionLoopDetected

    msg = _llm_endpoint_fatal(RepetitionLoopDetected("reasoning", 2400, "x", "y"))
    assert "repeating itself" in msg


def test_shaper_stage_routes_like_a_tool_module():
    ok = {"stage_idx": 0, "stages": {0: {"module": "shaper", "status": "ok"}}}
    failed = {"stage_idx": 0, "stages": {0: {"module": "shaper", "status": "failed"}}}
    assert route_after_run_stage(ok) == "advance"
    assert route_after_run_stage(failed) == "retry_gate"


def _blocks_for(monkeypatch, plan, request):
    from salome_agent import graph
    from salome_agent.specialists.tool_specialist import ToolSpecialist

    captured = {}

    def fake_run(self, stage_plan, llm, client, iteration_cap, *, blocks=(), **kw):
        captured["blocks"] = blocks
        return {"call_log": [], "result_names": ["x_1"]}

    monkeypatch.setattr(ToolSpecialist, "run", fake_run)
    state = {
        "user_request": request, "stage_idx": 0, "plan": plan,
        "stages": {i: {"module": "geom", "status": "pending"} for i in range(len(plan))},
        "config": {"policy": {"tool_iteration_cap": 5, "on_stage_failure": "abort"}},
        "last_error": None,
    }
    graph._run_stage_node(state, {"configurable": {"llm_client": MagicMock(),
                                                   "executor_client": MagicMock()}})
    return captured["blocks"]


def test_stage_gets_the_users_request_as_written(monkeypatch):
    stage = {"module": "geom", "intent": "badge", "params": {}, "inputs": []}
    blocks = _blocks_for(monkeypatch, [stage], "a badge: ring, four bars, gap cut from the central bars")
    assert blocks[-1].endswith("\na badge: ring, four bars, gap cut from the central bars")
    assert "Other stages" not in blocks[-1]


def test_multi_stage_request_block_says_other_stages_build_the_rest(monkeypatch):
    stage = {"module": "geom", "intent": "part", "params": {}, "inputs": []}
    blocks = _blocks_for(monkeypatch, [stage, dict(stage)], "a table and a chair")
    assert "Other stages build the rest" in blocks[-1]


def test_no_request_block_without_a_request(monkeypatch):
    stage = {"module": "geom", "intent": "box", "params": {}, "inputs": []}
    assert _blocks_for(monkeypatch, [stage], "  ") == ()
