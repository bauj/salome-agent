from unittest.mock import MagicMock

from langgraph.types import Command

from salome_agent.graph import (build_graph, route_after_run_stage,
                                route_after_verify_exec, select_upstream)


def test_select_upstream_passes_all_published_names_by_default():
    stages = {0: {"status": "ok", "result_names": ["box_1", "box_2"]}}
    assert select_upstream({"inputs": []}, stages, 1) == ["box_1", "box_2"]


def test_select_upstream_scopes_to_declared_inputs_when_they_resolve():
    stages = {0: {"status": "ok", "result_names": ["box_1", "cyl_1"]}}
    assert select_upstream({"inputs": ["cyl_1"]}, stages, 1) == ["cyl_1"]


def test_select_upstream_falls_back_when_inputs_do_not_resolve():
    stages = {0: {"status": "ok", "result_names": ["box_1"]}}
    assert select_upstream({"inputs": ["cube"]}, stages, 1) == ["box_1"]


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
