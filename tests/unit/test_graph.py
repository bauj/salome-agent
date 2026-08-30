from salome_agent.graph import route_after_run_stage, route_after_verify_exec, select_upstream


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
