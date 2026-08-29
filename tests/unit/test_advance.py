from salome_agent.nodes.advance import advance


def test_geom_stage_codegens_immediately_on_advance():
    state = {
        "stage_idx": 0,
        "stages": {0: {"module": "geom", "status": "pending", "script": None,
                       "call_log": [{"fn": "MakeBoxDXDYDZ", "args": [], "out_vars": ["box_1"],
                                     "publish": ["box_1"]}],
                       "static_retries": 0, "exec_retries": 0}},
    }
    update = advance(state, result_names=["box_1"])
    assert "box_1 = geompy.MakeBoxDXDYDZ()" in update["stages"][0]["script"]
    assert update["stages"][0]["status"] == "ok"


def test_smesh_stage_keeps_its_own_script():
    state = {
        "stage_idx": 1,
        "stages": {1: {"module": "smesh", "status": "pending", "script": "mesh.Compute()",
                       "call_log": None, "static_retries": 0, "exec_retries": 0}},
    }
    update = advance(state)
    assert update["stages"][1]["script"] == "mesh.Compute()"
    assert update["stages"][1]["status"] == "ok"
