import ast

from salome_agent.nodes.assembler import assemble


def test_assembles_ordered_stages_with_preamble():
    state = {
        "stages": {
            0: {"module": "geom", "status": "ok", "script": "box_1 = geompy.MakeBoxDXDYDZ(1,1,1)"},
            1: {"module": "smesh", "status": "ok", "script": "mesh.Compute()"},
        },
        "run_id": "20260829-abcd",
        "user_request": "make a box and mesh it",
        "plan": [],
    }
    result = assemble(state)
    ast.parse(result)
    assert result.index("salome_init") < result.index("MakeBoxDXDYDZ") < result.index("Compute")


def test_skipped_stages_are_excluded():
    state = {
        "stages": {0: {"module": "geom", "status": "skipped", "script": None}},
        "run_id": "x", "user_request": "y", "plan": [],
    }
    result = assemble(state)
    assert "MakeBoxDXDYDZ" not in result


def test_geom_only_result_omits_smesh_and_find_object():
    state = {
        "stages": {0: {"module": "geom", "status": "ok",
                       "script": "box_1 = geompy.MakeBoxDXDYDZ(10,10,10)"}},
        "run_id": "x", "user_request": "make a box", "plan": [],
    }
    result = assemble(state)
    ast.parse(result)
    assert "smeshBuilder" not in result
    assert "find_object" not in result
    assert "geomBuilder" in result


def test_find_object_helper_included_when_script_uses_it():
    state = {
        "stages": {0: {"module": "smesh", "status": "ok",
                       "script": 'shape = find_object("box_1")\nmesh.Compute()'}},
        "run_id": "x", "user_request": "mesh the box", "plan": [],
    }
    result = assemble(state)
    ast.parse(result)
    assert "def find_object" in result
    assert "smeshBuilder" in result
    # study_ns is only defined when a stage script actually references it
    assert "study_ns" not in result


def test_study_ns_defined_only_when_referenced():
    state = {
        "stages": {0: {"module": "smesh", "status": "ok",
                       "script": 'mesh.Mesh(s, study_ns + "_m")\nmesh.Compute()'}},
        "run_id": "x", "user_request": "mesh it", "plan": [],
    }
    result = assemble(state)
    ast.parse(result)
    assert 'study_ns = "salome_agent_run"' in result


def test_salomeds_import_added_when_color_used():
    state = {
        "stages": {0: {"module": "geom", "status": "ok",
                       "script": "box_1 = geompy.MakeBoxDXDYDZ(10,10,10)\nbox_1.SetColor(SALOMEDS.Color(1.0, 0.0, 0.0))"}},
        "run_id": "x", "user_request": "make a red box", "plan": [],
    }
    result = assemble(state)
    ast.parse(result)
    assert "from salome.kernel import SALOMEDS" in result
    assert "SALOMEDS.Color" in result


def test_gui_import_added_when_gg_used():
    state = {
        "stages": {0: {"module": "geom", "status": "ok",
                       "script": "box_1 = geompy.MakeBoxDXDYDZ(10,10,10)\ntry:\n    gg.setTransparency(salome.ObjectToID(box_1), 0.5)\nexcept Exception:\n    pass"}},
        "run_id": "x", "user_request": "make a transparent box", "plan": [],
    }
    result = assemble(state)
    ast.parse(result)
    assert 'gg = salome.ImportComponentGUI("GEOM")' in result
    assert "gg.setTransparency" in result


def test_both_salomeds_and_gui_imports_when_both_used():
    state = {
        "stages": {0: {"module": "geom", "status": "ok",
                       "script": "box_1 = geompy.MakeBoxDXDYDZ(10,10,10)\nbox_1.SetColor(SALOMEDS.Color(1.0, 0.0, 0.0))\ntry:\n    gg.setTransparency(salome.ObjectToID(box_1), 0.5)\nexcept Exception:\n    pass"}},
        "run_id": "x", "user_request": "make a colored and transparent box", "plan": [],
    }
    result = assemble(state)
    ast.parse(result)
    assert "from salome.kernel import SALOMEDS" in result
    assert 'gg = salome.ImportComponentGUI("GEOM")' in result


def test_geom_preamble_comes_from_the_module_spec():
    from salome_agent.modules import spec_for

    state = {"run_id": "r", "user_request": "x", "stages": {
        0: {"module": "geom", "status": "ok", "script": "box_1 = geompy.MakeBoxDXDYDZ(1, 1, 1)"}}}
    assert spec_for("geom").preamble in assemble(state)


def test_smesh_preamble_comes_from_the_module_spec():
    from salome_agent.modules import spec_for

    state = {"run_id": "r", "user_request": "x", "stages": {
        0: {"module": "smesh", "status": "ok", "script": "mesh.Compute()"}}}
    assert spec_for("smesh").preamble in assemble(state)


def test_the_two_copies_of_each_preamble_cannot_drift():
    # assembler keeps its own constants for the frozen body-sniffing
    # fallbacks; the specs carry the same text for the module-driven path.
    from salome_agent.modules import spec_for
    from salome_agent.nodes.assembler import _GEOM_INIT, _SMESH_INIT

    assert spec_for("geom").preamble == _GEOM_INIT
    assert spec_for("smesh").preamble == _SMESH_INIT
    assert spec_for("paravis").preamble == _SMESH_INIT
