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


def test_geom_smesh_preamble_is_exactly_as_before():
    """Assemble with geom and smesh and verify the full preamble is EXACTLY
    as before: kernel init, then geom init, then smesh init exactly once.
    """
    from salome_agent.nodes.assembler import _KERNEL_INIT, _GEOM_INIT, _SMESH_INIT

    state = {
        "run_id": "r",
        "user_request": "x",
        "stages": {
            0: {"module": "geom", "status": "ok", "script": "box_1 = geompy.MakeBoxDXDYDZ(1, 1, 1)"},
            1: {"module": "smesh", "status": "ok", "script": "mesh.Compute()"}
        }
    }
    result = assemble(state)

    # Verify the preamble contains exactly one of each init
    assert result.count("from salome.geom import geomBuilder") == 1, "geom init should appear exactly once"
    assert result.count("from salome.smesh import smeshBuilder") == 1, "smesh init should appear exactly once"

    # Verify order: kernel init < geom init < smesh init
    kernel_pos = result.index("salome_init()")
    geom_pos = result.index("geomBuilder")
    smesh_pos = result.index("smeshBuilder")
    assert kernel_pos < geom_pos < smesh_pos, f"Wrong order: kernel={kernel_pos}, geom={geom_pos}, smesh={smesh_pos}"

    # Verify the stage scripts come after the preamble
    stage_pos = result.index("# stage 1/2:")
    assert geom_pos < stage_pos and smesh_pos < stage_pos, "Init should come before stages"


def test_geom_init_precedes_smesh_init_with_no_geom_stage():
    # modules={"smesh"} but the smesh script references geompy (an object
    # built in an earlier, already-assembled run) — geom init still comes
    # first, matching main's fixed order.
    from salome_agent.nodes.assembler import _GEOM_INIT, _SMESH_INIT

    state = {"run_id": "r", "user_request": "x", "stages": {
        0: {"module": "smesh", "status": "ok",
            "script": "box_1 = geompy.MakeBoxDXDYDZ(1, 1, 1)\nmesh.Compute()"}}}
    result = assemble(state)

    assert result.index(_GEOM_INIT) < result.index(_SMESH_INIT)


def _box(name, dz=1.0):
    return {"fn": "MakeBoxDXDYDZ", "args": [{"kind": "literal", "value": 1.0},
                                            {"kind": "literal", "value": 1.0},
                                            {"kind": "literal", "value": dz}],
            "out_vars": [name], "publish": [name]}


def _fuse(out, a, b):
    return {"fn": "MakeFuse", "args": [{"kind": "ref", "value": a}, {"kind": "ref", "value": b}],
            "out_vars": [out], "publish": [out]}


def _state_with_calls():
    s1 = [{"tool": "make_box", "tool_args": {}, "records": [_box("box_1")]},
          {"tool": "make_box", "tool_args": {}, "records": [_box("box_2")]}]
    s2 = [{"tool": "fuse", "tool_args": {}, "records": [_fuse("fuse_1", "box_1", "box_2")]}]
    return {"run_id": "r1", "user_request": "two boxes fused",
            "stages": {0: {"module": "geom", "status": "ok", "script": "IGNORED",
                           "calls": s1, "result_names": ["box_1"]},
                       1: {"module": "geom", "status": "ok", "script": "IGNORED",
                           "calls": s2, "result_names": ["fuse_1"]}}}


def test_graph_projection_keeps_stage_headers_and_crosses_stage_refs():
    out = assemble(_state_with_calls())
    assert "# stage 1/2: geom\nbox_1 = geompy.MakeBoxDXDYDZ(1.0, 1.0, 1.0)" in out
    # box_2 is not in stage 1's result_names but stage 2 needs it: whole-graph reachability
    assert "box_2 = geompy.MakeBoxDXDYDZ(1.0, 1.0, 1.0)" in out
    assert "# stage 2/2: geom\nfuse_1 = geompy.MakeFuse(box_1, box_2)" in out
    assert "IGNORED" not in out


def test_commit_stages_labels_later_turns_and_records_requests():
    from salome_agent.build_graph import BuildGraph
    from salome_agent.nodes.assembler import commit_stages, render_script
    g = commit_stages(BuildGraph(), _state_with_calls())
    st = _state_with_calls()
    st.update(turn_idx=1, user_request="add a third", stages={
        0: {"module": "geom", "status": "ok", "script": None, "result_names": ["box_3"],
            "calls": [{"tool": "make_box", "tool_args": {}, "records": [_box("box_3")]}]}})
    commit_stages(g, st)
    assert [b.label for b in g.blocks][-1] == "turn 2 stage 1/1: geom"
    out = render_script(g, "r1")
    assert out.startswith("# generated by salome-agent, run r1\n# request: two boxes fused\n"
                          "# turn 2: add a third\n")


def test_script_block_for_geom_stage_without_calls_is_todays_output():
    state = {"run_id": "r1", "user_request": "x",
             "stages": {0: {"module": "geom", "status": "ok", "script": "a = 1"}}}
    assert "# stage 1/1: geom\na = 1" in assemble(state)


def test_multiline_request_stays_a_comment():
    state = {
        "stages": {0: {"module": "geom", "status": "ok",
                       "script": "box_1 = geompy.MakeBoxDXDYDZ(10,10,10)"}},
        "run_id": "x", "user_request": "a plate:\n- x=±54\n\n- holes", "plan": [],
    }
    result = assemble(state)
    ast.parse(result)
    assert "# request: a plate:\n# - x=±54\n#\n# - holes\n" in result
