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
    # the SMESH tool preamble extends the script one (imports, helpers)
    assert spec_for("smesh").preamble.startswith(_SMESH_INIT)


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


def _shaper_stage():
    sketch = {"fn": "addSketch", "emit": "stmt",
              "code": 'sketch_1 = model.addSketch(shaper_doc, model.defaultPlane("XOY"))',
              "args": [], "out_vars": ["sketch_1"], "publish": ["sketch_1"]}
    line = {"fn": "addLine", "emit": "stmt", "recv": "sketch_1",
            "code": "sketch_1_L1 = sketch_1.addLine(0, 0, 1, 0)",
            "args": [], "out_vars": ["sketch_1_L1"], "publish": None}
    return {"module": "shaper", "status": "ok", "script": None,
            "result_names": ["sketch_1"], "call_log": [sketch, line],
            "calls": [{"tool": "create_sketch", "tool_args": {"plane": "XOY"},
                       "records": [sketch]},
                      {"tool": "sketch_add", "tool_args": {}, "records": [line]}]}


def test_shaper_stage_renders_its_records_between_preamble_and_postamble():
    state = {"stages": {0: _shaper_stage()}, "run_id": "x",
             "user_request": "in shaper, a sketch", "plan": []}
    result = assemble(state)
    ast.parse(result)
    assert "from salome.shaper import model" in result
    assert "geomBuilder" not in result
    body = result.index('sketch_1 = model.addSketch(shaper_doc, model.defaultPlane("XOY"))')
    assert result.index("shaper_doc = model.addPart") < body
    assert body < result.index("_name(sketch_1, 'sketch_1')") < result.index("sketch_1_L1 =")
    assert result.rstrip().endswith("model.publishToShaperStudy()")


def test_shaper_calls_are_kept_as_shaper_nodes():
    from salome_agent.build_graph import BuildGraph
    from salome_agent.nodes.assembler import commit_stages
    state = {"stages": {0: _shaper_stage()}, "user_request": "r"}
    graph = commit_stages(BuildGraph(), state)
    assert [b.module for b in graph.blocks] == ["shaper"]
    nodes = list(graph.nodes.values())
    assert [n.module for n in nodes] == ["shaper", "shaper"]
    assert nodes[1].deps == [nodes[0].id]


def _ext_stage():
    ext = {"fn": "addExtrusion", "emit": "stmt",
           "code": "ext_1 = model.addExtrusion(shaper_doc, [], model.selection(), 5, 0)",
           "args": [], "out_vars": ["ext_1"], "publish": ["ext_1"]}
    return {"module": "shaper", "status": "ok", "script": None, "result_names": ["ext_1"],
            "calls": [{"tool": "extrude", "tool_args": {}, "records": [ext]}]}


def test_a_script_stage_after_shaper_work_sees_it_published():
    mesh = {"module": "smesh", "status": "ok", "script": 'm = smesh.Mesh(find_object("ext_1"))'}
    out = assemble({"stages": {0: _ext_stage(), 1: mesh}, "run_id": "x",
                    "user_request": "r", "plan": []})
    ast.parse(out)
    bridge = out.index("model.end()\nmodel.publishToShaperStudy()\nmodel.begin()\n")
    assert out.index("ext_1 = model.addExtrusion") < bridge < out.index("smesh.Mesh(")
    assert out.rstrip().endswith("model.publishToShaperStudy()")


def test_a_shaper_only_result_has_no_bridge():
    out = assemble({"stages": {0: _ext_stage()}, "run_id": "x", "user_request": "r",
                    "plan": []})
    assert out.count("publishToShaperStudy") == 1


def _render_shaper_then_smesh():
    from salome_agent.build_graph import BuildGraph
    from salome_agent.nodes.assembler import render_script
    ext = _ext_stage()["calls"][0]
    mesh = {"fn": "Mesh", "emit": "stmt",
            "code": "mesh_1 = smesh.Mesh(_shaper_shape('ext_1'), 'mesh_1')",
            "out_vars": ["mesh_1"], "args": [{"kind": "expr", "value": "", "refs": ["ext_1"]}],
            "publish": None}
    g = BuildGraph()
    b1 = g.add_block(0, "stage 1/2: shaper", "shaper", ["ext_1"])
    g.append(b1, ext["tool"], ext["tool_args"], ext["records"], 0, module="shaper")
    b2 = g.add_block(0, "stage 2/2: smesh", "smesh", ["mesh_1"])
    g.append(b2, "create_mesh", {"shape": "ext_1"}, [mesh], 0, module="smesh")
    return render_script(g, "r1")


def test_smesh_tool_block_after_shaper_gets_the_shaper_results_published():
    text = _render_shaper_then_smesh()
    shaper_end = text.index("# stage 2/2: smesh")
    assert text[shaper_end:].split("\n")[1:4] == ["model.end()",
                                                  "model.publishToShaperStudy()",
                                                  "model.begin()"]
    assert text.count("smesh = smeshBuilder.New()") == 1
    assert "from salome.kernel import SMESH, StdMeshers" in text


def test_an_amend_block_that_only_sets_up_an_earlier_mesh_renders_without_a_warning(caplog):
    from salome_agent.build_graph import BuildGraph
    from salome_agent.nodes.assembler import render_script
    mesh = {"fn": "Mesh", "emit": "stmt", "code": "mesh_1 = smesh.Mesh(box_1, 'mesh_1')",
            "out_vars": ["mesh_1"], "args": [], "publish": None}
    compute = {"fn": "Compute", "emit": "stmt", "code": "mesh_1.Compute()",
               "out_vars": [], "args": [], "recv": "mesh_1"}
    g = BuildGraph()
    b1 = g.add_block(0, "stage 1/1: smesh", "smesh", ["mesh_1"])
    g.append(b1, "create_mesh", {"shape": "box_1"}, [mesh], 0, module="smesh")
    b2 = g.add_block(1, "turn 2: amend", "smesh", [])
    g.append(b2, "compute", {"mesh": "mesh_1"}, [compute], 1, module="smesh")
    with caplog.at_level("WARNING", logger="salome_agent"):
        text = render_script(g, "r1")
    assert "mesh_1.Compute()" in text
    assert "match no built shape" not in caplog.text
