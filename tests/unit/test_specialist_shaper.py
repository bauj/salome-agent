from unittest.mock import MagicMock

from salome_agent.modules import llm_module_names, spec_for


def _rec(out, code="x = 1", recv=None, refs=()):
    r = {"fn": "f", "emit": "stmt", "code": code, "out_vars": [out],
         "args": [{"kind": "expr", "value": "", "refs": list(refs)}] if refs else [],
         "publish": [out]}
    if recv:
        r["recv"] = recv
        r["publish"] = None
    return r


def test_shaper_is_a_registered_tool_module():
    spec = spec_for("shaper")
    assert spec.needs_session is True
    assert spec.call_prefix
    assert spec.forgets is True
    assert "shaper" in llm_module_names()


def test_execute_and_tools_go_to_the_shaper_module():
    client = MagicMock()
    client.list_tools.return_value = {"tools": [{"name": "create_sketch"}]}
    client.call_tool.return_value = {"ok": True}
    spec = spec_for("shaper")
    assert list(spec.tools_for(client)) == ["create_sketch"]
    client.list_tools.assert_called_with(module="shaper")
    spec.execute(client, "create_sketch", {"plane": "XOY"})
    client.call_tool.assert_called_with("create_sketch", {"plane": "XOY"}, module="shaper")


def test_render_result_hides_bookkeeping():
    text = spec_for("shaper").render_result(
        {"ok": True, "records": [1], "result_shape": "s", "shape_info": {}, "dof": 2,
         "regions": []})
    assert "records" not in text and "shape_info" not in text and "result_shape" not in text
    assert "'dof': 2" in text


def test_gate_refuses_while_a_result_sketch_has_loose_ends():
    log = [_rec("sketch_1"), _rec("ext_1", refs=["sketch_1"])]
    infos = {"sketch_1": {"dof": 3, "loose_ends": [{"ref": "L1.end", "at": [1, 0]}]}}
    gate = spec_for("shaper").gate
    msg = gate({"result_names": ["ext_1"]}, log, infos)
    assert "sketch_1" in msg and "L1.end" in msg


def test_gate_ignores_closed_and_unreached_sketches():
    gate = spec_for("shaper").gate
    log = [_rec("sketch_1"), _rec("sketch_2"), _rec("ext_1", refs=["sketch_1"])]
    infos = {"sketch_1": {"dof": 0, "loose_ends": []},
             "sketch_2": {"dof": 0, "loose_ends": [{"ref": "A.end", "at": [0, 0]}]}}
    assert gate({"result_names": ["ext_1"]}, log, infos) is None


def test_finish_returns_result_names_and_call_log():
    log = [_rec("sketch_1")]
    out = spec_for("shaper").terminator.handle({"result_names": ["sketch_1"], "built": "x"}, log)
    assert out == {"result_names": ["sketch_1"], "call_log": log, "built": "x"}


def test_prompt_offers_both_routes_for_a_part_in_pieces():
    from salome_agent.specialists.shaper import SHAPER_SYSTEM_PROMPT
    assert "make_face on the regions" in SHAPER_SYSTEM_PROMPT
    assert "fuse, cut and common" in SHAPER_SYSTEM_PROMPT


def test_upstream_block_names_and_measures_earlier_results():
    client = MagicMock()
    client.call_tool.return_value = {"ok": True, "measurements": {
        "face_1": {"bbox": [-50.0, 50.0, -50.0, 50.0, 0.0, 0.0], "size": [100.0, 100.0, 0.0],
                   "centre": [0.0, 0.0, 0.0], "area": 2443.3737, "volume": 0.0,
                   "faces": 1, "edges": 4, "valid": True},
        "sketch_1": {"dof": 0, "loose_ends": [], "regions": [{"area": 1.0}, {"area": 2.0}]},
        "box_1": None}}
    block = spec_for("shaper").render_upstream(client, ["face_1", "sketch_1", "box_1"])
    client.call_tool.assert_called_with(
        "measure", {"names": ["face_1", "sketch_1", "box_1"]}, module="shaper")
    lines = block.splitlines()
    assert "EARLIER stage" in lines[0]
    assert lines[1].startswith("  face_1 — ") and "area 2443.3737" in lines[1]
    assert "1 face" in lines[1] and "volume" not in lines[1]
    assert lines[2] == "  sketch_1 — a sketch, 2 regions"
    assert lines[3] == "  box_1"


def test_upstream_block_leaves_out_results_a_later_boolean_used_up():
    client = MagicMock()
    client.call_tool.return_value = {"ok": True, "measurements": {
        "face_1": {"area": 1.0, "faces": 1, "concealed": True},
        "fuse_1": {"area": 2.0, "faces": 1}}}
    lines = spec_for("shaper").render_upstream(client, ["face_1", "fuse_1"]).splitlines()
    assert [l.split(" — ")[0] for l in lines[1:]] == ["  fuse_1"]


def test_upstream_block_is_dropped_when_every_result_is_used_up():
    client = MagicMock()
    client.call_tool.return_value = {"ok": True, "measurements": {
        "face_1": {"area": 1.0, "faces": 1, "concealed": True}}}
    assert spec_for("shaper").render_upstream(client, ["face_1"]) is None


def test_upstream_block_names_alone_when_measure_fails():
    client = MagicMock()
    client.call_tool.side_effect = RuntimeError("down")
    assert spec_for("shaper").render_upstream(client, ["face_1"]).splitlines()[1:] == ["  face_1"]
    assert spec_for("shaper").render_upstream(client, []) is None


def test_prompt_follows_the_verify_cadence():
    prompt = spec_for("shaper").prompt
    assert "{verify}" not in prompt("final")
    assert "measure on your result(s) once before finish_stage" in prompt("final")
    assert "After each fuse, cut" in prompt("strict")
    assert "bbox" not in prompt("none").split("Every feature reply carries")[1].split("\n\n")[0]


def test_tool_categories_are_the_geom_ones():
    from executor.shaper import list_tools
    spec = spec_for("shaper")
    assert spec.categories == spec_for("geom").categories
    assert {t["category"] for t in list_tools()} <= set(spec.categories)
    names = {t["name"] for t in list_tools()}
    assert spec.core_tools < names


def test_planner_catalogue_names_every_shaper_tool():
    from executor.shaper import list_tools
    from salome_agent.specialists.shaper import PLANNER_TOOL_LINES
    assert set(PLANNER_TOOL_LINES) == {t["name"] for t in list_tools()}


def test_load_tools_is_offered_with_the_shaper_categories():
    from salome_agent.specialists.tool_specialist import ToolSpecialist
    load = ToolSpecialist(spec_for("shaper"))._load_tools_def()
    assert load["parameters"]["properties"]["categories"]["items"]["enum"] == \
        list(spec_for("shaper").categories)


def test_initial_offer_uses_the_modules_core_set():
    from salome_agent.specialists.tool_selection import initial_offered
    got = initial_offered({"tools": ["revolve"]}, {"sketch_add", "revolve", "loft"},
                          frozenset({"sketch_add"}))
    assert got == {"sketch_add", "revolve"}


def test_prompt_lists_every_shaper_tool_to_load():
    from salome_agent.specialists.shaper import PLANNER_TOOL_LINES
    prompt = spec_for("shaper").prompt("final")
    assert "call\nload_tools with the names" in prompt
    for name, line in PLANNER_TOOL_LINES.items():
        assert f"  {name} -- {line}" in prompt


def test_shaper_core_set_offers_primitives_and_colour():
    core = spec_for("shaper").core_tools
    assert {"make_box", "make_cylinder", "set_color", "create_sketch", "extrude"} <= core
    assert "make_torus" not in core


def test_retry_block_names_the_error_hint_and_built_objects():
    text = spec_for("shaper").render_retry({
        "last_error": {"message": "sketch_1 has open ends"}, "hint": "close it",
        "built": ["sketch_1", "ext_1"],
        "built_info": {"sketch_1": {"regions": [{}]},
                       "ext_1": {"bbox": [0, 1, 0, 1, 0, 1], "area": 6, "faces": 6,
                                 "volume": 1}}})
    assert "failed: sketch_1 has open ends" in text
    assert "User hint for this retry: close it" in text
    assert "sketch_1 — a sketch, 1 regions" in text
    assert "ext_1 — bbox [0, 1, 0, 1, 0, 1], area 6, 6 faces, volume 1" in text


def _call(tool, rec):
    return {"tool": tool, "tool_args": {}, "records": [rec]}


def test_audit_flags_an_unused_solid_and_an_unknown_name():
    sk, ext1, ext2 = _rec("sketch_1"), _rec("ext_1", refs=["sketch_1"]), _rec("ext_2")
    calls = [_call("create_sketch", sk), _call("extrude", ext1), _call("revolve", ext2)]
    report = spec_for("shaper").audit({"result_names": ["ext_1", "typo_1"]}, calls,
                                      [sk, ext1, ext2], frozenset(), {})
    assert "typo_1" in report.split("not a shape built in this run: ")[1].splitlines()[0]
    assert "ext_2" in report.split("used by another shape: ")[1].splitlines()[0]


def test_audit_passes_a_clean_stage_and_ignores_sketches_without_size():
    sk, ext1 = _rec("sketch_1"), _rec("ext_1", refs=["sketch_1"])
    calls = [_call("create_sketch", sk), _call("extrude", ext1)]
    infos = {"sketch_1": {"regions": []}, "ext_1": {"size": [1, 1, 1]}}
    assert spec_for("shaper").audit({"result_names": ["ext_1"]}, calls, [sk, ext1],
                                    frozenset(), infos) is None


def test_audit_flags_an_empty_result():
    ext1 = _rec("ext_1")
    report = spec_for("shaper").audit({"result_names": ["ext_1"]}, [_call("extrude", ext1)],
                                      [ext1], frozenset(),
                                      {"ext_1": {"size": [0, 0, 0], "bbox": [0] * 6,
                                                 "area": 0, "faces": 0}})
    assert "empty or a single point" in report and "ext_1 — bbox" in report
