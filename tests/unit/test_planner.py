from unittest.mock import MagicMock

from salome_agent.nodes.planner import PLANNER_SYSTEM_PROMPT, plan


def test_prompt_documents_the_optional_tools_hint():
    assert '"tools"' in PLANNER_SYSTEM_PROMPT
    assert "geom tool names" in PLANNER_SYSTEM_PROMPT.lower()


def test_prompt_caps_planner_reasoning_output():
    p = PLANNER_SYSTEM_PROMPT.lower()
    assert "output only the json" in p
    assert "no analysis" in p


def test_a_returned_tools_key_survives_into_the_plan():
    llm = MagicMock()
    llm.chat.return_value = {"stages": [
        {"module": "geom", "intent": "red box", "params": {}, "inputs": [],
         "tools": ["make_box", "set_color"]},
    ]}
    stages = plan("make a red box", llm)
    assert stages[0]["tools"] == ["make_box", "set_color"]


def test_repeated_module_yields_two_stages():
    llm = MagicMock()
    llm.chat.return_value = {
        "stages": [
            {"module": "geom", "intent": "make a 20mm cube", "params": {"size": 20}, "inputs": []},
            {"module": "geom", "intent": "fillet edges 2mm", "params": {"radius": 2}, "inputs": ["cube"]},
            {"module": "smesh", "intent": "mesh with 5mm tets", "params": {"max_size": 5}, "inputs": []},
        ]
    }
    stages = plan("make a 20mm cube, fillet its edges 2mm, mesh it with 5mm tetrahedra", llm)
    assert [s["module"] for s in stages] == ["geom", "geom", "smesh"]
    assert stages[1]["params"]["radius"] == 2  # mm, unconverted


def test_geom_only_request_yields_no_smesh_stage():
    llm = MagicMock()
    llm.chat.return_value = {"stages": [{"module": "geom", "intent": "make a box",
                                          "params": {}, "inputs": []}]}
    stages = plan("make a box", llm)
    assert len(stages) == 1
    assert stages[0]["module"] == "geom"


def test_planner_prompt_lists_geom_tool_catalogue():
    from salome_agent.nodes.planner import PLANNER_SYSTEM_PROMPT
    assert "make_thick_solid" in PLANNER_SYSTEM_PROMPT
    assert "make_box" in PLANNER_SYSTEM_PROMPT
    assert "fillet_edges" in PLANNER_SYSTEM_PROMPT


def test_plan_detail_hint_appended_to_system_prompt():
    from salome_agent.nodes.planner import _DETAIL_HINT
    llm = MagicMock()
    llm.chat.return_value = {"stages": []}
    plan("make a widget", llm, detail="fine")
    system = llm.chat.call_args.kwargs["messages"][0]["content"]
    assert _DETAIL_HINT["fine"] in system
    assert system.startswith("You turn a natural-language SALOME request")


def test_plan_without_image_sends_plain_text_content():
    llm = MagicMock()
    llm.chat.return_value = {"stages": []}
    plan("make a box", llm)
    assert llm.chat.call_args.kwargs["messages"][1]["content"] == "make a box"


def test_plan_unknown_detail_falls_back_to_no_hint():
    llm = MagicMock()
    llm.chat.return_value = {"stages": []}
    plan("x", llm, detail="bogus")  # no crash, no extra hint
    system = llm.chat.call_args.kwargs["messages"][0]["content"]
    from salome_agent.nodes.planner import PLANNER_SYSTEM_PROMPT
    assert system == PLANNER_SYSTEM_PROMPT


def test_plan_appends_the_compact_spec_to_the_request():
    from tests.unit.test_vision_spec import p1_spec
    llm = MagicMock()
    llm.chat.return_value = {"stages": [{"module": "geom", "intent": "plate", "params": {},
                                         "inputs": [], "parts": ["P1"]}]}
    stages = plan("make this plate", llm, image_spec=p1_spec())
    user = llm.chat.call_args.kwargs["messages"][1]["content"]
    assert user.startswith("make this plate\n\nReference image spec")
    assert "- P1 plate: profile in XY" in user
    assert stages[0]["parts"] == ["P1"]


def test_unknown_part_ids_are_sent_back_once_then_dropped():
    from tests.unit.test_vision_spec import p1_spec
    llm = MagicMock()
    bad = {"stages": [{"module": "geom", "intent": "plate", "params": {}, "inputs": [],
                       "parts": ["P1", "P7"]}]}
    llm.chat.side_effect = [bad, bad]
    stages = plan("make this plate", llm, image_spec=p1_spec())
    assert llm.chat.call_count == 2
    retry = llm.chat.call_args_list[1].kwargs["messages"]
    assert retry[-1]["role"] == "user"
    assert 'Unknown part id(s) P7 in "parts"; the parts are P1.' in retry[-1]["content"]
    assert stages[0]["parts"] == ["P1"]


def test_a_bare_string_parts_value_is_read_as_a_list():
    from tests.unit.test_vision_spec import p1_spec
    llm = MagicMock()
    llm.chat.return_value = {"stages": [{"module": "geom", "intent": "plate", "params": {},
                                         "inputs": [], "parts": "P1"}]}
    stages = plan("make this plate", llm, image_spec=p1_spec())
    assert llm.chat.call_count == 1
    assert stages[0]["parts"] == ["P1"]


def test_prompt_tells_the_planner_to_list_part_ids_not_numbers():
    assert '"parts"' in PLANNER_SYSTEM_PROMPT
    assert "Reference image spec" in PLANNER_SYSTEM_PROMPT
    assert "never restate" in PLANNER_SYSTEM_PROMPT.lower()


def test_every_routable_module_contributes_its_catalogue():
    from salome_agent.modules import routable

    specs = {s.name for s in routable()}
    assert specs == {"geom", "shaper", "smesh"}
    for spec in routable():
        assert spec.catalogue.strip() in PLANNER_SYSTEM_PROMPT


def test_a_module_with_no_catalogue_is_not_routable():
    from salome_agent.modules import ModuleSpec, Terminator, routable

    spec = ModuleSpec(
        name="amend", prompt=lambda c: "",
        terminator=Terminator("finish_turn", {}, "", lambda a, c: {}),
        execute=lambda c, n, a: {"ok": True}, render_result=str,
        tools_for=lambda c: {}, no_tool_nudge="", on_exhausted=lambda c, e: {})
    assert spec not in routable()


def test_a_module_named_in_the_request_wins():
    assert "always wins" in PLANNER_SYSTEM_PROMPT
    assert "(geom|shaper|smesh)" in PLANNER_SYSTEM_PROMPT


def test_a_looping_planner_is_re_asked_once_without_the_loop():
    from salome_agent.llm import RepetitionLoopDetected
    llm = MagicMock()
    stages = [{"module": "shaper", "intent": "badge", "params": {}, "inputs": []}]
    llm.chat.side_effect = [RepetitionLoopDetected("reasoning", 2400, "Ring R50, bars 10 wide.",
                                                   "But we need to specify holes."),
                            {"stages": stages}]
    llm.loop_retries = 2
    assert plan("a badge", llm) == stages
    nudge = llm.chat.call_args.kwargs["messages"][-1]
    assert nudge["role"] == "user"
    assert "Ring R50, bars 10 wide." in nudge["content"]
    assert "But we need to specify holes." in nudge["content"]
    assert "JSON plan" in nudge["content"]


def test_a_planner_that_keeps_looping_past_loop_retries_raises():
    import pytest
    from salome_agent.llm import RepetitionLoopDetected
    llm = MagicMock()
    llm.loop_retries = 2
    llm.chat.side_effect = [RepetitionLoopDetected("reasoning", 2400, "x", "y y y.")] * 3
    with pytest.raises(RepetitionLoopDetected):
        plan("a badge", llm)
    assert llm.chat.call_count == 3


def test_a_planner_answering_within_loop_retries_is_kept():
    from salome_agent.llm import RepetitionLoopDetected
    llm = MagicMock()
    llm.loop_retries = 2
    stages = [{"module": "shaper", "intent": "badge", "params": {}, "inputs": []}]
    llm.chat.side_effect = [RepetitionLoopDetected("reasoning", 2400, "x", "y y y.")] * 2 + [
        {"stages": stages}]
    assert plan("a badge", llm) == stages


def test_planner_prompt_lists_shaper_tool_catalogue():
    assert "SHAPER tools available to a single SHAPER stage" in PLANNER_SYSTEM_PROMPT
    assert "make_face -- flat face" in PLANNER_SYSTEM_PROMPT


def test_shaper_planner_lines_cover_every_shaper_tool():
    import executor.shaper
    from salome_agent.specialists.shaper import PLANNER_TOOL_LINES
    assert set(PLANNER_TOOL_LINES) == {t["name"] for t in executor.shaper.list_tools()}


def test_named_cad_module():
    from salome_agent.nodes.planner import named_cad_module
    assert named_cad_module("draw the gasket in SHAPER") == "shaper"
    assert named_cad_module("a plate using the shaper module") == "shaper"
    assert named_cad_module("a box with GEOM") == "geom"
    assert named_cad_module("a box with a hole") is None
    assert named_cad_module("sketch in shaper, then fillet using geom") is None
    assert named_cad_module("mesh it with smesh") is None


def test_named_module_is_told_to_the_planner():
    llm = MagicMock()
    llm.chat.return_value = {"stages": []}
    plan("a gasket in shaper", llm)
    system = llm.chat.call_args.kwargs["messages"][0]["content"]
    assert system.endswith('every CAD geometry stage uses module "shaper".')


def test_named_module_moves_cad_stages_and_keeps_smesh():
    llm = MagicMock()
    llm.chat.return_value = {"stages": [
        {"module": "shaper", "intent": "sketch", "params": {}, "inputs": []},
        {"module": "geom", "intent": "face", "params": {}, "inputs": [],
         "tools": ["make_profile_face"]},
        {"module": "smesh", "intent": "mesh", "params": {}, "inputs": []},
    ]}
    stages = plan("a gasket in shaper, then mesh it", llm)
    assert [s["module"] for s in stages] == ["shaper", "shaper", "smesh"]
    assert "tools" not in stages[1]


def test_no_named_module_leaves_routing_alone():
    llm = MagicMock()
    llm.chat.return_value = {"stages": [
        {"module": "geom", "intent": "box", "params": {}, "inputs": [], "tools": ["make_box"]}]}
    stages = plan("a box", llm)
    assert stages[0]["module"] == "geom" and stages[0]["tools"] == ["make_box"]
