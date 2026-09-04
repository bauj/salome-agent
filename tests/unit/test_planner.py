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


def test_plan_forwards_reference_images_as_multimodal_content():
    llm = MagicMock()
    llm.chat.return_value = {"stages": []}
    plan("make a box like this", llm, images=["data:image/png;base64,AAAA",
                                              "data:image/bmp;base64,BBBB"])
    user_content = llm.chat.call_args.kwargs["messages"][1]["content"]
    assert user_content == [
        {"type": "text", "text": "make a box like this"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        {"type": "image_url", "image_url": {"url": "data:image/bmp;base64,BBBB"}},
    ]


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
