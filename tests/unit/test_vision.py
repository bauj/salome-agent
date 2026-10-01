from unittest.mock import MagicMock

from salome_agent.nodes.vision import describe_images, format_description


def test_describe_images_sends_images_to_the_vision_module():
    llm = MagicMock()
    llm.chat.return_value = {"summary": "a box"}
    out = describe_images("make this", ["data:image/png;base64,AAAA"], llm)
    assert out == {"summary": "a box"}
    kwargs = llm.chat.call_args.kwargs
    assert kwargs["module"] == "vision"
    content = kwargs["messages"][1]["content"]
    assert content[0]["type"] == "text" and "make this" in content[0]["text"]
    assert content[1] == {"type": "image_url",
                          "image_url": {"url": "data:image/png;base64,AAAA"}}


def test_format_description_renders_the_main_sections():
    text = format_description({
        "summary": "An L-bracket",
        "views": [{"image": 1, "view": "front", "shows": "the profile"}],
        "parts": [{"name": "base", "shape": "box", "dimensions": {"x": 40},
                   "position": "at origin"}],
        "dimensions_read": ["40 mm wide"],
        "uncertainties": ["hole depth"],
        "extra": "kept",
    })
    assert "Summary: An L-bracket" in text
    assert "Image 1 (front): the profile" in text
    assert "base -- box" in text and '"x": 40' in text
    assert "40 mm wide" in text and "hole depth" in text
    assert "extra: kept" in text


def test_format_description_tolerates_an_empty_answer():
    assert format_description({}) == "{}"


def test_format_description_leads_with_a_2d_target_and_its_plane():
    text = format_description({"dimensionality": "2d", "dimensionality_reason": "a lone outline",
                               "plane": "XZ", "summary": "a gasket profile"})
    assert text.splitlines()[0] == ("Target: 2D (a lone outline) -- a planar shape in the "
                                    "XZ plane, build it without thickness")
    assert "plane:" not in text


def test_format_description_states_a_3d_target():
    text = format_description({"dimensionality": "3D", "summary": "a bracket", "plane": None})
    assert text.splitlines()[0] == "Target: 3D"
