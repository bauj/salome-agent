from unittest.mock import MagicMock

from salome_agent.nodes.vision import VISION_SYSTEM_PROMPT, describe_images
from tests.unit.test_vision_spec import box_spec


def test_describe_images_sends_images_to_the_vision_module():
    llm = MagicMock()
    llm.chat.return_value = box_spec()
    out = describe_images("make this", ["data:image/png;base64,AAAA"], llm)
    assert out == box_spec()
    assert llm.chat.call_count == 1
    kwargs = llm.chat.call_args.kwargs
    assert kwargs["module"] == "vision"
    content = kwargs["messages"][1]["content"]
    assert content[0]["type"] == "text" and "make this" in content[0]["text"]
    assert content[1] == {"type": "image_url",
                          "image_url": {"url": "data:image/png;base64,AAAA"}}


def test_a_broken_spec_is_sent_back_with_its_errors():
    bad = box_spec()
    bad["parts"][1]["ops"] = [{"op": "fuse", "with": "P9"}]
    llm = MagicMock()
    llm.chat.side_effect = [bad, box_spec()]
    out = describe_images("make this", ["data:image/png;base64,AAAA"], llm)
    assert out == box_spec()
    messages = llm.chat.call_args_list[1].kwargs["messages"]
    assert messages[2]["role"] == "assistant" and '"P9"' in messages[2]["content"]
    assert messages[3]["role"] == "user"
    assert "- P2: ops[1].with must name another part (P1, P2)" in messages[3]["content"]
    assert "whole corrected JSON" in messages[3]["content"]


def test_errors_left_after_the_retries_become_conflicts():
    bad = box_spec()
    bad["parts"][1]["ops"] = [{"op": "fuse", "with": "P9"}]
    llm = MagicMock()
    llm.chat.return_value = bad
    out = describe_images("make this", ["x"], llm, fix_retries=1)
    assert llm.chat.call_count == 2
    assert out["parts"][1]["conflicts"] == ["ops[1].with must name another part (P1, P2)"]


def test_zero_retries_never_re_asks():
    llm = MagicMock()
    llm.chat.return_value = {"summary": "a box"}
    out = describe_images("make this", ["x"], llm, fix_retries=0)
    assert llm.chat.call_count == 1
    assert "parts must be a non-empty list" in out["conflicts"]


def test_prompt_asks_for_dimensions_read_before_parts_and_profiles_as_drawn():
    p = VISION_SYSTEM_PROMPT
    assert p.index('"dimensions_read"') < p.index('"parts"')
    assert '"kind": "profile"' in p and '"kind": "approx"' in p
    assert "the tool finds the tangent points" in p
    assert '"assumed": true' in p


def test_prompt_makes_every_linear_dimension_name_both_ends():
    p = VISION_SYSTEM_PROMPT
    assert '"from": what the first extension line sits on' in p
    assert "centre-to-centre" in p
