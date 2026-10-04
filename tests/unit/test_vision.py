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


def _uri(data: bytes, mime="image/png") -> str:
    import base64
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


def _png(w, h):
    import struct
    return b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR" + struct.pack(">II", w, h) + b"\x08\x02\x00\x00\x00"


def test_image_size_reads_png_jpeg_gif_bmp_webp_headers():
    import struct
    from salome_agent.nodes.vision import image_size

    assert image_size(_uri(_png(510, 550))) == (510, 550)
    jpeg = (b"\xff\xd8" + b"\xff\xe0" + struct.pack(">H", 4) + b"JF"
            + b"\xff\xc0" + struct.pack(">HBHH", 11, 8, 300, 640) + b"\x03\x01\x22\x00")
    assert image_size(_uri(jpeg, "image/jpeg")) == (640, 300)
    assert image_size(_uri(b"GIF89a" + struct.pack("<HH", 12, 34), "image/gif")) == (12, 34)
    bmp = b"BM" + b"\x00" * 16 + struct.pack("<ii", 77, -88)
    assert image_size(_uri(bmp, "image/bmp")) == (77, 88)
    vp8x = b"RIFF\x00\x00\x00\x00WEBPVP8X" + b"\x00" * 8 + (99).to_bytes(3, "little") + (49).to_bytes(3, "little")
    assert image_size(_uri(vp8x, "image/webp")) == (100, 50)
    vp8l = b"RIFF\x00\x00\x00\x00WEBPVP8L" + b"\x00" * 4 + b"\x2f" + struct.pack("<I", (20 - 1) | ((30 - 1) << 14))
    assert image_size(_uri(vp8l, "image/webp")) == (20, 30)
    vp8 = (b"RIFF\x00\x00\x00\x00WEBPVP8 " + b"\x00" * 4 + b"\x00\x00\x00\x9d\x01\x2a"
           + struct.pack("<HH", 64, 48))
    assert image_size(_uri(vp8, "image/webp")) == (64, 48)


def test_image_size_is_none_when_it_cannot_tell():
    from salome_agent.nodes.vision import image_size

    assert image_size("data:image/png;base64,AAAA") is None
    assert image_size("not a uri") is None


def test_describe_images_tells_the_model_each_image_pixel_size():
    llm = MagicMock()
    llm.chat.return_value = box_spec()
    describe_images("make this", [_uri(_png(510, 550)), "data:image/png;base64,AAAA"], llm)
    text = llm.chat.call_args.kwargs["messages"][1]["content"][0]["text"]
    assert "image 1: 510 x 550 px" in text
    assert "image 2" not in text


def test_image_size_survives_truncated_and_garbage_headers():
    import random
    import struct
    from salome_agent.nodes.vision import image_size

    heads = [_png(510, 550),
             b"\xff\xd8\xff\xe0" + struct.pack(">H", 4) + b"JF\xff\xc0" + struct.pack(">HBHH", 11, 8, 3, 6),
             b"GIF89a" + struct.pack("<HH", 1, 2),
             b"BM" + b"\x00" * 16 + struct.pack("<ii", 3, 4),
             b"RIFF\x00\x00\x00\x00WEBPVP8 " + b"\x00" * 7 + struct.pack("<HH", 1, 1)]
    rng = random.Random(0)
    for head in heads:
        for cut in range(len(head)):
            image_size(_uri(head[:cut]))
        for _ in range(50):
            junk = bytearray(head)
            junk[rng.randrange(len(junk))] = rng.randrange(256)
            image_size(_uri(bytes(junk) + bytes(rng.randrange(256) for _ in range(8))))
    for _ in range(200):
        image_size(_uri(bytes(rng.randrange(256) for _ in range(rng.randrange(40)))))


def _recorder():
    seen = []
    return seen, lambda kind, **data: seen.append((kind, data))


def test_a_clean_read_reports_start_and_done():
    llm = MagicMock()
    llm.endpoint.return_value.model = "reader"
    llm.chat.return_value = box_spec()
    seen, on_progress = _recorder()
    describe_images("make this", [_uri(_png(510, 550)), "x"], llm, on_progress=on_progress)
    llm.endpoint.assert_called_with("vision")
    assert seen[0] == ("vision_start", {"images": 2, "sizes": ["510x550"], "model": "reader"})
    kind, data = seen[1]
    assert kind == "vision_done" and len(seen) == 2
    assert (data["parts"], data["conflicts"]) == (len(box_spec()["parts"]), 0)
    assert isinstance(data["seconds"], float)


def test_each_fix_round_reports_its_problems():
    bad = box_spec()
    bad["parts"][1]["ops"] = [{"op": "fuse", "with": "P9"}]
    llm = MagicMock()
    llm.chat.side_effect = [bad, box_spec()]
    seen, on_progress = _recorder()
    describe_images("make this", ["x"], llm, fix_retries=2, on_progress=on_progress)
    assert [k for k, _ in seen] == ["vision_start", "vision_check", "vision_done"]
    assert seen[1][1] == {"attempt": 1, "of": 2,
                          "errors": ["P2: ops[1].with must name another part (P1, P2)"]}
    assert seen[2][1]["conflicts"] == 0


def test_problems_left_after_the_last_round_are_counted_as_conflicts():
    bad = box_spec()
    bad["parts"][1]["ops"] = [{"op": "fuse", "with": "P9"}]
    llm = MagicMock()
    llm.chat.return_value = bad
    seen, on_progress = _recorder()
    describe_images("make this", ["x"], llm, fix_retries=1, on_progress=on_progress)
    assert [k for k, _ in seen] == ["vision_start", "vision_check", "vision_done"]
    assert seen[2][1]["conflicts"] == 1


def test_no_callback_no_endpoint_lookup():
    llm = MagicMock()
    llm.chat.return_value = box_spec()
    describe_images("make this", ["x"], llm)
    llm.endpoint.assert_not_called()


def _timed_out(reasoning="R36 is a boss, so"):
    from salome_agent.llm import ReasoningTimeExceeded
    return ReasoningTimeExceeded(300, 9000, reasoning)


def test_a_vision_call_cut_for_reasoning_too_long_is_re_asked_to_decide():
    from salome_agent.nodes.vision import DECIDE_NOW
    llm = MagicMock()
    llm.chat.side_effect = [_timed_out(), box_spec()]
    out = describe_images("make this", ["x"], llm)
    assert out == box_spec()
    first, second = llm.chat.call_args_list
    assert first.kwargs["reasoning_share"] == 0.5
    assert second.kwargs["reasoning_share"] == 0.5
    assert second.kwargs["messages"][:-1] == first.kwargs["messages"]
    nudge = second.kwargs["messages"][-1]
    assert nudge["role"] == "user"
    assert nudge["content"].startswith(DECIDE_NOW)
    assert nudge["content"].endswith("R36 is a boss, so")


def test_the_conclude_nudge_keeps_only_the_tail_of_long_reasoning():
    from salome_agent.nodes.vision import decide_now
    msg = decide_now("a" * 50_000 + "THE END")
    assert "(earlier part omitted)" in msg
    assert msg.endswith("THE END")
    assert len(msg) < 41_000


def test_a_second_reasoning_cut_ends_the_vision_step():
    import pytest
    from salome_agent.llm import ReasoningTimeExceeded
    llm = MagicMock()
    llm.chat.side_effect = [_timed_out(), _timed_out()]
    with pytest.raises(ReasoningTimeExceeded):
        describe_images("make this", ["x"], llm)


def test_the_vision_prompt_says_to_move_on_from_an_unclear_feature():
    assert "do not reopen it" in VISION_SYSTEM_PROMPT
