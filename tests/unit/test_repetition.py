import re
from pathlib import Path

import pytest

from salome_agent.repetition import RepetitionWatch

_FIX = Path(__file__).parent.parent / "fixtures" / "repetition"


def _feed(text: str, chunk: int = 23):
    watch = RepetitionWatch()
    for i in range(0, len(text), chunk):
        loop = watch.feed(text[i:i + chunk])
        if loop:
            return loop, i + chunk
    return None, len(text)


def _unique_words(n: int) -> str:
    return " ".join(f"w{i}" for i in range(n)) + ". "


@pytest.mark.parametrize("name", ["loop_gpt_oss_planner", "loop_nemotron_shaper"])
def test_real_loops_are_caught_early(name):
    text = (_FIX / f"{name}.txt").read_text()
    loop, at = _feed(text)
    assert loop is not None
    assert len(re.findall(r"\w+", text[:at])) < 3200
    assert loop.start < at
    # the carried part holds the looped sentence at most once
    assert text[:loop.start].count(loop.sample) <= 1
    assert text[loop.start:at].count(loop.sample) >= 2


@pytest.mark.parametrize("name", ["ok_nemotron_cut_by_budget", "ok_nemotron_turn1",
                                  "ok_deepseek_shaper", "ok_deepseek_planner",
                                  "ok_kimi_planner"])
def test_real_healthy_reasoning_is_left_alone(name):
    loop, _ = _feed((_FIX / f"{name}.txt").read_text())
    assert loop is None


def test_short_text_never_fires():
    sentence = "But we can create a sketch that includes the bars as the outer contour. "
    loop, _ = _feed(sentence * 30)  # ~400 words, below the window
    assert loop is None


def test_verbatim_cycle_after_real_work_starts_at_the_cycle():
    head = _unique_words(800)
    sentence = ("But we can create a sketch that includes the bars as part of the "
                "outer contour, then approximate it as a polygon with many edges. ")
    text = head + sentence * 60
    loop, at = _feed(text)
    assert loop is not None
    assert len(head) - 400 <= loop.start <= len(head) + len(sentence)
    assert loop.sample.strip() == sentence.strip()


def test_chunking_does_not_change_the_result():
    text = _unique_words(800) + "the outer bars stop at the central bar's far edge. " * 80
    a, _ = _feed(text, chunk=7)
    b, _ = _feed(text, chunk=500)
    assert a is not None and b is not None
    assert a.start == b.start and a.sample == b.sample
