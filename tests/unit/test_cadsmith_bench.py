import os
import sys

import pytest

trimesh = pytest.importorskip("trimesh")
pytest.importorskip("scipy")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "bench"))
from cadsmith_bench import _tokens, score  # noqa: E402


def _box(tmp_path, name, extents, at=(0, 0, 0)):
    m = trimesh.creation.box(extents=extents)
    m.apply_translation(at)
    path = str(tmp_path / f"{name}.stl")
    m.export(path)
    return path


def test_same_part_scores_perfect(tmp_path):
    a = _box(tmp_path, "a", (40, 20, 10))
    s = score(a, a)
    assert s["f1"] == pytest.approx(1.0, abs=1e-3)
    assert s["iou"] == pytest.approx(1.0) and s["iou_paper"] == pytest.approx(1.0)
    assert s["cd"] < 0.5
    assert s["ref_volume"] == pytest.approx(8000)


def test_moved_part_only_hurts_placed_scores(tmp_path):
    ref = _box(tmp_path, "ref", (40, 20, 10))
    moved = _box(tmp_path, "moved", (40, 20, 10), at=(5, 0, 0))
    s = score(moved, ref)
    assert s["f1"] > 0.99 and s["iou"] > 0.99
    assert s["f1_placed"] < 0.9 and s["cd_placed"] > 5


def test_wrong_size_scores_low(tmp_path):
    ref = _box(tmp_path, "ref", (40, 20, 10))
    big = _box(tmp_path, "big", (60, 20, 10))
    s = score(big, ref)
    assert s["iou"] < 0.75 and s["f1"] < 0.9


def test_tokens_from_report(tmp_path):
    (tmp_path / "report.md").write_text(
        "- calls: 12\n- input tokens: 75,816\n- output tokens: 3,379\n"
        "- stage 1/2 (shaper): 13.4s wall, 12.4s llm\n- stage 1/2 (shaper): ok — 2 turns\n- stage 2/2 (shaper): failed — 5 turns\n")
    t = _tokens(str(tmp_path))
    assert t["llm_calls"] == 12 and t["input_tokens"] == 75816
    assert t["modules"] == ["shaper", "shaper"] and t["stages_ok"] is False


def test_surface_scores_ignore_part_size(tmp_path):
    pytest.importorskip("rtree")
    big = _box(tmp_path, "big", (120, 120, 30))
    s = score(big, big)
    assert s["f1"] < 0.95
    assert s["f1_surf"] == pytest.approx(1.0) and s["d99_surf"] < 0.01


def test_surface_scores_see_a_moved_part(tmp_path):
    pytest.importorskip("rtree")
    ref = _box(tmp_path, "ref", (40, 20, 10))
    moved = _box(tmp_path, "moved", (40, 20, 10), at=(5, 0, 0))
    s = score(moved, ref)
    assert s["d99_surf"] < 0.1
    assert s["d99_surf_placed"] > 4
