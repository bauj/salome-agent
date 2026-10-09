import pytest

from executor.smesh import compute as c
from executor.smesh import groups as gr


def test_volume_check():
    assert c.volume_off(6000.0, 6000.0) == 0.0
    assert c.volume_off(5900.0, 6000.0) > 0.01
    assert c.volume_off(10.0, 0.0) is None


def test_refusal_text_names_the_count_and_cap():
    msg = c.cap_refusal(2_500_000, 1_000_000)
    assert "2 500 000" in msg and "1 000 000" in msg and "evaluate" not in msg


def test_failure_line():
    f = {"algorithm": "Quadrangle_2D", "reason": "Face must have 4 sides but not 1",
         "type": "face", "centre": [0.0, 0.0, 10.0], "size": 78.54}
    assert c.failure_line(f) == ("Quadrangle_2D failed on a face at (0.0, 0.0, 10.0), "
                                 "size 78.54: Face must have 4 sides but not 1")


def test_quality_criteria_cover_the_tool_enum():
    assert set(c.QUALITY) == {"aspect_ratio", "minimum_angle", "scaled_jacobian", "skew",
                              "warping", "volume", "area"}


def test_group_code():
    assert (gr.group_code("m_inlet", "m", '_shaper_shape("inlet")', "inlet", "FACE")
            == 'm_inlet = m.GroupOnGeom(_shaper_shape("inlet"), \'inlet\', SMESH.FACE)')


def test_element_type_from_sub_type():
    assert gr.ELEMENT_TYPES["solid"] == "VOLUME"
    assert gr.element_for("face", None) == "FACE"
    assert gr.element_for("face", "node") == "NODE"
    with pytest.raises(ValueError):
        gr.element_for("face", "cells")
