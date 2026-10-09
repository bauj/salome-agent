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


def _m(**kw):
    base = {"nodes": 10, "edges": 0, "faces": 20, "volumes": 30, "volume": 6000.0,
            "shape_volume": 6000.0, "area": 2200.0, "shape_area": 2200.0, "errors": [],
            "groups": {"inlet": 12}}
    base.update(kw)
    return base


def test_mesh_status_ok():
    assert c.mesh_status(_m(), 3) == {"status": "ok"}


@pytest.mark.parametrize("kw,dim,word", [
    ({"nodes": 0, "faces": 0, "volumes": 0}, 3, "not computed"),
    ({"errors": ["Quadrangle_2D failed ..."]}, 3, "meshing errors"),
    ({"volumes": 0}, 3, "no volume elements"),
    ({"faces": 0}, 2, "no face elements"),
    ({"groups": {"inlet": 0}}, 3, "inlet is empty"),
    ({"volume": 5900.0}, 3, "1.67%"),
])
def test_mesh_status_refusals(kw, dim, word):
    r = c.mesh_status(_m(**kw), dim)
    assert r["status"] == "not_meshed" and word in r["detail"]


def test_check_line_exists():
    from salome_agent.specialists.tool_specialist import _CHECK_LINES
    assert "{detail}" in _CHECK_LINES["not_meshed"]


def _no_copy(*a):
    raise AssertionError("geom_copy called")


def test_failures_make_no_copy_when_nothing_failed(monkeypatch):
    import types
    monkeypatch.setattr(c, "geom_copy", _no_copy)
    st = types.SimpleNamespace(ns={"geompy": None})
    m = types.SimpleNamespace(GetComputeErrors=lambda: [], GetShape=lambda: "shaperstudy object")
    assert c.failures(st, None, m) == []


def test_shape_props_of_a_shaper_shape_measure_it_in_shaper(monkeypatch):
    import types
    from executor.smesh import state as sm
    import executor.shaper.state as shs
    monkeypatch.setattr(c, "geom_copy", _no_copy)
    monkeypatch.setattr(shs, "measure", lambda ex, name: {
        "area": 2200.0, "volume": 6000.0, "faces": 6, "edges": 12, "bbox": []})
    monkeypatch.setattr(shs, "shape_type", lambda ex, name: "solid")
    monkeypatch.setattr(shs, "consumed", lambda ex, name: False)
    ex = types.SimpleNamespace(geom_bindings={})
    ex.__dict__["smesh_state"] = sm.SmeshState({})
    ex.__dict__["shaper_state"] = types.SimpleNamespace(objects={"box_1": "solid"}, groups={})
    assert c.shape_props(ex, "box_1") == {"type": "solid", "area": 2200.0,
                                          "volume": 6000.0, "faces": 6}
