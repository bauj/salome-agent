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


def test_count_trusts_evaluate_when_it_counts_the_mesh_dimension():
    assert c.count({"Edge": 340, "Quadrangle": 4000, "Hexa": 12500}, 3, None, None) == 16840
    assert c.count({"Edge": 10, "Triangle": 50}, 2, None, None) == 60


def test_count_is_none_when_evaluate_misses_the_mesh_dimension():
    assert c.count({"Edge": 340}, 2, None, None) is None
    assert c.count({"Edge": 12, "Quadrangle": 6}, 3, None, None) is None
    assert c.count({}, 3, None, None) is None


def test_count_grid_rule_counts_cells_of_the_bounding_box():
    bbox = [0.0, 100.0, 0.0, 50.0, 0.0, 20.0]
    assert c.count({}, 3, ("grid", 2.0), bbox) == 12500
    assert c.count({}, 3, ("grid", 3.0), bbox) == 34 * 17 * 7
    assert c.count({}, 3, ("grid", 2.0), [0.0, 10.0, 0.0, 10.0, 5.0, 5.0]) == 25


def test_count_layers_rule_sweeps_the_end_face():
    assert c.count({"Edge": 72, "Triangle": 188}, 3, ("layers", 20), None) == 260 + 188 * 20


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


def _fake_smesh_executor():
    import types
    from executor.smesh import state as sm
    ex = types.SimpleNamespace()
    removed = []
    ns = {"smesh": types.SimpleNamespace(RemoveMesh=removed.append),
          "salome": types.SimpleNamespace(myStudy=None),
          "mesh_1": "mesh", "mesh_1_a3": "algo", "mesh_1_netgen": "params",
          "mesh_1_lid": "group", "mesh_10": "other mesh", "mesh_1_keep": "geo group"}
    st = ex.__dict__["smesh_state"] = sm.SmeshState(ns)
    st.meshes["mesh_1"] = sm.MeshInfo("box_1", "geom")
    st.meshes["mesh_10"] = sm.MeshInfo("box_1", "geom")
    st.geo_groups["mesh_1_keep"] = sm.GeoGroup("box_1", "geom", "face")
    return ex, st, removed


def test_forget_a_mesh_drops_its_variables():
    from executor.smesh.editing import forget
    ex, st, removed = _fake_smesh_executor()
    assert forget(ex, ["mesh_1"]) == ["mesh_1"]
    assert removed == ["mesh"]
    assert set(st.ns) == {"smesh", "salome", "mesh_10", "mesh_1_keep"}


def test_export_reply_says_where_result_py_writes(monkeypatch, tmp_path):
    from executor.smesh import editing
    ex, st, _ = _fake_smesh_executor()
    ex.run_dir = str(tmp_path)
    monkeypatch.setattr(editing, "run", lambda st, code: None)
    r = editing.tool_export_mesh(ex, {"mesh": "mesh_1", "format": "med"})
    assert r["path"] == str(tmp_path / "mesh_1.med")
    assert r["records"][0]["code"] == "mesh_1.ExportMED('mesh_1.med')"
    assert "result.py" in r["note"] and "mesh_1.med" in r["note"]
