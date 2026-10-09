import pytest

from executor.smesh import local as lo
from executor.smesh import setup as su


def test_tetra_code_defaults_and_options():
    code = su.tetra_code("mesh_1", {"max_size": 5})
    assert "mesh_1_a3 = mesh_1.Tetrahedron(algo=smeshBuilder.NETGEN_1D2D3D)" in code
    assert "mesh_1_netgen = mesh_1_a3.Parameters()" in code
    assert "mesh_1_netgen.SetMaxSize(5)" in code
    full = su.tetra_code("mesh_1", {"max_size": 5, "min_size": 1, "fineness": "fine",
                                    "growth_rate": 0.2, "second_order": True})
    assert "SetMinSize(1)" in full and "SetFineness(smeshBuilder.Fine)" in full
    assert "SetGrowthRate(0.2)" in full and "SetSecondOrder(True)" in full


@pytest.mark.parametrize("args", [{}, {"max_size": 0}, {"max_size": "5"},
                                  {"max_size": 5, "fineness": "medium"}])
def test_tetra_code_rejects(args):
    with pytest.raises(ValueError):
        su.tetra_code("mesh_1", args)


def test_surface_and_hexa_and_body_fitted():
    s = su.surface_code("m", {"max_size": 2, "quads": True})
    assert "m.Triangle(algo=smeshBuilder.NETGEN_1D2D)" in s and "SetQuadAllowed(True)" in s
    h = su.hexa_code("m", {"segments": 4})
    assert "m.Segment().NumberOfSegments(4)" in h and "m.Quadrangle()" in h
    assert "m.Hexahedron()" in h
    assert "LocalLength(2.5)" in su.hexa_code("m", {"segment_length": 2.5})
    with pytest.raises(ValueError):
        su.hexa_code("m", {"segments": 4, "segment_length": 2})
    assert "m.BodyFitted().SetGrid('3', '3', '3', 4.0)" in su.body_fitted_code("m", {"size": 3})


def test_algorithm_code_global_and_local():
    assert su.algorithm_code("m_a1", "m", "segment", None) == "m_a1 = m.Segment()"
    assert (su.algorithm_code("m_a2_top", "m", "quadrangle", '_shaper_shape("top")')
            == 'm_a2_top = m.Quadrangle(geom=_shaper_shape("top"))')
    assert "smeshBuilder.NETGEN_2D" in su.algorithm_code("v", "m", "netgen_2d", None)
    with pytest.raises(ValueError, match="algorithm"):
        su.algorithm_code("v", "m", "delaunay", None)


def test_hypothesis_code():
    assert su.hypothesis_code("a", "number_of_segments", {"count": 8}) == "a.NumberOfSegments(8)"
    assert (su.hypothesis_code("a", "number_of_segments", {"count": 8, "ratio": 3})
            == "a.NumberOfSegments(8, 3)")
    assert su.hypothesis_code("a", "local_length", {"length": 2}) == "a.LocalLength(2)"
    assert su.hypothesis_code("a", "propagation", {}) == "a.Propagation()"
    assert (su.hypothesis_code("a", "quadrangle_parameters", {"type": "reduced"})
            == "a.QuadrangleParameters(StdMeshers.QUAD_REDUCED)")
    with pytest.raises(ValueError, match="length"):
        su.hypothesis_code("a", "local_length", {})


def test_hypothesis_dimension():
    assert su.HYPOTHESES["local_length"][0] == 1
    assert su.HYPOTHESES["max_element_area"][0] == 2
    assert su.HYPOTHESES["max_element_volume"][0] == 3


def test_local_size_netgen_and_segment():
    assert (lo.local_size_code("m_netgen", "m", "bolt", '_shaper_shape("bolt")', 1)
            == 'm_netgen.SetLocalSizeOnShape(_shaper_shape("bolt"), 1)')
    assert (lo.local_size_code(None, "m", "bolt", "bolt", 1)
            == "m_a1_bolt = m.Segment(geom=bolt)\nm_a1_bolt.LocalLength(1)")


def test_edge_segments_code():
    code = lo.edge_segments_code("m_a1_e", "m", "e", {"count": 40, "propagate": True})
    assert code == "m_a1_e = m.Segment(geom=e)\nm_a1_e.NumberOfSegments(40)\nm_a1_e.Propagation()"
    assert "NumberOfSegments(10, 4)" in lo.edge_segments_code("v", "m", "e", {"count": 10, "ratio": 4})
    assert "LocalLength(2)" in lo.edge_segments_code("v", "m", "e", {"length": 2})
    with pytest.raises(ValueError):
        lo.edge_segments_code("v", "m", "e", {})


def test_viscous_code_3d_and_2d():
    code = lo.viscous_code("m_a3", 3, {"thickness": 1, "layers": 3, "stretch": 1.2,
                                       "faces_are": "without_layers"},
                           ['_shaper_shape("inlet")'])
    assert code == ('m_a3.ViscousLayers(1, 3, 1.2, '
                    '_ids(_shaper_shape("inlet")), True)')
    code2 = lo.viscous_code("m_a2", 2, {"thickness": 1, "layers": 3, "stretch": 1.2,
                                        "faces_are": "with_layers"}, ["w1", "w2"])
    assert code2 == "m_a2.ViscousLayers2D(1, 3, 1.2, _ids(w1) + _ids(w2), False)"


def test_prism_code():
    code = su.prism_code("m", '_shaper_shape("base")', {"layers": 10, "face_size": 2})
    assert code.splitlines() == [
        "m_a1 = m.Segment().NumberOfSegments(10)",
        "m_a3 = m.Prism()",
        'm_a1_end = m.Segment(geom=_shaper_shape("base"))',
        "m_a1_end.LocalLength(2)",
        'm_a2_end = m.Triangle(algo=smeshBuilder.NETGEN_2D, geom=_shaper_shape("base"))',
    ]


def test_a_new_netgen_preset_keeps_the_local_sizes(monkeypatch):
    from executor.smesh import state as sm
    monkeypatch.setattr(su, "run", lambda st, code: None)
    monkeypatch.setattr(lo, "run", lambda st, code: None)
    ex = type("Ex", (), {})()
    st = ex.__dict__["smesh_state"] = sm.SmeshState({})
    st.meshes["m"] = sm.MeshInfo("plate_1", "shaper")
    st.geo_groups["bolt"] = sm.GeoGroup("plate_1", "shaper", "face")
    su.tool_mesh_tetra(ex, {"mesh": "m", "max_size": 5})
    lo.tool_local_size(ex, {"mesh": "m", "on": ["bolt"], "size": 1})
    r = su.tool_mesh_tetra(ex, {"mesh": "m", "max_size": 8})
    assert r["records"][0]["code"].endswith('m_netgen.SetLocalSizeOnShape(_shaper_shape("bolt"), 1)')
    r = su.tool_mesh_hexa(ex, {"mesh": "m", "segments": 4})
    assert r["dropped_local_sizes"] == ["bolt"]
    assert st.meshes["m"].local_sizes == {}
