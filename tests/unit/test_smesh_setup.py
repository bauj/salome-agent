import pytest

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
