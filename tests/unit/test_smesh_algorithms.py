"""The meshers, hypotheses and tools added from the StdMeshers, NETGEN and GMSH
builders: the code each one records, and what it refuses before running."""
import pytest

from executor.smesh import editing as ed
from executor.smesh import groups as gr
from executor.smesh import local as lo
from executor.smesh import setup as su
from executor.smesh import state as sm


def test_gmsh_tetra_and_surface_code():
    t = su.tetra_code("m", {"max_size": 5, "min_size": 1, "mesher": "gmsh",
                            "second_order": True})
    assert t.splitlines() == ["m_a3 = m.Tetrahedron(algo=smeshBuilder.GMSH)",
                              "m_gmsh = m_a3.Parameters()", "m_gmsh.SetMaxSize(5)",
                              "m_gmsh.SetMinSize(1)", "m_gmsh.SetSecondOrder(True)"]
    s = su.surface_code("m", {"max_size": 2, "mesher": "gmsh", "quads": True})
    assert "m.Triangle(algo=smeshBuilder.GMSH_2D)" in s and "SetRecombineAll(True)" in s
    assert "NETGEN_1D2D3D" in su.tetra_code("m", {"max_size": 5, "mesher": "netgen"})


@pytest.mark.parametrize("args,match", [
    ({"max_size": 5, "mesher": "tetgen"}, "mesher"),
    ({"max_size": 5, "mesher": "gmsh", "fineness": "fine"}, "NETGEN options"),
    ({"max_size": 5, "mesher": "gmsh", "growth_rate": 0.3}, "NETGEN options"),
])
def test_gmsh_refuses_netgen_options(args, match):
    with pytest.raises(ValueError, match=match):
        su.tetra_code("m", args)


def test_new_algorithms_are_listed_with_their_dimension():
    assert su.ALGORITHMS["gmsh_1d2d3d"] == (3, "Tetrahedron", "smeshBuilder.GMSH")
    assert su.ALGORITHMS["gmsh_2d"][0] == 2 and su.ALGORITHMS["gmsh_3d"][0] == 3
    assert (su.algorithm_code("v", "m", "gmsh_2d", "f")
            == "v = m.Triangle(algo=smeshBuilder.GMSH_2D, geom=f)")


@pytest.mark.parametrize("kind,params,code", [
    ("max_size", {"length": 2}, "a.MaxSize(2)"),
    ("adaptive", {"min": 0.5, "max": 3, "deflection": 0.1}, "a.Adaptive(0.5, 3, 0.1)"),
    ("deflection", {"deflection": 0.05}, "a.Deflection1D(0.05)"),
    ("automatic_length", {"fineness": 0}, "a.AutomaticLength(0)"),
    ("fixed_points", {"points": [0.2, 0.5]}, "a.FixedPoints1D([0.2, 0.5])"),
    ("fixed_points", {"points": [0.5], "segments": [2, 4]}, "a.FixedPoints1D([0.5], [2, 4])"),
    ("propagation_of_distribution", {}, "a.PropagationOfDistribution()"),
    ("number_of_layers", {"count": 4}, "a.NumberOfLayers(4)"),
])
def test_new_hypothesis_code(kind, params, code):
    assert su.hypothesis_code("a", kind, params) == code


@pytest.mark.parametrize("kind,params,match", [
    ("adaptive", {"min": 3, "max": 1, "deflection": 0.1}, "min below max"),
    ("automatic_length", {"fineness": 1.5}, "fineness"),
    ("fixed_points", {"points": [0.5, 0.2]}, "increasing"),
    ("fixed_points", {"points": [0, 0.5]}, "strictly between"),
    ("fixed_points", {"points": [0.5], "segments": [2]}, "2 whole numbers"),
    ("number_of_layers", {"count": 0}, "count"),
    ("number_of_layers", {}, "needs count"),
])
def test_new_hypothesis_code_refuses(kind, params, match):
    with pytest.raises(ValueError, match=match):
        su.hypothesis_code("a", kind, params)


def test_reversed_edges_go_by_keyword_and_only_on_graded_rules():
    assert (su.hypothesis_code("a", "arithmetic_progression", {"start": 1, "end": 3},
                               ["e1", '_shaper_shape("e2")'])
            == 'a.Arithmetic1D(1, 3, reversedEdges=_ids(e1) + _ids(_shaper_shape("e2")))')
    assert (su.hypothesis_code("a", "number_of_segments", {"count": 8, "ratio": 3}, ["e"])
            == "a.NumberOfSegments(8, 3, reversedEdges=_ids(e))")
    with pytest.raises(ValueError, match="no reversed edges"):
        su.hypothesis_code("a", "local_length", {"length": 1}, ["e"])


def test_netgen_and_gmsh_parameters_code():
    code = su.hypothesis_code("m_a2", "netgen_parameters",
                              {"max_size": 4, "fineness": "fine", "chordal_error": 0.1,
                               "quads": True})
    assert code.splitlines() == ["m_a2_p = m_a2.Parameters()", "m_a2_p.SetMaxSize(4)",
                                 "m_a2_p.SetFineness(smeshBuilder.Fine)",
                                 "m_a2_p.SetChordalErrorEnabled(True)",
                                 "m_a2_p.SetChordalError(0.1)", "m_a2_p.SetQuadAllowed(True)"]
    code = su.hypothesis_code("v", "gmsh_parameters",
                              {"max_size": 3, "algorithm_2d": "frontal",
                               "algorithm_3d": "hxt", "quads": False})
    assert "v_p.Set2DAlgo(3)" in code and "v_p.Set3DAlgo(4)" in code
    assert "v_p.SetRecombineAll(False)" in code
    with pytest.raises(ValueError, match="not growth_rate"):
        su.hypothesis_code("v", "gmsh_parameters", {"growth_rate": 0.3})
    with pytest.raises(ValueError, match="algorithm_2d"):
        su.hypothesis_code("v", "gmsh_parameters", {"algorithm_2d": "mefisto"})
    with pytest.raises(ValueError, match="takes some of"):
        su.hypothesis_code("v", "netgen_parameters", {})


def test_prism_with_quads_allows_them_on_the_end_face():
    code = su.prism_code("m", "base", {"layers": 3, "face_size": 2, "quads": True})
    assert code.endswith("m_a2_end.SetQuadAllowed(True)")


class _Algo:
    def __init__(self, name):
        self._name = name

    def GetAlgorithm(self):
        return self

    def GetName(self):
        return self._name


def _ex(monkeypatch, ns=None):
    for mod in (su, lo, gr):
        monkeypatch.setattr(mod, "run", lambda st, code: None)
    ex = type("Ex", (), {})()
    st = ex.__dict__["smesh_state"] = sm.SmeshState(dict(ns or {}))
    st.meshes["m"] = sm.MeshInfo("plate_1", "shaper")
    for g, t in (("top", "face"), ("bottom", "face"), ("rim", "edge"), ("side", "face")):
        st.geo_groups[g] = sm.GeoGroup("plate_1", "shaper", t)
    return ex, st


def test_layers_go_on_the_radial_algorithm_of_either_dimension(monkeypatch):
    ex, st = _ex(monkeypatch, {"m_a2": _Algo("RadialQuadrangle_1D2D")})
    su.tool_set_algorithm(ex, {"mesh": "m", "dimension": 2, "algorithm": "radial_quadrangle"})
    r = su.tool_set_hypothesis(ex, {"mesh": "m", "kind": "number_of_layers",
                                    "params": {"count": 5}})
    assert r["records"][0]["code"] == "m_a2.NumberOfLayers(5)"


def test_layers_on_a_plain_prism_are_refused(monkeypatch):
    ex, st = _ex(monkeypatch, {"m_a3": _Algo("Prism_3D")})
    su.tool_set_algorithm(ex, {"mesh": "m", "dimension": 3, "algorithm": "prism"})
    with pytest.raises(ValueError, match="does not apply to the Prism_3D"):
        su.tool_set_hypothesis(ex, {"mesh": "m", "kind": "number_of_layers",
                                    "params": {"count": 5}})


def test_reversed_edges_are_refs_of_the_record(monkeypatch):
    ex, st = _ex(monkeypatch)
    su.tool_set_algorithm(ex, {"mesh": "m", "dimension": 1, "algorithm": "segment"})
    r = su.tool_set_hypothesis(ex, {"mesh": "m", "kind": "geometric_progression",
                                    "params": {"start": 1, "ratio": 1.2, "reversed": ["rim"]}})
    assert r["records"][0]["code"] == (
        'm_a1.GeometricProgression(1, 1.2, reversedEdges=_ids(_shaper_shape("rim")))')
    assert "rim" in str(r["records"][0])


def test_project_mesh_faces_splits_a_netgen_tetra_first(monkeypatch):
    class P:
        def GetMaxSize(self): return 5.0
        def GetMinSize(self): return 0.0
        def GetFineness(self): return 2
        def GetGrowthRate(self): return 0.3
    ex, st = _ex(monkeypatch, {"m_netgen": P()})
    su.tool_mesh_tetra(ex, {"mesh": "m", "max_size": 5})
    lo.tool_local_size(ex, {"mesh": "m", "on": ["side"], "size": 1})
    r = lo.tool_project_mesh(ex, {"mesh": "m", "source": "bottom", "on": ["top"]})
    code = r["records"][0]["code"]
    assert code.startswith("m.RemoveHypothesis(m_a3)")
    assert 'm_netgen2.SetLocalSizeOnShape(_shaper_shape("side"), 1)' in code
    assert code.endswith('m_proj_top = m.Projection1D2D(geom=_shaper_shape("top"))\n'
                         'm_proj_top.SourceFace(_shaper_shape("bottom"))')
    info = st.meshes["m"]
    assert info.algos[(2, "top")] == "m_proj_top" and info.submeshes["top"] == "m_proj_top"
    # split once: a second projection adds only its own lines
    r = lo.tool_project_mesh(ex, {"mesh": "m", "source": "bottom", "on": ["side"]})
    assert "RemoveHypothesis" not in r["records"][0]["code"]


def test_project_mesh_refusals(monkeypatch):
    ex, st = _ex(monkeypatch)
    with pytest.raises(ValueError, match="no setup"):
        lo.tool_project_mesh(ex, {"mesh": "m", "source": "bottom", "on": ["top"]})
    su.tool_mesh_tetra(ex, {"mesh": "m", "max_size": 5, "mesher": "gmsh"})
    with pytest.raises(ValueError, match="mesher netgen"):
        lo.tool_project_mesh(ex, {"mesh": "m", "source": "bottom", "on": ["top"]})
    su.tool_mesh_hexa(ex, {"mesh": "m", "segments": 3})
    with pytest.raises(ValueError, match="must be face groups"):
        lo.tool_project_mesh(ex, {"mesh": "m", "source": "bottom", "on": ["rim"]})
    with pytest.raises(ValueError, match="onto itself"):
        lo.tool_project_mesh(ex, {"mesh": "m", "source": "bottom", "on": ["bottom"]})


def test_edge_projection_code():
    assert (lo.projection_code("v", "m", "e2", "edge", "e1")
            == "v = m.Projection1D(geom=e2)\nv.SourceEdge(e1)")


def test_combine_code_union_intersect_cut():
    u = gr.combine_code("m_all", "m", "all", "FACE", "union", ["m_a", "m_b"], [])
    assert u.startswith("m_all = m.GroupOnFilter(SMESH.FACE, 'all', smesh.GetFilterFromCriteria([")
    assert u.count("FT_LogicalOR") == 1 and u.count("BinaryOp=SMESH.FT_Undefined") == 1
    i = gr.combine_code("v", "m", "x", "FACE", "intersect", ["m_a", "m_b", "m_c"], [])
    assert i.count("FT_LogicalAND") == 2
    c = gr.combine_code("v", "m", "x", "FACE", "cut", ["m_a", "m_b"], ["m_c"])
    crits = [line for line in c.splitlines() if "GetCriterion" in line]
    # (a AND NOT c) OR (b AND NOT c)
    assert ["m_a," in crits[0], "m_c, UnaryOp" in crits[1], "m_b," in crits[2],
            "m_c, UnaryOp" in crits[3]] == [True] * 4
    assert [x.split("BinaryOp=SMESH.")[1].rstrip("),]") for x in crits] == [
        "FT_LogicalAND", "FT_LogicalOR", "FT_LogicalAND", "FT_Undefined"]


class _Group:
    def __init__(self, kind, size=3):
        self.kind, self.size = kind, size

    def GetType(self):
        return self.kind

    def Size(self):
        return self.size


def test_combine_groups_refusals_and_record(monkeypatch):
    ex, st = _ex(monkeypatch, {"m_top": _Group("FACE"), "m_bottom": _Group("FACE"),
                               "m_rim": _Group("EDGE"), "m_both": _Group("FACE", 6)})
    info = st.meshes["m"]
    info.groups.update({"top": "top", "bottom": "bottom", "rim": "rim"})
    def call(**kw):
        return gr.tool_combine_groups(ex, {"mesh": "m", **kw})
    with pytest.raises(ValueError, match="two or more"):
        call(name="u", operation="union", groups=["top"])
    with pytest.raises(ValueError, match="needs the groups to remove"):
        call(name="u", operation="cut", groups=["top"])
    with pytest.raises(ValueError, match="mix element types"):
        call(name="u", operation="union", groups=["top", "rim"])
    with pytest.raises(ValueError, match="no mesh group nope"):
        call(name="u", operation="union", groups=["top", "nope"])
    with pytest.raises(ValueError, match="already has"):
        call(name="top", operation="union", groups=["top", "bottom"])
    r = call(name="both", operation="union", groups=["top", "bottom"])
    assert r["size"] == 6 and info.groups["both"] is None
    assert info.combined["both"] == ["top", "bottom"]


def test_forgetting_a_group_removes_the_combined_groups_made_from_it():
    removed = []

    class Mesh:
        def RemoveGroup(self, g):
            removed.append(g)
    ex = type("Ex", (), {})()
    st = ex.__dict__["smesh_state"] = sm.SmeshState(
        {"m": Mesh(), "m_top": "G_top", "m_both": "G_both", "m_more": "G_more",
         "m_bottom": "G_bottom"})
    info = st.meshes["m"] = sm.MeshInfo("plate_1", "shaper")
    info.groups.update({"top": "top", "bottom": "bottom", "both": None, "more": None})
    info.combined.update({"both": ["top", "bottom"], "more": ["both", "bottom"]})
    ed._remove_mesh_group(st, "m", info, "top")
    assert removed == ["G_top", "G_both", "G_more"]
    assert set(info.groups) == {"bottom"} and info.combined == {}


def test_remove_hypothesis_matches_smesh_names():
    assert ed.hypothesis_names("max_size") == ("MaxLength",)
    assert ed.hypothesis_names("quadrangle_parameters") == ("QuadrangleParams",)
    assert ed.hypothesis_names("local_length") == ("LocalLength",)
    assert "NumberOfLayers2D".startswith(ed.hypothesis_names("number_of_layers"))


def test_every_hypothesis_kind_can_be_removed():
    import executor.smesh.setup as s
    for kind in s.HYPOTHESES:
        assert ed.hypothesis_names(kind)
