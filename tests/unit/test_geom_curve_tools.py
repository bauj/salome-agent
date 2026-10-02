"""make_ellipse / make_polyline / make_curve / make_profile_face."""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from executor import geom as geom_tools


def _geompy(coords=None):
    """A fake geompy whose MakeVertex makes ("V", x, y, z) points; `coords`
    gives the coordinates of named point objects."""
    coords = dict(coords or {})
    g = MagicMock()
    g.ShapeType = {"EDGE": 6, "VERTEX": 7}
    g.MakeVertex.side_effect = lambda x, y, z: ("V", x, y, z)
    g.MakeVectorDXDYDZ.side_effect = lambda x, y, z: ("D", x, y, z)
    g.PointCoordinates.side_effect = lambda p: p[1:] if isinstance(p, tuple) and p[0] == "V" else coords[p]
    g.MakeVertexOnCurve.side_effect = lambda edge, t: ("V", float(t), 0.0, 0.0)
    g.SubShapeAll.return_value = []
    g.BasicProperties.return_value = (12.5, 40.0, 0.0)
    g.CheckShape.return_value = (True, "")
    g.KindOfShape.return_value = ["PLANAR", 0, 0, 0]
    g.MakeFaceWires.return_value = "FACE"
    return g


def _executor(geompy, bindings=None):
    geom_tools.reset_counters()
    return SimpleNamespace(name_prefix="", namespace={"geompy": geompy},
                           geom_bindings=dict(bindings or {}))


def _fns(result):
    return [r["fn"] for r in result["records"]]


class TestMakeEllipse:
    def test_full_ellipse(self):
        g = _geompy({"C": (0, 0, 0)})
        g.MakeEllipse.return_value = "ELL"
        ex = _executor(g, {"c": "C"})
        r = geom_tools.tool_make_ellipse(ex, {"center": "c", "normal": "0,0,1",
                                              "r_major": 10, "r_minor": 5})
        assert r["ok"] is True, r.get("error")
        g.MakeEllipse.assert_called_once_with("C", ("D", 0.0, 0.0, 1.0), 10, 5)
        assert r["records"][-1]["fn"] == "MakeEllipse"
        assert ex.geom_bindings["ellipse_1"] == "ELL"
        assert r["ellipse"] == {"length": 12.5}

    def test_major_dir(self):
        g = _geompy({"C": (0, 0, 0)})
        ex = _executor(g, {"c": "C"})
        r = geom_tools.tool_make_ellipse(ex, {"center": "c", "normal": "0,0,1", "r_major": 10,
                                              "r_minor": 5, "major_dir": "0,1,0"})
        assert r["ok"] is True
        assert g.MakeEllipse.call_args.args[-1] == ("D", 0.0, 1.0, 0.0)

    def test_minor_bigger_than_major(self):
        g = _geompy({"C": (0, 0, 0)})
        r = geom_tools.tool_make_ellipse(_executor(g, {"c": "C"}), {
            "center": "c", "normal": "0,0,1", "r_major": 5, "r_minor": 10})
        assert r["ok"] is False and "swap" in r["error"]
        g.MakeEllipse.assert_not_called()

    def test_quarter_ellipse(self):
        g = _geompy()
        r = geom_tools.tool_make_ellipse(_executor(g), {"center": "0,0,0", "start": "10,0,0",
                                                        "end": "0,5,0"})
        assert r["ok"] is True, r.get("error")
        g.MakeArcOfEllipse.assert_called_once_with(("V", 0.0, 0.0, 0.0), ("V", 10.0, 0.0, 0.0),
                                                   ("V", 0.0, 5.0, 0.0))
        assert set(r["ellipse"]) == {"start", "end", "length"}

    def test_quarter_needs_a_right_angle(self):
        g = _geompy()
        r = geom_tools.tool_make_ellipse(_executor(g), {"center": "0,0,0", "start": "10,0,0",
                                                        "end": "5,5,0"})
        assert r["ok"] is False and "45 degrees" in r["error"]

    @pytest.mark.parametrize("args", [
        {"center": "0,0,0"},
        {"center": "0,0,0", "normal": "0,0,1", "r_major": 2},
        {"center": "0,0,0", "normal": "0,0,1", "r_major": 2, "r_minor": 1, "start": "1,0,0"},
    ])
    def test_needs_exactly_one_form(self, args):
        r = geom_tools.tool_make_ellipse(_executor(_geompy()), args)
        assert r["ok"] is False and "either" in r["error"]


class TestMakePolyline:
    def test_closed(self):
        g = _geompy()
        r = geom_tools.tool_make_polyline(_executor(g), {"points": ["0,0,0", "1,0,0", "1,1,0"],
                                                         "closed": True})
        assert r["ok"] is True, r.get("error")
        assert g.MakePolyline.call_args.args[1] is True
        rec = r["records"][-1]
        assert rec["fn"] == "MakePolyline" and rec["publish"] == ["polyline_1"]
        assert r["length"] == 12.5

    def test_repeated_point(self):
        r = geom_tools.tool_make_polyline(_executor(_geompy()), {"points": ["0,0,0", "0,0,0"]})
        assert r["ok"] is False and "same point" in r["error"]


class TestMakeCurve:
    def test_interpolation_default(self):
        g = _geompy()
        r = geom_tools.tool_make_curve(_executor(g), {"points": ["0,0,0", "1,1,0", "2,0,0"]})
        assert r["ok"] is True, r.get("error")
        assert g.MakeInterpol.call_args.args[1:] == (False, False)
        assert r["records"][-1]["fn"] == "MakeInterpol"
        assert r["curve"]["start"] == [0, 0, 0] and r["curve"]["end"] == [1, 0, 0]

    def test_bezier(self):
        g = _geompy()
        r = geom_tools.tool_make_curve(_executor(g), {"points": ["0,0,0", "1,1,0", "2,0,0"],
                                                      "kind": "bezier", "closed": True})
        assert r["ok"] is True
        assert g.MakeBezier.call_args.args[1] is True

    def test_tangents(self):
        g = _geompy()
        r = geom_tools.tool_make_curve(_executor(g), {"points": ["0,0,0", "2,0,0"],
                                                      "start_tangent": "0,1,0",
                                                      "end_tangent": "0,-1,0"})
        assert r["ok"] is True, r.get("error")
        assert g.MakeInterpolWithTangents.call_args.args[1:] == (("D", 0.0, 1.0, 0.0),
                                                                 ("D", 0.0, -1.0, 0.0))

    @pytest.mark.parametrize("extra", [
        {"start_tangent": "0,1,0"},
        {"start_tangent": "0,1,0", "end_tangent": "0,1,0", "kind": "bezier"},
        {"start_tangent": "0,1,0", "end_tangent": "0,1,0", "closed": True},
    ])
    def test_tangent_misuse(self, extra):
        r = geom_tools.tool_make_curve(_executor(_geompy()), {"points": ["0,0,0", "2,0,0"], **extra})
        assert r["ok"] is False and "tangent" in r["error"]


class TestMakeProfileFace:
    def _run(self, args, g=None):
        g = g or _geompy()
        ex = _executor(g)
        return g, ex, geom_tools.tool_make_profile_face(ex, args)

    def test_slot_reuses_its_corner_vertices(self):
        g, ex, r = self._run({"start": "0,-5,0", "segments": [
            {"to": "20,-5,0"}, {"to": "20,5,0", "through": "25,0,0"},
            {"to": "0,5,0"}, {"to": "0,-5,0", "through": "-5,0,0"}]})
        assert r["ok"] is True, r.get("error")
        assert _fns(r).count("MakeVertex") == 6
        assert [f for f in _fns(r) if f != "MakeVertex"] == [
            "MakeLineTwoPnt", "MakeArc", "MakeLineTwoPnt", "MakeArc", "MakeWire", "MakeFaceWires"]
        assert "closed_with_line" not in r
        assert r["records"][-1]["publish"] == ["face_1"]
        assert all(rec["publish"] is None for rec in r["records"][:-1])
        assert r["area"] == 40.0

    def test_open_outline_is_closed_with_a_line(self):
        _, _, r = self._run({"start": "0,0,0", "segments": [{"to": "10,0,0"}, {"to": "0,10,0"}]})
        assert r["ok"] is True
        assert _fns(r).count("MakeLineTwoPnt") == 3
        assert r["closed_with_line"] == ["outline"]

    def test_centre_arc_snaps_a_nearly_right_end(self):
        g, _, r = self._run({"start": "10,0,0", "segments": [
            {"to": "0,10.001,0", "center": "0,0,0"}, {"to": "0,0,0"}]})
        assert r["ok"] is True, r.get("error")
        g.MakeArcCenter.assert_called_once()
        assert g.MakeArcCenter.call_args.args[2] == ("V", 0.0, 10.0, 0.0)

    def test_centre_arc_with_a_wrong_end_names_the_right_one(self):
        g, _, r = self._run({"start": "10,0,0", "segments": [
            {"to": "0,12,0", "center": "0,0,0"}]})
        assert r["ok"] is False
        assert "segment 1" in r["error"] and "radius" in r["error"] and "[0.0, 10.0, 0.0]" in r["error"]
        g.MakeArcCenter.assert_not_called()

    def test_half_circle_by_centre_asks_for_through(self):
        _, _, r = self._run({"start": "10,0,0", "segments": [
            {"to": "-10,0,0", "center": "0,0,0"}]})
        assert r["ok"] is False and "through" in r["error"]

    def test_via_makes_a_spline(self):
        g, _, r = self._run({"start": "0,0,0", "segments": [
            {"to": "10,0,0", "via": ["5,2,0"]}, {"to": "0,0,0"}]})
        assert r["ok"] is True, r.get("error")
        rec = next(x for x in r["records"] if x["fn"] == "MakeInterpol")
        assert len(rec["args"][0]["value"]) == 3

    def test_round_hole(self):
        g, _, r = self._run({"start": "0,0,0", "segments": [
            {"to": "10,0,0"}, {"to": "10,10,0"}, {"to": "0,10,0"}],
            "holes": [{"center": "5,5,0", "radius": 2}]})
        assert r["ok"] is True, r.get("error")
        normal = g.MakeCircle.call_args.args[1]
        assert normal[:3] == ("D", 0.0, 0.0) and abs(normal[3]) == 1.0
        assert g.MakeCircle.call_args.args[2] == 2.0
        assert len(g.MakeFaceWires.call_args.args[0]) == 2

    def test_outline_hole(self):
        g, _, r = self._run({"start": "0,0,0", "segments": [
            {"to": "10,0,0"}, {"to": "10,10,0"}, {"to": "0,10,0"}],
            "holes": [{"start": "2,2,0", "segments": [{"to": "4,2,0"}, {"to": "4,4,0"}]}]})
        assert r["ok"] is True, r.get("error")
        assert r["closed_with_line"] == ["outline", "hole 1"]
        assert _fns(r).count("MakeWire") == 2

    def test_two_arc_forms_is_an_error(self):
        _, _, r = self._run({"start": "10,0,0", "segments": [
            {"to": "0,10,0", "center": "0,0,0", "through": "7,7,0"}]})
        assert r["ok"] is False and "at most one" in r["error"]

    def test_curved_face_is_refused(self):
        g = _geompy()
        g.KindOfShape.return_value = ["FACE", 0]
        _, _, r = self._run({"start": "0,0,0", "segments": [{"to": "10,0,0"}, {"to": "0,10,5"}]}, g)
        assert r["ok"] is False and "one plane" in r["error"]

    SQUARE = [{"to": "10,0,0"}, {"to": "10,10,0"}, {"to": "0,10,0"}, {"to": "0,0,0"}]

    def _fillet_geompy(self):
        """Wire vertices = the square's corners; MakeFillet1D returns a new wire
        that keeps them (enough for the id lookup of a second radius)."""
        g = _geompy()
        corners = [("V", 0.0, 0.0, 0.0), ("V", 10.0, 0.0, 0.0),
                   ("V", 10.0, 10.0, 0.0), ("V", 0.0, 10.0, 0.0)]
        g.SubShapeAll.side_effect = lambda shape, kind: corners if kind == 7 else []
        g.GetSubShapeID.side_effect = lambda shape, v: 10 + corners.index(v)
        g.MakeWire.return_value = "WIRE"
        g.MakeFillet1D.side_effect = lambda wire, r, ids: f"W{r}"
        return g

    def test_fillet_rounds_the_corner_at_to(self):
        segs = [dict(s) for s in self.SQUARE]
        segs[0]["fillet"] = 2
        g, ex, r = self._run({"start": "0,0,0", "segments": segs}, self._fillet_geompy())
        assert r["ok"] is True, r.get("error")
        g.MakeFillet1D.assert_called_once_with("WIRE", 2.0, [11])
        assert g.MakeFaceWires.call_args.args[0] == ["W2.0"]
        assert _fns(r)[-3:] == ["MakeWire", "MakeFillet1D", "MakeFaceWires"]
        assert r["records"][-1]["publish"] == ["face_1"]

    def test_fillets_grouped_by_radius(self):
        segs = [dict(s) for s in self.SQUARE]
        segs[0]["fillet"], segs[1]["fillet"], segs[3]["fillet"] = 2, 3, 2
        g, _, r = self._run({"start": "0,0,0", "segments": segs}, self._fillet_geompy())
        assert r["ok"] is True, r.get("error")
        calls = [c.args for c in g.MakeFillet1D.call_args_list]
        assert calls == [("WIRE", 2.0, [11, 10]), ("W2.0", 3.0, [12])]

    def test_fillet_in_a_hole_rounds_only_that_wire(self):
        g = self._fillet_geompy()
        _, _, r = self._run({"start": "-5,-5,0", "segments": [
            {"to": "15,-5,0"}, {"to": "15,15,0"}, {"to": "-5,15,0"}],
            "holes": [{"start": "0,0,0", "segments": [
                {"to": "10,0,0"}, {"to": "10,10,0", "fillet": 1}, {"to": "0,10,0"}]}]}, g)
        assert r["ok"] is True, r.get("error")
        g.MakeFillet1D.assert_called_once_with("WIRE", 1.0, [12])
        assert g.MakeFaceWires.call_args.args[0] == ["WIRE", "W1.0"]

    @pytest.mark.parametrize("bad", [0, -1, "big"])
    def test_bad_fillet_radius(self, bad):
        segs = [dict(s) for s in self.SQUARE]
        segs[0]["fillet"] = bad
        _, _, r = self._run({"start": "0,0,0", "segments": segs}, self._fillet_geompy())
        assert r["ok"] is False and "segment 1" in r["error"] and "fillet" in r["error"]

    def test_fillet_occ_failure_names_the_corner(self):
        g = self._fillet_geompy()
        g.MakeFillet1D.side_effect = RuntimeError("Fillet1D failed")
        segs = [dict(s) for s in self.SQUARE]
        segs[1]["fillet"] = 20
        _, _, r = self._run({"start": "0,0,0", "segments": segs}, g)
        assert r["ok"] is False
        assert "[10.0, 10.0, 0.0]" in r["error"] and "20" in r["error"]

    def test_fillet_codegen(self):
        from salome_agent.codegen import generate
        segs = [dict(s) for s in self.SQUARE]
        segs[0]["fillet"] = 2
        _, _, r = self._run({"start": "0,0,0", "segments": segs}, self._fillet_geompy())
        code = generate(r["records"], ["face_1"])
        assert "wire_2 = geompy.MakeFillet1D(wire_1, 2.0, [11])" in code
        assert "face_1 = geompy.MakeFaceWires([wire_2], 1)" in code

    def test_codegen(self):
        from salome_agent.codegen import generate
        _, _, r = self._run({"start": "0,0,0", "segments": [{"to": "10,0,0"}, {"to": "0,10,0"}]})
        code = generate(r["records"], ["face_1"])
        assert "geompy.MakeWire([edge_1, edge_2, edge_3], 1e-07)" in code
        assert code.endswith("face_1 = geompy.MakeFaceWires([wire_1], 1)\n"
                             "geompy.addToStudy(face_1, 'face_1')")


def test_profile_face_comes_with_the_profile_tools():
    from salome_agent.specialists.tool_selection import initial_offered

    assert "make_profile_face" in initial_offered({"tools": ["make_arc"]}, set(geom_tools.TOOLS))
