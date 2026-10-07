"""make_arc / make_wire / make_face_from_wires: 2D profiles out of lines and arcs."""
import math
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from executor import geom as geom_tools
from executor.geom.sketch import _loose_ends


def _geompy(coords, edges=None, vertices=None):
    """A fake geompy: `coords` maps a point object to (x, y, z); `edges` maps a
    shape to its edge objects, `vertices` an edge to its vertex objects."""
    g = MagicMock()
    g.ShapeType = {"EDGE": 6, "VERTEX": 7}
    g.PointCoordinates.side_effect = lambda p: coords[p]
    g.MakeVertexOnCurve.side_effect = lambda edge, t: ("on", edge, t)
    g.BasicProperties.return_value = (15.7, 0.0, 0.0)
    g.CheckShape.return_value = (True, "")
    kinds = {6: edges or {}, 7: vertices or {}}
    g.SubShapeAll.side_effect = lambda shape, kind: kinds[kind][shape]
    return g


def _executor(geompy, bindings):
    geom_tools.reset_counters()
    return SimpleNamespace(name_prefix="", namespace={"geompy": geompy},
                           geom_bindings=dict(bindings))


class TestMakeArc:
    def _arc_executor(self, fn):
        coords = {"C": (0, 0, 0), "S": (10, 0, 0), "E": (0, 10, 0), "M": (7.07, 7.07, 0),
                  ("on", "ARC", 0.0): (10, 0, 0), ("on", "ARC", 1.0): (0, 10, 0)}
        g = _geompy(coords)
        getattr(g, fn).return_value = "ARC"
        return g, _executor(g, {"c": "C", "s": "S", "e": "E", "m": "M"})

    def test_center_form(self):
        g, ex = self._arc_executor("MakeArcCenter")
        r = geom_tools.tool_make_arc(ex, {"center": "c", "start": "s", "end": "e"})
        assert r["ok"] is True, r.get("error")
        g.MakeArcCenter.assert_called_once_with("C", "S", "E", False)
        rec = r["records"][-1]
        assert rec["fn"] == "MakeArcCenter"
        assert rec["args"] == [{"kind": "ref", "value": "c"}, {"kind": "ref", "value": "s"},
                               {"kind": "ref", "value": "e"}, {"kind": "literal", "value": False}]
        assert rec["publish"] == rec["out_vars"] == ["arc_1"]
        assert ex.geom_bindings["arc_1"] == "ARC"

    def test_center_form_reverse(self):
        g, ex = self._arc_executor("MakeArcCenter")
        r = geom_tools.tool_make_arc(ex, {"center": "c", "start": "s", "end": "e", "reverse": True})
        assert r["ok"] is True
        g.MakeArcCenter.assert_called_once_with("C", "S", "E", True)

    def test_through_form(self):
        g, ex = self._arc_executor("MakeArc")
        r = geom_tools.tool_make_arc(ex, {"start": "s", "through": "m", "end": "e"})
        assert r["ok"] is True, r.get("error")
        g.MakeArc.assert_called_once_with("S", "M", "E")
        assert r["records"][-1]["args"] == [{"kind": "ref", "value": v} for v in ("s", "m", "e")]

    def test_reports_real_end_points_and_length(self):
        _, ex = self._arc_executor("MakeArcCenter")
        r = geom_tools.tool_make_arc(ex, {"center": "c", "start": "s", "end": "e"})
        assert r["arc"] == {"start": [10, 0, 0], "end": [0, 10, 0], "length": 15.7}

    @pytest.mark.parametrize("args", [
        {"start": "s", "end": "e"},
        {"center": "c", "through": "m", "start": "s", "end": "e"},
    ])
    def test_needs_exactly_one_form(self, args):
        _, ex = self._arc_executor("MakeArc")
        r = geom_tools.tool_make_arc(ex, args)
        assert r["ok"] is False
        assert "either `center`" in r["error"]

    def test_through_form_collinear_says_so(self):
        g = _geompy({"A": (0, 0, 0), "B": (1, 1, 0), "D": (2, 2, 0)})
        ex = _executor(g, {"a": "A", "b": "B", "d": "D"})
        r = geom_tools.tool_make_arc(ex, {"start": "a", "through": "b", "end": "d"})
        assert r["ok"] is False and "collinear" in r["error"]
        g.MakeArc.assert_not_called()

    def test_center_form_zero_radius_says_so(self):
        g = _geompy({"C": (0, 0, 0), "E": (0, 10, 0)})
        ex = _executor(g, {"c": "C", "e": "E"})
        r = geom_tools.tool_make_arc(ex, {"center": "c", "start": "c", "end": "e"})
        assert r["ok"] is False and "radius" in r["error"]
        g.MakeArcCenter.assert_not_called()

    def test_end_on_the_start_says_so(self):
        g = _geompy({"C": (0, 0, 0), "S": (10, 0, 0)})
        ex = _executor(g, {"c": "C", "s": "S"})
        r = geom_tools.tool_make_arc(ex, {"center": "c", "start": "s", "end": "s"})
        assert r["ok"] is False and "same point" in r["error"]

    def test_codegen(self):
        from salome_agent.codegen import generate
        _, ex = self._arc_executor("MakeArcCenter")
        r = geom_tools.tool_make_arc(ex, {"center": "c", "start": "s", "end": "e"})
        assert generate(r["records"], ["arc_1"]) == (
            "arc_1 = geompy.MakeArcCenter(c, s, e, False)\n"
            "geompy.addToStudy(arc_1, 'arc_1')")


class TestLooseEnds:
    def test_closed_square_has_none(self):
        sq = [[(0, 0, 0), (1, 0, 0)], [(1, 0, 0), (1, 1, 0)],
              [(1, 1, 0), (0, 1, 0)], [(0, 1, 0), (0, 0, 0)]]
        assert _loose_ends(sq) == []

    def test_open_chain_has_two(self):
        assert _loose_ends([[(0, 0, 0), (1, 0, 0)], [(1, 0, 0), (1, 1, 0)]]) == \
            [(0, 0, 0), (1, 1, 0)]

    def test_a_closed_edge_has_none(self):
        assert _loose_ends([[(5, 0, 0)]]) == []

    def test_points_within_tolerance_join(self):
        assert _loose_ends([[(0, 0, 0), (1, 0, 0)], [(1 + 1e-9, 0, 0), (0, 0, 0)]]) == []


def _square(g_edges, g_vertices, coords, prefix=""):
    """Four edges of a unit square, registered in the fake geompy maps."""
    pts = [(0, 0, 0), (10, 0, 0), (10, 10, 0), (0, 10, 0)]
    names = []
    for i in range(4):
        e = f"{prefix}E{i}"
        a, b = f"{prefix}V{i}a", f"{prefix}V{i}b"
        coords[a], coords[b] = pts[i], pts[(i + 1) % 4]
        g_edges[e] = [e]
        g_vertices[e] = [a, b]
        names.append(e)
    return names


class TestMakeWire:
    def _setup(self, edges_count=4):
        coords, edges, vertices = {}, {}, {}
        names = _square(edges, vertices, coords)[:edges_count]
        g = _geompy(coords, edges, vertices)
        g.MakeWire.return_value = "WIRE"
        return g, _executor(g, {n.lower(): n for n in names}), [n.lower() for n in names]

    def test_closed_wire(self):
        g, ex, names = self._setup()
        r = geom_tools.tool_make_wire(ex, {"edges": names})
        assert r["ok"] is True, r.get("error")
        g.MakeWire.assert_called_once_with(["E0", "E1", "E2", "E3"], 1e-07)
        assert r["closed"] is True and "loose_ends" not in r
        rec = r["records"][-1]
        assert rec["fn"] == "MakeWire"
        assert rec["args"] == [{"kind": "list", "value": [{"kind": "ref", "value": n} for n in names]},
                               {"kind": "literal", "value": 1e-07}]
        assert rec["publish"] == ["wire_1"]

    def test_open_wire_reports_its_ends(self):
        _, ex, names = self._setup(edges_count=3)
        r = geom_tools.tool_make_wire(ex, {"edges": names})
        assert r["ok"] is True
        assert r["closed"] is False
        assert r["loose_ends"] == [[0, 0, 0], [0, 10, 0]]

    def test_a_gap_names_the_loose_ends_and_distance(self):
        g, ex, names = self._setup(edges_count=3)
        g.MakeWire.side_effect = RuntimeError("MakeWire : Wire is not connected")
        r = geom_tools.tool_make_wire(ex, {"edges": names})
        assert r["ok"] is False
        assert "[0, 0, 0]" in r["error"] and "[0, 10, 0]" in r["error"]
        assert "10" in r["error"]

    def test_needs_an_edge(self):
        g = _geompy({})
        r = geom_tools.tool_make_wire(_executor(g, {}), {"edges": []})
        assert r["ok"] is False


class TestMakeFaceFromWires:
    def _setup(self, kind="PLANAR"):
        coords, edges, vertices = {}, {}, {}
        _square(edges, vertices, coords)
        edges["OUTER"] = [f"E{i}" for i in range(4)]
        coords["HV"] = (5, 5, 0)
        edges["HOLE"] = ["HE"]
        vertices["HE"] = ["HV"]
        g = _geompy(coords, edges, vertices)
        g.MakeFaceWires.return_value = "FACE"
        g.KindOfShape.return_value = [kind, 0, 0, 0]
        g.BasicProperties.return_value = (0.0, 87.43, 0.0)
        return g, _executor(g, {"outer": "OUTER", "hole": "HOLE"})

    def test_face_with_a_hole(self):
        g, ex = self._setup()
        r = geom_tools.tool_make_face_from_wires(ex, {"wires": ["outer", "hole"]})
        assert r["ok"] is True, r.get("error")
        g.MakeFaceWires.assert_called_once_with(["OUTER", "HOLE"], 1)
        rec = r["records"][-1]
        assert rec["fn"] == "MakeFaceWires"
        assert rec["args"] == [{"kind": "list", "value": [{"kind": "ref", "value": "outer"},
                                                          {"kind": "ref", "value": "hole"}]},
                               {"kind": "literal", "value": 1}]
        assert rec["publish"] == ["face_1"]
        assert r["area"] == pytest.approx(87.43)

    def test_an_open_wire_is_refused(self):
        coords, edges, vertices = {}, {}, {}
        _square(edges, vertices, coords)
        edges["OPEN"] = ["E0", "E1"]
        g = _geompy(coords, edges, vertices)
        ex = _executor(g, {"open": "OPEN"})
        r = geom_tools.tool_make_face_from_wires(ex, {"wires": ["open"]})
        assert r["ok"] is False
        assert "open" in r["error"] and "[0, 0, 0]" in r["error"]
        g.MakeFaceWires.assert_not_called()

    def test_a_curved_face_is_refused(self):
        _, ex = self._setup(kind="FACE")
        r = geom_tools.tool_make_face_from_wires(ex, {"wires": ["outer"]})
        assert r["ok"] is False and "plane" in r["error"]

    def test_geom_failure_is_reported(self):
        g, ex = self._setup()
        g.MakeFaceWires.side_effect = RuntimeError("boom")
        r = geom_tools.tool_make_face_from_wires(ex, {"wires": ["outer"]})
        assert r["ok"] is False and "boom" in r["error"]

    def test_codegen(self):
        from salome_agent.codegen import generate
        _, ex = self._setup()
        r = geom_tools.tool_make_face_from_wires(ex, {"wires": ["outer", "hole"]})
        assert generate(r["records"], ["face_1"]) == (
            "face_1 = geompy.MakeFaceWires([outer, hole], 1)\n"
            "geompy.addToStudy(face_1, 'face_1')")


def test_profile_tools_come_together():
    from salome_agent.specialists.geom import CORE_TOOLS
    from salome_agent.modules import spec_for
    groups = spec_for("geom").tool_groups
    from salome_agent.specialists.tool_selection import initial_offered, resolve_load

    every = set(geom_tools.TOOLS)
    profile = {"make_line", "make_arc", "make_circle", "make_wire", "make_face_from_wires"}
    assert profile <= initial_offered({"tools": ["make_arc"]}, every, CORE_TOOLS, groups)
    defs = {t["name"]: t for t in geom_tools.list_tools()}
    assert profile <= resolve_load(["make_face_from_wires"], [], defs, groups)
    assert "make_arc" not in initial_offered({"tools": ["make_line"]}, every, CORE_TOOLS, groups)
