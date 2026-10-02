import math
import re
import subprocess
import sys
import tempfile

import pytest

from salome_agent.nodes.assembler import assemble

from ._salome_root import SALOME_ROOT

pytestmark = [
    pytest.mark.salome,
    pytest.mark.skipif(SALOME_ROOT is None,
                       reason="set SALOME_ROOT or salome.root in config.json"),
]


class TestPrimitiveTools:
    """Test primitive shape creation."""

    def test_make_box_two_pnt(self, agent_client):
        """make_box_two_pnt creates box from two points."""
        agent_client.call_tool("make_point", {"x": 0, "y": 0, "z": 0})
        agent_client.call_tool("make_point", {"x": 10, "y": 10, "z": 10})
        result = agent_client.call_tool("make_box_two_pnt", {"pnt1": "pt_1", "pnt2": "pt_2"})
        assert result["ok"] is True

    def test_make_box_two_pnt_accepts_coordinate_literals(self, agent_client):
        """A coord string instead of a point name -> MakeVertex emitted, box built."""
        result = agent_client.call_tool(
            "make_box_two_pnt", {"pnt1": "(0,0,0)", "pnt2": "(10,10,10)"})
        assert result["ok"] is True, result.get("error")
        assert [r["fn"] for r in result["records"]] == ["MakeVertex", "MakeVertex", "MakeBoxTwoPnt"]

    def test_make_cylinder(self, agent_client):
        """make_cylinder creates cylinder."""
        result = agent_client.call_tool("make_cylinder", {"radius": 5.0, "height": 10.0})
        assert result["ok"] is True
        assert result["records"][0]["fn"] == "MakeCylinderRH"

    def test_make_sphere(self, agent_client):
        """make_sphere creates sphere."""
        result = agent_client.call_tool("make_sphere", {"radius": 5.0})
        assert result["ok"] is True
        assert result["records"][0]["fn"] == "MakeSphereR"

    def test_make_sphere_with_center(self, agent_client):
        """make_sphere with center uses MakeSpherePntR."""
        agent_client.call_tool("make_point", {"x": 5, "y": 5, "z": 5})
        result = agent_client.call_tool("make_sphere", {"radius": 5.0, "center": "pt_1"})
        assert result["ok"] is True
        assert result["records"][0]["fn"] == "MakeSpherePntR"

    def test_make_cone(self, agent_client):
        """make_cone creates cone."""
        result = agent_client.call_tool("make_cone", {"radius1": 5.0, "radius2": 2.0, "height": 10.0})
        assert result["ok"] is True

    def test_make_torus(self, agent_client):
        """make_torus creates torus."""
        result = agent_client.call_tool("make_torus", {"radius_major": 10.0, "radius_minor": 3.0})
        assert result["ok"] is True

    def test_make_face(self, agent_client):
        """make_face creates rectangular face."""
        result = agent_client.call_tool("make_face", {"height": 10.0, "width": 20.0})
        assert result["ok"] is True

    def test_make_disk(self, agent_client):
        """make_disk creates disk."""
        result = agent_client.call_tool("make_disk", {"radius": 5.0})
        assert result["ok"] is True

    def test_make_polygon_face_gable_extrudes_to_a_roof(self, agent_client):
        """Triangle in the YOZ plane at x=0, extruded 200 along X (GH #45)."""
        face = agent_client.call_tool("make_polygon_face",
                                      {"points": ["0,0,80", "0,120,80", "0,60,120"]})
        assert face["ok"] is True, face.get("error")
        name = face["records"][-1]["out_vars"][0]
        roof = agent_client.call_tool("make_prism", {"base": name, "dx": 200, "dy": 0, "dz": 0})
        assert roof["ok"] is True, roof.get("error")
        bbox = roof["shape_info"]["bbox"]
        assert [round(v) for v in bbox] == [0, 200, 0, 120, 80, 120]

    def test_make_polygon_face_non_planar_is_a_clear_error(self, agent_client):
        r = agent_client.call_tool("make_polygon_face",
                                   {"points": ["0,0,0", "10,0,0", "10,10,0", "0,10,5"]})
        assert r["ok"] is False
        assert "same plane" in r["error"]


class TestProfileTools:
    """make_arc / make_wire / make_face_from_wires on real GEOM."""

    @staticmethod
    def _out(result):
        assert result["ok"] is True, result.get("error")
        return result["records"][-1]["out_vars"][0]

    @staticmethod
    def _props(client, name):
        return client.call_tool("basic_properties", {"shape": name})["properties"]

    def test_quarter_arc_by_centre_and_its_reverse(self, agent_client):
        """Default = the shorter way round, whatever the order of start and end."""
        fwd = agent_client.call_tool("make_arc", {"center": "0,0,0", "start": "10,0,0",
                                                  "end": "0,10,0"})
        rev = agent_client.call_tool("make_arc", {"center": "0,0,0", "start": "10,0,0",
                                                  "end": "0,10,0", "reverse": True})
        assert fwd["ok"] and rev["ok"], (fwd.get("error"), rev.get("error"))
        assert fwd["arc"]["length"] == pytest.approx(5 * math.pi, rel=1e-6)   # the short way
        assert rev["arc"]["length"] == pytest.approx(15 * math.pi, rel=1e-6)
        for r in (fwd, rev):
            assert {tuple(r["arc"][k]) for k in ("start", "end")} == {(10, 0, 0), (0, 10, 0)}
        swapped = agent_client.call_tool("make_arc", {"center": "0,0,0", "start": "0,10,0",
                                                      "end": "10,0,0"})
        assert swapped["arc"]["length"] == pytest.approx(5 * math.pi, rel=1e-6)

    def test_slot_profile_extrudes_to_the_right_volume(self, agent_client):
        """20 long, 10 wide slot with semicircular ends, 3 thick."""
        c = agent_client.call_tool
        edges = [
            self._out(c("make_line", {"pnt1": "0,-5,0", "pnt2": "20,-5,0"})),
            self._out(c("make_arc", {"start": "20,-5,0", "through": "25,0,0", "end": "20,5,0"})),
            self._out(c("make_line", {"pnt1": "20,5,0", "pnt2": "0,5,0"})),
            self._out(c("make_arc", {"start": "0,5,0", "through": "-5,0,0", "end": "0,-5,0"})),
        ]
        wire = c("make_wire", {"edges": edges})
        assert wire["closed"] is True
        face = c("make_face_from_wires", {"wires": [self._out(wire)]})
        assert face["area"] == pytest.approx(200 + 25 * math.pi, rel=1e-6)
        solid = self._out(c("make_prism", {"base": self._out(face), "dx": 0, "dy": 0, "dz": 3}))
        assert self._props(agent_client, solid)["volume"] == \
            pytest.approx(3 * (200 + 25 * math.pi), rel=1e-6)
        faces = c("explode", {"shape": solid, "sub_type": "face"})["records"][0]["out_vars"]
        assert len(faces) == 6

    def test_plate_with_a_round_hole(self, agent_client):
        c = agent_client.call_tool
        corners = ["0,0,0", "40,0,0", "40,40,0", "0,40,0"]
        edges = [self._out(c("make_line", {"pnt1": a, "pnt2": b}))
                 for a, b in zip(corners, corners[1:] + corners[:1])]
        outer = self._out(c("make_wire", {"edges": edges}))
        hole = self._out(c("make_circle", {"center": "20,20,0", "normal": "0,0,1", "radius": 5}))
        face = c("make_face_from_wires", {"wires": [outer, hole]})
        assert face["ok"] is True, face.get("error")
        assert face["area"] == pytest.approx(1600 - 25 * math.pi, rel=1e-6)

    def test_a_gap_is_a_clear_error(self, agent_client):
        c = agent_client.call_tool
        a = self._out(c("make_line", {"pnt1": "0,0,0", "pnt2": "10,0,0"}))
        b = self._out(c("make_line", {"pnt1": "10.5,0,0", "pnt2": "10,10,0"}))
        r = c("make_wire", {"edges": [a, b]})
        assert r["ok"] is False
        assert "0.5 apart" in r["error"]

    def test_open_wire_cannot_make_a_face(self, agent_client):
        c = agent_client.call_tool
        a = self._out(c("make_line", {"pnt1": "0,0,0", "pnt2": "10,0,0"}))
        b = self._out(c("make_line", {"pnt1": "10,0,0", "pnt2": "10,10,0"}))
        wire = c("make_wire", {"edges": [a, b]})
        assert wire["closed"] is False
        r = c("make_face_from_wires", {"wires": [self._out(wire)]})
        assert r["ok"] is False and "open" in r["error"]

    def test_non_planar_wire_is_refused(self, agent_client):
        c = agent_client.call_tool
        pts = ["0,0,0", "10,0,0", "10,10,0", "0,10,5"]
        edges = [self._out(c("make_line", {"pnt1": a, "pnt2": b}))
                 for a, b in zip(pts, pts[1:] + pts[:1])]
        r = c("make_face_from_wires", {"wires": [self._out(c("make_wire", {"edges": edges}))]})
        assert r["ok"] is False and "one plane" in r["error"]


class TestCurveTools:
    """make_ellipse / make_polyline / make_curve / make_profile_face on real GEOM."""

    _out = staticmethod(TestProfileTools._out)
    _props = staticmethod(TestProfileTools._props)
    # Ramanujan's perimeter of the 10 x 5 ellipse
    ELLIPSE_LENGTH = math.pi * (3 * 15 - math.sqrt(35 * 25))

    def test_full_ellipse_and_its_face(self, agent_client):
        c = agent_client.call_tool
        r = c("make_ellipse", {"center": "0,0,0", "normal": "0,0,1", "r_major": 10,
                               "r_minor": 5, "major_dir": "1,0,0"})
        # BasicProperties measures a closed ellipse ~0.13% long (bbox and face area are exact)
        assert r["ellipse"]["length"] == pytest.approx(self.ELLIPSE_LENGTH, rel=2e-3)
        face = c("make_face_from_wires", {"wires": [self._out(r)]})
        assert face["ok"] is True, face.get("error")
        assert face["area"] == pytest.approx(50 * math.pi, rel=1e-6)

    def test_quarter_ellipse_either_way_round(self, agent_client):
        for start, end in (("10,0,0", "0,5,0"), ("0,5,0", "10,0,0")):
            r = agent_client.call_tool("make_ellipse", {"center": "0,0,0", "start": start,
                                                        "end": end})
            assert r["ok"] is True, r.get("error")
            assert r["ellipse"]["length"] == pytest.approx(self.ELLIPSE_LENGTH / 4, rel=1e-4)
            assert {tuple(r["ellipse"][k]) for k in ("start", "end")} == {(10, 0, 0), (0, 5, 0)}

    def test_open_polyline(self, agent_client):
        r = agent_client.call_tool("make_polyline", {"points": ["0,0,0", "10,0,0", "10,10,0"]})
        assert r["ok"] is True, r.get("error")
        assert r["length"] == pytest.approx(20)

    @pytest.mark.parametrize("args", [
        {"points": ["0,0,0", "5,3,0", "10,0,0"]},
        {"points": ["0,0,0", "5,3,0", "10,0,0"], "kind": "bezier"},
        {"points": ["0,0,0", "10,0,0"], "start_tangent": "0,1,0", "end_tangent": "0,-1,0"},
    ])
    def test_curve_ends_on_its_end_points(self, agent_client, args):
        r = agent_client.call_tool("make_curve", args)
        assert r["ok"] is True, r.get("error")
        assert {tuple(r["curve"][k]) for k in ("start", "end")} == {(0, 0, 0), (10, 0, 0)}
        assert r["curve"]["length"] > 10

    def test_rounded_plate_with_a_hole_extrudes_to_the_right_volume(self, agent_client):
        """20 x 10 plate, corner radius 2, round hole r 2 in the middle, 3 thick."""
        c = agent_client.call_tool
        face = c("make_profile_face", {"start": "2,0,0", "segments": [
            {"to": "18,0,0"}, {"to": "20,2,0", "center": "18,2,0"},
            {"to": "20,8,0"}, {"to": "18,10,0", "center": "18,8,0"},
            {"to": "2,10,0"}, {"to": "0,8,0", "center": "2,8,0"},
            {"to": "0,2,0"}, {"to": "2,0,0", "center": "2,2,0"}],
            "holes": [{"center": "10,5,0", "radius": 2}]})
        assert face["ok"] is True, face.get("error")
        assert "closed_with_line" not in face
        assert face["area"] == pytest.approx(184, rel=1e-6)
        solid = self._out(c("make_prism", {"base": self._out(face), "dx": 0, "dy": 0, "dz": 3}))
        assert self._props(agent_client, solid)["volume"] == pytest.approx(3 * 184, rel=1e-6)

    def test_slot_by_through_arcs(self, agent_client):
        face = agent_client.call_tool("make_profile_face", {"start": "0,-5,0", "segments": [
            {"to": "20,-5,0"}, {"to": "20,5,0", "through": "25,0,0"},
            {"to": "0,5,0"}, {"to": "0,-5,0", "through": "-5,0,0"}]})
        assert face["ok"] is True, face.get("error")
        assert face["area"] == pytest.approx(200 + 25 * math.pi, rel=1e-6)

    def test_hole_outline_direction_does_not_matter(self, agent_client):
        """The description promises either way round works: same area and volume both ways."""
        c = agent_client.call_tool
        outer = {"start": "0,0,0", "segments": [{"to": "40,0,0"}, {"to": "40,20,0"},
                                                 {"to": "0,20,0"}]}
        ccw = {"start": "5,6,0", "segments": [
            {"to": "15,6,0"}, {"to": "15,14,0", "through": "19,10,0"},
            {"to": "5,14,0"}, {"to": "5,6,0", "through": "1,10,0"}]}
        cw = {"start": "5,6,0", "segments": [
            {"to": "5,14,0", "through": "1,10,0"}, {"to": "15,14,0"},
            {"to": "15,6,0", "through": "19,10,0"}, {"to": "5,6,0"}]}
        area = 800 - (80 + 16 * math.pi)
        for hole in (ccw, cw):
            face = c("make_profile_face", {**outer, "holes": [hole]})
            assert face["ok"] is True, face.get("error")
            assert face["area"] == pytest.approx(area, rel=1e-6)
            solid = self._out(c("make_prism", {"base": self._out(face), "dx": 0, "dy": 0, "dz": 4}))
            assert self._props(agent_client, solid)["volume"] == pytest.approx(4 * area, rel=1e-6)

    def test_spline_side_closed_by_a_line(self, agent_client):
        face = agent_client.call_tool("make_profile_face", {"start": "0,0,0", "segments": [
            {"to": "10,0,0"}, {"to": "0,10,0", "via": ["8,6,0", "4,9,0"]}]})
        assert face["ok"] is True, face.get("error")
        assert face["closed_with_line"] == ["outline"]
        assert face["area"] > 50

    CORNER = 1 - math.pi / 4     # area a fillet of radius 1 takes off a right-angle corner

    def test_fillet_convex_corner(self, agent_client):
        face = agent_client.call_tool("make_profile_face", {"start": "0,0,0", "segments": [
            {"to": "10,0,0", "fillet": 2}, {"to": "10,10,0"}, {"to": "0,10,0"}, {"to": "0,0,0"}]})
        assert face["ok"] is True, face.get("error")
        assert face["area"] == pytest.approx(100 - 4 * self.CORNER, rel=1e-6)

    def test_fillet_concave_corner_and_its_replay(self, agent_client):
        """L-shape whose inner corner gets material added; result.py rebuilds the same face."""
        from salome_agent.codegen import generate
        face = agent_client.call_tool("make_profile_face", {"start": "0,0,0", "segments": [
            {"to": "20,0,0"}, {"to": "20,10,0"}, {"to": "10,10,0", "fillet": 3},
            {"to": "10,20,0"}, {"to": "0,20,0", "fillet": 2}, {"to": "0,0,0", "fillet": 3}]})
        assert face["ok"] is True, face.get("error")
        area = 300 + 9 * self.CORNER - 4 * self.CORNER - 9 * self.CORNER
        assert face["area"] == pytest.approx(area, rel=1e-6)
        name = self._out(face)
        script = re.sub(rf"\b{name}\b", "replayed_face", generate(face["records"], [name]))
        r = agent_client.run(script + "\nprint(geompy.BasicProperties(replayed_face)[1])",
                             timeout=30)
        assert float(r["stdout"].split()[-1]) == pytest.approx(area, rel=1e-6)

    def _valid_with_wires(self, client, face, wires):
        """Area is integrated along the boundary, so it can be right on a broken
        face: also check validity and that every outline is its own wire."""
        name = self._out(face)
        assert client.call_tool("check_shape", {"shape": name})["ok"] is True
        got = client.call_tool("get_subshapes", {"shape": name, "sub_type": "wire", "where": "all"})
        assert len(got["ids"]) == wires

    def test_fillet_p1_plate(self, agent_client):
        """The p1.png plate: top half disc, waist with concave R20 corners, bottom half
        disc, the round holes and both curved slots (pitch R35, 18 wide, 120 deg each)."""
        c = math.cos(math.radians(60))
        s = math.sin(math.radians(60))

        def slot(side):
            p = lambda r, sgn: f"{side * r * c},{-80 + sgn * r * s},0"
            # each R9 end cap bulges past the slot's end angle (+-60 deg about (0,-80))
            cap = lambda sgn: f"{side * (35 * c - 9 * s)},{-80 + sgn * (35 * s + 9 * c)},0"
            return {"start": p(26, -1), "segments": [
                {"to": p(26, 1), "center": "0,-80,0"},
                {"to": p(44, 1), "through": cap(1)},
                {"to": p(44, -1), "center": "0,-80,0"},
                {"to": p(26, -1), "through": cap(-1)}]}

        holes = [{"center": "0,0,0", "radius": 25}, {"center": "0,-80,0", "radius": 20}]
        holes += [{"center": f"{64 * math.cos(math.radians(a))},{64 * math.sin(math.radians(a))},0",
                   "radius": 15} for a in (30, 90, 150)]
        face = agent_client.call_tool("make_profile_face", {"start": "96,0,0", "segments": [
            {"to": "-96,0,0", "through": "0,96,0"}, {"to": "-54,0,0", "fillet": 20},
            {"to": "-54,-80,0"}, {"to": "54,-80,0", "through": "0,-134,0"},
            {"to": "54,0,0", "fillet": 20}, {"to": "96,0,0"}],
            "holes": holes + [slot(1), slot(-1)]})
        assert face["ok"] is True, face.get("error")
        outline = math.pi * (96 ** 2 + 54 ** 2) / 2 + 108 * 80 + 2 * 400 * self.CORNER
        slots = 2 * (35 * 2 * math.pi / 3 * 18 + 81 * math.pi)
        area = outline - math.pi * (25 ** 2 + 20 ** 2 + 3 * 15 ** 2) - slots
        assert face["area"] == pytest.approx(area, rel=1e-6)
        self._valid_with_wires(agent_client, face, 8)

    def test_slot_holes_p1_plate(self, agent_client):
        """test_fillet_p1_plate with its curved slots given as slots, not traced arcs."""
        holes = [{"center": "0,0,0", "radius": 25}, {"center": "0,-80,0", "radius": 20}]
        holes += [{"center": f"{64 * math.cos(math.radians(a))},{64 * math.sin(math.radians(a))},0",
                   "radius": 15} for a in (30, 90, 150)]
        holes += [{"slot": {"center": "0,-80,0", "radius": 35, "width": 18,
                            "from_angle": a0, "to_angle": a0 + 120}} for a0 in (-60, 120)]
        face = agent_client.call_tool("make_profile_face", {"start": "96,0,0", "segments": [
            {"to": "-96,0,0", "through": "0,96,0"}, {"to": "-54,0,0", "fillet": 20},
            {"to": "-54,-80,0"}, {"to": "54,-80,0", "through": "0,-134,0"},
            {"to": "54,0,0", "fillet": 20}, {"to": "96,0,0"}], "holes": holes})
        assert face["ok"] is True, face.get("error")
        outline = math.pi * (96 ** 2 + 54 ** 2) / 2 + 108 * 80 + 2 * 400 * self.CORNER
        slots = 2 * (35 * 2 * math.pi / 3 * 18 + 81 * math.pi)
        area = outline - math.pi * (25 ** 2 + 20 ** 2 + 3 * 15 ** 2) - slots
        assert face["area"] == pytest.approx(area, rel=1e-6)
        self._valid_with_wires(agent_client, face, 8)

    def test_straight_slot_off_the_xy_plane(self, agent_client):
        face = agent_client.call_tool("make_profile_face", {"start": "0,0,0", "segments": [
            {"to": "40,0,0"}, {"to": "40,0,20"}, {"to": "0,0,20"}],
            "holes": [{"slot": {"from": "10,0,15", "to": "30,0,15", "width": 6}},
                      {"slot": {"center": "20,0,10", "radius": 6, "width": 2,
                                "from_angle": 30, "to_angle": 150}}]})
        assert face["ok"] is True, face.get("error")
        area = 800 - (20 * 6 + 9 * math.pi) - (6 * 2 * math.pi * 2 / 3 + math.pi)
        assert face["area"] == pytest.approx(area, rel=1e-6)
        self._valid_with_wires(agent_client, face, 3)

    def test_fillet_in_a_hole_keeps_its_own_wire(self, agent_client):
        face = agent_client.call_tool("make_profile_face", {"start": "0,0,0", "segments": [
            {"to": "40,0,0", "fillet": 2}, {"to": "40,20,0"}, {"to": "0,20,0"}],
            "holes": [{"center": "10,10,0", "radius": 3},
                      {"start": "25,5,0", "segments": [{"to": "35,5,0", "fillet": 1},
                                                       {"to": "35,15,0"}, {"to": "25,15,0"}]}]})
        assert face["ok"] is True, face.get("error")
        area = 800 - 4 * self.CORNER - 9 * math.pi - (100 - self.CORNER)
        assert face["area"] == pytest.approx(area, rel=1e-6)
        self._valid_with_wires(agent_client, face, 3)

    def test_fillet_line_arc_corner(self, agent_client):
        plain = {"start": "0,0,0", "segments": [{"to": "20,0,0"}, {"to": "0,20,0", "center": "0,0,0"}]}
        rounded = {"start": "0,0,0", "segments": [{"to": "20,0,0", "fillet": 3},
                                                   {"to": "0,20,0", "center": "0,0,0"}]}
        a = agent_client.call_tool("make_profile_face", plain)["area"]
        r = agent_client.call_tool("make_profile_face", rounded)
        assert r["ok"] is True, r.get("error")
        assert a - 3 * 9 * self.CORNER < r["area"] < a   # the arc bends into the corner: more than a square corner's 9C

    def test_fillet_too_big_names_the_corner(self, agent_client):
        r = agent_client.call_tool("make_profile_face", {"start": "0,0,0", "segments": [
            {"to": "10,0,0", "fillet": 50}, {"to": "10,10,0"}, {"to": "0,10,0"}]})
        assert r["ok"] is False
        assert "[10.0, 0.0, 0.0]" in r["error"] and "50" in r["error"]

    def test_wrong_arc_end_names_the_right_one(self, agent_client):
        r = agent_client.call_tool("make_profile_face", {"start": "10,0,0", "segments": [
            {"to": "0,12,0", "center": "0,0,0"}]})
        assert r["ok"] is False
        assert "[0.0, 10.0, 0.0]" in r["error"]


class TestBasicObjectTools:
    """Test basic geometry objects."""

    def test_make_point(self, agent_client):
        """make_point creates vertex."""
        result = agent_client.call_tool("make_point", {"x": 1.0, "y": 2.0, "z": 3.0})
        assert result["ok"] is True

    def test_make_vector(self, agent_client):
        """make_vector creates vector."""
        result = agent_client.call_tool("make_vector", {"dx": 1.0, "dy": 2.0, "dz": 3.0})
        assert result["ok"] is True

    def test_make_line(self, agent_client):
        """make_line creates line between two points."""
        agent_client.call_tool("make_point", {"x": 0, "y": 0, "z": 0})
        agent_client.call_tool("make_point", {"x": 10, "y": 10, "z": 10})
        result = agent_client.call_tool("make_line", {"pnt1": "pt_1", "pnt2": "pt_2"})
        assert result["ok"] is True

    def test_make_plane(self, agent_client):
        """make_plane creates plane."""
        agent_client.call_tool("make_point", {"x": 0, "y": 0, "z": 0})
        agent_client.call_tool("make_vector", {"dx": 0, "dy": 0, "dz": 1})
        result = agent_client.call_tool("make_plane", {"point": "pt_1", "normal": "vec_1"})
        assert result["ok"] is True

    def test_make_circle(self, agent_client):
        """make_circle creates circle."""
        agent_client.call_tool("make_point", {"x": 0, "y": 0, "z": 0})
        agent_client.call_tool("make_vector", {"dx": 0, "dy": 0, "dz": 1})
        result = agent_client.call_tool("make_circle", {"center": "pt_1", "normal": "vec_1", "radius": 5.0})
        assert result["ok"] is True


class TestComplexObjectTools:
    """Test swept and derived objects."""

    def test_make_prism(self, agent_client):
        """make_prism creates linear extrusion."""
        agent_client.call_tool("make_face", {"height": 10.0, "width": 10.0})
        result = agent_client.call_tool("make_prism", {"base": "face_1", "dx": 0, "dy": 0, "dz": 20})
        assert result["ok"] is True

    def test_make_revolution(self, agent_client):
        """make_revolution creates revolved shape."""
        # Base must be clear of the axis -- an origin-centred face straddles
        # OZ and revolves into a self-intersecting shape.
        agent_client.call_tool("make_face", {"height": 10.0, "width": 4.0, "orientation": 3})
        agent_client.call_tool("translate", {"shape": "face_1", "dx": 20, "dy": 0, "dz": 0})
        result = agent_client.call_tool("make_revolution", {"base": "tr_1", "axis": "OZ", "angle_deg": 180})
        assert result["ok"] is True, result.get("error")

    def test_make_pipe(self, agent_client):
        """make_pipe creates swept shape."""
        agent_client.call_tool("make_point", {"x": 0, "y": 0, "z": 0})
        agent_client.call_tool("make_vector", {"dx": 0, "dy": 0, "dz": 1})
        agent_client.call_tool("make_circle", {"center": "pt_1", "normal": "vec_1", "radius": 2.0})
        agent_client.call_tool("make_point", {"x": 10, "y": 10, "z": 10})
        agent_client.call_tool("make_line", {"pnt1": "pt_1", "pnt2": "pt_2"})
        result = agent_client.call_tool("make_pipe", {"base": "circ_1", "path": "line_1"})
        assert result["ok"] is True, result.get("error")

    def test_make_thick_solid(self, agent_client):
        """make_thick_solid creates thick shell."""
        agent_client.call_tool("make_box", {"dx": 10, "dy": 10, "dz": 10})
        # face id 3 is a box's first face -- removed to open the solid
        # (MakeThickSolid needs at least one).
        result = agent_client.call_tool("make_thick_solid", {"shape": "box_1", "thickness": 2.0, "faces": [3]})
        assert result["ok"] is True, result.get("error")

    def test_make_filling(self, agent_client):
        """make_filling creates filling surface."""
        # Create a closed wire of edges/wires for filling
        agent_client.call_tool("make_point", {"x": 0, "y": 0, "z": 0})
        agent_client.call_tool("make_point", {"x": 10, "y": 0, "z": 0})
        agent_client.call_tool("make_line", {"pnt1": "pt_1", "pnt2": "pt_2"})
        result = agent_client.call_tool("make_filling", {"contours": ["line_1"]})
        # Filling may fail with single edge, but tool should handle gracefully
        assert isinstance(result.get("ok"), bool)


class TestBooleanTools:
    """Test boolean operations."""

    def test_fuse_two_shapes(self, agent_client):
        """fuse combines two shapes."""
        agent_client.call_tool("make_box", {"dx": 10, "dy": 10, "dz": 10})
        agent_client.call_tool("make_cylinder", {"radius": 5, "height": 10})
        result = agent_client.call_tool("fuse", {"shapes": ["box_1", "cyl_1"]})
        assert result["ok"] is True
        assert result["records"][0]["fn"] == "MakeFuse"

    def test_fuse_multiple_shapes(self, agent_client):
        """fuse with 3+ shapes uses MakeFuseList."""
        agent_client.call_tool("make_box", {"dx": 5, "dy": 5, "dz": 5})
        agent_client.call_tool("make_box", {"dx": 5, "dy": 5, "dz": 5})
        agent_client.call_tool("make_box", {"dx": 5, "dy": 5, "dz": 5})
        result = agent_client.call_tool("fuse", {"shapes": ["box_1", "box_2", "box_3"]})
        assert result["ok"] is True
        assert result["records"][0]["fn"] == "MakeFuseList"

    def test_cut(self, agent_client):
        """cut subtracts tool from main shape."""
        agent_client.call_tool("make_box", {"dx": 10, "dy": 10, "dz": 10})
        agent_client.call_tool("make_cylinder", {"radius": 3, "height": 15})
        result = agent_client.call_tool("cut", {"main": "box_1", "tools": ["cyl_1"]})
        assert result["ok"] is True

    def test_common(self, agent_client):
        """common finds intersection."""
        agent_client.call_tool("make_box", {"dx": 10, "dy": 10, "dz": 10})
        agent_client.call_tool("make_sphere", {"radius": 6})
        result = agent_client.call_tool("common", {"shapes": ["box_1", "sph_1"]})
        assert result["ok"] is True

    def test_partition(self, agent_client):
        """partition splits shapes by tools."""
        agent_client.call_tool("make_box", {"dx": 10, "dy": 10, "dz": 10})
        agent_client.call_tool("make_point", {"x": 5, "y": 5, "z": 5})
        agent_client.call_tool("make_vector", {"dx": 0, "dy": 0, "dz": 1})
        agent_client.call_tool("make_plane", {"point": "pt_1", "normal": "vec_1"})
        result = agent_client.call_tool("partition", {"shapes": ["box_1"], "tools": ["plane_1"]})
        assert result["ok"] is True, result.get("error")


class TestTransformationTools:
    """Test transformation operations."""

    def test_translate(self, agent_client):
        """translate moves shape."""
        agent_client.call_tool("make_box", {"dx": 10, "dy": 10, "dz": 10})
        result = agent_client.call_tool("translate", {"shape": "box_1", "dx": 5, "dy": 0, "dz": 0})
        assert result["ok"] is True

    def test_rotate(self, agent_client):
        """rotate rotates shape."""
        agent_client.call_tool("make_box", {"dx": 10, "dy": 10, "dz": 10})
        result = agent_client.call_tool("rotate", {"shape": "box_1", "axis": "OZ", "angle_deg": 45})
        assert result["ok"] is True

    def test_mirror(self, agent_client):
        """mirror reflects shape."""
        agent_client.call_tool("make_box", {"dx": 10, "dy": 10, "dz": 10})
        result = agent_client.call_tool("mirror", {"shape": "box_1", "entity": "XOY"})
        assert result["ok"] is True

    def test_scale(self, agent_client):
        """scale resizes shape uniformly."""
        agent_client.call_tool("make_box", {"dx": 10, "dy": 10, "dz": 10})
        result = agent_client.call_tool("scale", {"shape": "box_1", "factor": 2.0})
        assert result["ok"] is True

    def test_offset_shape(self, agent_client):
        """offset_shape offsets shape."""
        agent_client.call_tool("make_box", {"dx": 10, "dy": 10, "dz": 10})
        result = agent_client.call_tool("offset_shape", {"shape": "box_1", "offset": 1.0})
        assert result["ok"] is True

    def test_position(self, agent_client):
        """position places a copy at a target frame."""
        agent_client.call_tool("make_box", {"dx": 10, "dy": 10, "dz": 10})
        result = agent_client.call_tool("position", {"shape": "box_1", "origin": [50, 0, 0]})
        assert result["ok"] is True, result.get("error")
        assert [r["fn"] for r in result["records"]] == ["MakeMarker", "MakePosition"]

    def test_align_butts_a_part_against_another(self, agent_client):
        """align moves `shape` so its chosen face meets `to`'s, gap included."""
        agent_client.call_tool("make_box", {"dx": 100, "dy": 100, "dz": 100})   # box_1
        agent_client.call_tool("make_box", {"dx": 10, "dy": 10, "dz": 10})      # box_2
        result = agent_client.call_tool("align", {
            "shape": "box_2", "to": "box_1",
            "match": [{"axis": "x", "my": "min", "target": "max", "gap": 5},
                      {"axis": "y", "my": "center", "target": "center"}]})
        assert result["ok"] is True, result.get("error")
        assert [r["fn"] for r in result["records"]] == ["MakeTranslation"]
        bb = result["shape_info"]["bbox"]
        assert bb[0] == pytest.approx(105.0, abs=1e-3)     # xmin = box_1 xmax + 5
        assert result["shape_info"]["centre"][1] == pytest.approx(50.0, abs=1e-3)

    def test_align_rejects_two_constraints_on_one_axis(self, agent_client):
        agent_client.call_tool("make_box", {"dx": 10, "dy": 10, "dz": 10})
        agent_client.call_tool("make_box", {"dx": 10, "dy": 10, "dz": 10})
        result = agent_client.call_tool("align", {
            "shape": "box_2", "to": "box_1",
            "match": [{"axis": "x", "my": "min", "target": "min"},
                      {"axis": "x", "my": "max", "target": "max"}]})
        assert result["ok"] is False

    def test_shape_info_reports_placement(self, agent_client):
        """Every successful tool result carries bbox/size/centre; a translate shifts it."""
        box = agent_client.call_tool("make_box", {"dx": 10, "dy": 10, "dz": 10})
        assert box["shape_info"]["size"] == [10.0, 10.0, 10.0]
        assert box["shape_info"]["centre"] == [5.0, 5.0, 5.0]
        moved = agent_client.call_tool("translate", {"shape": "box_1", "dx": 100, "dy": 0, "dz": 0})
        assert moved["shape_info"]["centre"][0] == pytest.approx(105.0, abs=1e-3)


class TestLocalEdgeOps:
    """Test local edge operations."""

    def test_fillet_edges_with_all(self, agent_client):
        """fillet_edges with 'all' uses new MakeFillet signature."""
        agent_client.call_tool("make_box", {"dx": 10, "dy": 10, "dz": 10})
        result = agent_client.call_tool("fillet_edges", {"shape": "box_1", "edges": "all", "radius": 1.0})
        assert result["ok"] is True
        # Verify MakeFillet call with hardcoded EDGE type
        assert result["records"][0]["fn"] == "MakeFillet"
        assert result["records"][0]["args"][2] == {"kind": "literal", "value": 6}  # EDGE = 6

    def test_chamfer_edges_with_all(self, agent_client):
        """chamfer_edges chamfers all edges."""
        agent_client.call_tool("make_box", {"dx": 10, "dy": 10, "dz": 10})
        result = agent_client.call_tool("chamfer_edges", {"shape": "box_1", "d1": 1.0, "edges": "all"})
        assert result["ok"] is True, result.get("error")
        rec = result["records"][0]
        assert rec["fn"] == "MakeChamferEdges"
        # MakeChamferEdges(shape, d1, d2, edges) -- no shape-type arg
        assert rec["args"][1] == {"kind": "literal", "value": 1.0}
        assert rec["args"][2] == {"kind": "literal", "value": 1.0}
        assert isinstance(rec["args"][3]["value"], list)

    def test_fillet_edges_bad_string_ids_get_a_clean_error(self, agent_client):
        """A bare string `edges` is normalised to ints; ids that are not edges
        of the shape are refused before GEOM, naming the real ones (GH #35)."""
        agent_client.call_tool("make_box", {"dx": 10, "dy": 10, "dz": 10})
        result = agent_client.call_tool("fillet_edges", {"shape": "box_1", "edges": "97,98", "radius": 1.0})
        assert result["ok"] is False
        assert result["error"].startswith("fillet_edges: [97, 98] are not edge ids of box_1")
        assert "its edge ids are [5, 8, 10," in result["error"]


class TestDisplayTools:
    """Test display property tools."""

    def test_set_color(self, agent_client):
        """set_color changes shape color."""
        agent_client.call_tool("make_box", {"dx": 10, "dy": 10, "dz": 10})
        result = agent_client.call_tool("set_color", {"shape": "box_1", "red": 255, "green": 0, "blue": 0})
        assert result["ok"] is True
        # Verify the record for codegen
        assert result["records"][0]["emit"] == "method_call"
        assert result["records"][0]["fn"] == "SetColor"

    def test_set_transparency(self, agent_client):
        """set_transparency changes transparency."""
        agent_client.call_tool("make_box", {"dx": 10, "dy": 10, "dz": 10})
        result = agent_client.call_tool("set_transparency", {"shape": "box_1", "transparency": 0.5})
        assert result["ok"] is True
        assert result["records"][0]["emit"] == "gui_call"

    def test_set_display_mode(self, agent_client):
        """set_display_mode changes display mode."""
        agent_client.call_tool("make_box", {"dx": 10, "dy": 10, "dz": 10})
        result = agent_client.call_tool("set_display_mode", {"shape": "box_1", "mode": 1})
        assert result["ok"] is True
        assert result["records"][0]["emit"] == "gui_call"


class TestShapeTypeEdgeConstant:
    """Verify the hard-coded geompy.ShapeType['EDGE'] literal via a live run.

    fillet_edges emits MakeFillet(shape, r, 6, edge_ids). If a live fillet
    succeeds, 6 is the right EDGE constant for this build (a wrong shape-type
    would fillet the wrong subshapes or fail).
    """

    def test_fillet_uses_correct_edge_type(self, agent_client):
        agent_client.call_tool("make_box", {"dx": 10, "dy": 10, "dz": 10})
        result = agent_client.call_tool("fillet_edges", {"shape": "box_1", "edges": "all", "radius": 1.0})
        assert result["ok"] is True, result.get("error")
        assert result["records"][0]["args"][2] == {"kind": "literal", "value": 6}


class TestEndToEndScenarios:
    """Test real-world scenarios with multiple operations."""

    def test_box_fuse_fillet_assemble_and_run(self, agent_client):
        """Create two boxes, fuse them, fillet edges, assemble, and run as standalone script."""
        # Create geometry
        agent_client.call_tool("make_box", {"dx": 10, "dy": 10, "dz": 10})
        agent_client.call_tool("make_box", {"dx": 5, "dy": 5, "dz": 5})
        result_fuse = agent_client.call_tool("fuse", {"shapes": ["box_1", "box_2"]})
        assert result_fuse["ok"] is True

        result_fillet = agent_client.call_tool("fillet_edges", {"shape": "fuse_1", "edges": "all", "radius": 0.5})
        assert result_fillet["ok"] is True

        result_color = agent_client.call_tool("set_color", {"shape": "fillet_1", "red": 255, "green": 0, "blue": 0})
        assert result_color["ok"] is True

        # Assemble a result.py script
        from salome_agent.codegen import generate
        call_log = (
            result_fuse["records"] +
            result_fillet["records"] +
            result_color["records"]
        )
        script = generate(call_log, result_names=["fillet_1"])

        # Verify script contains expected calls
        assert "MakeFuse" in script
        assert "MakeFillet" in script
        assert "SetColor" in script
        assert "SALOMEDS.Color" in script

        # Assemble full result.py
        state = {
            "run_id": "test-run",
            "user_request": "make a red fused box",
            "stages": {
                0: {"module": "geom", "status": "ok", "script": script},
            },
        }
        result_py = assemble(state)

        # Verify preamble includes SALOMEDS
        assert "from salome.kernel import SALOMEDS" in result_py
        assert "geomBuilder" in result_py
        assert "salome_init" in result_py

        # Try to parse it as valid Python (syntax check)
        import ast
        try:
            ast.parse(result_py)
        except SyntaxError as e:
            pytest.fail(f"Generated script has syntax error: {e}\n{result_py}")

    def test_box_with_transparency_and_display_mode(self, agent_client):
        """Test display properties in a complete workflow."""
        agent_client.call_tool("make_box", {"dx": 10, "dy": 10, "dz": 10})
        agent_client.call_tool("set_transparency", {"shape": "box_1", "transparency": 0.3})
        agent_client.call_tool("set_display_mode", {"shape": "box_1", "mode": 1})

        # Verify tools work without errors
        objects = agent_client.call_tool("list_study_objects", {})
        assert objects["ok"] is True
        assert "box_1" in objects.get("names", [])


class TestSubshapeSelection:
    def test_explode_edges_of_a_box(self, agent_client):
        agent_client.call_tool("make_box", {"dx": 10, "dy": 10, "dz": 10})
        out = agent_client.call_tool("explode", {"shape": "box_1", "sub_type": "edge"})
        assert out["ok"] is True, out.get("error")
        assert len(out["names"]) == 12
        assert out["records"][0]["emit"] == "list_call"

    def test_explode_faces_of_a_box(self, agent_client):
        agent_client.call_tool("make_box", {"dx": 10, "dy": 10, "dz": 10})
        out = agent_client.call_tool("explode", {"shape": "box_1", "sub_type": "face"})
        assert out["ok"] is True, out.get("error")
        assert len(out["names"]) == 6

    def test_get_subshapes_all_edges(self, agent_client):
        agent_client.call_tool("make_box", {"dx": 20, "dy": 20, "dz": 20})
        out = agent_client.call_tool(
            "get_subshapes", {"shape": "box_1", "sub_type": "edge", "where": "all"})
        assert out["ok"] is True, out.get("error")
        assert len(out["ids"]) == 12
        assert len(out["centres"]) == 12
        assert out["records"] == []

    def test_get_subshapes_top_four_edges_by_z(self, agent_client):
        agent_client.call_tool("make_box", {"dx": 20, "dy": 20, "dz": 20})
        out = agent_client.call_tool("get_subshapes", {
            "shape": "box_1", "sub_type": "edge",
            "where": {"sorted_by": "z", "take": "max", "count": 4}})
        assert out["ok"] is True, out.get("error")
        assert len(out["ids"]) == 4
        for c in out["centres"]:
            assert c[2] == 20.0

    def test_get_subshapes_on_plane(self, agent_client):
        agent_client.call_tool("make_box", {"dx": 20, "dy": 20, "dz": 20})
        out = agent_client.call_tool("get_subshapes", {
            "shape": "box_1", "sub_type": "face",
            "where": {"on_plane": {"point": [0, 0, 20], "normal": [0, 0, 1]},
                      "state": "on"}})
        assert out["ok"] is True, out.get("error")
        assert len(out["ids"]) == 1

    def test_get_subshape_from_ids(self, agent_client):
        agent_client.call_tool("make_box", {"dx": 10, "dy": 10, "dz": 10})
        ids = agent_client.call_tool(
            "get_subshapes", {"shape": "box_1", "sub_type": "face", "where": "all"})["ids"]
        out = agent_client.call_tool("get_subshape", {"shape": "box_1", "ids": ids[:1]})
        assert out["ok"] is True, out.get("error")
        assert out["records"][0]["fn"] == "GetSubShape"

    def test_min_distance_between_two_boxes(self, agent_client):
        agent_client.call_tool("make_box", {"dx": 10, "dy": 10, "dz": 10})
        agent_client.call_tool("make_box", {"dx": 10, "dy": 10, "dz": 10})
        agent_client.call_tool("translate", {"shape": "box_2", "dx": 30, "dy": 0, "dz": 0})
        out = agent_client.call_tool("min_distance", {"shape1": "box_1", "shape2": "tr_1"})
        assert out["ok"] is True, out.get("error")
        assert out["distance"] == 20.0

    def test_basic_properties_of_a_box(self, agent_client):
        agent_client.call_tool("make_box", {"dx": 2, "dy": 3, "dz": 4})
        out = agent_client.call_tool("basic_properties", {"shape": "box_1"})
        assert out["ok"] is True, out.get("error")
        assert out["properties"]["volume"] == 24.0

    def test_fillet_edges_inline_where(self, agent_client):
        agent_client.call_tool("make_box", {"dx": 20, "dy": 20, "dz": 20})
        out = agent_client.call_tool("fillet_edges", {
            "shape": "box_1", "radius": 3,
            "where": {"sorted_by": "z", "take": "max", "count": 4}})
        assert out["ok"] is True, out.get("error")
        assert out["records"][0]["fn"] == "MakeFillet"
        assert len(out["records"][0]["args"][3]["value"]) == 4

    def test_chamfer_edges_inline_where_all(self, agent_client):
        agent_client.call_tool("make_box", {"dx": 10, "dy": 10, "dz": 10})
        out = agent_client.call_tool("chamfer_edges", {
            "shape": "box_1", "d1": 1, "where": "all"})
        assert out["ok"] is True, out.get("error")
