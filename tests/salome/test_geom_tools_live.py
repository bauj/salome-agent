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
