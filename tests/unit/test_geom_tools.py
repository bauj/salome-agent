import math
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from executor import geom_tools

_VALID_CATEGORIES = {
    "primitive", "sketch", "swept", "boolean",
    "transform", "feature", "display", "query",
}


def test_every_tool_has_a_valid_category():
    missing = [n for n in geom_tools.TOOLS if n not in geom_tools.TOOL_CATEGORY]
    assert not missing, f"tools with no TOOL_CATEGORY entry: {missing}"
    bad = {n: c for n, c in geom_tools.TOOL_CATEGORY.items() if c not in _VALID_CATEGORIES}
    assert not bad, f"tools with an unknown category: {bad}"
    extra = [n for n in geom_tools.TOOL_CATEGORY if n not in geom_tools.TOOLS]
    assert not extra, f"TOOL_CATEGORY names not in TOOLS: {extra}"


def test_list_tools_entries_carry_a_category():
    for entry in geom_tools.list_tools():
        assert entry["category"] in _VALID_CATEGORIES, entry


class TestToolDescriptions:
    def test_every_llm_facing_tool_has_a_description(self):
        for name in geom_tools.TOOLS:
            assert geom_tools.TOOL_DESCRIPTIONS.get(name), f"no description for {name}"

    def test_list_tools_carries_name_params_description(self):
        for t in geom_tools.list_tools():
            assert t["name"] and t["parameters"] and t["description"]


class TestMeasureTool:
    def _executor(self, bbox_by_shape):
        geompy = MagicMock()
        geompy.BoundingBox.side_effect = lambda s: bbox_by_shape[s]
        return SimpleNamespace(namespace={"geompy": geompy},
                               geom_bindings={n: n for n in bbox_by_shape})

    def test_measure_returns_shape_info_per_name(self):
        ex = self._executor({"box_1": [0.0, 10.0, 0.0, 20.0, 0.0, 30.0]})
        out = geom_tools.tool_measure(ex, {"names": ["box_1"]})
        assert out["ok"] is True
        assert out["records"] == []
        assert out["measurements"]["box_1"] == {
            "bbox": [0.0, 10.0, 0.0, 20.0, 0.0, 30.0],
            "size": [10.0, 20.0, 30.0],
            "centre": [5.0, 10.0, 15.0],
        }

    def test_measure_unknown_name_maps_to_none(self):
        ex = self._executor({"box_1": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]})
        out = geom_tools.tool_measure(ex, {"names": ["box_1", "ghost"]})
        assert out["measurements"]["ghost"] is None
        assert out["measurements"]["box_1"] is not None

    def test_measure_is_registered_with_schema_and_description(self):
        assert "measure" in geom_tools.TOOLS
        assert geom_tools.TOOL_SCHEMAS["measure"]["required"] == ["names"]
        assert geom_tools.TOOL_DESCRIPTIONS["measure"]


class TestResolverHelpers:
    """Unit tests for _resolve_axis, _resolve_plane, _resolve_point helpers."""

    def test_resolve_axis_shorthand_ox(self):
        """_resolve_axis converts 'OX' to a vector ref."""
        geom_tools.reset_counters()
        geompy = MagicMock()
        geompy.MakeVectorDXDYDZ.return_value = MagicMock(name="vec_obj")
        ex = SimpleNamespace(
            name_prefix="sa_",
            namespace={"geompy": geompy},
            geom_bindings={},
        )
        records = []
        result = geom_tools._resolve_axis(ex, "OX", records)

        # Should emit a vector CallRecord and return the name
        assert len(records) == 1
        assert records[0]["fn"] == "MakeVectorDXDYDZ"
        assert records[0]["args"] == [
            {"kind": "literal", "value": 1},
            {"kind": "literal", "value": 0},
            {"kind": "literal", "value": 0},
        ]
        assert len(records[0]["out_vars"]) == 1
        assert result == records[0]["out_vars"][0]

    def test_resolve_axis_shorthand_oy(self):
        """_resolve_axis converts 'OY' to a vector ref."""
        geom_tools.reset_counters()
        geompy = MagicMock()
        geompy.MakeVectorDXDYDZ.return_value = MagicMock(name="vec_obj")
        ex = SimpleNamespace(
            name_prefix="sa_",
            namespace={"geompy": geompy},
            geom_bindings={},
        )
        records = []
        result = geom_tools._resolve_axis(ex, "OY", records)

        assert len(records) == 1
        assert records[0]["args"] == [
            {"kind": "literal", "value": 0},
            {"kind": "literal", "value": 1},
            {"kind": "literal", "value": 0},
        ]
        assert result == records[0]["out_vars"][0]

    def test_resolve_axis_shorthand_oz(self):
        """_resolve_axis converts 'OZ' to a vector ref."""
        geom_tools.reset_counters()
        geompy = MagicMock()
        geompy.MakeVectorDXDYDZ.return_value = MagicMock(name="vec_obj")
        ex = SimpleNamespace(
            name_prefix="sa_",
            namespace={"geompy": geompy},
            geom_bindings={},
        )
        records = []
        result = geom_tools._resolve_axis(ex, "OZ", records)

        assert len(records) == 1
        assert records[0]["args"] == [
            {"kind": "literal", "value": 0},
            {"kind": "literal", "value": 0},
            {"kind": "literal", "value": 1},
        ]
        assert result == records[0]["out_vars"][0]

    def test_resolve_axis_ref(self):
        """_resolve_axis passes through a ref name unchanged."""
        geom_tools.reset_counters()
        ex = SimpleNamespace()
        records = []
        result = geom_tools._resolve_axis(ex, "my_vector", records)

        # No record should be emitted, just return the name
        assert len(records) == 0
        assert result == "my_vector"

    def test_resolve_point_accepts_a_coordinate_literal(self):
        """A real LLM passes a point arg as "(10,10,0)" / [10,10,0] instead of a
        shape name -> emit MakeVertex rather than KeyError on geom_bindings."""
        geom_tools.reset_counters()
        geompy = MagicMock()
        geompy.MakeVertex.return_value = MagicMock()
        ex = SimpleNamespace(name_prefix="sa_", namespace={"geompy": geompy}, geom_bindings={})
        for arg in ("(10, 10, 0)", "10,10,0", "[10,10,0]", [10, 10, 0], (10, 10, 0)):
            geom_tools.reset_counters()
            records = []
            name = geom_tools._resolve_point(ex, arg, records)
            assert records[0]["fn"] == "MakeVertex"
            assert [a["value"] for a in records[0]["args"]] == [10.0, 10.0, 0.0]
            assert name == records[0]["out_vars"][0]
            assert ex.geom_bindings[name] is geompy.MakeVertex.return_value

    def test_resolve_axis_accepts_a_coordinate_literal(self):
        geom_tools.reset_counters()
        geompy = MagicMock()
        geompy.MakeVectorDXDYDZ.return_value = MagicMock()
        ex = SimpleNamespace(name_prefix="sa_", namespace={"geompy": geompy}, geom_bindings={})
        records = []
        name = geom_tools._resolve_axis(ex, "0,0,1", records)
        assert records[0]["fn"] == "MakeVectorDXDYDZ"
        assert [a["value"] for a in records[0]["args"]] == [0.0, 0.0, 1.0]
        assert name == records[0]["out_vars"][0]

    def test_resolve_point_passes_a_plain_name_through(self):
        assert geom_tools._resolve_point(SimpleNamespace(), "my_point", []) == "my_point"

    def test_resolve_plane_shorthand_xoy(self):
        """_resolve_plane converts 'XOY' to a plane ref."""
        geom_tools.reset_counters()
        geompy = MagicMock()
        geompy.MakeVertex.return_value = MagicMock(name="pt_obj")
        geompy.MakeVectorDXDYDZ.return_value = MagicMock(name="vec_obj")
        geompy.MakePlane.return_value = MagicMock(name="plane_obj")
        ex = SimpleNamespace(
            name_prefix="sa_",
            namespace={"geompy": geompy},
            geom_bindings={},
        )
        records = []
        result = geom_tools._resolve_plane(ex, "XOY", records)

        # Should emit a plane CallRecord and return the name
        assert len(records) == 1
        assert records[0]["fn"] == "MakePlane"
        assert len(records[0]["out_vars"]) == 1
        assert result == records[0]["out_vars"][0]

    def test_resolve_plane_shorthand_yoz(self):
        """_resolve_plane converts 'YOZ' to a plane ref."""
        geom_tools.reset_counters()
        geompy = MagicMock()
        geompy.MakeVertex.return_value = MagicMock(name="pt_obj")
        geompy.MakeVectorDXDYDZ.return_value = MagicMock(name="vec_obj")
        geompy.MakePlane.return_value = MagicMock(name="plane_obj")
        ex = SimpleNamespace(
            name_prefix="sa_",
            namespace={"geompy": geompy},
            geom_bindings={},
        )
        records = []
        result = geom_tools._resolve_plane(ex, "YOZ", records)

        assert len(records) == 1
        assert records[0]["fn"] == "MakePlane"
        assert result == records[0]["out_vars"][0]

    def test_resolve_plane_shorthand_zox(self):
        """_resolve_plane converts 'ZOX' to a plane ref."""
        geom_tools.reset_counters()
        geompy = MagicMock()
        geompy.MakeVertex.return_value = MagicMock(name="pt_obj")
        geompy.MakeVectorDXDYDZ.return_value = MagicMock(name="vec_obj")
        geompy.MakePlane.return_value = MagicMock(name="plane_obj")
        ex = SimpleNamespace(
            name_prefix="sa_",
            namespace={"geompy": geompy},
            geom_bindings={},
        )
        records = []
        result = geom_tools._resolve_plane(ex, "ZOX", records)

        assert len(records) == 1
        assert records[0]["fn"] == "MakePlane"
        assert result == records[0]["out_vars"][0]

    def test_resolve_plane_ref(self):
        """_resolve_plane passes through a ref name unchanged."""
        geom_tools.reset_counters()
        ex = SimpleNamespace()
        records = []
        result = geom_tools._resolve_plane(ex, "my_plane", records)

        # No record should be emitted, just return the name
        assert len(records) == 0
        assert result == "my_plane"

    def test_resolve_point_shorthand_o(self):
        """_resolve_point converts 'O' to origin point ref."""
        geom_tools.reset_counters()
        geompy = MagicMock()
        geompy.MakeVertex.return_value = MagicMock(name="pt_obj")
        ex = SimpleNamespace(
            name_prefix="sa_",
            namespace={"geompy": geompy},
            geom_bindings={},
        )
        records = []
        result = geom_tools._resolve_point(ex, "O", records)

        # Should emit a point CallRecord and return the name
        assert len(records) == 1
        assert records[0]["fn"] == "MakeVertex"
        assert records[0]["args"] == [
            {"kind": "literal", "value": 0},
            {"kind": "literal", "value": 0},
            {"kind": "literal", "value": 0},
        ]
        assert len(records[0]["out_vars"]) == 1
        assert result == records[0]["out_vars"][0]

    def test_resolve_point_ref(self):
        """_resolve_point passes through a ref name unchanged."""
        geom_tools.reset_counters()
        ex = SimpleNamespace()
        records = []
        result = geom_tools._resolve_point(ex, "my_point", records)

        # No record should be emitted, just return the name
        assert len(records) == 0
        assert result == "my_point"


class TestPrimitiveTools:
    """Unit tests for primitive tools."""

    def test_make_box_two_pnt(self):
        """make_box_two_pnt creates a box from two points."""
        geom_tools.reset_counters()
        geompy = MagicMock()
        box = MagicMock()
        geompy.MakeBoxTwoPnt.return_value = box
        geompy.CheckShape.return_value = (True, "")
        ex = SimpleNamespace(
            name_prefix="sa_",
            namespace={"geompy": geompy},
            geom_bindings={"pt1": MagicMock(), "pt2": MagicMock()},
        )

        result = geom_tools.tool_make_box_two_pnt(ex, {"pnt1": "pt1", "pnt2": "pt2"})

        assert result["ok"] is True
        assert len(result["records"]) == 1
        record = result["records"][0]
        assert record["fn"] == "MakeBoxTwoPnt"
        assert record["args"][0] == {"kind": "ref", "value": "pt1"}
        assert record["args"][1] == {"kind": "ref", "value": "pt2"}

    def test_make_box_two_pnt_accepts_coordinate_literals(self):
        """`pnt1`/`pnt2` given as "(0,0,0)" / "(10,10,10)" -> MakeVertex records
        emitted, box built from them (was an opaque KeyError, cost ~3 calls)."""
        geom_tools.reset_counters()
        geompy = MagicMock()
        geompy.MakeBoxTwoPnt.return_value = MagicMock()
        geompy.MakeVertex.side_effect = [MagicMock(), MagicMock()]
        geompy.CheckShape.return_value = (True, "")
        ex = SimpleNamespace(name_prefix="sa_", namespace={"geompy": geompy}, geom_bindings={})

        result = geom_tools.tool_make_box_two_pnt(ex, {"pnt1": "(0,0,0)", "pnt2": "(10,10,10)"})

        assert result["ok"] is True
        assert [r["fn"] for r in result["records"]] == ["MakeVertex", "MakeVertex", "MakeBoxTwoPnt"]
        box_rec = result["records"][-1]
        assert box_rec["args"][0]["kind"] == "ref"
        assert box_rec["args"][0]["value"] == result["records"][0]["out_vars"][0]

    def test_make_cylinder(self):
        """make_cylinder creates a cylinder."""
        geom_tools.reset_counters()
        geompy = MagicMock()
        cyl = MagicMock()
        geompy.MakeCylinderRH.return_value = cyl
        geompy.CheckShape.return_value = (True, "")
        ex = SimpleNamespace(
            name_prefix="sa_",
            namespace={"geompy": geompy},
            geom_bindings={},
        )

        result = geom_tools.tool_make_cylinder(ex, {"radius": 5.0, "height": 10.0})

        assert result["ok"] is True
        assert len(result["records"]) == 1
        record = result["records"][0]
        assert record["fn"] == "MakeCylinderRH"
        assert record["args"][0] == {"kind": "literal", "value": 5.0}
        assert record["args"][1] == {"kind": "literal", "value": 10.0}

    def test_make_sphere_without_center(self):
        """make_sphere without center uses MakeSphereR."""
        geom_tools.reset_counters()
        geompy = MagicMock()
        sph = MagicMock()
        geompy.MakeSphereR.return_value = sph
        geompy.CheckShape.return_value = (True, "")
        ex = SimpleNamespace(
            name_prefix="sa_",
            namespace={"geompy": geompy},
            geom_bindings={},
        )

        result = geom_tools.tool_make_sphere(ex, {"radius": 5.0})

        assert result["ok"] is True
        assert len(result["records"]) == 1
        record = result["records"][0]
        assert record["fn"] == "MakeSphereR"
        assert record["args"][0] == {"kind": "literal", "value": 5.0}

    def test_make_sphere_with_center(self):
        """make_sphere with center uses MakeSpherePntR."""
        geom_tools.reset_counters()
        geompy = MagicMock()
        sph = MagicMock()
        geompy.MakeSpherePntR.return_value = sph
        geompy.CheckShape.return_value = (True, "")
        ex = SimpleNamespace(
            name_prefix="sa_",
            namespace={"geompy": geompy},
            geom_bindings={"center": MagicMock()},
        )

        result = geom_tools.tool_make_sphere(ex, {"radius": 5.0, "center": "center"})

        assert result["ok"] is True
        assert len(result["records"]) == 1
        record = result["records"][0]
        assert record["fn"] == "MakeSpherePntR"
        assert record["args"][0] == {"kind": "ref", "value": "center"}
        assert record["args"][1] == {"kind": "literal", "value": 5.0}

    def test_make_cone(self):
        """make_cone creates a cone."""
        geom_tools.reset_counters()
        geompy = MagicMock()
        cone = MagicMock()
        geompy.MakeConeR1R2H.return_value = cone
        geompy.CheckShape.return_value = (True, "")
        ex = SimpleNamespace(
            name_prefix="sa_",
            namespace={"geompy": geompy},
            geom_bindings={},
        )

        result = geom_tools.tool_make_cone(
            ex, {"radius1": 5.0, "radius2": 2.0, "height": 10.0}
        )

        assert result["ok"] is True
        assert len(result["records"]) == 1
        record = result["records"][0]
        assert record["fn"] == "MakeConeR1R2H"
        assert record["args"][0] == {"kind": "literal", "value": 5.0}
        assert record["args"][1] == {"kind": "literal", "value": 2.0}
        assert record["args"][2] == {"kind": "literal", "value": 10.0}

    def test_make_torus(self):
        """make_torus creates a torus."""
        geom_tools.reset_counters()
        geompy = MagicMock()
        torus = MagicMock()
        geompy.MakeTorusRR.return_value = torus
        geompy.CheckShape.return_value = (True, "")
        ex = SimpleNamespace(
            name_prefix="sa_",
            namespace={"geompy": geompy},
            geom_bindings={},
        )

        result = geom_tools.tool_make_torus(
            ex, {"radius_major": 10.0, "radius_minor": 3.0}
        )

        assert result["ok"] is True
        assert len(result["records"]) == 1
        record = result["records"][0]
        assert record["fn"] == "MakeTorusRR"
        assert record["args"][0] == {"kind": "literal", "value": 10.0}
        assert record["args"][1] == {"kind": "literal", "value": 3.0}

    def test_make_face(self):
        """make_face creates a rectangular face."""
        geom_tools.reset_counters()
        geompy = MagicMock()
        face = MagicMock()
        geompy.MakeFaceHW.return_value = face
        geompy.CheckShape.return_value = (True, "")
        ex = SimpleNamespace(
            name_prefix="sa_",
            namespace={"geompy": geompy},
            geom_bindings={},
        )

        result = geom_tools.tool_make_face(ex, {"height": 10.0, "width": 20.0})

        assert result["ok"] is True
        assert len(result["records"]) == 1
        record = result["records"][0]
        assert record["fn"] == "MakeFaceHW"
        assert record["args"][0] == {"kind": "literal", "value": 10.0}
        assert record["args"][1] == {"kind": "literal", "value": 20.0}
        assert record["args"][2] == {"kind": "literal", "value": 1}

    def test_make_disk(self):
        """make_disk creates a disk."""
        geom_tools.reset_counters()
        geompy = MagicMock()
        disk = MagicMock()
        geompy.MakeDiskR.return_value = disk
        geompy.CheckShape.return_value = (True, "")
        ex = SimpleNamespace(
            name_prefix="sa_",
            namespace={"geompy": geompy},
            geom_bindings={},
        )

        result = geom_tools.tool_make_disk(ex, {"radius": 5.0})

        assert result["ok"] is True
        assert len(result["records"]) == 1
        record = result["records"][0]
        assert record["fn"] == "MakeDiskR"
        assert record["args"][0] == {"kind": "literal", "value": 5.0}
        assert record["args"][1] == {"kind": "literal", "value": 1}


class TestBasicObjectTools:
    """Unit tests for basic object tools."""

    def test_make_point(self):
        """make_point creates a vertex."""
        geom_tools.reset_counters()
        geompy = MagicMock()
        pt = MagicMock()
        geompy.MakeVertex.return_value = pt
        ex = SimpleNamespace(
            name_prefix="sa_",
            namespace={"geompy": geompy},
            geom_bindings={},
        )

        result = geom_tools.tool_make_point(ex, {"x": 1.0, "y": 2.0, "z": 3.0})

        assert result["ok"] is True
        assert len(result["records"]) == 1
        record = result["records"][0]
        assert record["fn"] == "MakeVertex"
        assert record["args"][0] == {"kind": "literal", "value": 1.0}
        assert record["args"][1] == {"kind": "literal", "value": 2.0}
        assert record["args"][2] == {"kind": "literal", "value": 3.0}

    def test_make_vector(self):
        """make_vector creates a vector."""
        geom_tools.reset_counters()
        geompy = MagicMock()
        vec = MagicMock()
        geompy.MakeVectorDXDYDZ.return_value = vec
        ex = SimpleNamespace(
            name_prefix="sa_",
            namespace={"geompy": geompy},
            geom_bindings={},
        )

        result = geom_tools.tool_make_vector(ex, {"dx": 1.0, "dy": 2.0, "dz": 3.0})

        assert result["ok"] is True
        assert len(result["records"]) == 1
        record = result["records"][0]
        assert record["fn"] == "MakeVectorDXDYDZ"
        assert record["args"][0] == {"kind": "literal", "value": 1.0}
        assert record["args"][1] == {"kind": "literal", "value": 2.0}
        assert record["args"][2] == {"kind": "literal", "value": 3.0}

    def test_make_line(self):
        """make_line creates a line from two points."""
        geom_tools.reset_counters()
        geompy = MagicMock()
        line = MagicMock()
        geompy.MakeLineTwoPnt.return_value = line
        ex = SimpleNamespace(
            name_prefix="sa_",
            namespace={"geompy": geompy},
            geom_bindings={"pt1": MagicMock(), "pt2": MagicMock()},
        )

        result = geom_tools.tool_make_line(ex, {"pnt1": "pt1", "pnt2": "pt2"})

        assert result["ok"] is True
        assert len(result["records"]) == 1
        record = result["records"][0]
        assert record["fn"] == "MakeLineTwoPnt"
        assert record["args"][0] == {"kind": "ref", "value": "pt1"}
        assert record["args"][1] == {"kind": "ref", "value": "pt2"}

    def test_make_plane(self):
        """make_plane creates a plane from point and normal."""
        geom_tools.reset_counters()
        geompy = MagicMock()
        plane = MagicMock()
        geompy.MakePlane.return_value = plane
        geompy.CheckShape.return_value = (True, "")
        ex = SimpleNamespace(
            name_prefix="sa_",
            namespace={"geompy": geompy},
            geom_bindings={"pt": MagicMock(), "vec": MagicMock()},
        )

        result = geom_tools.tool_make_plane(ex, {"point": "pt", "normal": "vec"})

        assert result["ok"] is True
        assert len(result["records"]) == 1
        record = result["records"][0]
        assert record["fn"] == "MakePlane"
        assert record["args"][0] == {"kind": "ref", "value": "pt"}
        assert record["args"][1] == {"kind": "ref", "value": "vec"}
        assert record["args"][2] == {"kind": "literal", "value": 2000}

    def test_make_plane_with_trim_size(self):
        """make_plane accepts a custom trim_size."""
        geom_tools.reset_counters()
        geompy = MagicMock()
        plane = MagicMock()
        geompy.MakePlane.return_value = plane
        geompy.CheckShape.return_value = (True, "")
        ex = SimpleNamespace(
            name_prefix="sa_",
            namespace={"geompy": geompy},
            geom_bindings={"pt": MagicMock(), "vec": MagicMock()},
        )

        result = geom_tools.tool_make_plane(ex, {"point": "pt", "normal": "vec", "trim_size": 1000})

        assert result["ok"] is True
        record = result["records"][0]
        assert record["args"][2] == {"kind": "literal", "value": 1000}

    def test_make_circle(self):
        """make_circle creates a circle."""
        geom_tools.reset_counters()
        geompy = MagicMock()
        circ = MagicMock()
        geompy.MakeCircle.return_value = circ
        ex = SimpleNamespace(
            name_prefix="sa_",
            namespace={"geompy": geompy},
            geom_bindings={"center": MagicMock(), "normal": MagicMock()},
        )

        result = geom_tools.tool_make_circle(ex, {"center": "center", "normal": "normal", "radius": 5.0})

        assert result["ok"] is True
        assert len(result["records"]) == 1
        record = result["records"][0]
        assert record["fn"] == "MakeCircle"
        assert record["args"][0] == {"kind": "ref", "value": "center"}
        assert record["args"][1] == {"kind": "ref", "value": "normal"}
        assert record["args"][2] == {"kind": "literal", "value": 5.0}


class TestComplexObjectTools:
    """Unit tests for complex (swept/derived) object tools."""

    def test_make_prism(self):
        """make_prism creates a linear extrusion."""
        geom_tools.reset_counters()
        geompy = MagicMock()
        prism = MagicMock()
        geompy.MakePrismDXDYDZ.return_value = prism
        geompy.CheckShape.return_value = (True, "")
        ex = SimpleNamespace(
            name_prefix="sa_",
            namespace={"geompy": geompy},
            geom_bindings={"base": MagicMock()},
        )

        result = geom_tools.tool_make_prism(ex, {"base": "base", "dx": 1.0, "dy": 2.0, "dz": 3.0})

        assert result["ok"] is True
        assert len(result["records"]) == 1
        record = result["records"][0]
        assert record["fn"] == "MakePrismDXDYDZ"
        assert record["args"][0] == {"kind": "ref", "value": "base"}
        assert record["args"][1] == {"kind": "literal", "value": 1.0}

    def test_make_revolution(self):
        """make_revolution creates a revolution."""
        geom_tools.reset_counters()
        geompy = MagicMock()
        rev = MagicMock()
        geompy.MakeRevolution.return_value = rev
        geompy.CheckShape.return_value = (True, "")
        ex = SimpleNamespace(
            name_prefix="sa_",
            namespace={"geompy": geompy},
            geom_bindings={"base": MagicMock(), "axis": MagicMock()},
        )

        result = geom_tools.tool_make_revolution(ex, {"base": "base", "axis": "axis", "angle_deg": 180.0})

        assert result["ok"] is True
        assert len(result["records"]) == 1
        record = result["records"][0]
        assert record["fn"] == "MakeRevolution"
        assert record["args"][0] == {"kind": "ref", "value": "base"}
        assert record["args"][1] == {"kind": "ref", "value": "axis"}
        # Angle should be converted to radians
        assert record["args"][2]["kind"] == "literal"
        assert abs(record["args"][2]["value"] - math.radians(180.0)) < 1e-10

    def test_make_revolution_with_axis_shorthand(self):
        """make_revolution resolves axis shorthand."""
        geom_tools.reset_counters()
        geompy = MagicMock()
        rev = MagicMock()
        geompy.MakeRevolution.return_value = rev
        geompy.MakeVectorDXDYDZ.return_value = MagicMock()
        geompy.CheckShape.return_value = (True, "")
        ex = SimpleNamespace(
            name_prefix="sa_",
            namespace={"geompy": geompy},
            geom_bindings={"base": MagicMock()},
        )

        result = geom_tools.tool_make_revolution(ex, {"base": "base", "axis": "OZ", "angle_deg": 90.0})

        assert result["ok"] is True
        # Should have 2 records: one for vector, one for revolution
        assert len(result["records"]) == 2
        assert result["records"][0]["fn"] == "MakeVectorDXDYDZ"
        assert result["records"][1]["fn"] == "MakeRevolution"

    def test_make_pipe(self):
        """make_pipe creates a sweep."""
        geom_tools.reset_counters()
        geompy = MagicMock()
        pipe = MagicMock()
        geompy.MakePipe.return_value = pipe
        geompy.CheckShape.return_value = (True, "")
        ex = SimpleNamespace(
            name_prefix="sa_",
            namespace={"geompy": geompy},
            geom_bindings={"base": MagicMock(), "path": MagicMock()},
        )

        result = geom_tools.tool_make_pipe(ex, {"base": "base", "path": "path"})

        assert result["ok"] is True
        assert len(result["records"]) == 1
        record = result["records"][0]
        assert record["fn"] == "MakePipe"
        assert record["args"][0] == {"kind": "ref", "value": "base"}
        assert record["args"][1] == {"kind": "ref", "value": "path"}

    def test_make_thick_solid(self):
        """make_thick_solid creates a thick solid."""
        geom_tools.reset_counters()
        geompy = MagicMock()
        thick = MagicMock()
        geompy.MakeThickSolid.return_value = thick
        geompy.CheckShape.return_value = (True, "")
        ex = SimpleNamespace(
            name_prefix="sa_",
            namespace={"geompy": geompy},
            geom_bindings={"shape": MagicMock()},
        )

        result = geom_tools.tool_make_thick_solid(
            ex, {"shape": "shape", "thickness": 2.0, "faces": [3]})

        assert result["ok"] is True
        assert len(result["records"]) == 1
        record = result["records"][0]
        assert record["fn"] == "MakeThickSolid"
        assert record["args"][0] == {"kind": "ref", "value": "shape"}
        assert record["args"][1] == {"kind": "literal", "value": 2.0}
        assert record["args"][2] == {"kind": "literal", "value": [3]}
        assert record["args"][3] == {"kind": "literal", "value": False}
        geompy.MakeThickSolid.assert_called_once_with(ex.geom_bindings["shape"], 2.0, [3], False)

    def test_make_thick_solid_with_inside(self):
        """make_thick_solid accepts inside flag."""
        geom_tools.reset_counters()
        geompy = MagicMock()
        thick = MagicMock()
        geompy.MakeThickSolid.return_value = thick
        geompy.CheckShape.return_value = (True, "")
        ex = SimpleNamespace(
            name_prefix="sa_",
            namespace={"geompy": geompy},
            geom_bindings={"shape": MagicMock()},
        )

        result = geom_tools.tool_make_thick_solid(
            ex, {"shape": "shape", "thickness": 2.0, "faces": [3], "inside": True})

        assert result["ok"] is True
        record = result["records"][0]
        assert record["args"][3] == {"kind": "literal", "value": True}

    def test_make_filling(self):
        """make_filling creates a filling."""
        geom_tools.reset_counters()
        geompy = MagicMock()
        fill = MagicMock()
        geompy.MakeFilling.return_value = fill
        geompy.CheckShape.return_value = (True, "")
        ex = SimpleNamespace(
            name_prefix="sa_",
            namespace={"geompy": geompy},
            geom_bindings={"edge1": MagicMock(), "edge2": MagicMock()},
        )

        result = geom_tools.tool_make_filling(ex, {"contours": ["edge1", "edge2"]})

        assert result["ok"] is True
        assert len(result["records"]) == 1
        record = result["records"][0]
        assert record["fn"] == "MakeFilling"
        assert record["args"][0]["kind"] == "list"
        assert len(record["args"][0]["value"]) == 2


class TestBooleanTools:
    """Unit tests for boolean operation tools."""

    def test_fuse_two_shapes(self):
        """fuse with two shapes uses MakeFuse."""
        geom_tools.reset_counters()
        geompy = MagicMock()
        result_shape = MagicMock()
        geompy.MakeFuse.return_value = result_shape
        geompy.CheckShape.return_value = (True, "")
        ex = SimpleNamespace(
            name_prefix="sa_",
            namespace={"geompy": geompy},
            geom_bindings={"shape1": MagicMock(), "shape2": MagicMock()},
        )

        result = geom_tools.tool_fuse(ex, {"shapes": ["shape1", "shape2"]})

        assert result["ok"] is True
        assert len(result["records"]) == 1
        record = result["records"][0]
        assert record["fn"] == "MakeFuse"

    def test_fuse_multiple_shapes(self):
        """fuse with more than two shapes uses MakeFuseList."""
        geom_tools.reset_counters()
        geompy = MagicMock()
        result_shape = MagicMock()
        geompy.MakeFuseList.return_value = result_shape
        geompy.CheckShape.return_value = (True, "")
        ex = SimpleNamespace(
            name_prefix="sa_",
            namespace={"geompy": geompy},
            geom_bindings={"shape1": MagicMock(), "shape2": MagicMock(), "shape3": MagicMock()},
        )

        result = geom_tools.tool_fuse(ex, {"shapes": ["shape1", "shape2", "shape3"]})

        assert result["ok"] is True
        record = result["records"][0]
        assert record["fn"] == "MakeFuseList"

    def test_cut_with_single_tool(self):
        """cut with one tool shape uses MakeCut."""
        geom_tools.reset_counters()
        geompy = MagicMock()
        result_shape = MagicMock()
        geompy.MakeCut.return_value = result_shape
        geompy.CheckShape.return_value = (True, "")
        ex = SimpleNamespace(
            name_prefix="sa_",
            namespace={"geompy": geompy},
            geom_bindings={"main": MagicMock(), "tool": MagicMock()},
        )

        result = geom_tools.tool_cut(ex, {"main": "main", "tools": ["tool"]})

        assert result["ok"] is True
        record = result["records"][0]
        assert record["fn"] == "MakeCut"

    def test_cut_with_multiple_tools(self):
        """cut with multiple tool shapes uses MakeCutList."""
        geom_tools.reset_counters()
        geompy = MagicMock()
        result_shape = MagicMock()
        geompy.MakeCutList.return_value = result_shape
        geompy.CheckShape.return_value = (True, "")
        ex = SimpleNamespace(
            name_prefix="sa_",
            namespace={"geompy": geompy},
            geom_bindings={"main": MagicMock(), "tool1": MagicMock(), "tool2": MagicMock()},
        )

        result = geom_tools.tool_cut(ex, {"main": "main", "tools": ["tool1", "tool2"]})

        assert result["ok"] is True
        record = result["records"][0]
        assert record["fn"] == "MakeCutList"

    def test_common(self):
        """common with multiple shapes."""
        geom_tools.reset_counters()
        geompy = MagicMock()
        result_shape = MagicMock()
        geompy.MakeCommonList.return_value = result_shape
        geompy.CheckShape.return_value = (True, "")
        ex = SimpleNamespace(
            name_prefix="sa_",
            namespace={"geompy": geompy},
            geom_bindings={"shape1": MagicMock(), "shape2": MagicMock()},
        )

        result = geom_tools.tool_common(ex, {"shapes": ["shape1", "shape2"]})

        assert result["ok"] is True
        record = result["records"][0]
        assert record["fn"] == "MakeCommonList"

    def test_partition(self):
        """partition with shapes and optional tools."""
        geom_tools.reset_counters()
        geompy = MagicMock()
        result_shape = MagicMock()
        geompy.MakePartition.return_value = result_shape
        geompy.CheckShape.return_value = (True, "")
        ex = SimpleNamespace(
            name_prefix="sa_",
            namespace={"geompy": geompy},
            geom_bindings={"shape1": MagicMock(), "shape2": MagicMock()},
        )

        result = geom_tools.tool_partition(ex, {"shapes": ["shape1", "shape2"]})

        assert result["ok"] is True
        record = result["records"][0]
        assert record["fn"] == "MakePartition"
        assert record["args"][1] == {"kind": "list", "value": []}


class TestTransformationTools:
    """Unit tests for transformation tools."""

    def test_translate(self):
        """translate creates a translated copy."""
        geom_tools.reset_counters()
        geompy = MagicMock()
        result = MagicMock()
        geompy.MakeTranslation.return_value = result
        geompy.CheckShape.return_value = (True, "")
        ex = SimpleNamespace(
            name_prefix="sa_",
            namespace={"geompy": geompy},
            geom_bindings={"shape": MagicMock()},
        )

        result = geom_tools.tool_translate(ex, {"shape": "shape", "dx": 1.0, "dy": 2.0, "dz": 3.0})

        assert result["ok"] is True
        record = result["records"][0]
        assert record["fn"] == "MakeTranslation"

    def test_rotate(self):
        """rotate creates a rotated copy."""
        geom_tools.reset_counters()
        geompy = MagicMock()
        result = MagicMock()
        geompy.MakeRotation.return_value = result
        geompy.CheckShape.return_value = (True, "")
        ex = SimpleNamespace(
            name_prefix="sa_",
            namespace={"geompy": geompy},
            geom_bindings={"shape": MagicMock(), "axis": MagicMock()},
        )

        result = geom_tools.tool_rotate(ex, {"shape": "shape", "axis": "axis", "angle_deg": 90.0})

        assert result["ok"] is True
        record = result["records"][0]
        assert record["fn"] == "MakeRotation"

    def test_mirror_by_plane(self):
        """mirror by default plane uses MakeMirrorByPlane."""
        geom_tools.reset_counters()
        geompy = MagicMock()
        result = MagicMock()
        geompy.MakeMirrorByPlane.return_value = result
        geompy.CheckShape.return_value = (True, "")
        ex = SimpleNamespace(
            name_prefix="sa_",
            namespace={"geompy": geompy},
            geom_bindings={"shape": MagicMock(), "plane": MagicMock()},
        )

        result = geom_tools.tool_mirror(ex, {"shape": "shape", "entity": "plane"})

        assert result["ok"] is True
        record = result["records"][0]
        assert record["fn"] == "MakeMirrorByPlane"

    def test_mirror_by_axis(self):
        """mirror by axis uses MakeMirrorByAxis."""
        geom_tools.reset_counters()
        geompy = MagicMock()
        result = MagicMock()
        geompy.MakeMirrorByAxis.return_value = result
        geompy.CheckShape.return_value = (True, "")
        ex = SimpleNamespace(
            name_prefix="sa_",
            namespace={"geompy": geompy},
            geom_bindings={"shape": MagicMock(), "axis": MagicMock()},
        )

        result = geom_tools.tool_mirror(ex, {"shape": "shape", "by": "axis", "entity": "axis"})

        assert result["ok"] is True
        record = result["records"][0]
        assert record["fn"] == "MakeMirrorByAxis"

    def test_mirror_by_point(self):
        """mirror by point uses MakeMirrorByPoint."""
        geom_tools.reset_counters()
        geompy = MagicMock()
        result = MagicMock()
        geompy.MakeMirrorByPoint.return_value = result
        geompy.CheckShape.return_value = (True, "")
        ex = SimpleNamespace(
            name_prefix="sa_",
            namespace={"geompy": geompy},
            geom_bindings={"shape": MagicMock(), "point": MagicMock()},
        )

        result = geom_tools.tool_mirror(ex, {"shape": "shape", "by": "point", "entity": "point"})

        assert result["ok"] is True
        record = result["records"][0]
        assert record["fn"] == "MakeMirrorByPoint"

    def test_scale_uniform(self):
        """scale with single factor uses MakeScaleTransform."""
        geom_tools.reset_counters()
        geompy = MagicMock()
        result = MagicMock()
        geompy.MakeScaleTransform.return_value = result
        geompy.CheckShape.return_value = (True, "")
        ex = SimpleNamespace(
            name_prefix="sa_",
            namespace={"geompy": geompy},
            geom_bindings={"shape": MagicMock(), "center": MagicMock()},
        )

        result = geom_tools.tool_scale(ex, {"shape": "shape", "factor": 2.0, "center": "center"})

        assert result["ok"] is True
        record = result["records"][0]
        assert record["fn"] == "MakeScaleTransform"

    def test_scale_per_axis(self):
        """scale with per-axis factors uses MakeScaleAlongAxes."""
        geom_tools.reset_counters()
        geompy = MagicMock()
        result = MagicMock()
        geompy.MakeScaleAlongAxes.return_value = result
        geompy.CheckShape.return_value = (True, "")
        ex = SimpleNamespace(
            name_prefix="sa_",
            namespace={"geompy": geompy},
            geom_bindings={"shape": MagicMock(), "center": MagicMock()},
        )

        result = geom_tools.tool_scale(
            ex, {"shape": "shape", "factor_x": 2.0, "factor_y": 3.0, "factor_z": 4.0, "center": "center"}
        )

        assert result["ok"] is True
        record = result["records"][0]
        assert record["fn"] == "MakeScaleAlongAxes"

    def test_multi_translate_1d(self):
        """multi_translate 1D creates multiple copies."""
        geom_tools.reset_counters()
        geompy = MagicMock()
        result = MagicMock()
        geompy.MakeMultiTranslation1D.return_value = result
        geompy.CheckShape.return_value = (True, "")
        ex = SimpleNamespace(
            name_prefix="sa_",
            namespace={"geompy": geompy},
            geom_bindings={"shape": MagicMock(), "vector": MagicMock()},
        )

        result = geom_tools.tool_multi_translate(ex, {"shape": "shape", "vector": "vector", "step": 10.0, "count": 3})

        assert result["ok"] is True
        record = result["records"][0]
        assert record["fn"] == "MakeMultiTranslation1D"

    def test_multi_translate_2d(self):
        """multi_translate 2D with vector2 creates 2D grid."""
        geom_tools.reset_counters()
        geompy = MagicMock()
        result = MagicMock()
        geompy.MakeMultiTranslation2D.return_value = result
        geompy.CheckShape.return_value = (True, "")
        ex = SimpleNamespace(
            name_prefix="sa_",
            namespace={"geompy": geompy},
            geom_bindings={"shape": MagicMock(), "vector1": MagicMock(), "vector2": MagicMock()},
        )

        result = geom_tools.tool_multi_translate(
            ex,
            {
                "shape": "shape",
                "vector": "vector1",
                "step": 10.0,
                "count": 3,
                "vector2": "vector2",
                "step2": 20.0,
                "count2": 4,
            },
        )

        assert result["ok"] is True
        record = result["records"][0]
        assert record["fn"] == "MakeMultiTranslation2D"

    def test_multi_rotate_equal_angles(self):
        """multi_rotate without angle_deg creates equal-spaced copies."""
        geom_tools.reset_counters()
        geompy = MagicMock()
        result = MagicMock()
        geompy.MultiRotate1DNbTimes.return_value = result
        geompy.CheckShape.return_value = (True, "")
        ex = SimpleNamespace(
            name_prefix="sa_",
            namespace={"geompy": geompy},
            geom_bindings={"shape": MagicMock(), "axis": MagicMock()},
        )

        result = geom_tools.tool_multi_rotate(ex, {"shape": "shape", "axis": "axis", "count": 4})

        assert result["ok"] is True
        record = result["records"][0]
        assert record["fn"] == "MultiRotate1DNbTimes"

    def test_multi_rotate_by_step(self):
        """multi_rotate with angle_deg uses ByStep variant."""
        geom_tools.reset_counters()
        geompy = MagicMock()
        result = MagicMock()
        geompy.MultiRotate1DByStep.return_value = result
        geompy.CheckShape.return_value = (True, "")
        ex = SimpleNamespace(
            name_prefix="sa_",
            namespace={"geompy": geompy},
            geom_bindings={"shape": MagicMock(), "axis": MagicMock()},
        )

        result = geom_tools.tool_multi_rotate(ex, {"shape": "shape", "axis": "axis", "count": 4, "angle_deg": 45.0})

        assert result["ok"] is True
        record = result["records"][0]
        assert record["fn"] == "MultiRotate1DByStep"

    def test_offset_shape(self):
        """offset_shape creates an offset copy."""
        geom_tools.reset_counters()
        geompy = MagicMock()
        result = MagicMock()
        geompy.MakeOffset.return_value = result
        geompy.CheckShape.return_value = (True, "")
        ex = SimpleNamespace(
            name_prefix="sa_",
            namespace={"geompy": geompy},
            geom_bindings={"shape": MagicMock()},
        )

        result = geom_tools.tool_offset_shape(ex, {"shape": "shape", "offset": 5.0})

        assert result["ok"] is True
        record = result["records"][0]
        assert record["fn"] == "MakeOffset"

    def test_position_defaults_and_records(self):
        """position builds a target marker + MakePosition(shape, None, marker)."""
        geom_tools.reset_counters()
        geompy = MagicMock()
        geompy.MakeMarker.return_value = MagicMock()
        geompy.MakePosition.return_value = MagicMock()
        geompy.CheckShape.return_value = (True, "")
        ex = SimpleNamespace(name_prefix="sa_", namespace={"geompy": geompy},
                             geom_bindings={"shape": MagicMock()})

        result = geom_tools.tool_position(ex, {"shape": "shape", "origin": ["50", 0, 0]})

        assert result["ok"] is True
        # string coords coerced
        geompy.MakeMarker.assert_called_once_with(50.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
        marker_rec, pos_rec = result["records"]
        assert marker_rec["fn"] == "MakeMarker"
        assert pos_rec["fn"] == "MakePosition"
        assert pos_rec["args"][0] == {"kind": "ref", "value": "shape"}
        assert pos_rec["args"][1] == {"kind": "literal", "value": None}
        assert pos_rec["args"][2] == {"kind": "ref", "value": marker_rec["out_vars"][0]}
        # start LCS is the global frame -> literal None in the generated call
        geompy.MakePosition.assert_called_once_with(
            ex.geom_bindings["shape"], None, geompy.MakeMarker.return_value)


class TestLocalOpsTools:
    """Unit tests for local edge operation tools."""

    def test_fillet_edges_with_all(self):
        """fillet_edges with 'all' expands to all edge ids."""
        geom_tools.reset_counters()
        geompy = MagicMock()
        shape = MagicMock()
        filleted = MagicMock()
        geompy.SubShapeAllIDs.return_value = [1, 2, 3, 4, 5]  # 5 edges
        geompy.MakeFillet.return_value = filleted
        geompy.CheckShape.return_value = (True, "")
        ex = SimpleNamespace(
            name_prefix="sa_",
            namespace={"geompy": geompy},
            geom_bindings={"shape": shape},
        )

        result = geom_tools.tool_fillet_edges(ex, {"shape": "shape", "radius": 2.0, "edges": "all"})

        assert result["ok"] is True
        record = result["records"][0]
        assert record["fn"] == "MakeFillet"
        # Should have hardcoded EDGE type (6) and edge_ids list
        assert record["args"][2] == {"kind": "literal", "value": 6}  # geompy.ShapeType["EDGE"]
        assert record["args"][3] == {"kind": "literal", "value": [1, 2, 3, 4, 5]}

    def test_fillet_edges_with_inline_where_predicate(self):
        geom_tools.reset_counters()
        geompy = MagicMock()
        geompy.ShapeType = {"EDGE": 6}
        geompy.SubShapeAllSortedCentres.return_value = ["a", "b", "c"]
        geompy.SubShapeAllSortedCentresIDs.return_value = [10, 11, 12]
        geompy.BoundingBox.side_effect = lambda s: {
            "a": [0, 0, 0, 0, 0, 0], "b": [0, 0, 0, 0, 9, 9],
            "c": [0, 0, 0, 0, 5, 5]}[s]
        geompy.MakeFillet.return_value = MagicMock()
        geompy.CheckShape.return_value = (True, "")
        ex = SimpleNamespace(name_prefix="sa_", namespace={"geompy": geompy},
                             geom_bindings={"shape": MagicMock()})
        result = geom_tools.tool_fillet_edges(ex, {
            "shape": "shape", "radius": 2.0,
            "where": {"sorted_by": "z", "take": "max", "count": 1}})
        assert result["ok"] is True
        assert result["records"][0]["args"][3] == {"kind": "literal", "value": [11]}

    def test_chamfer_edges_with_inline_where_predicate(self):
        geom_tools.reset_counters()
        geompy = MagicMock()
        geompy.ShapeType = {"EDGE": 6}
        geompy.SubShapeAllSortedCentres.return_value = ["a", "b"]
        geompy.SubShapeAllSortedCentresIDs.return_value = [1, 2]
        geompy.BoundingBox.return_value = [0, 0, 0, 0, 0, 0]
        geompy.MakeChamferEdges.return_value = MagicMock()
        geompy.CheckShape.return_value = (True, "")
        ex = SimpleNamespace(name_prefix="sa_", namespace={"geompy": geompy},
                             geom_bindings={"shape": MagicMock()})
        result = geom_tools.tool_chamfer_edges(ex, {
            "shape": "shape", "d1": 1.0, "where": "all"})
        assert result["ok"] is True
        assert result["records"][0]["args"][3] == {"kind": "literal", "value": [1, 2]}

    def test_fillet_edges_without_edges_or_where_is_a_clean_error(self):
        geom_tools.reset_counters()
        geompy = MagicMock()
        ex = SimpleNamespace(name_prefix="sa_", namespace={"geompy": geompy},
                             geom_bindings={"shape": MagicMock()})
        result = geom_tools.tool_fillet_edges(ex, {"shape": "shape", "radius": 2.0})
        assert result["ok"] is False
        assert "where" in result["error"]

    def test_fillet_edges_with_edge_ids(self):
        """fillet_edges with explicit edge ids."""
        geom_tools.reset_counters()
        geompy = MagicMock()
        shape = MagicMock()
        filleted = MagicMock()
        geompy.MakeFillet.return_value = filleted
        geompy.CheckShape.return_value = (True, "")
        ex = SimpleNamespace(
            name_prefix="sa_",
            namespace={"geompy": geompy},
            geom_bindings={"shape": shape},
        )

        result = geom_tools.tool_fillet_edges(ex, {"shape": "shape", "radius": 2.0, "edges": [1, 3, 5]})

        assert result["ok"] is True
        record = result["records"][0]
        assert record["fn"] == "MakeFillet"
        assert record["args"][2] == {"kind": "literal", "value": 6}
        assert record["args"][3] == {"kind": "literal", "value": [1, 3, 5]}

    def test_chamfer_edges_with_all(self):
        """chamfer_edges with 'all' expands to all edge ids."""
        geom_tools.reset_counters()
        geompy = MagicMock()
        shape = MagicMock()
        chamfered = MagicMock()
        geompy.SubShapeAllIDs.return_value = [1, 2, 3, 4]
        geompy.MakeChamferEdges.return_value = chamfered
        geompy.CheckShape.return_value = (True, "")
        ex = SimpleNamespace(
            name_prefix="sa_",
            namespace={"geompy": geompy},
            geom_bindings={"shape": shape},
        )

        result = geom_tools.tool_chamfer_edges(ex, {"shape": "shape", "d1": 1.0, "edges": "all"})

        assert result["ok"] is True
        record = result["records"][0]
        assert record["fn"] == "MakeChamferEdges"
        # d2 should default to d1
        assert record["args"][1] == {"kind": "literal", "value": 1.0}
        assert record["args"][2] == {"kind": "literal", "value": 1.0}
        assert record["args"][3] == {"kind": "literal", "value": [1, 2, 3, 4]}
        # MakeChamferEdges(theShape, theD1, theD2, theEdges) -- no shape-type arg
        geompy.MakeChamferEdges.assert_called_once_with(shape, 1.0, 1.0, [1, 2, 3, 4])

    def _fillet_executor(self):
        geompy = MagicMock()
        geompy.MakeFillet.return_value = MagicMock()
        geompy.CheckShape.return_value = (True, "")
        return SimpleNamespace(name_prefix="sa_", namespace={"geompy": geompy},
                               geom_bindings={"shape": MagicMock()})

    def test_fillet_edges_accepts_a_string_id_list(self):
        """A real LLM has passed `edges` as "1,2,3" / "(1, 2, 3)" — normalise it
        rather than letting a bare string reach geompy (opaque CORBA error)."""
        geom_tools.reset_counters()
        ex = self._fillet_executor()
        for edges in ("2,4,6", "(2, 4, 6)", "[2,4,6]"):
            result = geom_tools.tool_fillet_edges(ex, {"shape": "shape", "radius": 1.0, "edges": edges})
            assert result["ok"] is True
            assert result["records"][0]["args"][3] == {"kind": "literal", "value": [2, 4, 6]}

    def test_fillet_edges_accepts_a_single_int(self):
        geom_tools.reset_counters()
        ex = self._fillet_executor()
        result = geom_tools.tool_fillet_edges(ex, {"shape": "shape", "radius": 1.0, "edges": 3})
        assert result["records"][0]["args"][3] == {"kind": "literal", "value": [3]}

    def test_fillet_edges_rejects_junk_edges_with_a_clear_error(self):
        ex = self._fillet_executor()
        # call_tool turns the ValueError into an {"ok": False} result for the model
        result = geom_tools.call_tool(ex, "fillet_edges",
                                      {"shape": "shape", "radius": 1.0, "edges": "the top ones"})
        assert result["ok"] is False
        assert "edges" in result["error"] and "integer" in result["error"]

    def test_fillet_edges_wraps_a_geompy_failure(self):
        """Radius too large -> geompy raises; the model gets a hint, not a bare
        RuntimeError repr."""
        ex = self._fillet_executor()
        ex.namespace["geompy"].MakeFillet.side_effect = RuntimeError(
            "MakeFilletEdges : Fillet can't be computed on the given shape with the given radius")
        result = geom_tools.tool_fillet_edges(ex, {"shape": "shape", "radius": 999.0, "edges": "all"})
        assert result["ok"] is False
        assert "smaller radius" in result["error"]

    def test_chamfer_edges_wraps_a_geompy_failure(self):
        geompy = MagicMock()
        geompy.MakeChamferEdges.side_effect = RuntimeError("can't be computed")
        ex = SimpleNamespace(name_prefix="sa_", namespace={"geompy": geompy},
                             geom_bindings={"shape": MagicMock()})
        result = geom_tools.tool_chamfer_edges(ex, {"shape": "shape", "d1": 99.0, "edges": [1, 2]})
        assert result["ok"] is False
        assert "chamfer_edges" in result["error"]

    def test_chamfer_edges_with_d2(self):
        """chamfer_edges with separate d2."""
        geom_tools.reset_counters()
        geompy = MagicMock()
        shape = MagicMock()
        chamfered = MagicMock()
        geompy.MakeChamferEdges.return_value = chamfered
        geompy.CheckShape.return_value = (True, "")
        ex = SimpleNamespace(
            name_prefix="sa_",
            namespace={"geompy": geompy},
            geom_bindings={"shape": shape},
        )

        result = geom_tools.tool_chamfer_edges(ex, {"shape": "shape", "d1": 1.0, "d2": 2.0, "edges": [1, 2]})

        assert result["ok"] is True
        record = result["records"][0]
        assert record["fn"] == "MakeChamferEdges"
        assert record["args"][1] == {"kind": "literal", "value": 1.0}
        assert record["args"][2] == {"kind": "literal", "value": 2.0}


class TestDisplayTools:
    """Unit tests for display tools."""

    def test_set_color(self, monkeypatch):
        """set_color creates a method_call record."""
        geom_tools.reset_counters()
        geompy = MagicMock()
        shape = MagicMock()

        # Mock SALOMEDS import
        mock_salomeds = MagicMock()
        mock_salomeds.Color.return_value = MagicMock()
        monkeypatch.setattr("sys.modules",
                          {**sys.modules, "salome": MagicMock(), "salome.kernel": MagicMock(SALOMEDS=mock_salomeds)},
                          raising=False)

        ex = SimpleNamespace(
            name_prefix="sa_",
            namespace={"geompy": geompy},
            geom_bindings={"shape": shape},
        )

        result = geom_tools.tool_set_color(ex, {"shape": "shape", "red": 255, "green": 0, "blue": 0})

        assert result["ok"] is True
        assert len(result["records"]) == 1
        record = result["records"][0]
        assert record["emit"] == "method_call"
        assert record["recv"] == "shape"
        assert record["fn"] == "SetColor"
        assert record["out_vars"] == []
        assert record["publish"] is None
        assert record["args"][0]["kind"] == "salomeds_color"
        assert record["args"][0]["value"] == [255, 0, 0]

    def test_set_transparency(self, monkeypatch):
        """set_transparency creates a gui_call record."""
        geom_tools.reset_counters()
        geompy = MagicMock()
        shape = MagicMock()

        # Mock salome imports
        mock_salome = MagicMock()
        mock_salome.ImportComponentGUI.return_value = MagicMock()
        mock_salome.ObjectToID.return_value = "obj_id"
        monkeypatch.setattr("sys.modules",
                          {**sys.modules, "salome": mock_salome},
                          raising=False)

        ex = SimpleNamespace(
            name_prefix="sa_",
            namespace={"geompy": geompy},
            geom_bindings={"shape": shape},
        )

        result = geom_tools.tool_set_transparency(ex, {"shape": "shape", "transparency": 0.5})

        assert result["ok"] is True
        record = result["records"][0]
        assert record["emit"] == "gui_call"
        assert record["recv"] == "shape"
        assert record["fn"] == "setTransparency"
        assert record["out_vars"] == []
        assert record["args"][0] == {"kind": "literal", "value": 0.5}

    def test_set_display_mode(self, monkeypatch):
        """set_display_mode creates a gui_call record."""
        geom_tools.reset_counters()
        geompy = MagicMock()
        shape = MagicMock()

        # Mock salome imports
        mock_salome = MagicMock()
        mock_salome.ImportComponentGUI.return_value = MagicMock()
        mock_salome.ObjectToID.return_value = "obj_id"
        monkeypatch.setattr("sys.modules",
                          {**sys.modules, "salome": mock_salome},
                          raising=False)

        ex = SimpleNamespace(
            name_prefix="sa_",
            namespace={"geompy": geompy},
            geom_bindings={"shape": shape},
        )

        result = geom_tools.tool_set_display_mode(ex, {"shape": "shape", "mode": 1})

        assert result["ok"] is True
        record = result["records"][0]
        assert record["emit"] == "gui_call"
        assert record["recv"] == "shape"
        assert record["fn"] == "setDisplayMode"
        assert record["out_vars"] == []
        assert record["args"][0] == {"kind": "literal", "value": 1}


class TestShapeInfoFeedback:
    """call_tool attaches bbox/size/centre feedback to a successful result."""

    def _ex(self, geompy):
        return SimpleNamespace(name_prefix="sa_", namespace={"geompy": geompy},
                               geom_bindings={})

    def test_call_tool_attaches_shape_info(self):
        geom_tools.reset_counters()
        geompy = MagicMock()
        geompy.MakeBoxDXDYDZ.return_value = MagicMock()
        geompy.CheckShape.return_value = (True, "")
        geompy.BoundingBox.return_value = [0.0, 10.0, 0.0, 20.0, 5.0, 8.0]
        ex = self._ex(geompy)

        result = geom_tools.call_tool(ex, "make_box", {"dx": 10, "dy": 20, "dz": 3})

        assert result["ok"] is True
        assert result["result_shape"] == "sa_box_1"
        assert result["shape_info"] == {
            "bbox": [0.0, 10.0, 0.0, 20.0, 5.0, 8.0],
            "size": [10.0, 20.0, 3.0],
            "centre": [5.0, 10.0, 6.5],
        }

    def test_no_shape_info_for_display_tools(self, monkeypatch):
        geom_tools.reset_counters()
        geompy = MagicMock()
        monkeypatch.setattr("sys.modules", {**sys.modules, "salome": MagicMock()}, raising=False)
        ex = SimpleNamespace(name_prefix="sa_", namespace={"geompy": geompy},
                             geom_bindings={"box_1": MagicMock()})

        result = geom_tools.call_tool(ex, "set_transparency", {"shape": "box_1", "transparency": 0.5})

        assert result["ok"] is True
        assert "shape_info" not in result

    def test_shape_info_survives_a_bounding_box_failure(self):
        geom_tools.reset_counters()
        geompy = MagicMock()
        geompy.MakeBoxDXDYDZ.return_value = MagicMock()
        geompy.CheckShape.return_value = (True, "")
        geompy.BoundingBox.side_effect = RuntimeError("nope")
        ex = self._ex(geompy)

        result = geom_tools.call_tool(ex, "make_box", {"dx": 1, "dy": 1, "dz": 1})

        assert result["ok"] is True
        assert "shape_info" not in result


class TestExplodeTool:
    def _executor(self):
        geompy = MagicMock()
        geompy.ShapeType = {"EDGE": 6, "FACE": 4, "SOLID": 2, "VERTEX": 7,
                            "WIRE": 5, "SHELL": 3}
        geompy.SubShapeAllSortedCentres.return_value = ["e1", "e2", "e3"]
        geompy.BoundingBox.return_value = [0, 1, 0, 1, 0, 1]
        return SimpleNamespace(namespace={"geompy": geompy},
                               geom_bindings={"box_1": "BOX"}, name_prefix="")

    def test_explode_publishes_each_subshape_and_emits_one_list_call(self):
        ex = self._executor()
        out = geom_tools.tool_explode(ex, {"shape": "box_1", "sub_type": "edge"})
        assert out["ok"] is True
        assert out["names"] == ["box_1_edge_1", "box_1_edge_2", "box_1_edge_3"]
        assert ex.geom_bindings["box_1_edge_2"] == "e2"
        (rec,) = out["records"]
        assert rec["fn"] == "SubShapeAllSortedCentres"
        assert rec["emit"] == "list_call"
        assert rec["out_vars"] == rec["publish"] == out["names"]
        assert rec["args"][0] == {"kind": "ref", "value": "box_1"}
        assert rec["args"][1] == {"kind": "literal", "value": 6}


class TestGetSubshapeTool:
    def _executor(self):
        geompy = MagicMock()
        geompy.GetSubShape.return_value = "SUB"
        geompy.BoundingBox.return_value = [0, 1, 0, 1, 0, 1]
        return SimpleNamespace(namespace={"geompy": geompy},
                               geom_bindings={"box_1": "BOX"}, name_prefix="")

    def test_get_subshape_builds_and_publishes_one_shape(self):
        ex = self._executor()
        out = geom_tools.tool_get_subshape(ex, {"shape": "box_1", "ids": [3, 7]})
        assert out["ok"] is True
        assert ex.geom_bindings["sub_1"] == "SUB"
        (rec,) = out["records"]
        assert rec["fn"] == "GetSubShape"
        assert rec["args"] == [{"kind": "ref", "value": "box_1"},
                               {"kind": "literal", "value": [3, 7]}]
        assert rec["out_vars"] == rec["publish"] == ["sub_1"]

    def test_get_subshape_coerces_string_ids_to_int(self):
        ex = self._executor()
        out = geom_tools.tool_get_subshape(ex, {"shape": "box_1", "ids": ["3", "7"]})
        assert out["records"][0]["args"][1] == {"kind": "literal", "value": [3, 7]}


class TestGetSubshapesQuery:
    def _executor(self, subs, ids, bboxes, props=None):
        geompy = MagicMock()
        geompy.ShapeType = {"EDGE": 6, "FACE": 4}
        geompy.SubShapeAllSortedCentres.return_value = subs
        geompy.SubShapeAllSortedCentresIDs.return_value = ids
        geompy.BoundingBox.side_effect = lambda s: bboxes[s]
        if props is not None:
            geompy.BasicProperties.side_effect = lambda s: props[s]
        return SimpleNamespace(namespace={"geompy": geompy},
                               geom_bindings={"box_1": "BOX"}, name_prefix="")

    def test_where_all_returns_every_id_and_centre(self):
        ex = self._executor(subs=["a", "b"], ids=[10, 11],
                            bboxes={"a": [0, 2, 0, 0, 0, 0], "b": [0, 0, 0, 4, 0, 0]})
        out = geom_tools.tool_get_subshapes(
            ex, {"shape": "box_1", "sub_type": "edge", "where": "all"})
        assert out["records"] == []
        assert out["ids"] == [10, 11]
        assert out["centres"] == [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]

    def test_where_sorted_by_z_take_max_count_2(self):
        ex = self._executor(subs=["a", "b", "c", "d"], ids=[1, 2, 3, 4],
                            bboxes={"a": [0, 0, 0, 0, 0, 0], "b": [0, 0, 0, 0, 10, 10],
                                    "c": [0, 0, 0, 0, 5, 5], "d": [0, 0, 0, 0, 10, 10]})
        out = geom_tools.tool_get_subshapes(ex, {
            "shape": "box_1", "sub_type": "edge",
            "where": {"sorted_by": "z", "take": "max", "count": 2}})
        assert sorted(out["ids"]) == [2, 4]

    def test_where_by_size_longest_count_1_uses_edge_length(self):
        ex = self._executor(subs=["a", "b"], ids=[1, 2],
                            bboxes={"a": [0] * 6, "b": [0] * 6},
                            props={"a": [3.0, 0.0, 0.0], "b": [9.0, 0.0, 0.0]})
        out = geom_tools.tool_get_subshapes(ex, {
            "shape": "box_1", "sub_type": "edge",
            "where": {"by_size": "longest", "count": 1}})
        assert out["ids"] == [2]

    def test_where_by_size_largest_count_1_uses_face_area(self):
        ex = self._executor(subs=["a", "b"], ids=[1, 2],
                            bboxes={"a": [0] * 6, "b": [0] * 6},
                            props={"a": [0.0, 2.0, 0.0], "b": [0.0, 8.0, 0.0]})
        out = geom_tools.tool_get_subshapes(ex, {
            "shape": "box_1", "sub_type": "face",
            "where": {"by_size": "largest", "count": 1}})
        assert out["ids"] == [2]

    def test_where_accepts_a_json_string_predicate(self):
        ex = self._executor(subs=["a", "b", "c", "d"], ids=[1, 2, 3, 4],
                            bboxes={"a": [0, 0, 0, 0, 0, 0], "b": [0, 0, 0, 0, 10, 10],
                                    "c": [0, 0, 0, 0, 5, 5], "d": [0, 0, 0, 0, 10, 10]})
        out = geom_tools.tool_get_subshapes(ex, {
            "shape": "box_1", "sub_type": "edge",
            "where": '{"sorted_by": "z", "take": "max", "count": 2}'})
        assert out["ok"] is True
        assert sorted(out["ids"]) == [2, 4]

    def test_unknown_where_is_an_error(self):
        ex = self._executor(subs=[], ids=[], bboxes={})
        out = geom_tools.tool_get_subshapes(
            ex, {"shape": "box_1", "sub_type": "edge", "where": {"bogus": 1}})
        assert out["ok"] is False


class TestMeasurementQueries:
    def _executor(self):
        geompy = MagicMock()
        return SimpleNamespace(namespace={"geompy": geompy},
                               geom_bindings={"a": "A", "b": "B"}, name_prefix="")

    def test_min_distance_returns_a_float(self):
        ex = self._executor()
        ex.namespace["geompy"].MinDistance.return_value = 4.23456
        out = geom_tools.tool_min_distance(ex, {"shape1": "a", "shape2": "b"})
        assert out["records"] == []
        assert out["distance"] == 4.2346

    def test_min_distance_tolerates_a_tuple_return(self):
        ex = self._executor()
        ex.namespace["geompy"].MinDistance.return_value = (5.0, 0, 0, 0, 5, 0, 0)
        out = geom_tools.tool_min_distance(ex, {"shape1": "a", "shape2": "b"})
        assert out["distance"] == 5.0

    def test_basic_properties_returns_length_area_volume(self):
        ex = self._executor()
        ex.namespace["geompy"].BasicProperties.return_value = [40.0, 100.0, 0.0]
        out = geom_tools.tool_basic_properties(ex, {"shape": "a"})
        assert out["records"] == []
        assert out["properties"] == {"length": 40.0, "area": 100.0, "volume": 0.0}


class TestAlignTool:
    def _executor(self, bbox_by_shape):
        geompy = MagicMock()
        objs = {n: f"<{n}>" for n in bbox_by_shape}
        geompy.BoundingBox.side_effect = lambda s: bbox_by_shape[
            next(n for n, o in objs.items() if o == s)]
        geompy.MakeTranslation.return_value = "<moved>"
        geompy.CheckShape.return_value = (True, "")
        return SimpleNamespace(name_prefix="sa_", namespace={"geompy": geompy},
                               geom_bindings=dict(objs))

    def test_single_axis_min_to_max_with_gap(self):
        geom_tools.reset_counters()
        ex = self._executor({
            "beam": [0.0, 10.0, 0.0, 5.0, 0.0, 5.0],
            "wall": [0.0, 100.0, 0.0, 100.0, 0.0, 100.0],
        })
        out = geom_tools.tool_align(ex, {
            "shape": "beam", "to": "wall",
            "match": [{"axis": "x", "my": "min", "target": "max", "gap": 10}]})
        assert out["ok"] is True
        geompy = ex.namespace["geompy"]
        geompy.MakeTranslation.assert_called_once_with("<beam>", 110.0, 0.0, 0.0)
        rec = out["records"][0]
        assert rec["fn"] == "MakeTranslation"
        assert rec["args"][0] == {"kind": "ref", "value": "beam"}
        assert [a["value"] for a in rec["args"][1:]] == [110.0, 0.0, 0.0]
        assert rec["out_vars"] == rec["publish"] == ["sa_tr_1"]

    def test_multi_axis_center_to_center(self):
        geom_tools.reset_counters()
        ex = self._executor({
            "lid": [0.0, 20.0, 0.0, 20.0, 0.0, 2.0],
            "pot": [10.0, 50.0, 10.0, 50.0, 0.0, 40.0],
        })
        out = geom_tools.tool_align(ex, {
            "shape": "lid", "to": "pot",
            "match": [{"axis": "x", "my": "center", "target": "center"},
                      {"axis": "y", "my": "center", "target": "center"},
                      {"axis": "z", "my": "min", "target": "max"}]})
        assert out["ok"] is True
        # lid centre x=10 -> pot centre x=30 => dx=20 ; same dy=20 ; z min 0 -> 40 => dz=40
        ex.namespace["geompy"].MakeTranslation.assert_called_once_with("<lid>", 20.0, 20.0, 40.0)

    def test_two_constraints_on_one_axis_is_an_error(self):
        geom_tools.reset_counters()
        ex = self._executor({"a": [0, 1, 0, 1, 0, 1], "b": [0, 1, 0, 1, 0, 1]})
        out = geom_tools.tool_align(ex, {
            "shape": "a", "to": "b",
            "match": [{"axis": "x", "my": "min", "target": "min"},
                      {"axis": "x", "my": "max", "target": "max"}]})
        assert out["ok"] is False
        assert "x" in out["error"]

    def test_unknown_shape_is_a_clean_error(self):
        geom_tools.reset_counters()
        ex = self._executor({"b": [0, 1, 0, 1, 0, 1]})
        out = geom_tools.tool_align(ex, {"shape": "ghost", "to": "b",
                                         "match": [{"axis": "x", "my": "min", "target": "min"}]})
        assert out["ok"] is False

    def test_registered_everywhere(self):
        assert geom_tools.TOOLS["align"] is geom_tools.tool_align
        assert geom_tools.TOOL_CATEGORY["align"] == "transform"
        assert geom_tools.TOOL_SCHEMAS["align"]["required"] == ["shape", "to", "match"]
        assert geom_tools.TOOL_DESCRIPTIONS["align"]

    def test_in_core_tools(self):
        from salome_agent.specialists.tool_selection import CORE_TOOLS
        assert "align" in CORE_TOOLS


class TestDeltaNote:
    """Anomaly-triggered numeric feedback on booleans and transforms (task 215)."""

    def _ex(self, geompy, bindings):
        return SimpleNamespace(namespace={"geompy": geompy}, geom_bindings=bindings)

    def _vols(self, geompy, mapping):
        geompy.BasicProperties.side_effect = lambda s: [0.0, 0.0, mapping[s]]

    def _bboxes(self, geompy, mapping):
        # mapping: shape -> [xmin, xmax, ymin, ymax, zmin, zmax]
        geompy.BoundingBox.side_effect = lambda s: mapping[s]

    def test_fuse_of_bbox_disjoint_parts_is_flagged(self):
        geompy = MagicMock()
        self._vols(geompy, {"a": 100.0, "b": 50.0, "fuse_1": 150.0})
        self._bboxes(geompy, {"a": [0, 10, 0, 10, 0, 10], "b": [50, 60, 0, 10, 0, 10]})
        ex = self._ex(geompy, {"a": "a", "b": "b", "fuse_1": "fuse_1"})
        note = geom_tools._delta_note(ex, "fuse", {"shapes": ["a", "b"]},
                                      {"result_shape": "fuse_1"})
        assert note and "bounding boxes are disjoint" in note

    def test_fuse_of_face_touching_parts_is_silent(self):
        # box_1 y[0,50] + tr_1 y[50,250] share a full face: volume adds up but
        # the bboxes abut, so the "parts are separated" note must not fire (task 222).
        geompy = MagicMock()
        self._vols(geompy, {"a": 100.0, "b": 50.0, "fuse_1": 150.0})
        self._bboxes(geompy, {"a": [0, 10, 0, 50, 0, 10], "b": [0, 10, 50, 250, 0, 10]})
        ex = self._ex(geompy, {"a": "a", "b": "b", "fuse_1": "fuse_1"})
        assert geom_tools._delta_note(ex, "fuse", {"shapes": ["a", "b"]},
                                      {"result_shape": "fuse_1"}) is None

    def test_fuse_of_overlapping_parts_is_silent(self):
        geompy = MagicMock()
        self._vols(geompy, {"a": 100.0, "b": 50.0, "fuse_1": 120.0})
        self._bboxes(geompy, {"a": [0, 10, 0, 10, 0, 10], "b": [5, 15, 0, 10, 0, 10]})
        ex = self._ex(geompy, {"a": "a", "b": "b", "fuse_1": "fuse_1"})
        assert geom_tools._delta_note(ex, "fuse", {"shapes": ["a", "b"]},
                                      {"result_shape": "fuse_1"}) is None

    def test_fuse_bbox_lookup_failure_suppresses_the_note(self):
        geompy = MagicMock()
        self._vols(geompy, {"a": 100.0, "b": 50.0, "fuse_1": 150.0})
        geompy.BoundingBox.side_effect = RuntimeError("no bbox")
        ex = self._ex(geompy, {"a": "a", "b": "b", "fuse_1": "fuse_1"})
        assert geom_tools._delta_note(ex, "fuse", {"shapes": ["a", "b"]},
                                      {"result_shape": "fuse_1"}) is None

    def test_cut_that_removes_nothing_is_flagged(self):
        geompy = MagicMock()
        self._vols(geompy, {"m": 100.0, "cut_1": 100.0})
        ex = self._ex(geompy, {"m": "m", "cut_1": "cut_1"})
        note = geom_tools._delta_note(ex, "cut", {"main": "m", "tools": ["x"]},
                                      {"result_shape": "cut_1"})
        assert note and "removed almost nothing" in note

    def test_cut_that_removes_everything_is_flagged(self):
        geompy = MagicMock()
        self._vols(geompy, {"m": 100.0, "cut_1": 0.1})
        ex = self._ex(geompy, {"m": "m", "cut_1": "cut_1"})
        note = geom_tools._delta_note(ex, "cut", {"main": "m", "tools": ["x"]},
                                      {"result_shape": "cut_1"})
        assert note and "removed almost everything" in note

    def test_normal_cut_is_silent(self):
        geompy = MagicMock()
        self._vols(geompy, {"m": 100.0, "cut_1": 60.0})
        ex = self._ex(geompy, {"m": "m", "cut_1": "cut_1"})
        assert geom_tools._delta_note(ex, "cut", {"main": "m", "tools": ["x"]},
                                      {"result_shape": "cut_1"}) is None

    def test_common_empty_intersection_is_flagged(self):
        geompy = MagicMock()
        self._vols(geompy, {"a": 100.0, "b": 80.0, "common_1": 0.0})
        ex = self._ex(geompy, {"a": "a", "b": "b", "common_1": "common_1"})
        note = geom_tools._delta_note(ex, "common", {"shapes": ["a", "b"]},
                                      {"result_shape": "common_1"})
        assert note and "empty or negligible" in note

    def test_rigid_transform_changing_volume_is_flagged(self):
        geompy = MagicMock()
        self._vols(geompy, {"src": 100.0, "tr_1": 130.0})
        ex = self._ex(geompy, {"src": "src", "tr_1": "tr_1"})
        note = geom_tools._delta_note(
            ex, "translate", {"shape": "src", "dx": 1, "dy": 0, "dz": 0},
            {"result_shape": "tr_1", "shape_info": None})
        assert note and "volume changed" in note

    def test_rigid_transform_that_did_not_move_is_flagged(self):
        geompy = MagicMock()
        self._vols(geompy, {"src": 100.0, "tr_1": 100.0})
        geompy.BoundingBox.return_value = [0.0, 10.0, 0.0, 10.0, 0.0, 10.0]
        ex = self._ex(geompy, {"src": "src", "tr_1": "tr_1"})
        note = geom_tools._delta_note(
            ex, "translate", {"shape": "src", "dx": 5, "dy": 0, "dz": 0},
            {"result_shape": "tr_1",
             "shape_info": {"bbox": [], "size": [], "centre": [5.0, 5.0, 5.0]}})
        assert note and "no effect" in note

    def test_good_rigid_transform_is_silent(self):
        geompy = MagicMock()
        self._vols(geompy, {"src": 100.0, "tr_1": 100.0})
        geompy.BoundingBox.return_value = [0.0, 10.0, 0.0, 10.0, 0.0, 10.0]
        ex = self._ex(geompy, {"src": "src", "tr_1": "tr_1"})
        note = geom_tools._delta_note(
            ex, "translate", {"shape": "src", "dx": 5, "dy": 0, "dz": 0},
            {"result_shape": "tr_1",
             "shape_info": {"bbox": [], "size": [], "centre": [10.0, 5.0, 5.0]}})
        assert note is None

    def test_scale_mismatch_is_flagged(self):
        geompy = MagicMock()
        self._vols(geompy, {"src": 100.0, "scale_1": 150.0})
        ex = self._ex(geompy, {"src": "src", "scale_1": "scale_1"})
        note = geom_tools._delta_note(ex, "scale", {"shape": "src", "factor": 2},
                                      {"result_shape": "scale_1"})
        assert note and "does not match" in note

    def test_scale_that_matches_is_silent(self):
        geompy = MagicMock()
        self._vols(geompy, {"src": 100.0, "scale_1": 800.0})
        ex = self._ex(geompy, {"src": "src", "scale_1": "scale_1"})
        assert geom_tools._delta_note(ex, "scale", {"shape": "src", "factor": 2},
                                      {"result_shape": "scale_1"}) is None

    def test_non_boolean_non_transform_is_silent(self):
        geompy = MagicMock()
        ex = self._ex(geompy, {"box_1": "box_1"})
        assert geom_tools._delta_note(ex, "make_box", {"dx": 1},
                                      {"result_shape": "box_1"}) is None

    def test_call_tool_attaches_delta_to_the_result(self):
        geom_tools.reset_counters()
        geompy = MagicMock()
        src = MagicMock()
        geompy.MakeTranslation.return_value = MagicMock()
        geompy.CheckShape.return_value = (True, "")
        geompy.BoundingBox.return_value = [0.0, 10.0, 0.0, 10.0, 0.0, 10.0]
        geompy.BasicProperties.side_effect = (
            lambda s: [0.0, 0.0, 1.0] if s is src else [0.0, 0.0, 9.0])
        ex = SimpleNamespace(name_prefix="sa_", namespace={"geompy": geompy},
                             geom_bindings={"box_1": src})
        result = geom_tools.call_tool(ex, "translate",
                                      {"shape": "box_1", "dx": 10, "dy": 0, "dz": 0})
        assert result["ok"] is True
        assert "volume changed" in result["delta"]


def test_every_registered_geom_tool_is_complete():
    from executor.geom_tools import registry

    for name in registry.names:
        assert registry.descriptions()[name], f"no description for {name}"
        assert registry.schemas()[name].get("type") == "object", f"bad schema for {name}"
        assert name in registry.arg_types(), f"no arg_types entry for {name}"
        assert registry.categories()[name] in _VALID_CATEGORIES, f"bad category for {name}"
