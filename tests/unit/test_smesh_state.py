import types

import pytest

from executor.smesh import state as sm


class _Exec:
    def __init__(self):
        self.name_prefix = ""
        self.geom_bindings = {"box_1": object()}
        self.namespace = {}


def _st(ex):
    st = sm.SmeshState({})
    ex.__dict__["smesh_state"] = st
    return st


def test_shape_ref_geom_binds_the_shape_into_the_namespace():
    ex = _Exec()
    st = _st(ex)
    assert sm.shape_ref(ex, "box_1") == ("box_1", "geom")
    assert st.ns["box_1"] is ex.geom_bindings["box_1"]


def test_shape_ref_shaper_uses_the_study_lookup():
    ex = _Exec()
    _st(ex)
    ex.__dict__["shaper_state"] = types.SimpleNamespace(objects={"plate_1": "solid"}, groups={})
    assert sm.shape_ref(ex, "plate_1") == ('_shaper_shape("plate_1")', "shaper")


def test_shape_ref_unknown_lists_known_names():
    ex = _Exec()
    _st(ex)
    with pytest.raises(ValueError, match="box_1"):
        sm.shape_ref(ex, "nope")


def test_group_ref_for_both_owners():
    ex = _Exec()
    st = _st(ex)
    st.geo_groups["inlet"] = sm.GeoGroup("plate_1", "shaper", "face")
    st.geo_groups["wall"] = sm.GeoGroup("box_1", "geom", "face")
    assert sm.group_ref(ex, "inlet")[0] == '_shaper_shape("inlet")'
    assert sm.group_ref(ex, "wall")[0] == "wall"


def test_mesh_of_unknown():
    ex = _Exec()
    _st(ex)
    with pytest.raises(ValueError, match="create_mesh"):
        sm.mesh_of(ex, "mesh_9")


def test_preamble_keeps_both_import_forms_and_the_smesh_init_lines():
    assert "from salome.kernel import SMESH, StdMeshers" in sm.PREAMBLE
    assert "import SMESH, StdMeshers" in sm.PREAMBLE
    assert sm.PREAMBLE.startswith("from salome.smesh import smeshBuilder\nsmesh = smeshBuilder.New()\n")


def test_timeout_for_compute_only():
    from executor import smesh
    assert smesh.timeout_for("compute") == sm.COMPUTE_TIMEOUT_S
    assert smesh.timeout_for("create_mesh") is None


def test_core_registers_smesh():
    from agent_server import core
    assert core._TOOL_MODULES["smesh"].__name__ == "executor.smesh"
    assert core._REFRESH["smesh"] == "SMESH"
