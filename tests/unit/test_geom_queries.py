"""check_shape."""
from types import SimpleNamespace
from unittest.mock import MagicMock

from executor import geom as geom_tools


def _run(check):
    g = MagicMock()
    g.CheckShape.side_effect = check
    ex = SimpleNamespace(name_prefix="", namespace={"geompy": g}, geom_bindings={"f": "F"})
    return g, geom_tools.tool_check_shape(ex, {"shape": "f"})


def test_valid_shape():
    _, r = _run(lambda s, geom, status: (True, ""))
    assert r["ok"] is True and r["sanity"] == {"ok": True, "message": ""}


def test_geometry_check_that_crashes_falls_back_to_the_topology_check():
    """GEOM's geometry check can die in CORBA (MARSHAL_InvalidEnumValue) on an
    invalid shape; the topology-only check still says what is wrong."""
    def check(s, geom, status):
        if geom:
            raise RuntimeError("CORBA.MARSHAL(omniORB.MARSHAL_InvalidEnumValue, CORBA.COMPLETED_NO)")
        return False, "Unorientable Shape ... 1"
    g, r = _run(check)
    assert r["ok"] is False
    assert "Unorientable Shape" in r["sanity"]["message"]
    assert "geometry check failed" in r["sanity"]["message"]
