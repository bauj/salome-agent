import pytest

from ._salome_root import SALOME_ROOT

pytestmark = [pytest.mark.salome,
              pytest.mark.skipif(SALOME_ROOT is None,
                                 reason="set SALOME_ROOT or salome.root in config.json")]


def _ok(client, tool, **args):
    r = client.call_tool(tool, args, module="shaper")
    assert r["ok"] is True, r.get("error")
    return r


def test_make_group_of_top_face(agent_client):
    b = _ok(agent_client, "make_box", dx=10, dy=20, dz=30, center=[5, 10, 15])["name"]
    top = _ok(agent_client, "get_subshapes", shape=b, sub_type="face",
              where={"sorted_by": "z", "take": "max", "count": 1})["subshapes"]
    g = _ok(agent_client, "make_group", shape=b, ids=[top[0]["id"]], name="top")
    assert g["name"] == "top" and g["count"] == 1
    assert g["measure"]["area"] == pytest.approx(200)
    listed = {o["name"] for o in _ok(agent_client, "list_study_objects")["objects"]}
    assert "top" in listed


def test_make_group_name_taken(agent_client):
    b = _ok(agent_client, "make_box", dx=10, dy=10, dz=10, center=[5, 5, 5])["name"]
    _ok(agent_client, "make_group", shape=b, ids=["F1"], name="side")
    r = agent_client.call_tool("make_group", {"shape": b, "ids": ["F2"], "name": "side"},
                               module="shaper")
    assert r["ok"] is False and "taken" in r["error"]
