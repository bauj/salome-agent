import pytest

from ._salome_root import SALOME_ROOT

pytestmark = [
    pytest.mark.salome,
    pytest.mark.skipif(SALOME_ROOT is None,
                       reason="set SALOME_ROOT or salome.root in config.json"),
]


def _call(client, name, **args):
    return client.call_tool(name, args, module="shaper")


def _ok(client, name, **args):
    r = _call(client, name, **args)
    assert r["ok"] is True, r.get("error")
    return r


def _face(client):
    sk = _ok(client, "create_sketch", plane="XOY")["sketch"]
    _ok(client, "sketch_add", sketch=sk, entities=[
        {"id": "R_1", "type": "rectangle", "corner1": [0, 0], "corner2": [10, 10]}])
    _ok(client, "close_sketch", sketch=sk)
    return sk, _ok(client, "make_face", sketch=sk, regions=[0])["name"]


def test_a_sketch_id_may_use_an_underscore_but_not_the_copy_suffix(agent_client):
    sk = _ok(agent_client, "create_sketch", plane="XOY")["sketch"]
    r = _ok(agent_client, "sketch_add", sketch=sk, entities=[
        {"id": "M_outline", "type": "polyline", "closed": True,
         "points": [[0, 0], [10, 0], [10, 10]]}])
    assert len(r["regions"]) == 1
    r = _call(agent_client, "sketch_add", sketch=sk, entities=[
        {"id": "P_op", "type": "circle", "center": [30, 0], "radius": 2}])
    assert r["ok"] is False and "reserved for sketch_copy" in r["error"]
    _ok(agent_client, "close_sketch", sketch=sk)


def test_a_used_up_operand_names_the_object_that_replaced_it(agent_client):
    a = _ok(agent_client, "make_box", dx=10, dy=10, dz=10, center=[5, 5, 5])["name"]
    b = _ok(agent_client, "make_box", dx=10, dy=10, dz=10, center=[10, 5, 5])["name"]
    fused = _ok(agent_client, "fuse", objects=[a, b])["name"]
    r = _call(agent_client, "translate", shape=a, dx=5, dy=0, dz=0)
    assert r["ok"] is False
    assert f"{a} was used up by {fused}" in r["error"] and f"use {fused}" in r["error"]
    _ok(agent_client, "translate", shape=fused, dx=5, dy=0, dz=0)


def test_get_subshapes_takes_one_by_default_and_names_its_arguments(agent_client):
    box = _ok(agent_client, "make_box", dx=10, dy=10, dz=10, center=[5, 5, 5])["name"]
    r = _ok(agent_client, "get_subshapes", shape=box, sub_type="face",
            where={"sorted_by": "z", "take": "max"})
    assert len(r["subshapes"]) == 1 and r["subshapes"][0]["centre"][2] == pytest.approx(10)
    r = _call(agent_client, "get_subshapes", shape=box, sub_type="face",
              filter={"sorted_by": "z"})
    assert r["ok"] is False and "not filter" in r["error"] and "`where`" in r["error"]
    r = _call(agent_client, "get_subshapes", shape=box, sub_type="face",
              where={"plane": {"point": [0, 0, 0], "normal": [0, 0, 1]}})
    assert r["ok"] is False and "none of sorted_by, by_size or on_plane" in r["error"]


def test_a_zero_length_extrusion_is_refused_before_shaper(agent_client):
    _, face = _face(agent_client)
    r = _call(agent_client, "extrude", base=face, size=5, from_size=-5)
    assert r["ok"] is False and "zero length" in r["error"] and "size -5" in r["error"]
    down = _ok(agent_client, "extrude", base=face, size=-5)["measure"]
    assert down["volume"] == pytest.approx(500) and down["bbox"][4] == pytest.approx(-5)
