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


def _square(client, side=10, plane="XOY"):
    sk = _ok(client, "create_sketch", plane=plane)["sketch"]
    _ok(client, "sketch_add", sketch=sk, entities=[
        {"id": "R", "type": "rectangle", "corner1": [0, 0], "corner2": [side, side]}])
    return sk


def test_one_sketch_is_edited_at_a_time(agent_client):
    sk = _square(agent_client)
    r = _call(agent_client, "create_sketch", plane="XOZ")
    assert r["ok"] is False and f"{sk} is still being edited" in r["error"]
    r = _call(agent_client, "extrude", base=sk, size=5)
    assert r["ok"] is False and "close_sketch" in r["error"]
    r = _call(agent_client, "make_box", dx=1, dy=1, dz=1, center=[0.5, 0.5, 0.5])
    assert r["ok"] is False and "close_sketch" in r["error"]
    _ok(agent_client, "sketch_status", sketch=sk)
    _ok(agent_client, "list_study_objects")
    closed = _ok(agent_client, "close_sketch", sketch=sk)
    assert closed["records"][0]["code"] == "model.do()"
    r = _call(agent_client, "close_sketch", sketch=sk)
    assert r["ok"] is False and "not being edited" in r["error"]
    solid = _ok(agent_client, "extrude", base=sk, size=5)
    assert solid["measure"]["volume"] == pytest.approx(500)


def test_changing_a_closed_sketch_edits_it_again(agent_client):
    one = _square(agent_client)
    _ok(agent_client, "close_sketch", sketch=one)
    two = _square(agent_client, plane="XOZ")
    r = _call(agent_client, "sketch_add", sketch=one, entities=[
        {"id": "C", "type": "circle", "center": [30, 0], "radius": 2}])
    assert r["ok"] is False and f"{two} is still being edited" in r["error"]
    _ok(agent_client, "close_sketch", sketch=two)
    _ok(agent_client, "sketch_add", sketch=one, entities=[
        {"id": "C", "type": "circle", "center": [30, 0], "radius": 2}])
    r = _call(agent_client, "create_sketch", plane="YOZ")
    assert r["ok"] is False and f"{one} is still being edited" in r["error"]
    _ok(agent_client, "close_sketch", sketch=one)
    _ok(agent_client, "create_sketch", plane="YOZ")


def test_an_edit_that_breaks_a_later_feature_keeps_the_sketch_edited(agent_client):
    sk = _square(agent_client)
    _ok(agent_client, "close_sketch", sketch=sk)
    solid = _ok(agent_client, "extrude", base=sk, size=5)["name"]
    gone = _ok(agent_client, "sketch_status", sketch=sk)["entities"]["R1"]
    r = _ok(agent_client, "sketch_remove", sketch=sk, ids=["R1"])
    assert solid in r["breaks"]
    r = _call(agent_client, "close_sketch", sketch=sk)
    assert r["ok"] is False and solid in r["error"] and "stays edited" in r["error"]
    _ok(agent_client, "sketch_add", sketch=sk, entities=[
        {"id": "L", "type": "line", "start": gone["start"], "end": gone["end"]}])
    _ok(agent_client, "close_sketch", sketch=sk)
    m = _ok(agent_client, "measure", names=[solid])
    assert m["measurements"][solid]["volume"] == pytest.approx(500)


def test_a_sketch_projects_only_what_was_built_before_it(agent_client):
    sk = _square(agent_client)
    _ok(agent_client, "close_sketch", sketch=sk)
    box = _ok(agent_client, "make_box", dx=10, dy=20, dz=30, center=[5, 10, 15])["name"]
    edges = _ok(agent_client, "get_subshapes", shape=box, sub_type="edge")["subshapes"]
    across = [u["id"] for u in edges if u["size"] == pytest.approx(20)][0]
    r = _call(agent_client, "sketch_add", sketch=sk, entities=[
        {"id": "P", "type": "projection", "edge": across, "of": box}])
    assert r["ok"] is False and "built after" in r["error"]


def test_finish_check_names_the_sketch_being_edited(agent_client):
    sk = _square(agent_client)
    r = agent_client.check_results([sk], module="shaper")
    assert r["results"][sk]["status"] == "editing"
