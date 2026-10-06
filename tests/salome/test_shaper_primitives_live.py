import math

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


def test_list_tools_has_the_primitives_and_queries(agent_client):
    names = {t["name"] for t in agent_client.list_tools(module="shaper")["tools"]}
    assert {"make_box", "make_box_two_pnt", "make_cylinder", "make_sphere", "make_cone",
            "make_torus", "make_point", "make_axis", "make_plane", "check_shape",
            "list_study_objects", "min_distance", "basic_properties", "set_color",
            "set_transparency"} <= names


def test_box_from_dimensions_and_from_two_corners(agent_client):
    box = _ok(agent_client, "make_box", dx=10, dy=20, dz=30)["measure"]
    assert box["volume"] == pytest.approx(6000)
    assert box["bbox"] == pytest.approx([0, 10, 0, 20, 0, 30])
    two = _ok(agent_client, "make_box_two_pnt", pnt1=[1, 2, 3], pnt2="11,12,13")["measure"]
    assert two["bbox"] == pytest.approx([1, 11, 2, 12, 3, 13])
    assert two["faces"] == 6


def test_round_primitives_are_placed_by_origin_and_axis(agent_client):
    cyl = _ok(agent_client, "make_cylinder", radius=5, height=10, origin=[10, 0, 0],
              axis="OX")["measure"]
    assert cyl["bbox"] == pytest.approx([10, 20, -5, 5, -5, 5], abs=1e-6)
    assert cyl["volume"] == pytest.approx(math.pi * 25 * 10, rel=1e-6)
    sph = _ok(agent_client, "make_sphere", radius=10, center=[0, 0, 5])["measure"]
    assert sph["bbox"] == pytest.approx([-10, 10, -10, 10, -5, 15], abs=1e-6)
    cone = _ok(agent_client, "make_cone", radius1=10, radius2=0, height=20)["measure"]
    assert cone["volume"] == pytest.approx(math.pi * 100 * 20 / 3, rel=1e-6)
    torus = _ok(agent_client, "make_torus", radius_major=15, radius_minor=3,
                axis="OY")["measure"]
    assert torus["bbox"] == pytest.approx([-18, 18, -3, 3, -18, 18], abs=1e-3)


def test_construction_point_axis_and_plane(agent_client):
    p = _ok(agent_client, "make_point", x=1, y=2, z=3)
    assert p["measure"] == {"kind": "point", "at": [1, 2, 3]}
    ax = _ok(agent_client, "make_axis", point=p["name"], direction=[0, 0, 2])
    assert ax["measure"]["through"] == [1, 2, 3]
    assert ax["measure"]["direction"] == pytest.approx([0, 0, 1])
    cyl = _ok(agent_client, "make_cylinder", radius=1, height=4, origin=p["name"],
              axis=ax["name"])["measure"]
    assert cyl["bbox"] == pytest.approx([0, 2, 1, 3, 3, 7], abs=1e-6)
    pl = _ok(agent_client, "make_plane", point=[0, 0, 5], normal=[0, 0, 1])["measure"]
    assert pl["normal"] == pytest.approx([0, 0, 1]) and pl["point"][2] == pytest.approx(5)
    m = _ok(agent_client, "measure", names=[p["name"]])["measurements"]
    assert m[p["name"]]["at"] == [1, 2, 3]


def test_a_construction_object_is_not_a_boolean_operand(agent_client):
    box = _ok(agent_client, "make_box", dx=5, dy=5, dz=5)["name"]
    pl = _ok(agent_client, "make_plane", base="XOY", offset=2)["name"]
    r = _call(agent_client, "cut", objects=[box], tools=[pl])
    assert r["ok"] is False and "construction plane" in r["error"]


def test_bad_arguments_are_refused_with_a_message(agent_client):
    r = _call(agent_client, "make_torus", radius_major=3, radius_minor=5)
    assert r["ok"] is False and "smaller" in r["error"]
    r = _call(agent_client, "make_cylinder", radius=1, height=2, origin="nope")
    assert r["ok"] is False and "make_point" in r["error"]
    r = _call(agent_client, "make_cylinder", radius=1, height=2, axis="nope")
    assert r["ok"] is False and "make_axis" in r["error"]


def test_min_distance_of_separate_overlapping_and_nested_solids(agent_client):
    # SHAPER's own solid-to-solid distance crashes on overlapping solids
    a = _ok(agent_client, "make_box", dx=10, dy=10, dz=10)["name"]
    far = _ok(agent_client, "make_box_two_pnt", pnt1=[30, 0, 0], pnt2=[40, 10, 10])["name"]
    over = _ok(agent_client, "make_cylinder", radius=5, height=10)["name"]
    inner = _ok(agent_client, "make_sphere", radius=1, center=[5, 5, 5])["name"]
    assert _ok(agent_client, "min_distance", shape1=a, shape2=far)["distance"] == pytest.approx(20)
    assert _ok(agent_client, "min_distance", shape1=a, shape2=over)["distance"] == 0
    assert _ok(agent_client, "min_distance", shape1=inner, shape2=a)["distance"] == 0
    assert _ok(agent_client, "list_study_objects")["ok"] is True   # server still up


def test_queries_check_list_and_properties(agent_client):
    a = _ok(agent_client, "make_box", dx=10, dy=10, dz=10)["name"]
    b = _ok(agent_client, "make_box_two_pnt", pnt1=[5, 0, 0], pnt2=[15, 10, 10])["name"]
    f = _ok(agent_client, "fuse", objects=[a, b])["name"]
    chk = _ok(agent_client, "check_shape", shape=f)
    assert chk["valid"] is True
    props = _ok(agent_client, "basic_properties", shape=f)
    assert props["volume"] == pytest.approx(1500) and props["length"] == 0
    listed = {o["name"]: o for o in _ok(agent_client, "list_study_objects")["objects"]}
    assert listed[a].get("used_up") is True and listed[b].get("used_up") is True
    assert listed[f]["kind"] == "boolean" and "used_up" not in listed[f]


def test_colour_and_transparency_reach_the_result_and_result_py(agent_client, tmp_path):
    import re
    import subprocess

    from salome_agent.nodes.assembler import assemble

    from .conftest import _salome_subprocess_env

    calls = []

    def call(name, **args):
        r = _ok(agent_client, name, **args)
        calls.append({"tool": name, "tool_args": args, "records": r["records"]})
        return r

    pt = call("make_point", x=0, y=0, z=4)["name"]
    cyl = call("make_cylinder", radius=4, height=6, origin=pt)
    box = call("make_box_two_pnt", pnt1=[-10, -10, 0], pnt2=[10, 10, 10])["name"]
    cut = call("cut", objects=[box], tools=[cyl["name"]])["name"]
    call("set_color", shape=cut, red=255, green=0, blue=0)
    call("set_transparency", shape=cut, transparency=0.25)
    log = [rec for c in calls for rec in c["records"]]
    script = assemble({"stages": {0: {"module": "shaper", "status": "ok", "script": None,
                                      "result_names": [cut], "call_log": log,
                                      "calls": calls}},
                       "run_id": "t", "user_request": "box", "plan": []})
    script += ("import ModelAPI\n"
               "from GeomAlgoAPI import GeomAlgoAPI_ShapeTools as _T\n"
               f"_r = {cut}.feature().firstResult()\n"
               "_c = _r.data().intArray(ModelAPI.ModelAPI_Result.COLOR_ID())\n"
               "print('VOLUME', _T.volume(_r.shape()), 'COLOR',"
               " [_c.value(i) for i in range(_c.size())], 'T',"
               " _r.data().real(ModelAPI.ModelAPI_Result.TRANSPARENCY_ID()).value())\n")
    path = tmp_path / "result.py"
    path.write_text(script)
    out = subprocess.run([f"{SALOME_ROOT}/salome", "shell", "--", "python3", str(path)],
                         env=_salome_subprocess_env(), capture_output=True, text=True,
                         timeout=240)
    m = re.search(r"VOLUME ([0-9.eE+-]+) COLOR \[(.*?)\] T ([0-9.]+)", out.stdout)
    assert m, out.stdout[-2000:] + out.stderr[-2000:]
    assert float(m.group(1)) == pytest.approx(4000 - math.pi * 16 * 6, abs=1e-3)
    assert m.group(2) == "255, 0, 0"
    assert float(m.group(3)) == pytest.approx(0.25)


def test_an_offset_plane_and_sketch_on_xoz_go_to_plus_y(agent_client):
    off = _ok(agent_client, "make_plane", base="XOZ", offset=7)["measure"]
    assert off["point"][1] == pytest.approx(7)
    sk = _ok(agent_client, "create_sketch", plane={"base": "XOZ", "offset": 7})["sketch"]
    _ok(agent_client, "sketch_add", sketch=sk, entities=[
        {"id": "C", "type": "circle", "center": [0, 0], "radius": 2}])
    ext = _ok(agent_client, "extrude", base=sk, size=3)["measure"]
    assert sorted(ext["bbox"][2:4]) in (pytest.approx([4, 7]), pytest.approx([7, 10]))
