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


def _box(client, dx=10, dy=10, dz=10):
    return _ok(client, "make_box", dx=dx, dy=dy, dz=dz)["name"]


def test_translate_moves_the_shape_and_uses_it_up(agent_client):
    b = _box(agent_client)
    t = _ok(agent_client, "translate", shape=b, dx=5, dy=0, dz=-2)
    assert t["measure"]["bbox"] == pytest.approx([5, 15, 0, 10, -2, 8])
    listed = {o["name"]: o for o in _ok(agent_client, "list_study_objects")["objects"]}
    assert listed[b].get("used_up") is True
    r = _call(agent_client, "translate", shape="nope", dx=1, dy=0, dz=0)
    assert r["ok"] is False


def test_rotate_about_oz(agent_client):
    r = _ok(agent_client, "rotate", shape=_box(agent_client, 10, 20, 5), axis="OZ",
            angle_deg=90)
    assert r["measure"]["bbox"] == pytest.approx([-20, 0, 0, 10, 0, 5], abs=1e-6)


def test_mirror_moves_or_keeps_the_original(agent_client):
    moved = _ok(agent_client, "mirror", shape=_box(agent_client), entity="YOZ")["measure"]
    assert moved["bbox"] == pytest.approx([-10, 0, 0, 10, 0, 10], abs=1e-6)
    both = _ok(agent_client, "mirror", shape=_box(agent_client), entity="XOY",
               keep_original=True)["measure"]
    assert both["volume"] == pytest.approx(2000)
    assert both["bbox"] == pytest.approx([0, 10, 0, 10, -10, 10], abs=1e-6)
    pt = _ok(agent_client, "make_point", x=20, y=0, z=0)["name"]
    about = _ok(agent_client, "mirror", shape=_box(agent_client), entity=pt)["measure"]
    assert about["bbox"] == pytest.approx([30, 40, -10, 0, -10, 0], abs=1e-6)


def test_scale_uniform_and_per_axis(agent_client):
    s = _ok(agent_client, "scale", shape=_box(agent_client), factor=2, center=[10, 10, 10])
    assert s["measure"]["bbox"] == pytest.approx([-10, 10, -10, 10, -10, 10], abs=1e-6)
    s = _ok(agent_client, "scale", shape=_box(agent_client), factor_x=2, factor_z=0.5)
    assert s["measure"]["size"] == pytest.approx([20, 10, 5])
    r = _call(agent_client, "scale", shape=_box(agent_client), factor=2, factor_x=3)
    assert r["ok"] is False and "not both" in r["error"]


def test_patterns_count_the_original(agent_client):
    row = _ok(agent_client, "multi_translate", shape=_box(agent_client), vector="OX",
              step=20, count=3)["measure"]
    assert row["volume"] == pytest.approx(3000) and row["size"][0] == pytest.approx(50)
    grid = _ok(agent_client, "multi_translate", shape=_box(agent_client), vector="OX",
               step=20, count=3, vector2="OY", step2=15, count2=2)["measure"]
    assert grid["volume"] == pytest.approx(6000) and grid["size"][1] == pytest.approx(25)
    peg = _ok(agent_client, "make_cylinder", radius=2, height=5, origin=[20, 0, 0])["name"]
    ring = _ok(agent_client, "multi_rotate", shape=peg, axis="OZ", count=4)["measure"]
    assert ring["volume"] == pytest.approx(4 * math.pi * 4 * 5, rel=1e-6)
    assert ring["bbox"] == pytest.approx([-22, 22, -22, 22, 0, 5], abs=1e-6)
    peg = _ok(agent_client, "make_cylinder", radius=2, height=5, origin=[20, 0, 0])["name"]
    arc = _ok(agent_client, "multi_rotate", shape=peg, axis="OZ", count=3,
              angle_deg=90)["measure"]
    assert arc["bbox"] == pytest.approx([-22, 22, -2, 22, 0, 5], abs=1e-6)


def test_offset_grows_and_shrinks(agent_client):
    inner = _ok(agent_client, "offset_shape", shape=_box(agent_client), offset=-1)["measure"]
    assert inner["volume"] == pytest.approx(512, rel=1e-6)
    outer = _ok(agent_client, "offset_shape", shape=_box(agent_client), offset=2)["measure"]
    assert outer["bbox"] == pytest.approx([-2, 12, -2, 12, -2, 12], abs=1e-6)


def test_position_places_the_world_frame_on_a_target_frame(agent_client):
    p = _ok(agent_client, "position", shape=_box(agent_client, 10, 20, 30),
            origin=[100, 0, 0], x_dir=[0, 1, 0], y_dir=[-1, 0, 0])["measure"]
    assert p["bbox"] == pytest.approx([80, 100, 0, 10, 0, 30], abs=1e-6)
    flip = _ok(agent_client, "position", shape=_box(agent_client, 10, 20, 30),
               x_dir=[-1, 0, 0], y_dir=[0, -1, 0])["measure"]
    assert flip["bbox"] == pytest.approx([-10, 0, -20, 0, 0, 30], abs=1e-6)
    only = _ok(agent_client, "position", shape=_box(agent_client), origin=[0, 0, 7])
    assert only["measure"]["bbox"] == pytest.approx([0, 10, 0, 10, 7, 17], abs=1e-6)


def test_align_puts_one_part_on_another(agent_client):
    table = _ok(agent_client, "make_box_two_pnt", pnt1=[-50, -20, 0], pnt2=[50, 20, 30])["name"]
    r = _ok(agent_client, "align", shape=_box(agent_client), to=table, match=[
        {"axis": "z", "my": "min", "target": "max"},
        {"axis": "x", "my": "center", "target": "center"},
        {"axis": "y", "my": "max", "target": "max", "gap": -1}])
    assert r["measure"]["bbox"] == pytest.approx([-5, 5, 9, 19, 30, 40], abs=1e-6)
    assert r["moved_by"] == pytest.approx([-5, 9, 30])


def test_result_py_rebuilds_positioned_mirrored_and_patterned_parts(agent_client, tmp_path):
    import re
    import subprocess

    from salome_agent.nodes.assembler import assemble

    from .conftest import _salome_subprocess_env

    calls = []

    def call(name, **args):
        r = _ok(agent_client, name, **args)
        calls.append({"tool": name, "tool_args": args, "records": r["records"]})
        return r

    b = call("make_box", dx=10, dy=20, dz=30)["name"]
    p = call("position", shape=b, origin=[100, 0, 0], x_dir=[0, 1, 0], y_dir=[-1, 0, 0])["name"]
    m = call("mirror", shape=p, entity="YOZ", keep_original=True)["name"]
    peg = call("make_cylinder", radius=2, height=5, origin=[20, 0, 0])["name"]
    ring = call("multi_rotate", shape=peg, axis="OZ", count=4)["name"]
    both = call("fuse", objects=[m, ring])
    live = both["measure"]
    log = [rec for c in calls for rec in c["records"]]
    script = assemble({"stages": {0: {"module": "shaper", "status": "ok", "script": None,
                                      "result_names": [both["name"]], "call_log": log,
                                      "calls": calls}},
                       "run_id": "t", "user_request": "parts", "plan": []})
    script += ("from GeomAlgoAPI import GeomAlgoAPI_ShapeTools as _T\n"
               f"_s = {both['name']}.feature().firstResult().shape()\n"
               "print('VOLUME', _T.volume(_s), 'BOX', _s.computeSize()[1:])\n")
    path = tmp_path / "result.py"
    path.write_text(script)
    out = subprocess.run([f"{SALOME_ROOT}/salome", "shell", "--", "python3", str(path)],
                         env=_salome_subprocess_env(), capture_output=True, text=True,
                         timeout=240)
    m = re.search(r"VOLUME ([0-9.eE+-]+) BOX \[(.*?)\]", out.stdout)
    assert m, out.stdout[-2000:] + out.stderr[-2000:]
    assert float(m.group(1)) == pytest.approx(live["volume"], abs=1e-3)
    assert live["volume"] == pytest.approx(2 * 6000 + 4 * math.pi * 4 * 5, rel=1e-6)
    xs = [float(v) for v in m.group(2).split(",")]
    assert xs[0] == pytest.approx(-100, abs=1e-3) and xs[3] == pytest.approx(100, abs=1e-3)


def test_review_fixes_refuse_bad_and_pointless_input(agent_client):
    b = _box(agent_client)
    r = _call(agent_client, "position", shape=b, x_dir=[1, 0])
    assert r["ok"] is False and "[x, y, z]" in r["error"]
    r = _call(agent_client, "align", shape=b, to=[b], match=[{"axis": "x", "my": "min",
                                                             "target": "min"}])
    assert r["ok"] is False
    r = _call(agent_client, "align", shape=b, to=b, match=[{"axis": "x", "my": "min",
                                                           "target": "max"}])
    assert r["ok"] is False and "another shape" in r["error"]
    r = _call(agent_client, "rotate", shape=b, axis=["OZ"], angle_deg=30)
    assert r["ok"] is False and "OX, OY, OZ" in r["error"]
    r = _call(agent_client, "mirror", shape=b, entity={"plane": "XOY"})
    assert r["ok"] is False
    r = _call(agent_client, "multi_translate", shape=b, vector="OX", step=0, count=3)
    assert r["ok"] is False and "step" in r["error"]
    r = _call(agent_client, "multi_translate", shape=b, vector="OX", step=20, count=2,
              vector2="OX", step2=5, count2=2)
    assert r["ok"] is False and "vector2" in r["error"]
    r = _call(agent_client, "multi_rotate", shape=b, axis="OZ", count=5, angle_deg=90)
    assert r["ok"] is False and "overlap" in r["error"]
    same = _ok(agent_client, "translate", shape=b, dx=0, dy=0, dz=0)
    assert same["name"] == b and same["records"] == []
    same = _ok(agent_client, "rotate", shape=b, axis="OZ", angle_deg=360)
    assert same["name"] == b



def test_align_flush_on_odd_coordinates_fuses_into_one_box(agent_client):
    # a rounded box would leave a 1e-4 sliver: two solids, not one
    other = _ok(agent_client, "make_box_two_pnt", pnt1=[10.123456789, 0, 0],
                pnt2=[20.987654321, 10, 10])["name"]
    r = _ok(agent_client, "align", shape=_box(agent_client), to=other, match=[
        {"axis": "x", "my": "max", "target": "min"}])
    fused = _ok(agent_client, "fuse", objects=[r["name"], other])["measure"]
    assert fused["faces"] == 6
    assert fused["size"][0] == pytest.approx(20.987654321 - 0.123456789, abs=1e-4)
