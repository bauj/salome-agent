# Copyright (C) 2026  salome-agent authors
# LGPL 2.1 or later
import math

import pytest

from ._salome_root import SALOME_ROOT

pytestmark = [
    pytest.mark.salome,
    pytest.mark.skipif(SALOME_ROOT is None,
                       reason="set SALOME_ROOT or salome.root in config.json"),
]

_TOP = {"on_plane": {"point": [0, 0, 30], "normal": [0, 0, 1]}}


def _call(client, name, **args):
    return client.call_tool(name, args, module="shaper")


def _ok(client, name, **args):
    r = _call(client, name, **args)
    assert r["ok"] is True, r.get("error")
    return r


def _box(client):
    return _ok(client, "make_box", dx=10, dy=20, dz=30, center=[5, 10, 15])["name"]


def _ids(client, shape, sub_type, where="all"):
    return [u["id"] for u in
            _ok(client, "get_subshapes", shape=shape, sub_type=sub_type, where=where)["subshapes"]]


def test_subshape_ids_are_stable_and_filtered(agent_client):
    b = _box(agent_client)
    edges = _ok(agent_client, "get_subshapes", shape=b, sub_type="edge")["subshapes"]
    assert len(edges) == 12 and edges[0]["id"] == "E1"
    assert _ids(agent_client, b, "edge") == [e["id"] for e in edges]
    assert len(_ids(agent_client, b, "face")) == 6
    top = _ok(agent_client, "get_subshapes", shape=b, sub_type="edge", where=_TOP)["subshapes"]
    assert len(top) == 4 and all(u["centre"][2] == pytest.approx(30) for u in top)
    longest = _ok(agent_client, "get_subshapes", shape=b, sub_type="edge",
                  where={"by_size": "longest", "count": 4})["subshapes"]
    assert all(u["size"] == pytest.approx(30) for u in longest)


def test_fillet_and_chamfer_by_ids_and_by_where(agent_client):
    b = _box(agent_client)
    top = _ids(agent_client, b, "edge", _TOP)
    f = _ok(agent_client, "fillet_edges", shape=b, radius=2, edges=top)
    assert f["edges"] == top and f["measure"]["valid"] is True
    assert 5900 < f["measure"]["volume"] < 6000
    c = _ok(agent_client, "chamfer_edges", shape=_box(agent_client), d1=1, edges="all")
    assert c["measure"]["valid"] is True and c["measure"]["faces"] == 26
    w = _ok(agent_client, "fillet_edges", shape=_box(agent_client), radius=1,
            where={"sorted_by": "z", "take": "min", "count": 4})
    assert len(w["edges"]) == 4
    r = _call(agent_client, "fillet_edges", shape=_box(agent_client), radius=1, edges=["E99"])
    assert r["ok"] is False and "no edge E99" in r["error"]
    r = _call(agent_client, "fillet_edges", shape=_box(agent_client), radius=50, edges="all")
    assert r["ok"] is False


def test_thick_solid_open_at_the_top(agent_client):
    b = _box(agent_client)
    top = _ids(agent_client, b, "face", _TOP)
    t = _ok(agent_client, "make_thick_solid", shape=b, thickness=1, faces=top)["measure"]
    assert t["volume"] == pytest.approx(6000 - 8 * 18 * 29, abs=1e-3)
    assert t["size"] == pytest.approx([10, 20, 30])


def test_partition_and_split(agent_client):
    a = _ok(agent_client, "make_box", dx=10, dy=10, dz=10, center=[5, 5, 5])["name"]
    c = _ok(agent_client, "make_cylinder", radius=4, height=20, origin=[0, 0, 0])["name"]
    p = _ok(agent_client, "partition", shapes=[a, c])["name"]
    assert len(_ids(agent_client, p, "solid")) == 3
    b = _box(agent_client)
    pl = _ok(agent_client, "make_plane", base="XOY", offset=10)["name"]
    s = _ok(agent_client, "split", objects=[b], tools=[pl])
    assert s["measure"]["volume"] == pytest.approx(6000)
    assert len(_ids(agent_client, s["name"], "solid")) == 2


def test_copy_and_explode_subshapes(agent_client):
    b = _box(agent_client)
    top = _ids(agent_client, b, "face", _TOP)
    face = _ok(agent_client, "get_subshape", shape=b, ids=top)["measure"]
    assert face["area"] == pytest.approx(200) and face["faces"] == 1
    ex = _ok(agent_client, "explode", shape=_box(agent_client), sub_type="edge")
    assert len(ex["names"]) == 12
    r = _call(agent_client, "get_subshape", shape=b, ids=["F1", "E1"])
    assert r["ok"] is False and "one type" in r["error"]


def test_sketch_on_a_face_of_a_solid_and_pocket(agent_client):
    b = _box(agent_client)
    top = _ids(agent_client, b, "face", _TOP)[0]
    sk = _ok(agent_client, "create_sketch", plane={"face": top, "of": b})
    o, x, y = (sk["frame"][k] for k in ("origin", "x_dir", "y_dir"))
    assert sk["frame"]["normal"] == pytest.approx([0, 0, 1])
    # the centre of the top face, in this sketch's coordinates
    c = [5 - o[0], 10 - o[1]]
    u = c[0] * x[0] + c[1] * x[1]
    v = c[0] * y[0] + c[1] * y[1]
    r = _ok(agent_client, "sketch_add", sketch=sk["sketch"], entities=[
        {"id": "C", "type": "circle", "center": [u, v], "radius": 3}])
    assert r["regions"][0]["inside"] == pytest.approx([u, v], abs=1.5)
    _ok(agent_client, "close_sketch", sketch=sk["sketch"])
    cut = _ok(agent_client, "extrude_cut", base=sk["sketch"], target=b, through_all=True)
    assert cut["measure"]["volume"] == pytest.approx(6000 - math.pi * 9 * 30, rel=1e-6)


def test_sketch_on_a_tilted_construction_plane(agent_client):
    pl = _ok(agent_client, "make_plane", point=[5, 0, 5], normal=[1, 0, 1])["name"]
    sk = _ok(agent_client, "create_sketch", plane=pl)
    assert sk["frame"]["normal"] == pytest.approx([0.7071068, 0, 0.7071068], abs=1e-6)
    _ok(agent_client, "sketch_add", sketch=sk["sketch"], entities=[
        {"id": "C", "type": "circle", "center": [0, 0], "radius": 2}])
    _ok(agent_client, "close_sketch", sketch=sk["sketch"])
    ext = _ok(agent_client, "extrude", base=sk["sketch"], size=4)["measure"]
    assert ext["volume"] == pytest.approx(math.pi * 4 * 4, rel=1e-6)
    cyl = _ok(agent_client, "make_cylinder", radius=2, height=4, origin=[0, 0, 0])["name"]
    side = _ids(agent_client, cyl, "face", {"by_size": "largest", "count": 1})[0]
    r = _call(agent_client, "create_sketch", plane={"face": side, "of": cyl})
    assert r["ok"] is False and ("flat" in r["error"])


def test_result_py_rebuilds_fillet_sketch_on_face_and_shell(agent_client, tmp_path):
    import re
    import subprocess

    from salome_agent.nodes.assembler import assemble

    from .conftest import _salome_subprocess_env

    calls = []

    def call(name, **args):
        r = _ok(agent_client, name, **args)
        calls.append({"tool": name, "tool_args": args, "records": r["records"]})
        return r

    b = call("make_box", dx=10, dy=20, dz=30, center=[5, 10, 15])["name"]
    edges = call("get_subshapes", shape=b, sub_type="edge",
                 where={"by_size": "longest", "count": 4})["subshapes"]
    f = call("fillet_edges", shape=b, radius=2, edges=[e["id"] for e in edges])["name"]
    top = [u["id"] for u in call("get_subshapes", shape=f, sub_type="face",
                                 where=_TOP)["subshapes"]]
    sk = call("create_sketch", plane={"face": top[0], "of": f})
    o, x, y = (sk["frame"][k] for k in ("origin", "x_dir", "y_dir"))
    c = [5 - o[0], 10 - o[1]]
    call("sketch_add", sketch=sk["sketch"], entities=[
        {"id": "C", "type": "circle", "center": [c[0] * x[0] + c[1] * x[1],
                                                 c[0] * y[0] + c[1] * y[1]], "radius": 3}])
    call("close_sketch", sketch=sk["sketch"])
    cut = call("extrude_cut", base=sk["sketch"], target=f, through_all=True)
    live = cut["measure"]["volume"]
    log = [rec for c_ in calls for rec in c_["records"]]
    script = assemble({"stages": {0: {"module": "shaper", "status": "ok", "script": None,
                                      "result_names": [cut["name"]], "call_log": log,
                                      "calls": calls}},
                       "run_id": "t", "user_request": "block", "plan": []})
    script += ("from GeomAlgoAPI import GeomAlgoAPI_ShapeTools as _T\n"
               f"print('VOLUME', _T.volume({cut['name']}.feature().firstResult().shape()))\n")
    path = tmp_path / "result.py"
    path.write_text(script)
    out = subprocess.run([f"{SALOME_ROOT}/salome", "shell", "--", "python3", str(path)],
                         env=_salome_subprocess_env(), capture_output=True, text=True,
                         timeout=240)
    m = re.search(r"VOLUME ([0-9.eE+-]+)", out.stdout)
    assert m, out.stdout[-2000:] + out.stderr[-2000:]
    assert float(m.group(1)) == pytest.approx(live, abs=1e-3)
    assert live < 6000 - math.pi * 9 * 30 + 1


def test_ids_of_coaxial_rims_are_stable_and_distinct(agent_client):
    outer = _ok(agent_client, "make_cylinder", radius=10, height=5, origin=[0, 0, 0])["name"]
    inner = _ok(agent_client, "make_cylinder", radius=5, height=5, origin=[0, 0, 0])["name"]
    tube = _ok(agent_client, "cut", objects=[outer], tools=[inner])["name"]
    first = _ok(agent_client, "get_subshapes", shape=tube, sub_type="edge")["subshapes"]
    again = _ok(agent_client, "get_subshapes", shape=tube, sub_type="edge")["subshapes"]
    assert first == again
    rims = [u for u in first if u["centre"] == pytest.approx([0, 0, 5])]
    assert sorted(round(u["size"], 3) for u in rims) == [round(2 * math.pi * 5, 3),
                                                         round(2 * math.pi * 10, 3)]


def test_a_face_shared_by_two_results_has_one_id(agent_client):
    a = _ok(agent_client, "make_box", dx=10, dy=10, dz=10, center=[5, 5, 5])["name"]
    b = _ok(agent_client, "make_box_two_pnt", pnt1=[10, 0, 0], pnt2=[20, 10, 10])["name"]
    p = _ok(agent_client, "partition", shapes=[a, b])["name"]
    assert len(_ids(agent_client, p, "face")) == 11


def test_review_fixes_refuse_repeats_and_odd_args(agent_client):
    a = _box(agent_client)
    r = _call(agent_client, "partition", shapes=[a, a])
    assert r["ok"] is False and "more than once" in r["error"]
    r = _call(agent_client, "split", objects=[a], tools=[a])
    assert r["ok"] is False and "more than once" in r["error"]
    r = _call(agent_client, "partition", shapes=[a], tools="")
    assert r["ok"] is False
    r = _call(agent_client, "create_sketch", plane={"face": ["F1"], "of": a})
    assert r["ok"] is False and "id" in r["error"]
    r = _call(agent_client, "get_subshapes", shape=a, sub_type=["edge"])
    assert r["ok"] is False
    ex = _ok(agent_client, "explode", shape=a, sub_type="face")
    assert all(not n["name"].startswith(a + "_") for n in ex["names"])


def test_a_revolved_profile_touching_its_axis_lists_its_real_edges(agent_client):
    # The quarter arc starts on the axis: OCC adds a zero-length edge at that pole,
    # which SHAPER cannot name; it must not fail the whole query (task 349).
    sk = _ok(agent_client, "create_sketch", plane="XOZ")["sketch"]
    _ok(agent_client, "sketch_add", sketch=sk, entities=[
        {"id": "A", "type": "arc", "center": [20, 0], "start": [0, 0], "end": [20, 20],
         "reversed": True},
        {"id": "L1", "type": "line", "start": [20, 20], "end": [20, 40]},
        {"id": "L2", "type": "line", "start": [20, 40], "end": [0, 40]},
        {"id": "L3", "type": "line", "start": [0, 40], "end": [0, 0]}])
    _ok(agent_client, "close_sketch", sketch=sk)
    rev = _ok(agent_client, "revolve", base=sk, axis="OZ", angle=360)
    edges = _ok(agent_client, "get_subshapes", shape=rev["name"], sub_type="edge")["subshapes"]
    assert all(e["size"] > 0 for e in edges)
    assert len(edges) == rev["measure"]["edges"]
    top = [f["id"] for f in _ok(agent_client, "get_subshapes", shape=rev["name"],
                                sub_type="face")["subshapes"] if f["centre"][2] == 40]
    shell = _ok(agent_client, "make_thick_solid", shape=rev["name"], faces=top,
                thickness=2, inside=True)["name"]
    rim = _ids(agent_client, shell, "edge", {"sorted_by": "z", "take": "max", "count": 2})
    filleted = _ok(agent_client, "fillet_edges", shape=shell, edges=rim, radius=0.5)
    assert filleted["measure"]["valid"]


def test_fillets_meeting_across_a_rim_are_refused_not_run(agent_client):
    # Vase of run 84e7b7cb7844: radius 1 on both circles of its 2 mm rim
    # segfaulted OCC and killed the server (task 354).
    sk = _ok(agent_client, "create_sketch", plane="XOZ")["sketch"]
    _ok(agent_client, "sketch_add", sketch=sk, entities=[
        {"id": "L1", "type": "line", "start": [0, 0], "end": [30, 0]},
        {"id": "L2", "type": "line", "start": [30, 0], "end": [30, 5]},
        {"id": "S1", "type": "spline", "points": [[30, 5], [50, 60], [40, 120]]},
        {"id": "L3", "type": "line", "start": [40, 120], "end": [38, 120]},
        {"id": "S2", "type": "spline", "points": [[38, 120], [48, 60], [28, 5]]},
        {"id": "L4", "type": "line", "start": [28, 5], "end": [0, 5]},
        {"id": "L5", "type": "line", "start": [0, 5], "end": [0, 0]}])
    _ok(agent_client, "close_sketch", sketch=sk)
    vase = _ok(agent_client, "revolve", base=sk, axis="OZ", angle=360)["name"]
    rim = _ids(agent_client, vase, "edge",
               {"on_plane": {"point": [0, 0, 120], "normal": [0, 0, 1]}, "state": "on"})
    assert len(rim) == 2
    r = _call(agent_client, "fillet_edges", shape=vase, radius=1, edges=rim)
    assert r["ok"] is False and "would meet" in r["error"] and "below 0.95" in r["error"]
    f = _ok(agent_client, "fillet_edges", shape=vase, radius=0.9, edges=rim)
    assert f["measure"]["valid"] is True


def _car_lower_body(client):
    # Lower body of run llm_trace_3 (stage 4 input): side profile extruded 80 mm
    # across, four r=18 wheel arches cut out; 0.75 mm of side wall is left above
    # each arch, under the top edges that slope 3.5 degrees down from the middle.
    sk = _ok(client, "create_sketch", plane={"base": "XOZ", "offset": -40})["sketch"]
    _ok(client, "sketch_add", sketch=sk, entities=[{"id": "P1", "type": "polyline", "closed": True,
        "points": [[-60, 20], [60, 20], [60, 24], [53, 33], [20, 35], [-20, 35], [-53, 33],
                   [-60, 24]]}])
    _ok(client, "close_sketch", sketch=sk)
    body = _ok(client, "extrude", base=sk, size=80, direction="OY")["name"]
    cyls = [_ok(client, "make_cylinder", radius=18, height=20, origin=[x, y, 15],
                axis="OY")["name"] for x in (40, -40) for y in (22, -42)]
    return _ok(client, "cut", objects=[body], tools=cyls)["name"]


def _edges_at(client, shape, test):
    return [u["id"] for u in _ok(client, "get_subshapes", shape=shape, sub_type="edge")
            ["subshapes"] if test(u["centre"])]


_TOP_SIDE = lambda c: c[0] == 0 and abs(c[1]) == 40 and c[2] == 35
_SLOPE_SIDE = lambda c: abs(c[0]) == 36.5 and abs(c[1]) == 40
_RIM = lambda c: abs(c[1]) == 40 and abs(c[0]) in (31.3542, 48.6458)


def test_fillet_reports_the_edges_shaper_rounds_with_the_picked_ones(agent_client):
    body = _car_lower_body(agent_client)
    top = _edges_at(agent_client, body, _TOP_SIDE)
    slopes = _edges_at(agent_client, body, _SLOPE_SIDE)
    assert len(top) == 2 and len(slopes) == 4
    f = _ok(agent_client, "fillet_edges", shape=body, radius=0.5, edges=top)
    assert sorted(f["also_rounded"]) == sorted(slopes)


def test_fillet_meeting_an_earlier_fillet_through_its_chain_is_refused(agent_client):
    body = _car_lower_body(agent_client)
    rimmed = _ok(agent_client, "fillet_edges", shape=body, radius=0.5,
                 edges=_edges_at(agent_client, body, _RIM))["name"]
    top = _edges_at(agent_client, rimmed, _TOP_SIDE)
    r = _call(agent_client, "fillet_edges", shape=rimmed, radius=3, edges=top)
    assert r["ok"] is False and "earlier fillet" in r["error"], r
    assert "also rounds" in r["error"]


def test_fillet_that_cuts_a_face_in_two_is_refused_with_the_width(agent_client):
    # The trace's rim fillet: r=3 rolls past the 0.75 mm of wall above the arch
    # and cuts the side face in two, so every later fillet next to it failed.
    body = _car_lower_body(agent_client)
    rim = _edges_at(agent_client, body, _RIM)
    r = _call(agent_client, "fillet_edges", shape=body, radius=3, edges=rim)
    assert r["ok"] is False and "cut it in two" in r["error"], r
    assert "radius below 0.7535" in r["error"]
    f = _ok(agent_client, "fillet_edges", shape=body, radius=0.7, edges=rim)
    assert f["edges"] == rim


def test_fillet_cut_through_an_added_edge_names_the_added_edges(agent_client):
    body = _car_lower_body(agent_client)
    top = _edges_at(agent_client, body, _TOP_SIDE)
    r = _call(agent_client, "fillet_edges", shape=body, radius=3, edges=top)
    assert r["ok"] is False and "cut it in two" in r["error"], r
    assert "radius below 0.7535" in r["error"]
    assert "also rounds" in r["error"]


def test_fillet_on_an_edge_where_faces_meet_smoothly_is_refused(agent_client):
    # The edges along which a fillet meets its faces have no corner: SHAPER fails
    # on them without saying why (run e84dd3a265b7: "the operation produced nothing").
    box = _box(agent_client)
    first = _ok(agent_client, "fillet_edges", shape=box, radius=5,
                where={"sorted_by": "z", "take": "max", "count": 1})["name"]
    edges = _ok(agent_client, "get_subshapes", shape=first, sub_type="edge")["subshapes"]
    smooth = [u["id"] for u in edges if u.get("smooth") is True]
    corners = [u["id"] for u in edges if "smooth" not in u]
    assert len(smooth) == 2, edges
    r = _call(agent_client, "fillet_edges", shape=first, radius=1, edges=smooth[:1])
    assert r["ok"] is False and "has no corner to round" in r["error"], r
    r = _call(agent_client, "chamfer_edges", shape=first, d1=1, edges=smooth)
    assert r["ok"] is False and "have no corner to round" in r["error"], r
    f = _ok(agent_client, "fillet_edges", shape=first, radius=1,
            edges=[corners[0], smooth[0]])
    assert f"SHAPER skipped {smooth[0]}" in f["note"], f


def test_fillet_all_edges_of_a_cylinder_skips_its_seam(agent_client):
    cyl = lambda: _ok(agent_client, "make_cylinder", radius=4, height=20,
                      origin=[0, 0, 0])["name"]
    _ok(agent_client, "fillet_edges", shape=cyl(), radius=1, edges="all")
    cyl = cyl()
    seam = [u["id"] for u in _ok(agent_client, "get_subshapes", shape=cyl,
                                 sub_type="edge")["subshapes"] if u.get("smooth")]
    r = _call(agent_client, "fillet_edges", shape=cyl, radius=1, edges=seam)
    assert r["ok"] is False and "seam" in r["error"], r


def _arch_body(client):
    # run 68f0c1abce7d: a wheel arch whose circle touches the top and bottom faces
    box = _ok(client, "make_box", dx=200, dy=50, dz=70, center=[100, 25, 35])["name"]
    cyl = _ok(client, "make_cylinder", radius=35, height=70, origin=[100, -10, 35],
              axis="OY")["name"]
    return _ok(client, "cut", objects=[box], tools=[cyl])["name"]


def test_fillet_on_an_arch_rim_ending_on_a_zero_thick_wall_is_refused(agent_client):
    body = _arch_body(agent_client)
    edges = _ok(agent_client, "get_subshapes", shape=body, sub_type="edge")["subshapes"]
    rims = [u["id"] for u in edges if u["centre"][1] in (pytest.approx(0), pytest.approx(50))
            and 65 < u["centre"][0] < 135 and 0 < u["centre"][2] < 70]
    assert rims, edges
    r = _call(agent_client, "fillet_edges", shape=body, radius=2, edges=rims[:1])
    assert r["ok"] is False and "zero thickness" in r["error"], r
    assert "cannot round " + rims[0] in r["error"]
    far = [u["id"] for u in edges if u["centre"][0] == pytest.approx(200)]
    _ok(agent_client, "fillet_edges", shape=body, radius=2, edges=far)


def test_vertical_edge_after_an_earlier_top_fillet_still_builds(agent_client):
    first = _ok(agent_client, "fillet_edges", shape=_box(agent_client), radius=3,
                where={"sorted_by": "z", "take": "max", "count": 1})["name"]
    edges = _ok(agent_client, "get_subshapes", shape=first, sub_type="edge")["subshapes"]
    vertical = [u["id"] for u in edges
                if u["size"] > 20 and u["centre"][2] == pytest.approx(13.5, abs=2)]
    assert vertical, edges
    f = _ok(agent_client, "fillet_edges", shape=first, radius=1, edges=vertical)
    assert "zero thickness" not in (f.get("note") or "")
