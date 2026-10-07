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


def _sketch(client, plane="XOY"):
    return _ok(client, "create_sketch", plane=plane)["sketch"]


def _areas(r):
    return sorted(round(g["area"], 3) for g in r["regions"])


def test_extruding_a_sketch_with_a_hole_is_refused(agent_client):
    sk = _sketch(agent_client)
    _ok(agent_client, "sketch_add", sketch=sk, entities=[
        {"id": "R", "type": "rectangle", "corner1": [0, 0], "corner2": [40, 20]},
        {"id": "C", "type": "circle", "center": [20, 10], "radius": 5}])
    r = _call(agent_client, "extrude", base=sk, size=5)
    assert r["ok"] is False and "make_face" in r["error"]
    two = _sketch(agent_client)
    _ok(agent_client, "sketch_add", sketch=two, entities=[
        {"id": "A", "type": "circle", "center": [0, 0], "radius": 2},
        {"id": "B", "type": "circle", "center": [10, 0], "radius": 2}])
    ext = _ok(agent_client, "extrude", base=two, size=1)["measure"]
    assert ext["volume"] == pytest.approx(2 * math.pi * 4, abs=1e-3)


def test_rectangle_and_polyline_are_closed_and_joined(agent_client):
    sk = _sketch(agent_client)
    r = _ok(agent_client, "sketch_add", sketch=sk, entities=[
        {"id": "R", "type": "rectangle", "corner1": [0, 0], "corner2": [20, 10]},
        {"id": "T", "type": "polyline", "points": [[30, 0], [40, 0], [30, 10]], "closed": True}])
    assert _areas(r) == [50, 200]
    assert r["loose_ends"] == []
    st = _ok(agent_client, "sketch_status", sketch=sk)
    assert {"R1", "R2", "R3", "R4", "T1", "T2", "T3"} <= set(st["entities"])
    # the sides stay joined when one is moved by a constraint
    r = _ok(agent_client, "sketch_constrain", sketch=sk, constraints=[
        {"type": "length", "refs": ["T1"], "value": 15}])
    assert r["loose_ends"] == []
    r = _call(agent_client, "sketch_add", sketch=sk, entities=[
        {"id": "R1", "type": "circle", "center": [0, 0], "radius": 1}])
    assert r["ok"] is False
    r = _call(agent_client, "sketch_add", sketch=sk, entities=[
        {"id": "T", "type": "rectangle", "corner1": [0, 0], "corner2": [1, 1]}])
    assert r["ok"] is False and "taken" in r["error"]


def test_ellipse_and_splines(agent_client):
    sk = _sketch(agent_client)
    r = _ok(agent_client, "sketch_add", sketch=sk, entities=[
        {"id": "E", "type": "ellipse", "center": [0, 0], "major_radius": 20,
         "minor_radius": 10, "angle": 30}])
    assert _areas(r) == pytest.approx([math.pi * 200], abs=1e-3)
    ext = _ok(agent_client, "extrude", base=sk, size=2)["measure"]
    assert ext["volume"] == pytest.approx(math.pi * 200 * 2, rel=1e-4)
    sp = _sketch(agent_client)
    r = _ok(agent_client, "sketch_add", sketch=sp, entities=[
        {"id": "S", "type": "spline", "points": [[0, 0], [10, 8], [20, 0]]},
        {"id": "L", "type": "line", "start": [20, 0], "end": [0, 0]}])
    assert r["loose_ends"] == []      # drawn end to end: touching
    r = _ok(agent_client, "sketch_constrain", sketch=sp, constraints=[
        {"type": "coincident", "refs": ["S.end", "L.start"]},
        {"type": "coincident", "refs": ["L.end", "S.start"]}])
    assert r["loose_ends"] == [] and len(r["regions"]) == 1
    loop = _sketch(agent_client)
    r = _ok(agent_client, "sketch_add", sketch=loop, entities=[
        {"id": "S", "type": "spline", "points": [[0, 0], [10, 0], [10, 10], [0, 10]],
         "closed": True}])
    assert r["loose_ends"] == [] and len(r["regions"]) == 1


def test_projection_of_a_solid_edge(agent_client):
    box = _ok(agent_client, "make_box", dx=10, dy=20, dz=30)["name"]
    edges = _ok(agent_client, "get_subshapes", shape=box, sub_type="edge")["subshapes"]
    upright = [u["id"] for u in edges if u["size"] == pytest.approx(30)][0]
    across = [u["id"] for u in edges if u["size"] == pytest.approx(20)][0]
    sk = _sketch(agent_client)
    r = _call(agent_client, "sketch_add", sketch=sk, entities=[
        {"id": "P", "type": "projection", "edge": upright, "of": box}])
    assert r["ok"] is False and "vertex" in r["error"]
    _ok(agent_client, "sketch_add", sketch=sk, entities=[
        {"id": "P", "type": "projection", "edge": across, "of": box}])
    st = _ok(agent_client, "sketch_status", sketch=sk)
    assert st["entities"]["P"]["type"] == "line"
    assert st["regions"] == [] and st["loose_ends"] == []


def test_corner_fillet_trims_the_rectangle(agent_client):
    sk = _sketch(agent_client)
    _ok(agent_client, "sketch_add", sketch=sk, entities=[
        {"id": "R", "type": "rectangle", "corner1": [0, 0], "corner2": [20, 10]}])
    corner = _ok(agent_client, "sketch_status", sketch=sk)["entities"]["R1"]
    r = _ok(agent_client, "sketch_fillet", sketch=sk, at="R1.end", radius=2, id="F")
    assert r["loose_ends"] == []
    assert _areas(r) == pytest.approx([200 - (4 - math.pi)], abs=1e-3)
    assert r["moved"].keys() <= {"R1", "R2", "R3", "R4"}
    assert corner is not None
    r = _call(agent_client, "sketch_fillet", sketch=sk, at="R1", radius=2, id="G")
    assert r["ok"] is False


def test_sketch_copies(agent_client):
    sk = _sketch(agent_client)
    _ok(agent_client, "sketch_add", sketch=sk, entities=[
        {"id": "C", "type": "circle", "center": [10, 0], "radius": 2},
        {"id": "A", "type": "line", "start": [0, -20], "end": [0, 20], "construction": True}])
    r = _ok(agent_client, "sketch_copy", sketch=sk, op="mirror", of=["C"], axis="A", id="M")
    assert r["made"] == ["M1"] and len(r["regions"]) == 2
    r = _ok(agent_client, "sketch_copy", sketch=sk, op="linear", of=["C"], vector=[0, 10],
            count=3, id="P")
    assert r["made"] == ["P1", "P2"] and len(r["regions"]) == 4
    r = _ok(agent_client, "sketch_copy", sketch=sk, op="circular", of=["M1"], center=[0, 0],
            count=4, id="Q")
    centres = sorted(tuple(round(c, 3) + 0.0 for c in v["center"]) for k, v in
                     _ok(agent_client, "sketch_status", sketch=sk)["entities"].items()
                     if k.startswith("Q"))
    assert len(r["made"]) == 3
    assert centres == [(0.0, -10.0), (0.0, 10.0), (10.0, 0.0)]
    off = _sketch(agent_client)
    _ok(agent_client, "sketch_add", sketch=off, entities=[
        {"id": "C", "type": "circle", "center": [0, 0], "radius": 5}])
    r = _ok(agent_client, "sketch_copy", sketch=off, op="offset", of=["C"], distance=2, id="O")
    st = _ok(agent_client, "sketch_status", sketch=off)["entities"]
    assert st["O1"]["radius"] == pytest.approx(7)
    r = _call(agent_client, "sketch_copy", sketch=off, op="offset", of=["C"], distance=1, id="O")
    assert r["ok"] is False and "taken" in r["error"]


def test_pipe_loft_and_filling(agent_client):
    path = _ok(agent_client, "make_polyline", points=[[0, 0, 0], [0, 0, 50]])["name"]
    sk = _sketch(agent_client)
    _ok(agent_client, "sketch_add", sketch=sk, entities=[
        {"id": "C", "type": "circle", "center": [0, 0], "radius": 2}])
    pipe = _ok(agent_client, "make_pipe", base=sk, path=path)["measure"]
    assert pipe["volume"] == pytest.approx(math.pi * 4 * 50, rel=1e-6)
    bottom = _sketch(agent_client)
    _ok(agent_client, "sketch_add", sketch=bottom, entities=[
        {"id": "C", "type": "circle", "center": [0, 0], "radius": 5}])
    f1 = _ok(agent_client, "make_face", sketch=bottom, regions="all")["name"]
    top = _ok(agent_client, "create_sketch", plane={"base": "XOY", "offset": 10})["sketch"]
    _ok(agent_client, "sketch_add", sketch=top, entities=[
        {"id": "C", "type": "circle", "center": [0, 0], "radius": 2}])
    f2 = _ok(agent_client, "make_face", sketch=top, regions="all")["name"]
    loft = _ok(agent_client, "make_loft", first=f1, second=f2)["measure"]
    assert loft["volume"] == pytest.approx(math.pi * 10 * (25 + 10 + 4) / 3, rel=1e-3)
    a = _ok(agent_client, "make_curve", points=[[0, 0, 0], [10, 0, 3], [20, 0, 0]])["name"]
    b = _ok(agent_client, "make_curve", points=[[0, 20, 0], [10, 20, -3], [20, 20, 0]])["name"]
    fill = _ok(agent_client, "make_filling", edges=[a, b])["measure"]
    assert fill["faces"] == 1 and fill["area"] > 400


def test_result_py_rebuilds_new_sketch_entities(agent_client, tmp_path):
    import re
    import subprocess

    from salome_agent.nodes.assembler import assemble

    from .conftest import _salome_subprocess_env

    calls = []

    def call(name, **args):
        r = _ok(agent_client, name, **args)
        calls.append({"tool": name, "tool_args": args, "records": r["records"]})
        return r

    sk = call("create_sketch", plane="XOY")["sketch"]
    call("sketch_add", sketch=sk, entities=[
        {"id": "R", "type": "rectangle", "corner1": [0, 0], "corner2": [40, 20]},
        {"id": "A", "type": "line", "start": [20, -5], "end": [20, 25], "construction": True}])
    call("sketch_fillet", sketch=sk, at="R1.end", radius=3, id="F")
    face = call("make_face", sketch=sk, regions="all")["name"]
    plate = call("extrude", base=face, size=4)["name"]
    holes = call("create_sketch", plane="XOY")["sketch"]
    call("sketch_add", sketch=holes, entities=[
        {"id": "E", "type": "ellipse", "center": [8, 10], "major_radius": 4, "minor_radius": 2},
        {"id": "S", "type": "spline", "points": [[25, 5], [33, 5], [33, 15], [25, 15]],
         "closed": True},
        {"id": "A", "type": "line", "start": [20, -5], "end": [20, 25], "construction": True}])
    call("sketch_copy", sketch=holes, op="mirror", of=["E"], axis="A", id="M")
    cut = call("extrude_cut", base=holes, target=plate, through_all=True)
    live = cut["measure"]["volume"]
    log = [rec for c_ in calls for rec in c_["records"]]
    script = assemble({"stages": {0: {"module": "shaper", "status": "ok", "script": None,
                                      "result_names": [cut["name"]], "call_log": log,
                                      "calls": calls}},
                       "run_id": "t", "user_request": "plate", "plan": []})
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
    assert live < (800 - (4 - math.pi) * 9 - 2 * math.pi * 8) * 4


def test_sub_batch_4_refusals(agent_client):
    sk = _sketch(agent_client)
    for ent in ({"id": "S", "type": "spline", "points": [[0, 0], [5, 5], [0, 0]]},
                {"id": "S", "type": "polyline", "points": [[0, 0], [5, 0], [5, 5], [0, 0]],
                 "closed": True},
                {"id": "S", "type": "ellipse", "center": [0, 0], "major_radius": 3,
                 "minor_radius": 3}):
        r = _call(agent_client, "sketch_add", sketch=sk, entities=[ent])
        assert r["ok"] is False, ent
    _ok(agent_client, "sketch_add", sketch=sk, entities=[
        {"id": "C", "type": "circle", "center": [0, 0], "radius": 2},
        {"id": "A", "type": "line", "start": [0, -5], "end": [0, 5], "construction": True}])
    for kw in ({"op": "mirror", "of": ["C", "C"], "axis": "A"},
               {"op": "mirror", "of": ["A"], "axis": "A"},
               {"op": "mirror", "of": ["C"], "axis": "A", "count": 3},
               {"op": "offset", "of": ["C"], "distance": -2}):
        r = _call(agent_client, "sketch_copy", sketch=sk, id="M", **kw)
        assert r["ok"] is False, kw
    assert _ok(agent_client, "sketch_status", sketch=sk)["entities"].keys() == {"C", "A"}
    a = _ok(agent_client, "make_point", x=0, y=0, z=0)["name"]
    r = _call(agent_client, "make_polyline", points=[a, a, [1, 0, 0]])
    assert r["ok"] is False


def test_swept_and_face_refusals(agent_client):
    box = _ok(agent_client, "make_box", dx=5, dy=5, dz=5)["name"]
    pts = [[0, 0, 0], [0, 0, 10]]
    path = _ok(agent_client, "make_polyline", points=pts)["name"]
    for kw in ({"base": box, "path": path}, {"base": path, "path": box}):
        assert _call(agent_client, "make_pipe", **kw)["ok"] is False
    assert _call(agent_client, "make_loft", first=box, second=path)["ok"] is False
    sk = _sketch(agent_client)
    _ok(agent_client, "sketch_add", sketch=sk, entities=[
        {"id": "O", "type": "circle", "center": [0, 0], "radius": 10},
        {"id": "I", "type": "circle", "center": [0, 0], "radius": 3}])
    assert _call(agent_client, "make_face", sketch=sk, regions="all")["ok"] is False
    r = _call(agent_client, "sketch_add", sketch=sk, entities=[
        {"id": "L", "type": "line", "start": [0, 0], "end": [1, 1], "closed": True}])
    assert r["ok"] is False


def test_projection_of_a_circle_seen_edge_on(agent_client):
    cyl = _ok(agent_client, "make_cylinder", radius=5, height=10)["name"]
    edges = _ok(agent_client, "get_subshapes", shape=cyl, sub_type="edge")["subshapes"]
    ring = [u["id"] for u in edges if u["size"] == pytest.approx(2 * math.pi * 5)][0]
    edge_on = _sketch(agent_client, "XOZ")
    r = _call(agent_client, "sketch_add", sketch=edge_on, entities=[
        {"id": "P", "type": "projection", "edge": ring, "of": cyl}])
    assert r["ok"] is False and "edge-on" in r["error"]
    face_on = _sketch(agent_client, "XOY")
    _ok(agent_client, "sketch_add", sketch=face_on, entities=[
        {"id": "P", "type": "projection", "edge": ring, "of": cyl}])
    assert _ok(agent_client, "sketch_status", sketch=face_on)["entities"]["P"]["type"] == "circle"
