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
    return _ok(client, "make_box", dx=10, dy=20, dz=30)["name"]


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
    a = _ok(agent_client, "make_box", dx=10, dy=10, dz=10)["name"]
    c = _ok(agent_client, "make_cylinder", radius=4, height=20)["name"]
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
    cut = _ok(agent_client, "extrude_cut", base=sk["sketch"], target=b, through_all=True)
    assert cut["measure"]["volume"] == pytest.approx(6000 - math.pi * 9 * 30, rel=1e-6)


def test_sketch_on_a_tilted_construction_plane(agent_client):
    pl = _ok(agent_client, "make_plane", point=[5, 0, 5], normal=[1, 0, 1])["name"]
    sk = _ok(agent_client, "create_sketch", plane=pl)
    assert sk["frame"]["normal"] == pytest.approx([0.7071068, 0, 0.7071068], abs=1e-6)
    _ok(agent_client, "sketch_add", sketch=sk["sketch"], entities=[
        {"id": "C", "type": "circle", "center": [0, 0], "radius": 2}])
    ext = _ok(agent_client, "extrude", base=sk["sketch"], size=4)["measure"]
    assert ext["volume"] == pytest.approx(math.pi * 4 * 4, rel=1e-6)
    cyl = _ok(agent_client, "make_cylinder", radius=2, height=4)["name"]
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

    b = call("make_box", dx=10, dy=20, dz=30)["name"]
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
