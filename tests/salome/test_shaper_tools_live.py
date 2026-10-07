import pytest

from ._salome_root import SALOME_ROOT

pytestmark = [
    pytest.mark.salome,
    pytest.mark.skipif(SALOME_ROOT is None,
                       reason="set SALOME_ROOT or salome.root in config.json"),
]

_PLATE = [
    {"id": "L1", "type": "line", "start": [0, 0], "end": [40, 0]},
    {"id": "L2", "type": "line", "start": [40, 0], "end": [40, 20]},
    {"id": "L3", "type": "line", "start": [40, 20], "end": [0, 20]},
    {"id": "L4", "type": "line", "start": [0, 20], "end": [0, 0]},
    {"id": "C1", "type": "circle", "center": [20, 10], "radius": 5},
]
_CLOSE = [{"type": "coincident", "refs": [a, b]} for a, b in
          (("L1.end", "L2.start"), ("L2.end", "L3.start"),
           ("L3.end", "L4.start"), ("L4.end", "L1.start"))]


def _call(client, name, **args):
    return client.call_tool(name, args, module="shaper")


def _plate(client):
    sk = _call(client, "create_sketch", plane="XOY")
    assert sk["ok"] is True, sk.get("error")
    name = sk["sketch"]
    r = _call(client, "sketch_add", sketch=name, entities=_PLATE)
    assert r["ok"] is True, r.get("error")
    r = _call(client, "sketch_constrain", sketch=name, constraints=_CLOSE)
    assert r["ok"] is True, r.get("error")
    return name, r


def test_list_tools_for_shaper(agent_client):
    names = {t["name"] for t in agent_client.list_tools(module="shaper")["tools"]}
    assert {"create_sketch", "sketch_add", "sketch_constrain", "sketch_remove",
            "sketch_status", "make_face", "extrude", "extrude_cut", "extrude_fuse",
            "revolve", "measure"} <= names


def test_new_sketch_is_empty_and_anchored(agent_client):
    sk = _call(agent_client, "create_sketch", plane="XOY")
    assert sk["ok"] is True, sk.get("error")
    assert sk["dof"] == 0
    assert sk["regions"] == [] and sk["loose_ends"] == []


def test_plate_with_hole_has_two_regions(agent_client):
    _, r = _plate(agent_client)
    areas = sorted(round(g["area"], 2) for g in r["regions"])
    assert areas == [78.54, 721.46]
    assert r["loose_ends"] == []
    assert r["rejected"] == []


def test_conflicting_constraint_is_rejected_by_id_and_the_rest_kept(agent_client):
    name, _ = _plate(agent_client)
    r = _call(agent_client, "sketch_constrain", sketch=name, constraints=[
        {"id": "H1", "type": "horizontal", "refs": ["L1"]},
        {"id": "H2", "type": "horizontal", "refs": ["L3"]},
        {"id": "V1", "type": "vertical", "refs": ["L2"]},
        {"id": "V2", "type": "vertical", "refs": ["L4"]},
        {"id": "W", "type": "length", "refs": ["L1"], "value": 40},
        {"id": "BAD", "type": "length", "refs": ["L3"], "value": 30},
    ])
    assert r["ok"] is True, r.get("error")
    assert [x["id"] for x in r["rejected"]] == ["BAD"]
    st = _call(agent_client, "sketch_status", sketch=name)
    assert {"H1", "H2", "V1", "V2", "W"} <= set(st["constraints"])
    assert "BAD" not in st["constraints"]


def test_solver_moves_are_reported(agent_client):
    name, _ = _plate(agent_client)
    r = _call(agent_client, "sketch_constrain", sketch=name, constraints=[
        {"type": "radius", "refs": ["C1"], "value": 7}])
    assert r["ok"] is True, r.get("error")
    assert "C1" in r["moved"]
    assert r["moved"]["C1"]["radius"] == pytest.approx(7)


def test_removing_an_entity_cascades_to_its_constraints(agent_client):
    name, _ = _plate(agent_client)
    r = _call(agent_client, "sketch_remove", sketch=name, ids=["L2"])
    assert r["ok"] is True, r.get("error")
    assert "L2" in r["removed"]
    assert len(r["removed"]) == 3  # L2 and its two coincidences
    ends = {e["ref"] for e in r["loose_ends"]}
    assert ends == {"L1.end", "L3.start"}


def test_invalid_batch_changes_nothing(agent_client):
    name, _ = _plate(agent_client)
    r = _call(agent_client, "sketch_add", sketch=name, entities=[
        {"id": "L5", "type": "line", "start": [0, 0], "end": [5, 5]},
        {"id": "L1", "type": "line", "start": [0, 0], "end": [5, 5]}])
    assert r["ok"] is False
    assert "L1" in r["error"]
    r = _call(agent_client, "sketch_constrain", sketch=name, constraints=[
        {"type": "radius", "refs": ["L1"], "value": 3}])
    assert r["ok"] is False
    st = _call(agent_client, "sketch_status", sketch=name)
    assert "L5" not in st["entities"]


def test_construction_lines_form_no_region_and_no_loose_end(agent_client):
    sk = _call(agent_client, "create_sketch", plane="XOY")
    r = _call(agent_client, "sketch_add", sketch=sk["sketch"], entities=[
        {"id": "AX", "type": "line", "start": [0, 0], "end": [30, 30],
         "construction": True}])
    assert r["ok"] is True, r.get("error")
    assert r["loose_ends"] == [] and r["regions"] == []


def test_face_extrude_and_cut(agent_client):
    name, r = _plate(agent_client)
    outer = next(g["index"] for g in r["regions"] if round(g["area"], 2) == 721.46)
    face = _call(agent_client, "make_face", sketch=name, regions=[outer])
    assert face["ok"] is True, face.get("error")
    assert face["measure"]["area"] == pytest.approx(721.46, abs=0.01)
    ext = _call(agent_client, "extrude", base=face["name"], size=10)
    assert ext["ok"] is True, ext.get("error")
    assert ext["measure"]["volume"] == pytest.approx(7214.602, abs=0.01)
    assert ext["measure"]["valid"] is True

    hole = _call(agent_client, "create_sketch", plane="XOY")
    _call(agent_client, "sketch_add", sketch=hole["sketch"], entities=[
        {"id": "C", "type": "circle", "center": [5, 5], "radius": 2}])
    cut = _call(agent_client, "extrude_cut", base=hole["sketch"], target=ext["name"],
                through_all=True)
    assert cut["ok"] is True, cut.get("error")
    assert cut["measure"]["volume"] == pytest.approx(7214.602 - 3.14159265 * 4 * 10,
                                                     abs=0.05)
    assert cut["measure"]["size"] == pytest.approx([40, 20, 10])
    # GEOM does not see SHAPER results: no GEOM copy is made of them
    m = agent_client.call_tool("measure", {"names": [cut["name"]]})
    assert not m.get("ok") or m["measurements"].get(cut["name"]) is None


def test_measure_gives_the_exact_box_of_a_curved_solid(agent_client):
    sk = _call(agent_client, "create_sketch", plane="XOY")
    _call(agent_client, "sketch_add", sketch=sk["sketch"], entities=[
        {"id": "C", "type": "circle", "center": [50, 0], "radius": 10}])
    face = _call(agent_client, "make_face", sketch=sk["sketch"], regions=[0])
    torus = _call(agent_client, "revolve", base=face["name"], axis="OY", angle=360)
    assert torus["ok"] is True, torus.get("error")
    # the fast box encloses control points (+-64.94); the exact one is +-60
    assert torus["measure"]["bbox"] == pytest.approx([-60, 60, -10, 10, -60, 60], abs=1e-3)
    assert torus["measure"]["volume"] == pytest.approx(2 * 3.14159265 ** 2 * 50 * 100, rel=1e-6)
    assert (torus["measure"]["faces"], torus["measure"]["edges"]) == (1, 2)


def test_cut_that_misses_carries_a_delta_note(agent_client):
    name, r = _plate(agent_client)
    outer = next(g["index"] for g in r["regions"] if round(g["area"], 2) == 721.46)
    face = _call(agent_client, "make_face", sketch=name, regions=[outer])
    ext = _call(agent_client, "extrude", base=face["name"], size=10)
    far = _call(agent_client, "create_sketch", plane="XOY")
    _call(agent_client, "sketch_add", sketch=far["sketch"], entities=[
        {"id": "C", "type": "circle", "center": [100, 100], "radius": 2}])
    cut = _call(agent_client, "extrude_cut", base=far["sketch"], target=ext["name"],
                through_all=True)
    assert cut["ok"] is True, cut.get("error")
    assert "delta" in cut


def test_revolve_about_an_axis(agent_client):
    sk = _call(agent_client, "create_sketch", plane="XOZ")
    _call(agent_client, "sketch_add", sketch=sk["sketch"], entities=[
        {"id": "A", "type": "line", "start": [10, 0], "end": [20, 0]},
        {"id": "B", "type": "line", "start": [20, 0], "end": [20, 5]},
        {"id": "C", "type": "line", "start": [20, 5], "end": [10, 5]},
        {"id": "D", "type": "line", "start": [10, 5], "end": [10, 0]}])
    rev = _call(agent_client, "revolve", base=sk["sketch"], axis="OZ", angle=360)
    assert rev["ok"] is True, rev.get("error")
    assert rev["measure"]["volume"] == pytest.approx(3.14159265 * (400 - 100) * 5, rel=1e-3)


def test_result_py_rebuilds_the_same_solid(agent_client, tmp_path):
    import re
    import subprocess

    from salome_agent.nodes.assembler import assemble

    from .conftest import _salome_subprocess_env

    calls = []

    def call(name, **args):
        r = _call(agent_client, name, **args)
        assert r["ok"] is True, r.get("error")
        calls.append({"tool": name, "tool_args": args, "records": r["records"]})
        return r

    sk = call("create_sketch", plane="XOY")["sketch"]
    call("sketch_add", sketch=sk, entities=_PLATE)
    # the constraint set a live run wrote (run 5359c1def3de): fully constrained
    r = call("sketch_constrain", sketch=sk, constraints=[
        {"type": "coincident", "refs": ["L1.start", "ORIGIN"]},
        {"type": "horizontal", "refs": ["L1"]}, {"type": "vertical", "refs": ["L2"]},
        {"type": "horizontal", "refs": ["L3"]}, {"type": "vertical", "refs": ["L4"]},
        *_CLOSE,
        {"type": "length", "refs": ["L1"], "value": 40},
        {"type": "length", "refs": ["L4"], "value": 20},
        {"type": "distance", "refs": ["C1.center", "L1"], "value": 10},
        {"type": "distance", "refs": ["C1.center", "L4"], "value": 20},
        {"type": "radius", "refs": ["C1"], "value": 6}])
    assert r["dof"] == 0 and r["rejected"] == []
    outer = max(r["regions"], key=lambda g: g["area"])["index"]
    face = call("make_face", sketch=sk, regions=[outer])["name"]
    ext = call("extrude", base=face, size=10)
    live = ext["measure"]["volume"]

    log = [rec for c in calls for rec in c["records"]]
    script = assemble({"stages": {0: {"module": "shaper", "status": "ok", "script": None,
                                      "result_names": [ext["name"]], "call_log": log,
                                      "calls": calls}},
                       "run_id": "t", "user_request": "plate", "plan": []})
    script += ("from GeomAlgoAPI import GeomAlgoAPI_ShapeTools as _T\n"
               f"print('VOLUME', _T.volume({ext['name']}.feature().firstResult().shape()))\n")
    path = tmp_path / "result.py"
    path.write_text(script)
    out = subprocess.run([f"{SALOME_ROOT}/salome", "shell", "--", "python3", str(path)],
                         env=_salome_subprocess_env(), capture_output=True, text=True,
                         timeout=240)
    m = re.search(r"VOLUME ([0-9.eE+-]+)", out.stdout)
    assert m, out.stdout[-2000:] + out.stderr[-2000:]
    assert float(m.group(1)) == pytest.approx(live, abs=1e-3)
    assert live == pytest.approx((800 - 3.14159265 * 36) * 10, abs=0.01)


def test_a_rejected_constraint_leaves_the_geometry_as_it_was(agent_client):
    name, _ = _plate(agent_client)
    r = _call(agent_client, "sketch_constrain", sketch=name, constraints=[
        {"type": "horizontal", "refs": ["L1"]}, {"type": "horizontal", "refs": ["L3"]},
        {"type": "vertical", "refs": ["L2"]}, {"type": "vertical", "refs": ["L4"]},
        {"type": "length", "refs": ["L1"], "value": 40}])
    assert r["rejected"] == []
    before = _call(agent_client, "sketch_status", sketch=name)["entities"]
    r = _call(agent_client, "sketch_constrain", sketch=name, constraints=[
        {"id": "BAD", "type": "length", "refs": ["L3"], "value": 25}])
    assert [x["id"] for x in r["rejected"]] == ["BAD"]
    after = _call(agent_client, "sketch_status", sketch=name)["entities"]
    for i, geo in before.items():
        for k, v in geo.items():
            assert after[i][k] == pytest.approx(v, abs=1e-6), (i, k)


def _rect(tag, x0, y0, x1, y1):
    pts = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
    return [{"id": f"{tag}{i}", "type": "line", "start": pts[i], "end": pts[(i + 1) % 4]}
            for i in range(4)]


def test_make_face_merges_touching_regions_and_result_py_rebuilds_it(agent_client, tmp_path):
    # a plus sign drawn as two whole overlapping bars: 5 regions, one outline
    import re
    import subprocess

    from salome_agent.nodes.assembler import assemble

    from .conftest import _salome_subprocess_env

    calls = []

    def call(name, **args):
        r = _call(agent_client, name, **args)
        assert r["ok"] is True, r.get("error")
        calls.append({"tool": name, "tool_args": args, "records": r["records"]})
        return r

    sk = call("create_sketch", plane="XOY")["sketch"]
    r = call("sketch_add", sketch=sk,
             entities=_rect("H", -20, -5, 20, 5) + _rect("V", -5, -20, 5, 20))
    assert len(r["regions"]) == 5
    face = call("make_face", sketch=sk, regions=[g["index"] for g in r["regions"]])
    assert face["measure"]["faces"] == 1
    assert face["measure"]["area"] == pytest.approx(700, abs=1e-6)

    log = [rec for c in calls for rec in c["records"]]
    script = assemble({"stages": {0: {"module": "shaper", "status": "ok", "script": None,
                                      "result_names": [face["name"]], "call_log": log,
                                      "calls": calls}},
                       "run_id": "t", "user_request": "plus", "plan": []})
    script += ("from GeomAlgoAPI import GeomAlgoAPI_ShapeTools as _T\n"
               f"_r = {face['name']}.feature().results()\n"
               "print('AREA', _T.area(_r[0].shape()), 'RESULTS', len(_r))\n")
    path = tmp_path / "result.py"
    path.write_text(script)
    out = subprocess.run([f"{SALOME_ROOT}/salome", "shell", "--", "python3", str(path)],
                         env=_salome_subprocess_env(), capture_output=True, text=True,
                         timeout=240)
    m = re.search(r"AREA ([0-9.eE+-]+) RESULTS (\d+)", out.stdout)
    assert m, out.stdout[-2000:] + out.stderr[-2000:]
    assert float(m.group(1)) == pytest.approx(700, abs=1e-6)
    assert m.group(2) == "1"


def test_a_merged_face_can_be_discarded(agent_client):
    sk = _call(agent_client, "create_sketch", plane="XOY")["sketch"]
    r = _call(agent_client, "sketch_add", sketch=sk,
              entities=_rect("H", -20, -5, 20, 5) + _rect("V", -5, -20, 5, 20))
    face = _call(agent_client, "make_face", sketch=sk,
                 regions=[g["index"] for g in r["regions"]])["name"]
    assert agent_client.forget([face])["removed"] == [face]
    again = _call(agent_client, "make_face", sketch=sk, regions="all")
    assert again["ok"] is True, again.get("error")
    assert again["measure"]["faces"] == 1


def _rect_face(client, x0, y0, x1, y1):
    sk = _call(client, "create_sketch", plane="XOY")["sketch"]
    _call(client, "sketch_add", sketch=sk, entities=_rect("R", x0, y0, x1, y1))
    face = _call(client, "make_face", sketch=sk, regions="all")
    assert face["ok"] is True, face.get("error")
    return face["name"]


def test_list_tools_has_the_booleans(agent_client):
    names = {t["name"] for t in agent_client.list_tools(module="shaper")["tools"]}
    assert {"fuse", "cut", "common"} <= names


def test_fuse_of_overlapping_faces_is_one_face(agent_client):
    a = _rect_face(agent_client, -20, -5, 20, 5)
    b = _rect_face(agent_client, -5, -20, 5, 20)
    r = _call(agent_client, "fuse", objects=[a, b])
    assert r["ok"] is True, r.get("error")
    assert r["measure"]["faces"] == 1
    assert r["measure"]["area"] == pytest.approx(700, abs=1e-6)
    assert r["measure"]["valid"] is True


def test_measure_marks_the_operands_a_fuse_used_up(agent_client):
    a = _rect_face(agent_client, -20, -5, 20, 5)
    b = _rect_face(agent_client, -5, -20, 5, 20)
    f = _call(agent_client, "fuse", objects=[a, b])["name"]
    m = _call(agent_client, "measure", names=[a, b, f])["measurements"]
    assert m[a]["concealed"] is True and m[b]["concealed"] is True
    assert "concealed" not in m[f]


def test_cut_and_common_of_faces(agent_client):
    a = _rect_face(agent_client, 0, 0, 40, 20)
    b = _rect_face(agent_client, 30, 0, 50, 20)
    r = _call(agent_client, "common", objects=[a], tools=[b])
    assert r["ok"] is True, r.get("error")
    assert r["measure"]["area"] == pytest.approx(200, abs=1e-6)
    a = _rect_face(agent_client, 0, 0, 40, 20)
    b = _rect_face(agent_client, 30, 0, 50, 20)
    r = _call(agent_client, "cut", objects=[a], tools=[b])
    assert r["ok"] is True, r.get("error")
    assert r["measure"]["area"] == pytest.approx(600, abs=1e-6)
    assert r["measure"]["faces"] == 1


def test_fuse_of_solids(agent_client):
    a = _call(agent_client, "extrude", base=_rect_face(agent_client, 0, 0, 10, 10), size=10)
    b = _call(agent_client, "extrude", base=_rect_face(agent_client, 5, 0, 15, 10), size=10)
    r = _call(agent_client, "fuse", objects=[a["name"], b["name"]])
    assert r["ok"] is True, r.get("error")
    assert r["measure"]["volume"] == pytest.approx(1500, abs=1e-6)
    assert r["measure"]["faces"] == 6


def test_a_cut_that_misses_carries_a_note(agent_client):
    a = _rect_face(agent_client, 0, 0, 10, 10)
    b = _rect_face(agent_client, 50, 50, 60, 60)
    r = _call(agent_client, "cut", objects=[a], tools=[b])
    assert r["ok"] is True, r.get("error")
    assert "delta" in r


def test_a_boolean_refuses_a_sketch(agent_client):
    a = _rect_face(agent_client, 0, 0, 10, 10)
    sk = _call(agent_client, "create_sketch", plane="XOY")["sketch"]
    _call(agent_client, "sketch_add", sketch=sk, entities=_rect("R", 5, 5, 15, 15))
    r = _call(agent_client, "fuse", objects=[a, sk])
    assert r["ok"] is False
    assert "make_face" in r["error"]


def test_result_py_rebuilds_a_boolean(agent_client, tmp_path):
    import re
    import subprocess

    from salome_agent.nodes.assembler import assemble

    from .conftest import _salome_subprocess_env

    calls = []

    def call(name, **args):
        r = _call(agent_client, name, **args)
        assert r["ok"] is True, r.get("error")
        calls.append({"tool": name, "tool_args": args, "records": r["records"]})
        return r

    faces = []
    for x0, y0, x1, y1 in ((-20, -5, 20, 5), (-5, -20, 5, 20), (-2, -2, 2, 2)):
        sk = call("create_sketch", plane="XOY")["sketch"]
        call("sketch_add", sketch=sk, entities=_rect("R", x0, y0, x1, y1))
        faces.append(call("make_face", sketch=sk, regions="all")["name"])
    plus = call("fuse", objects=faces[:2])["name"]
    holed = call("cut", objects=[plus], tools=[faces[2]])
    assert holed["measure"]["area"] == pytest.approx(684, abs=1e-6)

    log = [rec for c in calls for rec in c["records"]]
    script = assemble({"stages": {0: {"module": "shaper", "status": "ok", "script": None,
                                      "result_names": [holed["name"]], "call_log": log,
                                      "calls": calls}},
                       "run_id": "t", "user_request": "plus", "plan": []})
    script += ("from GeomAlgoAPI import GeomAlgoAPI_ShapeTools as _T\n"
               f"_r = {holed['name']}.feature().results()\n"
               "print('AREA', _T.area(_r[0].shape()), 'RESULTS', len(_r))\n")
    path = tmp_path / "result.py"
    path.write_text(script)
    out = subprocess.run([f"{SALOME_ROOT}/salome", "shell", "--", "python3", str(path)],
                         env=_salome_subprocess_env(), capture_output=True, text=True,
                         timeout=240)
    m = re.search(r"AREA ([0-9.eE+-]+) RESULTS (\d+)", out.stdout)
    assert m, out.stdout[-2000:] + out.stderr[-2000:]
    assert float(m.group(1)) == pytest.approx(684, abs=1e-6)
    assert m.group(2) == "1"


def test_region_names_use_only_published_entity_names(agent_client):
    # SHAPER's own counters (SketchLine_N) differ in result.py once anything
    # built live was discarded, so a face selected by them is lost on replay
    name = _call(agent_client, "create_sketch", plane="XOY")["sketch"]
    r = _call(agent_client, "sketch_add", sketch=name, entities=_rect("R", 0, 0, 10, 5))
    face = _call(agent_client, "make_face", sketch=name, regions=[0])
    assert face["ok"] is True, face.get("error")
    code = "\n".join(rec["code"] for rec in face["records"])
    assert "SketchLine_" not in code and "SketchCircle_" not in code, code


def test_result_py_selects_a_merged_face_by_its_name(agent_client, tmp_path):
    # make_face on several regions publishes a fuse made after a helper face;
    # result.py must give the name to the fuse, or a later selection by that
    # name lands on the helper, which the fuse has consumed
    import re
    import subprocess

    from salome_agent.nodes.assembler import assemble

    from .conftest import _salome_subprocess_env

    calls = []

    def call(name, **args):
        r = _call(agent_client, name, **args)
        assert r["ok"] is True, r.get("error")
        calls.append({"tool": name, "tool_args": args, "records": r["records"]})
        return r

    sk = call("create_sketch", plane="XOY")["sketch"]
    r = call("sketch_add", sketch=sk,
             entities=_rect("H", -20, -5, 20, 5) + _rect("V", -5, -20, 5, 20))
    plus = call("make_face", sketch=sk, regions=[g["index"] for g in r["regions"]])["name"]
    sk2 = call("create_sketch", plane="XOY")["sketch"]
    call("sketch_add", sketch=sk2, entities=_rect("R", 15, -2, 30, 2))
    bar = call("make_face", sketch=sk2, regions="all")["name"]
    both = call("fuse", objects=[plus, bar])
    assert both["measure"]["area"] == pytest.approx(700 + 40, abs=1e-6)

    log = [rec for c in calls for rec in c["records"]]
    script = assemble({"stages": {0: {"module": "shaper", "status": "ok", "script": None,
                                      "result_names": [both["name"]], "call_log": log,
                                      "calls": calls}},
                       "run_id": "t", "user_request": "plus", "plan": []})
    script += ("from GeomAlgoAPI import GeomAlgoAPI_ShapeTools as _T\n"
               f"_r = {both['name']}.feature().results()\n"
               "print('AREA', _T.area(_r[0].shape()), 'RESULTS', len(_r))\n")
    path = tmp_path / "result.py"
    path.write_text(script)
    out = subprocess.run([f"{SALOME_ROOT}/salome", "shell", "--", "python3", str(path)],
                         env=_salome_subprocess_env(), capture_output=True, text=True,
                         timeout=240)
    m = re.search(r"AREA ([0-9.eE+-]+) RESULTS (\d+)", out.stdout)
    assert m, out.stdout[-2000:] + out.stderr[-2000:]
    assert float(m.group(1)) == pytest.approx(740, abs=1e-6)


def test_check_results_sees_a_face_whose_sketch_was_edited(agent_client):
    # session 77: make_face, then sketch_remove of the lines it was built from
    sk = _call(agent_client, "create_sketch", plane="XOY")["sketch"]
    _call(agent_client, "sketch_add", sketch=sk, entities=_rect("R", 0, 0, 10, 5))
    face = _call(agent_client, "make_face", sketch=sk, regions="all")["name"]
    good = _rect_face(agent_client, 20, 0, 30, 5)
    r = agent_client.check_results([face, good, sk, "nope"], module="shaper")
    assert r["ok"] is True, r
    assert r["results"][face]["status"] == "ok"
    assert r["results"][sk]["status"] == "skip"
    assert r["results"]["nope"]["status"] == "missing"
    _call(agent_client, "sketch_remove", sketch=sk, ids=["R0", "R1", "R2", "R3"])
    r = agent_client.check_results([face, good], module="shaper")
    assert r["results"][face]["status"] == "broken", r
    assert r["results"][good]["status"] == "ok"
