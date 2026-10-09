import pytest

from ._salome_root import SALOME_ROOT

pytestmark = [pytest.mark.salome,
              pytest.mark.skipif(SALOME_ROOT is None,
                                 reason="set SALOME_ROOT or salome.root in config.json")]


def _call(client, module, tool, **args):
    return client.call_tool(tool, args, module=module)


def _ok(client, module, tool, **args):
    r = _call(client, module, tool, **args)
    assert r["ok"] is True, r.get("error")
    return r


def shaper_box(client, dx=10, dy=20, dz=30):
    return _ok(client, "shaper", "make_box", dx=dx, dy=dy, dz=dz,
               center=[dx / 2, dy / 2, dz / 2])["name"]


def geom_box(client, dx=10, dy=20, dz=30):
    return _ok(client, "geom", "make_box", dx=dx, dy=dy, dz=dz)["result_shape"]


def top_face_group(client, shape, name="top"):
    faces = _ok(client, "smesh", "get_subshapes", shape=shape, sub_type="face",
                where={"sorted_by": "z", "take": "max", "count": 1})["subshapes"]
    return _ok(client, "smesh", "name_subshapes", shape=shape,
               ids=[faces[0]["id"]], name=name)


def test_create_mesh_on_shaper_and_geom(agent_client):
    for shape in (shaper_box(agent_client), geom_box(agent_client)):
        m = _ok(agent_client, "smesh", "create_mesh", shape=shape)
        assert m["name"].startswith("mesh_") and m["shape"] == shape


def test_name_subshapes_on_shaper_shape(agent_client):
    b = shaper_box(agent_client)
    g = top_face_group(agent_client, b)
    assert g["sub_type"] == "face" and g["count"] == 1
    assert g["measure"]["area"] == pytest.approx(200)
    info = _ok(agent_client, "smesh", "shape_info", names=[b, "top"])["shapes"]
    assert info[b]["volume"] == pytest.approx(6000) and info["top"]["area"] == pytest.approx(200)


def test_geom_shape_group_and_mesh(agent_client):
    b = geom_box(agent_client)
    g = top_face_group(agent_client, b, name="lid")
    assert g["sub_type"] == "face" and g["measure"]["area"] == pytest.approx(200)
    _ok(agent_client, "smesh", "create_mesh", shape="lid")


def test_name_taken_is_refused(agent_client):
    b = shaper_box(agent_client)
    top_face_group(agent_client, b, name="inlet")
    r = _call(agent_client, "smesh", "name_subshapes", shape=b, ids=["F1"], name="inlet")
    assert r["ok"] is False and "taken" in r["error"]


def test_every_algorithm_constant_resolves(agent_client):
    from executor.smesh.setup import ALGORITHMS
    b = shaper_box(agent_client)
    for algorithm, (dim, _, _) in ALGORITHMS.items():
        m = _ok(agent_client, "smesh", "create_mesh", shape=b)["name"]
        _ok(agent_client, "smesh", "set_algorithm", mesh=m, dimension=dim, algorithm=algorithm)


def test_preset_replaces_preset(agent_client):
    b = shaper_box(agent_client)
    m = _ok(agent_client, "smesh", "create_mesh", shape=b)["name"]
    _ok(agent_client, "smesh", "mesh_hexa", mesh=m, segments=4)
    r = _ok(agent_client, "smesh", "mesh_tetra", mesh=m, max_size=5)
    assert "RemoveHypothesis" in r["records"][0]["code"]


def test_hypothesis_needs_its_algorithm(agent_client):
    b = shaper_box(agent_client)
    m = _ok(agent_client, "smesh", "create_mesh", shape=b)["name"]
    r = _call(agent_client, "smesh", "set_hypothesis", mesh=m, kind="local_length",
              params={"length": 2})
    assert r["ok"] is False and "set_algorithm(dimension=1" in r["error"]
    _ok(agent_client, "smesh", "set_algorithm", mesh=m, dimension=1, algorithm="segment")
    _ok(agent_client, "smesh", "set_hypothesis", mesh=m, kind="local_length",
        params={"length": 2})


def test_tetra_compute_measures(agent_client):
    b = shaper_box(agent_client)
    m = _ok(agent_client, "smesh", "create_mesh", shape=b)["name"]
    _ok(agent_client, "smesh", "mesh_tetra", mesh=m, max_size=5)
    assert _ok(agent_client, "smesh", "evaluate", mesh=m)["predicted_elements"] > 0
    r = _ok(agent_client, "smesh", "compute", mesh=m)
    assert r["computed"] is True and r["errors"] == []
    assert r["tetra"] > 0 and r["volume"] == pytest.approx(6000, rel=1e-6)
    assert r["shape_volume"] == pytest.approx(6000)
    assert 1.0 <= r["quality"]["aspect_ratio"]["min"] <= r["quality"]["aspect_ratio"]["max"]
    q = _ok(agent_client, "smesh", "check_quality", mesh=m, criterion="aspect_ratio",
            threshold=2.0)
    assert q["above_threshold"] >= 0


def test_hexa_on_cylinder_names_the_failed_faces(agent_client):
    cyl = _ok(agent_client, "shaper", "make_cylinder", radius=5, height=10,
              origin=[0, 0, 0])["name"]
    m = _ok(agent_client, "smesh", "create_mesh", shape=cyl)["name"]
    _ok(agent_client, "smesh", "mesh_hexa", mesh=m, segments=4)
    r = _ok(agent_client, "smesh", "compute", mesh=m)
    assert r["computed"] is False
    assert any("a face at (0.0, 0.0, 10.0)" in e and "4 sides" in e for e in r["errors"])


def test_compute_refuses_above_cap(agent_client):
    b = shaper_box(agent_client, dx=100, dy=100, dz=100)
    m = _ok(agent_client, "smesh", "create_mesh", shape=b)["name"]
    _ok(agent_client, "smesh", "mesh_tetra", mesh=m, max_size=0.5)
    r = _call(agent_client, "smesh", "compute", mesh=m)
    assert r["ok"] is False and "limit" in r["error"]


def test_body_fitted_and_prism_are_predicted(agent_client):
    b = shaper_box(agent_client, dx=100, dy=50, dz=20)
    m = _ok(agent_client, "smesh", "create_mesh", shape=b)["name"]
    _ok(agent_client, "smesh", "mesh_body_fitted", mesh=m, size=2)
    assert _ok(agent_client, "smesh", "evaluate", mesh=m)["predicted_elements"] == 12500
    _ok(agent_client, "smesh", "mesh_body_fitted", mesh=m, size=0.1)
    r = _call(agent_client, "smesh", "compute", mesh=m)
    assert r["ok"] is False and "limit" in r["error"]
    base = _ok(agent_client, "smesh", "get_subshapes", shape=b, sub_type="face",
               where={"sorted_by": "z", "take": "min", "count": 1})["subshapes"]
    _ok(agent_client, "smesh", "name_subshapes", shape=b, ids=[base[0]["id"]], name="floor")
    p = _ok(agent_client, "smesh", "create_mesh", shape=b)["name"]
    _ok(agent_client, "smesh", "mesh_prism", mesh=p, end_face="floor", layers=10, face_size=5)
    n = _ok(agent_client, "smesh", "evaluate", mesh=p)["predicted_elements"]
    r = _ok(agent_client, "smesh", "compute", mesh=p)
    assert r["prism"] * 0.8 <= n <= (r["prism"] + r["faces"] + r["edges"]) * 1.2


def test_unpredictable_setup_computes_with_a_note(agent_client):
    b = shaper_box(agent_client)
    m = _ok(agent_client, "smesh", "create_mesh", shape=b)["name"]
    _ok(agent_client, "smesh", "set_algorithm", mesh=m, dimension=1, algorithm="segment")
    _ok(agent_client, "smesh", "set_hypothesis", mesh=m, kind="local_length",
        params={"length": 2})
    _ok(agent_client, "smesh", "set_algorithm", mesh=m, dimension=2,
        algorithm="quadrangle_medial_axis")
    e = _ok(agent_client, "smesh", "evaluate", mesh=m)
    assert e["predicted_elements"] is None and "cannot predict" in e["note"]
    r = _ok(agent_client, "smesh", "compute", mesh=m)
    assert r["computed"] and r["faces"] > 0 and "without the" in r["note"]


def test_local_size_and_edge_segments(agent_client):
    b = shaper_box(agent_client)
    top_face_group(agent_client, b, name="lid")
    edges = _ok(agent_client, "smesh", "get_subshapes", shape=b, sub_type="edge",
                where={"by_size": "longest", "count": 1})["subshapes"]
    _ok(agent_client, "smesh", "name_subshapes", shape=b, ids=[edges[0]["id"]], name="long")
    m = _ok(agent_client, "smesh", "create_mesh", shape=b)["name"]
    _ok(agent_client, "smesh", "mesh_tetra", mesh=m, max_size=5)
    _ok(agent_client, "smesh", "local_size", mesh=m, on=["lid"], size=1)
    _ok(agent_client, "smesh", "edge_segments", mesh=m, on=["long"], count=40)
    _ok(agent_client, "smesh", "group_on_geometry", mesh=m, group="long")
    r = _ok(agent_client, "smesh", "compute", mesh=m)
    assert r["computed"] and r["groups"]["long"] == 40


def test_viscous_layers_on_tetra(agent_client):
    b = shaper_box(agent_client, dx=20, dy=20, dz=20)
    top_face_group(agent_client, b, name="outlet")
    m = _ok(agent_client, "smesh", "create_mesh", shape=b)["name"]
    _ok(agent_client, "smesh", "mesh_tetra", mesh=m, max_size=4)
    _ok(agent_client, "smesh", "viscous_layers", mesh=m, thickness=1, layers=3,
        stretch=1.2, faces=["outlet"], faces_are="without_layers")
    r = _ok(agent_client, "smesh", "compute", mesh=m)
    assert r["computed"] and r["errors"] == [] and r["prism"] > 0
    assert r["volume"] == pytest.approx(8000, rel=1e-3)


def test_group_on_late_shaper_group_and_from_geometry(agent_client):
    b = shaper_box(agent_client)
    _ok(agent_client, "shaper", "make_group", shape=b, ids=["F1"], name="side")
    m = _ok(agent_client, "smesh", "create_mesh", shape=b)["name"]
    _ok(agent_client, "smesh", "mesh_tetra", mesh=m, max_size=5)
    top_face_group(agent_client, b, name="inlet")
    g = _ok(agent_client, "smesh", "groups_from_geometry", mesh=m)
    assert set(g["groups"]) == {"side", "inlet"}
    r = _ok(agent_client, "smesh", "compute", mesh=m)
    assert r["groups"]["inlet"] > 0 and r["groups"]["side"] > 0
    assert _call(agent_client, "smesh", "group_on_geometry", mesh=m, group="inlet")["ok"] is False


def test_check_results_refuses_until_computed(agent_client):
    b = shaper_box(agent_client)
    m = _ok(agent_client, "smesh", "create_mesh", shape=b)["name"]
    _ok(agent_client, "smesh", "mesh_tetra", mesh=m, max_size=5)
    r = agent_client.check_results([m], module="smesh")["results"][m]
    assert r["status"] == "not_meshed" and "not computed" in r["detail"]
    _ok(agent_client, "smesh", "compute", mesh=m)
    assert agent_client.check_results([m], module="smesh")["results"][m] == {"status": "ok"}


def _in_study(client, name):
    r = client.run(f"print(len(salome.myStudy.FindObjectByName({name!r}, 'SMESH')) + "
                   f"len(salome.myStudy.FindObjectByName({name!r}, 'SHAPERSTUDY')))", 30)
    return int(r["stdout"].strip())


def test_discard_mesh_and_group(agent_client):
    b = shaper_box(agent_client)
    top_face_group(agent_client, b, name="gone")
    m = _ok(agent_client, "smesh", "create_mesh", shape=b)["name"]
    before = _in_study(agent_client, m)
    assert _in_study(agent_client, "gone") == 1
    removed = agent_client.forget([m, "gone"])["removed"]
    assert set(removed) >= {m, "gone"}
    assert _in_study(agent_client, m) == before - 1
    assert _in_study(agent_client, "gone") == 0


def test_export_med(agent_client, tmp_path):
    b = shaper_box(agent_client)
    m = _ok(agent_client, "smesh", "create_mesh", shape=b)["name"]
    _ok(agent_client, "smesh", "mesh_tetra", mesh=m, max_size=5)
    _ok(agent_client, "smesh", "compute", mesh=m)
    r = _ok(agent_client, "smesh", "export_mesh", mesh=m, format="med")
    import os
    try:
        assert os.path.getsize(r["path"]) > 0
    finally:
        os.remove(r["path"])


def test_prism_on_extruded_plate(agent_client):
    b = shaper_box(agent_client, dx=40, dy=20, dz=10)
    bottom = _ok(agent_client, "smesh", "get_subshapes", shape=b, sub_type="face",
                 where={"sorted_by": "z", "take": "min", "count": 1})["subshapes"]
    _ok(agent_client, "smesh", "name_subshapes", shape=b, ids=[bottom[0]["id"]], name="base")
    m = _ok(agent_client, "smesh", "create_mesh", shape=b)["name"]
    _ok(agent_client, "smesh", "mesh_prism", mesh=m, end_face="base", layers=5, face_size=4)
    r = _ok(agent_client, "smesh", "compute", mesh=m)
    assert r["computed"] and r["prism"] > 0 and r["volume"] == pytest.approx(8000, rel=1e-6)


def test_shape_info_reports_a_consumed_shape_per_name(agent_client):
    # A run prefix keeps the earlier tests' box_1 (still in the study) out of it.
    import json
    from salome_agent.executor_client import ExecutorClient
    from .conftest import _HS
    agent_client.close()
    hs = json.loads(open(_HS).read())
    client = ExecutorClient(hs["host"], hs["port"], hs["token"], "consumed0001")
    try:
        b = shaper_box(client)
        cyl = _ok(client, "shaper", "make_cylinder", radius=2, height=30,
                  origin=[5, 10, 0])["name"]
        cut = _ok(client, "shaper", "cut", objects=[b], tools=[cyl])["name"]
        info = _ok(client, "smesh", "shape_info", names=[b, cut])["shapes"]
        assert "error" in info[b]
        assert info[cut]["volume"] == pytest.approx(6000 - 3.14159265 * 4 * 30, rel=1e-4)
    finally:
        client.close()


def test_forget_a_group_drops_its_mesh_groups(agent_client):
    b = shaper_box(agent_client)
    top_face_group(agent_client, b, name="cap")
    m = _ok(agent_client, "smesh", "create_mesh", shape=b)["name"]
    _ok(agent_client, "smesh", "mesh_tetra", mesh=m, max_size=5)
    _ok(agent_client, "smesh", "group_on_geometry", mesh=m, group="cap")
    assert "cap" in agent_client.forget(["cap"])["removed"]
    r = _ok(agent_client, "smesh", "compute", mesh=m)
    assert "cap" not in r["groups"]
    g = _call(agent_client, "smesh", "groups_from_geometry", mesh=m)
    assert g["ok"] is False


def test_a_mesh_on_a_shaper_group_measures_the_group(agent_client):
    b = shaper_box(agent_client)
    top_face_group(agent_client, b, name="roof")
    m = _ok(agent_client, "smesh", "create_mesh", shape="roof")["name"]
    _ok(agent_client, "smesh", "mesh_surface", mesh=m, max_size=2)
    r = _ok(agent_client, "smesh", "compute", mesh=m)
    assert r["shape_area"] == pytest.approx(200) and r["shape_volume"] == 0
    assert r["area"] == pytest.approx(200, rel=1e-6)
    info = _ok(agent_client, "smesh", "shape_info", names=[b, "roof"])["shapes"]
    assert info[b] == {"type": "solid", "area": pytest.approx(2200),
                       "volume": pytest.approx(6000), "faces": 6}
    assert info["roof"]["type"] == "face" and info["roof"]["area"] == pytest.approx(200)
