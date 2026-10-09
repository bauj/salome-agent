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
