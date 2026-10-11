"""Live checks of the meshers, hypotheses and tools taken from the StdMeshers,
NETGEN and GMSH builders (maillages/)."""
import json
import subprocess

import pytest

from ._salome_root import SALOME_ROOT
from .conftest import _salome_subprocess_env
from .test_smesh_tools_live import _call, _ok, shaper_box, top_face_group

pytestmark = [pytest.mark.salome,
              pytest.mark.skipif(SALOME_ROOT is None,
                                 reason="set SALOME_ROOT or salome.root in config.json")]


def _mesh(client, shape):
    return _ok(client, "smesh", "create_mesh", shape=shape)["name"]


def _face(client, shape, axis, take, name):
    faces = _ok(client, "smesh", "get_subshapes", shape=shape, sub_type="face",
                where={"sorted_by": axis, "take": take, "count": 1})["subshapes"]
    return _ok(client, "smesh", "name_subshapes", shape=shape, ids=[faces[0]["id"]],
               name=name)


def test_gmsh_tetra_and_surface(agent_client):
    b = shaper_box(agent_client)
    m = _mesh(agent_client, b)
    _ok(agent_client, "smesh", "mesh_tetra", mesh=m, max_size=5, mesher="gmsh")
    # GMSH's Evaluate() predicts nothing: compute runs without the limit and says so
    assert _ok(agent_client, "smesh", "evaluate", mesh=m)["predicted_elements"] is None
    r = _ok(agent_client, "smesh", "compute", mesh=m)
    assert r["computed"] and r["errors"] == [] and r["tetra"] > 0
    assert r["volume"] == pytest.approx(6000, rel=1e-6) and "without the" in r["note"]
    s = _mesh(agent_client, b)
    _ok(agent_client, "smesh", "mesh_surface", mesh=s, max_size=3, mesher="gmsh", quads=True)
    r = _ok(agent_client, "smesh", "compute", mesh=s)
    assert r["computed"] and r["quadrangles"] > r["triangles"]
    r = _call(agent_client, "smesh", "mesh_tetra", mesh=m, max_size=5, mesher="gmsh",
              fineness="fine")
    assert r["ok"] is False and "NETGEN options" in r["error"]


def test_gmsh_local_size_goes_on_edges(agent_client):
    b = shaper_box(agent_client)
    top_face_group(agent_client, b, name="fine_top")
    m = _mesh(agent_client, b)
    _ok(agent_client, "smesh", "mesh_tetra", mesh=m, max_size=5, mesher="gmsh")
    _ok(agent_client, "smesh", "group_on_geometry", mesh=m, group="fine_top")
    coarse = _ok(agent_client, "smesh", "compute", mesh=m)["groups"]["fine_top"]
    _ok(agent_client, "smesh", "local_size", mesh=m, on=["fine_top"], size=1)
    r = _ok(agent_client, "smesh", "compute", mesh=m)
    assert r["errors"] == [] and r["groups"]["fine_top"] > 3 * coarse


def test_new_1d_hypotheses_build_on_a_hexa_box(agent_client):
    b = shaper_box(agent_client)
    _ok(agent_client, "smesh", "get_subshapes", shape=b, sub_type="edge", where="all")
    for kind, params in [("max_size", {"length": 3}),
                         ("adaptive", {"min": 0.5, "max": 4, "deflection": 0.1}),
                         ("deflection", {"deflection": 0.1}),
                         ("automatic_length", {"fineness": 0.5}),
                         ("fixed_points", {"points": [0.25, 0.5], "segments": [2, 3, 4]}),
                         ("arithmetic_progression", {"start": 1, "end": 3})]:
        m = _mesh(agent_client, b)
        _ok(agent_client, "smesh", "set_algorithm", mesh=m, dimension=1, algorithm="segment")
        _ok(agent_client, "smesh", "set_algorithm", mesh=m, dimension=2, algorithm="quadrangle")
        _ok(agent_client, "smesh", "set_algorithm", mesh=m, dimension=3, algorithm="hexahedron")
        _ok(agent_client, "smesh", "set_hypothesis", mesh=m, kind=kind, params=params)
        r = _ok(agent_client, "smesh", "compute", mesh=m)
        assert r["computed"] and r["hexa"] > 0, (kind, r["errors"])
        assert r["volume"] == pytest.approx(6000, rel=1e-6)


def test_reversed_edges_grade_the_other_way(agent_client):
    b = shaper_box(agent_client)
    edges = _ok(agent_client, "smesh", "get_subshapes", shape=b, sub_type="edge",
                where={"sorted_by": "z", "take": "max", "count": 4})["subshapes"]
    _ok(agent_client, "smesh", "name_subshapes", shape=b, ids=[edges[0]["id"]], name="flip")
    m = _mesh(agent_client, b)
    _ok(agent_client, "smesh", "set_algorithm", mesh=m, dimension=1, algorithm="segment")
    _ok(agent_client, "smesh", "set_hypothesis", mesh=m, kind="geometric_progression",
        params={"start": 0.5, "ratio": 1.3, "reversed": ["flip"]})
    _ok(agent_client, "smesh", "set_algorithm", mesh=m, dimension=2, algorithm="netgen_2d")
    r = _ok(agent_client, "smesh", "compute", mesh=m)
    assert r["computed"] and r["errors"] == []


def test_netgen_split_setup_with_parameters(agent_client):
    b = shaper_box(agent_client)
    m = _mesh(agent_client, b)
    _ok(agent_client, "smesh", "set_algorithm", mesh=m, dimension=2, algorithm="gmsh_2d")
    _ok(agent_client, "smesh", "set_hypothesis", mesh=m, kind="gmsh_parameters",
        params={"max_size": 4, "algorithm_2d": "frontal"})
    _ok(agent_client, "smesh", "set_algorithm", mesh=m, dimension=3, algorithm="netgen_3d")
    _ok(agent_client, "smesh", "set_hypothesis", mesh=m, kind="netgen_parameters",
        params={"max_size": 4, "optimize": True})
    r = _ok(agent_client, "smesh", "compute", mesh=m)
    assert r["computed"] and r["tetra"] > 0 and r["volume"] == pytest.approx(6000, rel=1e-6)
    r = _call(agent_client, "smesh", "set_hypothesis", mesh=m, kind="number_of_layers",
              params={"count": 3})
    assert r["ok"] is False and "does not apply to the NETGEN_3D" in r["error"]
    _ok(agent_client, "smesh", "remove_hypothesis", mesh=m, kind="netgen_parameters")
    _ok(agent_client, "smesh", "clear_mesh", mesh=m)
    assert _ok(agent_client, "smesh", "compute", mesh=m)["tetra"] > 0


def test_remove_hypothesis_removes_renamed_kinds(agent_client):
    b = shaper_box(agent_client)
    m = _mesh(agent_client, b)
    _ok(agent_client, "smesh", "set_algorithm", mesh=m, dimension=1, algorithm="segment")
    _ok(agent_client, "smesh", "set_hypothesis", mesh=m, kind="max_size", params={"length": 3})
    _ok(agent_client, "smesh", "set_algorithm", mesh=m, dimension=2, algorithm="quadrangle")
    _ok(agent_client, "smesh", "set_hypothesis", mesh=m, kind="quadrangle_parameters",
        params={"type": "reduced"})
    for kind in ("max_size", "quadrangle_parameters"):
        r = _ok(agent_client, "smesh", "remove_hypothesis", mesh=m, kind=kind)
    left = _ok(agent_client, "smesh", "set_hypothesis", mesh=m, kind="local_length",
               params={"length": 5})
    r = _ok(agent_client, "smesh", "compute", mesh=m)
    assert r["computed"] and r["errors"] == [], left


def test_radial_quadrangle_on_a_disk(agent_client):
    d = _ok(agent_client, "geom", "make_disk", radius=10)["result_shape"]
    m = _mesh(agent_client, d)
    _ok(agent_client, "smesh", "set_algorithm", mesh=m, dimension=2,
        algorithm="radial_quadrangle")
    _ok(agent_client, "smesh", "set_hypothesis", mesh=m, kind="number_of_layers",
        params={"count": 5})
    r = _ok(agent_client, "smesh", "compute", mesh=m)
    assert r["computed"] and r["quadrangles"] > 0 and r["errors"] == []


def test_radial_prism_between_shells_with_projection(agent_client):
    outer = _ok(agent_client, "geom", "make_sphere", radius=10)["result_shape"]
    inner = _ok(agent_client, "geom", "make_sphere", radius=5)["result_shape"]
    shell = _ok(agent_client, "geom", "cut", main=outer, tools=[inner])["result_shape"]
    faces = _ok(agent_client, "smesh", "get_subshapes", shape=shell, sub_type="face",
                where={"by_size": "smallest", "count": 2})["subshapes"]
    small, big = sorted(faces, key=lambda f: f["size"])
    _ok(agent_client, "smesh", "name_subshapes", shape=shell, ids=[small["id"]], name="core")
    _ok(agent_client, "smesh", "name_subshapes", shape=shell, ids=[big["id"]], name="skin")
    m = _mesh(agent_client, shell)
    _ok(agent_client, "smesh", "set_algorithm", mesh=m, dimension=2, algorithm="netgen_1d2d")
    _ok(agent_client, "smesh", "set_hypothesis", mesh=m, kind="netgen_parameters",
        params={"max_size": 2})
    _ok(agent_client, "smesh", "set_algorithm", mesh=m, dimension=3, algorithm="radial_prism")
    _ok(agent_client, "smesh", "set_hypothesis", mesh=m, kind="number_of_layers",
        params={"count": 4})
    _ok(agent_client, "smesh", "project_mesh", mesh=m, source="skin", on=["core"])
    assert _ok(agent_client, "smesh", "evaluate", mesh=m)["predicted_elements"] > 0
    r = _ok(agent_client, "smesh", "compute", mesh=m)
    assert r["computed"] and r["errors"] == [] and r["prism"] > 0


def test_project_mesh_makes_opposite_faces_match(agent_client):
    b = shaper_box(agent_client)
    _face(agent_client, b, "z", "min", "floor")
    _face(agent_client, b, "z", "max", "roof")
    m = _mesh(agent_client, b)
    _ok(agent_client, "smesh", "mesh_tetra", mesh=m, max_size=4)
    _ok(agent_client, "smesh", "local_size", mesh=m, on=["floor"], size=1.5)
    r = _ok(agent_client, "smesh", "project_mesh", mesh=m, source="floor", on=["roof"])
    assert "RemoveHypothesis" in r["records"][0]["code"]
    _ok(agent_client, "smesh", "group_on_geometry", mesh=m, group="floor")
    _ok(agent_client, "smesh", "group_on_geometry", mesh=m, group="roof")
    r = _ok(agent_client, "smesh", "compute", mesh=m)
    assert r["computed"] and r["errors"] == []
    assert r["groups"]["floor"] == r["groups"]["roof"] > 50
    assert r["volume"] == pytest.approx(6000, rel=1e-6)
    r = _call(agent_client, "smesh", "project_mesh", mesh=m, source="floor", on=["floor"])
    assert r["ok"] is False and "onto itself" in r["error"]


def test_viscous_layers_keep_the_local_sizes(agent_client):
    b = shaper_box(agent_client)
    _face(agent_client, b, "z", "min", "wall")
    top_face_group(agent_client, b, name="opening")
    m = _mesh(agent_client, b)
    _ok(agent_client, "smesh", "mesh_tetra", mesh=m, max_size=5)
    _ok(agent_client, "smesh", "local_size", mesh=m, on=["wall"], size=1)
    _ok(agent_client, "smesh", "group_on_geometry", mesh=m, group="wall")
    fine = _ok(agent_client, "smesh", "compute", mesh=m)["groups"]["wall"]
    r = _ok(agent_client, "smesh", "viscous_layers", mesh=m, thickness=0.5, layers=2,
            stretch=1.2, faces=["opening"], faces_are="without_layers")
    assert "SetLocalSizeOnShape" in r["records"][0]["code"]
    r = _ok(agent_client, "smesh", "compute", mesh=m)
    assert r["computed"] and r["prism"] > 0
    assert r["groups"]["wall"] > fine / 2


def test_combine_groups_follow_a_recompute(agent_client):
    b = shaper_box(agent_client)
    faces = _ok(agent_client, "smesh", "get_subshapes", shape=b, sub_type="face",
                where="all")["subshapes"]
    _ok(agent_client, "smesh", "name_subshapes", shape=b, ids=[f["id"] for f in faces],
        name="skin")
    _face(agent_client, b, "z", "max", "inlet")
    _face(agent_client, b, "z", "min", "outlet")
    m = _mesh(agent_client, b)
    _ok(agent_client, "smesh", "mesh_tetra", mesh=m, max_size=5)
    for g in ("skin", "inlet", "outlet"):
        _ok(agent_client, "smesh", "group_on_geometry", mesh=m, group=g)
    _ok(agent_client, "smesh", "combine_groups", mesh=m, name="wall", operation="cut",
        groups=["skin"], minus=["inlet", "outlet"])
    _ok(agent_client, "smesh", "combine_groups", mesh=m, name="ends", operation="union",
        groups=["inlet", "outlet"])
    r = _ok(agent_client, "smesh", "compute", mesh=m)
    g = r["groups"]
    assert g["wall"] + g["inlet"] + g["outlet"] == g["skin"] and g["wall"] > 0
    assert g["ends"] == g["inlet"] + g["outlet"]
    _ok(agent_client, "smesh", "mesh_tetra", mesh=m, max_size=3)
    g2 = _ok(agent_client, "smesh", "compute", mesh=m)["groups"]
    assert g2["skin"] > g["skin"] and g2["wall"] + g2["inlet"] + g2["outlet"] == g2["skin"]
    assert "outlet" in agent_client.forget(["outlet"])["removed"]
    left = _ok(agent_client, "smesh", "mesh_info", mesh=m)["groups"]
    assert set(left) == {"skin", "inlet"}
    assert agent_client.check_results([m], module="smesh")["results"][m] == {"status": "ok"}


def test_split_elements_and_check_topology(agent_client):
    b = shaper_box(agent_client)
    m = _mesh(agent_client, b)
    _ok(agent_client, "smesh", "mesh_hexa", mesh=m, segments=3)
    r = _call(agent_client, "smesh", "split_elements", mesh=m, into="tetrahedra")
    assert r["ok"] is False and "not computed" in r["error"]
    _ok(agent_client, "smesh", "compute", mesh=m)
    r = _ok(agent_client, "smesh", "split_elements", mesh=m, into="tetrahedra")
    assert r["hexa"] == 0 and r["tetra"] > 0 and r["quadrangles"] == 0
    t = _ok(agent_client, "smesh", "check_topology", mesh=m)
    assert t["coincident_nodes"] == 0 and t["bare_border_volumes"] == 0
    info = _ok(agent_client, "smesh", "mesh_info", mesh=m)
    assert info["volume"] == pytest.approx(6000, rel=1e-6)
    r = _call(agent_client, "smesh", "split_elements", mesh=m, into="triangles")
    assert r["ok"] is False and "nothing to split" in r["error"]
    q = _ok(agent_client, "smesh", "check_quality", mesh=m, criterion="max_element_length")
    assert q["max"] > 0


def test_check_topology_finds_the_open_border_of_a_surface(agent_client):
    d = _ok(agent_client, "geom", "make_disk", radius=10)["result_shape"]
    m = _mesh(agent_client, d)
    _ok(agent_client, "smesh", "mesh_surface", mesh=m, max_size=2)
    _ok(agent_client, "smesh", "compute", mesh=m)
    t = _ok(agent_client, "smesh", "check_topology", mesh=m)
    assert t["faces_with_free_edges"] > 0 and t["coincident_nodes"] == 0


def test_prism_with_quads_sweeps_hexahedra(agent_client):
    cyl = _ok(agent_client, "shaper", "make_cylinder", radius=5, height=10,
              origin=[0, 0, 0])["name"]
    _face(agent_client, cyl, "z", "min", "base")
    m = _mesh(agent_client, cyl)
    _ok(agent_client, "smesh", "mesh_prism", mesh=m, end_face="base", layers=5,
        face_size=1, quads=True)
    r = _ok(agent_client, "smesh", "compute", mesh=m)
    assert r["computed"] and r["errors"] == [] and r["hexa"] > r["prism"]


def test_export_gmf_and_dat(agent_client, tmp_path):
    import os
    b = shaper_box(agent_client)
    m = _mesh(agent_client, b)
    _ok(agent_client, "smesh", "mesh_tetra", mesh=m, max_size=5)
    _ok(agent_client, "smesh", "compute", mesh=m)
    for fmt, ext in (("gmf", "mesh"), ("dat", "dat")):
        r = _ok(agent_client, "smesh", "export_mesh", mesh=m, format=fmt)
        assert r["path"].endswith(f"{m}.{ext}") and os.path.getsize(r["path"]) > 0
        assert r["records"][0]["code"].endswith(f"{m}.{ext}')")
        os.unlink(r["path"])


def test_result_py_replays_the_new_tools(agent_client, tmp_path):
    from salome_agent.build_graph import BuildGraph
    from salome_agent.nodes.assembler import render_script

    calls = {"shaper": [], "smesh": []}

    def rec(module, tool, **args):
        r = _ok(agent_client, module, tool, **args)
        calls[module].append({"tool": tool, "tool_args": args, "records": r.get("records", [])})
        return r

    box = rec("shaper", "make_box", dx=10, dy=20, dz=30, center=[5, 10, 15])["name"]
    for axis_take, name in ((("z", "min"), "floor"), (("z", "max"), "roof")):
        f = _ok(agent_client, "smesh", "get_subshapes", shape=box, sub_type="face",
                where={"sorted_by": axis_take[0], "take": axis_take[1], "count": 1})
        rec("smesh", "name_subshapes", shape=box, ids=[f["subshapes"][0]["id"]], name=name)
    m = rec("smesh", "create_mesh", shape=box)["name"]
    rec("smesh", "mesh_tetra", mesh=m, max_size=4)
    rec("smesh", "local_size", mesh=m, on=["floor"], size=1.5)
    rec("smesh", "project_mesh", mesh=m, source="floor", on=["roof"])
    rec("smesh", "group_on_geometry", mesh=m, group="floor")
    rec("smesh", "group_on_geometry", mesh=m, group="roof")
    rec("smesh", "combine_groups", mesh=m, name="ends", operation="union",
        groups=["floor", "roof"])
    live = rec("smesh", "compute", mesh=m)

    g = BuildGraph()
    for i, (module, names) in enumerate((("shaper", [box]), ("smesh", [m]))):
        bid = g.add_block(0, f"stage {i + 1}/2: {module}", module, names)
        for c in calls[module]:
            g.append(bid, c["tool"], c["tool_args"], c["records"], 0, module=module)
    script = render_script(g, "replay")
    script += (f"\nimport json\nprint('COUNTS', json.dumps([{m}.NbTetras(), "
               f"{m}_floor.Size(), {m}_roof.Size(), {m}_ends.Size()]))\n")
    path = tmp_path / "result.py"
    path.write_text(script)
    out = subprocess.run([f"{SALOME_ROOT}/salome", "shell", "--", "python3", str(path)],
                         env=_salome_subprocess_env(), capture_output=True, text=True,
                         timeout=300)
    line = next((l for l in out.stdout.splitlines() if l.startswith("COUNTS")), None)
    assert line, out.stdout[-2000:] + out.stderr[-2000:]
    tetra, floor, roof, ends = json.loads(line.split(" ", 1)[1])
    assert tetra == live["tetra"]
    assert floor == roof == live["groups"]["floor"] and ends == live["groups"]["ends"]
