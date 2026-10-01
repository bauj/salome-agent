import pytest

from salome_agent.build_graph import BuildGraph, GraphError


def _rec(fn, out, refs=(), publish=True, **extra):
    r = {"fn": fn, "args": [{"kind": "ref", "value": x} for x in refs]
         + [{"kind": "literal", "value": 1}],
         "out_vars": list(out), "publish": list(out) if publish else None}
    r.update(extra)
    return r


def _house():
    g = BuildGraph()
    b = g.add_block(0, "stage 1/1: geom", "geom", ["fuse_1"])
    g.append(b, "make_box", {"dx": 200, "dy": 120, "dz": 80}, [_rec("MakeBoxDXDYDZ", ["box_1"])], 0)
    g.append(b, "make_box", {"dx": 200, "dy": 120, "dz": 40},
             [_rec("MakeVectorDXDYDZ", ["vec_1"], publish=False),
              _rec("MakeTranslation", ["box_2"], refs=["vec_1"])], 0)
    g.append(b, "fuse", {"shapes": ["box_1", "box_2"]},
             [_rec("MakeFuse", ["fuse_1"], refs=["box_1", "box_2"])], 0)
    g.append(b, "set_color", {"shape": "fuse_1", "red": 255, "green": 0, "blue": 0},
             [{"fn": "SetColor", "args": [], "out_vars": [], "publish": None,
               "emit": "method_call", "recv": "fuse_1"}], 0)
    return g, b


def test_append_assigns_ids_and_deps_from_refs_and_recv():
    g, _ = _house()
    assert list(g.nodes) == ["n1", "n2", "n3", "n4"]
    assert g.nodes["n2"].deps == []          # vec_1 is its own record, not a dep
    assert g.nodes["n3"].deps == ["n1", "n2"]
    assert g.nodes["n4"].deps == ["n3"]      # method_call recv counts


def test_out_vars_and_published():
    g, _ = _house()
    assert g.nodes["n2"].out_vars == ["vec_1", "box_2"]
    assert g.nodes["n2"].published == ["box_2"]


def test_affected_is_forward_closure_in_id_order():
    g, _ = _house()
    assert g.affected("n1") == ["n1", "n3", "n4"]
    assert g.affected("n4") == ["n4"]


def test_edit_arg_replaces_a_literal():
    g, _ = _house()
    g.edit_arg("n1", "dz", 120)
    assert g.nodes["n1"].tool_args["dz"] == 120


def test_edit_arg_may_add_an_absent_literal_param():
    g, _ = _house()
    g.edit_arg("n1", "note", 3)
    assert g.nodes["n1"].tool_args["note"] == 3


def test_edit_arg_refuses_to_rewrite_a_ref():
    g, _ = _house()
    with pytest.raises(GraphError, match="delete_node"):
        g.edit_arg("n3", "shapes", ["box_1"])
    with pytest.raises(GraphError):
        g.edit_arg("n4", "shape", "box_1")


def test_edit_arg_refuses_a_literal_becoming_a_ref():
    g, _ = _house()
    with pytest.raises(GraphError):
        g.edit_arg("n1", "dz", "box_2")


def test_dead_node_error_names_the_live_ones():
    g, _ = _house()
    g.delete("n4")
    with pytest.raises(GraphError, match=r"n4 .*n1, n2, n3"):
        g.edit_arg("n4", "red", 0)


def test_delete_removes_downstream_and_prunes_result_names():
    g, b = _house()
    removed = g.delete("n2")
    assert [n.id for n in removed] == ["n2", "n3", "n4"]
    assert list(g.nodes) == ["n1"]
    assert g.blocks[0].result_names == []


def test_delete_drops_an_emptied_geom_block_but_keeps_script_blocks():
    g, b = _house()
    g.add_block(0, "stage 2/2: smesh", "smesh", [], script="mesh = 1")
    g.delete("n1"); g.delete("n2")
    assert [bl.module for bl in g.blocks] == ["smesh"]


def test_ids_are_never_reused():
    g, b = _house()
    g.delete("n4")
    assert g.append(b, "make_box", {"dx": 1, "dy": 1, "dz": 1},
                    [_rec("MakeBoxDXDYDZ", ["box_3"])], 1) == "n5"


def test_result_names_falls_back_to_published():
    g, b = _house()
    b2 = g.add_block(1, "turn 2: amend", "geom", [])
    g.append(b2, "make_box", {"dx": 1, "dy": 1, "dz": 1}, [_rec("MakeBoxDXDYDZ", ["box_3"])], 1)
    assert g.result_names() == ["fuse_1", "box_3"]


def test_set_records_recomputes_deps():
    g, _ = _house()
    g.set_records("n3", [_rec("MakeFuse", ["fuse_1"], refs=["box_1"])])
    assert g.nodes["n3"].deps == ["n1"]


def test_round_trip_and_copy_are_independent():
    g, _ = _house()
    g.requests.append("a house")
    g2 = BuildGraph.from_dict(g.to_dict())
    assert g2.to_dict() == g.to_dict()
    c = g.copy()
    c.edit_arg("n1", "dz", 1)
    assert g.nodes["n1"].tool_args["dz"] == 80
    assert BuildGraph.from_dict(None).to_dict() == BuildGraph().to_dict()


def test_render_is_tool_level():
    g, _ = _house()
    text = g.render()
    assert "# stage 1/1: geom" in text
    assert 'n1  box_1 = make_box(dx=200, dy=120, dz=80)' in text
    assert 'n3  fuse_1 = fuse(shapes=["box_1", "box_2"])  <- n1, n2' in text
    assert 'n4  set_color(shape="fuse_1", red=255, green=0, blue=0)  <- n3' in text
