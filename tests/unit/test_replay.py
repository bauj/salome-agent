import pytest

from salome_agent.build_graph import BuildGraph, GraphError
from salome_agent.replay import (ReplayFailed, RollbackFailed, apply_delete, apply_edit,
                                 rebuild_all, sync)


class FakeClient:
    """Builds boxes (dz is the only thing that matters) and fuses; `fail_on`
    makes a tool call with those exact args fail."""

    def __init__(self):
        self.study = {}
        self.log = []
        self.fail_on = []

    def forget(self, names):
        self.log.append(("forget", sorted(names)))
        for n in names:
            self.study.pop(n, None)
        return {"ok": True, "removed": []}

    def call_tool(self, name, args, pin=None):
        self.log.append(("call", name, dict(args), list(pin or [])))
        if (name, args) in self.fail_on:
            return {"ok": False, "error": "boom"}
        out = pin[0]
        self.study[out] = (name, dict(args))
        refs = args.get("shapes", [])
        return {"ok": True, "records": [{"fn": name, "args": [{"kind": "ref", "value": r}
                                                              for r in refs],
                                         "out_vars": [out], "publish": [out]}]}


def _graph(client):
    g = BuildGraph()
    b = g.add_block(0, "stage 1/1: geom", "geom", ["fuse_1"])
    for name, args, out in [("make_box", {"dz": 80}, "box_1"),
                            ("make_box", {"dz": 40}, "box_2"),
                            ("fuse", {"shapes": ["box_1", "box_2"]}, "fuse_1")]:
        r = client.call_tool(name, args, pin=[out])
        g.append(b, name, args, r["records"], 0)
    client.log.clear()
    return g


def test_apply_edit_forgets_then_rebuilds_the_closure_with_pinned_names():
    c = FakeClient()
    g = _graph(c)
    new, ids = apply_edit(c, g, "n2", "dz", 120)
    assert ids == ["n2", "n3"]
    assert c.log == [("forget", ["box_2", "fuse_1"]),
                     ("call", "make_box", {"dz": 120}, ["box_2"]),
                     ("call", "fuse", {"shapes": ["box_1", "box_2"]}, ["fuse_1"])]
    assert c.study["box_2"] == ("make_box", {"dz": 120})
    assert g.nodes["n2"].tool_args["dz"] == 40      # input graph untouched
    assert new.nodes["n2"].tool_args["dz"] == 120


def test_apply_edit_failure_restores_the_pre_edit_subtree():
    c = FakeClient()
    g = _graph(c)
    c.fail_on = [("make_box", {"dz": -1})]
    with pytest.raises(ReplayFailed) as e:
        apply_edit(c, g, "n2", "dz", -1)
    assert e.value.node_id == "n2"
    assert c.study["box_2"] == ("make_box", {"dz": 40})
    assert c.study["fuse_1"][0] == "fuse"


def test_apply_edit_rollback_failure_is_rollbackfailed():
    c = FakeClient()
    g = _graph(c)
    c.fail_on = [("make_box", {"dz": -1}), ("make_box", {"dz": 40})]
    with pytest.raises(RollbackFailed):
        apply_edit(c, g, "n2", "dz", -1)


def test_apply_edit_bad_node_is_grapherror_and_touches_nothing():
    c = FakeClient()
    g = _graph(c)
    with pytest.raises(GraphError):
        apply_edit(c, g, "n9", "dz", 1)
    assert c.log == []


def test_apply_delete_forgets_the_closure():
    c = FakeClient()
    g = _graph(c)
    new, removed = apply_delete(c, g, "n1")
    assert removed == ["n1", "n3"]
    assert c.log == [("forget", ["box_1", "fuse_1"])]
    assert list(new.nodes) == ["n2"]


def test_sync_reverts_an_edit_an_append_and_a_delete():
    c = FakeClient()
    before = _graph(c)
    after, _ = apply_edit(c, before, "n1", "dz", 5)
    after, _ = apply_delete(c, after, "n2")          # also removes n3
    b = after.blocks[0].id if after.blocks else after.add_block(1, "turn 2: amend", "geom", [])
    r = c.call_tool("make_box", {"dz": 9}, pin=["box_3"])
    after.append(b, "make_box", {"dz": 9}, r["records"], 1)
    restored = sync(c, current=after, target=before)
    assert set(c.study) == {"box_1", "box_2", "fuse_1"}
    assert c.study["box_1"] == ("make_box", {"dz": 80})
    assert restored.to_dict() == before.to_dict()


def test_rebuild_all_forgets_everything_and_replays_in_order():
    c = FakeClient()
    g = _graph(c)
    rebuild_all(c, g, extra_names=["stray_1"])
    assert c.log[0] == ("forget", ["box_1", "box_2", "fuse_1", "stray_1"])
    assert [e[1] for e in c.log[1:]] == ["make_box", "make_box", "fuse"]


def test_replay_sends_a_non_geom_node_to_its_module():
    from salome_agent.replay import replay
    seen = []

    class Client:
        def call_tool(self, name, args, pin=None, module=None):
            seen.append((name, module))
            return {"ok": True, "records": [{"fn": "f", "emit": "stmt", "code": "",
                                             "args": [], "out_vars": pin,
                                             "publish": pin}]}

    g = BuildGraph()
    b = g.add_block(0, "stage 1/1: shaper", "shaper", ["sketch_1"])
    nid = g.append(b, "create_sketch", {"plane": "XOY"},
                   [{"fn": "f", "emit": "stmt", "code": "", "args": [],
                     "out_vars": ["sketch_1"], "publish": ["sketch_1"]}], 0, module="shaper")
    replay(Client(), g, [nid])
    assert seen == [("create_sketch", "shaper")]


class SketchClient:
    """A sketch whose entities stay until the sketch itself is forgotten, and
    which refuses an id it already has, as SHAPER does."""

    def __init__(self):
        self.sketches = {}
        self.log = []

    def forget(self, names):
        self.log.append(("forget", sorted(names)))
        for n in names:
            self.sketches.pop(n, None)
        return {"ok": True, "removed": []}

    def call_tool(self, name, args, pin=None, module=None):
        self.log.append(("call", name))
        if name == "create_sketch":
            self.sketches[pin[0]] = []
            return {"ok": True, "records": [{"fn": name, "args": [], "recv": pin[0],
                                             "out_vars": [pin[0]], "publish": [pin[0]]}]}
        sk = self.sketches[args["sketch"]]
        if name == "close_sketch":
            return {"ok": True, "records": [{"fn": name, "args": [], "recv": args["sketch"],
                                             "out_vars": []}]}
        ids = [e["id"] for e in args["entities"]]
        if set(ids) & set(sk):
            return {"ok": False, "error": "id already used in this sketch"}
        sk += ids
        return {"ok": True, "records": [{"fn": name, "args": [], "recv": args["sketch"],
                                         "out_vars": [f"{args['sketch']}_{i}" for i in ids]}]}


def _sketch_graph(client):
    g = BuildGraph()
    b = g.add_block(0, "stage 1/1: shaper", "shaper", ["sketch_1"])
    for name, args in [("create_sketch", {"plane": "XOY"}),
                       ("sketch_add", {"sketch": "sketch_1", "entities": [{"id": "H1"}, {"id": "H2"}]}),
                       ("sketch_add", {"sketch": "sketch_1", "entities": [{"id": "L1"}]}),
                       ("close_sketch", {"sketch": "sketch_1"})]:
        r = client.call_tool(name, args, pin=["sketch_1"] if name == "create_sketch" else [])
        g.append(b, name, args, r["records"], 0, module="shaper")
    client.log.clear()
    return g


def test_apply_edit_of_a_sketch_entity_rebuilds_the_sketch():
    c = SketchClient()
    g = _sketch_graph(c)
    new, ids = apply_edit(c, g, "n2", "entities", [{"id": "H1"}])
    assert ids == ["n1", "n2", "n3", "n4"]
    assert c.log[0] == ("forget", ["sketch_1", "sketch_1_H1", "sketch_1_H2", "sketch_1_L1"])
    assert c.sketches["sketch_1"] == ["H1", "L1"]


def test_apply_delete_of_a_sketch_entity_rebuilds_the_sketch_without_it():
    c = SketchClient()
    g = _sketch_graph(c)
    new, removed = apply_delete(c, g, "n2")
    assert removed == ["n2"]
    assert list(new.nodes) == ["n1", "n3", "n4"]
    assert c.sketches["sketch_1"] == ["L1"]


def test_apply_delete_failure_restores_the_sketch():
    c = SketchClient()
    g = _sketch_graph(c)
    real = c.call_tool

    def failing(name, args, pin=None, module=None):
        if name == "close_sketch" and c.log.count(("call", "close_sketch")) == 0:
            c.log.append(("call", name))
            return {"ok": False, "error": "boom"}
        return real(name, args, pin, module)

    c.call_tool = failing
    with pytest.raises(ReplayFailed):
        apply_delete(c, g, "n2")
    assert c.sketches["sketch_1"] == ["H1", "H2", "L1"]


def test_sync_undoes_a_sketch_entity_edit():
    c = SketchClient()
    before = _sketch_graph(c)
    after, _ = apply_edit(c, before, "n2", "entities", [{"id": "H3"}])
    restored = sync(c, current=after, target=before)
    assert c.sketches["sketch_1"] == ["H1", "H2", "L1"]
    assert restored.to_dict() == before.to_dict()


def test_sync_undoes_a_sketch_entity_delete():
    c = SketchClient()
    before = _sketch_graph(c)
    after, _ = apply_delete(c, before, "n3")
    assert c.sketches["sketch_1"] == ["H1", "H2"]
    sync(c, current=after, target=before)
    assert c.sketches["sketch_1"] == ["H1", "H2", "L1"]
