from unittest.mock import MagicMock

from salome_agent.build_graph import BuildGraph
from salome_agent.specialists.amend import AmendSession, render_transcript, run_amend


_SMESH_TOOLS = [("create_mesh", "setup"), ("mesh_tetra", "setup"), ("compute", "compute"),
                ("get_subshapes", "naming"), ("name_subshapes", "naming"),
                ("viscous_layers", "local")]


def _smesh_list():
    return {"tools": [{"name": n, "parameters": {}, "description": n, "category": c}
                      for n, c in _SMESH_TOOLS], "module": "smesh"}


class FakeClient:
    def __init__(self):
        self.calls = []

    def list_tools(self, module=None):
        if module == "smesh":
            return _smesh_list()
        return {"tools": [{"name": n, "parameters": {}, "description": n, "category": c}
                          for n, c in [("make_box", "primitive"), ("measure", "query"),
                                       ("fuse", "boolean"), ("translate", "transform"),
                                       ("align", "transform"), ("cut", "boolean"),
                                       ("check_shape", "query"), ("get_subshapes", "query"),
                                       ("list_study_objects", "query")]]}

    def forget(self, names):
        self.calls.append(("forget", names))
        return {"ok": True, "removed": []}

    def call_tool(self, name, args, pin=None):
        self.calls.append((name, args, pin))
        if name == "measure":
            return {"ok": True, "measurements": {n: {"bbox": [0] * 6, "size": [1, 1, 1],
                                                     "centre": [0, 0, 0]} for n in args["names"]}}
        out = (pin or [f"{name}_new"])[0]
        return {"ok": True, "records": [{"fn": "F", "args": [], "out_vars": [out],
                                         "publish": [out]}]}


def _graph():
    g = BuildGraph()
    g.requests.append("a box")
    b = g.add_block(0, "stage 1/1: geom", "geom", ["box_1"])
    g.append(b, "make_box", {"dx": 1, "dy": 1, "dz": 1},
             [{"fn": "F", "args": [], "out_vars": ["box_1"], "publish": ["box_1"]}], 0)
    return g


def _resp(*calls):
    return {"tool_calls": [{"name": n, "args": a, "id": None} for n, a in calls],
            "content": None, "raw_message": None}


POLICY = {"tool_iteration_cap": 5, "tool_selection": "hints", "verify_cadence": "none"}


def test_edit_arg_replays_and_records_events():
    llm = MagicMock()
    llm.chat.side_effect = [_resp(("edit_arg", {"node": "n1", "param": "dz", "value": "3"})),
                            _resp(("finish_turn", {"summary": "taller"}))]
    client = FakeClient()
    session, outcome = run_amend("make it taller", _graph(), llm, client, 1, POLICY)
    assert outcome == {"summary": "taller", "question": None}
    assert session.graph.nodes["n1"].tool_args["dz"] == 3
    assert ("make_box", {"dx": 1, "dy": 1, "dz": 3}, ["box_1"]) in client.calls
    assert session.events[0] == {"event": "graph_edit", "op": "edit", "node": "n1",
                                 "param": "dz", "old": 1, "new": 3}
    assert session.events[1] == {"event": "replay", "nodes": ["n1"]}


def test_prompt_carries_graph_snapshot_turn_text_and_amend_tools():
    llm = MagicMock()
    llm.chat.side_effect = [_resp(("finish_turn", {"summary": "nothing"}))]
    run_amend("make it red", _graph(), llm, FakeClient(), 1, POLICY,
              history=[{"event": "turn", "turn": 0, "text": "a box", "status": "ok",
                        "summary": "built"}])
    kw = llm.chat.call_args.kwargs
    user = kw["messages"][1]["content"]
    assert "n1  box_1 = make_box(dx=1, dy=1, dz=1)" in user
    assert "box_1 — bbox" in user
    assert "turn 1 (ok): a box -> built" in user
    assert "make it red" in user
    names = {t["name"] for t in kw["tools"]}
    assert {"edit_arg", "delete_node", "finish_turn", "replan", "load_tools"} <= names
    assert kw["module"] == "amend"


def test_selection_block_renders_only_when_given():
    llm = MagicMock()
    llm.chat.side_effect = [_resp(("finish_turn", {"summary": "x"}))] * 2
    run_amend("t", _graph(), llm, FakeClient(), 1, POLICY)
    assert "currently has these selected" not in llm.chat.call_args.kwargs["messages"][1]["content"]
    run_amend("t", _graph(), llm, FakeClient(), 1, POLICY,
              selection=[{"object": "box_1", "subshape": {"type": "FACE", "index": 3}}])
    assert "currently has these selected" in llm.chat.call_args.kwargs["messages"][1]["content"]


def test_image_description_block_renders_only_when_given():
    llm = MagicMock()
    llm.chat.side_effect = [_resp(("finish_turn", {"summary": "x"}))] * 2
    run_amend("t", _graph(), llm, FakeClient(), 1, POLICY)
    assert "reference image" not in llm.chat.call_args.kwargs["messages"][1]["content"]
    run_amend("add this handle", _graph(), llm, FakeClient(), 1, POLICY,
              image_description="a D-shaped handle")
    user = llm.chat.call_args.kwargs["messages"][1]["content"]
    assert "reference image" in user and "a D-shaped handle" in user
    assert user.index("a D-shaped handle") < user.index("The user says: add this handle")


def test_append_goes_into_one_amend_block_for_the_turn():
    llm = MagicMock()
    llm.chat.side_effect = [_resp(("make_box", {"dx": 2, "dy": 2, "dz": 2})),
                            _resp(("make_box", {"dx": 3, "dy": 3, "dz": 3})),
                            _resp(("finish_turn", {"summary": "added"}))]
    session, _ = run_amend("add two", _graph(), llm, FakeClient(), 1, POLICY)
    labels = [b.label for b in session.graph.blocks]
    assert labels == ["stage 1/1: geom", "turn 2: amend"]
    assert [n.block for n in session.graph.nodes.values()] == ["b1", "b2", "b2"]


def test_bad_node_is_a_tool_error_with_the_remedy():
    session = AmendSession(_graph(), FakeClient(), 1)
    r = session.execute(None, "edit_arg", {"node": "n7", "param": "dz", "value": "1"})
    assert r["ok"] is False and "current nodes are n1" in r["error"]


def test_replan_and_question_outcomes():
    llm = MagicMock()
    llm.chat.side_effect = [_resp(("replan", {"reason": "whole new wing"}))]
    _, outcome = run_amend("add a wing", _graph(), llm, FakeClient(), 1, POLICY)
    assert outcome == {"replan": "whole new wing"}
    llm.chat.side_effect = [_resp(("finish_turn", {"summary": "", "question": "which box?"}))]
    _, outcome = run_amend("move the box", _graph(), llm, FakeClient(), 1, POLICY)
    assert outcome["question"] == "which box?"


def test_render_transcript_keeps_the_last_ten_turns():
    hist = [{"event": "turn", "turn": i, "text": f"t{i}", "status": "ok", "summary": "s"}
            for i in range(15)] + [{"event": "retry_hint", "hint": "x"}]
    text = render_transcript(hist)
    assert "t4" not in text and "t5" in text and "t14" in text


def test_amend_prompt_only_deletes_on_request_and_adds_new_objects_alongside():
    from salome_agent.specialists.amend import AMEND_SYSTEM_PROMPT as p
    assert "Never call delete_node unless the user asked to remove or replace" in p
    assert "added alongside" in p
    assert "add or replace" in p


class ShaperClient(FakeClient):
    def list_tools(self, module=None):
        if module == "smesh":
            return _smesh_list()
        assert module == "shaper"
        return {"tools": [{"name": "extrude", "parameters": {}, "description": "e",
                           "category": "swept"}], "module": "shaper"}

    def call_tool(self, name, args, pin=None, module=None):
        self.calls.append((name, args, pin, module))
        if name == "measure":
            return {"ok": True, "measurements": {n: {"bbox": [0] * 6, "area": 1, "faces": 1}
                                                 for n in args["names"]}}
        out = (pin or [f"{name}_new"])[0]
        return {"ok": True, "records": [{"fn": "F", "args": [], "out_vars": [out],
                                         "publish": [out]}]}


def _shaper_graph():
    g = BuildGraph()
    b = g.add_block(0, "stage 1/1: shaper", "shaper", ["extrusion_1"])
    g.append(b, "extrude", {"base": "face_1", "size": 5},
             [{"fn": "F", "args": [], "out_vars": ["extrusion_1"],
               "publish": ["extrusion_1"]}], 0, module="shaper")
    return g


def test_a_shaper_amend_builds_measures_and_replays_in_shaper():
    client = ShaperClient()
    session = AmendSession(_shaper_graph(), client, 1, "shaper")
    r = session.execute(client, "extrude", {"base": "face_2", "size": 3})
    node = session.graph.nodes[r["node"]]
    assert node.module == "shaper" and client.calls[-1][3] == "shaper"
    assert session.graph.blocks[-1].module == "shaper"
    r = session.execute(client, "edit_arg", {"node": "n1", "param": "size", "value": "8"})
    assert r["ok"] and r["measurements"]["extrusion_1"]["faces"] == 1
    replayed = [c for c in client.calls if c[0] == "extrude" and c[2] == ["extrusion_1"]]
    assert replayed and replayed[0][3] == "shaper"


def test_amend_prompt_names_the_modules_tools():
    from salome_agent.specialists.amend import _prompt
    geom = _prompt("geom", "final")
    assert "call the geom tools as usual" in geom and "SHAPER" not in geom
    shaper = _prompt("shaper", "final")
    assert "call the shaper tools as usual" in shaper
    assert "finish_turn ends it" in shaper and "create_sketch" in shaper


def test_a_shaper_amend_offers_shaper_tools_and_the_snapshot_reads_shaper():
    from salome_agent.specialists.amend import _snapshot, _spec
    client = ShaperClient()
    spec = _spec(AmendSession(_shaper_graph(), client, 1, "shaper"))
    defs = spec.tools_for(client)
    assert set(defs) == {"extrude", "edit_arg", "delete_node"} | {n for n, _ in _SMESH_TOOLS}
    assert "extrusion_1 — bbox [0, 0, 0, 0, 0, 0], area 1, 1 face" in _snapshot(
        client, _shaper_graph())


def _meshed_graph():
    g = _shaper_graph()
    b = g.add_block(0, "stage 2/2: smesh", "smesh", [])
    g.append(b, "create_mesh", {"shape": "extrusion_1"},
             [{"fn": "smesh.Mesh", "args": [{"kind": "ref", "value": "extrusion_1"}],
               "out_vars": ["mesh_1"], "publish": ["mesh_1"]}], 0, module="smesh")
    g.append(b, "compute", {"mesh": "mesh_1"},
             [{"fn": "Compute", "args": [], "recv": "mesh_1", "out_vars": []}], 0,
             module="smesh")
    return g


def test_amend_adds_smesh_tools_routed_to_smesh_and_cad_keeps_the_shared_name():
    from salome_agent.specialists.amend import _spec

    class Both(ShaperClient):
        def list_tools(self, module=None):
            if module == "shaper":
                return {"tools": [{"name": n, "parameters": {}, "description": n,
                                   "category": c} for n, c in
                                  [("extrude", "swept"), ("get_subshapes", "query")]],
                        "module": "shaper"}
            return super().list_tools(module)

    client = Both()
    session = AmendSession(_shaper_graph(), client, 1, "shaper")
    spec = _spec(session)
    defs = spec.tools_for(client)
    assert session.tool_module["get_subshapes"] == "shaper"
    assert defs["get_subshapes"]["category"] == "query"
    assert session.tool_module["create_mesh"] == "smesh"
    assert defs["create_mesh"]["category"] == "smesh"
    assert defs["viscous_layers"]["category"] == "smesh_local"
    assert "smesh" in spec.categories
    session.execute(client, "create_mesh", {"shape": "extrusion_1"})
    assert client.calls[-1][0] == "create_mesh" and client.calls[-1][3] == "smesh"


def test_smesh_tools_are_offered_up_front_only_on_a_meshed_model():
    from salome_agent.specialists.amend import _spec
    client = ShaperClient()
    plain = _spec(AmendSession(_shaper_graph(), client, 1, "shaper"))
    assert "create_mesh" not in plain.core_tools
    assert "How the SMESH tools work" not in plain.prompt("final")
    assert "load_tools with the category smesh" in plain.prompt("final")
    meshed = _spec(AmendSession(_meshed_graph(), client, 1, "shaper"))
    assert {"create_mesh", "compute"} <= meshed.core_tools
    assert "How the SMESH tools work" in meshed.prompt("final")


def test_loading_smesh_returns_the_howto_once():
    llm = MagicMock()
    llm.chat.side_effect = [_resp(("load_tools", {"categories": ["smesh"]})),
                            _resp(("load_tools", {"names": ["viscous_layers"]})),
                            _resp(("finish_turn", {"summary": "nothing"}))]
    run_amend("mesh it", _shaper_graph(), llm, ShaperClient(), 1, POLICY | {"verify_cadence": "none"},
              session=AmendSession(_shaper_graph(), ShaperClient(), 1, "shaper"))
    replies = [m["content"] for m in llm.chat.call_args_list[-1].kwargs["messages"]
               if m["role"] == "tool"]
    assert "create_mesh" in replies[0] and "How the SMESH tools work" in replies[0]
    assert "viscous_layers" in replies[1] and "How the SMESH tools work" not in replies[1]
    tools = {t["name"] for t in llm.chat.call_args_list[-1].kwargs["tools"]}
    assert {"create_mesh", "mesh_tetra", "compute", "viscous_layers"} <= tools


def test_a_turns_cad_block_goes_before_its_smesh_block():
    client = ShaperClient()
    session = AmendSession(_meshed_graph(), client, 1, "shaper")
    session.tool_module = {"create_mesh": "smesh", "extrude": "shaper"}
    session.execute(client, "create_mesh", {"shape": "extrusion_1"})
    session.execute(client, "extrude", {"base": "face_2", "size": 3})
    assert [b.label for b in session.graph.blocks] == [
        "stage 1/1: shaper", "stage 2/2: smesh", "turn 2: amend", "turn 2: amend (SMESH)"]
    assert [b.module for b in session.graph.blocks][2:] == ["shaper", "smesh"]
    assert session.graph.blocks[3].result_names == ["create_mesh_new"]


class SmeshRecords(ShaperClient):
    """SMESH tools publish nothing, like the real ones."""

    def call_tool(self, name, args, pin=None, module=None):
        if module != "smesh":
            return super().call_tool(name, args, pin, module)
        self.calls.append((name, args, pin, module))
        rec = {"fn": "Mesh", "emit": "stmt", "publish": None,
               "args": [{"kind": "expr", "value": "", "refs": [args.get("shape", "")]}]}
        if name == "create_mesh":
            rec |= {"code": "mesh_9 = smesh.Mesh(x)", "out_vars": ["mesh_9"]}
        else:
            rec |= {"code": "mesh_9.Compute()", "out_vars": [], "recv": "mesh_9", "args": []}
        return {"ok": True, "records": [rec]}


def test_a_mesh_built_in_an_amend_turn_reaches_result_py():
    from salome_agent.nodes.assembler import render_script
    client = SmeshRecords()
    session = AmendSession(_shaper_graph(), client, 1, "shaper")
    session.tool_module = {"create_mesh": "smesh", "compute": "smesh"}
    session.graph.blocks[0].result_names = ["extrusion_1"]
    session.execute(client, "create_mesh", {"shape": "extrusion_1"})
    session.execute(client, "compute", {"mesh": "mesh_9"})
    script = render_script(session.graph, "r1")
    assert "# turn 2: amend (SMESH)" in script
    assert "mesh_9 = smesh.Mesh(x)" in script and "mesh_9.Compute()" in script


def test_a_delete_that_takes_a_mesh_with_it_says_so():
    client = ShaperClient()
    session = AmendSession(_meshed_graph(), client, 1, "shaper")
    r = session.execute(client, "delete_node", {"node": "n1"})
    assert r["deleted"] == ["n1", "n2", "n3"]
    assert r["note"] == ("This also removed the SMESH nodes n2, n3 (mesh_1), which were "
                         "built on what you deleted. Rebuild them on the new shape with the "
                         "SMESH tools unless the user asked to remove them.")
    session = AmendSession(_meshed_graph(), client, 1, "shaper")
    r = session.execute(client, "delete_node", {"node": "n3"})
    assert r == {"ok": True, "deleted": ["n3"]}


def test_a_compute_of_an_untouched_mesh_stays_out_of_the_graph():
    client = SmeshRecords()
    session = AmendSession(_shaper_graph(), client, 1, "shaper")
    session.tool_module = {"create_mesh": "smesh", "compute": "smesh", "local_size": "smesh"}
    session.execute(client, "create_mesh", {"shape": "extrusion_1"})
    session.execute(client, "compute", {"mesh": "mesh_9"})
    n = len(session.graph.nodes)
    r = session.execute(client, "compute", {"mesh": "mesh_9"})
    assert r["ok"] and client.calls[-1][0] == "compute"
    assert len(session.graph.nodes) == n
    session.execute(client, "local_size", {"mesh": "mesh_9"})
    session.execute(client, "compute", {"mesh": "mesh_9"})
    assert [x.tool for x in session.graph.nodes.values()][-2:] == ["local_size", "compute"]


def test_the_measure_nudge_names_the_amend_terminator():
    llm = MagicMock()
    llm.chat.side_effect = [_resp(("translate", {"shape": "box_1", "dx": 1})),
                            _resp(("finish_turn", {"summary": "moved"})),
                            _resp(("finish_turn", {"summary": "moved"}))]
    run_amend("move it", _graph(), llm, FakeClient(), 1, POLICY | {"verify_cadence": "final"})
    replies = [m["content"] for m in llm.chat.call_args_list[-1].kwargs["messages"]
               if m["role"] == "tool"]
    assert any("then call finish_turn again" in c for c in replies)
    assert not any("finish_stage" in c for c in replies)
