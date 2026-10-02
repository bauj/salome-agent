from unittest.mock import MagicMock

from salome_agent.build_graph import BuildGraph
from salome_agent.specialists.amend import AmendSession, render_transcript, run_amend


class FakeClient:
    def __init__(self):
        self.calls = []

    def list_tools(self):
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
