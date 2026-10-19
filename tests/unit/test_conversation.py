from unittest.mock import MagicMock

import pytest

from salome_agent.build_graph import BuildGraph
from salome_agent import conversation as conv_mod
from salome_agent.conversation import Conversation

CONFIG = {"work_dir": None, "llm": {}, "policy": {
    "tool_iteration_cap": 5, "tool_selection": "hints", "verify_cadence": "none",
    "plan_detail": "balanced",
    "on_stage_failure": "abort"}}


def _graph_with(n_boxes):
    g = BuildGraph()
    g.requests.append("boxes")
    b = g.add_block(0, "stage 1/1: geom", "geom", [])
    for i in range(1, n_boxes + 1):
        g.append(b, "make_box", {"dz": i}, [{"fn": "F", "args": [], "out_vars": [f"box_{i}"],
                                            "publish": [f"box_{i}"]}], 0)
    return g


@pytest.fixture
def conv(tmp_path, monkeypatch):
    cfg = {**CONFIG, "work_dir": str(tmp_path)}
    c = Conversation(cfg, client=MagicMock(), run_id="r1", batch=True, llm=MagicMock(usage={}))
    return c


def _fake_drive(monkeypatch, final_update):
    def fake(app, inp, cfg, mapper, llm, on_event, on_gate, cancel):
        return {**inp, **final_update}
    monkeypatch.setattr(conv_mod, "drive", fake)


def test_ok_turn_commits_graph_and_snapshot(conv, monkeypatch):
    g = _graph_with(1)
    _fake_drive(monkeypatch, {"build_graph": g.to_dict(), "final_script": "x"})
    events = []
    r = conv.turn("one box", on_event=events.append, on_gate=None)
    assert r.status == "ok"
    assert conv.turn_idx == 1 and list(conv.graph.nodes) == ["n1"]
    assert [e["kind"] for e in events] == ["turn_start", "turn_done"]
    assert conv.history[-1]["event"] == "turn" and conv.history[-1]["status"] == "ok"


def test_turn_hands_its_images_to_the_graph(conv, monkeypatch):
    seen = []

    def fake(app, inp, cfg, mapper, llm, on_event, on_gate, cancel):
        seen.append(inp["reference_images"])
        return {**inp, "build_graph": _graph_with(1).to_dict()}

    monkeypatch.setattr(conv_mod, "drive", fake)
    conv.turn("like this", images=["data:image/png;base64,AAAA"], on_event=lambda e: None,
              on_gate=None)
    conv.turn("taller", on_event=lambda e: None, on_gate=None)
    assert seen == [["data:image/png;base64,AAAA"], None]


def test_failed_turn_rolls_back_via_sync(conv, monkeypatch):
    conv.graph = _graph_with(1); conv.turn_idx = 1
    bad = _graph_with(2)
    _fake_drive(monkeypatch, {"build_graph": bad.to_dict(), "fatal": "LLM endpoint unreachable"})
    seen = {}
    monkeypatch.setattr(conv_mod, "sync",
                        lambda client, current, target: seen.setdefault("t", target).copy())
    r = conv.turn("more", on_event=lambda e: None, on_gate=None)
    assert r.status == "failed"
    assert list(seen["t"].nodes) == ["n1"]
    assert conv.turn_idx == 1 and list(conv.graph.nodes) == ["n1"]


def test_needs_input_commits_without_failing(conv, monkeypatch):
    conv.graph = _graph_with(1); conv.turn_idx = 1
    _fake_drive(monkeypatch, {"build_graph": conv.graph.to_dict(),
                              "turn_outcome": {"summary": "", "question": "which one?"}})
    r = conv.turn("move it", on_event=lambda e: None, on_gate=None)
    assert r.status == "needs_input" and "which one?" in r.message


def test_session_loss_ends_the_conversation(conv, monkeypatch):
    _fake_drive(monkeypatch, {"fatal": "SALOME agent server unavailable: socket closed"})
    r = conv.turn("x", on_event=lambda e: None, on_gate=None)
    assert r.status == "ended" and conv.ended
    assert conv.turn("y", on_event=lambda e: None, on_gate=None).status == "ended"


def test_degraded_blocks_turns_until_rebuild(conv, monkeypatch):
    conv.graph = _graph_with(1); conv.turn_idx = 1
    _fake_drive(monkeypatch, {"build_graph": _graph_with(2).to_dict(), "degraded": True,
                              "fatal": "rollback failed: x"})
    assert conv.turn("x", on_event=lambda e: None, on_gate=None).status == "degraded"
    assert conv.turn("y", on_event=lambda e: None, on_gate=None).status == "degraded"
    calls = {}

    def fake_rebuild_all(client, graph, extra_names=()):
        calls["extra"] = list(extra_names)
        return graph.copy()

    monkeypatch.setattr(conv_mod, "rebuild_all", fake_rebuild_all)
    assert conv.rebuild().status == "ok"
    assert "box_2" in calls["extra"] and not conv.degraded


def test_undo_restores_the_previous_snapshot(conv, monkeypatch):
    _fake_drive(monkeypatch, {"build_graph": _graph_with(1).to_dict()})
    conv.turn("one", on_event=lambda e: None, on_gate=None)
    _fake_drive(monkeypatch, {"build_graph": _graph_with(2).to_dict()})
    conv.turn("two", on_event=lambda e: None, on_gate=None)
    monkeypatch.setattr(conv_mod, "sync", lambda client, current, target: target.copy())
    assert conv.undo().status == "ok"
    assert list(conv.graph.nodes) == ["n1"] and conv.turn_idx == 1
    assert conv.undo().status == "ok" and conv.turn_idx == 0
    assert conv.undo().status == "failed"


def test_real_drive_runs_amend_turn_through_langgraph(conv):
    import os

    conv.client.mode = "headless"
    conv.client.name_prefix = "p_"
    conv.client.list_tools.return_value = {"tools": []}
    conv.client.call_tool.return_value = {"ok": False}
    conv.llm.chat.return_value = {"content": None, "raw_message": None,
                                  "tool_calls": [{"name": "finish_turn",
                                                  "args": {"summary": "made it red"}}]}
    conv.graph = _graph_with(1)
    conv.turn_idx = 1

    r = conv.turn("make it red", on_event=lambda e: None, on_gate=None)

    assert r.status == "ok"
    assert conv.turn_idx == 2
    assert os.path.exists(os.path.join(conv.run_dir, "result.py"))
    assert os.path.exists(os.path.join(conv.run_dir, "build_graph.json"))


def test_turn_input_carries_graph_turn_and_selection(conv, monkeypatch):
    seen = {}

    def fake(app, inp, cfg, mapper, llm, on_event, on_gate, cancel):
        seen.update(inp)
        return {**inp, "build_graph": inp["build_graph"]}

    monkeypatch.setattr(conv_mod, "drive", fake)
    conv.graph = _graph_with(1); conv.turn_idx = 1
    sel = [{"object": "box_1", "subshape": None}]
    conv.turn("x", selection=sel, on_event=lambda e: None, on_gate=None)
    assert seen["turn_idx"] == 1 and seen["selection"] == sel
    assert seen["build_graph"]["nodes"][0]["id"] == "n1"


def test_ok_turn_undo_and_rebuild_display_the_results(conv, monkeypatch):
    shown = []
    monkeypatch.setattr(conv_mod, "show_results", lambda client, g: shown.append(g))
    _fake_drive(monkeypatch, {"build_graph": _graph_with(1).to_dict()})
    conv.turn("one", on_event=lambda e: None, on_gate=None)
    monkeypatch.setattr(conv_mod, "sync", lambda client, current, target: target.copy())
    conv.undo()
    monkeypatch.setattr(conv_mod, "rebuild_all", lambda client, graph, extra_names=(): graph.copy())
    conv.rebuild()
    assert len(shown) == 3


def test_redo_reapplies_what_undo_took_back(conv, monkeypatch):
    monkeypatch.setattr(conv_mod, "sync", lambda client, current, target: target.copy())
    _fake_drive(monkeypatch, {"build_graph": _graph_with(1).to_dict()})
    conv.turn("one", on_event=lambda e: None, on_gate=None)
    _fake_drive(monkeypatch, {"build_graph": _graph_with(2).to_dict()})
    conv.turn("two", on_event=lambda e: None, on_gate=None)
    hist = list(conv.history)
    conv.undo()
    assert list(conv.graph.nodes) == ["n1"] and conv.turn_idx == 1
    r = conv.redo()
    assert r.status == "ok"
    assert list(conv.graph.nodes) == ["n1", "n2"] and conv.turn_idx == 2
    assert conv.history == hist
    assert conv.redo().status == "failed"          # nothing left to redo
    conv.undo()                                     # and undo still works after a redo
    assert list(conv.graph.nodes) == ["n1"]


def test_a_new_turn_clears_redo(conv, monkeypatch):
    monkeypatch.setattr(conv_mod, "sync", lambda client, current, target: target.copy())
    _fake_drive(monkeypatch, {"build_graph": _graph_with(1).to_dict()})
    conv.turn("one", on_event=lambda e: None, on_gate=None)
    conv.undo()
    _fake_drive(monkeypatch, {"build_graph": _graph_with(2).to_dict()})
    conv.turn("other", on_event=lambda e: None, on_gate=None)
    assert conv.redo().status == "failed"


def test_redo_failure_degrades(conv, monkeypatch):
    monkeypatch.setattr(conv_mod, "sync", lambda client, current, target: target.copy())
    _fake_drive(monkeypatch, {"build_graph": _graph_with(1).to_dict()})
    conv.turn("one", on_event=lambda e: None, on_gate=None)
    conv.undo()

    def boom(client, current, target):
        raise conv_mod.ReplayFailed("n1", "x")

    monkeypatch.setattr(conv_mod, "sync", boom)
    assert conv.redo().status == "degraded" and conv.degraded


def _stage_calls(*names, status="failed"):
    return {0: {"module": "geom", "status": status, "call_log": [
        {"fn": "F", "args": [], "out_vars": [n], "publish": [n]} for n in names]}}


def test_cancelled_turn_forgets_objects_its_stages_built(conv, monkeypatch):
    # a fresh-plan turn cancelled mid-stage never reaches assemble, so its
    # objects are only in the stage's call_log, not in build_graph (task 259)
    conv.graph = _graph_with(1); conv.turn_idx = 1
    conv.client.forget.return_value = {"ok": True}
    _fake_drive(monkeypatch, {"aborted": True, "stages": _stage_calls("box_1", "box_2", "vec_1")})
    r = conv.turn("more", on_event=lambda e: None, on_gate=None)
    assert r.status == "failed"
    conv.client.forget.assert_called_once_with(["box_2", "vec_1"])
    assert list(conv.graph.nodes) == ["n1"]


def test_failed_stage_objects_join_the_stray_set_when_their_forget_fails(conv, monkeypatch):
    conv.client.forget.return_value = {"ok": False, "error": "boom"}
    _fake_drive(monkeypatch, {"fatal": "stage 1 failed", "stages": _stage_calls("box_9")})
    r = conv.turn("x", on_event=lambda e: None, on_gate=None)
    assert r.status == "degraded"
    assert "box_9" in conv._stray


def test_ok_turn_does_not_forget_stage_objects(conv, monkeypatch):
    _fake_drive(monkeypatch, {"build_graph": _graph_with(1).to_dict(),
                              "stages": _stage_calls("box_1", status="ok")})
    conv.turn("one", on_event=lambda e: None, on_gate=None)
    conv.client.forget.assert_not_called()


def _drive_with_stages(monkeypatch, final_update, stages):
    """stages: list of name lists, each sent as a stage_done event."""
    def fake(app, inp, cfg, mapper, llm, on_event, on_gate, cancel):
        for i, names in enumerate(stages):
            on_event({"kind": "stage_done", "run_id": "r1", "data": {"index": i, "names": names}})
        return {**inp, **final_update}
    monkeypatch.setattr(conv_mod, "drive", fake)


def test_each_finished_stage_is_shown_in_the_viewer_at_once(conv, monkeypatch):
    monkeypatch.setattr(conv_mod, "show_results", lambda client, g: None)
    order = []
    conv.client.display.side_effect = lambda names: order.append(("display", names))
    events = []

    def on_event(e):
        events.append(e)
        order.append(("event", e["kind"]))

    _drive_with_stages(monkeypatch, {"build_graph": _graph_with(2).to_dict()},
                       [["box_1"], [], ["box_2"]])
    conv.turn("two boxes", on_event=on_event, on_gate=None)
    assert [o for o in order if o[0] == "display"] == [("display", ["box_1"]),
                                                        ("display", ["box_2"])]
    assert order.index(("display", ["box_1"])) < order.index(("event", "turn_done"))


def test_stage_objects_that_are_not_results_are_hidden_at_turn_end(conv, monkeypatch):
    monkeypatch.setattr(conv_mod, "show_results", lambda client, g: None)
    _drive_with_stages(monkeypatch, {"build_graph": _graph_with(1).to_dict()},
                       [["box_1", "tool_1"], ["cut_1"]])
    conv.turn("x", on_event=lambda e: None, on_gate=None)
    conv.client.hide.assert_called_once_with(["tool_1", "cut_1"])


def test_nothing_to_hide_sends_no_hide(conv, monkeypatch):
    monkeypatch.setattr(conv_mod, "show_results", lambda client, g: None)
    _drive_with_stages(monkeypatch, {"build_graph": _graph_with(1).to_dict()}, [["box_1"]])
    conv.turn("x", on_event=lambda e: None, on_gate=None)
    conv.client.hide.assert_not_called()


def test_a_failing_stage_display_does_not_fail_the_turn(conv, monkeypatch):
    monkeypatch.setattr(conv_mod, "show_results", lambda client, g: None)
    conv.client.display.side_effect = RuntimeError("viewer gone")
    conv.client.hide.side_effect = RuntimeError("viewer gone")
    _drive_with_stages(monkeypatch, {"build_graph": _graph_with(1).to_dict()},
                       [["box_1", "tool_1"]])
    assert conv.turn("x", on_event=lambda e: None, on_gate=None).status == "ok"


def test_turn_done_carries_the_usage_totals_even_when_cancelled(conv, monkeypatch):
    conv.llm.usage = {"calls": 7, "prompt_tokens": 900}
    _fake_drive(monkeypatch, {"aborted": True})
    monkeypatch.setattr(conv_mod, "sync", lambda client, current, target: target.copy())
    events = []
    conv.turn("x", on_event=events.append, on_gate=None)
    done = [e for e in events if e["kind"] == "turn_done"][-1]
    assert done["data"]["status"] == "failed"
    assert done["data"]["usage"] == {"calls": 7, "prompt_tokens": 900}
    conv.llm.usage["calls"] = 99
    assert done["data"]["usage"]["calls"] == 7          # a snapshot, not the live counter


def test_an_image_reading_reaches_later_turns_until_undone(conv, monkeypatch):
    seen = []

    def fake(app, inp, cfg, mapper, llm, on_event, on_gate, cancel):
        seen.append([e for e in inp["history"] if e["event"] == "image"])
        spec = {"summary": "a plate"} if inp["reference_images"] else None
        return {**inp, "build_graph": _graph_with(1).to_dict(), "image_spec": spec}

    monkeypatch.setattr(conv_mod, "drive", fake)
    monkeypatch.setattr(conv_mod, "sync", lambda client, current, target: target.copy())
    conv.turn("like this", images=["data:image/png;base64,AAAA"], on_event=lambda e: None,
              on_gate=None)
    conv.turn("missing the bars", on_event=lambda e: None, on_gate=None)
    assert seen[1] == [{"event": "image", "turn": 0, "spec": {"summary": "a plate"}}]
    conv.undo(); conv.undo()
    conv.turn("a box", on_event=lambda e: None, on_gate=None)
    assert seen[2] == []


def test_a_replanned_turn_without_images_records_no_reading_of_its_own(conv, monkeypatch):
    # a replan carries an earlier turn's reading in image_spec; it is not this turn's
    finals = iter([{"image_spec": {"summary": "a plate"}}, {"image_spec": {"summary": "a plate"}}])

    def fake(app, inp, cfg, mapper, llm, on_event, on_gate, cancel):
        return {**inp, "build_graph": _graph_with(1).to_dict(), **next(finals)}

    monkeypatch.setattr(conv_mod, "drive", fake)
    monkeypatch.setattr(conv_mod, "sync", lambda client, current, target: target.copy())
    conv.turn("like this", images=["data:image/png;base64,AAAA"], on_event=lambda e: None,
              on_gate=None)
    conv.turn("start over", on_event=lambda e: None, on_gate=None)
    assert [e["turn"] for e in conv.history if e["event"] == "image"] == [0]


def test_a_planned_turn_records_what_its_stages_built(conv, monkeypatch):
    _fake_drive(monkeypatch, {"build_graph": _graph_with(1).to_dict(), "stages": {
        0: {"status": "ok", "built": "a plate of diameter 100"},
        1: {"status": "failed", "built": "half a slot"},
        2: {"status": "ok", "built": "four slots"}}})
    conv.turn("like this", on_event=lambda e: None, on_gate=None)
    assert conv.history[-1]["summary"] == "a plate of diameter 100; four slots"


def test_images_stay_with_the_conversation_and_a_reread_replaces_the_reading(conv, monkeypatch):
    seen = []
    finals = iter([
        {"build_graph": _graph_with(1).to_dict(), "image_spec": {"summary": "rings"}},
        {"build_graph": _graph_with(1).to_dict(),
         "turn_outcome": {"summary": "re-read", "image_rereads": [(0, {"summary": "slots"})]}},
    ])

    def fake(app, inp, cfg, mapper, llm, on_event, on_gate, cancel):
        seen.append(cfg["configurable"]["turn_images"])
        return {**inp, **next(finals)}

    monkeypatch.setattr(conv_mod, "drive", fake)
    conv.turn("this plate", images=["data:image/png;base64,AAAA"], on_event=lambda e: None,
              on_gate=None)
    conv.turn("you misread it", on_event=lambda e: None, on_gate=None)
    assert seen[1] == {0: {"request": "this plate", "images": ["data:image/png;base64,AAAA"]}}
    from salome_agent.graph import _earlier_images, _readings
    assert _readings({"history": conv.history}) == {0: {"summary": "slots"}}
    assert len(_earlier_images({"history": conv.history})) == 1
