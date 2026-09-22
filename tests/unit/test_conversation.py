from unittest.mock import MagicMock

import pytest

from salome_agent.build_graph import BuildGraph
from salome_agent import conversation as conv_mod
from salome_agent.conversation import Conversation

CONFIG = {"work_dir": None, "llm": {}, "policy": {
    "tool_iteration_cap": 5, "static_retries": 1, "exec_retries": 1, "exec_timeout_s": 120,
    "tool_selection": "hints", "verify_cadence": "none", "plan_detail": "balanced",
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
