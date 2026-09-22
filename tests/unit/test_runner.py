import os

from salome_agent.runner import EventMapper

PLAN = [{"module": "geom", "intent": "box", "params": {}, "inputs": []},
        {"module": "smesh", "intent": "mesh it", "params": {}, "inputs": []}]


def _kinds(evs):
    return [e["kind"] for e in evs]


def test_plan_update_emits_plan_event():
    m = EventMapper("r1", "/w/r1")
    evs = m.events("plan", {"plan": PLAN, "stages": {}, "stage_idx": 0})
    assert _kinds(evs) == ["plan"]
    assert evs[0] == {"kind": "plan", "run_id": "r1", "data": {"stages": PLAN}}


def test_edited_plan_at_the_gate_re_emits_plan():
    m = EventMapper("r1", "/w/r1")
    m.events("plan", {"plan": PLAN})
    evs = m.events("plan_gate", {"plan": PLAN[:1], "stages": {}, "stage_idx": 0})
    assert _kinds(evs) == ["plan"]
    assert evs[0]["data"]["stages"] == PLAN[:1]


def test_first_run_stage_emits_stage_start_once():
    m = EventMapper("r1", "/w/r1")
    m.events("plan", {"plan": PLAN})
    first = m.events("run_stage", {"stages": {}})
    again = m.events("run_stage", {"stages": {}})
    assert _kinds(first) == ["stage_start"]
    assert first[0]["data"] == {"index": 0, "module": "geom", "intent": "box"}
    assert again == []


def test_failed_verify_emits_failed_attempt():
    m = EventMapper("r1", "/w/r1")
    m.events("plan", {"plan": PLAN})
    m.events("run_stage", {})
    err = {"phase": "exec", "message": "boom", "offending_line": 3, "traceback": None}
    evs = m.events("verify_exec", {"stages": {}, "last_error": err})
    assert evs == [{"kind": "stage_attempt", "run_id": "r1",
                    "data": {"index": 0, "phase": "exec", "ok": False, "error": "boom"}}]


def test_passing_verify_static_with_no_update_emits_ok_attempt():
    m = EventMapper("r1", "/w/r1")
    m.events("plan", {"plan": PLAN})
    evs = m.events("verify_static", None)
    assert evs[0]["data"] == {"index": 0, "phase": "static", "ok": True, "error": None}


def test_advance_emits_stage_done_and_moves_index():
    m = EventMapper("r1", "/w/r1")
    m.events("plan", {"plan": PLAN})
    m.events("run_stage", {})
    evs = m.events("advance", {"stages": {0: {"status": "ok", "result_names": ["box_1"]}},
                               "stage_idx": 1, "last_error": None})
    assert evs[0] == {"kind": "stage_done", "run_id": "r1",
                      "data": {"index": 0, "names": ["box_1"]}}
    nxt = m.events("run_stage", {})
    assert nxt[0]["data"]["index"] == 1


def test_skip_emits_stage_skipped():
    m = EventMapper("r1", "/w/r1")
    m.events("plan", {"plan": PLAN})
    evs = m.events("skip", {"stages": {0: {"status": "skipped"}}, "stage_idx": 1})
    assert _kinds(evs) == ["stage_skipped"]
    assert evs[0]["data"] == {"index": 0}


def test_fatal_and_aborted_and_report():
    m = EventMapper("r1", "/w/r1")
    assert m.events("plan", {"fatal": "LLM endpoint down"})[0] == {
        "kind": "fatal", "run_id": "r1", "data": {"message": "LLM endpoint down"}}
    assert _kinds(m.events("plan_gate", {"aborted": True})) == ["aborted"]
    rep = m.events("report", {"report": "# Report"})[0]
    assert rep["kind"] == "report"
    assert rep["data"] == {"result_py": "/w/r1/result.py", "report_md": "/w/r1/report.md",
                           "text": "# Report"}


def test_fatal_from_run_stage_is_not_also_an_attempt():
    m = EventMapper("r1", "/w/r1")
    m.events("plan", {"plan": PLAN})
    m.events("run_stage", {})
    assert _kinds(m.events("run_stage", {"fatal": "x"})) == ["fatal"]


import threading
from unittest.mock import MagicMock

import requests

from salome_agent.runner import run_request

ONE_STAGE = {"stages": [{"module": "geom", "intent": "box", "params": {}, "inputs": []}]}


def _config(tmp_path):
    return {"llm": {"base_url": "http://x", "model": "m"}, "work_dir": str(tmp_path),
            "policy": {"tool_iteration_cap": 3, "on_stage_failure": "abort"}}


def _fake_llm(monkeypatch, llm):
    monkeypatch.setattr("salome_agent.cli.build_llm", lambda *a, **k: llm)


def _http_error(code):
    resp = requests.Response()
    resp.status_code = code
    return requests.HTTPError(f"{code} Server Error", response=resp)


def test_plan_gate_abort_emits_plan_then_aborted_and_writes_nothing(tmp_path, monkeypatch):
    llm = MagicMock()
    llm.chat.return_value = ONE_STAGE
    llm.usage = None
    _fake_llm(monkeypatch, llm)
    events, gates = [], []

    def on_gate(payload):
        gates.append(payload)
        return {"action": "abort"}

    res = run_request("make a box", _config(tmp_path), client=None,
                      on_event=events.append, on_gate=on_gate, run_id="r1")
    assert [e["kind"] for e in events] == ["plan", "aborted"]
    assert gates == [{"plan": ONE_STAGE["stages"]}]
    assert res.exit_code == 1
    assert res.run_id == "r1"
    assert res.run_dir == str(tmp_path / "r1")
    assert res.final_state["aborted"] is True
    assert list(tmp_path.iterdir()) == []


def test_edit_at_the_gate_re_prompts_with_the_edited_plan(tmp_path, monkeypatch):
    llm = MagicMock()
    llm.chat.return_value = ONE_STAGE
    llm.usage = None
    _fake_llm(monkeypatch, llm)
    edited = [{"module": "geom", "intent": "cylinder", "params": {}, "inputs": []}]
    answers = iter([{"action": "edit", "plan": edited}, {"action": "abort"}])
    gates = []

    def on_gate(payload):
        gates.append(payload)
        return next(answers)

    run_request("x", _config(tmp_path), client=None, on_event=lambda e: None,
                on_gate=on_gate, run_id="r1")
    assert gates[1] == {"plan": edited}


def test_endpoint_fatal_in_batch_emits_fatal_and_report(tmp_path, monkeypatch):
    llm = MagicMock()
    llm.chat.side_effect = _http_error(502)
    llm.usage = {"calls": 1}
    _fake_llm(monkeypatch, llm)
    events = []

    res = run_request("x", _config(tmp_path), client=None, on_event=events.append,
                      on_gate=lambda p: {"action": "abort"}, batch=True, run_id="r1")
    kinds = [e["kind"] for e in events]
    assert "fatal" in kinds and kinds[-1] == "report"
    assert events[-1]["data"]["usage"] == {"calls": 1}
    assert res.exit_code == 1
    assert (tmp_path / "r1" / "report.md").exists()


def test_cancel_before_the_gate_answers_ends_aborted_without_resuming(tmp_path, monkeypatch):
    llm = MagicMock()
    llm.chat.return_value = ONE_STAGE
    llm.usage = None
    _fake_llm(monkeypatch, llm)
    cancel = threading.Event()
    events = []

    def on_gate(payload):
        cancel.set()
        return {"action": "approve"}

    res = run_request("x", _config(tmp_path), client=None, on_event=events.append,
                      on_gate=on_gate, cancel=cancel, run_id="r1")
    assert [e["kind"] for e in events] == ["plan", "aborted"]
    assert res.final_state["aborted"] is True
    assert res.exit_code == 1
    assert llm.chat.call_count == 1          # planner only, no specialist turn


def test_run_dir_is_absolute_even_with_a_relative_work_dir(tmp_path, monkeypatch):
    llm = MagicMock()
    llm.chat.return_value = ONE_STAGE
    llm.usage = None
    _fake_llm(monkeypatch, llm)
    monkeypatch.chdir(tmp_path)
    config = {"llm": {"base_url": "http://x", "model": "m"}, "work_dir": "rel",
              "policy": {"tool_iteration_cap": 3, "on_stage_failure": "abort"}}

    res = run_request("x", config, client=None, on_event=lambda e: None,
                      on_gate=lambda p: {"action": "abort"}, run_id="r1")
    assert os.path.isabs(res.run_dir)
    assert res.run_dir == str(tmp_path / "rel" / "r1")


def test_client_supplies_the_session_handle(tmp_path, monkeypatch):
    llm = MagicMock()
    llm.chat.return_value = ONE_STAGE
    llm.usage = None
    _fake_llm(monkeypatch, llm)
    client = MagicMock(mode="gui", name_prefix="p_")
    res = run_request("x", _config(tmp_path), client=client, on_event=lambda e: None,
                      on_gate=lambda p: {"action": "abort"}, run_id="r1")
    assert res.final_state["session"]["mode"] == "gui"
    assert res.final_state["session"]["name_prefix"] == "p_"


def test_amend_history_events_become_ui_events():
    m = EventMapper("r1", "/w/r1")
    evs = m.events("amend", {"history": [
        {"event": "graph_edit", "op": "edit", "node": "n1", "param": "dz", "old": 1, "new": 2},
        {"event": "replay", "nodes": ["n1", "n3"]},
        {"event": "retry_hint", "hint": "x"}]})
    assert [e["kind"] for e in evs] == ["graph_edit", "replay"]
    assert evs[1]["data"] == {"nodes": ["n1", "n3"]}


def test_show_results_displays_the_graph_result_names():
    from unittest.mock import MagicMock

    from salome_agent.build_graph import BuildGraph
    from salome_agent.runner import show_results

    g = BuildGraph()
    b = g.add_block(0, "stage 1/1: geom", "geom", ["fuse_1"])
    g.append(b, "fuse", {}, [{"fn": "F", "args": [], "out_vars": ["fuse_1"],
                              "publish": ["fuse_1"]}], 0)
    client = MagicMock()
    show_results(client, g.to_dict())
    client.display.assert_called_once_with(["fuse_1"])


def test_show_results_is_best_effort():
    from unittest.mock import MagicMock

    from salome_agent.runner import show_results

    client = MagicMock()
    client.display.side_effect = RuntimeError("no viewer")
    show_results(client, {"nodes": [], "blocks": []})    # nothing to show: no call
    client.display.assert_not_called()
    show_results(None, {"nodes": [], "blocks": []})
    from salome_agent.build_graph import BuildGraph
    g = BuildGraph()
    b = g.add_block(0, "s", "geom", [])
    g.append(b, "make_box", {}, [{"fn": "F", "args": [], "out_vars": ["box_1"],
                                  "publish": ["box_1"]}], 0)
    show_results(client, g.to_dict())                     # raises inside: swallowed
