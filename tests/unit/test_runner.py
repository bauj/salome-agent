import os

from salome_agent.runner import EventMapper

PLAN = [{"module": "geom", "intent": "box", "params": {}, "inputs": []},
        {"module": "smesh", "intent": "mesh it", "params": {}, "inputs": []}]


def test_reasoning_source_follows_planner_stage_and_edit():
    m = EventMapper("r1", "/w/r1")
    ev = m.reasoning("planner", 1, "reasoning", "hmm")
    assert ev == {"kind": "reasoning", "run_id": "r1",
                  "data": {"source": "planner", "index": None, "call": 1,
                           "channel": "reasoning", "text": "hmm"}}
    assert m.reasoning("geom", 2, "prose", "x")["data"]["source"] == "edit"
    m.events("plan", {"plan": PLAN})
    m.events("advance", {"stages": {}, "stage_idx": 1})
    d = m.reasoning("smesh", 3, "prose", "x")["data"]
    assert (d["source"], d["index"]) == ("stage", 1)


def test_reasoning_relay_throttles_and_flushes_on_end_and_channel_switch():
    from salome_agent.runner import ReasoningRelay
    now = [0.0]
    out = []
    relay = ReasoningRelay(EventMapper("r1", "/w"), out.append, clock=lambda: now[0])
    relay({"type": "start", "module": "planner"})
    relay({"type": "delta", "channel": "reasoning", "text": "a"})
    now[0] = 0.2
    relay({"type": "delta", "channel": "reasoning", "text": "b"})
    assert out == []
    now[0] = 0.6
    relay({"type": "delta", "channel": "reasoning", "text": "c"})
    relay({"type": "delta", "channel": "prose", "text": "p"})
    relay({"type": "end"})
    assert [(e["data"]["call"], e["data"]["channel"], e["data"]["text"]) for e in out] == [
        (1, "reasoning", "abc"), (1, "prose", "p")]
    relay({"type": "start", "module": "planner"})
    relay({"type": "end"})
    assert len(out) == 2                     # an empty call emits nothing


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


def test_cancel_during_a_stage_stops_its_tool_loop(tmp_path, monkeypatch):
    cancel = threading.Event()
    llm = MagicMock()
    llm.usage = None

    def chat(**kw):
        if llm.chat.call_count == 1:
            return ONE_STAGE
        cancel.set()
        return {"tool_calls": [{"name": "make_box", "args": {"dx": 1}, "id": None}],
                "content": None, "raw_message": None}

    llm.chat.side_effect = chat
    _fake_llm(monkeypatch, llm)
    client = MagicMock(mode="headless", name_prefix="")
    client.list_tools.return_value = {"tools": []}
    res = run_request("x", _config(tmp_path), client=client, on_event=lambda e: None,
                      on_gate=lambda p: {"action": "approve"}, batch=True,
                      cancel=cancel, run_id="r1")
    assert res.final_state["aborted"] is True
    assert llm.chat.call_count == 2          # planner + one specialist turn
    client.call_tool.assert_not_called()


def test_cancel_while_the_planner_streams_ends_aborted(tmp_path, monkeypatch):
    from salome_agent.llm import LLMCancelled
    cancel = threading.Event()
    llm = MagicMock()
    llm.usage = None

    def chat(**kw):
        assert llm.cancel is cancel          # the turn's event reaches the client
        cancel.set()
        raise LLMCancelled()

    llm.chat.side_effect = chat
    _fake_llm(monkeypatch, llm)
    events = []
    res = run_request("x", _config(tmp_path), client=None, on_event=events.append,
                      on_gate=lambda p: {"action": "approve"}, cancel=cancel, run_id="r1")
    assert [e["kind"] for e in events] == ["aborted"]
    assert res.final_state["aborted"] is True
    assert res.exit_code == 1


def test_drive_relays_streamed_reasoning_as_events(tmp_path, monkeypatch):
    llm = MagicMock()
    llm.usage = None

    def chat(**kw):
        llm.on_stream({"type": "start", "module": "planner"})
        llm.on_stream({"type": "delta", "channel": "reasoning", "text": "one box"})
        llm.on_stream({"type": "end"})
        return ONE_STAGE

    llm.chat.side_effect = chat
    _fake_llm(monkeypatch, llm)
    events = []
    run_request("x", _config(tmp_path), client=None, on_event=events.append,
                on_gate=lambda p: {"action": "abort"}, run_id="r1")
    assert events[0]["kind"] == "reasoning"
    assert events[0]["data"]["text"] == "one box"
    assert llm.on_stream is None             # removed when the turn ends


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


from types import SimpleNamespace

from salome_agent.runner import drive


class _StreamApp:
    """A compiled graph stand-in: yields fixed (mode, chunk) pairs."""

    def __init__(self, chunks):
        self.chunks = chunks
        self.modes = None

    def stream(self, inp, config, stream_mode):
        self.modes = stream_mode
        yield from self.chunks

    def get_state(self, cfg):
        return SimpleNamespace(values={})


def test_drive_passes_custom_chunks_through_as_events():
    m = EventMapper("r1", "/w/r1")
    app = _StreamApp([("custom", {"kind": "vision_start", "data": {"images": 1}}),
                      ("updates", {"plan": {"plan": PLAN}})])
    events = []
    drive(app, {}, {"configurable": {}}, m, MagicMock(), events.append, None, None)
    assert app.modes == ["updates", "custom"]
    assert [e["kind"] for e in events] == ["vision_start", "plan"]
    assert events[0] == {"kind": "vision_start", "run_id": "r1", "data": {"images": 1}}


def test_vision_progress_reaches_on_event_before_the_plan(tmp_path, monkeypatch):
    from tests.unit.test_vision_spec import box_spec
    llm = MagicMock()
    llm.usage = None
    llm.endpoint.return_value.model = "reader"
    llm.chat.side_effect = lambda **kw: box_spec() if kw.get("module") == "vision" else ONE_STAGE
    _fake_llm(monkeypatch, llm)
    events = []
    run_request("x", _config(tmp_path), client=None, on_event=events.append,
                on_gate=lambda p: {"action": "abort"}, run_id="r1",
                reference_images=["data:image/png;base64,AAAA"])
    kinds = [e["kind"] for e in events if e["kind"] != "reasoning"]
    assert kinds[:3] == ["vision_start", "vision_done", "plan"]
    start = next(e for e in events if e["kind"] == "vision_start")
    assert start["data"] == {"images": 1, "sizes": [], "model": "reader"}


def test_vision_reasoning_has_its_own_source_with_or_without_a_plan():
    m = EventMapper("r1", "/w/r1")
    assert m.reasoning("vision", 1, "reasoning", "hmm")["data"]["source"] == "vision"
    m.events("plan", {"plan": PLAN})
    d = m.reasoning("vision", 2, "reasoning", "x")["data"]
    assert (d["source"], d["index"]) == ("vision", None)


def test_a_stage_given_up_after_loops_emits_a_failed_attempt():
    m = EventMapper("r1", "/w/r1")
    m.events("plan", {"plan": PLAN, "stages": {}, "stage_idx": 0})
    evs = m.events("run_stage", {"last_error": {
        "phase": "llm", "message": "stage ended: looped 4 times in a row"}})
    attempt = [e for e in evs if e["kind"] == "stage_attempt"][0]["data"]
    assert attempt["phase"] == "llm" and attempt["ok"] is False
    assert "looped" in attempt["error"]
