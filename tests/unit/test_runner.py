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
