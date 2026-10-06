import json
import os
import uuid

from salome_agent.build_graph import BuildGraph
from salome_agent.graph import build_graph, route_start

POLICY = {"tool_iteration_cap": 5, "static_retries": 1, "exec_retries": 1,
          "exec_timeout_s": 120, "tool_selection": "hints", "verify_cadence": "none",
          "plan_detail": "balanced", "on_stage_failure": "abort"}


def _state(tmp_path, graph, text, turn=1):
    return {"run_id": "r1", "user_request": text,
            "config": {"batch": True, "work_dir": str(tmp_path), "policy": POLICY},
            "plan": [], "plan_approved": True, "stage_idx": 0, "plan_timing": None,
            "session": {"mode": "none", "pid": None, "salome_port": None, "name_prefix": ""},
            "study_ns": "salome_agent_run", "stages": {}, "last_error": None,
            "history": [], "final_script": None, "report": None, "fatal": None,
            "aborted": None, "build_graph": graph.to_dict(), "turn_idx": turn,
            "selection": None, "turn_outcome": None, "replan_context": None,
            "degraded": None}


def _graph():
    g = BuildGraph()
    g.requests.append("a box")
    b = g.add_block(0, "stage 1/1: geom", "geom", ["box_1"])
    g.append(b, "make_box", {"dx": 1, "dy": 1, "dz": 1},
             [{"fn": "MakeBoxDXDYDZ", "args": [{"kind": "literal", "value": 1.0}] * 3,
               "out_vars": ["box_1"], "publish": ["box_1"]}], 0)
    return g


def test_route_start():
    assert route_start({"turn_idx": 0}) == "plan"
    assert route_start({}) == "plan"
    assert route_start({"turn_idx": 2}) == "amend"
    # images go through describe_image (the "plan" branch) on any turn
    assert route_start({"turn_idx": 2, "reference_images": ["data:image/png;base64,AA"]}) == "plan"


def test_a_follow_up_naming_shaper_is_planned_not_amended():
    # amend has only GEOM tools
    assert route_start({"turn_idx": 1, "user_request": "add a rib in SHAPER"}) == "plan"
    assert route_start({"turn_idx": 1, "user_request": "add a rib in GEOM"}) == "amend"
    assert route_start({"turn_idx": 1, "user_request": "add a rib"}) == "amend"


def test_amend_turn_rewrites_result_and_graph_json(tmp_path, monkeypatch):
    from salome_agent.specialists import amend as amend_mod

    def fake_run_amend(text, graph, llm, client, turn, policy, *, history=(), selection=None,
                       session=None, cancel=None):
        session = amend_mod.AmendSession(graph, client, turn)
        session.graph = graph.copy()
        session.graph.nodes["n1"].tool_args["dz"] = 3
        session.graph.nodes["n1"].records[0]["args"][2] = {"kind": "literal", "value": 3.0}
        session.events = [{"event": "replay", "nodes": ["n1"]}]
        return session, {"summary": "taller", "question": None}

    monkeypatch.setattr(amend_mod, "run_amend", fake_run_amend)
    app = build_graph(batch=True)
    cfg = {"configurable": {"thread_id": uuid.uuid4().hex, "llm_client": None,
                            "executor_client": None}}
    out = app.invoke(_state(tmp_path, _graph(), "make it taller"), config=cfg)
    assert out["turn_outcome"] == {"summary": "taller", "question": None}
    assert "MakeBoxDXDYDZ(1.0, 1.0, 3.0)" in out["final_script"]
    assert "# turn 2: make it taller" in out["final_script"]
    run_dir = tmp_path / "r1"
    assert json.loads((run_dir / "build_graph.json").read_text())["nodes"][0]["tool_args"]["dz"] == 3
    assert "# Turn 2" in (run_dir / "report.md").read_text()


def test_replan_goes_to_the_planner_with_context(tmp_path, monkeypatch):
    from salome_agent.specialists import amend as amend_mod
    from salome_agent.nodes import planner as planner_mod

    monkeypatch.setattr(amend_mod, "run_amend",
                        lambda *a, **k: (amend_mod.AmendSession(a[1], a[3], a[4]),
                                         {"replan": "new wing"}))
    seen = {}

    def fake_plan(request, llm, detail="balanced", **kw):
        seen["request"] = request
        raise RuntimeError("stop after planning")   # _plan_node turns this into a clean fatal

    monkeypatch.setattr(planner_mod, "plan", fake_plan)
    app = build_graph(batch=True)
    cfg = {"configurable": {"thread_id": uuid.uuid4().hex, "llm_client": None,
                            "executor_client": None}}
    app.invoke(_state(tmp_path, _graph(), "add a wing"), config=cfg)
    assert seen["request"].startswith("add a wing")
    assert "box_1" in seen["request"]


def test_llm_failure_mid_amend_keeps_the_edits_already_applied(tmp_path, monkeypatch):
    from salome_agent.specialists import amend as amend_mod

    def fake_run_amend(text, graph, llm, client, turn, policy, *, history=(), selection=None,
                       session=None, cancel=None):
        session.graph = graph.copy()
        session.graph.nodes["n1"].tool_args["dz"] = 3
        raise RuntimeError("endpoint died")

    monkeypatch.setattr(amend_mod, "run_amend", fake_run_amend)
    app = build_graph(batch=True)
    cfg = {"configurable": {"thread_id": uuid.uuid4().hex, "llm_client": None,
                            "executor_client": None}}
    out = app.invoke(_state(tmp_path, _graph(), "make it taller"), config=cfg)
    assert out["fatal"].startswith("LLM endpoint error")
    assert out["build_graph"]["nodes"][0]["tool_args"]["dz"] == 3


def test_follow_up_with_images_is_described_then_amended(tmp_path, monkeypatch):
    from salome_agent.nodes import vision as vision_mod
    from salome_agent.specialists import amend as amend_mod

    monkeypatch.setattr(vision_mod, "describe_images",
                        lambda request, images, llm, **kw: {"summary": "a D-shaped handle",
                                                                      "parts": []})
    seen = {}

    def fake_run_amend(text, graph, llm, client, turn, policy, **kw):
        seen.update(kw)
        return amend_mod.AmendSession(graph, client, turn), {"summary": "ok", "question": None}

    monkeypatch.setattr(amend_mod, "run_amend", fake_run_amend)
    state = _state(tmp_path, _graph(), "add this handle")
    state["reference_images"] = ["data:image/png;base64,AAAA"]
    app = build_graph(batch=True)
    cfg = {"configurable": {"thread_id": uuid.uuid4().hex, "llm_client": None,
                            "executor_client": None}}
    out = app.invoke(state, config=cfg)
    assert "a D-shaped handle" in seen["image_description"]
    assert seen["image_description"].startswith("Reference image spec")
    assert out["turn_outcome"]["summary"] == "ok"


def _replan_seen(tmp_path, monkeypatch, history):
    from salome_agent.specialists import amend as amend_mod
    from salome_agent.nodes import planner as planner_mod

    monkeypatch.setattr(amend_mod, "run_amend",
                        lambda *a, **k: (amend_mod.AmendSession(a[1], a[3], a[4]),
                                         {"replan": "start over"}))
    seen = {}

    def fake_plan(request, llm, detail="balanced", **kw):
        seen["request"] = request
        raise RuntimeError("stop after planning")

    monkeypatch.setattr(planner_mod, "plan", fake_plan)
    state = _state(tmp_path, BuildGraph(), "retry", turn=2)
    state["history"] = history
    app = build_graph(batch=True)
    cfg = {"configurable": {"thread_id": uuid.uuid4().hex, "llm_client": None,
                            "executor_client": None}}
    app.invoke(state, config=cfg)
    return seen["request"]


def test_replan_tells_the_planner_the_earlier_requests(tmp_path, monkeypatch):
    request = _replan_seen(tmp_path, monkeypatch, [
        {"event": "turn", "turn": 0, "text": "A small car with 4 torus wheels",
         "status": "failed", "summary": None},
        {"event": "graph_edit", "op": "edit"},
        {"event": "turn", "turn": 1, "text": "make it red", "status": "ok", "summary": "red"}])
    assert request.startswith("retry")
    assert "A small car with 4 torus wheels" in request and "make it red" in request
    assert request.index("A small car") < request.index("make it red")
    assert "did not finish" in request


def _first_plan_seen(tmp_path, monkeypatch, history):
    from salome_agent.nodes import planner as planner_mod
    seen = {}

    def fake_plan(request, llm, detail="balanced", **kw):
        seen["request"] = request
        raise RuntimeError("stop after planning")

    monkeypatch.setattr(planner_mod, "plan", fake_plan)
    state = _state(tmp_path, BuildGraph(), "retry", turn=0)
    state["history"] = history
    cfg = {"configurable": {"thread_id": uuid.uuid4().hex, "llm_client": None,
                            "executor_client": None}}
    build_graph(batch=True).invoke(state, config=cfg)
    return seen["request"]


def test_retry_after_a_cancelled_first_turn_tells_the_planner_what_was_asked(tmp_path,
                                                                            monkeypatch):
    # a rolled-back turn leaves turn_idx at 0, so "retry" is planned, not amended
    request = _first_plan_seen(tmp_path, monkeypatch, [
        {"event": "turn", "turn": 0, "text": "A small car with 4 torus wheels",
         "status": "failed", "summary": None}])
    assert request.startswith("retry")
    assert "A small car with 4 torus wheels (did not finish)" in request


def test_first_plan_without_earlier_turns_is_the_bare_request(tmp_path, monkeypatch):
    assert _first_plan_seen(tmp_path, monkeypatch, []) == "retry"


def test_replan_lists_the_earlier_requests_once(tmp_path, monkeypatch):
    request = _replan_seen(tmp_path, monkeypatch, [
        {"event": "turn", "turn": 0, "text": "a box", "status": "ok", "summary": "box"}])
    assert request.count("Earlier requests") == 1


def test_replan_without_earlier_turns_adds_no_request_list(tmp_path, monkeypatch):
    assert "Earlier requests" not in _replan_seen(tmp_path, monkeypatch, [])


def test_an_empty_plan_becomes_a_question_not_a_gate(tmp_path, monkeypatch):
    from salome_agent.nodes import planner as planner_mod

    monkeypatch.setattr(planner_mod, "plan", lambda request, llm, detail="balanced", **kw: [])
    state = _state(tmp_path, BuildGraph(), "retry", turn=0)
    app = build_graph(batch=False)
    cfg = {"configurable": {"thread_id": uuid.uuid4().hex, "llm_client": None,
                            "executor_client": None}}
    out = app.invoke(state, config=cfg)
    assert "__interrupt__" not in out
    assert out["turn_outcome"]["question"]
    assert out["plan"] == []


def _shaper_follow_up_seen(tmp_path, monkeypatch, images=False, graph=None):
    from salome_agent.nodes import planner as planner_mod
    from salome_agent.nodes import vision as vision_mod
    from salome_agent.specialists import amend as amend_mod

    def no_amend(*a, **k):
        raise AssertionError("a SHAPER follow-up must not reach amend")

    monkeypatch.setattr(amend_mod, "run_amend", no_amend)
    monkeypatch.setattr(vision_mod, "describe_images",
                        lambda request, images, llm, **kw: {"summary": "a rib", "parts": []})
    seen = {}

    def fake_plan(request, llm, detail="balanced", **kw):
        seen["request"] = request
        seen.update(kw)
        raise RuntimeError("stop after planning")

    monkeypatch.setattr(planner_mod, "plan", fake_plan)
    state = _state(tmp_path, graph or _graph(), "add a rib in SHAPER")
    state["history"] = [{"event": "turn", "turn": 0, "text": "a box in GEOM",
                         "status": "ok", "summary": "box"}]
    if images:
        state["reference_images"] = ["data:image/png;base64,AAAA"]
    cfg = {"configurable": {"thread_id": uuid.uuid4().hex, "llm_client": None,
                            "executor_client": None}}
    seen["out"] = build_graph(batch=True).invoke(state, config=cfg)
    return seen


def _shaper_graph():
    g = BuildGraph()
    g.requests.append("a plate in SHAPER")
    b = g.add_block(0, "stage 1/1: shaper", "shaper", ["extrusion_1"])
    g.append(b, "extrude", {"base": "face_1", "size": 5},
             [{"fn": "addExtrusion", "args": [], "out_vars": ["extrusion_1"],
               "publish": ["extrusion_1"]}], 0, module="shaper")
    return g


def test_a_shaper_follow_up_on_a_geom_study_is_answered_without_planning(tmp_path,
                                                                         monkeypatch):
    seen = _shaper_follow_up_seen(tmp_path, monkeypatch)
    assert "request" not in seen
    question = seen["out"]["turn_outcome"]["question"]
    assert question.startswith("The objects in the study (box_1) were built in GEOM")
    assert seen["out"]["turn_outcome"]["summary"] is None


def test_a_shaper_follow_up_on_a_shaper_study_is_planned(tmp_path, monkeypatch):
    seen = _shaper_follow_up_seen(tmp_path, monkeypatch, graph=_shaper_graph())
    assert "extrusion_1" in seen["request"]


def test_a_shaper_follow_up_is_planned_with_the_study_and_earlier_requests(tmp_path,
                                                                          monkeypatch):
    seen = _shaper_follow_up_seen(tmp_path, monkeypatch, graph=_shaper_graph())
    assert seen["request"].startswith("add a rib in SHAPER")
    assert "The study already contains: extrusion_1." in seen["request"]
    assert "a box in GEOM" in seen["request"]
    assert seen["pin_from"] == "add a rib in SHAPER"


def test_a_shaper_follow_up_with_images_is_described_then_planned(tmp_path, monkeypatch):
    seen = _shaper_follow_up_seen(tmp_path, monkeypatch, images=True, graph=_shaper_graph())
    assert "extrusion_1" in seen["request"]
    assert seen["image_spec"]["summary"] == "a rib"


def test_the_geom_refusal_names_objects_without_the_run_prefix():
    from salome_agent.graph import _shaper_on_geom
    g = BuildGraph()
    g.add_block(0, "stage 1/1: geom", "geom", ["sa_r1_box_1"])
    state = {"turn_idx": 1, "user_request": "add a rib in SHAPER", "build_graph": g.to_dict(),
             "session": {"name_prefix": "sa_r1_"}}
    assert "(box_1)" in _shaper_on_geom(state)
