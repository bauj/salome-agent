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


def test_amend_turn_rewrites_result_and_graph_json(tmp_path, monkeypatch):
    from salome_agent.specialists import amend as amend_mod

    def fake_run_amend(text, graph, llm, client, turn, policy, *, history=(), selection=None):
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

    def fake_plan(request, llm, detail="balanced"):
        seen["request"] = request
        raise RuntimeError("stop after planning")   # _plan_node turns this into a clean fatal

    monkeypatch.setattr(planner_mod, "plan", fake_plan)
    app = build_graph(batch=True)
    cfg = {"configurable": {"thread_id": uuid.uuid4().hex, "llm_client": None,
                            "executor_client": None}}
    app.invoke(_state(tmp_path, _graph(), "add a wing"), config=cfg)
    assert seen["request"].startswith("add a wing")
    assert "box_1" in seen["request"]
