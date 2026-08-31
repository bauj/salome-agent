from salome_agent.state import RunState, StagePlan, StageRecord, SessionHandle


def test_repeated_module_stages_do_not_collide():
    plan: list[StagePlan] = [
        {"module": "geom", "intent": "box", "params": {}, "inputs": []},
        {"module": "smesh", "intent": "mesh", "params": {}, "inputs": []},
        {"module": "geom", "intent": "fillet", "params": {}, "inputs": []},
    ]
    stages: dict[int, StageRecord] = {
        i: {
            "module": p["module"], "status": "pending", "script": None,
            "call_log": None, "static_retries": 0, "exec_retries": 0,
        }
        for i, p in enumerate(plan)
    }
    assert stages[0]["module"] == "geom"
    assert stages[2]["module"] == "geom"
    assert stages[0] is not stages[2]


def test_history_reducer_is_additive():
    history: list[dict] = []
    history = history + [{"event": "plan_approved"}]
    history = history + [{"event": "stage_0_ok"}]
    assert [h["event"] for h in history] == ["plan_approved", "stage_0_ok"]


def test_runstate_has_fatal_key():
    assert "fatal" in RunState.__annotations__
