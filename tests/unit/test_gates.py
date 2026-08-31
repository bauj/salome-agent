import salome_agent.nodes.gates as gates
from langgraph.graph import END


def _patch_interrupt(monkeypatch, decision):
    monkeypatch.setattr(gates, "interrupt", lambda payload: decision)


def test_plan_gate_abort_routes_to_end_without_assembling(monkeypatch):
    _patch_interrupt(monkeypatch, {"action": "abort"})
    cmd = gates.plan_gate({"config": {}, "plan": [], "stages": {}})
    assert cmd.goto == END
    assert cmd.update == {"aborted": True}


def test_retry_gate_abort_routes_to_end_without_assembling(monkeypatch):
    _patch_interrupt(monkeypatch, {"action": "abort"})
    cmd = gates.retry_gate({"config": {}, "last_error": {"message": "boom"}})
    assert cmd.goto == END
    assert cmd.update == {"aborted": True}


def test_plan_gate_approve_still_runs_stage(monkeypatch):
    _patch_interrupt(monkeypatch, {"action": "approve"})
    cmd = gates.plan_gate({"config": {}, "plan": [], "stages": {}})
    assert cmd.goto == "run_stage"
    assert cmd.update == {"plan_approved": True}


def test_retry_gate_skip_still_skips(monkeypatch):
    _patch_interrupt(monkeypatch, {"action": "skip"})
    cmd = gates.retry_gate({"config": {}, "last_error": {"message": "boom"}})
    assert cmd.goto == "skip"


def test_batch_retry_gate_abort_still_assembles_a_report():
    # --batch on_stage_failure=abort is a real failure, not a user abort:
    # it must still route through assemble/report.
    state = {"config": {"batch": True, "policy": {"on_stage_failure": "abort"}},
             "last_error": {"message": "boom"}}
    cmd = gates.retry_gate(state)
    assert cmd.goto == "assemble"
