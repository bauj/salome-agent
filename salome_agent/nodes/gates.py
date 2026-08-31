from langgraph.graph import END
from langgraph.types import Command, interrupt

from salome_agent.logs import log


def plan_gate(state: dict) -> Command:
    if state["config"].get("batch"):
        log.info("plan auto-approved (--batch)")
        return Command(update={"plan_approved": True}, goto="run_stage")
    decision = interrupt({"plan": state["plan"]})
    if decision["action"] == "approve":
        return Command(update={"plan_approved": True}, goto="run_stage")
    if decision["action"] == "edit":
        from salome_agent.nodes.planner import blank_stages

        new_plan = decision["plan"]
        # Re-prompt with the edited plan — NOT goto="plan", which would
        # re-run the planner and throw the edit away. stages must be rebuilt
        # since the edit can add or drop stages.
        return Command(update={"plan": new_plan, "stages": blank_stages(new_plan),
                               "stage_idx": 0}, goto="plan_gate")
    # User aborted at the plan gate: stop now, do NOT run assemble/report
    # (which would write result.py / report.md for a run that produced nothing).
    log.info("plan rejected - run aborted, nothing written")
    return Command(update={"aborted": True}, goto=END)


def retry_gate(state: dict) -> Command:
    if state["config"].get("batch"):
        policy = state["config"]["policy"]["on_stage_failure"]
        log.warning("stage failed, on_stage_failure=%s: %s", policy,
                    (state.get("last_error") or {}).get("message"))
        if policy == "continue":
            return Command(goto="skip")
        return Command(goto="assemble")
    decision = interrupt({"last_error": state["last_error"]})
    if decision["action"] == "hint":
        return Command(update={"history": [{"event": "retry_hint", "hint": decision["hint"]}]},
                        goto="run_stage")
    if decision["action"] == "skip":
        return Command(goto="skip")
    log.info("run aborted at the retry gate - nothing written")
    return Command(update={"aborted": True}, goto=END)
