from langgraph.types import Command, interrupt


def plan_gate(state: dict) -> Command:
    if state["config"].get("batch"):
        return Command(update={"plan_approved": True}, goto="run_stage")
    decision = interrupt({"plan": state["plan"]})
    if decision["action"] == "approve":
        return Command(update={"plan_approved": True}, goto="run_stage")
    if decision["action"] == "edit":
        return Command(update={"plan": decision["plan"]}, goto="plan")
    return Command(goto="assemble")


def retry_gate(state: dict) -> Command:
    if state["config"].get("batch"):
        policy = state["config"]["policy"]["on_stage_failure"]
        if policy == "continue":
            return Command(goto="skip")
        return Command(goto="assemble")
    decision = interrupt({"last_error": state["last_error"]})
    if decision["action"] == "hint":
        return Command(update={"history": [{"event": "retry_hint", "hint": decision["hint"]}]},
                        goto="run_stage")
    if decision["action"] == "skip":
        return Command(goto="skip")
    return Command(goto="assemble")
