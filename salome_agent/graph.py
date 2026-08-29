def route_after_run_stage(state: dict) -> str:
    stage = state["stages"][state["stage_idx"]]
    if stage["module"] == "geom":
        if stage["status"] == "ok":
            return "advance"
        return "retry_gate"
    return "verify_static"


def route_after_verify_exec(state: dict, session_died: bool = False) -> str:
    if session_died:
        return "assemble"
    return "advance"


def build_graph(batch: bool):
    from langgraph.graph import StateGraph, START
    from langgraph.checkpoint.memory import MemorySaver
    from salome_agent.state import RunState
    from salome_agent.nodes.gates import plan_gate, retry_gate

    g = StateGraph(RunState)
    g.add_node("plan_gate", plan_gate)
    g.add_node("retry_gate", retry_gate)
    g.add_conditional_edges("run_stage", route_after_run_stage,
                             {"advance": "advance", "retry_gate": "retry_gate",
                              "verify_static": "verify_static"})
    g.add_edge(START, "plan")
    return g.compile(checkpointer=MemorySaver())
