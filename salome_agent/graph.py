from langchain_core.runnables import RunnableConfig

from salome_agent.logs import log


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


def select_upstream(stage_plan: dict, stages: dict, idx: int) -> list[str]:
    """Names of study objects to hand the specialist for this stage. Every
    object published by a prior accepted stage is available; if the plan's
    `inputs` name specific ones that actually exist, scope down to those
    (keeps multi-stage prompts tight). Otherwise pass everything."""
    available: list[str] = []
    for j in range(idx):
        prior = stages.get(j)
        if prior and prior.get("status") == "ok":
            available += prior.get("result_names") or []
    declared = [n for n in (stage_plan.get("inputs") or []) if n in available]
    return declared or available


# llm_client/executor_client are live, non-serializable objects — they live
# in the graph's runnable config["configurable"] (never checkpointed), not
# in RunState itself. Putting them in state made MemorySaver.put() try to
# msgpack them and fail with "Type is not msgpack serializable: MagicMock"
# the first time a live end-to-end run actually exercised the checkpointer.

def _plan_node(state: dict, config: "RunnableConfig"):
    from langgraph.types import Command
    from salome_agent.nodes.planner import blank_stages, plan

    llm = config["configurable"]["llm_client"]
    detail = state["config"]["policy"].get("plan_detail", "balanced")
    log.info("planning... (plan_detail=%s)", detail)
    stage_plans = plan(state["user_request"], llm, detail)
    log.info("plan: %d stage(s)", len(stage_plans))
    for i, p in enumerate(stage_plans):
        log.info("  stage %d/%d [%s] %s", i + 1, len(stage_plans), p["module"],
                 p.get("intent", ""))
    return Command(update={"plan": stage_plans, "stages": blank_stages(stage_plans),
                           "stage_idx": 0},
                   goto="plan_gate")


def _run_stage_node(state: dict, config: "RunnableConfig"):
    from langgraph.types import Command

    idx = state["stage_idx"]
    stage_plan = state["plan"][idx]
    module = stage_plan["module"]
    log.info("stage %d/%d [%s] %s", idx + 1, len(state["plan"]), module,
             stage_plan.get("intent", ""))
    llm = config["configurable"]["llm_client"]
    client = config["configurable"].get("executor_client")
    cap = state["config"]["policy"]["tool_iteration_cap"]

    stages = dict(state["stages"])
    stage = dict(stages[idx])
    update: dict = {}

    if module == "geom":
        from salome_agent.specialists.geom import run_geom_specialist

        upstream_info = None
        upstream = select_upstream(stage_plan, state["stages"], idx)
        if upstream and client is not None:
            try:
                m = client.call_tool("measure", {"names": upstream})
                if m.get("ok"):
                    upstream_info = m.get("measurements")
                else:
                    log.warning("  measure(upstream) failed: %s", m.get("error"))
            except Exception as e:  # noqa: BLE001 — best-effort, must not abort the stage
                log.warning("  measure(upstream) errored: %s", e)
        result = run_geom_specialist(stage_plan, llm, client, cap, upstream=upstream_info)
        if "fatal" in result:
            log.error("  %s", result["fatal"])
            return Command(update={"fatal": result["fatal"]}, goto="assemble")
        stage["call_log"] = result.get("call_log")
        if "result_names" in result:
            stage["status"] = "ok"
            stage["result_names"] = result["result_names"]
            update["last_error"] = None
            log.info("  geom specialist done: %d tool call(s), results=%s",
                     len(result.get("call_log") or []), result["result_names"])
        else:
            stage["status"] = "failed"
            update["last_error"] = result.get("last_error")
            log.warning("  geom specialist failed: %s", result.get("last_error"))
    else:
        from salome_agent.specialists.base import run_specialist
        from salome_agent.specialists.paravis import PARAVIS_SYSTEM_PROMPT
        from salome_agent.specialists.smesh import SMESH_SYSTEM_PROMPT

        system_prompt = {"smesh": SMESH_SYSTEM_PROMPT, "paravis": PARAVIS_SYSTEM_PROMPT}.get(module)
        tools = {}
        if client is not None:
            tools = {
                "api_introspect": lambda name: str(client.introspect([name])),
                "list_study": lambda: str(client.list_study()),
            }
        upstream = select_upstream(stage_plan, state["stages"], idx)
        result = run_specialist(stage_plan, llm, tools, cap, system_prompt=system_prompt,
                                upstream_names=upstream, last_error=state.get("last_error"))
        stage["script"] = result["script"]
        if result.get("warning"):
            log.warning("  %s specialist: %s", module, result["warning"])
        log.info("  %s specialist produced a %d-char script", module, len(result["script"] or ""))

    stages[idx] = stage
    update["stages"] = stages
    goto = route_after_run_stage({**state, "stages": stages})
    return Command(update=update, goto=goto)


def _verify_static_node(state: dict, config: "RunnableConfig"):
    from langgraph.types import Command
    from salome_agent.verify.introspect import build_api_map
    from salome_agent.verify.static import check_static

    idx = state["stage_idx"]
    stage = state["stages"][idx]
    client = config["configurable"].get("executor_client")
    api_map = build_api_map(client)
    err = check_static(stage["script"], api_map)
    if err is None:
        log.info("  static check: ok")
        return Command(goto="verify_exec")

    stages = dict(state["stages"])
    stage = dict(stage)
    stage["static_retries"] += 1
    stages[idx] = stage
    cap = state["config"]["policy"]["static_retries"]
    goto = "run_stage" if stage["static_retries"] <= cap else "retry_gate"
    log.warning("  static check failed (retry %d/%d): %s", stage["static_retries"], cap,
                err.get("message"))
    return Command(update={"stages": stages, "last_error": err}, goto=goto)


def _verify_exec_node(state: dict, config: "RunnableConfig"):
    from langgraph.types import Command
    from salome_agent.verify.exec_check import check_exec

    idx = state["stage_idx"]
    stage = state["stages"][idx]
    client = config["configurable"]["executor_client"]
    timeout = state["config"]["policy"]["exec_timeout_s"]
    log.info("  exec check: running stage script under SALOME (timeout %ss)...", timeout)
    result = check_exec(stage["script"], stage["module"], client, timeout)

    if result == "session_died":
        log.error("  SALOME session died during exec - assembling what we have")
        return Command(goto="assemble")
    if result == "session_poisoned":
        log.error("  SALOME agent server poisoned by a timeout - aborting run")
        return Command(update={"fatal": "SALOME agent server poisoned by a stage timeout"},
                       goto="assemble")
    if result is None:
        log.info("  exec check: ok")
        return Command(goto=route_after_verify_exec(state))

    stages = dict(state["stages"])
    stage = dict(stage)
    stage["exec_retries"] += 1
    stages[idx] = stage
    cap = state["config"]["policy"]["exec_retries"]
    goto = "run_stage" if stage["exec_retries"] <= cap else "retry_gate"
    log.warning("  exec check failed (retry %d/%d): %s", stage["exec_retries"], cap,
                result.get("message"))
    return Command(update={"stages": stages, "last_error": result}, goto=goto)


def _advance_node(state: dict):
    from langgraph.types import Command
    from salome_agent.nodes.advance import advance

    idx = state["stage_idx"]
    result_names = state["stages"][idx].get("result_names")
    update = advance(state, result_names=result_names)
    update["last_error"] = None
    log.info("  stage %d/%d accepted", idx + 1, len(state["plan"]))
    goto = "assemble" if update["stage_idx"] >= len(state["plan"]) else "run_stage"
    return Command(update=update, goto=goto)


def _skip_node(state: dict):
    from langgraph.types import Command
    from salome_agent.nodes.advance import skip

    update = skip(state)
    goto = "assemble" if update["stage_idx"] >= len(state["plan"]) else "run_stage"
    return Command(update=update, goto=goto)


def _assemble_node(state: dict):
    from langgraph.types import Command
    from salome_agent.nodes.assembler import assemble

    log.info("assembling result.py")
    return Command(update={"final_script": assemble(state)}, goto="report")


def _report_node(state: dict):
    import os

    from langgraph.graph import END
    from langgraph.types import Command
    from salome_agent.nodes.report import write_report

    report_text = write_report(state)
    work_dir = os.path.expanduser(state["config"].get("work_dir") or ".")
    run_dir = os.path.join(work_dir, state["run_id"] or "run")
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "result.py"), "w") as f:
        f.write(state.get("final_script") or "")
    with open(os.path.join(run_dir, "report.md"), "w") as f:
        f.write(report_text)
    return Command(update={"report": report_text}, goto=END)


def build_graph(batch: bool):
    from langgraph.graph import StateGraph, START
    from langgraph.checkpoint.memory import MemorySaver
    from salome_agent.state import RunState
    from salome_agent.nodes.gates import plan_gate, retry_gate

    g = StateGraph(RunState)
    g.add_node("plan", _plan_node)
    g.add_node("plan_gate", plan_gate)
    g.add_node("run_stage", _run_stage_node)
    g.add_node("verify_static", _verify_static_node)
    g.add_node("verify_exec", _verify_exec_node)
    g.add_node("retry_gate", retry_gate)
    g.add_node("advance", _advance_node)
    g.add_node("skip", _skip_node)
    g.add_node("assemble", _assemble_node)
    g.add_node("report", _report_node)
    g.add_edge(START, "plan")
    return g.compile(checkpointer=MemorySaver())
