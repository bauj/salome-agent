import json
import time

import requests
from langchain_core.runnables import RunnableConfig

from salome_agent.executor_client import ServerOutdated
from salome_agent.llm import (ReasoningTimeExceeded, RepetitionLoopDetected, StreamBudgetExceeded,
                              StreamErrorFrame)
from salome_agent.logs import log


def _usage_snap(llm) -> dict | None:
    """The LLM client's running counters, or None if it has no real usage
    dict (a bare MagicMock in tests) — timing is then skipped."""
    u = getattr(llm, "usage", None)
    if not isinstance(u, dict):
        return None
    return {"calls": u.get("calls", 0), "prompt_tokens": u.get("prompt_tokens", 0),
            "completion_tokens": u.get("completion_tokens", 0),
            "llm_seconds": u.get("llm_seconds", 0.0)}


def _timing_delta(before: dict, after: dict, wall: float) -> dict:
    llm_s = after["llm_seconds"] - before["llm_seconds"]
    return {"wall": wall, "llm_s": llm_s, "core_s": wall - llm_s,
            "turns": after["calls"] - before["calls"],
            "in_tok": after["prompt_tokens"] - before["prompt_tokens"],
            "out_tok": after["completion_tokens"] - before["completion_tokens"]}


def _accum_timing(existing: dict | None, delta: dict) -> dict:
    """Merge a timing delta into a stage's running timing (retries re-enter
    the same node, so numbers accumulate)."""
    t = dict(existing or {})
    for k, v in delta.items():
        t[k] = round((t.get(k) or 0) + v, 2)
    return t


def _body_snippet(resp, limit: int = 300) -> str:
    try:
        text = " ".join((resp.text or "").split())
    except Exception:  # noqa: BLE001 — body already consumed or unreadable
        return ""
    return text if len(text) <= limit else text[:limit] + "…"


def _llm_endpoint_fatal(e: Exception) -> str | None:
    """A terminal LLM-endpoint failure (gateway 5xx after retries, connection
    loss, non-JSON body) — return a one-line abort message so the run ends
    cleanly with a report instead of a raw traceback escaping cli.main()
    (GH #31). Returns None for any other exception, which keeps propagating."""
    if isinstance(e, requests.HTTPError):
        code = getattr(e.response, "status_code", "?")
        if code == 429:
            return ("LLM endpoint rate-limited (HTTP 429) after retries — quota "
                    "or requests-per-minute exceeded; wait or use a higher-tier key")
        if isinstance(code, int) and 400 <= code < 500:
            body = _body_snippet(e.response)
            return (f"LLM endpoint rejected the request (HTTP {code}) — check "
                    f"base_url, model name, API key and request parameters"
                    + (f"; endpoint said: {body}" if body else ""))
        return (f"LLM endpoint unavailable (HTTP {code}) — the model backend is "
                f"down or slow; retry later")
    if isinstance(e, StreamBudgetExceeded):
        return ("LLM call exceeded llm.timeout total budget — the model kept "
                "generating (likely a runaway reasoning loop); raise llm.timeout "
                "or lower the reasoning effort")
    if isinstance(e, ReasoningTimeExceeded):
        return (f"LLM kept reasoning without answering, even after being told to "
                f"decide ({e}) — use a faster model for this step or lower its "
                f"reasoning effort")
    if isinstance(e, RepetitionLoopDetected):
        return (f"LLM got stuck repeating itself ({e}) — this model loops on this "
                f"step; try another model or reword the request")
    if isinstance(e, StreamErrorFrame):
        return f"LLM endpoint error, still failing after retries — {e}"
    if isinstance(e, (requests.ConnectionError, requests.Timeout)):
        return f"LLM endpoint unreachable ({type(e).__name__}) — retry later"
    if isinstance(e, ServerOutdated):
        return str(e)
    if isinstance(e, RuntimeError):
        return f"LLM endpoint error — {e}"
    return None


def route_start(state: dict) -> str:
    """"plan" goes through describe_image first, so a follow-up turn that
    brings reference images is routed there too; describe_image hands it on to
    amend."""
    if state.get("reference_images"):
        return "plan"
    return _follow_up(state)


def _follow_up(state: dict) -> str:
    """A later turn is amended, except a SHAPER request on a GEOM study: the
    plan node answers that one with why it cannot."""
    if (state.get("turn_idx") or 0) == 0:
        return "plan"
    return "plan" if _cross_module(state) else "amend"


def _amend_module(state: dict, graph) -> str:
    """The module the request names, else the one that built the study's latest
    CAD stage, else SHAPER."""
    from salome_agent.nodes.planner import named_cad_module

    named = named_cad_module(state.get("user_request") or "")
    if named:
        return named
    for b in reversed(graph.blocks):
        if b.module in ("geom", "shaper"):
            return b.module
    return "shaper"


def _graph_upstream(state: dict) -> list[str]:
    from salome_agent.build_graph import BuildGraph

    if not state.get("build_graph"):
        return []
    return BuildGraph.from_dict(state["build_graph"]).result_names()


def _request_block(state: dict) -> str:
    """The planner's brief can drop what the user wrote; the specialist reads the
    original text too, but builds only its own stage."""
    request = (state.get("user_request") or "").strip()
    if not request:
        return ""
    others = (" Other stages build the rest; do not build their parts."
              if len(state.get("plan") or []) > 1 else "")
    return ("The user's request, as written. Your job is the stage above; use this "
            f"for any detail the stage leaves out.{others}\n" + request)


def _earlier_requests(state: dict, last: int = 5) -> str:
    """A follow-up such as "retry" means nothing to the planner on its own."""
    turns = [h for h in state.get("history") or [] if h.get("event") == "turn"][-last:]
    if not turns:
        return ""
    lines = [f"- {t.get('text', '')}" + ("" if t.get("status") == "ok" else " (did not finish)")
             for t in turns]
    return "\n\nEarlier requests in this conversation, oldest first:\n" + "\n".join(lines)


def _replan_context(state: dict, graph, reason: str = "") -> str:
    names = ", ".join(graph.result_names()) or "(nothing yet)"
    return (f"The study already contains: {names}. Build on those objects by name; "
            f"do not rebuild them." + (f" ({reason})" if reason else "")
            + _earlier_requests(state))


def _amend_node(state: dict, config: "RunnableConfig"):
    from langgraph.types import Command
    from salome_agent.build_graph import BuildGraph
    from salome_agent.replay import RollbackFailed
    from salome_agent.specialists import amend

    llm = config["configurable"]["llm_client"]
    client = config["configurable"].get("executor_client")
    graph = BuildGraph.from_dict(state.get("build_graph"))
    turn = state.get("turn_idx") or 0
    log.info("amend turn %d: %s", turn + 1, state["user_request"])
    # Created here, not in run_amend: edits already applied to the study must
    # still reach the conversation's rollback when the LLM call fails mid-loop.
    session = amend.AmendSession(graph, client, turn, _amend_module(state, graph))
    extra = {}
    if state.get("image_spec"):
        from salome_agent.nodes.vision_spec import amend_block
        extra = {"image_description": amend_block(state["image_spec"])}
    try:
        session, outcome = amend.run_amend(
            state["user_request"], graph, llm, client, turn,
            state["config"]["policy"], history=state.get("history") or [],
            selection=state.get("selection"), session=session,
            cancel=config["configurable"].get("cancel"), **extra)
    except RollbackFailed as e:
        log.error("  %s", e)
        return Command(update={"fatal": f"rollback failed: {e}", "degraded": True,
                               "build_graph": e.graph.to_dict()}, goto="assemble")
    except Exception as e:  # noqa: BLE001
        msg = _llm_endpoint_fatal(e)
        if msg is None:
            raise
        log.error("  %s", msg)
        return Command(update={"fatal": msg, "build_graph": session.graph.to_dict()},
                       goto="assemble")
    update = {"build_graph": session.graph.to_dict(), "turn_outcome": outcome,
              "history": session.events}
    if "fatal" in outcome:
        update["fatal"] = outcome["fatal"]
        return Command(update=update, goto="assemble")
    if "replan" in outcome:
        update["replan_context"] = _replan_context(state, session.graph, outcome["replan"])
        log.info("  amend -> replan: %s", outcome["replan"])
        return Command(update=update, goto="plan")
    return Command(update=update, goto="assemble")


def route_after_run_stage(state: dict) -> str:
    from salome_agent.modules import spec_for

    stage = state["stages"][state["stage_idx"]]
    if spec_for(stage["module"]).call_prefix:
        if stage["status"] == "ok":
            return "advance"
        return "retry_gate"
    return "verify_static"


def route_after_verify_exec(state: dict, session_died: bool = False) -> str:
    if session_died:
        return "assemble"
    return "advance"


def select_upstream(stage_plan: dict, stages: dict, idx: int) -> list[str]:
    """Names of study objects to hand the specialist for this stage: everything
    published by a prior accepted stage.

    `stage_plan["inputs"]` is deliberately not consulted — planner.py tells the
    planner to put plain words there, never object names (it cannot know the
    auto-generated ones), so the scoping branch this replaced never fired."""
    available: list[str] = []
    for j in range(idx):
        prior = stages.get(j)
        if prior and prior.get("status") == "ok":
            available += prior.get("result_names") or []
    return available


def retry_context(state: dict, idx: int, client) -> dict | None:
    """What a tool stage's PREVIOUS attempt left behind, for the retry prompt
    (GH #38). None on a first attempt. The failed attempt's call_log is
    reused as-is (both here, for the built-shapes list, and by the caller as
    the new attempt's seed_call_log) — retry_gate only ever re-enters
    run_stage for the same stage right after appending at most one
    retry_hint event, so the LAST such event in history is always the one
    meant for this retry."""
    stage = state["stages"].get(idx)
    if not stage or stage.get("status") != "failed":
        return None
    hint = None
    for event in reversed(state.get("history") or []):
        if event.get("event") == "retry_hint":
            hint = event.get("hint")
            break
    call_log = stage.get("call_log") or []
    built = [n for call in call_log for n in (call.get("publish") or [])]
    built_info = None
    if built and client is not None:
        module = stage.get("module") or "geom"
        extra = {} if module == "geom" else {"module": module}
        try:
            m = client.call_tool("measure", {"names": built}, **extra)
            if m.get("ok"):
                built_info = m.get("measurements")
            else:
                log.warning("  measure(retry built) failed: %s", m.get("error"))
        except Exception as e:  # noqa: BLE001 — best-effort, must not abort the retry
            log.warning("  measure(retry built) errored: %s", e)
    return {"last_error": state.get("last_error"), "hint": hint, "built": built,
            "built_info": built_info, "call_log": call_log, "calls": stage.get("calls") or []}


# llm_client/executor_client are live, non-serializable objects — they live
# in the graph's runnable config["configurable"] (never checkpointed), not
# in RunState itself. Putting them in state made MemorySaver.put() try to
# msgpack them and fail with "Type is not msgpack serializable: MagicMock"
# the first time a live end-to-end run actually exercised the checkpointer.

def _progress_writer():
    """A node's live progress, out through runner.drive's custom stream; None
    when the node runs outside a graph run (called directly)."""
    from langgraph.config import get_stream_writer

    try:
        write = get_stream_writer()
    except RuntimeError:
        return None
    return lambda kind, **data: write({"kind": kind, "data": data})


def _describe_image_node(state: dict, config: "RunnableConfig"):
    from langgraph.types import Command
    from salome_agent.nodes.vision import describe_images

    images = state.get("reference_images")
    after = _follow_up(state)
    if not images or _cross_module(state):
        return Command(goto=after)
    llm = config["configurable"]["llm_client"]
    retries = ((state.get("config") or {}).get("llm") or {}).get("vision_fix_retries", 2)
    log.info("describing %d reference image(s)...", len(images))
    snap0 = _usage_snap(llm)
    t0 = time.monotonic()
    try:
        spec = describe_images(state["user_request"], images, llm, fix_retries=retries,
                               on_progress=_progress_writer())
    except Exception as e:  # noqa: BLE001
        msg = _llm_endpoint_fatal(e)
        if msg is None:
            raise
        log.error("%s", msg)
        return Command(update={"fatal": msg}, goto="assemble")
    log.info("image spec:\n%s", json.dumps(spec, ensure_ascii=False, indent=1))
    # only this node reads the images; the checkpointer would otherwise carry
    # the base64 through every later step
    update = {"image_spec": spec, "reference_images": None}
    if snap0 is not None:
        vt = _timing_delta(snap0, _usage_snap(llm), time.monotonic() - t0)
        log.info("  vision timing: wall=%.1fs llm=%.1fs", vt["wall"], vt["llm_s"])
    return Command(update=update, goto=after)


def _cross_module(state: dict) -> str | None:
    """GEOM and SHAPER cannot use each other's objects, so a later request that
    names one on a study the other built is answered without planning."""
    from salome_agent.build_graph import BuildGraph
    from salome_agent.nodes.planner import named_cad_module

    named = named_cad_module(state.get("user_request") or "")
    if (state.get("turn_idx") or 0) == 0 or named not in ("geom", "shaper"):
        return None
    other = "shaper" if named == "geom" else "geom"
    graph = BuildGraph.from_dict(state.get("build_graph"))
    prefix = (state.get("session") or {}).get("name_prefix") or ""
    names = [n.removeprefix(prefix) if prefix else n
             for b in graph.blocks if b.module == other
             for n in b.result_names or
             [p for node in graph.block_nodes(b.id) for p in node.published]]
    if not names:
        return None
    built, asked = other.upper(), named.upper()
    return (f"The objects in the study ({', '.join(names)}) were built in {built}, and "
            f"{asked} cannot use {built} objects. Ask for this change in {built}, or "
            f"start a new conversation to build it in {asked}.")


def _plan_node(state: dict, config: "RunnableConfig"):
    from langgraph.types import Command
    from salome_agent.nodes.planner import blank_stages, plan

    refusal = _cross_module(state)
    if refusal:
        log.info("  %s", refusal)
        return Command(update={"plan": [], "stages": {}, "stage_idx": 0,
                               "turn_outcome": {"question": refusal, "summary": None}},
                       goto="assemble")
    llm = config["configurable"]["llm_client"]
    detail = state["config"]["policy"].get("plan_detail", "balanced")
    log.info("planning... (plan_detail=%s)", detail)
    snap0 = _usage_snap(llm)
    t0 = time.monotonic()
    request = state["user_request"]
    if state.get("replan_context"):
        request = f"{request}\n\n{state['replan_context']}"
    else:
        # a first turn that was cancelled or failed rolls back, so a follow-up
        # "retry" is planned from scratch and must still see what was asked
        request += _earlier_requests(state)
    try:
        image_spec = state.get("image_spec")
        extra = {"image_spec": image_spec} if image_spec else {}
        stage_plans = plan(request, llm, detail, pin_from=state["user_request"], **extra)
    except Exception as e:  # noqa: BLE001
        msg = _llm_endpoint_fatal(e)
        if msg is None:
            raise
        log.error("%s", msg)
        return Command(update={"fatal": msg}, goto="assemble")
    log.info("plan: %d stage(s)", len(stage_plans))
    if not stage_plans:
        question = ("I could not work out what to build from that. "
                    "What should I make?")
        return Command(update={"plan": [], "stages": {}, "stage_idx": 0,
                               "turn_outcome": {"question": question, "summary": None}},
                       goto="assemble")
    for i, p in enumerate(stage_plans):
        log.info("  stage %d/%d [%s] %s", i + 1, len(stage_plans), p["module"],
                 p.get("intent", ""))
    update = {"plan": stage_plans, "stages": blank_stages(stage_plans), "stage_idx": 0}
    if snap0 is not None:
        pt = {k: round(v, 2) for k, v in
              _timing_delta(snap0, _usage_snap(llm), time.monotonic() - t0).items()}
        update["plan_timing"] = pt
        log.info("  planner timing: wall=%ss llm=%ss (%d call)",
                 pt["wall"], pt["llm_s"], pt["turns"])
    return Command(update=update, goto="plan_gate")


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
    snap0 = _usage_snap(llm)
    t0 = time.monotonic()

    stages = dict(state["stages"])
    stage = dict(stages[idx])
    update: dict = {}

    from salome_agent.modules import MODULES, spec_for
    from salome_agent.specialists.tool_specialist import ToolSpecialist
    import salome_agent.specialists.geom  # noqa: F401 — registers GEOM_SPEC
    import salome_agent.specialists.script_modules  # noqa: F401 — registers the script specs
    import salome_agent.specialists.shaper  # noqa: F401 — registers SHAPER_SPEC

    if module not in MODULES:
        return Command(update={"fatal": f"planner produced unknown module {module!r}"},
                       goto="assemble")

    spec = spec_for(module)
    upstream = _graph_upstream(state) + select_upstream(stage_plan, state["stages"], idx)
    blocks = []
    if spec.render_upstream is not None:
        block = spec.render_upstream(client, upstream)
        if block:
            blocks.append(block)
    if module in ("geom", "shaper"):
        # smesh never sees the image spec: it only feeds CAD construction
        image_spec = state.get("image_spec")
        if image_spec:
            from salome_agent.nodes.vision_spec import parts_block, stage_part_ids
            ids = stage_part_ids(state["plan"], idx, image_spec)
            if ids:
                blocks.append(parts_block(image_spec, ids))
    request_block = _request_block(state)
    if request_block:
        blocks.append(request_block)
    if spec.call_prefix:
        retry = retry_context(state, idx, client)
        seed = (retry or {}).get("call_log")
        seed_calls = (retry or {}).get("calls")
    else:
        retry = {"last_error": state.get("last_error")}
        seed = None
        seed_calls = None
    policy = state["config"]["policy"]
    try:
        result = ToolSpecialist(spec).run(
            stage_plan, llm, client, cap, blocks=tuple(blocks),
            tool_selection=policy.get("tool_selection", "hints"),
            verify_cadence=policy.get("verify_cadence", "final"),
            retry=retry, seed_call_log=seed, seed_calls=seed_calls,
            cancel=config["configurable"].get("cancel"), upstream=upstream)
    except Exception as e:  # noqa: BLE001
        msg = _llm_endpoint_fatal(e)
        if msg is None:
            raise
        log.error("  %s", msg)
        return Command(update={"fatal": msg}, goto="assemble")
    if "fatal" in result:
        log.error("  %s", result["fatal"])
        return Command(update={"fatal": result["fatal"]}, goto="assemble")
    if "script" in result:
        stage["script"] = result["script"]
        if result.get("warning"):
            log.warning("  %s specialist: %s", module, result["warning"])
        log.info("  %s specialist produced a %d-char script", module,
                 len(result["script"] or ""))
    else:
        stage["call_log"] = result.get("call_log")
        stage["calls"] = result.get("calls")
        if "result_names" in result:
            stage["status"] = "ok"
            stage["result_names"] = result["result_names"]
            update["last_error"] = None
            log.info("  %s specialist done: %d tool call(s), %d %s call(s), results=%s",
                     module, len(result.get("calls") or []),
                     len(result.get("call_log") or []), spec.call_prefix.rstrip("."),
                     result["result_names"])
        else:
            stage["status"] = "failed"
            update["last_error"] = result.get("last_error")
            log.warning("  %s specialist failed: %s", module, result.get("last_error"))

    if snap0 is not None:
        stage["timing"] = _accum_timing(
            stage.get("timing"),
            _timing_delta(snap0, _usage_snap(llm), time.monotonic() - t0))
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
    t0 = time.monotonic()
    result = check_exec(stage["script"], stage["module"], client, timeout)
    exec_wall = time.monotonic() - t0

    if result == "session_died":
        log.error("  SALOME session died during exec - assembling what we have")
        return Command(goto="assemble")
    if result == "session_poisoned":
        log.error("  SALOME agent server poisoned by a timeout - aborting run")
        return Command(update={"fatal": "SALOME agent server poisoned by a stage timeout"},
                       goto="assemble")

    stages = dict(state["stages"])
    stage = dict(stage)
    stage["timing"] = _accum_timing(stage.get("timing"), {"exec_s": exec_wall})
    stages[idx] = stage

    if result is None:
        log.info("  exec check: ok")
        return Command(update={"stages": stages}, goto=route_after_verify_exec(state))

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
    tm = (state["stages"][idx].get("timing") or {})
    if tm:
        log.info("  stage %d/%d timing: wall=%ss llm=%ss (%s turns) exec=%ss core=%ss",
                 idx + 1, len(state["plan"]), tm.get("wall", 0), tm.get("llm_s", 0),
                 tm.get("turns", 0), tm.get("exec_s", 0), tm.get("core_s", 0))
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
    from salome_agent.build_graph import BuildGraph
    from salome_agent.nodes.assembler import commit_stages, render_script

    log.info("assembling result.py")
    graph = commit_stages(BuildGraph.from_dict(state.get("build_graph")), state)
    return Command(update={"final_script": render_script(graph, state["run_id"]),
                           "build_graph": graph.to_dict()}, goto="report")


def _report_node(state: dict, config: "RunnableConfig"):
    import json
    import os

    from langgraph.graph import END
    from langgraph.types import Command
    from salome_agent.nodes.report import write_report

    llm = config["configurable"].get("llm_client")
    report_text = write_report(state, usage=getattr(llm, "usage", None))
    work_dir = os.path.expanduser(state["config"].get("work_dir") or ".")
    run_dir = os.path.join(work_dir, state["run_id"] or "run")
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "result.py"), "w") as f:
        f.write(state.get("final_script") or "")
    if state.get("build_graph") is not None:
        with open(os.path.join(run_dir, "build_graph.json"), "w") as f:
            json.dump(state["build_graph"], f, indent=1)
    turn = state.get("turn_idx") or 0
    mode = "a" if turn > 0 else "w"
    with open(os.path.join(run_dir, "report.md"), mode) as f:
        f.write(f"\n\n# Turn {turn + 1}\n\n{report_text}" if turn > 0 else report_text)
    return Command(update={"report": report_text}, goto=END)


def build_graph(batch: bool):
    from langgraph.graph import StateGraph, START
    from langgraph.checkpoint.memory import MemorySaver
    from salome_agent.state import RunState
    from salome_agent.nodes.gates import plan_gate, retry_gate

    g = StateGraph(RunState)
    g.add_node("describe_image", _describe_image_node)
    g.add_node("plan", _plan_node)
    g.add_node("amend", _amend_node)
    g.add_node("plan_gate", plan_gate)
    g.add_node("run_stage", _run_stage_node)
    g.add_node("verify_static", _verify_static_node)
    g.add_node("verify_exec", _verify_exec_node)
    g.add_node("retry_gate", retry_gate)
    g.add_node("advance", _advance_node)
    g.add_node("skip", _skip_node)
    g.add_node("assemble", _assemble_node)
    g.add_node("report", _report_node)
    # a first turn goes through describe_image, which hands over to plan
    # straight away when there is no reference image
    g.add_conditional_edges(START, route_start, {"plan": "describe_image", "amend": "amend"})
    return g.compile(checkpointer=MemorySaver())
