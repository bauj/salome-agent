import json

from salome_agent.executor_client import SessionDied, SessionPoisoned
from salome_agent.llm import assistant_tool_turn, continuity_fields
from salome_agent.logs import log
from salome_agent.modules import ModuleSpec
from salome_agent.specialists.tool_selection import initial_offered, resolve_load

# strict: nudge once every N risky ops with no intervening measure.
_STRICT_NUDGE_EVERY = 3

_LOAD_TOOLS_DESCRIPTION = (
    "Add geom tools to your available set for the rest of this stage. Pass "
    "tool `names`, a list of `categories`, or both. Use it whenever the tool "
    "you need is not in your current list."
)


def _call_key(name: str, args: dict) -> tuple:
    return (name, json.dumps(args, sort_keys=True, default=str))


class ToolSpecialist:
    """One tool-calling loop for every module.

    Everything that differs between modules is in `spec`; every geom-only
    lever goes inert on an empty spec field rather than behind a branch.
    """

    def __init__(self, spec: ModuleSpec):
        self.spec = spec

    def _load_tools_def(self) -> dict | None:
        if not self.spec.categories:
            return None
        return {"name": "load_tools", "description": _LOAD_TOOLS_DESCRIPTION,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "names": {"type": "array", "items": {"type": "string"},
                                  "description": "specific geom tool names to add"},
                        "categories": {"type": "array", "items": {
                            "type": "string",
                            "enum": list(self.spec.categories)},
                            "description": "whole tool categories to add"},
                    }}}

    def run(self, stage_plan, llm, client, iteration_cap, *, blocks=(),
            tool_selection="hints", verify_cadence="final", retry=None,
            seed_call_log=None, seed_calls=None, cancel=None, upstream=()) -> dict:
        try:
            return self._run(stage_plan, llm, client, iteration_cap, blocks,
                             tool_selection, verify_cadence, retry, seed_call_log, seed_calls,
                             cancel, frozenset(upstream))
        except (SessionDied, SessionPoisoned) as e:
            return {"fatal": f"SALOME agent server unavailable: {e}"}

    def _run(self, stage_plan, llm, client, iteration_cap, blocks,
             tool_selection, verify_cadence, retry, seed_call_log, seed_calls,
             cancel, upstream) -> dict:
        spec = self.spec
        parts = [str(stage_plan)] + [b for b in blocks if b]
        if retry and spec.render_retry:
            rendered = spec.render_retry(retry)
            if rendered is not None:
                parts.append(rendered)
        messages = [{"role": "system", "content": spec.prompt(verify_cadence)},
                    {"role": "user", "content": "\n\n".join(parts)}]
        # A retry seeds call_log with the failed attempt's records so a shape the
        # model reuses by name (rather than rebuilding) stays traceable by
        # codegen's reachability pass instead of silently dropping out of
        # result.py (GH #38).
        call_log: list[dict] = list(seed_call_log) if seed_call_log else []
        calls: list[dict] = list(seed_calls) if seed_calls else []
        terms = {spec.terminator.name: spec.terminator,
                 **{t.name: t for t in spec.extra_terminators}}
        extra_defs = [{"name": t.name, "parameters": t.schema, "description": t.description}
                      for t in spec.extra_terminators]
        last_error = None
        # Stage-local record of calls that already failed: hash((tool, args)) ->
        # error. A byte-identical repeat is answered here without a round-trip to
        # the server, and the model is told it is repeating itself (seen live:
        # weak models loop 4x on the same failing call, inventing a new wrong
        # diagnosis each time — task 215). Reset per stage.
        failed_calls: dict[tuple, str] = {}
        # policy.verify_cadence bookkeeping: count placement-risky ops (boolean /
        # transform) since the last measure/check_shape. "final" nudges once at
        # finish_stage if any are outstanding; "strict" also nudges mid-loop every
        # _STRICT_NUDGE_EVERY of them.
        risky_since_measure = 0
        nudged_final = False
        # Per-stage tool retrieval: offer CORE_TOOLS + the planner's hints, not the
        # whole registry (which was ~66% of a run's input tokens). The model widens
        # its set on demand via load_tools. "all" restores the send-everything
        # behaviour. finish_stage/load_tools are always appended, never in `offered`.
        all_defs = spec.tools_for(client)
        finish_def = {"name": spec.terminator.name, "parameters": spec.terminator.schema,
                      "description": spec.terminator.description}
        load_def = self._load_tools_def()
        if tool_selection == "all" or not spec.core_tools:
            offered = set(all_defs)
        else:
            offered = initial_offered(stage_plan, set(all_defs))
        log.info("  %s stage: offered %d/%d tools (%s)", spec.name,
                 len(offered), len(all_defs), ", ".join(sorted(offered)))
        # Cancel is checked here, not only between graph nodes (task 256): the
        # stage ends like an exhausted one, carrying the calls already made, so
        # the caller's rollback can undo them.
        def cancelled() -> bool:
            if cancel is not None and cancel.is_set():
                log.info("  %s stage: cancelled", spec.name)
                return True
            return False

        for i in range(iteration_cap):
            if cancelled():
                return _with_calls(spec.on_exhausted(call_log, last_error), calls)
            log.debug("    %s iter %d/%d: asking LLM...", spec.name, i + 1, iteration_cap)
            tools = ([all_defs[n] for n in sorted(offered) if n in all_defs]
                     + [finish_def] + extra_defs
                     + ([load_def] if load_def is not None else []))
            # Tool results are stored already compacted (see below), so `messages`
            # is sent verbatim: each turn's request extends the previous one byte
            # for byte, which keeps a gateway's prefix/KV cache alive. `load_tools`
            # widening `tools` mid-stage is the one remaining prefix break.
            # A shallow copy per call: the message dicts are never mutated after
            # append, so the copy is byte-identical content, and each request is a
            # verbatim extension of the last.
            response = llm.chat(messages=list(messages), tools=tools,
                                module=stage_plan.get("module"))
            if response.get("content"):
                log.debug("    LLM reasoning: %s", response["content"])
            tool_calls = response["tool_calls"]
            log.debug("    LLM -> %s", [(c["name"], c["args"]) for c in tool_calls])
            if not tool_calls:
                # tool_choice="auto" — the model replied with prose and no call.
                # Keep its message, nudge it, and let the loop try again.
                messages.append({"role": "assistant", "content": response["content"] or "(no reply)",
                                 **continuity_fields(response.get("raw_message"))})
                messages.append({"role": "user", "content": spec.no_tool_nudge})
                continue
            # The assistant's own tool-call turn must be echoed back before any
            # tool result — without it the model has no memory of what it just
            # did (each turn looks like a fresh start) and, seen live against a
            # real LLM, it just calls make_box forever instead of ever finishing.
            # The model may batch several independent calls in one turn; every
            # one gets executed and answered below, in order.
            for j, c in enumerate(tool_calls):
                c.setdefault("id", None)
                c["id"] = c["id"] or f"call_{i}_{j}"
            messages.append(assistant_tool_turn(tool_calls, response.get("raw_message")))
            batch_error = None
            finish_call = None
            for k, c in enumerate(tool_calls):
                if cancelled():
                    return _with_calls(spec.on_exhausted(call_log, batch_error), calls)
                name, args, call_id = c["name"], c["args"], c["id"]
                if name in terms:
                    finish_call = (call_id, args, terms[name])
                    # Anything the model batched after finish_stage still needs a
                    # tool response for its id, or the next request is malformed.
                    for skipped in tool_calls[k + 1:]:
                        messages.append({"role": "tool", "tool_call_id": skipped["id"],
                                         "content": "skipped — finish_stage was called"})
                    break
                if name == "load_tools" and load_def is not None:
                    added = resolve_load(args.get("names"), args.get("categories"), all_defs)
                    new = added - offered
                    offered |= added
                    obs = (f"added: {', '.join(sorted(new)) or '(nothing new)'} "
                           f"— {len(offered)} tools available")
                    log.info("  load_tools(names=%s, categories=%s) -> +%d, %d available",
                             args.get("names"), args.get("categories"), len(new), len(offered))
                    messages.append({"role": "tool", "tool_call_id": call_id, "content": obs})
                    continue
                key = _call_key(name, args)
                if key in failed_calls:
                    log.info("  repeat of a call that already failed (%s) — answered "
                             "locally, no round-trip", name)
                    result = {"ok": False, "repeat": True, "error": (
                        f"identical call already failed with: {failed_calls[key]}. "
                        "Do not repeat it — change the arguments or the approach, or "
                        "gather information first (measure / get_subshapes / check_shape).")}
                    batch_error = {"phase": "exec", "message": failed_calls[key],
                                   "offending_line": None, "traceback": None}
                    messages.append({"role": "tool", "tool_call_id": call_id,
                                     "content": spec.render_result(result)})
                    continue
                result = spec.execute(client, name, args)
                if result["ok"]:
                    call_log.extend(result.get("records") or [])
                    if result.get("records"):
                        calls.append({"tool": name, "tool_args": args,
                                      "records": list(result["records"])})
                    log.debug("    tool %s ok", name)
                    if spec.risky_categories:
                        if name in spec.measure_tools:
                            risky_since_measure = 0
                        elif all_defs.get(name, {}).get("category") in spec.risky_categories:
                            risky_since_measure += 1
                else:
                    failed_calls[key] = result["error"]
                    batch_error = {"phase": "exec", "message": result["error"],
                                   "offending_line": None, "traceback": None}
                    log.debug("    tool %s failed: %s", name, result["error"])
                content = spec.render_result(result)
                if result["ok"]:
                    content += _upstream_change_note(result.get("records"), upstream)
                messages.append({"role": "tool", "tool_call_id": call_id,
                                 "content": content})
            # A later ok call in the same batch must not mask an earlier failure —
            # settle last_error once per turn from the batch outcome.
            last_error = batch_error

            if finish_call is not None:
                call_id, args, term = finish_call
                if (term is spec.terminator and spec.measure_tools
                        and verify_cadence in ("final", "strict")
                        and risky_since_measure > 0 and not nudged_final):
                    nudged_final = True
                    log.info("  verify_cadence=%s: nudging a measure before finish_stage",
                             verify_cadence)
                    messages.append({"role": "tool", "tool_call_id": call_id, "content":
                        "You have moved or combined shapes since your last measure. "
                        "Call `measure` on your result shape(s) and confirm their "
                        "placement, then call finish_stage again."})
                    continue
                return _with_calls(term.handle(args, call_log), calls)

            if (spec.risky_categories and verify_cadence == "strict"
                    and risky_since_measure >= _STRICT_NUDGE_EVERY):
                log.info("  verify_cadence=strict: nudging a measure (%d risky ops)",
                         risky_since_measure)
                messages.append({"role": "user", "content":
                    f"You have made {risky_since_measure} shapes since your last "
                    "measure. Call `measure` on your recent shapes and confirm "
                    "their placement before continuing."})
                risky_since_measure = 0
        return _with_calls(spec.on_exhausted(call_log, last_error), calls)


def _upstream_change_note(records, upstream: frozenset) -> str:
    """One sentence when a call changed an earlier stage's shape in place (a
    record with a receiver and no new shape, e.g. set_color). Copies that only
    read an upstream shape are the normal geom->geom handoff and stay silent.
    Reported, never blocked: the prompt already says "don't", this lands at the
    moment it happens (seen live: a beam from stage 1 painted with stage 3's
    walls — task 216)."""
    hit = sorted({r["recv"] for r in records or ()
                  if not r.get("out_vars") and r.get("recv") in upstream})
    if not hit:
        return ""
    names = ", ".join(hit)
    log.info("  cross-stage change: %s", names)
    return (f"\nnote: {names} was built in an earlier stage and you just changed it "
            "in place — do that only if this stage's own task asks for it.")


def _with_calls(result: dict, calls: list) -> dict:
    if "call_log" in result:
        return {**result, "calls": calls}
    return result
