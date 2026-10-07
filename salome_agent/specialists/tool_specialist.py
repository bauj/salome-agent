import json

from salome_agent.calc import CALC_DEF, run_calc
from salome_agent.codegen import _reachable, _refs_in
from salome_agent.executor_client import SessionDied, SessionPoisoned
from salome_agent.llm import (LLMCancelled, ReasoningBudgetExceeded, RepetitionLoopDetected,
                              assistant_tool_turn, continuity_fields, reasoning_field)
from salome_agent.logs import log
from salome_agent.modules import ModuleSpec
from salome_agent.repetition import loop_at_end, loop_nudge
from salome_agent.specialists.tool_selection import (
    initial_offered, resolve_load, suggest_tool_names,
)

# strict: nudge once every N risky ops with no intervening measure.
_STRICT_NUDGE_EVERY = 3

_LOAD_TOOLS_DESCRIPTION = (
    "Add tools to your available set for the rest of this stage. Pass "
    "tool `names`, a list of `categories`, or both. Use it whenever the tool "
    "you need is not in your current list."
)


_DISCARD_DESCRIPTION = (
    "Remove shapes you built in this stage for an attempt you abandoned (e.g. "
    "a part you are rebuilding differently) from the study and the result. "
    "Name every shape of that attempt, including the ones built from it."
)
_DISCARD_DEF = {"name": "discard", "description": _DISCARD_DESCRIPTION,
                "parameters": {"type": "object", "properties": {
                    "names": {"type": "array", "items": {"type": "string"},
                              "description": "published names to remove"}},
                    "required": ["names"]}}


# How much of a cut turn's reasoning goes back in the nudge: the tail is where
# the model was heading. A 20000-token budget cut is ~80k chars.
_REASONING_TAIL = 40_000


def _overrun_nudge(e, calc: bool) -> str:
    calc_line = " Put any arithmetic still to do in a calc call." if calc else ""
    if not e.reasoning:
        return (f"You spent ~{e.tokens} tokens deliberating without acting, and that "
                "answer was discarded. Pick the most likely reading, state it in one "
                f"sentence, and make your tool call now.{calc_line} Tool results will "
                "tell you if it was wrong.")
    reasoning = e.reasoning
    if len(reasoning) > _REASONING_TAIL:
        reasoning = "(earlier part omitted)\n" + reasoning[-_REASONING_TAIL:]
    return (f"You spent ~{e.tokens} tokens deliberating without acting and were "
            "stopped. Your reasoning so far is below. Do not start over: continue "
            f"from it, take the most likely reading, and make your tool call now.{calc_line} "
            "Tool results will tell you if it was wrong.\n\nYour reasoning so far:\n"
            + reasoning)


def _drops(record: dict, names: set) -> bool:
    out = record.get("out_vars") or []
    if record.get("recv") in names:
        return True
    return bool(out) and set(out) <= names


def prune_records(call_log: list, calls: list, names: set) -> tuple[list, list, list]:
    """call_log and calls without the records that made (or restyled) `names`.
    Also returns the indices of the calls left with no record at all."""
    kept_log = [r for r in call_log if not _drops(r, names)]
    kept_calls, removed = [], []
    for i, c in enumerate(calls):
        records = [r for r in c["records"] if not _drops(r, names)]
        if records:
            kept_calls.append({**c, "records": records})
        else:
            removed.append(i)
    return kept_log, kept_calls, removed


def _discard_refusal(names: list, call_log: list) -> str | None:
    wanted = set(names)
    built = {v for r in call_log for v in r.get("out_vars") or []}
    lines = [f"{n} was not built in this stage — you can only discard your own shapes."
             for n in names if n not in built]
    for r in call_log:
        out = r.get("out_vars") or []
        if not out or _drops(r, wanted):
            continue
        together = sorted(wanted & set(out))
        if together:
            rest = ", ".join(v for v in out if v not in wanted)
            lines.append(f"{', '.join(together)} was made by one call together with "
                         f"{rest} — discard them together.")
        refs = set()
        for a in r.get("args") or []:
            refs |= _refs_in(a)
        for n in sorted(wanted & refs):
            lines.append(f"{n} is still used by {', '.join(out)} — discard that too, "
                         f"or keep {n}.")
    return "\n".join(lines) or None


def _dead_ends(call_log: list, result_names: list) -> list[str]:
    """What codegen leaves out of result.py: shapes the results do not reach.
    Empty when no result name resolves — codegen then keeps everything."""
    built = {v for r in call_log for v in r.get("out_vars") or []}
    built |= {r["recv"] for r in call_log if not r.get("out_vars") and r.get("recv")}
    if not set(result_names) & built:
        return []
    keep = _reachable(call_log, result_names)
    return list(dict.fromkeys(v for r in call_log for v in r.get("out_vars") or []
                              if v not in keep))


def _call_key(name: str, args: dict) -> tuple:
    return (name, json.dumps(args, sort_keys=True, default=str))


def _unknown_names_note(names, all_defs: dict, categories) -> str:
    """One line per load_tools name that is not a real tool, so a wrong guess
    reads as a typo to fix instead of "this tool does not exist"."""
    lines = []
    for n in names or []:
        if n in all_defs:
            continue
        close = suggest_tool_names(n, all_defs)
        if close:
            hint = "did you mean " + ", ".join(
                f"{c} ({all_defs[c].get('category')})" for c in close) + "?"
        else:
            hint = "no close match; categories: " + ", ".join(categories)
        lines.append(f"\nunknown tool name {n} — {hint}")
    return "".join(lines)


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
                                  "description": "specific tool names to add"},
                        "categories": {"type": "array", "items": {
                            "type": "string",
                            "enum": list(self.spec.categories)},
                            "description": "whole tool categories to add"},
                    }}}

    def run(self, stage_plan, llm, client, iteration_cap, *, blocks=(),
            tool_selection="hints", verify_cadence="final", retry=None,
            seed_call_log=None, seed_calls=None, cancel=None, upstream=()) -> dict:
        if client is None and self.spec.needs_session:
            return {"fatal": f"a {self.spec.name} stage needs a SALOME agent server; "
                             f"--no-session only runs smesh stages"}
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
        if spec.kickoff:
            parts.append(spec.kickoff)
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
        # finish_stage is checked once: a clean stage closes on the first call,
        # an odd one is reopened with a report and closes on the next (task 217).
        audited = False
        # A call cut for deliberating past llm.reasoning_budget is re-asked once
        # with the tail of its own reasoning and a "continue and act" nudge; a
        # second cut in a row ends the stage. A call cut for looping (or a
        # budget cut whose reasoning ends in a loop) is re-asked without the
        # loop, up to llm.loop_retries cuts in a row (task 317).
        cuts = 0
        shape_infos: dict[str, dict] = {}
        n_seed_calls = len(calls)
        # Per-stage tool retrieval: offer CORE_TOOLS + the planner's hints, not the
        # whole registry (which was ~66% of a run's input tokens). The model widens
        # its set on demand via load_tools. "all" restores the send-everything
        # behaviour. finish_stage/load_tools are always appended, never in `offered`.
        all_defs = spec.tools_for(client)
        finish_def = {"name": spec.terminator.name, "parameters": spec.terminator.schema,
                      "description": spec.terminator.description}
        load_def = self._load_tools_def()
        discard_def = _DISCARD_DEF if spec.forgets else None
        calc_def = CALC_DEF if spec.calc else None
        if tool_selection == "all" or not spec.core_tools:
            offered = set(all_defs)
        else:
            offered = initial_offered(stage_plan, set(all_defs), spec.core_tools,
                                      spec.tool_groups)
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
                     + ([load_def] if load_def is not None else [])
                     + ([discard_def] if discard_def is not None else [])
                     + ([calc_def] if calc_def is not None else []))
            # Tool results are stored already compacted (see below), so `messages`
            # is sent verbatim: each turn's request extends the previous one byte
            # for byte, which keeps a gateway's prefix/KV cache alive. `load_tools`
            # widening `tools` mid-stage is the one remaining prefix break.
            # A shallow copy per call: the message dicts are never mutated after
            # append, so the copy is byte-identical content, and each request is a
            # verbatim extension of the last.
            try:
                response = llm.chat(messages=list(messages), tools=tools,
                                    module=stage_plan.get("module"))
            except LLMCancelled:
                log.info("  %s stage: cancelled mid-call", spec.name)
                return _with_calls(spec.on_exhausted(call_log, last_error), calls)
            except (ReasoningBudgetExceeded, RepetitionLoopDetected) as e:
                cuts += 1
                if isinstance(e, RepetitionLoopDetected):
                    loop = (e.sample, e.before)
                else:
                    found = loop_at_end(e.reasoning)
                    loop = found and (found.sample, e.reasoning[:found.start])
                if cuts > (llm.loop_retries if loop else 1):
                    log.warning("  %s stage: %s, cut %d in a row; giving up", spec.name, e, cuts)
                    what = "looped" if loop else "reasoning budget exceeded"
                    return _with_calls(spec.on_exhausted(call_log, {
                        "phase": "llm",
                        "message": f"stage ended: {what} {cuts} times in a row ({e})"}), calls)
                log.warning("  %s stage: %s%s; nudging it to act", spec.name, e,
                            " (ends in a loop)" if loop and not isinstance(
                                e, RepetitionLoopDetected) else "")
                if loop:
                    calc_line = (" Put any arithmetic still to do in a calc call."
                                 if calc_def else "")
                    nudge = loop_nudge(*loop, f"make your tool call now.{calc_line} Tool "
                                       "results will tell you if it was wrong.", _REASONING_TAIL)
                else:
                    nudge = _overrun_nudge(e, calc_def is not None)
                # a cut call adds nothing to the history, so after a cut in a
                # row the last message is the previous nudge: replace it
                if cuts > 1:
                    messages.pop()
                messages.append({"role": "user", "content": nudge})
                continue
            cuts = 0
            if response.get("content"):
                log.debug("    LLM reasoning: %s", response["content"])
            tool_calls = response["tool_calls"]
            log.debug("    LLM -> %s", [(c["name"], c["args"]) for c in tool_calls])
            if not tool_calls:
                # tool_choice="auto" — the model replied with prose and no call.
                # Keep its message, nudge it, and let the loop try again.
                messages.append({"role": "assistant", "content": response["content"] or "(no reply)",
                                 **continuity_fields(response.get("raw_message")),
                                 **reasoning_field(response.get("reasoning"),
                                                   response.get("raw_message"))})
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
            messages.append(assistant_tool_turn(tool_calls, response.get("raw_message"),
                                                response.get("reasoning")))
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
                    names = args.get("names")
                    if isinstance(names, str):
                        names = [names]
                    added = resolve_load(names, args.get("categories"), all_defs,
                                         spec.tool_groups)
                    new = added - offered
                    offered |= added
                    obs = (f"added: {', '.join(sorted(new)) or '(nothing new)'} "
                           f"— {len(offered)} tools available")
                    obs += _unknown_names_note(names, all_defs, spec.categories)
                    log.info("  load_tools(names=%s, categories=%s) -> +%d, %d available",
                             args.get("names"), args.get("categories"), len(new), len(offered))
                    messages.append({"role": "tool", "tool_call_id": call_id, "content": obs})
                    continue
                if name == "calc" and calc_def is not None:
                    obs = run_calc(args.get("code") or "")
                    log.info("  calc -> %s", obs if len(obs) <= 200 else obs[:200] + "...")
                    messages.append({"role": "tool", "tool_call_id": call_id, "content": obs})
                    continue
                if name == "discard" and discard_def is not None:
                    names = args.get("names") or []
                    if isinstance(names, str):
                        names = [names]
                    names = list(dict.fromkeys(names))
                    obs = _discard_refusal(names, call_log)
                    if obs is None:
                        r = client.forget(names)
                        if r.get("ok"):
                            call_log, calls, gone = prune_records(call_log, calls, set(names))
                            n_seed_calls -= sum(1 for g in gone if g < n_seed_calls)
                            for n in names:
                                shape_infos.pop(n, None)
                            obs = f"discarded: {', '.join(names)} — removed from the study"
                        else:
                            obs = f"discard failed: {r.get('error') or 'forget failed'}"
                    log.info("  discard(%s) -> %s", names, obs)
                    messages.append({"role": "tool", "tool_call_id": call_id, "content": obs})
                    continue
                # A known tool the model never saw the definition of: it guesses the
                # parameters from the catalogue one-liner, so load it and ask again.
                if load_def is not None and name in all_defs and name not in offered:
                    added = resolve_load([name], None, all_defs, spec.tool_groups)
                    offered |= added
                    obs = (f"{name} was not loaded, so this call was not run. It is "
                           f"loaded now ({', '.join(sorted(added))}): read its parameters "
                           "in your tool list and call it again.")
                    log.info("  %s called before loading -> loaded, not run", name)
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
                    if result.get("result_shape") and result.get("shape_info"):
                        shape_infos[result["result_shape"]] = result["shape_info"]
                    if isinstance(result.get("measurements"), dict):
                        shape_infos.update({n: v for n, v in result["measurements"].items() if v})
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
                if term is spec.terminator and spec.checks_results:
                    refusal = _check_results(client, args, spec.name)
                    if refusal:
                        log.info("  %s: finish refused, a result does not hold up",
                                 spec.terminator.name)
                        messages.append({"role": "tool", "tool_call_id": call_id,
                                         "content": refusal})
                        continue
                if term is spec.terminator and spec.gate is not None:
                    refusal = spec.gate(args, call_log, shape_infos)
                    if refusal:
                        log.info("  %s: finish refused by the gate", spec.terminator.name)
                        messages.append({"role": "tool", "tool_call_id": call_id,
                                         "content": refusal})
                        continue
                if term is spec.terminator and not audited:
                    parts = []
                    if (spec.measure_tools and verify_cadence in ("final", "strict")
                            and risky_since_measure > 0):
                        log.info("  verify_cadence=%s: nudging a measure before finish_stage",
                                 verify_cadence)
                        parts.append(
                            "You have moved or combined shapes since your last measure. "
                            "Call `measure` on your result shape(s) and confirm their "
                            "placement, then call finish_stage again.")
                    if spec.audit is not None:
                        report = spec.audit(args, calls[n_seed_calls:], call_log,
                                            upstream, shape_infos)
                        if report:
                            log.info("  finish_stage audit: reopening the stage")
                            parts.append(report)
                    if parts:
                        audited = True
                        messages.append({"role": "tool", "tool_call_id": call_id,
                                         "content": "\n\n".join(parts)})
                        continue
                result = term.handle(args, call_log)
                if term is spec.terminator and spec.forgets:
                    call_log, calls = _sweep(client, call_log, calls,
                                             result.get("result_names") or [])
                    result = {**result, "call_log": call_log}
                return _with_calls(result, calls)

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


def _check_results(client, args: dict, module: str) -> str | None:
    names = args.get("result_names") or []
    if isinstance(names, str):
        names = [names]
    if not names:
        return None
    try:
        r = client.check_results(list(names), module=module)
    except (SessionDied, SessionPoisoned):
        raise
    except Exception as e:  # noqa: BLE001 — a failed check never blocks a stage
        log.warning("  result check failed, finish accepted: %s", e)
        return None
    if not r.get("ok"):
        log.warning("  result check failed, finish accepted: %s", r.get("error"))
        return None
    return _result_check_refusal(r.get("results") or {})


_CHECK_LINES = {
    "missing": "{name} does not exist",
    "broken": "{name} no longer builds: {detail} -- something it was built from "
              "changed after it was made",
    "invalid": "{name} is not a valid shape: {detail}",
    "empty": "{name} is empty: {detail}",
    "editing": "{name} is still being edited: call close_sketch on it",
}


def _result_check_refusal(results: dict) -> str | None:
    lines = [f"- {_CHECK_LINES[r['status']].format(name=name, detail=r.get('detail', ''))}"
             for name, r in results.items() if r.get("status") in _CHECK_LINES]
    if not lines:
        return None
    return ("Not finished: a result does not hold up.\n" + "\n".join(lines)
            + "\nRebuild or fix it, or leave it out of result_names, then call "
            "finish_stage again.")


def _sweep(client, call_log: list, calls: list, result_names: list) -> tuple[list, list]:
    """Remove this stage's dead ends from the study at a clean finish, so the
    live study matches result.py. Best effort: if forget fails, the records
    stay and the study keeps them too."""
    dead = _dead_ends(call_log, result_names)
    if not dead:
        return call_log, calls
    r = client.forget(dead)
    if not r.get("ok"):
        log.warning("  finish_stage: could not remove unused %s: %s", dead, r.get("error"))
        return call_log, calls
    log.info("  finish_stage: removed unused %s", ", ".join(dead))
    call_log, calls, _ = prune_records(call_log, calls, set(dead))
    return call_log, calls


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
