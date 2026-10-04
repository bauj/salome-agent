def _fmt_arg(arg: dict) -> str:
    kind = arg["kind"]
    if kind == "literal":
        return repr(arg["value"])
    if kind == "ref":
        return arg["value"]
    if kind == "list":
        return "[" + ", ".join(_fmt_arg(a) for a in arg["value"]) + "]"
    if kind == "study":
        return f'find_object("{arg["value"]}")'
    if kind == "expr":
        return arg["value"]
    if kind == "salomeds_color":
        r, g, b = arg["value"]
        return f"SALOMEDS.Color({r/255:.4f}, {g/255:.4f}, {b/255:.4f})"
    raise ValueError(f"unknown ArgRef kind {kind!r}")


def _refs_in(arg: dict) -> set[str]:
    if arg["kind"] == "ref":
        return {arg["value"]}
    if arg["kind"] == "list":
        return set().union(*(_refs_in(a) for a in arg["value"])) if arg["value"] else set()
    if arg["kind"] == "expr":
        return set(arg.get("refs") or ())
    return set()


def _reachable(call_log: list[dict], result_names: list[str]) -> set[str]:
    wanted = set(result_names)
    changed = True
    while changed:
        changed = False
        for call in call_log:
            # A record on a kept receiver (a sketch's lines and constraints)
            # comes with it, along with what it references.
            if call.get("recv") in wanted and not set(call["out_vars"]) <= wanted:
                wanted |= set(call["out_vars"])
                changed = True
            if wanted & set(call["out_vars"]) or call.get("recv") in wanted:
                for arg in call["args"]:
                    for ref in _refs_in(arg):
                        if ref not in wanted:
                            wanted.add(ref)
                            changed = True
    return wanted


def _reachable_with_display(call_log: list[dict], reachable: set[str]) -> set[tuple]:
    """Return set of (index, display_flag) tuples for reachable calls.

    display_flag is True for display records (no out_vars but recv in reachable set).
    """
    reachable_calls = set()
    for i, call in enumerate(call_log):
        out_vars = call.get("out_vars", [])
        if reachable & set(out_vars):
            reachable_calls.add((i, False))
        elif call.get("recv") in reachable:
            # Display record: no out_vars but recv is reachable
            reachable_calls.add((i, True))
    return reachable_calls


def roots_for(call_log: list[dict], result_names: list[str]) -> list[str]:
    # A weak model can call finish_stage with names that match no shape it
    # actually built (invented names, or an empty list). Pruning to those would
    # emit nothing and silently drop the whole stage from result.py (issue #7),
    # so fall back to keeping every published shape when nothing resolves.
    built = {v for call in call_log for v in call.get("out_vars", [])}
    # A display/method_call record (set_color, set_transparency, ...) has no
    # out_vars — it mutates an existing object named by `recv` — but naming
    # that object in finish_stage is a legitimate match, not an invented name.
    built |= {call["recv"] for call in call_log if not call.get("out_vars") and call.get("recv")}
    if set(result_names) & built:
        return list(result_names)
    from salome_agent.logs import log

    if call_log:
        log.warning("codegen: result_names %s match no built shape — keeping "
                    "every published shape for this stage", result_names)
    return [n for call in call_log for n in (call.get("publish") or [])]


_ADD_TO_STUDY = "geompy.addToStudy({var}, {name!r})"


def generate(call_log: list[dict], result_names: list[str],
             call_prefix: str = "geompy.", keep: set | None = None,
             publish_line: str = _ADD_TO_STUDY) -> str:
    if keep is None:
        reachable = _reachable(call_log, roots_for(call_log, result_names))
    else:
        reachable = keep
    reachable_calls = _reachable_with_display(call_log, reachable)
    lines = []
    for i, call in enumerate(call_log):
        if (i, False) not in reachable_calls and (i, True) not in reachable_calls:
            continue

        emit = call.get("emit", "geompy_call")
        args_src = ", ".join(_fmt_arg(a) for a in call["args"])

        if emit == "geompy_call":
            assign_target = call["out_vars"][0] if call["out_vars"] else "_"
            lines.append(f"{assign_target} = {call_prefix}{call['fn']}({args_src})")
            if call.get("publish"):
                for var, name in zip(call["out_vars"], call["publish"]):
                    # geompy.addToStudy is the publish API, not the module's call prefix
                    lines.append(publish_line.format(var=var, name=name))
        elif emit == "stmt":
            lines.append(call["code"])
            for var, name in zip(call["out_vars"], call.get("publish") or []):
                lines.append(publish_line.format(var=var, name=name))
        elif emit == "method_call":
            recv = call.get("recv")
            lines.append(f"{recv}.{call['fn']}({args_src})")
        elif emit == "gui_call":
            recv = call.get("recv")
            lines.append("try:")
            lines.append(f"    gg.{call['fn']}(salome.ObjectToID({recv}), {args_src})")
            lines.append("except Exception:")
            lines.append("    pass")
        elif emit == "list_call":
            tmp = f"_sub_{i}"
            lines.append(f"{tmp} = {call_prefix}{call['fn']}({args_src})")
            pubs = dict(zip(call["out_vars"], call.get("publish") or []))
            for idx, var in enumerate(call["out_vars"]):
                if var not in reachable:
                    continue
                lines.append(f"{var} = {tmp}[{idx}]")
                if var in pubs:
                    # geompy.addToStudy is the publish API, not the module's call prefix
                    lines.append(publish_line.format(var=var, name=pubs[var]))

    return "\n".join(lines)
