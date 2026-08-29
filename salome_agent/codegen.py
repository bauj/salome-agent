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
    raise ValueError(f"unknown ArgRef kind {kind!r}")


def _refs_in(arg: dict) -> set[str]:
    if arg["kind"] == "ref":
        return {arg["value"]}
    if arg["kind"] == "list":
        return set().union(*(_refs_in(a) for a in arg["value"])) if arg["value"] else set()
    return set()


def _reachable(call_log: list[dict], result_names: list[str]) -> set[str]:
    wanted = set(result_names)
    changed = True
    while changed:
        changed = False
        for call in call_log:
            if wanted & set(call["out_vars"]):
                for arg in call["args"]:
                    for ref in _refs_in(arg):
                        if ref not in wanted:
                            wanted.add(ref)
                            changed = True
    return wanted


def generate(call_log: list[dict], result_names: list[str]) -> str:
    reachable = _reachable(call_log, result_names)
    lines = []
    for call in call_log:
        if not (reachable & set(call["out_vars"])):
            continue
        args_src = ", ".join(_fmt_arg(a) for a in call["args"])
        assign_target = call["out_vars"][0] if call["out_vars"] else "_"
        lines.append(f"{assign_target} = geompy.{call['fn']}({args_src})")
        if call.get("publish"):
            for var, name in zip(call["out_vars"], call["publish"]):
                lines.append(f"geompy.addToStudy({var}, {name!r})")
    return "\n".join(lines)
