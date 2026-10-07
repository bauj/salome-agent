"""Per-stage tool retrieval (see
docs/superpowers/specs/2026-09-02-geom-tool-retrieval-design.md).

Pure functions, no I/O -- the specialist loop uses these to decide which tool
definitions to send the LLM on each turn instead of the whole registry. Each
module keeps its own starting set and tool groups (ModuleSpec.core_tools,
tool_groups).
"""

import difflib


def _with_companions(names: set[str], all_tool_names, groups=()) -> set[str]:
    """`groups`: (triggers, group) pairs -- offering a trigger offers its group."""
    for triggers, group in groups:
        if names & triggers:
            names |= group & set(all_tool_names)
    return names


def hint_tool_names(stage_plan: dict) -> list[str]:
    """Tool names the planner expects this stage to use.

    Reads `stage_plan["tools"]` (list of names) or, as a fallback,
    `stage_plan["operations"]` (list of {"tool": name, ...} dicts). Tolerates
    either bare strings or {"tool": name} dicts in either key.
    """
    hints = stage_plan.get("tools") or stage_plan.get("operations") or []
    out: list[str] = []
    for h in hints:
        if isinstance(h, str):
            out.append(h)
        elif isinstance(h, dict) and isinstance(h.get("tool"), str):
            out.append(h["tool"])
    return out


def initial_offered(stage_plan: dict, all_tool_names: set[str],
                    core, groups=()) -> set[str]:
    """The module's core tools plus any planner-hinted names that are real tools."""
    hinted = {n for n in hint_tool_names(stage_plan) if n in all_tool_names}
    return set(core) | _with_companions(hinted, all_tool_names, groups)


def resolve_load(names, categories, all_defs: dict[str, dict], groups=()) -> set[str]:
    """Expand a load_tools(names=, categories=) request to concrete tool names.

    `all_defs`: tool name -> its list_tools() entry (must carry "category").
    Unknown names and unknown categories are dropped (see suggest_tool_names).
    """
    want = {n for n in (names or []) if n in all_defs}
    cats = set(categories or [])
    want |= {n for n, d in all_defs.items() if d.get("category") in cats}
    return _with_companions(want, all_defs, groups)


def suggest_tool_names(name: str, all_defs: dict[str, dict]) -> list[str]:
    """Real tool names close to a guessed one (at most 3): names containing the
    guess first (`thick_solid` -> `make_thick_solid`), then spelling matches."""
    contains = sorted(n for n in all_defs if name in n)
    close = difflib.get_close_matches(name, list(all_defs), n=3, cutoff=0.75)
    return list(dict.fromkeys(contains + close))[:3]
