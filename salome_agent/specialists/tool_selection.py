"""Per-stage geom tool retrieval (see
docs/superpowers/specs/2026-09-02-geom-tool-retrieval-design.md).

Pure functions, no I/O -- the geom specialist loop uses these to decide which
tool definitions to send the LLM on each turn instead of the whole registry.
"""

CORE_TOOLS = frozenset({
    "measure", "check_shape", "list_study_objects", "get_subshapes",
    "make_box", "fuse", "cut", "translate",
})


def hint_tool_names(stage_plan: dict) -> list[str]:
    """Geom tool names the planner expects this stage to use.

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


def initial_offered(stage_plan: dict, all_tool_names: set[str]) -> set[str]:
    """CORE_TOOLS plus any planner-hinted names that are real geom tools."""
    return set(CORE_TOOLS) | {n for n in hint_tool_names(stage_plan) if n in all_tool_names}


def resolve_load(names, categories, all_defs: dict[str, dict]) -> set[str]:
    """Expand a load_tools(names=, categories=) request to concrete tool names.

    `all_defs`: tool name -> its list_tools() entry (must carry "category").
    Unknown names and unknown categories are silently dropped.
    """
    want = {n for n in (names or []) if n in all_defs}
    cats = set(categories or [])
    want |= {n for n, d in all_defs.items() if d.get("category") in cats}
    return want
