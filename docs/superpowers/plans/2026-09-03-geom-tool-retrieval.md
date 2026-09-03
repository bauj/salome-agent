# Geom Tool Retrieval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop sending all ~38 geom tool schemas on every geom-specialist LLM call; offer a small per-stage subset (~10) plus a `load_tools` escape hatch, cutting run input tokens ~300k → <100k.

**Architecture:** The planner emits an optional per-stage `tools` hint (names it expects that stage to use). At stage start the geom specialist offers `CORE_TOOLS ∪ valid-hints` + `finish_stage` + `load_tools`. When the model needs a tool it does not have, it calls `load_tools(names=[...], categories=[...])`, which widens the offered set for the rest of that stage. No embeddings, no extra LLM call. A `policy.tool_selection: "all"` knob restores today's send-everything behaviour.

**Tech Stack:** Python 3.10, pytest, LangGraph `StateGraph`. LLM tool-calling over an OpenAI-compatible endpoint. Geom tools live in `executor/geom_tools.py` and reach the specialist via the agent server (`agent_server/core.py` → `executor_client.list_tools()` → `{"tools": [ {name, parameters, description}, ... ]}`).

**Spec:** `docs/superpowers/specs/2026-09-02-geom-tool-retrieval-design.md` (approved 2026-09-02; currently on branch `spec/geom-tool-retrieval`, not merged — read it with `git show 4a6e81b:docs/superpowers/specs/2026-09-02-geom-tool-retrieval-design.md`, or merge that branch first).

## Global Constraints

- **Python floor: 3.10.** No `match`-only-3.12 syntax, no `type` statement. `X | None` annotations are fine.
- **Branch per feature.** Do this on `feat/193-geom-tool-retrieval`, never straight on `main`. ff-merge only after live verification (Task 6).
- **Commit messages: bare subject line only.** No body, no `Co-Authored-By`, no trailers. `feat:` / `test:` / `docs:` prefixes.
- **Stage named files only** when committing (`git add path/a path/b`). Never `git add -A` / `git add .`.
- **No distribution paths** (`/storage/...`, `...-FD44`) in repo code, comments, docs, or tests. Resolve from config/env.
- **SERVER-RESTART GOTCHA:** the SALOME GUI plugin server caches `executor.geom_tools` in `sys.modules`. After ANY change to `executor/geom_tools.py`, the whole SALOME process must be restarted before a live run sees it — re-clicking the plugin menu does NOT reload the module. Relevant only to Task 6 (live).
- Unit tests must pass offline with no SALOME and no LLM (`pytest tests/unit`). Current baseline: ~205 passing.

---

## File Structure

| File | Change | Responsibility |
|---|---|---|
| `executor/geom_tools.py` | modify | Add `TOOL_CATEGORY` dict; `list_tools()` emits a `category` per entry. |
| `salome_agent/nodes/planner.py` | modify | Planner prompt asks for an optional per-stage `tools` list of geom tool names. |
| `salome_agent/interactive.py` | modify | `_PLAN_KEYS` gains `"tools"`; edited plans keep the key. |
| `salome_agent/specialists/tool_selection.py` | **create** | Pure functions: `CORE_TOOLS`, `hint_tool_names`, `initial_offered`, `resolve_load`. No I/O. |
| `salome_agent/specialists/geom.py` | modify | `LOAD_TOOLS_SCHEMA`/`_DESCRIPTION`; `_run` builds a per-iteration tool list from an `offered` set; handles `load_tools`; new `tool_selection` param; prompt paragraph; logging. |
| `salome_agent/graph.py` | modify | `_run_stage_node` reads `policy.tool_selection`, passes it to `run_geom_specialist`. |
| `salome_agent/cli.py` | modify | `--tools {hints,all}` flag → `config["policy"]["tool_selection"]`. |
| `config.example.json` | modify | `"tool_selection": "hints"` under `policy`. |
| `README.md` | modify | Options table row for `--tools`; policy block gains the key. |
| `tests/unit/test_geom_tools.py` | modify | Guard: every tool categorized; every `list_tools()` entry carries a category. |
| `tests/unit/test_planner.py` | modify | Planner prompt mentions the `tools` hint; a returned `tools` key survives into the plan. |
| `tests/unit/test_interactive.py` | modify | `tools` key preserved through an edit round-trip. |
| `tests/unit/test_tool_selection.py` | **create** | All four pure functions. |
| `tests/unit/test_specialist_geom.py` | modify | Offered subset is the expected small set; `load_tools` widens the next call; `"all"` mode offers everything; hint names honoured. |
| `tests/unit/test_cli.py` | modify | `--tools all` sets `policy.tool_selection`. |

---

## Task 1: Tool categories in the geom registry

**Files:**
- Modify: `executor/geom_tools.py` (add `TOOL_CATEGORY` near `TOOLS` at line ~1207; extend `list_tools()` at line 1713)
- Test: `tests/unit/test_geom_tools.py`

**Interfaces:**
- Consumes: existing `TOOLS: dict[str, Callable]` (the authoritative tool-name list), `TOOL_SCHEMAS`, `TOOL_DESCRIPTIONS`.
- Produces:
  - `TOOL_CATEGORY: dict[str, str]` — every key of `TOOLS` mapped to exactly one of
    `"primitive" | "sketch" | "swept" | "boolean" | "transform" | "feature" | "display" | "query"`.
  - `list_tools() -> list[dict]` — each entry now `{"name": str, "parameters": dict, "description": str, "category": str}`. `category` is `""` for any tool missing from `TOOL_CATEGORY` (should never happen — the guard test forbids it).

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/test_geom_tools.py`:

```python
from executor.geom_tools import TOOLS, TOOL_CATEGORY, list_tools

_VALID_CATEGORIES = {
    "primitive", "sketch", "swept", "boolean",
    "transform", "feature", "display", "query",
}


def test_every_tool_has_a_valid_category():
    missing = [n for n in TOOLS if n not in TOOL_CATEGORY]
    assert not missing, f"tools with no TOOL_CATEGORY entry: {missing}"
    bad = {n: c for n, c in TOOL_CATEGORY.items() if c not in _VALID_CATEGORIES}
    assert not bad, f"tools with an unknown category: {bad}"
    extra = [n for n in TOOL_CATEGORY if n not in TOOLS]
    assert not extra, f"TOOL_CATEGORY names not in TOOLS: {extra}"


def test_list_tools_entries_carry_a_category():
    for entry in list_tools():
        assert entry["category"] in _VALID_CATEGORIES, entry
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_geom_tools.py::test_every_tool_has_a_valid_category tests/unit/test_geom_tools.py::test_list_tools_entries_carry_a_category -v`
Expected: FAIL — `ImportError: cannot import name 'TOOL_CATEGORY'`.

- [ ] **Step 3: Add `TOOL_CATEGORY` and extend `list_tools()`**

In `executor/geom_tools.py`, immediately after the `TOOLS` dict closes (line ~1207), add:

```python
# One category per tool, for the geom-specialist tool-retrieval mechanism
# (salome_agent/specialists/tool_selection.py). Keep in sync with TOOLS —
# tests/unit/test_geom_tools.py guards that every TOOLS key appears here.
TOOL_CATEGORY: dict[str, str] = {
    "make_box": "primitive",
    "make_box_two_pnt": "primitive",
    "make_cylinder": "primitive",
    "make_sphere": "primitive",
    "make_cone": "primitive",
    "make_torus": "primitive",
    "make_face": "sketch",
    "make_disk": "sketch",
    "make_point": "sketch",
    "make_vector": "sketch",
    "make_line": "sketch",
    "make_plane": "sketch",
    "make_circle": "sketch",
    "make_prism": "swept",
    "make_revolution": "swept",
    "make_pipe": "swept",
    "make_thick_solid": "swept",
    "make_filling": "swept",
    "fuse": "boolean",
    "cut": "boolean",
    "common": "boolean",
    "partition": "boolean",
    "translate": "transform",
    "rotate": "transform",
    "mirror": "transform",
    "scale": "transform",
    "multi_translate": "transform",
    "multi_rotate": "transform",
    "offset_shape": "transform",
    "position": "transform",
    "fillet_edges": "feature",
    "chamfer_edges": "feature",
    "set_color": "display",
    "set_transparency": "display",
    "set_display_mode": "display",
    "check_shape": "query",
    "list_study_objects": "query",
    "measure": "query",
}
```

Replace `list_tools()` (line 1713) with:

```python
def list_tools() -> list[dict]:
    return [{"name": n,
             "parameters": TOOL_SCHEMAS.get(n, {"type": "object", "properties": {}}),
             "description": TOOL_DESCRIPTIONS.get(n, ""),
             "category": TOOL_CATEGORY.get(n, "")}
            for n in TOOLS]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_geom_tools.py -v`
Expected: PASS (all, including pre-existing).

- [ ] **Step 5: Commit**

```bash
git add executor/geom_tools.py tests/unit/test_geom_tools.py
git commit -m "feat: tag each geom tool with a category and expose it from list_tools"
```

---

## Task 2: Planner emits a per-stage tool hint

**Files:**
- Modify: `salome_agent/nodes/planner.py` (`_PLANNER_BASE` string, ~line 11)
- Modify: `salome_agent/interactive.py` (`_edit_plan_in_editor`, the `setdefault` block at lines 35-37; `_PLAN_KEYS` at line 9 is dead code — update it too for consistency but nothing reads it)
- Test: `tests/unit/test_planner.py`, `tests/unit/test_interactive.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: stage dicts from `plan()` MAY carry `"tools": list[str]` — geom tool names the planner expects that stage to use. Absent or `[]` is valid and common. Consumed by Task 3's `hint_tool_names`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/unit/test_planner.py`:

```python
from salome_agent.nodes.planner import PLANNER_SYSTEM_PROMPT


def test_prompt_documents_the_optional_tools_hint():
    assert '"tools"' in PLANNER_SYSTEM_PROMPT
    assert "geom tool names" in PLANNER_SYSTEM_PROMPT.lower()


def test_a_returned_tools_key_survives_into_the_plan():
    from unittest.mock import MagicMock
    from salome_agent.nodes.planner import plan

    llm = MagicMock()
    llm.chat.return_value = {"stages": [
        {"module": "geom", "intent": "red box", "params": {}, "inputs": [],
         "tools": ["make_box", "set_color"]},
    ]}
    stages = plan("make a red box", llm)
    assert stages[0]["tools"] == ["make_box", "set_color"]
```

Add to `tests/unit/test_interactive.py` (match the module's existing style;
`_edit_plan_in_editor` shells out to `$EDITOR`, so drive it via a fake editor
that just leaves the file untouched — or, simpler, test the setdefault
behaviour by pointing `$EDITOR` at `true`):

```python
def test_edit_preserves_the_tools_hint(monkeypatch, tmp_path):
    from salome_agent import interactive

    monkeypatch.setenv("EDITOR", "true")  # no-op editor, file unchanged
    plan = [{"module": "geom", "intent": "x", "params": {}, "inputs": [],
             "tools": ["make_box"]}]
    out = interactive._edit_plan_in_editor(plan)
    assert out[0]["tools"] == ["make_box"]


def test_edit_defaults_missing_tools_to_empty_list(monkeypatch):
    from salome_agent import interactive

    monkeypatch.setenv("EDITOR", "true")
    plan = [{"module": "geom", "intent": "x", "params": {}, "inputs": []}]
    out = interactive._edit_plan_in_editor(plan)
    assert out[0]["tools"] == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_planner.py::test_prompt_documents_the_optional_tools_hint tests/unit/test_planner.py::test_a_returned_tools_key_survives_into_the_plan tests/unit/test_interactive.py -v`
Expected: planner prompt test FAILS (string absent); interactive test FAILS or ERRORS (key dropped / helper name wrong).

- [ ] **Step 3: Update the planner prompt**

In `salome_agent/nodes/planner.py`, in `_PLANNER_BASE`, change the stage-shape sentence. Current:

```
ordered list of stages. Each stage has: module (geom|smesh|paravis), intent
(short description of the job for that stage), params (dims in millimetres,
unconverted), inputs (named upstream objects this stage needs). Respond as
JSON: {"stages": [...]}.
```

Replace with:

```
ordered list of stages. Each stage has: module (geom|smesh|paravis), intent
(short description of the job for that stage), params (dims in millimetres,
unconverted), inputs (named upstream objects this stage needs), and — for a
geom stage — an optional "tools" list naming the geom tool names you expect
that stage to use (from the catalogue below; names only, e.g.
["make_box", "translate", "set_color"]). The "tools" list is a hint that
narrows what the geom specialist is offered; omit it if unsure. Respond as
JSON: {"stages": [...]}.
```

- [ ] **Step 4: Update `interactive.py`**

Line 9 (dead code, kept in sync):

```python
_PLAN_KEYS = ("module", "intent", "params", "inputs", "tools")
```

In `_edit_plan_in_editor`, in the `for stage in edited:` loop after `stage.setdefault("inputs", [])` (line 37):

```python
        stage.setdefault("tools", [])
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/unit/test_planner.py tests/unit/test_interactive.py -v`
Expected: PASS (all, including pre-existing).

- [ ] **Step 6: Commit**

```bash
git add salome_agent/nodes/planner.py salome_agent/interactive.py tests/unit/test_planner.py tests/unit/test_interactive.py
git commit -m "feat: planner emits an optional per-stage geom tool hint"
```

---

## Task 3: `tool_selection` pure module

**Files:**
- Create: `salome_agent/specialists/tool_selection.py`
- Test: `tests/unit/test_tool_selection.py`

**Interfaces:**
- Consumes: stage-plan dicts (Task 2's `"tools"` key), and `list_tools()` output shape (Task 1's `category` field).
- Produces:
  - `CORE_TOOLS: frozenset[str]` = `{"measure", "check_shape", "list_study_objects", "make_box", "fuse", "cut", "translate"}`
  - `hint_tool_names(stage_plan: dict) -> list[str]`
  - `initial_offered(stage_plan: dict, all_tool_names: set[str]) -> set[str]`
  - `resolve_load(names, categories, all_defs: dict[str, dict]) -> set[str]` where `all_defs` maps tool name → its `list_tools()` entry (must carry `"category"`).

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_tool_selection.py`:

```python
from salome_agent.specialists.tool_selection import (
    CORE_TOOLS, hint_tool_names, initial_offered, resolve_load,
)

ALL = {"make_box", "make_cylinder", "translate", "set_color", "fuse", "cut",
       "measure", "check_shape", "list_study_objects", "make_thick_solid"}
DEFS = {
    "make_box": {"name": "make_box", "category": "primitive"},
    "make_cylinder": {"name": "make_cylinder", "category": "primitive"},
    "translate": {"name": "translate", "category": "transform"},
    "set_color": {"name": "set_color", "category": "display"},
    "make_thick_solid": {"name": "make_thick_solid", "category": "swept"},
}


def test_hint_tool_names_reads_the_tools_key():
    assert hint_tool_names({"tools": ["make_box", "translate"]}) == ["make_box", "translate"]


def test_hint_tool_names_also_reads_the_operations_key_and_dict_items():
    assert hint_tool_names({"operations": [{"tool": "make_box"}, {"tool": "cut"}]}) == ["make_box", "cut"]


def test_hint_tool_names_ignores_malformed_entries_and_missing_keys():
    assert hint_tool_names({}) == []
    assert hint_tool_names({"tools": [1, None, {"nope": "x"}, "make_box"]}) == ["make_box"]


def test_initial_offered_is_core_union_valid_hints():
    got = initial_offered({"tools": ["make_cylinder", "not_a_tool"]}, ALL)
    assert got == set(CORE_TOOLS) | {"make_cylinder"}
    assert "not_a_tool" not in got


def test_initial_offered_with_no_hint_is_exactly_core():
    assert initial_offered({}, ALL) == set(CORE_TOOLS)


def test_resolve_load_expands_names_categories_and_both():
    assert resolve_load(["make_box"], None, DEFS) == {"make_box"}
    assert resolve_load(None, ["primitive"], DEFS) == {"make_box", "make_cylinder"}
    assert resolve_load(["translate"], ["swept"], DEFS) == {"translate", "make_thick_solid"}


def test_resolve_load_ignores_unknown_names_and_categories():
    assert resolve_load(["ghost"], ["nonsense"], DEFS) == set()
    assert resolve_load(None, None, DEFS) == set()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_tool_selection.py -v`
Expected: FAIL — `ModuleNotFoundError: salome_agent.specialists.tool_selection`.

- [ ] **Step 3: Write the module**

Create `salome_agent/specialists/tool_selection.py`:

```python
"""Per-stage geom tool retrieval (see
docs/superpowers/specs/2026-09-02-geom-tool-retrieval-design.md).

Pure functions, no I/O — the geom specialist loop uses these to decide which
tool definitions to send the LLM on each turn instead of the whole registry.
"""

CORE_TOOLS = frozenset({
    "measure", "check_shape", "list_study_objects",
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_tool_selection.py -v`
Expected: PASS (7 tests).

- [ ] **Step 5: Commit**

```bash
git add salome_agent/specialists/tool_selection.py tests/unit/test_tool_selection.py
git commit -m "feat: add tool_selection helpers for per-stage geom tool retrieval"
```

---

## Task 4: `load_tools` meta-tool and the geom specialist loop

**Files:**
- Modify: `salome_agent/specialists/geom.py` (`GEOM_SYSTEM_PROMPT`; add `LOAD_TOOLS_SCHEMA`/`LOAD_TOOLS_DESCRIPTION`; `run_geom_specialist` + `_run` signatures; the loop body)
- Test: `tests/unit/test_specialist_geom.py`

**Interfaces:**
- Consumes: `CORE_TOOLS`, `hint_tool_names`, `initial_offered`, `resolve_load` from Task 3; `list_tools()` entries carrying `category` from Task 1.
- Produces:
  - `run_geom_specialist(stage_plan, llm, client, iteration_cap, upstream=None, tool_selection="hints") -> dict` — unchanged return contract (`{"call_log", "result_names"}` on finish, `{"call_log", "last_error"}` on cap, `{"fatal": ...}` on session death). New keyword-only-in-practice `tool_selection: str` — `"hints"` (default) or `"all"`.
  - `_run(...)` gains the same trailing `tool_selection="hints"` parameter.

- [ ] **Step 1: Write the failing tests**

Add to `tests/unit/test_specialist_geom.py`:

```python
def _defs():
    # Mimics list_tools() output with categories.
    cats = {
        "make_box": "primitive", "make_cylinder": "primitive",
        "translate": "transform", "set_color": "display",
        "fuse": "boolean", "cut": "boolean", "measure": "query",
        "check_shape": "query", "list_study_objects": "query",
        "make_thick_solid": "swept",
    }
    return [{"name": n, "parameters": {"type": "object", "properties": {}},
             "description": "", "category": c} for n, c in cats.items()]


def _tool_names(chat_call):
    return {t["name"] for t in chat_call.kwargs["tools"]}


def test_hinted_stage_offers_core_plus_hints_plus_meta_tools_only():
    llm = MagicMock()
    llm.chat.return_value = {"tool_call": "finish_stage", "args": {"result_names": ["x"]}}
    client = MagicMock()
    client.list_tools.return_value = {"tools": _defs()}
    run_geom_specialist(
        {"module": "geom", "intent": "red box", "params": {}, "inputs": [],
         "tools": ["make_box", "set_color"]},
        llm, client, iteration_cap=3,
    )
    offered = _tool_names(llm.chat.call_args)
    assert offered == {
        "measure", "check_shape", "list_study_objects", "make_box", "fuse",
        "cut", "translate",            # CORE_TOOLS
        "set_color",                   # hint (make_box already in core)
        "finish_stage", "load_tools",  # meta
    }
    assert "make_thick_solid" not in offered


def test_load_tools_widens_the_next_call():
    llm = MagicMock()
    llm.chat.side_effect = [
        {"tool_call": "load_tools", "args": {"categories": ["swept"]}},
        {"tool_call": "finish_stage", "args": {"result_names": ["x"]}},
    ]
    client = MagicMock()
    client.list_tools.return_value = {"tools": _defs()}
    run_geom_specialist({"module": "geom", "intent": "hollow", "params": {}, "inputs": []},
                        llm, client, iteration_cap=5)
    first = _tool_names(llm.chat.call_args_list[0])
    second = _tool_names(llm.chat.call_args_list[1])
    assert "make_thick_solid" not in first
    assert "make_thick_solid" in second
    # a tool observation was fed back
    msgs = llm.chat.call_args_list[1].kwargs["messages"]
    assert any(m.get("role") == "tool" and "make_thick_solid" in str(m.get("content"))
               for m in msgs)


def test_tool_selection_all_offers_the_full_registry():
    llm = MagicMock()
    llm.chat.return_value = {"tool_call": "finish_stage", "args": {"result_names": ["x"]}}
    client = MagicMock()
    client.list_tools.return_value = {"tools": _defs()}
    run_geom_specialist({"module": "geom", "intent": "x", "params": {}, "inputs": [],
                         "tools": ["make_box"]},
                        llm, client, iteration_cap=3, tool_selection="all")
    offered = _tool_names(llm.chat.call_args)
    assert "make_thick_solid" in offered
    assert "make_cylinder" in offered
```

Also update the three PRE-EXISTING tests in this file that set
`client.list_tools.return_value = {"tools": [{"name": "make_box"}]}` — add
`"category": "primitive"` to that entry so `_defs`-free tests still exercise a
realistic shape. (`test_finish_stage_returns_call_log_and_result_names`,
`test_no_tool_call_is_nudged_then_loop_continues`,
`test_upstream_shape_info_is_injected_into_the_opening_prompt`.) They must
still pass unchanged in intent.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_specialist_geom.py -v`
Expected: the three new tests FAIL (`load_tools` not offered / not handled; `tool_selection` kwarg unexpected).

- [ ] **Step 3: Implement**

In `salome_agent/specialists/geom.py`:

**3a.** Add the import at the top:

```python
from salome_agent.specialists.tool_selection import initial_offered, resolve_load
```

**3b.** Add after `FINISH_STAGE_DESCRIPTION` (line ~43):

```python
LOAD_TOOLS_SCHEMA = {
    "type": "object",
    "properties": {
        "names": {"type": "array", "items": {"type": "string"},
                  "description": "specific geom tool names to add"},
        "categories": {"type": "array", "items": {
            "type": "string",
            "enum": ["primitive", "sketch", "swept", "boolean",
                     "transform", "feature", "display", "query"]},
            "description": "whole tool categories to add"},
    },
}
LOAD_TOOLS_DESCRIPTION = (
    "Add geom tools to your available set for the rest of this stage. Pass "
    "tool `names`, a list of `categories`, or both. Use it whenever the tool "
    "you need is not in your current list."
)
```

**3c.** Add to `GEOM_SYSTEM_PROMPT` (append a paragraph before the closing `"""`):

```
You are given only a subset of the geom tools, chosen for this stage. If you
need a tool that is not in your list, call load_tools with specific tool
`names` or whole `categories` (primitive, sketch, swept, boolean, transform,
feature, display, query) to add them, then call the tool you wanted.
```

**3d.** Change both signatures:

```python
def run_geom_specialist(stage_plan: dict, llm, client, iteration_cap: int,
                        upstream: dict | None = None,
                        tool_selection: str = "hints") -> dict:
    try:
        return _run(stage_plan, llm, client, iteration_cap, upstream, tool_selection)
    except (SessionDied, SessionPoisoned) as e:
        return {"fatal": f"SALOME agent server unavailable: {e}"}


def _run(stage_plan: dict, llm, client, iteration_cap: int,
         upstream: dict | None = None, tool_selection: str = "hints") -> dict:
```

**3e.** Replace the tool-list setup (lines 71-77) with:

```python
    all_defs = {t["name"]: t for t in client.list_tools()["tools"]}
    finish_def = {"name": "finish_stage", "parameters": FINISH_STAGE_SCHEMA,
                  "description": FINISH_STAGE_DESCRIPTION}
    load_def = {"name": "load_tools", "parameters": LOAD_TOOLS_SCHEMA,
                "description": LOAD_TOOLS_DESCRIPTION}
    if tool_selection == "all":
        offered = set(all_defs)
    else:
        offered = initial_offered(stage_plan, set(all_defs))
    log.info("  geom stage: offered %d/%d tools (%s)",
             len(offered), len(all_defs), ", ".join(sorted(offered)))
```

**3f.** Inside the `for i in range(iteration_cap):` loop, as the FIRST statement, build the per-iteration list:

```python
        tools = ([all_defs[n] for n in sorted(offered) if n in all_defs]
                 + [finish_def, load_def])
```

(Delete the old single pre-loop `tools = ...`; the `llm.chat(messages=messages, tools=tools)` call is unchanged.)

**3g.** Handle `load_tools` — place this block right after the assistant tool-call turn is appended to `messages` (after line 99, before the `finish_stage` check at line 100):

```python
        if response["tool_call"] == "load_tools":
            added = resolve_load(response["args"].get("names"),
                                 response["args"].get("categories"), all_defs)
            new = added - offered
            offered |= added
            obs = (f"added: {', '.join(sorted(new)) or '(nothing new)'} "
                   f"— {len(offered)} tools available")
            log.info("  load_tools(names=%s, categories=%s) -> +%d, %d available",
                     response["args"].get("names"), response["args"].get("categories"),
                     len(new), len(offered))
            messages.append({"role": "tool", "tool_call_id": call_id, "content": obs})
            continue
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_specialist_geom.py -v`
Expected: PASS (all — 3 new + pre-existing).

- [ ] **Step 5: Commit**

```bash
git add salome_agent/specialists/geom.py tests/unit/test_specialist_geom.py
git commit -m "feat: geom specialist offers a per-stage tool subset with a load_tools escape hatch"
```

---

## Task 5: Wire `policy.tool_selection` through the graph, CLI, config and docs

**Files:**
- Modify: `salome_agent/graph.py` (`_run_stage_node`, geom branch, lines 68-88)
- Modify: `salome_agent/cli.py` (`parse_args`, `main`)
- Modify: `config.example.json`
- Modify: `README.md`
- Test: `tests/unit/test_cli.py`, `tests/unit/test_graph.py` (if it exercises the geom branch)

**Interfaces:**
- Consumes: `run_geom_specialist(..., tool_selection=...)` from Task 4.
- Produces: `config["policy"]["tool_selection"]` — `"hints"` (default when absent) or `"all"`. Set from `--tools {hints,all}` when given.

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/test_cli.py` (match the file's existing `parse_args` test style):

```python
def test_tools_flag_sets_policy(tmp_path, monkeypatch):
    from salome_agent.cli import parse_args
    args = parse_args(["do a thing", "--tools", "all"])
    assert args.tools == "all"


def test_tools_flag_defaults_to_none():
    from salome_agent.cli import parse_args
    args = parse_args(["do a thing"])
    assert args.tools is None
```

If `tests/unit/test_graph.py` has a geom-branch test with a fake config, add:

```python
def test_geom_branch_passes_tool_selection_from_policy():
    # ... build state with state["config"]["policy"]["tool_selection"] = "all"
    # ... assert run_geom_specialist was called with tool_selection="all"
```

(Only if a suitable harness already exists in that file — do not build one from scratch here.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_cli.py -v -k tools`
Expected: FAIL — `AttributeError: 'Namespace' object has no attribute 'tools'`.

- [ ] **Step 3: Implement**

**3a.** `salome_agent/cli.py`, in `parse_args`, after the `--on-failure` line:

```python
    parser.add_argument("--tools", choices=["hints", "all"], default=None,
                        help="geom tool retrieval: 'hints' (default) offers the "
                             "specialist a per-stage subset; 'all' offers the "
                             "full registry every call")
```

In `main`, after the `if args.on_failure:` block:

```python
    if args.tools:
        config["policy"]["tool_selection"] = args.tools
```

**3b.** `salome_agent/graph.py`, `_run_stage_node`, geom branch. After `cap = state["config"]["policy"]["tool_iteration_cap"]` (line 68) is fine, but do it inside the `if module == "geom":` block for locality — just before the `run_geom_specialist(...)` call (line 88):

```python
        tool_selection = state["config"]["policy"].get("tool_selection", "hints")
        result = run_geom_specialist(stage_plan, llm, client, cap,
                                     upstream=upstream_info,
                                     tool_selection=tool_selection)
```

(Replace the existing `result = run_geom_specialist(stage_plan, llm, client, cap, upstream=upstream_info)` line.)

**3c.** `config.example.json` — add to `"policy"`:

```json
    "plan_detail": "balanced",
    "tool_selection": "hints",
```

(Keep valid JSON — the `plan_detail` line already exists; add `tool_selection` after it.)

**3d.** `README.md`:

- In the policy block (lines 82-89), add `"tool_selection": "hints",` after `"tool_iteration_cap": 12,`.
- In the options table (after the `--on-failure` row, line 154), add:

```
| `--tools {hints,all}` | Geom tool retrieval. `hints` (default): the geom specialist is offered a small per-stage tool subset (planner hint + a core set) and widens it on demand via `load_tools`. `all`: offer the full geom registry on every call (pre-retrieval behaviour). Overrides `policy.tool_selection`. |
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_cli.py tests/unit/test_graph.py -v`
Expected: PASS.

- [ ] **Step 5: Full unit sweep**

Run: `pytest tests/unit -q`
Expected: PASS, count = baseline + ~14.

- [ ] **Step 6: Commit**

```bash
git add salome_agent/graph.py salome_agent/cli.py config.example.json README.md tests/unit/test_cli.py tests/unit/test_graph.py
git commit -m "feat: wire policy.tool_selection and the --tools flag to the geom specialist"
```

---

## Task 6: Live verification and token re-measurement

**Files:** none (verification only). May produce a follow-up commit if a bug is found.

**Preconditions:**
- A real `config.json` with working LLM creds, run from the repo root.
- **Fully restart the SALOME process** and relaunch the agent server (Task 1 changed `executor/geom_tools.py`; the plugin server caches it — see Global Constraints).
- Note the baseline: run `aaa61d4c15bd` (toy-house prompt) measured ~300k total input tokens, tool schemas ≈ 202k of that.

- [ ] **Step 1: Hinted-stage smoke — subset offered, no `load_tools` needed**

Run: `salome-agent "a 20mm cube, then 4 thin walls around it, colour the walls red" -vv`

Check `<run_dir>/llm_trace.jsonl` and the `-v` log:
- Each geom stage logs `geom stage: offered N/M tools` with N ≈ 8-12, not 40.
- The per-call `tools` array in the trace is the small set, every call.
- No `load_tools` call needed for this prompt (walls = make_box + translate + set_color, all core-or-hinted).
- `result.py` output is equivalent to a pre-change run (walls enclose the cube, red).

- [ ] **Step 2: Vague-plan smoke — model calls `load_tools`**

Run: `salome-agent "make a hollow enclosure around a 30mm sphere, 3mm walls" -vv`

Check:
- The geom stage starts with ~7 tools (no hint → core only).
- The trace shows a `load_tools` call (expect `categories=["swept"]` or `names=["make_thick_solid"]`), then the widened set on the next call.
- The `-v` log shows `load_tools(...) -> +K, N available`.

- [ ] **Step 3: Re-measure total input tokens**

Re-run the toy-house prompt used for baseline `aaa61d4c15bd`:

Run: `salome-agent "<the toy-house prompt>" -vv`

Sum input across all calls. Since the `-vv` trace under-reports (task 195 / it logs tool names not schemas), compute the real number: for each call, `len(json.dumps(tools_sent)) / 4` approx, summed. Compare to ~300k baseline.
- **Target: < 100k total input.** (Spec: MVP cuts 85-90% of the ~202k schema cost.)
- Record the number in the task-193 notes / GH issue #8.

- [ ] **Step 4: `--tools all` regression check**

Run: `salome-agent "a 10mm cube" --tools all -vv`
Expected: geom stage logs `offered 40/40 tools`; behaviour identical to pre-change. Confirms the safety valve.

- [ ] **Step 5: Commit any fixes, then finish the branch**

If Steps 1-4 surfaced a bug, fix it TDD-style (unit test first), commit with a bare `fix:` subject.

Then follow **superpowers:finishing-a-development-branch**: full `pytest tests/unit -q`, ff-merge `feat/193-geom-tool-retrieval` into `main`, delete the branch. User pushes.

- [ ] **Step 6: Update project memory**

Update `salome-agent-project.md` (session entry) and the remember-da-project task 193 → `done`, with the measured before/after token numbers. Note that 185's future tools must get `TOOL_CATEGORY` entries (the guard test enforces it).

---

## Self-Review

**Spec coverage:**

| Spec section | Task |
|---|---|
| §4.1 tool categories | Task 1 |
| §4.2 `select_tools` module (`CORE_TOOLS`, `hint_tool_names`, `initial_offered`, `resolve_load`) | Task 3 |
| §4.3 `load_tools` meta-tool schema/description | Task 4 (3b) |
| §4.4 loop rework, per-iteration tool list, `load_tools` handling, retry recompute | Task 4 (3e-3g). Retry: geom specialist is re-invoked fresh by `_run_stage_node` on retry (it is not passed `last_error`), so `initial_offered` is recomputed automatically — no explicit retry branch needed. |
| §4.5 prompt paragraph | Task 4 (3c) |
| §4.6 `policy.tool_selection` knob + `--tools` flag | Task 5 |
| §4.7 logging (stage start, per `load_tools`) | Task 4 (3e, 3g) |
| §5 worked example | Task 6 Steps 1-2 |
| §6 unit tests | Tasks 1, 3, 4, 5 |
| §6 live tests | Task 6 |
| §7 "do now" = categories ahead of 185 | Task 1 (first task, standalone commit) |
| §8 risk: stale server omits `category` | `resolve_load` drops unknown categories → logged no-op (Task 3 docstring, Task 4 log line) |

**Deviation from spec:** §3 says "no change to the planner." Reality check during planning found the planner does NOT currently emit `operations`/`tools` hints (schema is `module/intent/params/inputs` only — `interactive.py:_PLAN_KEYS`). Without a hint source the mechanism degrades to bare `CORE_TOOLS` on every stage. Task 2 adds an optional `tools` output field to the planner — the spec's own §9 lists this as the clean path ("Let the planner emit an explicit categories field per stage"). Approved by the user 2026-09-03.

**Placeholder scan:** `test_edit_round_trip_preserves_the_tools_hint` in Task 2 references `_normalise_plan` with a caveat to adapt to the real helper name — the executor must read `interactive.py` lines 1-45 first. Task 5's optional `test_graph.py` addition is gated on an existing harness. Both are flagged, not silent.

**Type consistency:** `tool_selection: str` default `"hints"` consistent across `run_geom_specialist` / `_run` (Task 4) and `config["policy"]["tool_selection"]` / `args.tools` (Task 5). `resolve_load(names, categories, all_defs)` signature identical in spec §4.2, Task 3 module, and Task 4 call site. `all_defs` is `name -> list_tools() entry` everywhere. `offered: set[str]` throughout Task 4.
