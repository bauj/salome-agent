# Geom Spatial Coordination Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the geom specialist the spatial facts it needs to place parts — upstream shapes' bounding boxes and a `measure` tool — so it stops guessing translation offsets.

**Architecture:** Add a `measure` tool to the geom tool registry (returns bbox/size/centre for named shapes, no codegen). Wire the existing `select_upstream()` output through the geom branch of `_run_stage_node`: the node calls `measure` on the upstream names and injects the results into the geom specialist's opening prompt. Strengthen the geom system prompt to mandate a `measure` self-check after transforms and booleans.

**Tech Stack:** Python 3.10, pytest, the in-repo `executor` / `salome_agent` packages. No new dependencies.

**Spec:** No standalone spec file. Design is captured in remember-da-project task **192** ("Spatial coordination: give the geom specialist the facts to place parts (A+C+B)") and the brainstorm in the session that produced this plan. Relevant background: tasks 174 (upstream handoff for smesh/paravis), 188 (`shape_info` already attached to every ok geom tool result), 185 (subshape selection — explicitly OUT of scope here).

## Global Constraints

- Target Python **3.10** (no `typing.NotRequired` — use `typing_extensions`; already imported in `executor/geom_tools.py`).
- Work on a feature branch, not `main` (`fix/192-geom-spatial-coordination` or similar). ff-merge + delete branch when live-verified; the user pushes.
- Commit messages: **subject line only**, no body, no `Co-Authored-By` / `Session` trailers.
- Never bake this box's SALOME install path into repo code/comments/docs.
- The `measure` tool must NOT emit a `CallRecord` — it is a query, like `check_shape` / `list_study_objects`, and must never appear in a generated `result.py`.
- Any change touching the LLM call path needs at least one live-LLM smoke run before it is considered done (see Task 5). Unit tests with a `MagicMock` LLM cannot catch prompt-wiring bugs.

---

### Task 1: `measure` tool in the geom registry

**Files:**
- Modify: `executor/geom_tools.py` (add `tool_measure`, register it in `TOOLS`, `TOOL_SCHEMAS`, `TOOL_DESCRIPTIONS`)
- Test: `tests/unit/test_geom_tools.py`

**Interfaces:**
- Consumes: `_shape_info(geompy, shape) -> dict | None` (existing, in `executor/geom_tools.py`), `executor.geom_bindings: dict[str, <shape>]`, `executor.namespace["geompy"]`.
- Produces: tool name `"measure"`. Handler `tool_measure(executor, args: dict) -> dict` where `args == {"names": list[str]}`. Return shape:
  `{"ok": True, "records": [], "sanity": {"ok": True}, "measurements": {name: {"bbox": [...], "size": [...], "centre": [...]} | None}}`.
  Every requested name is a key; an unknown or unbound name maps to `None`.

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/test_geom_tools.py` (top-level, after the `TestToolDescriptions` class):

```python
class TestMeasureTool:
    def _executor(self, bbox_by_shape):
        geompy = MagicMock()
        geompy.BoundingBox.side_effect = lambda s: bbox_by_shape[s]
        return SimpleNamespace(namespace={"geompy": geompy},
                               geom_bindings={n: n for n in bbox_by_shape})

    def test_measure_returns_shape_info_per_name(self):
        ex = self._executor({"box_1": [0.0, 10.0, 0.0, 20.0, 0.0, 30.0]})
        out = geom_tools.tool_measure(ex, {"names": ["box_1"]})
        assert out["ok"] is True
        assert out["records"] == []
        assert out["measurements"]["box_1"] == {
            "bbox": [0.0, 10.0, 0.0, 20.0, 0.0, 30.0],
            "size": [10.0, 20.0, 30.0],
            "centre": [5.0, 10.0, 15.0],
        }

    def test_measure_unknown_name_maps_to_none(self):
        ex = self._executor({"box_1": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]})
        out = geom_tools.tool_measure(ex, {"names": ["box_1", "ghost"]})
        assert out["measurements"]["ghost"] is None
        assert out["measurements"]["box_1"] is not None

    def test_measure_is_registered_with_schema_and_description(self):
        assert "measure" in geom_tools.TOOLS
        assert geom_tools.TOOL_SCHEMAS["measure"]["required"] == ["names"]
        assert geom_tools.TOOL_DESCRIPTIONS["measure"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/test_geom_tools.py::TestMeasureTool -v`
Expected: FAIL — `AttributeError: module 'executor.geom_tools' has no attribute 'tool_measure'` (and `KeyError` on the registry test).

- [ ] **Step 3: Implement `tool_measure` and register it**

In `executor/geom_tools.py`, add the handler next to `tool_list_study_objects` (around line 1150):

```python
def tool_measure(executor, args: dict):
    geompy = executor.namespace["geompy"]
    out = {}
    for name in args["names"]:
        shape = executor.geom_bindings.get(name)
        out[name] = _shape_info(geompy, shape) if shape is not None else None
    return {"ok": True, "records": [], "sanity": {"ok": True}, "measurements": out}
```

In the `TOOLS` dict (around line 1196), add after `"list_study_objects": tool_list_study_objects,`:

```python
    "measure": tool_measure,
```

In `TOOL_SCHEMAS` (around line 1592, after the `list_study_objects` entry), add:

```python
    "measure": {
        "type": "object",
        "properties": {
            "names": {"type": "array", "items": {"type": "string"},
                      "description": "published shape names to measure"},
        },
        "required": ["names"],
    },
```

In `TOOL_DESCRIPTIONS` (around line 1636, after the `list_study_objects` entry), add:

```python
    "measure": "Report bbox/size/centre (mm) of one or more published shapes by name. Use it to check where a shape landed and to see upstream shapes you did not build.",
```

Do NOT add a `measure` entry to `ARG_TYPES` — `names` is a list of strings, no numeric coercion.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit/test_geom_tools.py -v`
Expected: PASS (including the pre-existing `TestToolDescriptions::test_every_llm_facing_tool_has_a_description` guard, which now also covers `measure`).

- [ ] **Step 5: Commit**

```bash
git add executor/geom_tools.py tests/unit/test_geom_tools.py
git commit -m "feat: add a measure tool that reports bbox/size/centre of published shapes"
```

---

### Task 2: `run_geom_specialist` accepts and injects upstream shape info

**Files:**
- Modify: `salome_agent/specialists/geom.py` (`run_geom_specialist` and `_run` signatures + opening prompt)
- Test: `tests/unit/test_specialist_geom.py`

**Interfaces:**
- Consumes: nothing new.
- Produces:
  - `run_geom_specialist(stage_plan, llm, client, iteration_cap, upstream=None)` — `upstream` is `dict[str, dict | None]` mapping a published name to its `{"bbox","size","centre"}` (or `None`). Default `None` == first stage / no upstream.
  - `_run(stage_plan, llm, client, iteration_cap, upstream=None)` — same param, forwarded.
  - When `upstream` is truthy, the opening user message gains a block:
    `\n\nObjects already in the study — reference them by these exact names:\n  <name> — bbox <bbox>, size <size>, centre <centre>` (one line per name; a name with `None` info is listed as `  <name> — (geometry unavailable)`).

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/test_specialist_geom.py`:

```python
def test_upstream_shape_info_is_injected_into_the_opening_prompt():
    llm = MagicMock()
    llm.chat.return_value = {"tool_call": "finish_stage", "args": {"result_names": ["x"]}}
    client = MagicMock()
    client.list_tools.return_value = {"tools": [{"name": "make_box"}]}
    run_geom_specialist(
        {"module": "geom", "intent": "walls", "params": {}, "inputs": []},
        llm, client, iteration_cap=3,
        upstream={"fuse_1": {"bbox": [0, 100, 0, 100, 0, 20],
                             "size": [100, 100, 20], "centre": [50, 50, 10]}},
    )
    sent = llm.chat.call_args.kwargs["messages"][1]["content"]
    assert "fuse_1" in sent
    assert "centre [50, 50, 10]" in sent


def test_no_upstream_leaves_the_prompt_unchanged():
    llm = MagicMock()
    llm.chat.return_value = {"tool_call": "finish_stage", "args": {"result_names": ["x"]}}
    client = MagicMock()
    client.list_tools.return_value = {"tools": [{"name": "make_box"}]}
    run_geom_specialist({"module": "geom", "intent": "box", "params": {}, "inputs": []},
                        llm, client, iteration_cap=3)
    sent = llm.chat.call_args.kwargs["messages"][1]["content"]
    assert "already in the study" not in sent
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/test_specialist_geom.py::test_upstream_shape_info_is_injected_into_the_opening_prompt tests/unit/test_specialist_geom.py::test_no_upstream_leaves_the_prompt_unchanged -v`
Expected: FAIL — `TypeError: run_geom_specialist() got an unexpected keyword argument 'upstream'`.

- [ ] **Step 3: Implement the signature + injection**

In `salome_agent/specialists/geom.py`:

Change `run_geom_specialist` (line 43):

```python
def run_geom_specialist(stage_plan: dict, llm, client, iteration_cap: int,
                        upstream: dict | None = None) -> dict:
    try:
        return _run(stage_plan, llm, client, iteration_cap, upstream)
    except (SessionDied, SessionPoisoned) as e:
        return {"fatal": f"SALOME agent server unavailable: {e}"}
```

Change `_run` (line 50) header and opening `messages`:

```python
def _run(stage_plan: dict, llm, client, iteration_cap: int,
         upstream: dict | None = None) -> dict:
    content = str(stage_plan)
    if upstream:
        lines = []
        for name, info in upstream.items():
            if info:
                lines.append(f"  {name} — bbox {info['bbox']}, size {info['size']}, "
                             f"centre {info['centre']}")
            else:
                lines.append(f"  {name} — (geometry unavailable)")
        content += ("\n\nObjects already in the study — reference them by these "
                    "exact names:\n" + "\n".join(lines))
    messages = [{"role": "system", "content": GEOM_SYSTEM_PROMPT},
                {"role": "user", "content": content}]
```

Leave the rest of `_run` unchanged.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit/test_specialist_geom.py -v`
Expected: PASS (all, including the pre-existing tests that call `run_geom_specialist` with no `upstream`).

- [ ] **Step 5: Commit**

```bash
git add salome_agent/specialists/geom.py tests/unit/test_specialist_geom.py
git commit -m "feat: feed the geom specialist upstream shapes' bbox/size/centre in its opening prompt"
```

---

### Task 3: `_run_stage_node` measures upstream shapes and passes them to the geom specialist

**Files:**
- Modify: `salome_agent/graph.py` (`_run_stage_node`, geom branch, lines ~74-91)
- Test: `tests/unit/test_graph.py`

**Interfaces:**
- Consumes: `select_upstream(stage_plan, stages, idx) -> list[str]` (existing, `salome_agent/graph.py:21`); `client.call_tool(name, args) -> dict` (existing, `ExecutorClient.call_tool`); `run_geom_specialist(..., upstream=...)` (Task 2).
- Produces: no new public interface. Behaviour: for a geom stage with `idx > 0`, the node calls `client.call_tool("measure", {"names": <upstream>})` when `upstream` is non-empty and `client` is not `None`, and forwards `result["measurements"]` as `run_geom_specialist(..., upstream=...)`. If the `measure` call fails (`ok` falsey or raises) the node logs a warning and proceeds with `upstream=None`.

- [ ] **Step 1: Write the failing test**

Check `tests/unit/test_graph.py` for the existing helper/fixture style first (there are already `_run_stage_node` tests). Add:

```python
def test_geom_stage_measures_upstream_and_forwards_it(monkeypatch):
    from salome_agent import graph

    captured = {}

    def fake_run_geom_specialist(stage_plan, llm, client, cap, upstream=None):
        captured["upstream"] = upstream
        return {"call_log": [], "result_names": ["walls_1"]}

    monkeypatch.setattr("salome_agent.specialists.geom.run_geom_specialist",
                        fake_run_geom_specialist)

    client = MagicMock()
    client.call_tool.return_value = {
        "ok": True,
        "measurements": {"fuse_1": {"bbox": [0, 100, 0, 100, 0, 20],
                                    "size": [100, 100, 20], "centre": [50, 50, 10]}},
    }
    state = {
        "stage_idx": 1,
        "plan": [
            {"module": "geom", "intent": "beams", "params": {}, "inputs": []},
            {"module": "geom", "intent": "walls", "params": {}, "inputs": []},
        ],
        "stages": {
            0: {"module": "geom", "status": "ok", "result_names": ["fuse_1"]},
            1: {"module": "geom", "status": "pending"},
        },
        "config": {"policy": {"tool_iteration_cap": 5, "on_stage_failure": "abort"}},
        "last_error": None,
    }
    config = {"configurable": {"llm_client": MagicMock(), "executor_client": client}}

    graph._run_stage_node(state, config)

    client.call_tool.assert_called_once_with("measure", {"names": ["fuse_1"]})
    assert captured["upstream"] == {
        "fuse_1": {"bbox": [0, 100, 0, 100, 0, 20],
                   "size": [100, 100, 20], "centre": [50, 50, 10]}
    }


def test_first_geom_stage_does_not_call_measure(monkeypatch):
    from salome_agent import graph

    def fake_run_geom_specialist(stage_plan, llm, client, cap, upstream=None):
        return {"call_log": [], "result_names": ["box_1"]}

    monkeypatch.setattr("salome_agent.specialists.geom.run_geom_specialist",
                        fake_run_geom_specialist)
    client = MagicMock()
    state = {
        "stage_idx": 0,
        "plan": [{"module": "geom", "intent": "box", "params": {}, "inputs": []}],
        "stages": {0: {"module": "geom", "status": "pending"}},
        "config": {"policy": {"tool_iteration_cap": 5, "on_stage_failure": "abort"}},
        "last_error": None,
    }
    config = {"configurable": {"llm_client": MagicMock(), "executor_client": client}}
    graph._run_stage_node(state, config)
    client.call_tool.assert_not_called()
```

If `test_graph.py`'s existing tests build `state` differently (e.g. a builder helper), match that style — the keys above (`stage_idx`, `plan`, `stages`, `config`, `last_error`) are what `_run_stage_node` reads today; verify against the current source before finalizing.

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/test_graph.py::test_geom_stage_measures_upstream_and_forwards_it tests/unit/test_graph.py::test_first_geom_stage_does_not_call_measure -v`
Expected: FAIL — `test_geom_stage_measures_upstream...` fails on `client.call_tool.assert_called_once_with(...)` (never called) and `captured["upstream"]` is `None`.

- [ ] **Step 3: Implement the measure-and-forward logic**

In `salome_agent/graph.py`, `_run_stage_node`, replace the start of the `if module == "geom":` branch (currently lines ~74-77):

```python
    if module == "geom":
        from salome_agent.specialists.geom import run_geom_specialist

        result = run_geom_specialist(stage_plan, llm, client, cap)
```

with:

```python
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
```

`select_upstream` and `log` are already imported/defined at module scope in `graph.py` — no new import.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit/test_graph.py -v`
Expected: PASS (all, including pre-existing `_run_stage_node` tests — they use `stage_idx: 0` or a mocked specialist, so the new branch is a no-op for them).

- [ ] **Step 5: Commit**

```bash
git add salome_agent/graph.py tests/unit/test_graph.py
git commit -m "feat: measure upstream shapes and hand them to the geom specialist for multi-stage placement"
```

---

### Task 4: Mandate a `measure` self-check in the geom system prompt

**Files:**
- Modify: `salome_agent/specialists/geom.py` (`GEOM_SYSTEM_PROMPT`)
- Test: `tests/unit/test_specialist_geom.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `GEOM_SYSTEM_PROMPT` contains an explicit instruction to `measure` (or read `shape_info`) after transforms and booleans.

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/test_specialist_geom.py`:

```python
def test_system_prompt_mandates_a_placement_self_check():
    from salome_agent.specialists.geom import GEOM_SYSTEM_PROMPT
    p = GEOM_SYSTEM_PROMPT.lower()
    assert "measure" in p
    assert "fuse" in p and "translate" in p
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/test_specialist_geom.py::test_system_prompt_mandates_a_placement_self_check -v`
Expected: FAIL — `assert "measure" in p` (the current prompt says "check shape_info" but never names the `measure` tool, and does not list the trigger operations).

- [ ] **Step 3: Update the prompt**

In `salome_agent/specialists/geom.py`, replace the existing `shape_info` paragraph in `GEOM_SYSTEM_PROMPT`:

```
Every successful tool result reports the shape's bbox, size and centre (mm)
in `shape_info` — check it before moving on, especially after a transform
or a boolean, to confirm each part landed where you intended.
```

with:

```
Every successful tool result reports the shape's bbox, size and centre (mm)
in `shape_info`. After each translate, rotate, fuse or cut, read that
`shape_info` (or call `measure`) and confirm the shape's centre and extent
match what you intended before continuing — if a part is in the wrong
place, correct it now, not at the end. Call `measure` on any upstream
object whose position you need but did not build yourself.
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit/test_specialist_geom.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add salome_agent/specialists/geom.py tests/unit/test_specialist_geom.py
git commit -m "feat: geom prompt mandates a measure self-check after transforms and booleans"
```

---

### Task 5: Full suite + live-LLM smoke run

**Files:**
- No production code. May touch `docs/superpowers/plans/2026-09-01-geom-spatial-coordination.md` (check off boxes) and remember-da-project task 192.

**Interfaces:** none.

- [ ] **Step 1: Run the full unit suite**

Run: `.venv/bin/python -m pytest tests/unit/ -q`
Expected: PASS, count = previous total (192) + new tests from Tasks 1-4 (8) = **200**. If any pre-existing test fails, fix the regression before proceeding.

- [ ] **Step 2: Start the SALOME agent server**

The user runs this — it needs a live SALOME GUI (`salome withsession` + the agent-server plugin) or the headless server (`scripts/start-headless-server.sh`). Confirm `~/.salome_agent/server.json` exists and `config.json` is present in the repo root.

- [ ] **Step 3: Live smoke run with the trace on**

Run: `salome-agent "make me a union of two perpendicular beams on the tip of each other. Then add 4 walls around it. Color the walls in red" -vv`

- [ ] **Step 4: Inspect the trace**

Run: `jq -r 'select(.request.tools!=[]) | .response.message.tool_calls[0].function.name' <work_dir>/<run_id>/llm_trace.jsonl`

Verify:
- Stage 2's opening prompt (line where `request.messages[1].content` mentions "already in the study") lists the stage-1 result name with a bbox.
- The stage-2 specialist references that upstream name and/or calls `measure`.
- The generated `result.py` walls enclose the beams' bbox (no large gaps, walls span the beam extent) — a visual check in the GUI plus reading the coordinates in `result.py`.

- [ ] **Step 5: Record the outcome**

Update remember-da-project task 192 with the run id, what improved, and any residual placement error. If the walls still do not enclose the beams, that is a prompt-tuning follow-up (not a rollback) — note it and stop.

- [ ] **Step 6: Merge**

```bash
git checkout main
git merge --ff-only fix/192-geom-spatial-coordination
git branch -d fix/192-geom-spatial-coordination
```

The user pushes.

---

## Self-Review

**1. Spec coverage** (design items A, B, C from task 192):
- **C** (`measure` tool) → Task 1. ✓
- **A** (upstream handoff into geom specialist) → Task 2 (specialist side) + Task 3 (graph wiring). ✓
- **B** (prompt self-check) → Task 4. ✓
- Live verification (Global Constraint) → Task 5. ✓
- Out of scope and correctly absent: subshape selection / `explode` (task 185), vision feedback loop.

**2. Placeholder scan:** No "TBD"/"handle edge cases"/"similar to Task N". The `measure`-failure path in Task 3 is spelled out (log + `upstream=None`). Task 3 Step 1 flags that `test_graph.py` may use a state builder — this is a real instruction to check the current source, with the concrete keys listed, not a placeholder.

**3. Type consistency:**
- `tool_measure` return key `"measurements"` — used identically in Task 1 (definition), Task 3 (`m.get("measurements")`), Task 3 test (`client.call_tool.return_value`).
- `run_geom_specialist(..., upstream=None)` — `upstream` is `dict[str, dict | None]` in Tasks 2 and 3; Task 3 passes `m.get("measurements")` which is exactly that shape.
- `_shape_info` returns keys `bbox` / `size` / `centre` — matched in Task 1's expected dict and Task 2's prompt f-string.
- `select_upstream` returns `list[str]` — Task 3 passes it straight to `{"names": ...}`. ✓
