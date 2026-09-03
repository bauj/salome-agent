# GEOM Subshape Selection & Query Family Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the geom specialist a way to discover subshape ids (edges/faces) and feed them to `fillet_edges` / `chamfer_edges` / `get_subshape`, plus two measurement queries, so a request like "fillet the top four edges of a cube" stops being a blind integer guess.

**Architecture:** Four new tools plus one codegen capability. `get_subshapes`, `min_distance` and `basic_properties` are **query tools** — they run live, put data (id lists, coordinates, numbers) into the tool result and emit **no `CallRecord`**; the model reads the payload and passes resolved ids as literals to a downstream call. `explode` and `get_subshape` **do** produce shapes and emit `CallRecord`s; `explode` returns a Python list from one geompy call, which needs a new `list_call` codegen emit (`_sub_N = geompy.Fn(...)` then indexed unpacking).

**Tech Stack:** Python 3.10+, geompy (SALOME GEOM Python API), pytest. Executor tool registry in `executor/geom_tools.py`; codegen in `salome_agent/codegen.py`; specialist loop in `salome_agent/specialists/geom.py`.

**Spec:** `docs/superpowers/specs/2026-08-31-geom-tool-catalogue.md` section 10 ("Deferred follow-up — subshape selection & query family (Task 176b)"). GH issue: https://github.com/bauj/salome-agent/issues/12

## Global Constraints

- **Python floor 3.10.** No bare-`typing` `NotRequired`/`Required`/`Self`; use `typing_extensions` or `total=False`. `from __future__ import annotations` before runtime-evaluated `X | None`.
- **No local SALOME install paths** in code, comments, docs or config — resolve at runtime from `$SALOME_ROOT` / `salome.root` in config.json. Live tests skip when neither is set (`tests/salome/_salome_root.py`).
- **Provider-agnostic; minimal comments.** No hardcoded provider specifics.
- **Commits:** subject line only, no body, no trailers. Append ` (#12)` to every commit subject in this plan (issue-tagged).
- **Staging:** `git add` named files only, never `git add -A`.
- **Branch:** do this work on a feature branch off `main` (`git checkout -b feat/185-subshape-selection`), not on `main`.
- **Server cache gotcha:** `executor/geom_tools.py` is imported and cached by the long-lived SALOME agent server (`agent_server.core` / `agent_server.backend`). After editing it, the SALOME process must be fully restarted before a live run picks up the change — a headless server restart is `kill` + `scripts/start-headless-server.sh`. Unit tests import fresh each run and are unaffected.
- **Do not fuzz fillet radius against a live server** — a too-large radius hangs `MakeFillet` in OCC and wedges the server (`ServerBusy` cascade). Keep live fillet radii small relative to the shape (e.g. 3 mm on a 20 mm cube).

---

## File Structure

| File | Responsibility | Change |
|------|----------------|--------|
| `salome_agent/codegen.py` | Turn the `CallRecord` log into `result.py` source | Add a `list_call` emit branch to `generate()` |
| `executor/geom_tools.py` | Executor-side tool handlers + registries | Add `tool_explode`, `tool_get_subshape`, `tool_get_subshapes`, `tool_min_distance`, `tool_basic_properties`; register each in `TOOLS`, `TOOL_CATEGORY`, `TOOL_SCHEMAS`, `TOOL_DESCRIPTIONS`, `ARG_TYPES` |
| `salome_agent/specialists/geom.py` | Geom specialist loop + system prompt | Extend `_compact_result` keep-list with query payload keys; add a query-then-act paragraph to `_GEOM_BASE` |
| `salome_agent/specialists/tool_selection.py` | Per-stage tool retrieval | Add `get_subshapes` to `CORE_TOOLS` |
| `tests/unit/test_codegen.py` | codegen unit tests | New tests for `list_call` |
| `tests/unit/test_geom_tools.py` | geom_tools unit tests (MagicMock geompy) | New test classes for each tool |
| `tests/unit/test_specialist_geom.py` | specialist unit tests | New tests for compaction + prompt |
| `tests/unit/test_tool_selection.py` | tool-selection unit tests | Update `CORE_TOOLS` assertion |
| `tests/salome/test_geom_tools_live.py` | live geompy tests (real SALOME) | New live test class per tool |
| `README.md` | user docs | Update the geom-tool count / known-limitations bullets |

---

## Task 1: codegen `list_call` emit — one geompy call, many published out_vars

**Files:**
- Modify: `salome_agent/codegen.py` (`generate()`, the per-call `for` loop, currently around lines 73–95)
- Test: `tests/unit/test_codegen.py`

**Interfaces:**
- Consumes: a `CallRecord` dict `{"fn": str, "args": list[ArgRef], "out_vars": list[str], "publish": list[str], "emit": "list_call"}` where `len(out_vars) == len(publish) == N` (N subshapes from one call).
- Produces: for a reachable `list_call` record, emits `_sub_<i> = geompy.<fn>(<args>)` followed, **per out_var that is in the reachable set**, by `<out_var> = _sub_<i>[<index>]` and `geompy.addToStudy(<out_var>, '<publish name>')`. `<i>` is the record's index in `call_log` (unique per call). Unreachable out_vars of a reachable call are skipped (the whole geompy call still runs — you cannot partially explode).

- [ ] **Step 1: Write the failing tests**

Add to `tests/unit/test_codegen.py`:

```python
def test_list_call_emit_unpacks_and_publishes_reachable_subshapes():
    call_log = [
        {"fn": "MakeBoxDXDYDZ", "args": [{"kind": "literal", "value": 10}] * 3,
         "out_vars": ["box_1"], "publish": ["box_1"]},
        {"fn": "SubShapeAllSortedCentres",
         "args": [{"kind": "ref", "value": "box_1"}, {"kind": "literal", "value": 6}],
         "out_vars": ["box_1_edge_1", "box_1_edge_2", "box_1_edge_3"],
         "publish": ["box_1_edge_1", "box_1_edge_2", "box_1_edge_3"],
         "emit": "list_call"},
        {"fn": "MakeFillet",
         "args": [{"kind": "ref", "value": "box_1"}, {"kind": "literal", "value": 2},
                  {"kind": "literal", "value": 6},
                  {"kind": "literal", "value": [2]}],
         "out_vars": ["fillet_1"], "publish": ["fillet_1"]},
    ]
    # fillet_1 depends on box_1 only (edge id 2 is a literal, not a ref), so the
    # explode call is NOT reachable from result_names=["fillet_1"].
    source = generate(call_log, result_names=["fillet_1"])
    ast.parse(source)
    assert "SubShapeAllSortedCentres" not in source

    # Now name an exploded subshape as a result -> the list_call is emitted,
    # but only the reachable index is unpacked/published.
    source = generate(call_log, result_names=["box_1_edge_2"])
    ast.parse(source)
    assert "_sub_1 = geompy.SubShapeAllSortedCentres(box_1, 6)" in source
    assert "box_1_edge_2 = _sub_1[1]" in source
    assert "geompy.addToStudy(box_1_edge_2, 'box_1_edge_2')" in source
    assert "box_1_edge_1 = _sub_1[0]" not in source
    assert "box_1_edge_3" not in source


def test_list_call_emit_unpacks_every_subshape_when_all_reachable():
    call_log = [
        {"fn": "MakeBoxDXDYDZ", "args": [{"kind": "literal", "value": 1}] * 3,
         "out_vars": ["box_1"], "publish": ["box_1"]},
        {"fn": "SubShapeAllSortedCentres",
         "args": [{"kind": "ref", "value": "box_1"}, {"kind": "literal", "value": 4}],
         "out_vars": ["box_1_face_1", "box_1_face_2"],
         "publish": ["box_1_face_1", "box_1_face_2"],
         "emit": "list_call"},
    ]
    source = generate(call_log, result_names=["box_1_face_1", "box_1_face_2"])
    ast.parse(source)
    assert "_sub_1 = geompy.SubShapeAllSortedCentres(box_1, 4)" in source
    assert "box_1_face_1 = _sub_1[0]" in source
    assert "box_1_face_2 = _sub_1[1]" in source
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/unit/test_codegen.py -k list_call -v`
Expected: FAIL — the `list_call` emit is unknown, so the calls are skipped and no `SubShapeAllSortedCentres` line is emitted (assertions on the emitted lines fail).

- [ ] **Step 3: Implement the `list_call` branch**

In `salome_agent/codegen.py`, inside `generate()`'s per-call loop, after the existing `elif emit == "gui_call":` block, add:

```python
        elif emit == "list_call":
            tmp = f"_sub_{i}"
            lines.append(f"{tmp} = geompy.{call['fn']}({args_src})")
            pubs = dict(zip(call["out_vars"], call.get("publish") or []))
            for idx, var in enumerate(call["out_vars"]):
                if var not in reachable:
                    continue
                lines.append(f"{var} = {tmp}[{idx}]")
                if var in pubs:
                    lines.append(f"geompy.addToStudy({var}, {pubs[var]!r})")
```

`reachable` is already in scope (computed just before the loop). `_reachable()` and `_reachable_with_display()` already handle multi-`out_var` calls correctly: a call is pulled in when `reachable & set(call["out_vars"])` is non-empty, and its arg refs are then added to `reachable`. No change needed there.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/unit/test_codegen.py -v`
Expected: PASS (all codegen tests, old and new).

- [ ] **Step 5: Commit**

```bash
git add salome_agent/codegen.py tests/unit/test_codegen.py
git commit -m "feat: codegen list_call emit for one-call many-subshape records (#12)"
```

---

## Task 2: specialist keeps query payloads across turns

**Files:**
- Modify: `salome_agent/specialists/geom.py` (`_compact_result`, the `for k in (...)` keep-tuple around line 112)
- Test: `tests/unit/test_specialist_geom.py`

**Interfaces:**
- Consumes: a tool result dict that may carry any of `ids` (`list[int]`), `centres` (`list[list[float]]`), `distance` (`float`), `properties` (`dict`) alongside the existing `ok`/`sanity`.
- Produces: `_compact_result` retains those keys on compacted (non-latest) tool-result messages, so the model still sees an edge-id list it queried three turns ago.

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/test_specialist_geom.py`:

```python
def test_compact_result_keeps_query_payload_keys():
    from salome_agent.specialists.geom import _compact_result

    full = {"ok": True, "records": [{"fn": "x", "args": [], "out_vars": [], "publish": None}],
            "sanity": {"ok": True},
            "ids": [3, 7, 11, 15], "centres": [[0, 0, 20], [10, 0, 20]],
            "distance": 4.2, "properties": {"length": 40.0, "area": 0.0, "volume": 0.0}}
    compact = _compact_result(full)
    assert compact["ids"] == [3, 7, 11, 15]
    assert compact["centres"] == [[0, 0, 20], [10, 0, 20]]
    assert compact["distance"] == 4.2
    assert compact["properties"] == {"length": 40.0, "area": 0.0, "volume": 0.0}
    assert "records" not in compact  # still stripped
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/pytest tests/unit/test_specialist_geom.py::test_compact_result_keeps_query_payload_keys -v`
Expected: FAIL — `KeyError`/`assert` on `compact["ids"]` (key dropped).

- [ ] **Step 3: Extend the keep-tuple**

In `salome_agent/specialists/geom.py`, `_compact_result`, change:

```python
    for k in ("result_shape", "shape_info", "measurements", "names", "sanity"):
```

to:

```python
    for k in ("result_shape", "shape_info", "measurements", "names", "sanity",
              "ids", "centres", "distance", "properties"):
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/pytest tests/unit/test_specialist_geom.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add salome_agent/specialists/geom.py tests/unit/test_specialist_geom.py
git commit -m "feat: geom specialist retains query payloads across compacted turns (#12)"
```

---

## Task 3: `explode` tool — publish every subshape as its own named object

**Files:**
- Modify: `executor/geom_tools.py` (add `tool_explode`; register in `TOOLS`, `TOOL_CATEGORY`, `TOOL_SCHEMAS`, `TOOL_DESCRIPTIONS`, `ARG_TYPES`)
- Test: `tests/unit/test_geom_tools.py`, `tests/salome/test_geom_tools_live.py`

**Interfaces:**
- Consumes: `args = {"shape": <ref name>, "sub_type": "vertex"|"edge"|"wire"|"face"|"shell"|"solid"}`.
- Produces: tool result `{"ok": True, "records": [<one list_call CallRecord>], "sanity": {"ok": True}, "names": [<published names>]}`. Published names are `f"{shape}_{sub_type}_{i}"` for `i` from 1. The `CallRecord` is `{"fn": "SubShapeAllSortedCentres", "args": [{"kind": "ref", "value": shape}, {"kind": "literal", "value": <int shape-type code>}], "out_vars": names, "publish": names, "emit": "list_call"}`. Each subshape is bound in `executor.geom_bindings` and published to the study via `_publish`.

- [ ] **Step 1: Write the failing unit test**

Add to `tests/unit/test_geom_tools.py`:

```python
class TestExplodeTool:
    def _executor(self):
        geompy = MagicMock()
        geompy.ShapeType = {"EDGE": 6, "FACE": 4, "SOLID": 2, "VERTEX": 7,
                            "WIRE": 5, "SHELL": 3}
        geompy.SubShapeAllSortedCentres.return_value = ["e1", "e2", "e3"]
        geompy.BoundingBox.return_value = [0, 1, 0, 1, 0, 1]
        return SimpleNamespace(namespace={"geompy": geompy},
                               geom_bindings={"box_1": "BOX"},
                               name_prefix="")

    def test_explode_publishes_each_subshape_and_emits_one_list_call(self):
        ex = self._executor()
        out = geom_tools.tool_explode(ex, {"shape": "box_1", "sub_type": "edge"})
        assert out["ok"] is True
        assert out["names"] == ["box_1_edge_1", "box_1_edge_2", "box_1_edge_3"]
        assert ex.geom_bindings["box_1_edge_2"] == "e2"
        (rec,) = out["records"]
        assert rec["fn"] == "SubShapeAllSortedCentres"
        assert rec["emit"] == "list_call"
        assert rec["out_vars"] == rec["publish"] == out["names"]
        assert rec["args"][0] == {"kind": "ref", "value": "box_1"}
        assert rec["args"][1] == {"kind": "literal", "value": 6}
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/pytest tests/unit/test_geom_tools.py::TestExplodeTool -v`
Expected: FAIL — `AttributeError: module 'executor.geom_tools' has no attribute 'tool_explode'`.

- [ ] **Step 3: Implement `tool_explode` and register it**

In `executor/geom_tools.py`, add the handler (near the other shape-producing tools):

```python
def tool_explode(executor, args: dict):
    geompy = executor.namespace["geompy"]
    shape_name = args["shape"]
    shape = executor.geom_bindings[shape_name]
    sub_type = args["sub_type"]
    type_code = int(geompy.ShapeType[sub_type.upper()])
    subs = geompy.SubShapeAllSortedCentres(shape, type_code)
    names = []
    for i, sub in enumerate(subs, start=1):
        nm = f"{shape_name}_{sub_type}_{i}"
        _publish(executor, sub, nm)
        names.append(nm)
    return {
        "ok": True,
        "records": [{
            "fn": "SubShapeAllSortedCentres",
            "args": [{"kind": "ref", "value": shape_name},
                     {"kind": "literal", "value": type_code}],
            "out_vars": names,
            "publish": names,
            "emit": "list_call",
        }],
        "sanity": {"ok": True},
        "names": names,
    }
```

Register in each dict:
- `TOOLS`: `"explode": tool_explode,`
- `TOOL_CATEGORY`: `"explode": "feature",`
- `ARG_TYPES`: `"explode": {},`
- `TOOL_SCHEMAS`: 
  ```python
      "explode": {
          "type": "object",
          "properties": {
              "shape": {"type": "string", "description": "name of the shape to decompose"},
              "sub_type": {"type": "string",
                           "enum": ["vertex", "edge", "wire", "face", "shell", "solid"]},
          },
          "required": ["shape", "sub_type"],
      },
  ```
- `TOOL_DESCRIPTIONS`: `"explode": "Publish every subshape of the given sub_type (edge, face, ...) of \`shape\` as its own named object, sorted by centre for a deterministic order. Use get_subshapes instead if you only need ids.",`

- [ ] **Step 4: Run the unit test to verify it passes**

Run: `.venv/bin/pytest tests/unit/test_geom_tools.py -v`
Expected: PASS (including the coverage guards `test_every_tool_has_a_valid_category` and `test_every_llm_facing_tool_has_a_description`).

- [ ] **Step 5: Write the live test**

Add to `tests/salome/test_geom_tools_live.py`:

```python
class TestSubshapeSelection:
    def test_explode_edges_of_a_box(self, agent_client):
        agent_client.call_tool("make_box", {"dx": 10, "dy": 10, "dz": 10})
        out = agent_client.call_tool("explode", {"shape": "box_1", "sub_type": "edge"})
        assert out["ok"] is True, out.get("error")
        assert len(out["names"]) == 12
        assert out["records"][0]["emit"] == "list_call"

    def test_exploded_edge_feeds_codegen(self, agent_client):
        agent_client.call_tool("make_box", {"dx": 10, "dy": 10, "dz": 10})
        agent_client.call_tool("explode", {"shape": "box_1", "sub_type": "face"})
        script = assemble([{"module": "geom",
                            "call_log": agent_client._explode_call_log("box_1", "face"),
                            "result_names": ["box_1_face_1"]}]) if False else None
        # codegen coverage is unit-tested in test_codegen.py; this live test only
        # confirms explode itself runs against real geompy.
        assert agent_client.call_tool("list_study_objects", {})["ok"] is True
```

Keep the live test minimal — the codegen path is already covered by Task 1's unit tests. If `assemble` end-to-end coverage is wanted, fold it into Task 8's smoke instead. Drop the dead `if False` scaffold; the intent is just:

```python
    def test_explode_faces_of_a_box(self, agent_client):
        agent_client.call_tool("make_box", {"dx": 10, "dy": 10, "dz": 10})
        out = agent_client.call_tool("explode", {"shape": "box_1", "sub_type": "face"})
        assert out["ok"] is True, out.get("error")
        assert len(out["names"]) == 6
```

- [ ] **Step 6: Run the live test**

Ensure `$SALOME_ROOT` is set (or `salome.root` in config.json), then:
Run: `.venv/bin/pytest tests/salome/test_geom_tools_live.py::TestSubshapeSelection -v`
Expected: PASS. If it errors on `agent_client`, check the SALOME server was restarted (server-cache gotcha).

- [ ] **Step 7: Commit**

```bash
git add executor/geom_tools.py tests/unit/test_geom_tools.py tests/salome/test_geom_tools_live.py
git commit -m "feat: explode tool publishes each subshape as a named object (#12)"
```

---

## Task 4: `get_subshape` tool — build one object from an id list

**Files:**
- Modify: `executor/geom_tools.py` (add `tool_get_subshape`; register in the five dicts)
- Test: `tests/unit/test_geom_tools.py`, `tests/salome/test_geom_tools_live.py`

**Interfaces:**
- Consumes: `args = {"shape": <ref name>, "ids": [<int>, ...]}`.
- Produces: `{"ok": True, "records": [<one geompy_call CallRecord>], "sanity": {"ok": True}}`. Publishes one shape named `sub_<n>` (via `_next_name(executor, "sub")`). `CallRecord` = `{"fn": "GetSubShape", "args": [{"kind": "ref", "value": shape}, {"kind": "literal", "value": [int,...]}], "out_vars": [name], "publish": [name]}`. No `CheckShape` (subshapes of a valid shape are valid; `CheckShape` on a lone face can spuriously fail).

- [ ] **Step 1: Write the failing unit test**

Add to `tests/unit/test_geom_tools.py`:

```python
class TestGetSubshapeTool:
    def _executor(self):
        geompy = MagicMock()
        geompy.GetSubShape.return_value = "SUB"
        geompy.BoundingBox.return_value = [0, 1, 0, 1, 0, 1]
        return SimpleNamespace(namespace={"geompy": geompy},
                               geom_bindings={"box_1": "BOX"},
                               name_prefix="")

    def test_get_subshape_builds_and_publishes_one_shape(self):
        ex = self._executor()
        out = geom_tools.tool_get_subshape(ex, {"shape": "box_1", "ids": [3, 7]})
        assert out["ok"] is True
        assert ex.geom_bindings["sub_1"] == "SUB"
        (rec,) = out["records"]
        assert rec["fn"] == "GetSubShape"
        assert rec["args"] == [{"kind": "ref", "value": "box_1"},
                               {"kind": "literal", "value": [3, 7]}]
        assert rec["out_vars"] == rec["publish"] == ["sub_1"]

    def test_get_subshape_coerces_string_ids_to_int(self):
        ex = self._executor()
        out = geom_tools.tool_get_subshape(ex, {"shape": "box_1", "ids": ["3", "7"]})
        assert out["records"][0]["args"][1] == {"kind": "literal", "value": [3, 7]}
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/pytest tests/unit/test_geom_tools.py::TestGetSubshapeTool -v`
Expected: FAIL — `AttributeError: ... has no attribute 'tool_get_subshape'`.

- [ ] **Step 3: Implement and register**

Handler in `executor/geom_tools.py`:

```python
def tool_get_subshape(executor, args: dict):
    geompy = executor.namespace["geompy"]
    shape_name = args["shape"]
    shape = executor.geom_bindings[shape_name]
    ids = [int(i) for i in args["ids"]]
    sub = geompy.GetSubShape(shape, ids)
    name = _next_name(executor, "sub")
    _publish(executor, sub, name)
    return {
        "ok": True,
        "records": [{
            "fn": "GetSubShape",
            "args": [{"kind": "ref", "value": shape_name},
                     {"kind": "literal", "value": ids}],
            "out_vars": [name],
            "publish": [name],
        }],
        "sanity": {"ok": True},
    }
```

Register:
- `TOOLS`: `"get_subshape": tool_get_subshape,`
- `TOOL_CATEGORY`: `"get_subshape": "feature",`
- `ARG_TYPES`: `"get_subshape": {},`
- `TOOL_SCHEMAS`:
  ```python
      "get_subshape": {
          "type": "object",
          "properties": {
              "shape": {"type": "string"},
              "ids": {"type": "array", "items": {"type": "integer"},
                      "description": "subshape ids from explode or get_subshapes"},
          },
          "required": ["shape", "ids"],
      },
  ```
- `TOOL_DESCRIPTIONS`: `"get_subshape": "Build one new named shape from a list of subshape ids of \`shape\` (e.g. a group of faces). Get the ids from get_subshapes or explode.",`

- [ ] **Step 4: Run the unit tests to verify they pass**

Run: `.venv/bin/pytest tests/unit/test_geom_tools.py -v`
Expected: PASS.

- [ ] **Step 5: Write the live test**

Add to `tests/salome/test_geom_tools_live.py`, class `TestSubshapeSelection`:

```python
    def test_get_subshape_from_ids(self, agent_client):
        agent_client.call_tool("make_box", {"dx": 10, "dy": 10, "dz": 10})
        ids = agent_client.call_tool(
            "get_subshapes", {"shape": "box_1", "sub_type": "face", "where": "all"})["ids"]
        out = agent_client.call_tool("get_subshape", {"shape": "box_1", "ids": ids[:1]})
        assert out["ok"] is True, out.get("error")
        assert out["records"][0]["fn"] == "GetSubShape"
```

(Depends on `get_subshapes` from Task 5 — order Task 5 before running this live test, or run it after Task 5 lands.)

- [ ] **Step 6: Run the live test** (after Task 5)

Run: `.venv/bin/pytest tests/salome/test_geom_tools_live.py::TestSubshapeSelection::test_get_subshape_from_ids -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add executor/geom_tools.py tests/unit/test_geom_tools.py tests/salome/test_geom_tools_live.py
git commit -m "feat: get_subshape tool builds one shape from an id list (#12)"
```

---

## Task 5: `get_subshapes` query tool — return ids + centres, no CallRecord

**Files:**
- Modify: `executor/geom_tools.py` (add `tool_get_subshapes` + a small `_state_enum` helper; register in the five dicts)
- Modify: `salome_agent/specialists/tool_selection.py` (`CORE_TOOLS` gains `"get_subshapes"`)
- Test: `tests/unit/test_geom_tools.py`, `tests/unit/test_tool_selection.py`, `tests/salome/test_geom_tools_live.py`

**Interfaces:**
- Consumes: `args = {"shape": <ref>, "sub_type": <enum>, "where": <predicate>}` where `where` is one of:
  - `"all"`
  - `{"sorted_by": "x"|"y"|"z", "take": "max"|"min", "count": <int>}` (`take` defaults to `"max"`)
  - `{"by_size": "longest"|"shortest"|"largest"|"smallest", "count": <int>}` (length for edges, area for faces)
  - `{"on_plane": {"point": [x,y,z], "normal": [x,y,z]}, "state": "on"|"in"|"out"|"onin"|"onout"}` (`state` defaults to `"on"`)
- Produces: `{"ok": True, "records": [], "sanity": {"ok": True}, "ids": [<int>, ...], "centres": [[x,y,z], ...]}`. **No `CallRecord`** — nothing from this call reaches `result.py`; the model passes the returned ids as literals to `fillet_edges` / `chamfer_edges` / `get_subshape`.

- [ ] **Step 1: Write the failing unit tests**

Add to `tests/unit/test_geom_tools.py`. The `all` / `sorted_by` / `by_size` paths are pure Python over geompy return values and are unit-testable with a `MagicMock` geompy; `on_plane` delegates to a geompy call and is covered live only.

```python
class TestGetSubshapesQuery:
    def _executor(self, subs, ids, bboxes, props=None):
        geompy = MagicMock()
        geompy.ShapeType = {"EDGE": 6, "FACE": 4}
        geompy.SubShapeAllSortedCentres.return_value = subs
        geompy.SubShapeAllSortedCentresIDs.return_value = ids
        geompy.BoundingBox.side_effect = lambda s: bboxes[s]
        if props is not None:
            geompy.BasicProperties.side_effect = lambda s: props[s]
        return SimpleNamespace(namespace={"geompy": geompy},
                               geom_bindings={"box_1": "BOX"}, name_prefix="")

    def test_where_all_returns_every_id_and_centre(self):
        ex = self._executor(
            subs=["a", "b"], ids=[10, 11],
            bboxes={"a": [0, 2, 0, 0, 0, 0], "b": [0, 0, 0, 4, 0, 0]})
        out = geom_tools.tool_get_subshapes(
            ex, {"shape": "box_1", "sub_type": "edge", "where": "all"})
        assert out["records"] == []
        assert out["ids"] == [10, 11]
        assert out["centres"] == [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]

    def test_where_sorted_by_z_take_max_count_2(self):
        ex = self._executor(
            subs=["a", "b", "c", "d"], ids=[1, 2, 3, 4],
            bboxes={"a": [0, 0, 0, 0, 0, 0], "b": [0, 0, 0, 0, 10, 10],
                    "c": [0, 0, 0, 0, 5, 5], "d": [0, 0, 0, 0, 10, 10]})
        out = geom_tools.tool_get_subshapes(ex, {
            "shape": "box_1", "sub_type": "edge",
            "where": {"sorted_by": "z", "take": "max", "count": 2}})
        assert sorted(out["ids"]) == [2, 4]  # the two highest-z edges

    def test_where_by_size_longest_count_1_uses_edge_length(self):
        ex = self._executor(
            subs=["a", "b"], ids=[1, 2],
            bboxes={"a": [0, 0, 0, 0, 0, 0], "b": [0, 0, 0, 0, 0, 0]},
            props={"a": [3.0, 0.0, 0.0], "b": [9.0, 0.0, 0.0]})
        out = geom_tools.tool_get_subshapes(ex, {
            "shape": "box_1", "sub_type": "edge",
            "where": {"by_size": "longest", "count": 1}})
        assert out["ids"] == [2]

    def test_where_by_size_largest_count_1_uses_face_area(self):
        ex = self._executor(
            subs=["a", "b"], ids=[1, 2],
            bboxes={"a": [0, 0, 0, 0, 0, 0], "b": [0, 0, 0, 0, 0, 0]},
            props={"a": [0.0, 2.0, 0.0], "b": [0.0, 8.0, 0.0]})
        out = geom_tools.tool_get_subshapes(ex, {
            "shape": "box_1", "sub_type": "face",
            "where": {"by_size": "largest", "count": 1}})
        assert out["ids"] == [2]
```

Also update `tests/unit/test_tool_selection.py`:

```python
def test_core_tools_includes_get_subshapes():
    assert "get_subshapes" in CORE_TOOLS
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/bin/pytest tests/unit/test_geom_tools.py::TestGetSubshapesQuery tests/unit/test_tool_selection.py::test_core_tools_includes_get_subshapes -v`
Expected: FAIL — no `tool_get_subshapes`; `get_subshapes` not in `CORE_TOOLS`.

- [ ] **Step 3: Implement `tool_get_subshapes` and the state-enum helper**

In `executor/geom_tools.py`:

```python
def _state_enum(state: str):
    """Map a "where" state string to the GEOM.ST_* enum value."""
    from salome.geom import geomBuilder
    g = geomBuilder.GEOM
    return {"on": g.ST_ON, "in": g.ST_IN, "out": g.ST_OUT,
            "onin": g.ST_ONIN, "onout": g.ST_ONOUT}[state]


def _centre(geompy, shape):
    xmin, xmax, ymin, ymax, zmin, zmax = geompy.BoundingBox(shape)
    r = lambda v: round(v, 4)
    return [r((xmin + xmax) / 2), r((ymin + ymax) / 2), r((zmin + zmax) / 2)]


def tool_get_subshapes(executor, args: dict):
    geompy = executor.namespace["geompy"]
    shape = executor.geom_bindings[args["shape"]]
    sub_type = args["sub_type"]
    type_code = int(geompy.ShapeType[sub_type.upper()])
    where = args["where"]

    subs = list(geompy.SubShapeAllSortedCentres(shape, type_code))
    all_ids = list(geompy.SubShapeAllSortedCentresIDs(shape, type_code))
    pairs = list(zip(all_ids, subs))  # aligned: sorted-by-centre order

    if where == "all":
        selected = pairs
    elif "sorted_by" in where:
        axis = {"x": 0, "y": 1, "z": 2}[where["sorted_by"]]
        take = where.get("take", "max")
        n = int(where["count"])
        ordered = sorted(pairs, key=lambda p: _centre(geompy, p[1])[axis],
                         reverse=(take == "max"))
        selected = ordered[:n]
    elif "by_size" in where:
        kind = where["by_size"]
        n = int(where["count"])
        metric_idx = 1 if kind in ("largest", "smallest") else 0  # area vs length
        biggest = kind in ("longest", "largest")
        ordered = sorted(pairs,
                         key=lambda p: geompy.BasicProperties(p[1])[metric_idx],
                         reverse=biggest)
        selected = ordered[:n]
    elif "on_plane" in where:
        op = where["on_plane"]
        pnt = geompy.MakeVertex(*[float(v) for v in op["point"]])
        nrm = geompy.MakeVectorDXDYDZ(*[float(v) for v in op["normal"]])
        st = _state_enum(where.get("state", "on"))
        sel_ids = set(geompy.GetShapesOnPlaneWithLocationIDs(
            shape, type_code, nrm, pnt, st))
        selected = [p for p in pairs if p[0] in sel_ids]
    else:
        return {"ok": False, "error": f"get_subshapes: unknown where {where!r}"}

    return {
        "ok": True,
        "records": [],
        "sanity": {"ok": True},
        "ids": [i for i, _ in selected],
        "centres": [_centre(geompy, s) for _, s in selected],
    }
```

Register:
- `TOOLS`: `"get_subshapes": tool_get_subshapes,`
- `TOOL_CATEGORY`: `"get_subshapes": "query",`
- `ARG_TYPES`: `"get_subshapes": {},`
- `TOOL_SCHEMAS`:
  ```python
      "get_subshapes": {
          "type": "object",
          "properties": {
              "shape": {"type": "string"},
              "sub_type": {"type": "string",
                           "enum": ["vertex", "edge", "wire", "face", "shell", "solid"]},
              "where": {
                  "description": ('"all", or {"sorted_by":"x|y|z","take":"max|min",'
                                  '"count":N}, or {"by_size":"longest|shortest|'
                                  'largest|smallest","count":N}, or {"on_plane":'
                                  '{"point":[x,y,z],"normal":[x,y,z]},"state":"on"}'),
              },
          },
          "required": ["shape", "sub_type", "where"],
      },
  ```
- `TOOL_DESCRIPTIONS`: `"get_subshapes": "Find subshape ids of \`shape\` without changing geometry. Returns {ids, centres}. Pass the ids to fillet_edges, chamfer_edges or get_subshape. where = \\"all\\", or select by a centre axis (sorted_by/take/count), by size (by_size/count), or by a plane (on_plane/state).",`

In `salome_agent/specialists/tool_selection.py`, add `"get_subshapes"` to the `CORE_TOOLS` frozenset.

- [ ] **Step 4: Run the unit tests to verify they pass**

Run: `.venv/bin/pytest tests/unit/test_geom_tools.py tests/unit/test_tool_selection.py tests/unit/test_specialist_geom.py -v`
Expected: PASS. (`test_specialist_geom.py` is included because `initial_offered` / `CORE_TOOLS` assertions there could be sensitive — confirm `test_hinted_stage_offers_core_plus_hints_plus_meta_tools_only` still passes; if it hard-codes the core list, update it to include `get_subshapes`.)

- [ ] **Step 5: Write the live tests**

Add to `tests/salome/test_geom_tools_live.py`, class `TestSubshapeSelection`:

```python
    def test_get_subshapes_all_edges(self, agent_client):
        agent_client.call_tool("make_box", {"dx": 20, "dy": 20, "dz": 20})
        out = agent_client.call_tool(
            "get_subshapes", {"shape": "box_1", "sub_type": "edge", "where": "all"})
        assert out["ok"] is True, out.get("error")
        assert len(out["ids"]) == 12
        assert len(out["centres"]) == 12
        assert out["records"] == []

    def test_get_subshapes_top_four_edges_by_z(self, agent_client):
        agent_client.call_tool("make_box", {"dx": 20, "dy": 20, "dz": 20})
        out = agent_client.call_tool("get_subshapes", {
            "shape": "box_1", "sub_type": "edge",
            "where": {"sorted_by": "z", "take": "max", "count": 4}})
        assert out["ok"] is True, out.get("error")
        assert len(out["ids"]) == 4
        for c in out["centres"]:
            assert c[2] == 20.0  # all on the top face

    def test_get_subshapes_on_plane(self, agent_client):
        agent_client.call_tool("make_box", {"dx": 20, "dy": 20, "dz": 20})
        out = agent_client.call_tool("get_subshapes", {
            "shape": "box_1", "sub_type": "face",
            "where": {"on_plane": {"point": [0, 0, 20], "normal": [0, 0, 1]},
                      "state": "on"}})
        assert out["ok"] is True, out.get("error")
        assert len(out["ids"]) == 1  # exactly the top face
```

- [ ] **Step 6: Run the live tests**

Run: `.venv/bin/pytest tests/salome/test_geom_tools_live.py::TestSubshapeSelection -v`
Expected: PASS. If `on_plane` fails on the geompy signature, introspect the live build:
`agent_client.call_tool` a throwaway and check `geompy.GetShapesOnPlaneWithLocationIDs.__doc__` order — spec verified `(shape, type, normalVec, pnt, state)` on 2026-08-31 but confirm before adjusting.

- [ ] **Step 7: Commit**

```bash
git add executor/geom_tools.py salome_agent/specialists/tool_selection.py tests/unit/test_geom_tools.py tests/unit/test_tool_selection.py tests/unit/test_specialist_geom.py tests/salome/test_geom_tools_live.py
git commit -m "feat: get_subshapes query tool for edge/face id selection (#12)"
```

---

## Task 6: `min_distance` + `basic_properties` measurement queries

**Files:**
- Modify: `executor/geom_tools.py` (add `tool_min_distance`, `tool_basic_properties`; register in the five dicts)
- Test: `tests/unit/test_geom_tools.py`, `tests/salome/test_geom_tools_live.py`

**Interfaces:**
- `min_distance`: `args = {"shape1": <ref>, "shape2": <ref>}` -> `{"ok": True, "records": [], "sanity": {"ok": True}, "distance": <float>}`.
- `basic_properties`: `args = {"shape": <ref>}` -> `{"ok": True, "records": [], "sanity": {"ok": True}, "properties": {"length": <float>, "area": <float>, "volume": <float>}}`.
- Both are queries — no `CallRecord`, nothing reaches `result.py`.
- Rationale for only these two: the existing `measure` tool already returns bbox / size / centre by name, which covers the spec's `bounding_box` and `point_coordinates` helpers.

- [ ] **Step 1: Write the failing unit tests**

Add to `tests/unit/test_geom_tools.py`:

```python
class TestMeasurementQueries:
    def _executor(self):
        geompy = MagicMock()
        return SimpleNamespace(namespace={"geompy": geompy},
                               geom_bindings={"a": "A", "b": "B"}, name_prefix="")

    def test_min_distance_returns_a_float(self):
        ex = self._executor()
        ex.namespace["geompy"].MinDistance.return_value = 4.23456
        out = geom_tools.tool_min_distance(ex, {"shape1": "a", "shape2": "b"})
        assert out["records"] == []
        assert out["distance"] == 4.2346

    def test_min_distance_tolerates_a_tuple_return(self):
        ex = self._executor()
        ex.namespace["geompy"].MinDistance.return_value = (5.0, 0, 0, 0, 5, 0, 0)
        out = geom_tools.tool_min_distance(ex, {"shape1": "a", "shape2": "b"})
        assert out["distance"] == 5.0

    def test_basic_properties_returns_length_area_volume(self):
        ex = self._executor()
        ex.namespace["geompy"].BasicProperties.return_value = [40.0, 100.0, 0.0]
        out = geom_tools.tool_basic_properties(ex, {"shape": "a"})
        assert out["records"] == []
        assert out["properties"] == {"length": 40.0, "area": 100.0, "volume": 0.0}
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/bin/pytest tests/unit/test_geom_tools.py::TestMeasurementQueries -v`
Expected: FAIL — handlers not defined.

- [ ] **Step 3: Implement and register**

```python
def tool_min_distance(executor, args: dict):
    geompy = executor.namespace["geompy"]
    s1 = executor.geom_bindings[args["shape1"]]
    s2 = executor.geom_bindings[args["shape2"]]
    d = geompy.MinDistance(s1, s2)
    if isinstance(d, (tuple, list)):
        d = d[0]
    return {"ok": True, "records": [], "sanity": {"ok": True},
            "distance": round(d, 4)}


def tool_basic_properties(executor, args: dict):
    geompy = executor.namespace["geompy"]
    shape = executor.geom_bindings[args["shape"]]
    length, area, volume = geompy.BasicProperties(shape)
    return {"ok": True, "records": [], "sanity": {"ok": True},
            "properties": {"length": round(length, 4),
                           "area": round(area, 4),
                           "volume": round(volume, 4)}}
```

Register both:
- `TOOLS`: `"min_distance": tool_min_distance,` and `"basic_properties": tool_basic_properties,`
- `TOOL_CATEGORY`: both `"query"`
- `ARG_TYPES`: both `{}`
- `TOOL_SCHEMAS`:
  ```python
      "min_distance": {
          "type": "object",
          "properties": {"shape1": {"type": "string"}, "shape2": {"type": "string"}},
          "required": ["shape1", "shape2"],
      },
      "basic_properties": {
          "type": "object",
          "properties": {"shape": {"type": "string"}},
          "required": ["shape"],
      },
  ```
- `TOOL_DESCRIPTIONS`:
  - `"min_distance": "Report the shortest distance (mm) between two published shapes."`
  - `"basic_properties": "Report [length, area, volume] of \`shape\` (mm). Length for an edge/wire, area for a face/shell, volume for a solid; the others are 0."`

- [ ] **Step 4: Run the unit tests to verify they pass**

Run: `.venv/bin/pytest tests/unit/test_geom_tools.py -v`
Expected: PASS.

- [ ] **Step 5: Write the live tests**

Add to `tests/salome/test_geom_tools_live.py`, class `TestSubshapeSelection`:

```python
    def test_min_distance_between_two_boxes(self, agent_client):
        agent_client.call_tool("make_box", {"dx": 10, "dy": 10, "dz": 10})
        agent_client.call_tool("make_box", {"dx": 10, "dy": 10, "dz": 10})
        agent_client.call_tool("translate", {"shape": "box_2", "dx": 30, "dy": 0, "dz": 0})
        out = agent_client.call_tool("min_distance", {"shape1": "box_1", "shape2": "tr_1"})
        assert out["ok"] is True, out.get("error")
        assert out["distance"] == 20.0

    def test_basic_properties_of_a_box(self, agent_client):
        agent_client.call_tool("make_box", {"dx": 2, "dy": 3, "dz": 4})
        out = agent_client.call_tool("basic_properties", {"shape": "box_1"})
        assert out["ok"] is True, out.get("error")
        assert out["properties"]["volume"] == 24.0
```

- [ ] **Step 6: Run the live tests**

Run: `.venv/bin/pytest tests/salome/test_geom_tools_live.py::TestSubshapeSelection -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add executor/geom_tools.py tests/unit/test_geom_tools.py tests/salome/test_geom_tools_live.py
git commit -m "feat: min_distance and basic_properties measurement queries (#12)"
```

---

## Task 7: geom system prompt — the query-then-act pattern

**Files:**
- Modify: `salome_agent/specialists/geom.py` (`_GEOM_BASE`)
- Test: `tests/unit/test_specialist_geom.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `GEOM_SYSTEM_PROMPT` (and every `_geom_prompt(cadence)` output) contains a paragraph telling the model to call `get_subshapes` for ids, then pass them to `fillet_edges` / `chamfer_edges` / `get_subshape`, and that `get_subshapes` changes no geometry.

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/test_specialist_geom.py`:

```python
def test_prompt_documents_the_query_then_act_pattern():
    from salome_agent.specialists.geom import _geom_prompt
    p = _geom_prompt("final")
    assert "get_subshapes" in p
    assert "fillet_edges" in p
    for cadence in ("none", "final", "strict"):
        assert "get_subshapes" in _geom_prompt(cadence)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/pytest tests/unit/test_specialist_geom.py::test_prompt_documents_the_query_then_act_pattern -v`
Expected: FAIL — `get_subshapes` not in the prompt.

- [ ] **Step 3: Add the paragraph**

In `salome_agent/specialists/geom.py`, in `_GEOM_BASE`, insert this paragraph immediately before the closing `load_tools` paragraph (so it is part of the always-present base text, not the cadence-specific `{verify}` block):

```
To act on specific edges or faces (e.g. "fillet the top four edges"), first
call get_subshapes to get their integer ids -- by where = "all", a centre
axis (sorted_by / take / count), size (by_size / count), or a plane
(on_plane / state) -- then pass that id list straight to fillet_edges,
chamfer_edges or get_subshape. get_subshapes only reports ids and centres;
it changes no geometry. Use explode when you instead want every subshape
published as its own named object.
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/unit/test_specialist_geom.py -v`
Expected: PASS (including the existing `test_system_prompt_*` and `test_verify_hint_has_all_three_cadences` — the new paragraph is cadence-independent so those are unaffected).

- [ ] **Step 5: Commit**

```bash
git add salome_agent/specialists/geom.py tests/unit/test_specialist_geom.py
git commit -m "feat: document the get_subshapes query-then-act pattern in the geom prompt (#12)"
```

---

## Task 8: live LLM smoke + docs

**Files:**
- Modify: `README.md` (geom-tool count and/or known-limitations bullets that mention subshape selection being a raw-int guess or specific-edge fillet being unsupported — grep for "fillet" and "subshape")
- Modify: `docs/superpowers/specs/2026-08-31-geom-tool-catalogue.md` (tick the section 10 checklist)
- No new pytest file — this is a manual live run whose transcript is the evidence.

**Interfaces:** none.

- [ ] **Step 1: Full unit suite green**

Run: `.venv/bin/pytest tests/unit -q`
Expected: PASS, count >= the pre-plan baseline plus the new tests.

- [ ] **Step 2: Full live geom_tools suite green**

Ensure `$SALOME_ROOT` set and the SALOME agent server restarted (server-cache gotcha).
Run: `.venv/bin/pytest tests/salome/test_geom_tools_live.py -q`
Expected: PASS.

- [ ] **Step 3: Live LLM smoke run**

Restart the SALOME agent server first. Then:

Run: `.venv/bin/salome-agent "fillet the top 4 edges of a 20mm cube, radius 3mm" -vv`

Expected: the geom specialist calls `get_subshapes(shape=box_1, sub_type=edge, where={sorted_by:z, take:max, count:4})` (loading it via `load_tools` only if the planner did not hint it — it is in `CORE_TOOLS` so it should be offered from the start), reads the 4 ids from the result, calls `fillet_edges(shape=box_1, edges=[...], radius=3)`, then `finish_stage`. The generated `result.py` runs standalone under `salome shell` and produces a cube with 4 rounded top edges.

Record the run id, call count, and input/output token totals from the `-vv` trace / `report.md` in the session memory note (compare against a pre-plan run of the same prompt if one exists).

- [ ] **Step 4: If the smoke run reveals a gap**

Common gaps and where they belong:
- Model passes ids as a JSON string not an array to `fillet_edges` -> already handled by `_normalise_edge_ids` (Task is done, no change).
- Model never calls `get_subshapes` and guesses ids -> strengthen the Task 7 paragraph (make it imperative), re-run.
- `sorted_by` picks bottom edges -> check the `take` default / axis mapping in Task 5, add a unit test for the failing case first.
Do not paper over a real handler bug in the prompt — fix the handler with a failing test.

- [ ] **Step 5: Update docs**

- `README.md`: bump the geom tool count; remove or soften any bullet saying specific-edge fillet / subshape selection is unsupported or a blind integer guess.
- `docs/superpowers/specs/2026-08-31-geom-tool-catalogue.md` section 10: mark checklist items 1–7 done; leave item 8 (the `where`-object sugar on `fillet_edges`) as an explicit future option.

- [ ] **Step 6: Commit**

```bash
git add README.md docs/superpowers/specs/2026-08-31-geom-tool-catalogue.md
git commit -m "docs: subshape selection & query family shipped (#12)"
```

- [ ] **Step 7: Finish the branch**

Use the `superpowers:finishing-a-development-branch` skill. Do not `gh issue close` — file a summary comment on issue #12 if useful; the user closes issues. Do not cite a commit hash in the issue comment before the branch is pushed.

---

## Self-Review

**1. Spec coverage (section 10):**
- "query tools that return data, no `CallRecord`" -> Tasks 5, 6 (`get_subshapes`, `min_distance`, `basic_properties` all return `"records": []`).
- `explode` + list-return codegen -> Tasks 1, 3.
- `get_subshapes` with `all` / `on_plane` / `sorted_by` / `by_size` -> Task 5 (the four checklist predicates; `on_cylinder` / `on_sphere` / `on_shape` explicitly deferred per the brainstorm).
- `get_subshape` -> Task 4.
- measurement helpers -> Task 6, scoped to the two genuinely new ones (`bounding_box` / `point_coordinates` are already covered by the existing `measure` tool — brainstorm decision).
- `GEOM_SYSTEM_PROMPT` query-then-act paragraph -> Task 7.
- live LLM smoke `"fillet the top 4 edges..."` -> Task 8.
- checklist item 2 (query result with no `records` round-trips) -> Task 2 makes it explicit and tested.
- checklist item 8 (inline `where` sugar on `fillet_edges`) -> explicitly left as future in Task 8 step 5.

**2. Placeholder scan:** No "TBD"/"handle edge cases"/"similar to Task N". Every code step has a full code block. Task 8 is inherently a manual run; its steps still spell out the exact command, expected behavior, and where a discovered gap is fixed.

**3. Type consistency:**
- `emit == "list_call"` produced by Task 3, consumed by Task 1. Temp var name `_sub_<i>` consistent.
- Query payload keys `ids` / `centres` / `distance` / `properties` produced by Tasks 5–6, retained by Task 2's `_compact_result` change, listed identically in both.
- `_centre(geompy, shape)` helper defined once in Task 5, reused by Task 5's `by_size`/`sorted_by`; Task 6 does not use it.
- `_state_enum` defined in Task 5, used only there.
- Registration dicts (`TOOLS`, `TOOL_CATEGORY`, `TOOL_SCHEMAS`, `TOOL_DESCRIPTIONS`, `ARG_TYPES`) — every new tool adds an entry to all five; the unit guards `test_every_tool_has_a_valid_category` and `test_every_llm_facing_tool_has_a_description` enforce completeness.
- `CORE_TOOLS` change in Task 5 — Task 5 step 4 flags the one existing specialist test that may hard-code the core list.
