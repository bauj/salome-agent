# Generic specialist loop, ModuleSpec and tool registry — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn `specialists/geom.py:_run` into one generic `ToolSpecialist` loop driven by a `ModuleSpec` data record, absorb smesh/paravis into it, and replace geom's five parallel tool dicts with a `ToolRegistry` — so the next tool-calling module (194's amend specialist, SHAPER, smesh-as-tools) is configuration, not a fork.

**Architecture:** A frozen `ModuleSpec` dataclass declares everything that differs between modules: the terminator record (name, schema, description, and a `handle` callable that produces the stage result), how a tool executes, how a tool result renders into a transcript message, which tools are offered, and the module-specific prompt blocks. `ToolSpecialist.run` is the current geom loop body with three substitutions (terminator name, `spec.execute`, spec-supplied category sets); every geom-only lever (per-stage tool subsets, `load_tools`, verify-cadence nudges) goes inert for smesh/paravis through **empty configuration**, never through a branch. smesh and paravis ride along so they keep working; the loop is shaped by geom's needs, and smesh's own tool layer is a later task. The tool registry is a `ToolRegistry` instance plus a `@tool(...)` decorator; `executor/geom_tools.py` stays a re-export shim until the last commit so the ~1800-line test file is untouched while the mechanical split lands.

**Tech Stack:** Python >= 3.10, stdlib `dataclasses`, LangGraph (unchanged), pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-09-21-specialist-refactor-design.md`

## Global Constraints

- **Branch:** all work on `spec/specialist-refactor` (already checked out, based on `main` at `41ff9ca`). Never commit straight to `main`.
- **Never switch git branches while a live CLI run or a live SALOME server is using this worktree** — an import mismatch mid-run raises a bare `TypeError` (session 23 incident).
- **Python floor is 3.10** (`pyproject.toml`). No `match`, no PEP 604 in runtime-evaluated positions that 3.10 rejects, no `typing.Self`.
- **Run the suite with the project venv:** `.venv/bin/python -m pytest tests/unit -q`. Baseline on `main` at plan time: **385 passed**. Every task must end green and never below 385 minus tests deliberately deleted by that task (each such deletion is named in the task).
- **`git add` named files only.** Never `git add -A` / `git add .` (a past `-A` leaked a config file containing an API key).
- **Commit messages: bare subject line only.** No body, no `Co-Authored-By`, no session trailers. Task 226 has GitHub issue **#40**, so every commit subject on this branch ends with ` (#40)`.
- **Never `gh issue close`.** The user closes issues.
- **Never bake this machine's SALOME install path** (anything under `/storage/salome_platform/...`, anything with `-FD44`) into repo code, comments, docs or fixtures. Resolve from env/config.
- **Provider-agnostic:** no provider or gateway specifics in source; those live in `config.json`.
- **Minimal comments.** Comment the non-obvious *why*, in the style of the surrounding code. Do not narrate what the code plainly says.
- **Standing behaviour constraint:** geom is the module being made to work, and **geom's** prompts, transcript and generated code must come out byte-identical — any difference there is a bug, not an improvement.
  smesh and paravis were built at the start of the project and are absorbed **structurally, not improved**: the bar for them is that they keep working, not that their prompt bytes are preserved. smesh gets real attention in its own task next (its own tools, plugged in the way geom is), which is when its prompt gets rewritten anyway. **paravis is out of scope entirely** — do not spend effort, tests or verification runs on it beyond keeping it importable and routable.
- **The `sys.modules` caching gotcha:** `agent_server` caches the tools module. After Task 6 a running SALOME server keeps serving the OLD registry and **fails silently** — stale tool definitions, no error. A full SALOME restart is mandatory before any live verification from Task 6 onward.

---

## File Structure

**Created:**

| File | Responsibility |
|---|---|
| `salome_agent/modules.py` | `Terminator`, `ModuleSpec`, the `MODULES` registry, and a `spec_for(name)` lookup. Data only — no loop logic. |
| `salome_agent/specialists/tool_specialist.py` | `ToolSpecialist` — the single tool-calling loop, moved verbatim from `geom.py:_run`. |
| `salome_agent/specialists/script_modules.py` | smesh + paravis `ModuleSpec`s: the `emit_script` terminator, the closure `execute`, their two prompt blocks. |
| `executor/registry.py` | `ToolRegistry` and its `tool(...)` decorator. Shared later with shaper/smesh; nothing geom-specific. |
| `executor/geom/__init__.py` | Imports every tool submodule for registration; exposes `TOOLS`, `TOOL_CATEGORY`, `ARG_TYPES`, `TOOL_SCHEMAS`, `TOOL_DESCRIPTIONS`, `list_tools`, `call_tool`, `reset_counters`, `_counters`, and the private helpers the tests reach for. |
| `executor/geom/helpers.py` | The geom `ToolRegistry` instance and the `tool` decorator bound to it, plus `_counters`, `reset_counters`, `_next_name`, `_publish`, `_resolve`, `_gui_call`, `_parse_coords`, `_resolve_axis`, `_resolve_plane`, `_resolve_point`, the `ArgRef`/`CallRecord` TypedDicts. |
| `executor/geom/primitives.py`, `sketch.py`, `swept.py`, `booleans.py`, `transforms.py`, `features.py`, `display.py`, `queries.py` | The tool handlers, one file per category, each decorated with `@tool(...)`. |
| `executor/geom/results.py` | `_attach_shape_info`, `_delta_note`, `_select_subshapes` and the other cross-category result machinery that is not a tool. |
| `tests/replay.py` | `ReplayLLM` and `ReplayExecutorClient` — the deterministic replay harness. |
| `scripts/make-cassette.py` | Turns a recorded run directory into a compact committed cassette. |
| `tests/fixtures/cassettes/*.json` | Recorded LLM + executor traffic for the vase and enclosure requests. |
| `tests/fixtures/golden/*.py` | The `result.py` each cassette must reproduce. |
| `tests/fixtures/golden/list_tools.json`, `planner_prompt.txt` | Byte-level guards on the two wire surfaces a refactor can silently reorder. |
| `tests/unit/test_replay_harness.py`, `test_golden_runs.py`, `test_modules.py`, `test_tool_specialist.py`, `test_registry.py` | Tests for the above. |

**Modified:**

| File | Change |
|---|---|
| `salome_agent/executor_client.py` | Optional `record_path`: append one JSON line per `_call`. |
| `salome_agent/runner.py:93-96` | Pass `record_path` when `verbose >= 2`. |
| `salome_agent/graph.py:187-256` | The `if module == "geom": ... else:` fork becomes one `MODULES[module]` path. |
| `salome_agent/graph.py:79-97` | Delete the dead `select_upstream` scoping branch. |
| `salome_agent/specialists/geom.py` | Shrinks to geom's `ModuleSpec` and its helpers; the loop leaves. |
| `salome_agent/specialists/base.py` | **Deleted** in Task 3. |
| `salome_agent/codegen.py:81-108` | `geompy.` becomes a per-stage `call_prefix`. |
| `salome_agent/nodes/assembler.py:68-72` | Module-driven preamble parts come from `spec.preamble`. |
| `salome_agent/nodes/planner.py` | Catalogue assembled over `MODULES`. |
| `salome_agent/state.py:5-9` | `StagePlan` gains `tools: list[str]`. |
| `agent_server/core.py`, `backend.py`, `gui_backend.py` | Import `executor.geom` instead of `executor.geom_tools` (Task 7). |
| `executor/geom_tools.py` | Registry-backed (Task 5), shim (Task 6), **deleted** (Task 7). |

---

### Task 0: The deterministic replay harness

Nothing below this task is falsifiable without it. "Same `result.py`" is meaningless against a live LLM that answers differently every run, so the harness comes first: a fake LLM replays a recorded `llm_trace.jsonl`, a fake executor client replays the matching tool responses, and the graph runs end to end with no network and no SALOME.

Two of the guards are capturable **offline, right now**: `list_tools()`'s serialized bytes and the assembled planner system prompt. Those are the two wire surfaces this refactor can silently reorder, and they cost nothing.

**Files:**
- Create: `tests/replay.py`
- Create: `scripts/make-cassette.py`
- Create: `tests/fixtures/cassettes/synthetic_box.json`
- Create: `tests/fixtures/golden/list_tools.json`
- Create: `tests/fixtures/golden/planner_prompt.txt`
- Create: `tests/unit/test_replay_harness.py`
- Modify: `salome_agent/executor_client.py`
- Modify: `salome_agent/runner.py:93-96`

**Interfaces:**
- Consumes: `salome_agent.llm.LLMClient._round_trip(payload, tool_names, extra_body) -> dict` (the provider `message` dict); `salome_agent.executor_client.ExecutorClient._call(cmd, **payload) -> dict`.
- Produces:
  - `tests.replay.ReplayLLM(cassette: dict, **kw)` — an `LLMClient` subclass whose `_round_trip` pops the next recorded message. Attribute `exhausted: bool`.
  - `tests.replay.ReplayExecutorClient(cassette: dict)` — exposes `list_tools()`, `call_tool(name, args)`, `introspect(names)`, `list_study()`, `run(...)`, `sanity(...)`, `reset_and_replay(...)`, `close()`, plus `mode = "none"` and `name_prefix = ""`. Raises `AssertionError` on a cmd/name mismatch against the recording.
  - `tests.replay.load_cassette(name: str) -> dict` — reads `tests/fixtures/cassettes/<name>.json`.
  - `ExecutorClient(host, port, token, run_id, record_path=None)` — when `record_path` is set, appends `{"cmd", "payload", "response"}` per call.

- [ ] **Step 1: Capture the two offline byte guards (no code change yet)**

These are captured from the current tree, which is behaviourally `main`.

```bash
cd /storage/Projects/salome-agent
mkdir -p tests/fixtures/golden tests/fixtures/cassettes
.venv/bin/python - <<'PY'
import json
from executor.geom_tools import list_tools
from salome_agent.nodes.planner import PLANNER_SYSTEM_PROMPT
with open("tests/fixtures/golden/list_tools.json", "w") as f:
    f.write(json.dumps(list_tools(), ensure_ascii=False, indent=2) + "\n")
with open("tests/fixtures/golden/planner_prompt.txt", "w") as f:
    f.write(PLANNER_SYSTEM_PROMPT)
PY
wc -c tests/fixtures/golden/list_tools.json tests/fixtures/golden/planner_prompt.txt
```

Expected: both files non-empty; `list_tools.json` contains 44 entries.

- [ ] **Step 2: Write the failing byte-guard tests**

Create `tests/unit/test_golden_surfaces.py`:

```python
import json
import os

_FIX = os.path.join(os.path.dirname(__file__), "..", "fixtures", "golden")


def _read(name):
    with open(os.path.join(_FIX, name)) as f:
        return f.read()


def test_list_tools_wire_output_is_byte_identical():
    from executor.geom_tools import list_tools

    got = json.dumps(list_tools(), ensure_ascii=False, indent=2) + "\n"
    assert got == _read("list_tools.json")


def test_list_tools_structure_is_unchanged():
    from executor.geom_tools import list_tools

    assert list_tools() == json.loads(_read("list_tools.json"))


def test_planner_system_prompt_is_byte_identical():
    from salome_agent.nodes.planner import PLANNER_SYSTEM_PROMPT

    assert PLANNER_SYSTEM_PROMPT == _read("planner_prompt.txt")
```

Both a parsed comparison and a serialized one: a field-order change is invisible to a dict comparison and very visible to a gateway's prefix cache.

- [ ] **Step 3: Run the byte-guard tests**

Run: `.venv/bin/python -m pytest tests/unit/test_golden_surfaces.py -q`
Expected: **3 passed** (the fixtures were just captured from this tree).

- [ ] **Step 4: Write the failing test for executor-call recording**

`tests/unit/test_executor_client.py` drives a **real socket against a fake server thread**. Its helpers are `_fake_server(port_box, responses, *, hello=None)` and `_connect(box)` at line 30, which currently hardcodes `ExecutorClient("127.0.0.1", box[0], "tok", "run1")`. Widen `_connect` to forward extra kwargs:

```python
def _connect(box, **kw):
    return ExecutorClient("127.0.0.1", box[0], "tok", "run1", **kw)
```

then append, following that file's existing fake-server idiom for starting the thread and collecting `responses`:

```python
def test_record_path_appends_one_json_line_per_call(tmp_path):
    import json

    path = tmp_path / "exec_trace.jsonl"
    box = _fake_server([0], [{"id": 1, "ok": True, "records": []}])
    client = _connect(box, record_path=str(path))
    client.call_tool("make_box", {"dx": 1, "dy": 2, "dz": 3})

    lines = [json.loads(l) for l in path.read_text().splitlines()]
    assert lines[-1]["cmd"] == "call_tool"
    assert lines[-1]["payload"] == {"name": "make_box", "args": {"dx": 1, "dy": 2, "dz": 3}}
    assert lines[-1]["response"]["ok"] is True
```

Read `_fake_server`'s actual signature and wire it the way the neighbouring tests do — the call above is the shape, not the specification. Do not invent a second fake-socket mechanism alongside the existing one.

- [ ] **Step 5: Run it to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/test_executor_client.py -q`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'record_path'`.

- [ ] **Step 6: Implement recording in `ExecutorClient`**

In `salome_agent/executor_client.py`, widen `__init__` and `_call`:

```python
    def __init__(self, host: str, port: int, token: str, run_id: str,
                 record_path: str | None = None):
        self.record_path = record_path
        self._sock = socket.create_connection((host, port), timeout=10)
```

(the rest of `__init__` is unchanged), and at the end of `_call`, just before `return resp`:

```python
        if self.record_path:
            self._record(cmd, payload, resp)
        return resp

    def _record(self, cmd: str, payload: dict, resp: dict) -> None:
        # Recorded so a run can be replayed offline against a golden result.py;
        # a failure here must never abort the run it is only observing.
        try:
            import os

            os.makedirs(os.path.dirname(self.record_path) or ".", exist_ok=True)
            with open(self.record_path, "a") as f:
                f.write(json.dumps({"cmd": cmd, "payload": payload, "response": resp},
                                   default=str, ensure_ascii=False) + "\n")
        except OSError:
            pass
```

- [ ] **Step 7: Run it to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/test_executor_client.py -q`
Expected: PASS.

- [ ] **Step 8: Wire `record_path` into the runner**

In `salome_agent/runner.py`, the client is built by the caller, not by `run_request` — so the flag belongs where the client is constructed. Find every `ExecutorClient(` construction:

```bash
grep -rn "ExecutorClient(" --include=*.py . | grep -v '^\./\.venv'
```

For each production construction site, pass `record_path=os.path.join(run_dir, "exec_trace.jsonl")` when the run is already writing an LLM trace (`verbose >= 2`), and `None` otherwise. Keep the two traces on the same condition so a recorded run always has both halves or neither — a cassette needs both.

- [ ] **Step 9: Run the full suite**

Run: `.venv/bin/python -m pytest tests/unit -q`
Expected: **389 passed** (385 + 3 byte guards + 1 recording test).

- [ ] **Step 10: Write the replay harness**

Create `tests/replay.py`:

```python
"""Deterministic replay of a recorded run: no network, no SALOME.

A cassette is {"llm": [<provider message dict>, ...],
               "exec": [{"cmd", "payload", "response"}, ...]}.
Both streams replay strictly in order; any divergence from the recording is
an AssertionError, which is the whole point — a refactor that changes the
order of calls is exactly what this is meant to catch.
"""
import json
import os

from salome_agent.executor_client import SessionDied
from salome_agent.llm import LLMClient

_FIX = os.path.join(os.path.dirname(__file__), "fixtures", "cassettes")


def load_cassette(name: str) -> dict:
    with open(os.path.join(_FIX, f"{name}.json")) as f:
        return json.load(f)


class ReplayLLM(LLMClient):
    """An LLMClient whose provider round-trip is a recording.

    Subclassing (rather than faking `chat`) keeps every parsing rule in
    llm.py — tool-call argument decoding, the JSON-retry path, continuity
    fields — on the tested path.
    """

    def __init__(self, cassette: dict):
        super().__init__(base_url="http://replay.invalid", api_key="none",
                         model="replay", timeout=1)
        self._messages = list(cassette["llm"])
        self._i = 0

    @property
    def exhausted(self) -> bool:
        return self._i >= len(self._messages)

    def _round_trip(self, payload, tool_names, extra_body):
        assert not self.exhausted, "cassette ran out of LLM responses"
        message = self._messages[self._i]
        self._i += 1
        self.usage["calls"] += 1
        return message


class ReplayExecutorClient:
    """Replays recorded executor responses, asserting the call sequence."""

    mode = "none"
    name_prefix = ""

    def __init__(self, cassette: dict):
        self._calls = list(cassette["exec"])
        self._i = 0

    def _next(self, cmd: str, payload: dict) -> dict:
        assert self._i < len(self._calls), f"cassette ran out at {cmd}({payload})"
        rec = self._calls[self._i]
        self._i += 1
        assert rec["cmd"] == cmd, f"call {self._i}: recorded {rec['cmd']}, got {cmd}"
        assert rec["payload"] == payload, (
            f"call {self._i} ({cmd}): recorded {rec['payload']}, got {payload}")
        return rec["response"]

    def list_tools(self):
        return self._next("list_tools", {})

    def call_tool(self, name, args):
        return self._next("call_tool", {"name": name, "args": args})

    def introspect(self, names):
        return self._next("introspect", {"names": names})

    def list_study(self):
        return self._next("list_study", {})

    def run(self, code, timeout, module=None):
        return self._next("run", {"code": code, "timeout": timeout, "module": module})

    def sanity(self, module, expectations):
        return self._next("sanity", {"module": module, "expectations": expectations})

    def reset_and_replay(self, scripts):
        return self._next("reset_and_replay", {"scripts": scripts})

    def close(self):
        pass
```

If `LLMClient.__init__` rejects those keyword names, read its signature at `salome_agent/llm.py:174` and pass whatever it actually requires — the point is a constructed-but-never-connected client.

- [ ] **Step 11: Write the synthetic cassette and its test**

Create `tests/fixtures/cassettes/synthetic_box.json` — a hand-written two-turn geom run, so the harness is green before any live recording exists:

```json
{
  "llm": [
    {"role": "assistant", "content": "One box.",
     "tool_calls": [{"id": "c1", "type": "function",
       "function": {"name": "make_box", "arguments": "{\"dx\": 10, \"dy\": 20, \"dz\": 30}"}}]},
    {"role": "assistant", "content": "Done.",
     "tool_calls": [{"id": "c2", "type": "function",
       "function": {"name": "finish_stage", "arguments": "{\"result_names\": [\"box_1\"]}"}}]}
  ],
  "exec": [
    {"cmd": "list_tools", "payload": {},
     "response": {"tools": [{"name": "make_box", "category": "primitive",
       "parameters": {"type": "object", "properties": {}}, "description": "box"}]}},
    {"cmd": "call_tool", "payload": {"name": "make_box", "args": {"dx": 10, "dy": 20, "dz": 30}},
     "response": {"ok": true, "records": [{"fn": "MakeBoxDXDYDZ",
       "args": [{"kind": "literal", "value": 10}, {"kind": "literal", "value": 20},
                {"kind": "literal", "value": 30}],
       "out_vars": ["box_1"], "publish": ["box_1"]}]}}
  ]
}
```

Create `tests/unit/test_replay_harness.py`:

```python
from tests.replay import ReplayExecutorClient, ReplayLLM, load_cassette


def test_replay_drives_the_geom_specialist_to_a_call_log():
    from salome_agent.specialists.geom import run_geom_specialist

    cassette = load_cassette("synthetic_box")
    llm = ReplayLLM(cassette)
    client = ReplayExecutorClient(cassette)
    result = run_geom_specialist(
        {"module": "geom", "intent": "a box", "params": {}, "inputs": []},
        llm, client, iteration_cap=12, verify_cadence="none")
    assert result["result_names"] == ["box_1"]
    assert [c["fn"] for c in result["call_log"]] == ["MakeBoxDXDYDZ"]
    assert llm.exhausted


def test_replay_executor_rejects_a_diverging_call():
    import pytest

    cassette = load_cassette("synthetic_box")
    client = ReplayExecutorClient(cassette)
    client.list_tools()
    with pytest.raises(AssertionError):
        client.call_tool("make_sphere", {"radius": 5})


def test_generated_script_from_a_replayed_call_log():
    from salome_agent.codegen import generate
    from salome_agent.specialists.geom import run_geom_specialist

    cassette = load_cassette("synthetic_box")
    result = run_geom_specialist(
        {"module": "geom", "intent": "a box", "params": {}, "inputs": []},
        ReplayLLM(cassette), ReplayExecutorClient(cassette),
        iteration_cap=12, verify_cadence="none")
    src = generate(result["call_log"], result["result_names"])
    assert src == ("box_1 = geompy.MakeBoxDXDYDZ(10, 20, 30)\n"
                   "geompy.addToStudy(box_1, 'box_1')")
```

`tests/` needs to be importable as a package for `from tests.replay import ...`. `tests/unit/__init__.py` exists; **`tests/__init__.py` does not** (verified) — create it empty.

- [ ] **Step 12: Run the harness tests**

Run: `.venv/bin/python -m pytest tests/unit/test_replay_harness.py -q`
Expected: **3 passed**. If `generate` emits a different string, fix the *assertion* to the real output (the golden is whatever `main` does), not the code.

- [ ] **Step 13: Write the cassette maker**

Create `scripts/make-cassette.py`:

```python
#!/usr/bin/env python3
"""Turn a recorded run directory into a committed cassette.

Usage: scripts/make-cassette.py <run_dir> <cassette_name>

Reads <run_dir>/llm_trace.jsonl and <run_dir>/exec_trace.jsonl, keeps only
what replay needs (the provider message per LLM turn, and each executor
call's cmd/payload/response), and writes tests/fixtures/cassettes/<name>.json
plus tests/fixtures/golden/<name>.py from the run's result.py.
"""
import json
import os
import shutil
import sys


def main(run_dir: str, name: str) -> int:
    llm = []
    with open(os.path.join(run_dir, "llm_trace.jsonl")) as f:
        for line in f:
            entry = json.loads(line)
            message = entry["response"].get("message")
            if message is None:
                print(f"skipping a trace line with no message: {entry.get('t')}")
                continue
            # reasoning_content is dropped from history by design, so it is
            # dead weight in a cassette.
            llm.append({k: v for k, v in message.items() if k != "reasoning_content"})
    exec_calls = []
    with open(os.path.join(run_dir, "exec_trace.jsonl")) as f:
        for line in f:
            exec_calls.append(json.loads(line))
    root = os.path.join(os.path.dirname(__file__), "..", "tests", "fixtures")
    os.makedirs(os.path.join(root, "cassettes"), exist_ok=True)
    os.makedirs(os.path.join(root, "golden"), exist_ok=True)
    with open(os.path.join(root, "cassettes", f"{name}.json"), "w") as f:
        json.dump({"llm": llm, "exec": exec_calls}, f, ensure_ascii=False, indent=1)
        f.write("\n")
    shutil.copyfile(os.path.join(run_dir, "result.py"),
                    os.path.join(root, "golden", f"{name}.py"))
    print(f"{name}: {len(llm)} llm turn(s), {len(exec_calls)} executor call(s)")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
```

- [ ] **Step 14: Run the full suite and commit**

Run: `.venv/bin/python -m pytest tests/unit -q`
Expected: **391 passed**.

```bash
git add tests/replay.py tests/__init__.py scripts/make-cassette.py \
        tests/fixtures/cassettes/synthetic_box.json \
        tests/fixtures/golden/list_tools.json tests/fixtures/golden/planner_prompt.txt \
        tests/unit/test_replay_harness.py tests/unit/test_golden_surfaces.py \
        tests/unit/test_executor_client.py \
        salome_agent/executor_client.py salome_agent/runner.py
git commit -m "test: deterministic replay harness and byte guards for the specialist refactor (#40)"
```

---

### Task 0b: Capture the live baselines (requires SALOME — the user runs this)

**This is the one task an agent cannot complete alone.** It needs a live SALOME server and a live LLM endpoint. Hand the commands to the user and wait for the fixtures to exist.

The capture runs from the branch at Task 0's commit, which is behaviourally identical to `main` — Task 0 added recording and tests, and touched no production decision. That is why capture is safe here and does not need a detour through `main`.

**Files:**
- Create: `tests/fixtures/cassettes/vase.json`, `tests/fixtures/golden/vase.py`
- Create: `tests/fixtures/cassettes/enclosure.json`, `tests/fixtures/golden/enclosure.py`
- Create: `tests/unit/test_golden_runs.py`

**Interfaces:**
- Consumes: `tests.replay.load_cassette`, `ReplayLLM`, `ReplayExecutorClient` from Task 0.
- Produces: two named cassettes every later task re-checks against.

- [ ] **Step 1: Start a clean SALOME agent server**

Ask the user to run the project's headless server script and confirm it is up. Do not hardcode any install path in anything committed.

- [ ] **Step 2: Record the two baseline runs**

```bash
cd /storage/Projects/salome-agent
.venv/bin/salome-agent "a vase 200mm tall, 80mm wide at the base, hollow with 4mm walls" --batch -vv
.venv/bin/salome-agent "a 300x200x150mm enclosure with 5mm walls, a 60mm circular vent on the front face, and all vertical edges filleted at 8mm" --batch -vv
```

Note each run id from the log line `LLM trace: .../<run_id>/llm_trace.jsonl`. Confirm each run produced a `result.py` that is not empty and that the run reported no `fatal`. **If a run fails, re-run it — a failed baseline is worse than none**, because every later task would be pinned to broken behaviour.

- [ ] **Step 3: Turn each run into a cassette**

```bash
.venv/bin/python scripts/make-cassette.py ~/.local/share/salome-agent/runs/<vase_run_id> vase
.venv/bin/python scripts/make-cassette.py ~/.local/share/salome-agent/runs/<enclosure_run_id> enclosure
du -h tests/fixtures/cassettes/*.json
```

If a cassette exceeds ~1 MB, trim it: the `exec` stream's `response.records` are needed, but a `response` that carries a large `measurements` or `ids` blob can stay — size is only a repo-hygiene concern, not a correctness one. Do not hand-edit the LLM stream.

- [ ] **Step 4: Write the golden-run test**

Create `tests/unit/test_golden_runs.py`:

```python
import os

import pytest

from tests.replay import ReplayExecutorClient, ReplayLLM, load_cassette

_GOLDEN = os.path.join(os.path.dirname(__file__), "..", "fixtures", "golden")
_CASSETTES = os.path.join(os.path.dirname(__file__), "..", "fixtures", "cassettes")

_REQUESTS = {
    "vase": "a vase 200mm tall, 80mm wide at the base, hollow with 4mm walls",
    "enclosure": ("a 300x200x150mm enclosure with 5mm walls, a 60mm circular vent "
                  "on the front face, and all vertical edges filleted at 8mm"),
}


def _has(name):
    return os.path.exists(os.path.join(_CASSETTES, f"{name}.json"))


@pytest.mark.parametrize("name", sorted(_REQUESTS))
def test_replayed_run_reproduces_its_golden_result_py(name):
    if not _has(name):
        pytest.skip(f"cassette {name} not captured yet (see plan Task 0b)")
    cassette = load_cassette(name)
    script = _replay_run(_REQUESTS[name], cassette)
    with open(os.path.join(_GOLDEN, f"{name}.py")) as f:
        assert script == f.read()


def _replay_run(request, cassette):
    """Drive the real graph end to end with both streams replayed."""
    import uuid

    from salome_agent.graph import build_graph

    llm = ReplayLLM(cassette)
    client = ReplayExecutorClient(cassette)
    graph = build_graph(batch=True)
    config = {"configurable": {"thread_id": uuid.uuid4().hex,
                               "llm_client": llm, "executor_client": client}}
    state = {
        "run_id": "replay", "user_request": request,
        "config": {"batch": True, "work_dir": None,
                   "policy": {"tool_iteration_cap": 40, "static_retries": 1,
                              "exec_retries": 1, "exec_timeout_s": 120,
                              "tool_selection": "hints", "verify_cadence": "final",
                              "plan_detail": "balanced"}},
        "plan": [], "plan_approved": True, "stage_idx": 0, "plan_timing": None,
        "session": {"mode": "none", "pid": None, "salome_port": None, "name_prefix": ""},
        "study_ns": "salome_agent_run", "stages": {}, "last_error": None, "history": [],
        "final_script": None, "report": None, "fatal": None, "aborted": None,
    }
    out = graph.invoke(state, config=config)
    assert not out.get("fatal"), out.get("fatal")
    return out["final_script"]
```

The `work_dir` handling in `_report_node` writes files; point `work_dir` at a `tmp_path` if the test pollutes the tree — read `graph.py:367-383` and adapt. The `_replay_run` helper is the reusable core; keep it in this file, every later task calls it through this test.

- [ ] **Step 5: Run the golden tests**

Run: `.venv/bin/python -m pytest tests/unit/test_golden_runs.py -q`
Expected: **2 passed**. If a replay diverges *before any refactor*, the harness is wrong, not the code — most likely the cassette's exec stream is missing calls that the graph makes outside the specialist (`measure` for upstream, `run`/`sanity` for verify). Fix the harness or widen the recording until both replay clean. **Do not proceed past this step with a skipping or failing golden test** — every task below spends this guard.

- [ ] **Step 6: Commit**

```bash
git add tests/fixtures/cassettes/vase.json tests/fixtures/cassettes/enclosure.json \
        tests/fixtures/golden/vase.py tests/fixtures/golden/enclosure.py \
        tests/unit/test_golden_runs.py
git commit -m "test: vase and enclosure golden baselines captured from main (#40)"
```

---

### Task 1: `ModuleSpec`, `Terminator` and geom's spec — data only, not yet wired

**Files:**
- Create: `salome_agent/modules.py`
- Create: `tests/unit/test_modules.py`
- Modify: `salome_agent/specialists/geom.py` (add the spec at the bottom; change nothing else)

**Interfaces:**
- Consumes: `salome_agent.specialists.geom.{_geom_prompt, FINISH_STAGE_SCHEMA, FINISH_STAGE_DESCRIPTION, _leaf_result_names, _compact_result, _retry_block}`; `salome_agent.specialists.tool_selection.CORE_TOOLS`.
- Produces:
  - `salome_agent.modules.Terminator(name, schema, description, handle)` — `handle(args: dict, call_log: list) -> dict`.
  - `salome_agent.modules.ModuleSpec(...)` — frozen dataclass, fields exactly as written in Step 3 below.
  - `salome_agent.modules.MODULES: dict[str, ModuleSpec]`, populated with `"geom"` in this task.
  - `salome_agent.modules.spec_for(name: str) -> ModuleSpec`.
  - `salome_agent.specialists.geom.GEOM_SPEC: ModuleSpec`.

**Note on two deviations from the spec document, both forced by the code:**
1. The spec writes `measure_tool: str | None`. The code has `_MEASURE_TOOLS = {"measure", "check_shape"}` — two names. The field is `measure_tools: frozenset[str]`.
2. `execute` in the spec returns `str`. geom's loop needs the whole result dict (`ok`, `records`, `error`), while the script modules need a plain observation string. So `execute` returns a **dict** and a second field, `render_result`, turns it into the transcript message. That keeps both modules' transcript wording exactly as it is today.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_modules.py`:

```python
from salome_agent.modules import MODULES, ModuleSpec, Terminator, spec_for


def test_geom_spec_is_registered_and_frozen():
    spec = spec_for("geom")
    assert isinstance(spec, ModuleSpec)
    assert spec.name == "geom"
    assert MODULES["geom"] is spec


def test_geom_terminator_is_finish_stage():
    t = spec_for("geom").terminator
    assert isinstance(t, Terminator)
    assert t.name == "finish_stage"
    assert t.schema["required"] == ["result_names"]
    assert t.description


def test_geom_terminator_handle_returns_call_log_and_leaf_names():
    t = spec_for("geom").terminator
    call_log = [{"fn": "MakeBoxDXDYDZ", "args": [], "out_vars": ["box_1"], "publish": ["box_1"]}]
    assert t.handle({"result_names": ["box_1"]}, call_log) == {
        "call_log": call_log, "result_names": ["box_1"]}


def test_geom_terminator_handle_decodes_a_json_encoded_result_names():
    # seen live: a model returns the array as a JSON string
    t = spec_for("geom").terminator
    out = t.handle({"result_names": '["box_1"]'}, [])
    assert out["result_names"] == ["box_1"]


def test_geom_levers_are_configured_on():
    spec = spec_for("geom")
    assert "make_box" in spec.core_tools
    assert spec.risky_categories == frozenset({"boolean", "transform"})
    assert spec.measure_tools == frozenset({"measure", "check_shape"})
    assert spec.categories
    assert spec.call_prefix == "geompy."
    assert spec.catalogue  # geom is routable


def test_module_spec_is_frozen():
    import dataclasses

    import pytest

    with pytest.raises(dataclasses.FrozenInstanceError):
        spec_for("geom").name = "nope"


def test_spec_for_rejects_an_unknown_module():
    import pytest

    with pytest.raises(KeyError):
        spec_for("shaper")
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/test_modules.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'salome_agent.modules'`.

- [ ] **Step 3: Write `salome_agent/modules.py`**

```python
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(frozen=True)
class Terminator:
    """The one tool that ends a stage.

    `handle` absorbs the entire difference between a tool-calling module and a
    script-emitting one: geom's returns {result_names, call_log}, a script
    module's returns {script, warning}. The loop never asks which mode it is in.
    """
    name: str
    schema: dict
    description: str
    handle: Callable[[dict, list], dict]


@dataclass(frozen=True)
class ModuleSpec:
    name: str
    # verify_cadence -> system prompt. Modules with no cadence ignore the arg.
    prompt: Callable[[str], str]
    terminator: Terminator
    # client, tool name, args -> the raw result dict the loop reasons about
    # (needs "ok"; may carry "records", "error").
    execute: Callable[[Any, str, dict], dict]
    # result dict -> the string that goes back as the tool message.
    render_result: Callable[[dict], str]
    # client -> {tool name: tool def}. The terminator is never in here.
    tools_for: Callable[[Any], dict]
    # What to say when the model replies with prose and no tool call. geom's
    # exact wording is pinned by its golden baselines.
    no_tool_nudge: str
    # call_log, last_error -> the result when the iteration cap runs out.
    on_exhausted: Callable[[list, dict | None], dict]

    # Everything below is a lever that goes INERT when left empty — that is
    # how smesh and paravis run the geom loop without being changed by it.
    core_tools: frozenset = frozenset()
    categories: frozenset = frozenset()
    risky_categories: frozenset = frozenset()
    measure_tools: frozenset = frozenset()

    render_upstream: Callable[..., "str | None"] | None = None
    render_retry: Callable[[dict], "str | None"] | None = None

    call_prefix: str | None = None
    preamble: str = ""
    # Empty means the planner never offers this module as a stage.
    catalogue: str = ""


MODULES: dict[str, ModuleSpec] = {}


def register(spec: ModuleSpec) -> ModuleSpec:
    MODULES[spec.name] = spec
    return spec


def spec_for(name: str) -> ModuleSpec:
    return MODULES[name]


def routable() -> list[ModuleSpec]:
    """Modules the planner may put in a plan — those with catalogue text."""
    return [s for s in MODULES.values() if s.catalogue]
```

The `from dataclasses import field` import is unused as written; drop it if no default needs a factory.

- [ ] **Step 4: Add geom's spec at the bottom of `specialists/geom.py`**

Nothing above it moves in this task. Append:

```python
def _finish_stage(args: dict, call_log: list) -> dict:
    result_names = args["result_names"]
    # A schema hint doesn't guarantee compliance — seen live, a model returned
    # this as a JSON-encoded string, which codegen's set() would then iterate
    # as individual characters.
    if isinstance(result_names, str):
        result_names = json.loads(result_names)
    return {"call_log": call_log, "result_names": _leaf_result_names(call_log, result_names)}


def _geom_tools_for(client) -> dict:
    return {t["name"]: t for t in client.list_tools()["tools"]}


def _geom_upstream(client, names: list[str]) -> str | None:
    """The measured upstream block. Measuring belongs here, not in graph.py:
    knowing what a bbox is is geom's business."""
    if not names or client is None:
        return None
    info = None
    try:
        m = client.call_tool("measure", {"names": names})
        if m.get("ok"):
            info = m.get("measurements")
        else:
            log.warning("  measure(upstream) failed: %s", m.get("error"))
    except Exception as e:  # noqa: BLE001 — best-effort, must not abort the stage
        log.warning("  measure(upstream) errored: %s", e)
    if not info:
        return None
    lines = []
    for name in names:
        detail = info.get(name)
        if detail:
            lines.append(f"  {name} — bbox {detail['bbox']}, size {detail['size']}, "
                         f"centre {detail['centre']}")
        else:
            lines.append(f"  {name} — (geometry unavailable)")
    return ("Objects built by an EARLIER stage, given only for reference "
            "(e.g. to measure or align against) — reference them by these exact "
            "names, but do not modify, colour or finish_stage them unless this "
            "stage's own task says to:\n" + "\n".join(lines))


GEOM_SPEC = register(ModuleSpec(
    name="geom",
    prompt=_geom_prompt,
    terminator=Terminator(name="finish_stage", schema=FINISH_STAGE_SCHEMA,
                          description=FINISH_STAGE_DESCRIPTION, handle=_finish_stage),
    execute=lambda client, name, args: client.call_tool(name, args),
    render_result=lambda result: str(_compact_result(result)),
    tools_for=_geom_tools_for,
    no_tool_nudge=("You did not call a tool. Call at least one tool now — "
                   "finish_stage once the geometry is done."),
    on_exhausted=lambda call_log, last_error: {"call_log": call_log, "last_error": last_error},
    core_tools=frozenset(CORE_TOOLS),
    categories=frozenset({"primitive", "sketch", "swept", "boolean",
                          "transform", "feature", "display", "query"}),
    risky_categories=frozenset(_PLACEMENT_RISKY),
    measure_tools=frozenset(_MEASURE_TOOLS),
    render_upstream=_geom_upstream,
    render_retry=lambda retry: _retry_block(retry) or None,
    call_prefix="geompy.",
    preamble=_GEOM_PREAMBLE,
    catalogue="",  # filled in Task 4, from the tool registry
))
```

with these imports added at the top of the file:

```python
from salome_agent.modules import ModuleSpec, Terminator, register
from salome_agent.specialists.tool_selection import CORE_TOOLS, initial_offered, resolve_load
```

and `_GEOM_PREAMBLE` defined next to the other constants as the exact text of `_GEOM_INIT` in `salome_agent/nodes/assembler.py:14-16`:

```python
# The result.py preamble a geom stage needs. Assembler owns when to emit it;
# the module owns what it is.
_GEOM_PREAMBLE = """from salome.geom import geomBuilder
geompy = geomBuilder.New()
"""
```

**Important:** `_geom_upstream`'s wording is verified against `geom.py:225-228` character for character, minus the two leading newlines (the loop joins blocks with `\n\n` in Task 2). The `_retry_block` wording is reused, not rewritten.

Watch for an import cycle: `modules.py` must not import anything from `specialists/`. It does not — geom imports modules, never the reverse.

- [ ] **Step 5: Run the test**

Run: `.venv/bin/python -m pytest tests/unit/test_modules.py -q`
Expected: **7 passed**. `test_geom_levers_are_configured_on` asserts `spec.catalogue` truthy — it is empty until Task 4, so **delete that one assertion line from the test now and restore it in Task 4**, where the catalogue is wired.

- [ ] **Step 6: Run the full suite**

Run: `.venv/bin/python -m pytest tests/unit -q`
Expected: **398 passed** (391 + 7). The golden tests still pass because nothing is wired yet.

- [ ] **Step 7: Commit**

```bash
git add salome_agent/modules.py salome_agent/specialists/geom.py tests/unit/test_modules.py
git commit -m "feat: ModuleSpec, Terminator and geom's module spec (#40)"
```

---

### Task 2: `ToolSpecialist` — the loop moves out of `geom.py`

This is the riskiest task in the plan and the reason Task 0 exists. **The loop body is a move, not a rewrite.** Work with `specialists/geom.py:212-385` open beside the new file and port it statement by statement. Three things generalise as they move, and nothing else changes:

1. `"finish_stage"` becomes `spec.terminator.name`, and its result goes through `spec.terminator.handle`.
2. `client.call_tool(name, args)` becomes `spec.execute(client, name, args)`.
3. `_PLACEMENT_RISKY` and `_MEASURE_TOOLS` become `spec.risky_categories` and `spec.measure_tools`.

The user message is now built from `blocks` — pre-rendered strings the caller passes — joined after `str(stage_plan)`. The retry block stays *inside* the loop because `retry` and `seed_call_log` are coupled (the seed keeps a reused-by-name shape traceable for codegen, GH #38) and splitting them across the boundary would invite passing one without the other.

**Files:**
- Create: `salome_agent/specialists/tool_specialist.py`
- Create: `tests/unit/test_tool_specialist.py`
- Modify: `salome_agent/specialists/geom.py` (delete `_run`; `run_geom_specialist` delegates)

**`graph.py` is NOT touched by this task.** `run_geom_specialist` keeps its `upstream` kwarg and builds `blocks` internally, so every caller is unaffected. Task 3 is what collapses the fork in `graph.py`.

**Interfaces:**
- Consumes: `ModuleSpec` and `GEOM_SPEC` from Task 1; `tool_selection.initial_offered`, `resolve_load`; `llm.assistant_tool_turn`, `llm.continuity_fields`.
- Produces:
  ```python
  class ToolSpecialist:
      def __init__(self, spec: ModuleSpec): ...
      def run(self, stage_plan: dict, llm, client, iteration_cap: int, *,
              blocks: tuple = (), tool_selection: str = "hints",
              verify_cadence: str = "final", retry: dict | None = None,
              seed_call_log: list | None = None) -> dict
  ```
  Returns exactly one of: `spec.terminator.handle(...)`'s dict, `{"fatal": str}` on `SessionDied`/`SessionPoisoned`, or `spec.on_exhausted(call_log, last_error)`. Those are the same three shapes `run_geom_specialist` returns today, so `graph.py`'s result handling is untouched.
- `run_geom_specialist(...)` keeps its exact current signature and return contract — it becomes a thin wrapper. Every existing test in `tests/unit/test_specialist_geom.py` must keep passing **unedited**; that unchanged file is the main evidence this move changed nothing.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_tool_specialist.py`. These test the generic contract, not geom:

```python
from unittest.mock import MagicMock

from salome_agent.modules import ModuleSpec, Terminator
from salome_agent.specialists.tool_specialist import ToolSpecialist


def _resp(*calls, content=None, raw_message=None):
    return {"tool_calls": [{"name": n, "args": a, "id": None} for n, a in calls],
            "content": content, "raw_message": raw_message}


def _spec(**over):
    base = dict(
        name="toy",
        prompt=lambda cadence: "SYSTEM",
        terminator=Terminator(name="done", schema={"type": "object", "properties": {}},
                              description="finish",
                              handle=lambda args, call_log: {"out": args, "call_log": call_log}),
        execute=lambda client, name, args: client.call_tool(name, args),
        render_result=lambda r: str(r),
        tools_for=lambda client: {"work": {"name": "work", "parameters": {}, "description": "w"}},
        no_tool_nudge="Call a tool.",
        on_exhausted=lambda call_log, last_error: {"exhausted": True, "last_error": last_error},
    )
    base.update(over)
    return ModuleSpec(**base)


def test_terminator_name_and_handle_come_from_the_spec():
    llm = MagicMock()
    llm.chat.side_effect = [_resp(("done", {"x": 1}))]
    result = ToolSpecialist(_spec()).run({"module": "toy"}, llm, MagicMock(), iteration_cap=3)
    assert result == {"out": {"x": 1}, "call_log": []}


def test_execute_comes_from_the_spec():
    seen = []
    llm = MagicMock()
    llm.chat.side_effect = [_resp(("work", {"a": 1})), _resp(("done", {}))]

    def execute(client, name, args):
        seen.append((name, args))
        return {"ok": True, "records": [{"fn": "F", "args": [], "out_vars": ["v"], "publish": ["v"]}]}

    result = ToolSpecialist(_spec(execute=execute)).run(
        {"module": "toy"}, llm, MagicMock(), iteration_cap=3)
    assert seen == [("work", {"a": 1})]
    assert result["call_log"] == [{"fn": "F", "args": [], "out_vars": ["v"], "publish": ["v"]}]


def test_blocks_are_joined_after_the_stage_plan():
    llm = MagicMock()
    llm.chat.side_effect = [_resp(("done", {}))]
    ToolSpecialist(_spec()).run({"module": "toy"}, llm, MagicMock(), iteration_cap=3,
                                blocks=("FIRST BLOCK", "SECOND BLOCK"))
    user = llm.chat.call_args.kwargs["messages"][1]["content"]
    assert user.endswith("FIRST BLOCK\n\nSECOND BLOCK")
    assert user.startswith("{'module': 'toy'}")


def test_falsy_blocks_are_skipped():
    llm = MagicMock()
    llm.chat.side_effect = [_resp(("done", {}))]
    ToolSpecialist(_spec()).run({"module": "toy"}, llm, MagicMock(), iteration_cap=3,
                                blocks=(None, "", "ONLY"))
    user = llm.chat.call_args.kwargs["messages"][1]["content"]
    assert user.count("\n\n") == 1


def test_system_prompt_comes_from_the_spec_with_the_cadence():
    seen = []
    llm = MagicMock()
    llm.chat.side_effect = [_resp(("done", {}))]
    ToolSpecialist(_spec(prompt=lambda c: seen.append(c) or f"P:{c}")).run(
        {"module": "toy"}, llm, MagicMock(), iteration_cap=3, verify_cadence="strict")
    assert seen == ["strict"]
    assert llm.chat.call_args.kwargs["messages"][0]["content"] == "P:strict"


def test_empty_categories_means_no_load_tools_tool_is_offered():
    llm = MagicMock()
    llm.chat.side_effect = [_resp(("done", {}))]
    ToolSpecialist(_spec()).run({"module": "toy"}, llm, MagicMock(), iteration_cap=3)
    offered = [t["name"] for t in llm.chat.call_args.kwargs["tools"]]
    assert offered == ["work", "done"]


def test_empty_risky_categories_never_nudges_a_measure():
    llm = MagicMock()
    llm.chat.side_effect = [_resp(("work", {})), _resp(("done", {}))]
    client = MagicMock()
    client.call_tool.return_value = {"ok": True, "records": []}
    result = ToolSpecialist(_spec()).run({"module": "toy"}, llm, client,
                                         iteration_cap=3, verify_cadence="strict")
    assert "out" in result
    assert llm.chat.call_count == 2


def test_no_tool_call_uses_the_spec_nudge():
    llm = MagicMock()
    llm.chat.side_effect = [_resp(content="prose"), _resp(("done", {}))]
    ToolSpecialist(_spec()).run({"module": "toy"}, llm, MagicMock(), iteration_cap=3)
    second = llm.chat.call_args_list[1].kwargs["messages"]
    assert second[-1]["content"] == "Call a tool."


def test_iteration_cap_uses_the_spec_on_exhausted():
    llm = MagicMock()
    llm.chat.return_value = _resp(("work", {}))
    client = MagicMock()
    client.call_tool.return_value = {"ok": False, "error": "boom"}
    result = ToolSpecialist(_spec()).run({"module": "toy"}, llm, client, iteration_cap=2)
    assert result["exhausted"] is True
    assert result["last_error"]["message"] == "boom"


def test_session_died_becomes_a_fatal_result():
    from salome_agent.executor_client import SessionDied

    llm = MagicMock()
    llm.chat.side_effect = [_resp(("work", {}))]

    def execute(client, name, args):
        raise SessionDied("socket closed")

    result = ToolSpecialist(_spec(execute=execute)).run(
        {"module": "toy"}, llm, MagicMock(), iteration_cap=3)
    assert result["fatal"].startswith("SALOME agent server unavailable")
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/test_tool_specialist.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'salome_agent.specialists.tool_specialist'`.

- [ ] **Step 3: Move the loop**

Create `salome_agent/specialists/tool_specialist.py`. Port `geom.py:212-385` with the three substitutions, keeping every comment that explains a live-found bug — those comments are the record of what the loop costs to rediscover:

```python
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
                            "enum": sorted(self.spec.categories)},
                            "description": "whole tool categories to add"},
                    }}}

    def run(self, stage_plan, llm, client, iteration_cap, *, blocks=(),
            tool_selection="hints", verify_cadence="final", retry=None,
            seed_call_log=None) -> dict:
        try:
            return self._run(stage_plan, llm, client, iteration_cap, blocks,
                             tool_selection, verify_cadence, retry, seed_call_log)
        except (SessionDied, SessionPoisoned) as e:
            return {"fatal": f"SALOME agent server unavailable: {e}"}

    def _run(self, stage_plan, llm, client, iteration_cap, blocks,
             tool_selection, verify_cadence, retry, seed_call_log) -> dict:
        spec = self.spec
        parts = [str(stage_plan)] + [b for b in blocks if b]
        if retry and spec.render_retry:
            rendered = spec.render_retry(retry)
            if rendered:
                parts.append(rendered)
        messages = [{"role": "system", "content": spec.prompt(verify_cadence)},
                    {"role": "user", "content": "\n\n".join(parts)}]
        ...
```

The body from `call_log: list[dict] = ...` down to the final `return` is the current `_run` verbatim, with:

- `finish_def` built from `spec.terminator` (`{"name": spec.terminator.name, "parameters": spec.terminator.schema, "description": spec.terminator.description}`);
- `load_def = self._load_tools_def()` and appended to `tools` only when it is not `None`;
- `all_defs = spec.tools_for(client)`;
- the offered set: `set(all_defs)` when `tool_selection == "all"` **or when `spec.core_tools` is empty**, else `initial_offered(stage_plan, set(all_defs))`;
- the offered order stays `sorted(offered)`, as geom does today. The script modules' two tools (`api_introspect`, `list_study`) already sort into their current order, so one rule covers both and no knob is needed;
- `if name == spec.terminator.name:` in place of the `"finish_stage"` literal;
- `if name == "load_tools" and load_def is not None:` for the load branch;
- `result = spec.execute(client, name, args)` in place of `client.call_tool(...)`;
- `result.get("records") or []` in place of `result["records"]` (a script module's execute carries none);
- `if name in spec.measure_tools:` / `elif all_defs.get(name, {}).get("category") in spec.risky_categories:` — and guard the whole counter block with `if spec.risky_categories:` so an empty set can never match a tool whose `category` key is absent;
- `spec.render_result(result)` in place of `str(_compact_result(result))`;
- the nudge-before-terminator block guarded by `spec.measure_tools and verify_cadence in ("final", "strict")`;
- `return spec.terminator.handle(args, call_log)` in place of the inline `finish_stage` handling;
- `return spec.on_exhausted(call_log, last_error)` as the final return;
- the log line `"  geom stage: offered %d/%d tools (%s)"` becomes `"  %s stage: offered %d/%d tools (%s)"` with `spec.name` first, and `"    geom iter %d/%d"` likewise. These are log strings, not prompt strings, so they may change.

- [ ] **Step 4: Make `run_geom_specialist` a wrapper**

In `specialists/geom.py`, delete `_run` and `_retry_block`'s caller, and replace `run_geom_specialist` with:

```python
def run_geom_specialist(stage_plan: dict, llm, client, iteration_cap: int,
                        upstream: dict | None = None,
                        tool_selection: str = "hints",
                        verify_cadence: str = "final",
                        retry: dict | None = None,
                        seed_call_log: list[dict] | None = None) -> dict:
    """Kept as the geom entry point for the existing tests and callers; the
    loop itself is generic now."""
    blocks = []
    if upstream:
        blocks.append(_upstream_block(upstream))
    return ToolSpecialist(GEOM_SPEC).run(
        stage_plan, llm, client, iteration_cap, blocks=tuple(blocks),
        tool_selection=tool_selection, verify_cadence=verify_cadence,
        retry=retry, seed_call_log=seed_call_log)
```

`_upstream_block(info: dict) -> str` is the *formatting* half of `_geom_upstream` from Task 1 (the part after the measure), extracted so both the wrapper and `_geom_upstream` share one copy of the wording:

```python
def _upstream_block(info: dict) -> str:
    lines = []
    for name, detail in info.items():
        if detail:
            lines.append(f"  {name} — bbox {detail['bbox']}, size {detail['size']}, "
                         f"centre {detail['centre']}")
        else:
            lines.append(f"  {name} — (geometry unavailable)")
    return ("Objects built by an EARLIER stage, given only for reference "
            "(e.g. to measure or align against) — reference them by these exact "
            "names, but do not modify, colour or finish_stage them unless this "
            "stage's own task says to:\n" + "\n".join(lines))
```

and `_geom_upstream` from Task 1 becomes `return _upstream_block(info)` after its measure. Note the iteration-order difference: the old code iterated `upstream.items()` (the measurements dict), `_geom_upstream` iterates the requested `names` list. Keep `_upstream_block` iterating the dict, and have `_geom_upstream` pass a dict built in `names` order — `{n: info.get(n) for n in names}` — so both paths order identically.

- [ ] **Step 5: Run the geom specialist's existing tests unedited**

Run: `.venv/bin/python -m pytest tests/unit/test_specialist_geom.py -q`
Expected: all pass with **zero edits to that file**. If a test needs editing, stop and find what the move changed — an edit here is the plan failing, not the test being stale. The single exception the plan allows: a test asserting on a **log message** that gained a `%s` module name.

- [ ] **Step 6: Run the new tests and the golden runs**

Run: `.venv/bin/python -m pytest tests/unit/test_tool_specialist.py tests/unit/test_golden_runs.py -q`
Expected: **12 passed** (10 new + 2 golden). A golden failure here is the point of the harness: diff the produced `final_script` against `tests/fixtures/golden/<name>.py` and find the wording or ordering that drifted.

- [ ] **Step 7: Run the full suite**

Run: `.venv/bin/python -m pytest tests/unit -q`
Expected: **408 passed** (398 + 10).

- [ ] **Step 8: Commit**

```bash
git add salome_agent/specialists/tool_specialist.py salome_agent/specialists/geom.py \
        tests/unit/test_tool_specialist.py
git commit -m "refactor: one generic ToolSpecialist loop, geom routed through it (#40)"
```

---

### Task 3: Absorb smesh and paravis; delete `base.py`

**Files:**
- Create: `salome_agent/specialists/script_modules.py`
- Delete: `salome_agent/specialists/base.py`
- Modify: `salome_agent/graph.py:187-256` (the fork collapses)
- Modify: `tests/unit/test_specialist_base.py` → rewritten against the spec-driven path
- Create: `tests/unit/test_script_modules.py`

**Interfaces:**
- Consumes: `ToolSpecialist`, `ModuleSpec`, `Terminator`, `register` from Tasks 1-2; `SMESH_SYSTEM_PROMPT`, `PARAVIS_SYSTEM_PROMPT`.
- Produces:
  - `salome_agent.specialists.script_modules.{SMESH_SPEC, PARAVIS_SPEC}`, both registered in `MODULES`.
  - `salome_agent.specialists.script_modules.EMIT_SCRIPT_SCHEMA`, `API_TOOL_SCHEMAS`, `API_TOOL_DESCRIPTIONS` — the three dicts that currently live in `base.py`.
  - `graph.py`'s `_run_stage_node` with no `if module == "geom"` fork.

**The bar here is "smesh still works", not "smesh's bytes are preserved."** smesh's prompt gets rewritten when it gets its own tool layer in a later task, and paravis is out of scope. So absorb them the straightforward way and take the generic loop's behaviour wherever it differs:

- **Batched calls** are now all answered. `base.py` executed only the first and left any other call in the turn without a tool message, which makes the next request malformed.
- **Synthetic tool-call ids** go from `"call_0"` to `f"call_{i}_{j}"`. Opaque to the model.
- **Unknown tool names** now get an `unknown tool` observation instead of no tool message at all — the old path left a dangling `tool_call_id`.

The two blocks worth keeping exact are the **upstream** and **retry** wordings, because those are what "absorbed structurally" means in practice and they cost six lines to pin. Do not spend further test effort on smesh/paravis wording.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_script_modules.py`:

```python
from unittest.mock import MagicMock

from salome_agent.modules import spec_for
from salome_agent.specialists.script_modules import PARAVIS_SPEC, SMESH_SPEC
from salome_agent.specialists.tool_specialist import ToolSpecialist


def _resp(*calls):
    return {"tool_calls": [{"name": n, "args": a, "id": None} for n, a in calls],
            "content": None, "raw_message": None}


def test_both_script_modules_are_registered():
    assert spec_for("smesh") is SMESH_SPEC
    assert spec_for("paravis") is PARAVIS_SPEC


def test_smesh_system_prompt_is_unchanged():
    from salome_agent.specialists.smesh import SMESH_SYSTEM_PROMPT

    assert SMESH_SPEC.prompt("final") == SMESH_SYSTEM_PROMPT
    assert SMESH_SPEC.prompt("strict") == SMESH_SYSTEM_PROMPT


def test_paravis_system_prompt_is_unchanged():
    from salome_agent.specialists.paravis import PARAVIS_SYSTEM_PROMPT

    assert PARAVIS_SPEC.prompt("final") == PARAVIS_SYSTEM_PROMPT


def test_emit_script_terminates_with_the_code():
    llm = MagicMock()
    llm.chat.side_effect = [_resp(("emit_script", {"code": "mesh.Compute()"}))]
    result = ToolSpecialist(SMESH_SPEC).run({"module": "smesh"}, llm, MagicMock(),
                                            iteration_cap=12)
    assert result == {"script": "mesh.Compute()", "warning": None}


def test_upstream_block_wording_is_unchanged():
    assert SMESH_SPEC.render_upstream(None, ["box_1"]) == (
        "Objects already in the study — call find_object() with "
        "exactly these names: ['box_1']")


def test_retry_block_wording_is_unchanged():
    assert SMESH_SPEC.render_retry({"last_error": {"message": "boom"}}) == (
        "Your previous attempt failed at execution with:\nboom\n"
        "Return a corrected script.")
    assert SMESH_SPEC.render_retry({"last_error": None}) is None


def test_api_introspect_executes_through_the_client():
    llm = MagicMock()
    llm.chat.side_effect = [_resp(("api_introspect", {"name": "geompy"})),
                            _resp(("emit_script", {"code": "x"}))]
    client = MagicMock()
    client.introspect.return_value = {"geompy": ["MakeBox"]}
    result = ToolSpecialist(SMESH_SPEC).run({"module": "smesh"}, llm, client,
                                            iteration_cap=12)
    client.introspect.assert_called_once_with(["geompy"])
    assert result["script"] == "x"
    second = llm.chat.call_args_list[1].kwargs["messages"]
    assert second[-1]["role"] == "tool"
    assert "MakeBox" in second[-1]["content"]


def test_no_client_offers_only_the_terminator():
    llm = MagicMock()
    llm.chat.side_effect = [_resp(("emit_script", {"code": "x"}))]
    ToolSpecialist(SMESH_SPEC).run({"module": "smesh"}, llm, None, iteration_cap=12)
    assert [t["name"] for t in llm.chat.call_args.kwargs["tools"]] == ["emit_script"]


def test_iteration_cap_warns_without_emit_script():
    llm = MagicMock()
    llm.chat.return_value = _resp(("list_study", {}))
    result = ToolSpecialist(SMESH_SPEC).run({"module": "smesh"}, llm, MagicMock(),
                                            iteration_cap=2)
    assert result == {"script": "", "warning": "iteration_cap reached without emit_script"}


def test_script_modules_are_never_nudged_to_measure():
    llm = MagicMock()
    llm.chat.side_effect = [_resp(("list_study", {})), _resp(("emit_script", {"code": "x"}))]
    result = ToolSpecialist(SMESH_SPEC).run({"module": "smesh"}, llm, MagicMock(),
                                            iteration_cap=12, verify_cadence="strict")
    assert result["script"] == "x"
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/test_script_modules.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'salome_agent.specialists.script_modules'`.

- [ ] **Step 3: Write `script_modules.py`**

```python
from salome_agent.modules import ModuleSpec, Terminator, register
from salome_agent.specialists.paravis import PARAVIS_SYSTEM_PROMPT
from salome_agent.specialists.smesh import SMESH_SYSTEM_PROMPT

API_TOOL_SCHEMAS = {
    "api_introspect": {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
    },
    "list_study": {"type": "object", "properties": {}},
}

API_TOOL_DESCRIPTIONS = {
    "api_introspect": "Report the members (methods/attributes) of a SALOME API object by name.",
    "list_study": "List the names of every object currently published in the study.",
}

EMIT_SCRIPT_SCHEMA = {
    "type": "object",
    "properties": {"code": {"type": "string"}},
    "required": ["code"],
}
EMIT_SCRIPT_DESCRIPTION = (
    "Submit the finished Python script for this stage as `code`. Call once, when done.")

_SCRIPT_PREAMBLE = """from salome.smesh import smeshBuilder
smesh = smeshBuilder.New()
"""


def _tools_for(client) -> dict:
    if client is None:
        return {}
    return {n: {"name": n, "parameters": API_TOOL_SCHEMAS[n],
                "description": API_TOOL_DESCRIPTIONS[n]}
            for n in ("api_introspect", "list_study")}


def _execute(client, name: str, args: dict) -> dict:
    if name == "api_introspect":
        return {"ok": True, "observation": str(client.introspect([args["name"]]))}
    if name == "list_study":
        return {"ok": True, "observation": str(client.list_study())}
    return {"ok": True, "observation": f"unknown tool {name!r}"}


def _upstream(client, names) -> "str | None":
    # The planner's stage_plan["inputs"] are names it invented from the
    # request ("cube"); the geom stage actually published something else
    # ("box_1"). Give the specialist the real study names to find_object.
    if not names:
        return None
    return ("Objects already in the study — call find_object() with "
            f"exactly these names: {list(names)}")


def _retry(retry: dict) -> "str | None":
    err = (retry or {}).get("last_error")
    if not err:
        return None
    return ("Your previous attempt failed at execution with:\n"
            f"{err.get('message')}\nReturn a corrected script.")


def _script_spec(name: str, system_prompt: str, catalogue: str) -> ModuleSpec:
    return ModuleSpec(
        name=name,
        prompt=lambda cadence, _p=system_prompt: _p,
        terminator=Terminator(name="emit_script", schema=EMIT_SCRIPT_SCHEMA,
                              description=EMIT_SCRIPT_DESCRIPTION,
                              handle=lambda args, call_log: {"script": args["code"],
                                                             "warning": None}),
        execute=_execute,
        render_result=lambda result: result["observation"],
        tools_for=_tools_for,
        no_tool_nudge=("You did not call a tool. Call exactly one tool now — "
                       "emit_script once the script is ready."),
        on_exhausted=lambda call_log, last_error: {
            "script": "", "warning": "iteration_cap reached without emit_script"},
        render_upstream=_upstream,
        render_retry=_retry,
        preamble=_SCRIPT_PREAMBLE,
        catalogue=catalogue,
    )


SMESH_SPEC = register(_script_spec("smesh", SMESH_SYSTEM_PROMPT, catalogue=""))
PARAVIS_SPEC = register(_script_spec("paravis", PARAVIS_SYSTEM_PROMPT, catalogue=""))
```

`catalogue=""` here and in Task 1 is temporary; Task 4 fills all three and asserts the planner prompt comes out byte-identical.

`_SCRIPT_PREAMBLE` is written out **literally** — `script_modules.py` imports nothing from `salome_agent.nodes.assembler` (a specialist importing from nodes is a dependency inversion). Verify it matches `assembler.py:18-20` character for character; Task 4 adds the test that pins the two copies together.

- [ ] **Step 4: Collapse the fork in `graph.py`**

Replace `_run_stage_node`'s body from line 187 (`if module == "geom":`) through line 256 with one path:

```python
    from salome_agent.modules import spec_for
    from salome_agent.specialists.tool_specialist import ToolSpecialist
    import salome_agent.specialists.geom  # noqa: F401 — registers GEOM_SPEC
    import salome_agent.specialists.script_modules  # noqa: F401 — registers the script specs

    spec = spec_for(module)
    upstream = select_upstream(stage_plan, state["stages"], idx)
    blocks = []
    if spec.render_upstream is not None:
        block = spec.render_upstream(client, upstream)
        if block:
            blocks.append(block)
    if module == "geom":
        retry = retry_context(state, idx, client)
        seed = (retry or {}).get("call_log")
    else:
        retry = {"last_error": state.get("last_error")}
        seed = None
    policy = state["config"]["policy"]
    try:
        result = ToolSpecialist(spec).run(
            stage_plan, llm, client, cap, blocks=tuple(blocks),
            tool_selection=policy.get("tool_selection", "hints"),
            verify_cadence=policy.get("verify_cadence", "final"),
            retry=retry, seed_call_log=seed)
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
        if "result_names" in result:
            stage["status"] = "ok"
            stage["result_names"] = result["result_names"]
            update["last_error"] = None
            log.info("  geom specialist done: %d tool call(s), results=%s",
                     len(result.get("call_log") or []), result["result_names"])
        else:
            stage["status"] = "failed"
            update["last_error"] = result.get("last_error")
            log.warning("  geom specialist failed: %s", result.get("last_error"))
```

The `if module == "geom"` that remains is about **retry semantics**, not about which loop runs: geom's retry carries a seeded call log and measured built shapes, a script module's carries the last exec error. That difference is real state, not dispatch, and collapsing it further belongs to a later task.

The result-shape branch is `"script" in result`, keyed on what the terminator produced — not on the module name.

- [ ] **Step 5: Delete `base.py` and rewrite its test**

```bash
git rm salome_agent/specialists/base.py
```

Delete `tests/unit/test_specialist_base.py` and let `test_script_modules.py` carry the cases — one test file per surviving unit. Before deleting, read every test in it: any case that covers a behaviour **`test_script_modules.py` does not already cover** gets ported. Do not port a case that only pins wording the divergences above already changed; smesh's prompt is not being preserved.

- [ ] **Step 6: Run the tests**

Run: `.venv/bin/python -m pytest tests/unit/test_script_modules.py tests/unit/test_graph.py tests/unit/test_golden_runs.py -q`
Expected: all pass. `test_graph.py` exercises `_run_stage_node` and is the place a fork collapse most plausibly breaks — read its failures carefully rather than patching assertions.

- [ ] **Step 7: Run the full suite**

Run: `.venv/bin/python -m pytest tests/unit -q`
Expected: green, with `test_specialist_base.py`'s cases now counted inside `test_script_modules.py`. Record the new number in the commit's verification note.

- [ ] **Step 8: Commit**

```bash
git add salome_agent/specialists/script_modules.py salome_agent/graph.py \
        tests/unit/test_script_modules.py
git rm --cached salome_agent/specialists/base.py tests/unit/test_specialist_base.py 2>/dev/null; true
git add -u salome_agent/specialists tests/unit
git commit -m "refactor: smesh and paravis absorbed into the generic loop, base.py deleted (#40)"
```

(`git add -u <path>` stages only tracked files under the named paths — it is not `git add -A`.)

---

### Task 4: `call_prefix`, `preamble` and the planner catalogue

**Files:**
- Modify: `salome_agent/codegen.py:56-108`
- Modify: `salome_agent/nodes/assembler.py:52-82`
- Modify: `salome_agent/nodes/planner.py:1-54`
- Modify: `salome_agent/specialists/geom.py` (catalogue text)
- Modify: `salome_agent/specialists/script_modules.py` (catalogue text)
- Modify: `tests/unit/test_codegen.py`, `test_assembler.py`, `test_planner.py`, `test_modules.py`

**Interfaces:**
- Consumes: `MODULES`, `routable()` from Task 1; `spec.call_prefix`, `spec.preamble`, `spec.catalogue`.
- Produces: `codegen.generate(call_log, result_names, call_prefix="geompy.")` — new third parameter, defaulted so no caller breaks; `planner.PLANNER_SYSTEM_PROMPT` assembled over `MODULES`.

The assertion for this task is that `PLANNER_SYSTEM_PROMPT` comes out **byte-identical** — `tests/unit/test_golden_surfaces.py` already guards it from Task 0.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_codegen.py`:

```python
def test_call_prefix_is_configurable():
    from salome_agent.codegen import generate

    call_log = [{"fn": "MakeBoxDXDYDZ", "args": [{"kind": "literal", "value": 1}],
                 "out_vars": ["box_1"], "publish": None}]
    assert generate(call_log, ["box_1"], call_prefix="shaper.") == (
        "box_1 = shaper.MakeBoxDXDYDZ(1)")


def test_call_prefix_defaults_to_geompy():
    from salome_agent.codegen import generate

    call_log = [{"fn": "MakeBoxDXDYDZ", "args": [{"kind": "literal", "value": 1}],
                 "out_vars": ["box_1"], "publish": None}]
    assert generate(call_log, ["box_1"]).startswith("box_1 = geompy.")
```

Append to `tests/unit/test_assembler.py`:

```python
def test_geom_preamble_comes_from_the_module_spec():
    from salome_agent.modules import spec_for
    from salome_agent.nodes.assembler import assemble

    state = {"run_id": "r", "user_request": "x", "stages": {
        0: {"module": "geom", "status": "ok", "script": "box_1 = geompy.MakeBoxDXDYDZ(1, 1, 1)"}}}
    assert spec_for("geom").preamble in assemble(state)


def test_smesh_preamble_comes_from_the_module_spec():
    from salome_agent.modules import spec_for
    from salome_agent.nodes.assembler import assemble

    state = {"run_id": "r", "user_request": "x", "stages": {
        0: {"module": "smesh", "status": "ok", "script": "mesh.Compute()"}}}
    assert spec_for("smesh").preamble in assemble(state)


def test_the_two_copies_of_each_preamble_cannot_drift():
    # assembler keeps its own constants for the frozen body-sniffing
    # fallbacks; the specs carry the same text for the module-driven path.
    from salome_agent.modules import spec_for
    from salome_agent.nodes.assembler import _GEOM_INIT, _SMESH_INIT

    assert spec_for("geom").preamble == _GEOM_INIT
    assert spec_for("smesh").preamble == _SMESH_INIT
    assert spec_for("paravis").preamble == _SMESH_INIT
```

Append to `tests/unit/test_planner.py`:

```python
def test_every_routable_module_contributes_its_catalogue():
    from salome_agent.modules import routable
    from salome_agent.nodes.planner import PLANNER_SYSTEM_PROMPT

    specs = {s.name for s in routable()}
    assert specs == {"geom", "smesh", "paravis"}
    for spec in routable():
        assert spec.catalogue.strip() in PLANNER_SYSTEM_PROMPT


def test_a_module_with_no_catalogue_is_not_routable():
    from salome_agent.modules import ModuleSpec, Terminator, routable

    spec = ModuleSpec(
        name="amend", prompt=lambda c: "",
        terminator=Terminator("finish_turn", {}, "", lambda a, c: {}),
        execute=lambda c, n, a: {"ok": True}, render_result=str,
        tools_for=lambda c: {}, no_tool_nudge="", on_exhausted=lambda c, e: {})
    assert spec not in routable()
```

And restore the assertion deleted in Task 1, Step 5:

```python
    assert spec.catalogue  # geom is routable
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/test_codegen.py tests/unit/test_assembler.py tests/unit/test_planner.py tests/unit/test_modules.py -q`
Expected: FAIL — `generate() got an unexpected keyword argument 'call_prefix'`, and the catalogue assertions fail on empty strings.

- [ ] **Step 3: Thread `call_prefix` through codegen**

In `salome_agent/codegen.py`, change the signature and the two emit sites:

```python
def generate(call_log: list[dict], result_names: list[str],
             call_prefix: str = "geompy.") -> str:
```

then `f"{assign_target} = {call_prefix}{call['fn']}({args_src})"` at what is line 86 today, and `f"{tmp} = {call_prefix}{call['fn']}({args_src})"` in the `list_call` branch at line 101. `method_call` and `gui_call` emit through `recv` and `gg.` and are untouched. `geompy.addToStudy` stays literal: it is the publish API, not the module's call prefix — note that in one comment.

Find `generate(`'s callers and pass the stage's prefix:

```bash
grep -rn "codegen.generate\|from salome_agent.codegen import" --include=*.py . | grep -v '^\./\.venv'
```

At each production call site, resolve the prefix as `spec_for(stage["module"]).call_prefix or "geompy."`.

- [ ] **Step 4: Drive the assembler preamble from the specs**

**`assembler.py` keeps its local `_GEOM_INIT` and `_SMESH_INIT` constants.** The body-sniffing fallbacks (`"geompy" in body`, `"smesh" in body`) reference them, and the spec freezes those heuristics — only the *module-driven* half moves to `spec.preamble`. That leaves the same text in two places, so Step 1 below adds a test pinning them equal.

In `salome_agent/nodes/assembler.py`, replace lines 69-72:

```python
    for name in sorted(modules):
        spec = spec_for(name)
        if spec.preamble and spec.preamble not in preamble:
            preamble += spec.preamble
    if "geompy" in body and _GEOM_INIT not in preamble:
        preamble += _GEOM_INIT
    if "smesh" in body and _SMESH_INIT not in preamble:
        preamble += _SMESH_INIT
```

**Verify the emitted order does not change.** Today `_GEOM_INIT` is always emitted before `_SMESH_INIT`. `sorted({"geom", "paravis", "smesh"})` gives `geom, paravis, smesh`, and paravis's preamble is the same smesh init text — so the `not in preamble` guard is what keeps it emitted once, in geom-then-smesh order. Confirm with a test that a geom+smesh+paravis state produces exactly the same preamble as before; if it does not, keep an explicit ordered list of module names rather than `sorted`.

The body-sniffing heuristics (`"SALOMEDS.Color(" in body`, `"gg." in body`, `"find_object(" in body`, `"study_ns" in body`) **stay exactly as they are**. They key on generated content, not on module; turning them into spec fields would be a behaviour change dressed as a refactor.

- [ ] **Step 5: Assemble the planner prompt over `MODULES`**

Give each spec its catalogue text, copied verbatim from `_PLANNER_BASE`'s "Module scope" block in `planner.py:45-52`, so the reassembled prompt is byte-identical:

- geom's `catalogue` is *generated*, from `TOOL_DESCRIPTIONS`, exactly as `_geom_catalogue()` does today. Make it a `ModuleSpec` field holding the assembled string, computed at import in `specialists/geom.py`.
- smesh's and paravis's are their one-liners from the "Module scope" block.

Then in `planner.py`:

```python
PLANNER_SYSTEM_PROMPT = _PLANNER_BASE + "\n\n" + spec_for("geom").catalogue
```

with `_PLANNER_BASE` keeping the smesh/paravis lines where they are for now — **do not restructure the prose**. The spec's `catalogue` field earns its keep as "is this module routable", and geom's additionally carries the generated tool list. Moving the three static one-liners out of `_PLANNER_BASE` would change the prompt's bytes for no gain; give smesh and paravis a catalogue string that equals their existing one-liner and assert it appears in the prompt (which it does, from `_PLANNER_BASE`). Note this explicitly in a comment so the next reader does not "finish the job" and break the bytes.

- [ ] **Step 6: Run the byte guards**

Run: `.venv/bin/python -m pytest tests/unit/test_golden_surfaces.py tests/unit/test_golden_runs.py -q`
Expected: **5 passed**. A planner-prompt byte failure here means the catalogue reassembly drifted — fix the assembly, never the fixture.

- [ ] **Step 7: Run the full suite**

Run: `.venv/bin/python -m pytest tests/unit -q`
Expected: green, +6 tests.

- [ ] **Step 8: Commit**

```bash
git add salome_agent/codegen.py salome_agent/nodes/assembler.py salome_agent/nodes/planner.py \
        salome_agent/specialists/geom.py salome_agent/specialists/script_modules.py \
        tests/unit/test_codegen.py tests/unit/test_assembler.py tests/unit/test_planner.py \
        tests/unit/test_modules.py
git commit -m "refactor: per-module call prefix, preamble and planner catalogue (#40)"
```

---

### Task 5: `ToolRegistry` and the `@tool` decorator — still one file

44 tools, each currently maintained across five parallel dicts (`TOOLS` 1437, `TOOL_CATEGORY` 1487, `ARG_TYPES` 1539, `TOOL_SCHEMAS` 1586, `TOOL_DESCRIPTIONS` 2020). Five edit sites per tool, and only `category` is guard-tested. A registry makes the invariant expressible: every registered tool has a schema, a description, arg types and a category from the known set — one test replacing five opportunities to forget an edit site.

**The wire output must stay byte-identical.** `list_tools()` iterates `TOOLS`, so the registration order must equal today's `TOOLS` literal order, and `TOOL_DESCRIPTIONS`'s order feeds the planner catalogue. Both are already pinned by Task 0's fixtures.

**Files:**
- Create: `executor/registry.py`
- Create: `tests/unit/test_registry.py`
- Modify: `executor/geom_tools.py` (decorate every handler; the five dicts become registry projections)

**Interfaces:**
- Consumes: nothing from earlier tasks — this half is independent.
- Produces:
  ```python
  class ToolRegistry:
      def tool(self, name, *, category, description, schema, arg_types=None): ...  # decorator
      def handlers(self) -> dict[str, Callable]
      def categories(self) -> dict[str, str]
      def arg_types(self) -> dict[str, dict]
      def schemas(self) -> dict[str, dict]
      def descriptions(self) -> dict[str, str]
      def list_tools(self) -> list[dict]
      def coerce_args(self, name: str, args: dict) -> dict
      def call(self, executor, name: str, args: dict) -> dict   # dispatch only, no result decoration
      names: property -> list[str]
  ```
  Every projection preserves registration order.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_registry.py`:

```python
import pytest

from executor.registry import ToolRegistry

_CATEGORIES = {"primitive", "sketch", "swept", "boolean", "transform",
               "feature", "display", "query"}


def _reg():
    r = ToolRegistry(valid_categories=_CATEGORIES)

    @r.tool("make_thing", category="primitive", description="Make a thing.",
            schema={"type": "object", "properties": {"n": {"type": "number"}},
                    "required": ["n"]},
            arg_types={"n": float})
    def tool_make_thing(executor, args):
        return {"ok": True, "n": args["n"]}

    return r


def test_registration_fills_every_projection():
    r = _reg()
    assert list(r.handlers()) == ["make_thing"]
    assert r.categories() == {"make_thing": "primitive"}
    assert r.arg_types() == {"make_thing": {"n": float}}
    assert r.schemas()["make_thing"]["required"] == ["n"]
    assert r.descriptions()["make_thing"] == "Make a thing."


def test_the_decorator_returns_the_function_unchanged():
    r = ToolRegistry(valid_categories=_CATEGORIES)

    def handler(executor, args):
        return {"ok": True}

    assert r.tool("t", category="query", description="d", schema={})(handler) is handler


def test_list_tools_entry_shape():
    assert _reg().list_tools() == [{
        "name": "make_thing",
        "parameters": {"type": "object", "properties": {"n": {"type": "number"}},
                       "required": ["n"]},
        "description": "Make a thing.",
        "category": "primitive"}]


def test_projections_preserve_registration_order():
    r = ToolRegistry(valid_categories=_CATEGORIES)
    for name in ("z", "a", "m"):
        r.tool(name, category="query", description="d", schema={})(lambda e, a: {"ok": True})
    assert list(r.handlers()) == ["z", "a", "m"]
    assert [t["name"] for t in r.list_tools()] == ["z", "a", "m"]


def test_coerce_args_applies_the_declared_types():
    assert _reg().coerce_args("make_thing", {"n": "3"}) == {"n": 3.0}


def test_coerce_args_leaves_an_unknown_tool_alone():
    assert _reg().coerce_args("nope", {"n": "3"}) == {"n": "3"}


def test_an_unknown_category_is_rejected_at_registration():
    r = ToolRegistry(valid_categories=_CATEGORIES)
    with pytest.raises(ValueError, match="category"):
        r.tool("t", category="bogus", description="d", schema={})(lambda e, a: None)


def test_a_missing_description_is_rejected_at_registration():
    r = ToolRegistry(valid_categories=_CATEGORIES)
    with pytest.raises(ValueError, match="description"):
        r.tool("t", category="query", description="", schema={})(lambda e, a: None)


def test_a_duplicate_name_is_rejected():
    r = _reg()
    with pytest.raises(ValueError, match="already registered"):
        r.tool("make_thing", category="query", description="d", schema={})(lambda e, a: None)
```

Add the registry-wide invariant to `tests/unit/test_geom_tools.py` (this is the test that replaces five forgettable edit sites):

```python
def test_every_registered_geom_tool_is_complete():
    from executor.geom_tools import registry

    for name in registry.names:
        assert registry.descriptions()[name], f"no description for {name}"
        assert registry.schemas()[name].get("type") == "object", f"bad schema for {name}"
        assert name in registry.arg_types(), f"no arg_types entry for {name}"
        assert registry.categories()[name] in _VALID_CATEGORIES, f"bad category for {name}"
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/test_registry.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'executor.registry'`.

- [ ] **Step 3: Write `executor/registry.py`**

```python
from typing import Callable


class ToolRegistry:
    """One module's tools, declared once instead of across five parallel dicts.

    shaper and smesh get their own instances of this class; nothing in here
    knows about geom.
    """

    def __init__(self, valid_categories: "set[str] | None" = None):
        self._valid = set(valid_categories or ())
        self._entries: dict[str, dict] = {}

    def tool(self, name: str, *, category: str, description: str, schema: dict,
             arg_types: "dict[str, Callable] | None" = None):
        if name in self._entries:
            raise ValueError(f"tool {name!r} already registered")
        if self._valid and category not in self._valid:
            raise ValueError(f"tool {name!r}: unknown category {category!r}")
        if not description:
            raise ValueError(f"tool {name!r}: empty description")

        def decorate(fn):
            self._entries[name] = {"fn": fn, "category": category,
                                   "description": description, "schema": schema,
                                   "arg_types": dict(arg_types or {})}
            return fn

        return decorate

    @property
    def names(self) -> list:
        return list(self._entries)

    def handlers(self) -> dict:
        return {n: e["fn"] for n, e in self._entries.items()}

    def categories(self) -> dict:
        return {n: e["category"] for n, e in self._entries.items()}

    def arg_types(self) -> dict:
        return {n: e["arg_types"] for n, e in self._entries.items()}

    def schemas(self) -> dict:
        return {n: e["schema"] for n, e in self._entries.items()}

    def descriptions(self) -> dict:
        return {n: e["description"] for n, e in self._entries.items()}

    def list_tools(self) -> list:
        return [{"name": n, "parameters": e["schema"] or {"type": "object", "properties": {}},
                 "description": e["description"], "category": e["category"]}
                for n, e in self._entries.items()]

    def coerce_args(self, name: str, args: dict) -> dict:
        types = self._entries.get(name, {}).get("arg_types") or {}
        out = dict(args)
        for key, caster in types.items():
            if key in out and out[key] is not None:
                try:
                    out[key] = caster(out[key])
                except (TypeError, ValueError):
                    pass
        return out

    def call(self, executor, name: str, args: dict) -> dict:
        entry = self._entries.get(name)
        if entry is None:
            return {"ok": False, "error": f"unknown tool {name!r}"}
        return entry["fn"](executor, self.coerce_args(name, args))
```

`coerce_args` must reproduce `geom_tools._coerce_args` (line 2068) **exactly**, including how it handles a missing key, a `None` value and a failed cast. Read that function first and port its body rather than the sketch above; the sketch is the shape, not the specification.

- [ ] **Step 4: Decorate every geom tool**

In `executor/geom_tools.py`:

```python
from executor.registry import ToolRegistry

_VALID_CATEGORIES = {"primitive", "sketch", "swept", "boolean", "transform",
                     "feature", "display", "query"}
registry = ToolRegistry(valid_categories=_VALID_CATEGORIES)
tool = registry.tool
```

**The name `registry` is load-bearing across three tasks.** Task 6 moves this same instance to `executor/geom/helpers.py` under the same name, and Task 7's test alias re-exports it. Do not rename it to `_REGISTRY` or anything else.

Then, for each of the 44 handlers, move its four facts onto the decorator and **delete its entry from the four dicts**:

```python
@tool("make_box", category="primitive",
      description="Axis-aligned solid box with one corner at the origin, extending +X/+Y/+Z by dx/dy/dz.",
      schema={"type": "object",
              "properties": {"dx": {"type": "number"}, "dy": {"type": "number"},
                             "dz": {"type": "number"}},
              "required": ["dx", "dy", "dz"]},
      arg_types={"dx": float, "dy": float, "dz": float})
def tool_make_box(executor, args):
    ...
```

Then replace the five literal dicts with projections, after every decorator has run:

```python
TOOLS = registry.handlers()
TOOL_CATEGORY = registry.categories()
ARG_TYPES = registry.arg_types()
TOOL_SCHEMAS = registry.schemas()
TOOL_DESCRIPTIONS = registry.descriptions()
```

and point `list_tools` / `call_tool` / `_coerce_args` at the registry, keeping `call_tool`'s result decoration (`_attach_shape_info`, `_delta_note`) exactly where it is:

```python
def _coerce_args(name: str, args: dict) -> dict:
    return registry.coerce_args(name, args)


def list_tools() -> list[dict]:
    return registry.list_tools()
```

**The order trap.** Registration order is *definition* order in the file; the old wire order was the `TOOLS` literal's order. Before deleting the literal, capture it and compare:

```bash
.venv/bin/python - <<'PY'
import re
src = open("executor/geom_tools.py").read()
literal = re.search(r"TOOLS: dict\[str, Callable\] = \{(.*?)\n\}", src, re.S).group(1)
print([m.group(1) for m in re.finditer(r'"([a-z_0-9]+)":', literal)])
print([m.group(1) for m in re.finditer(r'^@tool\("([a-z_0-9]+)"', src, re.M)])
PY
```

If the two lists differ, **reorder the `@tool` decorated definitions** to match the old literal — never the fixture. `ARG_TYPES` had no entry for a handful of tools that take no coercible args; those become `arg_types={}` (which is what `.get(name, {})` already produced), and the new completeness test asserts the key exists rather than that it is non-empty.

- [ ] **Step 5: Run the byte guards**

Run: `.venv/bin/python -m pytest tests/unit/test_golden_surfaces.py -q`
Expected: **3 passed** — both `list_tools()` serializations and the planner prompt unchanged. This is the assertion the spec asks for, and it is the whole value of doing the registry before the split.

- [ ] **Step 6: Run the full suite**

Run: `.venv/bin/python -m pytest tests/unit -q`
Expected: green, +10 tests. `tests/unit/test_geom_tools.py` is **untouched apart from the one added invariant test** — an unchanged 1800-line suite passing is the evidence that 44 mechanical moves changed nothing.

- [ ] **Step 7: Commit**

```bash
git add executor/registry.py executor/geom_tools.py \
        tests/unit/test_registry.py tests/unit/test_geom_tools.py
git commit -m "refactor: geom tools declared through a ToolRegistry decorator (#40)"
```

---

### Task 6: The package split, with `geom_tools.py` as a shim

**Files:**
- Create: `executor/geom/__init__.py`, `helpers.py`, `results.py`, `primitives.py`, `sketch.py`, `swept.py`, `booleans.py`, `transforms.py`, `features.py`, `display.py`, `queries.py`
- Modify: `executor/geom_tools.py` → a re-export shim
- Modify: `pyproject.toml` (`packages` list gains `executor.geom`)

**Interfaces:**
- Consumes: `ToolRegistry` from Task 5.
- Produces: `executor.geom` exporting everything `executor.geom_tools` exported — `TOOLS`, `TOOL_CATEGORY`, `ARG_TYPES`, `TOOL_SCHEMAS`, `TOOL_DESCRIPTIONS`, `list_tools`, `call_tool`, `reset_counters`, `_counters`, `_next_name`, `_resolve`, `_resolve_axis`, `_resolve_plane`, `_resolve_point`, `_publish`, `_parse_coords`, `_gui_call`, `_delta_note`, `_attach_shape_info`, `_select_subshapes`, every `tool_*` handler, and `ArgRef` / `CallRecord`.

**Deviation from the spec's file list:** the spec puts the registry object in `executor/registry.py` (it does) and the geom instance implicitly in the package. Putting the *instance* in `executor/geom/__init__.py` makes every tool submodule import its own parent, which works only by careful ordering. The instance goes in `helpers.py` instead — every tool module already imports helpers, so there is no cycle and no ordering subtlety.

**The caching gotcha:** `agent_server` caches the tools module in `sys.modules`. From this commit on, a running SALOME server keeps serving the OLD registry and the failure is **silent** — stale tool definitions, no error. A full SALOME restart is mandatory before any live verification.

- [ ] **Step 1: Create the package skeleton with the registry instance**

`executor/geom/helpers.py` — move, unchanged: the `ArgRef`/`CallRecord` TypedDicts (`geom_tools.py:5-18`), `_counters`, `reset_counters`, `_next_name`, `_resolve`, `_publish`, `_gui_call`, `_parse_coords`, `_resolve_axis`, `_resolve_plane`, `_resolve_point` — plus:

```python
from executor.registry import ToolRegistry

_VALID_CATEGORIES = {"primitive", "sketch", "swept", "boolean", "transform",
                     "feature", "display", "query"}
registry = ToolRegistry(valid_categories=_VALID_CATEGORIES)
tool = registry.tool
```

`executor/geom/results.py` — move, unchanged: `_attach_shape_info`, `_delta_note`, `_select_subshapes`, and any other shared non-tool result machinery. Find them with:

```bash
grep -n "^def _" executor/geom_tools.py
```

Anything not a `tool_*` handler and not in the helpers list above lands here.

- [ ] **Step 2: Move the handlers, one category file at a time**

For each category in `TOOL_CATEGORY`, move its `@tool`-decorated handlers into the matching file, each importing what it needs from `.helpers` and `.results`. Do one file, run the suite, then the next — a single failing move is far cheaper to locate than eight.

After each file: `.venv/bin/python -m pytest tests/unit -q`

Note `explode` and `get_subshape` are category `feature`, while `get_subshapes` is `query` — follow `TOOL_CATEGORY`, not intuition, so the split matches the declared categories exactly.

- [ ] **Step 3: Write `executor/geom/__init__.py`**

```python
"""The geom tool package. Importing it registers every tool."""
from executor.geom.helpers import (  # noqa: F401
    ArgRef, CallRecord, _counters, _gui_call, _next_name, _parse_coords,
    _publish, _resolve, _resolve_axis, _resolve_plane, _resolve_point,
    registry, reset_counters,
)
from executor.geom.results import (  # noqa: F401
    _attach_shape_info, _delta_note, _select_subshapes,
)
from executor.geom import (  # noqa: F401  — import for registration side effects
    primitives, sketch, swept, booleans, transforms, features, display, queries,
)

# Registration order = the order these submodules are imported above, which
# must match the order list_tools() served before the split (guarded by
# tests/fixtures/golden/list_tools.json).
TOOLS = registry.handlers()
TOOL_CATEGORY = registry.categories()
ARG_TYPES = registry.arg_types()
TOOL_SCHEMAS = registry.schemas()
TOOL_DESCRIPTIONS = registry.descriptions()


def _coerce_args(name: str, args: dict) -> dict:
    return registry.coerce_args(name, args)


def list_tools() -> list[dict]:
    return registry.list_tools()
```

plus `call_tool` moved verbatim from `geom_tools.py:2230-2249`, and a `from executor.geom.<mod> import tool_*` re-export for every handler name the test file reaches for.

**The order trap again:** registration order is now submodule import order × definition order within each file, which will almost certainly differ from the pre-split order. The `list_tools.json` fixture will catch it. Fix it by ordering the imports and, if needed, by having `__init__` project the five dicts through an explicit name order captured from the fixture:

```python
_WIRE_ORDER = [t["name"] for t in json.load(open(...))]  # last resort only
```

Prefer reordering the definitions. Reach for an explicit order list only if the categories genuinely interleave in the old order — check first, and say in the commit which one you did.

- [ ] **Step 4: Turn `geom_tools.py` into a shim**

Replace the whole file with:

```python
"""Compatibility shim — the tools moved to executor.geom.

Kept for one commit so the ~1800-line test file and the five production
import sites stay untouched while the split is proven. Deleted in Task 7.
"""
from executor.geom import *  # noqa: F401,F403
from executor.geom import (  # noqa: F401 — private names star-import skips
    _attach_shape_info, _coerce_args, _counters, _delta_note, _gui_call,
    _next_name, _parse_coords, _publish, _resolve, _resolve_axis,
    _resolve_plane, _resolve_point, _select_subshapes,
)
```

`_counters` is a **mutable module-level dict**: `tests/unit/test_backend.py:11` writes `geom_tools._counters["box"] = 7` and expects `reset_counters()` to clear it. A star import binds the same dict object, so mutation through either name is shared and `reset_counters` (which calls `_counters.clear()` on the original) still works. **Rebinding** would break it — nothing may do `geom_tools._counters = {}`. Confirm `test_backend.py` passes before moving on; it is the canary for this exact hazard.

- [ ] **Step 5: Register the new package**

In `pyproject.toml`, add `"executor.geom"` to `[tool.setuptools] packages`.

- [ ] **Step 6: Run the full suite**

Run: `.venv/bin/python -m pytest tests/unit -q`
Expected: green, with `tests/unit/test_geom_tools.py` **completely untouched**. That is the evidence for this task. Also confirm the byte guards:

Run: `.venv/bin/python -m pytest tests/unit/test_golden_surfaces.py tests/unit/test_golden_runs.py -q`
Expected: **5 passed**.

- [ ] **Step 7: Commit**

```bash
git add executor/geom executor/geom_tools.py pyproject.toml
git commit -m "refactor: split the geom tools into a package behind a re-export shim (#40)"
```

---

### Task 7: Cleanup — imports rewritten, shim deleted, dead branch removed

**Files:**
- Modify: `agent_server/core.py:1`, `agent_server/backend.py:7`, `agent_server/gui_backend.py:10`, `salome_agent/nodes/planner.py:1`
- Modify: `tests/unit/test_geom_tools.py`, `test_geom_tools_naming.py`, `test_backend.py` (import lines only)
- Delete: `executor/geom_tools.py`
- Modify: `salome_agent/state.py:5-9` (`StagePlan.tools`)
- Modify: `salome_agent/graph.py:79-97` (delete the dead scoping branch)
- Modify: `tests/unit/test_graph.py`, `tests/unit/test_state.py`

**Interfaces:**
- Consumes: `executor.geom` from Task 6.
- Produces: `StagePlan` with `tools: list[str]`; `select_upstream(stage_plan, stages, idx)` unchanged in signature, always returning every available name.

**Two corrections carried from the spec, both verified in the code:**

- **`StagePlan.inputs` is NOT removed.** `PARAVIS_SYSTEM_PROMPT` (`specialists/paravis.py:29`) tells the model to take the MED file path from it. What *is* dead is the scoping branch in `select_upstream` (`graph.py:89-96`): `planner.py:27` instructs the planner to put plain words in `inputs`, never object names, so `len(declared) == len(inputs)` essentially never holds. Delete the branch, keep the field. Removing the field would mean editing the planner prompt, which changes the plan JSON, which invalidates every golden baseline this branch depends on.
- **`StagePlan.tools`** has been emitted by the planner since task 193 and never typed. `interactive.py:_PLAN_KEYS` already lists it.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_state.py`:

```python
def test_stage_plan_declares_the_planner_tools_hint():
    from salome_agent.state import StagePlan

    assert StagePlan.__annotations__["tools"] == list[str]
    assert "inputs" in StagePlan.__annotations__  # paravis reads the MED path from it
```

Append to `tests/unit/test_graph.py`:

```python
def test_select_upstream_always_returns_every_available_name():
    from salome_agent.graph import select_upstream

    stages = {0: {"status": "ok", "result_names": ["box_1", "cyl_1"]}}
    # `inputs` holds plain words by design (planner.py), never object names —
    # a full match used to scope down and never fires in practice.
    assert select_upstream({"inputs": ["the box"]}, stages, 1) == ["box_1", "cyl_1"]
    assert select_upstream({"inputs": ["box_1"]}, stages, 1) == ["box_1", "cyl_1"]
    assert select_upstream({"inputs": []}, stages, 1) == ["box_1", "cyl_1"]
```

Create `tests/unit/test_no_shim.py`:

```python
def test_the_geom_tools_shim_is_gone():
    import pytest

    with pytest.raises(ImportError):
        import executor.geom_tools  # noqa: F401
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/test_state.py tests/unit/test_graph.py tests/unit/test_no_shim.py -q`
Expected: three failures — missing `tools` annotation, the scoping branch still returning `["box_1"]`, and the shim still importable.

- [ ] **Step 3: Rewrite the five production imports**

```bash
grep -rn "geom_tools" --include=*.py . | grep -v '^\./\.venv'
```

- `agent_server/core.py:1`: `from executor import geom` — keep the existing comment (`# top-level — salome_init mutates sys.path later`), it is still the reason.
- `agent_server/backend.py:7` and `agent_server/gui_backend.py:10`: same, and `geom_tools.reset_counters()` → `geom.reset_counters()` at the three call sites.
- `salome_agent/nodes/planner.py:1`: `from executor.geom import TOOL_DESCRIPTIONS`.

Then the test imports in `test_geom_tools.py`, `test_geom_tools_naming.py` and `test_backend.py` — a mechanical `from executor import geom_tools` → `from executor import geom as geom_tools`, which keeps every one of the ~1800 lines of assertions unedited. Use the alias deliberately: renaming the local binding too would bury a real failure in a thousand-line diff. Say so in one comment at the top of each file.

- [ ] **Step 4: Delete the shim**

```bash
git rm executor/geom_tools.py
```

- [ ] **Step 5: Type `StagePlan.tools`**

```python
class StagePlan(TypedDict):
    module: Literal["geom", "smesh", "paravis"]
    intent: str
    params: dict
    # Plain-words description of what this stage needs, never object names
    # (planner.py). paravis reads the MED file path from it.
    inputs: list[str]
    # Optional planner hint: the geom tool names this stage expects to use
    # (task 193). Absent on smesh/paravis stages.
    tools: list[str]
```

`TypedDict` makes every key required by default; since `tools` is optional in practice, mark it with `NotRequired` (already a dependency via `typing_extensions`, imported the same way `executor/geom/helpers.py` does) or declare a `total=False` sibling, following whichever pattern `StageRecord` uses at `state.py:19`. Adjust the test's assertion to match the form chosen.

- [ ] **Step 6: Delete the dead scoping branch**

In `salome_agent/graph.py`, `select_upstream` becomes:

```python
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
```

The `stage_plan` parameter stays in the signature — callers pass it, and 194's amend path will want it back.

- [ ] **Step 7: Run the full suite**

Run: `.venv/bin/python -m pytest tests/unit -q`
Expected: green. Confirm the byte guards and the goldens one last time:

Run: `.venv/bin/python -m pytest tests/unit/test_golden_surfaces.py tests/unit/test_golden_runs.py -q`
Expected: **5 passed**.

- [ ] **Step 8: Commit**

```bash
git add agent_server/core.py agent_server/backend.py agent_server/gui_backend.py \
        salome_agent/nodes/planner.py salome_agent/state.py salome_agent/graph.py \
        tests/unit/test_geom_tools.py tests/unit/test_geom_tools_naming.py \
        tests/unit/test_backend.py tests/unit/test_state.py tests/unit/test_graph.py \
        tests/unit/test_no_shim.py
git add -u executor
git commit -m "refactor: drop the geom_tools shim, type StagePlan.tools, remove the dead upstream scoping (#40)"
```

---

### Task 8: Live verification

Unit tests and replayed goldens prove the refactor is faithful to a recording. They cannot prove SALOME still builds geometry. This task is the smoke test, and the user drives it.

**Files:** none. This task produces evidence, not code.

- [ ] **Step 1: Full SALOME restart — mandatory**

`agent_server` caches the tools module in `sys.modules`. After Task 6 a server started before this branch keeps serving the **old** registry, silently. Stop SALOME completely — the server process *and* the GUI — and start it fresh. Confirm from the server's own startup log, not from the fact that a connection succeeds.

Watch for the session-26 gotcha: a headless server started without `--serve-forever` serves exactly one CLI run and exits.

- [ ] **Step 2: Run the two geom baselines live**

```bash
.venv/bin/salome-agent "a vase 200mm tall, 80mm wide at the base, hollow with 4mm walls" --batch -vv
.venv/bin/salome-agent "a 300x200x150mm enclosure with 5mm walls, a 60mm circular vent on the front face, and all vertical edges filleted at 8mm" --batch -vv
```

Compare **bounding boxes within tolerance, not bytes** — a live LLM will not reproduce a trace. The check is that each run produces working geometry of roughly the recorded size, with no `fatal` and no stage marked failed.

- [ ] **Step 3: Confirm each `result.py` runs standalone**

```bash
salome shell -- python3 ~/.local/share/salome-agent/runs/<run_id>/result.py
```

Expected: no traceback. This exercises the assembler preamble path that Task 4 changed.

- [ ] **Step 4: Smoke smesh once**

One smesh stage still meshes. That is the whole claim — the check that "absorbed structurally" did not quietly mean "broken". smesh gets real verification when it gets its own tool layer.

```bash
.venv/bin/salome-agent "a 50mm cube, then a tetrahedral mesh of it with max element size 10mm" --batch -vv
```

**No paravis run.** paravis is out of scope for this branch; it only has to stay importable and routable, which the unit tests already cover. If a paravis stage is broken by this refactor, it is a finding to file, not a step to fix here.

- [ ] **Step 5: Confirm the tool wire is the new registry, not a stale cache**

With the server running, check that `list_tools()` over the wire still returns 44 tools and that a tool added to no dict — there are none, which is the point — cannot go missing:

```bash
.venv/bin/python - <<'PY'
import json
from executor.geom import list_tools
print(len(list_tools()))
PY
```

Expected: `44`. If a live run reports an unknown tool the registry clearly has, the server is serving a cached module — go back to Step 1.

- [ ] **Step 6: Report, do not close**

Post the results as a comment on GH **#40**. Do **not** close the issue; the user closes issues. Do not cite a commit hash that has not been pushed.

---

## Self-Review

**Spec coverage.** Every section of the design maps to a task: `ModuleSpec` contract → Task 1; terminator and "inert by configuration" → Tasks 1-3; `ToolSpecialist` and prompt blocks moving out of the loop → Task 2; smesh/paravis absorbed and `base.py` deleted → Task 3; `codegen` `call_prefix`, `assembler` preamble, planner catalogue → Task 4; registry + `@tool` → Task 5; package split and the `sys.modules` gotcha → Tasks 6 and 8; the five production imports, `StagePlan.tools`, the dead `select_upstream` branch → Task 7; Step-0 replay harness → Tasks 0 and 0b; live verification → Task 8. The spec's seven-commit sequence is Tasks 1-7 in order, with Task 0/0b in front of them as the spec's "Step 0".

**Three places this plan deviates from the spec document, each for a reason found in the code:**
1. `measure_tool: str | None` → `measure_tools: frozenset[str]`. The code tracks two names (`measure`, `check_shape`).
2. `execute` returning `str` → returning a result `dict`, plus a `render_result` field. geom's loop reasons about `ok`/`records`/`error`; a single string cannot carry that, and collapsing them would change geom's transcript.
3. The geom registry **instance** lives in `executor/geom/helpers.py`, not in `executor/geom/__init__.py`. The instance in `__init__` would make every tool submodule import its own parent package.

Plus three fields the spec's dataclass sketch does not list: `no_tool_nudge` and `render_result` (geom's exact wording and transcript are pinned by its goldens, so both must be per-module) and `on_exhausted` (a genuine result-shape difference — `{call_log, last_error}` vs `{script, warning}`, not wording). A fourth, `sort_tools`, was in an earlier draft purely to preserve smesh's tool order; it is gone — `sorted()` already yields `api_introspect, list_study`, so one rule covers both modules.

**Three behaviour divergences in smesh/paravis**, listed in Task 3 and deliberately accepted rather than engineered around: batched calls are now all answered, synthetic tool-call ids change shape, and an unknown tool name now gets a tool message instead of a dangling id. The first and third were latent malformed-request bugs. smesh's prompt is not preserved byte-for-byte because it is being rewritten in its own task next; **geom's is**, and that is what the goldens guard.

**Placeholder scan.** No TBDs. Four steps deliberately say "read the current code and port its body" rather than restating it — `_coerce_args` (Task 5 Step 3), the loop body (Task 2 Step 3), `call_tool`'s result decoration (Task 6 Step 3), and `_fake_server` (Task 0 Step 4). In each case the existing code *is* the specification, and transcribing ~150 lines into a plan would invite a transcription error that the "move, not rewrite" rule exists to prevent. Every such step names the exact file and line range.

**Corrections applied after a pre-execution scan of this plan against the code** (each is also a ruling in the SDD ledger): Task 2's file list claimed a `graph.py` edit its steps never make (removed); Task 3 imported `_SMESH_INIT` from `assembler.py` and then said to drop it (now a literal, with a Task 4 test pinning the copies); Task 4 referenced assembler preamble constants under three different names (now `_GEOM_INIT`/`_SMESH_INIT` throughout); Tasks 5 and 6 named the same registry object `_REGISTRY` and `registry` (now `registry` everywhere); Task 0's recording test used a `_connected_client` helper that does not exist (rewritten against the real `_fake_server`/`_connect` idiom); Task 0's Step 9 count was off by one (389, not 388); `tests/__init__.py` confirmed absent rather than "may not exist".

**Type consistency.** `ModuleSpec`'s field names are used identically in Tasks 1, 2, 3 and 4: `prompt`, `terminator`, `execute`, `render_result`, `tools_for`, `no_tool_nudge`, `on_exhausted`, `core_tools`, `categories`, `risky_categories`, `measure_tools`, `render_upstream`, `render_retry`, `call_prefix`, `preamble`, `catalogue`. `Terminator.handle(args, call_log)` has the same two-argument shape in Tasks 1, 2 and 3. `ToolRegistry`'s five projections (`handlers`, `categories`, `arg_types`, `schemas`, `descriptions`) are named identically in Tasks 5, 6 and 7. `generate(call_log, result_names, call_prefix)` is defined in Task 4 and used nowhere earlier.

**One risk the plan cannot remove:** Task 0b needs a live SALOME and a live LLM. Tasks 1 and 5 can proceed without it (their guards are the offline byte fixtures), but Tasks 2, 3, 4, 6 and 7 all spend the golden-run check, and without captured cassettes those tests skip rather than fail. A skipping golden test looks green. If Task 0b cannot run, say so explicitly before starting Task 2 rather than proceeding with a guard that is silently absent.
