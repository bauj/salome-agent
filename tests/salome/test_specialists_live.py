"""Live-LLM smoke test for the smesh + paravis specialists (task 177).

Every other specialist test uses a MagicMock LLM, which cannot catch
message-shape / prompt-wiring / schema bugs — those only surface against a
real endpoint (see the geom-specialist bugs 19-23). This drives the real
smesh/paravis path against the configured LLM and a live SALOME session.

Skipped unless BOTH a real ./config.json is present AND -m salome is run.
"""
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "config.json"

pytestmark = [
    pytest.mark.salome,
    pytest.mark.skipif(not CONFIG_PATH.exists(),
                       reason="no config.json — needs a real LLM endpoint"),
]


@pytest.fixture(scope="module")
def config() -> dict:
    return json.loads(CONFIG_PATH.read_text())


@pytest.fixture(scope="module")
def live_llm(config):
    from salome_agent.llm import LLMClient

    c = config["llm"]
    return LLMClient(base_url=c["base_url"], api_key=c.get("api_key", ""),
                     model=c["model"], ssl_cert_file=c.get("ssl_cert_file", ""),
                     temperature=c.get("temperature", 0))


@pytest.fixture
def client():
    """Own throwaway server per test (check_exec here can poison the backend).

    Uses conftest's _start_server / _kill_tree so the SALOME tree is spawned in
    its own process group and reaped -- a bare proc.terminate(); proc.kill()
    with no follow-up wait() left the launcher <defunct> under pytest (issue #5).
    """
    from salome_agent.executor_client import ExecutorClient

    from .conftest import _HS, SALOME_ROOT, _kill_tree, _repo_root, _start_server

    if SALOME_ROOT is None:
        pytest.skip("set SALOME_ROOT or salome.root in config.json")
    proc = _start_server(_repo_root())
    data = json.loads(open(_HS).read())
    conn = ExecutorClient(data["host"], data["port"], data["token"], "")  # "" = unprefixed names
    try:
        yield conn
    finally:
        try:
            conn._sock.close()
        except Exception:
            pass
        _kill_tree(proc)


def _assert_tool_turns_well_formed(messages):
    """Every {role: tool} message must be preceded by an assistant message
    carrying a tool_calls[] entry with the matching id (the OpenAI contract
    that bug 20 violated)."""
    seen_call_ids = set()
    for m in messages:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                seen_call_ids.add(tc["id"])
        if m.get("role") == "tool":
            assert m["tool_call_id"] in seen_call_ids, \
                f"tool message {m['tool_call_id']} has no preceding assistant tool_call"


def test_smesh_specialist_live_end_to_end(client, live_llm):
    from salome_agent.specialists.script_modules import SMESH_SPEC
    from salome_agent.specialists.tool_specialist import ToolSpecialist
    from salome_agent.verify.exec_check import check_exec
    from salome_agent.verify.introspect import build_api_map
    from salome_agent.verify.static import check_static

    # upstream geom shape the smesh stage will find_object() — published as box_1
    box = client.call_tool("make_box", {"dx": 20, "dy": 20, "dz": 20})
    assert box["ok"], box

    stage_plan = {"module": "smesh", "intent": "tetrahedral mesh of the box, ~5mm elements",
                  "params": {"max_size": 5}, "inputs": ["box_1"]}

    # ToolSpecialist.run keeps `messages` local; grab it via a wrapper so we
    # can assert the tool-turn shape afterwards.
    captured = {}
    real_chat = live_llm.chat

    def spy_chat(messages, tools=None, module=None):
        captured["messages"] = messages
        return real_chat(messages=messages, tools=tools, module=module)

    live_llm.chat = spy_chat
    try:
        result = ToolSpecialist(SMESH_SPEC).run(stage_plan, live_llm, client, iteration_cap=12)
    finally:
        live_llm.chat = real_chat

    assert result["warning"] is None, result["warning"]
    script = result["script"]
    assert script and "Compute" in script, script

    print("\n--- generated smesh script ---\n" + script + "\n---\n")
    _assert_tool_turns_well_formed(captured["messages"])

    static_err = check_static(script, build_api_map(client))
    assert static_err is None, static_err

    exec_err = check_exec(script, "smesh", client, timeout=180)
    assert exec_err is None, exec_err


def test_geom_to_smesh_multistage_live(config, tmp_path, running_server):
    """Full graph, real LLM: geom stage publishes a box, smesh stage must
    find_object() it by the ACTUAL published name (not the planner's guess)
    and mesh it. Exercises the cross-stage handoff + verify routing + the
    assembled result.py."""
    import ast

    from salome_agent.cli import main

    cfg = dict(config)
    cfg["work_dir"] = str(tmp_path)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(cfg))

    rc = main(["make a 20mm cube and mesh it with 5mm tetrahedra", "--batch",
               "--work-dir", str(tmp_path), "--config", str(config_path)])
    assert rc == 0

    report = next(tmp_path.glob("*/report.md")).read_text()
    assert "(geom): ok" in report, report
    assert "(smesh): ok" in report, report

    src = next(tmp_path.glob("*/result.py")).read_text()
    ast.parse(src)
    assert "MakeBoxDXDYDZ" in src
    assert "find_object(" in src
    assert "Compute()" in src


def test_paravis_specialist_live_generates_valid_script(client, live_llm):
    """Lighter than the smesh test: paravis exec-verify needs a real MED file
    on disk from an upstream stage (that path is task 174's geom->smesh->paravis
    chain). Here we only confirm the specialist talks to the real LLM correctly
    and emits a syntactically valid, statically-clean script."""
    from salome_agent.specialists.script_modules import PARAVIS_SPEC
    from salome_agent.specialists.tool_specialist import ToolSpecialist
    from salome_agent.verify.introspect import build_api_map
    from salome_agent.verify.static import check_static

    stage_plan = {"module": "paravis", "intent": "colour the mesh surface by the 'temperature' field",
                  "params": {"field": "temperature"}, "inputs": ["/tmp/mesh_out.med"]}

    captured = {}
    real_chat = live_llm.chat

    def spy_chat(messages, tools=None, module=None):
        captured["messages"] = messages
        return real_chat(messages=messages, tools=tools, module=module)

    live_llm.chat = spy_chat
    try:
        result = ToolSpecialist(PARAVIS_SPEC).run(stage_plan, live_llm, client, iteration_cap=12)
    finally:
        live_llm.chat = real_chat

    assert result["warning"] is None, result["warning"]
    script = result["script"]
    assert script
    print("\n--- generated paravis script ---\n" + script + "\n---\n")

    _assert_tool_turns_well_formed(captured["messages"])
    static_err = check_static(script, build_api_map(client))
    assert static_err is None, static_err
