"""Live-LLM smoke test for the smesh specialist (task 177).

Every other specialist test uses a MagicMock LLM, which cannot catch
message-shape / prompt-wiring / schema bugs — those only surface against a
real endpoint (see the geom-specialist bugs 19-23). This drives the real
smesh path against the configured LLM and a live SALOME session.

Spends tokens: deselected by default; run with `pytest -m llm`. Also skipped
without a real ./config.json.
"""
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "config.json"

pytestmark = [
    pytest.mark.salome,
    pytest.mark.llm,
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
