"""Live-LLM smoke test for the smesh + paravis specialists (task 177).

Every other specialist test uses a MagicMock LLM, which cannot catch
message-shape / prompt-wiring / schema bugs — those only surface against a
real endpoint (see the geom-specialist bugs 19-23). This drives the real
smesh/paravis path against the configured LLM and a live SALOME session.

Skipped unless BOTH a real ./config.json is present AND -m salome is run.
"""
import json
import os
import socket
import subprocess
import tempfile
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "config.json"

pytestmark = [
    pytest.mark.salome,
    pytest.mark.skipif(not CONFIG_PATH.exists(),
                       reason="no config.json — needs a real LLM endpoint"),
]


def _salome_subprocess_env() -> dict:
    env = os.environ.copy()
    env["PATH"] = os.pathsep.join(
        p for p in env.get("PATH", "").split(os.pathsep) if ".venv" not in p
    )
    env.pop("VIRTUAL_ENV", None)
    return env


@pytest.fixture(scope="module")
def config() -> dict:
    return json.loads(CONFIG_PATH.read_text())


@pytest.fixture(scope="module")
def live_llm(config):
    from salome_agent.llm import LLMClient

    c = config["llm"]
    return LLMClient(base_url=c["base_url"], api_key=c.get("api_key", ""),
                     model=c["model"], temperature=c.get("temperature", 0))


@pytest.fixture
def client(config):
    from salome_agent.executor_client import ExecutorClient

    sock_path = tempfile.mktemp(suffix=".sock")
    proc = subprocess.Popen(
        [f"{config['salome']['root']}/salome", "shell", "--", "python3",
         str(REPO_ROOT / "executor" / "executor.py"), "--socket", sock_path],
        env=_salome_subprocess_env(),
    )
    try:
        for _ in range(400):
            if os.path.exists(sock_path):
                break
            time.sleep(0.2)
        else:
            raise RuntimeError("executor socket never appeared")
        yield ExecutorClient(sock_path)
    finally:
        try:
            proc.terminate()
            proc.wait(timeout=15)
        except Exception:
            proc.kill()


def _smesh_tools(client):
    return {
        "api_introspect": lambda name: str(client.introspect([name])),
        "list_study": lambda: str(client.list_study()),
    }


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
    from salome_agent.specialists.base import run_specialist
    from salome_agent.specialists.smesh import SMESH_SYSTEM_PROMPT
    from salome_agent.verify.exec_check import check_exec
    from salome_agent.verify.introspect import build_api_map
    from salome_agent.verify.static import check_static

    # upstream geom shape the smesh stage will find_object() — published as box_1
    box = client.call_tool("make_box", {"dx": 20, "dy": 20, "dz": 20})
    assert box["ok"], box

    stage_plan = {"module": "smesh", "intent": "tetrahedral mesh of the box, ~5mm elements",
                  "params": {"max_size": 5}, "inputs": ["box_1"]}

    # run_specialist mutates a local `messages`; grab it via a wrapper so we
    # can assert the tool-turn shape afterwards.
    captured = {}
    real_chat = live_llm.chat

    def spy_chat(messages, tools=None):
        captured["messages"] = messages
        return real_chat(messages=messages, tools=tools)

    live_llm.chat = spy_chat
    try:
        result = run_specialist(stage_plan, live_llm, _smesh_tools(client),
                                iteration_cap=12, system_prompt=SMESH_SYSTEM_PROMPT)
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


def test_paravis_specialist_live_generates_valid_script(client, live_llm):
    """Lighter than the smesh test: paravis exec-verify needs a real MED file
    on disk from an upstream stage (that path is task 174's geom->smesh->paravis
    chain). Here we only confirm the specialist talks to the real LLM correctly
    and emits a syntactically valid, statically-clean script."""
    from salome_agent.specialists.base import run_specialist
    from salome_agent.specialists.paravis import PARAVIS_SYSTEM_PROMPT
    from salome_agent.verify.introspect import build_api_map
    from salome_agent.verify.static import check_static

    stage_plan = {"module": "paravis", "intent": "colour the mesh surface by the 'temperature' field",
                  "params": {"field": "temperature"}, "inputs": ["/tmp/mesh_out.med"]}

    captured = {}
    real_chat = live_llm.chat

    def spy_chat(messages, tools=None):
        captured["messages"] = messages
        return real_chat(messages=messages, tools=tools)

    live_llm.chat = spy_chat
    try:
        result = run_specialist(stage_plan, live_llm, _smesh_tools(client),
                                iteration_cap=12, system_prompt=PARAVIS_SYSTEM_PROMPT)
    finally:
        live_llm.chat = real_chat

    assert result["warning"] is None, result["warning"]
    script = result["script"]
    assert script
    print("\n--- generated paravis script ---\n" + script + "\n---\n")

    _assert_tool_turns_well_formed(captured["messages"])
    static_err = check_static(script, build_api_map(client))
    assert static_err is None, static_err
