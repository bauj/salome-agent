import json
import os
import subprocess
import time

import pytest

from ._salome_root import SALOME_ROOT

_HS = os.path.expanduser("~/.salome_agent/server.json")


def _salome_subprocess_env() -> dict:
    # Confirmed by hands-on testing: (1) an absolute non-"python3" interpreter
    # path sends salomeContext_impl through a codepath that imports psutil and
    # fails; (2) if this process's own venv is active, bare "python3" on PATH
    # resolves to the venv's python (missing psutil / SALOME's sys.path) since
    # venv/bin is prepended. Strip it. (scripts/start-headless-server.sh does
    # the same for real use.)
    env = os.environ.copy()
    env["PATH"] = os.pathsep.join(
        p for p in env.get("PATH", "").split(os.pathsep) if ".venv" not in p
    )
    env.pop("VIRTUAL_ENV", None)
    return env


def _start_server(repo_root: str) -> subprocess.Popen:
    if os.path.exists(_HS):
        os.unlink(_HS)
    proc = subprocess.Popen(
        [f"{SALOME_ROOT}/salome", "shell", "--", "python3", "-m", "agent_server",
         "--serve-forever"],
        cwd=repo_root, env=_salome_subprocess_env(),
    )
    for _ in range(300):
        if os.path.exists(_HS):
            return proc
        time.sleep(0.2)
    proc.terminate()
    raise TimeoutError("agent_server never wrote its handshake")


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def running_server():
    """A headless agent server process; tests reach it via the handshake file
    (used by cli.main-driven tests)."""
    if SALOME_ROOT is None:
        pytest.skip("set SALOME_ROOT or salome.root in config.json")
    proc = _start_server(_repo_root())
    yield proc
    proc.terminate()
    proc.wait(timeout=15)


@pytest.fixture
def agent_client(running_server):
    from salome_agent.executor_client import ExecutorClient

    data = json.loads(open(_HS).read())
    client = ExecutorClient(data["host"], data["port"], data["token"], "test")
    yield client
    try:
        client.shutdown()
    except Exception:
        pass
