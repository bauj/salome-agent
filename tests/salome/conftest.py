import json
import os
import signal
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
    # Refuse to clobber a server someone else started (a hand-run
    # start-headless-server.sh, or the GUI plugin) — this fixture unlinks the
    # handshake and SIGKILLs a process group at teardown (GH #26).
    from agent_server.handshake import live_owner
    if live_owner() is not None:
        raise RuntimeError(
            "another agent server owns ~/.salome_agent/server.json — stop it "
            "before running the live suite (it would be orphaned otherwise)")
    if os.path.exists(_HS):
        os.unlink(_HS)
    proc = subprocess.Popen(
        [f"{SALOME_ROOT}/salome", "shell", "--", "python3", "-m", "agent_server",
         "--serve-forever"],
        cwd=repo_root, env=_salome_subprocess_env(),
        # Own process group so teardown can SIGKILL the whole SALOME tree
        # (the `salome` launcher forks helpers that a bare proc.terminate()
        # leaves orphaned -- that leak, times one-server-per-test, is an
        # OOM).
        start_new_session=True,
    )
    for _ in range(300):
        if os.path.exists(_HS):
            return proc
        time.sleep(0.2)
    _kill_tree(proc)
    raise TimeoutError("agent_server never wrote its handshake")


def _kill_tree(proc: subprocess.Popen) -> None:
    """Graceful shutdown request, then SIGKILL the whole process group."""
    try:
        data = json.loads(open(_HS).read())
        from salome_agent.executor_client import ExecutorClient
        ExecutorClient(data["host"], data["port"], data["token"], "teardown").shutdown()
    except Exception:
        pass
    try:
        proc.wait(timeout=10)
    except Exception:
        pass
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    try:
        proc.wait(timeout=10)
    except Exception:
        pass
    if os.path.exists(_HS):
        try:
            os.unlink(_HS)
        except OSError:
            pass


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="module")
def running_server():
    """One headless agent-server process for the whole test module.

    Module-scoped on purpose: each server is a full SALOME session (~hundreds
    of MB). Per-test isolation comes from the `hello` handshake, which resets
    geom counters + bindings on every new client connection.
    """
    if SALOME_ROOT is None:
        pytest.skip("set SALOME_ROOT or salome.root in config.json")
    proc = _start_server(_repo_root())
    yield proc
    _kill_tree(proc)


@pytest.fixture
def agent_client(running_server):
    from salome_agent.executor_client import ExecutorClient

    data = json.loads(open(_HS).read())
    client = ExecutorClient(data["host"], data["port"], data["token"], "test")
    yield client
    # Do NOT shutdown() -- the server is shared across the module. Just drop
    # the connection.
    try:
        client._sock.close()
    except Exception:
        pass
