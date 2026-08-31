import json
import os
import socket
import subprocess
import tempfile
import time

import pytest

from ._salome_root import SALOME_ROOT


def _salome_subprocess_env() -> dict:
    # Two things confirmed by hands-on testing against a live SALOME
    # install (2026-08-29): (1) an absolute non-"python3" interpreter path
    # sends salomeContext_impl through a codepath that imports psutil and
    # fails; (2) if this process's own venv is active, bare "python3" on
    # PATH resolves to the venv's python (still missing psutil / SALOME's
    # sys.path) since venv/bin is prepended. Strip it out here.
    env = os.environ.copy()
    env["PATH"] = os.pathsep.join(
        p for p in env.get("PATH", "").split(os.pathsep) if ".venv" not in p
    )
    env.pop("VIRTUAL_ENV", None)
    return env


@pytest.fixture
def executor_conn():
    if SALOME_ROOT is None:
        pytest.skip("set SALOME_ROOT or salome.root in config.json")
    sock_path = tempfile.mktemp(suffix=".sock")
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    proc = subprocess.Popen(
        [f"{SALOME_ROOT}/salome", "shell", "--", "python3",
         os.path.join(repo_root, "executor", "executor.py"), "--socket", sock_path],
        env=_salome_subprocess_env(),
    )
    for _ in range(300):
        if os.path.exists(sock_path):
            break
        time.sleep(0.2)
    conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    conn.connect(sock_path)
    f = conn.makefile("rwb")
    ready = json.loads(f.readline())
    assert ready == {"ready": True}
    yield f
    try:
        send(f, "shutdown")
    except Exception:
        pass
    proc.terminate()
    proc.wait(timeout=15)


def send(f, cmd, **payload):
    f.write((json.dumps({"id": 1, "cmd": cmd, **payload}) + "\n").encode())
    f.flush()
    return json.loads(f.readline())
