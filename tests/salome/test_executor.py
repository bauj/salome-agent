import json
import os
import socket
import subprocess
import tempfile
import time

import pytest

SALOME_ROOT = "/storage/salome_platform/devel/SALOME-master-native-FD44"


@pytest.fixture
def executor_conn():
    sock_path = tempfile.mktemp(suffix=".sock")
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    # Strip the pytest venv's bin/ from PATH (and unset VIRTUAL_ENV) before
    # launching salome: with the venv active, bare "python3" on PATH
    # resolves to .venv/bin/python3, which lacks psutil and isn't set up
    # with SALOME's sys.path — salomeContext_impl's _runSession then fails
    # importing psutil. Confirmed 2026-08-29: `salome shell -- python3
    # script.py` works from a plain (non-venv) shell; it does not from one
    # with the venv activated, because "python3" resolves differently.
    env = os.environ.copy()
    env["PATH"] = os.pathsep.join(
        p for p in env.get("PATH", "").split(os.pathsep) if ".venv" not in p
    )
    env.pop("VIRTUAL_ENV", None)
    proc = subprocess.Popen(
        [f"{SALOME_ROOT}/salome", "shell", "--", "python3",
         os.path.join(repo_root, "executor", "executor.py"), "--socket", sock_path],
        env=env,
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


@pytest.mark.salome
def test_run_captures_stdout(executor_conn):
    result = send(executor_conn, "run", code="print('hi')", timeout=5)
    assert result["ok"] is True
    assert "hi" in result["stdout"]


@pytest.mark.salome
def test_run_syntax_error_reports_offending_line(executor_conn):
    result = send(executor_conn, "run", code="x = (\n", timeout=5)
    assert result["ok"] is False
    assert result["offending_line"] == 1


@pytest.mark.salome
def test_timeout_kills_infinite_loop(executor_conn):
    result = send(executor_conn, "run", code="while True: pass", timeout=1)
    assert result["ok"] is False
    assert "timeout" in result["exception"].lower()


@pytest.mark.salome
def test_geompy_call_actually_executes(executor_conn):
    result = send(executor_conn, "run",
                   code="geompy.addToStudy(geompy.MakeBoxDXDYDZ(1, 1, 1), 'box_1')",
                   timeout=10)
    assert result["ok"] is True, result
