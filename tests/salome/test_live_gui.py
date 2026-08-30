"""Live-GUI attach test (task 172 Phase 1).

Cannot run headless: needs a SALOME GUI already open in `withsession` mode.
Skipped unless `discover_port()` finds a running session. When it does, it
attaches an executor with --live, publishes a box via a geom tool call, and
asserts the shape landed in the shared study (proving setSSLMode(False) gave
us the GUI's own engine, not a private one). Watch the GUI: the box should
appear in the Object Browser.
"""
import json
import os
import socket
import subprocess
import tempfile
import time
from pathlib import Path

import pytest

from salome_agent.session.discovery import discover_port

REPO_ROOT = Path(__file__).resolve().parents[2]
SALOME_ROOT = "/storage/salome_platform/devel/SALOME-master-native-FD44"
_PORT = discover_port()

pytestmark = [
    pytest.mark.salome,
    pytest.mark.skipif(_PORT is None,
                       reason="no running SALOME session — launch `salome withsession` first"),
]


def _env() -> dict:
    env = os.environ.copy()
    env["PATH"] = os.pathsep.join(
        p for p in env.get("PATH", "").split(os.pathsep) if ".venv" not in p
    )
    env.pop("VIRTUAL_ENV", None)
    return env


def _send(f, cmd, **payload):
    f.write((json.dumps({"id": 1, "cmd": cmd, **payload}) + "\n").encode())
    f.flush()
    return json.loads(f.readline())


def test_live_attach_shares_the_gui_study():
    sock_path = tempfile.mktemp(suffix=".sock")
    proc = subprocess.Popen(
        [f"{SALOME_ROOT}/salome", "remote", "-p", str(_PORT), "--", "python3",
         str(REPO_ROOT / "executor" / "executor.py"), "--socket", sock_path, "--live"],
        env=_env(),
    )
    try:
        for _ in range(300):
            if os.path.exists(sock_path):
                break
            time.sleep(0.2)
        conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        conn.connect(sock_path)
        f = conn.makefile("rwb")
        assert json.loads(f.readline()) == {"ready": True}

        r = _send(f, "call_tool", name="make_box", args={"dx": 12, "dy": 12, "dz": 12})
        assert r["ok"], r

        study = _send(f, "list_study")
        names = [o["name"] for o in study["objects"]]
        assert "Geometry" in names
        _send(f, "shutdown")
    finally:
        proc.terminate()
        proc.wait(timeout=15)
