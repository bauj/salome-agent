"""Live-GUI attach test (task 172 Phase 1).

Cannot run headless: needs a SALOME GUI already open in `withsession` mode.
Skipped unless `discover_port()` finds a running session. When it does, it
attaches an executor with --live, publishes a box via a geom tool call, and
asserts the shape landed in the shared study (proving setSSLMode(False) gave
us the GUI's own engine, not a private one). Watch the GUI: the box should
appear in the Object Browser.

WARNING: attaching/detaching external SALOME clients to a live GUI session
has crashed the session at least once (task 172, 2026-08-31 — no core, root
cause not found). Run this against a GUI you can afford to lose, save your
work first, and don't stack it on top of several prior --live-gui runs in
the same session.
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

from ._salome_root import SALOME_ROOT

REPO_ROOT = Path(__file__).resolve().parents[2]
_PORT = discover_port()

pytestmark = [
    pytest.mark.salome,
    pytest.mark.skipif(_PORT is None,
                       reason="no running SALOME session — launch `salome withsession` first"),
    pytest.mark.skipif(SALOME_ROOT is None,
                       reason="set SALOME_ROOT or salome.root in config.json"),
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

        # Graceful shutdown: let the executor close its own CORBA connections
        # and exit before we fall through to terminate(). A hard SIGTERM while
        # it still holds handles to the live session is a suspected trigger
        # for the session crash noted in the module docstring.
        _send(f, "shutdown")
        f.close()
        conn.close()
        proc.wait(timeout=15)
    finally:
        if proc.poll() is None:
            proc.terminate()
            proc.wait(timeout=15)
