# Copyright (C) 2026  salome-agent authors
# LGPL 2.1 or later
import json
import os
import signal
import threading
import time

import pytest

from ._salome_root import SALOME_ROOT
from .conftest import _HS, _kill_tree, _repo_root, _start_server

pytestmark = [pytest.mark.salome,
              pytest.mark.skipif(SALOME_ROOT is None,
                                 reason="set SALOME_ROOT or salome.root in config.json")]


@pytest.fixture
def server():
    proc = _start_server(_repo_root())
    yield json.loads(open(_HS).read())
    _kill_tree(proc)


def _gone(pid, timeout=15):
    end = time.time() + timeout
    while time.time() < end:
        try:
            with open(f"/proc/{pid}/stat") as f:
                if f.read().rsplit(")", 1)[1].split()[0] == "Z":
                    return True
        except OSError:
            return True
        time.sleep(0.2)
    return False


def _assert_stops(data, sig):
    os.kill(data["pid"], sig)
    assert _gone(data["pid"]), f"server still running after {sig.name}"
    assert not os.path.exists(_HS), "server.json left behind"


def _client(data):
    from salome_agent.executor_client import ExecutorClient
    return ExecutorClient(data["host"], data["port"], data["token"], "")


def _ok(client, module, tool, **args):
    r = client.call_tool(tool, args, module=module)
    assert r["ok"] is True, r.get("error")
    return r


@pytest.mark.parametrize("sig", [signal.SIGINT, signal.SIGTERM, signal.SIGHUP],
                         ids=lambda s: s.name)
def test_signal_stops_an_idle_server(server, sig):
    _assert_stops(server, sig)


@pytest.mark.parametrize("sig", [signal.SIGINT, signal.SIGHUP], ids=lambda s: s.name)
def test_signal_stops_the_server_after_every_module_ran(server, sig):
    client = _client(server)
    box = _ok(client, "shaper", "make_box", dx=10, dy=20, dz=30,
              center=[5, 10, 15])["name"]
    _ok(client, "geom", "make_box", dx=10, dy=20, dz=30)
    hexa = _ok(client, "smesh", "create_mesh", shape=box)["name"]
    _ok(client, "smesh", "mesh_hexa", mesh=hexa, segments=4)  # Regular_1D
    assert _ok(client, "smesh", "compute", mesh=hexa)["computed"] is True
    tetra = _ok(client, "smesh", "create_mesh", shape=box)["name"]
    _ok(client, "smesh", "mesh_tetra", mesh=tetra, max_size=5)
    assert _ok(client, "smesh", "compute", mesh=tetra)["computed"] is True
    client.close()
    _assert_stops(server, sig)


def test_signal_stops_the_server_with_a_client_connected(server):
    client = _client(server)
    _ok(client, "shaper", "make_box", dx=10, dy=20, dz=30, center=[5, 10, 15])
    _assert_stops(server, signal.SIGINT)
    client.close()


def test_signal_stops_the_server_during_a_compute(server):
    client = _client(server)
    box = _ok(client, "shaper", "make_box", dx=100, dy=100, dz=100,
              center=[50, 50, 50])["name"]
    mesh = _ok(client, "smesh", "create_mesh", shape=box)["name"]
    _ok(client, "smesh", "mesh_tetra", mesh=mesh, max_size=4)
    done = threading.Event()

    def compute():
        try:
            client.call_tool("compute", {"mesh": mesh}, module="smesh")
        except Exception:
            pass
        done.set()

    threading.Thread(target=compute, daemon=True).start()
    time.sleep(1)
    assert not done.is_set(), "compute ended before the signal: the test needs a bigger mesh"
    _assert_stops(server, signal.SIGINT)
