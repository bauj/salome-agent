import json
import socket
import threading

import pytest

from salome_agent import server_conn
from salome_agent.executor_client import SessionPoisoned


def _fake_server(port_box, replies):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port_box.append(srv.getsockname()[1])
    conn, _ = srv.accept()
    f = conn.makefile("rwb")
    f.readline()
    for r in replies:
        f.write((json.dumps(r) + "\n").encode())
        f.flush()
    conn.close()
    srv.close()


@pytest.fixture
def _home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr("os.path.expanduser",
                        lambda p: p.replace("~", str(tmp_path), 1))
    d = tmp_path / ".salome_agent"
    d.mkdir()
    return d


def _write_hs(_home, port, token="t"):
    (_home / "server.json").write_text(json.dumps(
        {"proto": 1, "host": "127.0.0.1", "port": port, "token": token,
         "pid": 1, "mode": "gui"}))


def test_connect_absent_handshake_raises(_home):
    with pytest.raises(server_conn.ServerUnavailable):
        server_conn.connect("run1")


def test_connect_happy_path(_home):
    box = []
    t = threading.Thread(target=_fake_server,
                         args=(box, [{"ready": True, "proto": 1, "mode": "gui",
                                      "name_prefix": "sa_run1_"}]))
    t.start()
    while not box:
        pass
    _write_hs(_home, box[0])
    client = server_conn.connect("run1")
    assert client.name_prefix == "sa_run1_"
    t.join()


def test_connect_poisoned_raises(_home):
    box = []
    t = threading.Thread(target=_fake_server,
                         args=(box, [{"ready": False, "fatal": True,
                                      "error": "poisoned"}]))
    t.start()
    while not box:
        pass
    _write_hs(_home, box[0])
    with pytest.raises((server_conn.ServerUnavailable, SessionPoisoned)):
        server_conn.connect("run1")
    t.join()
