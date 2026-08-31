import json
import socket
import threading

import pytest

from salome_agent.executor_client import (ExecutorClient, SessionDied,
                                          SessionPoisoned)


def _fake_server(port_box, responses, *, hello=None):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port_box.append(srv.getsockname()[1])
    conn, _ = srv.accept()
    f = conn.makefile("rwb")
    f.readline()  # the hello line
    f.write((json.dumps(hello or {"ready": True, "proto": 1,
                                  "mode": "headless"}) + "\n").encode())
    f.flush()
    for resp in responses:
        f.readline()
        f.write((json.dumps(resp) + "\n").encode())
        f.flush()
    conn.close()
    srv.close()


def _connect(box):
    while not box:
        pass
    return ExecutorClient("127.0.0.1", box[0], "tok", "run1")


def test_run_returns_result():
    box = []
    t = threading.Thread(target=_fake_server,
                         args=(box, [{"id": 1, "ok": True, "stdout": "hi"}]))
    t.start()
    client = _connect(box)
    assert client.run("print('hi')", timeout=5)["stdout"] == "hi"
    t.join()


def test_dropped_socket_raises_session_died():
    box = []
    t = threading.Thread(target=_fake_server, args=(box, []))
    t.start()
    client = _connect(box)
    with pytest.raises(SessionDied):
        client.run("print(1)", timeout=5)
    t.join()


def test_fatal_reply_raises_session_poisoned():
    box = []
    t = threading.Thread(target=_fake_server,
                         args=(box, [{"id": 1, "ok": False, "fatal": True,
                                      "error": "poisoned"}]))
    t.start()
    client = _connect(box)
    with pytest.raises(SessionPoisoned):
        client.run("x", timeout=5)
    t.join()


def test_hello_fatal_raises_session_poisoned():
    box = []
    t = threading.Thread(target=_fake_server,
                         args=(box, []),
                         kwargs={"hello": {"fatal": True, "error": "poisoned"}})
    t.start()
    while not box:
        pass
    with pytest.raises(SessionPoisoned):
        ExecutorClient("127.0.0.1", box[0], "tok", "run1")
    t.join()
