import json
import os
import socket
import tempfile
import threading
import time

import pytest

from salome_agent.executor_client import ExecutorClient, SessionDied


def _wait_for_socket(sock_path, timeout=5):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if os.path.exists(sock_path):
            return
        time.sleep(0.01)
    raise TimeoutError(f"socket never appeared: {sock_path}")


def _fake_server(sock_path, responses):
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(sock_path)
    srv.listen(1)
    conn, _ = srv.accept()
    f = conn.makefile("rwb")
    f.write((json.dumps({"ready": True}) + "\n").encode())
    f.flush()
    for resp in responses:
        f.readline()
        f.write((json.dumps(resp) + "\n").encode())
        f.flush()
    conn.close()


def test_run_returns_result():
    sock_path = tempfile.mktemp(suffix=".sock")
    t = threading.Thread(target=_fake_server, args=(sock_path, [{"id": 1, "ok": True, "stdout": "hi"}]))
    t.start()
    _wait_for_socket(sock_path)
    client = ExecutorClient(sock_path)
    result = client.run("print('hi')", timeout=5)
    assert result["stdout"] == "hi"
    t.join()


def test_dropped_socket_raises_session_died():
    sock_path = tempfile.mktemp(suffix=".sock")
    t = threading.Thread(target=_fake_server, args=(sock_path, []))
    t.start()
    _wait_for_socket(sock_path)
    client = ExecutorClient(sock_path)
    with pytest.raises(SessionDied):
        client.run("print(1)", timeout=5)
    t.join()
