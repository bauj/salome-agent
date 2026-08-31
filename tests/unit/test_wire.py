import json
import socket
import threading
import time

import pytest

from agent_server.wire import Server


class FakeCore:
    def handle(self, req):
        if req.get("cmd") == "hello":
            return {"id": req.get("id"), "ready": True, "mode": "headless"}
        return {"id": req.get("id"), "ok": True, "echo": req.get("cmd")}


def _client(port):
    s = socket.create_connection(("127.0.0.1", port), timeout=2)
    return s, s.makefile("rwb")


@pytest.fixture
def server():
    srv = Server(FakeCore(), token="secret")
    port = srv.bind()
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield srv, port
    srv.stop()
    t.join(timeout=2)


def test_hello_with_good_token_then_a_command(server):
    _, port = server
    s, f = _client(port)
    f.write(b'{"cmd":"hello","token":"secret","id":1}\n'); f.flush()
    assert json.loads(f.readline())["ready"] is True
    f.write(b'{"cmd":"run","token":"secret","id":2}\n'); f.flush()
    assert json.loads(f.readline())["echo"] == "run"
    s.close()


def test_bad_token_drops_connection(server):
    _, port = server
    s, f = _client(port)
    f.write(b'{"cmd":"hello","token":"WRONG","id":1}\n'); f.flush()
    assert f.readline() == b""            # closed, no response
    s.close()


def test_pre_hello_timeout_frees_the_listener(server):
    srv, port = server
    srv.PRE_HELLO_TIMEOUT = 0.3
    s, _ = _client(port)                  # connect, send nothing
    time.sleep(0.6)
    s2, f2 = _client(port)               # a second client must still get in
    f2.write(b'{"cmd":"hello","token":"secret","id":1}\n'); f2.flush()
    assert json.loads(f2.readline())["ready"] is True
    s.close(); s2.close()


def test_second_live_client_gets_busy(server):
    _, port = server
    s1, f1 = _client(port)
    f1.write(b'{"cmd":"hello","token":"secret","id":1}\n'); f1.flush()
    f1.readline()
    s2, f2 = _client(port)
    f2.write(b'{"cmd":"hello","token":"secret","id":1}\n'); f2.flush()
    assert json.loads(f2.readline())["error"] == "busy"
    s1.close(); s2.close()


def test_serve_forever_once_returns_after_one_client():
    srv = Server(FakeCore(), token="t")
    port = srv.bind()
    t = threading.Thread(target=srv.serve_forever_once, daemon=True)
    t.start()
    s, f = _client(port)
    f.write(b'{"cmd":"hello","token":"t","id":1}\n'); f.flush()
    f.readline()
    f.write(b'{"cmd":"shutdown","token":"t","id":2}\n'); f.flush()
    f.readline()
    s.close()
    t.join(timeout=2)
    assert not t.is_alive()
