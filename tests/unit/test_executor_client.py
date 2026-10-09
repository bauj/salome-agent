import json
import socket
import threading

import pytest

from salome_agent.executor_client import (ExecutorClient, ServerOutdated, SessionDied,
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


def _connect(box, **kw):
    while not box:
        pass
    return ExecutorClient("127.0.0.1", box[0], "tok", "run1", **kw)


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


def test_close_drops_the_connection_without_sending_shutdown():
    box, seen = [], []

    def server():
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.bind(("127.0.0.1", 0))
        srv.listen(1)
        box.append(srv.getsockname()[1])
        conn, _ = srv.accept()
        f = conn.makefile("rwb")
        f.readline()  # hello
        f.write((json.dumps({"ready": True, "proto": 1, "mode": "headless"}) + "\n").encode())
        f.flush()
        seen.append(f.readline())  # EOF after close(), not a shutdown line
        conn.close()
        srv.close()

    t = threading.Thread(target=server)
    t.start()
    client = _connect(box)
    client.close()
    t.join()
    assert seen == [b""]


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


def test_record_path_appends_one_json_line_per_call(tmp_path):
    path = tmp_path / "exec_trace.jsonl"
    box = []
    t = threading.Thread(target=_fake_server,
                         args=(box, [{"id": 1, "ok": True, "records": []}]),
                         daemon=True)
    t.start()
    client = _connect(box, record_path=str(path))
    client.call_tool("make_box", {"dx": 1, "dy": 2, "dz": 3})
    t.join()

    lines = [json.loads(l) for l in path.read_text().splitlines()]
    assert lines[-1]["cmd"] == "call_tool"
    assert lines[-1]["payload"] == {"name": "make_box", "args": {"dx": 1, "dy": 2, "dz": 3}}
    assert lines[-1]["response"]["ok"] is True


def test_hello_leftovers_and_clear_study():
    box = []
    t = threading.Thread(target=_fake_server,
                         args=(box, [{"id": 1, "ok": True, "removed": 3}]),
                         kwargs={"hello": {"ready": True, "proto": 1, "mode": "headless",
                                           "name_prefix": "sa_run1_", "leftovers": 3}})
    t.start()
    client = _connect(box)
    assert client.leftovers == 3
    assert client.clear_study() == {"id": 1, "ok": True, "removed": 3}
    t.join()


def test_hello_without_leftovers_defaults_to_zero():
    box = []
    t = threading.Thread(target=_fake_server, args=(box, []))
    t.start()
    client = _connect(box)
    assert client.leftovers == 0
    client.close()
    t.join()


def test_module_goes_on_the_wire_only_when_given(tmp_path):
    path = tmp_path / "exec_trace.jsonl"
    box = []
    t = threading.Thread(target=_fake_server,
                         args=(box, [{"id": 1, "tools": [], "module": "shaper"},
                                     {"id": 2, "ok": True, "records": []}]),
                         daemon=True)
    t.start()
    client = _connect(box, record_path=str(path))
    client.list_tools(module="shaper")
    client.call_tool("create_sketch", {"plane": "XOY"}, module="shaper")
    t.join()

    lines = [json.loads(l) for l in path.read_text().splitlines()]
    assert lines[-2]["payload"] == {"module": "shaper"}
    assert lines[-1]["payload"] == {"name": "create_sketch", "args": {"plane": "XOY"},
                                    "module": "shaper"}


def test_list_tools_refuses_a_server_that_ignored_the_module():
    # A server older than SHAPER routing answers every list_tools with the geom
    # tools and no "module" key; a SHAPER stage must not be handed those.
    box = []
    t = threading.Thread(target=_fake_server,
                         args=(box, [{"id": 1, "tools": [{"name": "make_box"}]}]),
                         daemon=True)
    t.start()
    client = _connect(box)
    with pytest.raises(ServerOutdated, match="shaper"):
        client.list_tools(module="shaper")
    t.join()


def test_list_tools_without_module_accepts_any_server():
    box = []
    t = threading.Thread(target=_fake_server,
                         args=(box, [{"id": 1, "tools": [{"name": "make_box"}]}]),
                         daemon=True)
    t.start()
    client = _connect(box)
    assert client.list_tools()["tools"] == [{"name": "make_box"}]
    t.join()


def _slow_server(port_box, delay):
    import time
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port_box.append(srv.getsockname()[1])
    conn, _ = srv.accept()
    f = conn.makefile("rwb")
    f.readline()
    f.write((json.dumps({"ready": True, "proto": 1, "mode": "headless"}) + "\n").encode())
    f.flush()
    for _ in range(2):
        line = f.readline()
        if not line:
            break
        time.sleep(delay)
        try:
            f.write((json.dumps({"id": 1, "ok": True}) + "\n").encode())
            f.flush()
        except OSError:
            break
    conn.close()
    srv.close()


def test_a_long_tool_waits_past_the_read_timeout(monkeypatch):
    import salome_agent.executor_client as ec
    monkeypatch.setattr(ec, "READ_TIMEOUT", 0.2)
    monkeypatch.setattr("executor.smesh.timeout_for", lambda n: 5 if n == "compute" else None)
    box = []
    t = threading.Thread(target=_slow_server, args=(box, 0.5))
    t.start()
    c = _connect(box)
    assert c.call_tool("compute", {"mesh": "mesh_1"}, module="smesh")["ok"] is True
    with pytest.raises(SessionDied):
        c.call_tool("create_mesh", {"shape": "box_1"}, module="smesh")
    c.close()
    t.join()


def test_run_dir_goes_in_the_hello():
    seen = []
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)

    def serve():
        conn, _ = srv.accept()
        f = conn.makefile("rwb")
        seen.append(json.loads(f.readline()))
        f.write((json.dumps({"ready": True, "proto": 1, "mode": "headless"}) + "\n").encode())
        f.flush()
        conn.close()
        srv.close()

    t = threading.Thread(target=serve, daemon=True)
    t.start()
    ExecutorClient("127.0.0.1", srv.getsockname()[1], "tok", "run1", run_dir="/w/run1").close()
    t.join()
    assert seen[0]["run_dir"] == "/w/run1"


def test_the_script_sanity_command_is_gone():
    assert not hasattr(ExecutorClient, "sanity")
