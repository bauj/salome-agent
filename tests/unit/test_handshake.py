import json
import os

import pytest

from agent_server import handshake


@pytest.fixture(autouse=True)
def _home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(os.path, "expanduser",
                        lambda p: p.replace("~", str(tmp_path), 1))
    yield tmp_path


def test_write_then_read_roundtrip():
    handshake.write(port=54123, token="abc", mode="gui",
                    log_path="/x/server.log")
    got = handshake.read()
    assert got["port"] == 54123
    assert got["token"] == "abc"
    assert got["mode"] == "gui"
    assert got["proto"] == handshake.PROTO
    assert got["pid"] == os.getpid()
    assert "pid_start" in got


def test_read_missing_returns_none():
    assert handshake.read() is None


def test_read_proto_mismatch_returns_none():
    p = handshake.path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"proto": 999, "port": 1}))
    assert handshake.read() is None


def test_remove_only_unlinks_own_pid():
    p = handshake.path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"proto": handshake.PROTO, "pid": os.getpid() + 999999}))
    handshake.remove()
    assert p.exists()                       # not ours -> left alone
    p.write_text(json.dumps({"proto": handshake.PROTO, "pid": os.getpid()}))
    handshake.remove()
    assert not p.exists()


def test_live_owner_none_when_missing():
    assert handshake.live_owner() is None


def test_live_owner_none_for_own_pid():
    handshake.write(port=1, token="t", mode="gui", log_path="/x")
    assert handshake.live_owner() is None   # our own entry is not an "owner"


def test_live_owner_none_for_dead_pid():
    p = handshake.path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"proto": handshake.PROTO, "pid": 2 ** 22,
                             "pid_start": 1}))
    assert handshake.live_owner() is None


def test_live_owner_returns_entry_for_live_other_pid():
    # PID 1 is always alive and is never this process.
    p = handshake.path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"proto": handshake.PROTO, "pid": 1, "port": 9,
                             "mode": "headless"}))   # no pid_start -> unverified, still live
    got = handshake.live_owner()
    assert got is not None and got["pid"] == 1


def test_live_owner_none_when_pid_start_mismatch():
    p = handshake.path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"proto": handshake.PROTO, "pid": 1,
                             "pid_start": -12345}))   # wrong start time -> reused pid
    assert handshake.live_owner() is None


def test_dir_and_file_permissions():
    handshake.write(port=1, token="t", mode="headless", log_path="/x")
    assert oct(handshake.path().parent.stat().st_mode)[-3:] == "700"
    assert oct(handshake.path().stat().st_mode)[-3:] == "600"
