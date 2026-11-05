# Copyright (C) 2026  salome-agent authors
# LGPL 2.1 or later
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


def test_files_live_under_the_data_dir(tmp_path):
    base = tmp_path / ".local" / "share" / "salome-agent"
    assert handshake.path() == base / "server.json"
    assert handshake.log_path() == str(base / "server.log")


def test_write_makes_the_handshake_owner_only():
    handshake.write(port=1, token="secret", mode="gui", log_path="/x")
    assert handshake.path().stat().st_mode & 0o777 == 0o600


def test_write_removes_an_older_install_s_leftovers(tmp_path):
    old = tmp_path / ".salome_agent"
    old.mkdir()
    for name in ("server.log", "salome-agent-gui.pid"):
        (old / name).write_text("x")
    (old / "server.json").write_text(json.dumps({"proto": handshake.PROTO, "pid": 2 ** 22,
                                                 "pid_start": 1}))
    handshake.write(port=1, token="t", mode="gui", log_path="/x")
    assert not old.exists()


def test_write_keeps_the_old_dir_while_an_old_server_runs_from_it(tmp_path):
    old = tmp_path / ".salome_agent"
    old.mkdir()
    (old / "server.json").write_text(json.dumps({"proto": handshake.PROTO, "pid": 1}))
    handshake.write(port=1, token="t", mode="gui", log_path="/x")
    assert (old / "server.json").exists()


def test_write_keeps_files_it_does_not_know_in_the_old_dir(tmp_path):
    old = tmp_path / ".salome_agent"
    old.mkdir()
    (old / "server.log").write_text("x")
    (old / "notes.txt").write_text("mine")
    handshake.write(port=1, token="t", mode="gui", log_path="/x")
    assert (old / "notes.txt").exists() and not (old / "server.log").exists()
