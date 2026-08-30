from unittest.mock import MagicMock, patch

import pytest

from salome_agent.session.manager import SessionManager


def _mgr():
    return SessionManager(salome_root="/opt/salome")


@patch("salome_agent.session.manager.SessionManager._wait_ready")
@patch("salome_agent.session.manager.subprocess.Popen")
@patch("salome_agent.session.manager.discover_port", return_value=2810)
def test_live_mode_spawns_executor_via_salome_remote(disc, popen, wait_ready):
    popen.return_value = MagicMock(pid=4242)
    wait_ready.return_value = MagicMock()

    handle, client = _mgr().resolve(no_session=False, port=None, live_gui=True)

    argv = popen.call_args[0][0]
    assert argv[:5] == ["/opt/salome/salome", "remote", "-p", "2810", "--"]
    assert argv[5:8] == ["python3", "executor/executor.py", "--socket"]
    assert argv[-1] == "--live"
    assert handle["mode"] == "live"
    assert handle["port"] == 2810
    assert handle["alive"] is True
    assert client is not None


@patch("salome_agent.session.manager.SessionManager._wait_ready")
@patch("salome_agent.session.manager.subprocess.Popen")
@patch("salome_agent.session.manager.discover_port", return_value=None)
def test_live_mode_raises_when_no_session_found(disc, popen, wait_ready):
    with pytest.raises(RuntimeError):
        _mgr().resolve(no_session=False, port=None, live_gui=True)
    popen.assert_not_called()


@patch("salome_agent.session.manager.SessionManager._wait_ready")
@patch("salome_agent.session.manager.subprocess.Popen")
@patch("salome_agent.session.manager.discover_port")
def test_live_mode_honours_explicit_port(disc, popen, wait_ready):
    popen.return_value = MagicMock(pid=1)
    handle, _ = _mgr().resolve(no_session=False, port=9999, live_gui=True)
    disc.assert_not_called()
    assert handle["port"] == 9999
    assert "-p" in popen.call_args[0][0]
    assert "9999" in popen.call_args[0][0]


@patch("salome_agent.session.manager.SessionManager._wait_ready")
@patch("salome_agent.session.manager.subprocess.Popen")
@patch("salome_agent.session.manager.discover_port", return_value=2810)
def test_non_live_still_uses_salome_shell(disc, popen, wait_ready):
    popen.return_value = MagicMock(pid=1)
    handle, _ = _mgr().resolve(no_session=False, port=None, live_gui=False)
    argv = popen.call_args[0][0]
    assert argv[:3] == ["/opt/salome/salome", "shell", "--"]
    assert "--live" not in argv
    assert handle["mode"] == "attach"
