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


@patch("salome_agent.session.manager.time.sleep")
@patch("salome_agent.session.manager.SessionManager._wait_ready")
@patch("salome_agent.session.manager.subprocess.Popen")
@patch("salome_agent.session.manager.discover_port")
def test_live_mode_auto_launches_gui_when_none_running(disc, popen, wait_ready, _sleep):
    # No session at first; after the GUI comes up, a port appears.
    disc.side_effect = [None, None, 3777]
    popen.return_value = MagicMock(pid=1, poll=MagicMock(return_value=None))
    wait_ready.return_value = MagicMock()

    handle, client = _mgr().resolve(no_session=False, port=None, live_gui=True)

    launch_argv = popen.call_args_list[0][0][0]
    assert launch_argv == ["/opt/salome/salome", "withsession", "-g"]
    exec_argv = popen.call_args_list[1][0][0]
    assert exec_argv[:5] == ["/opt/salome/salome", "remote", "-p", "3777", "--"]
    assert exec_argv[-1] == "--live"
    assert handle["mode"] == "live"
    assert handle["port"] == 3777
    assert handle["owned"] is True


@patch("salome_agent.session.manager.time.sleep")
@patch("salome_agent.session.manager.SessionManager._wait_ready")
@patch("salome_agent.session.manager.subprocess.Popen")
@patch("salome_agent.session.manager.discover_port", return_value=None)
def test_live_mode_raises_when_gui_launch_exits_early(disc, popen, wait_ready, _sleep):
    popen.return_value = MagicMock(pid=1, returncode=1,
                                   poll=MagicMock(return_value=1))
    with pytest.raises(RuntimeError):
        _mgr().resolve(no_session=False, port=None, live_gui=True)


@patch("salome_agent.session.manager.SessionManager._wait_ready")
@patch("salome_agent.session.manager.subprocess.Popen")
@patch("salome_agent.session.manager.discover_port", return_value=2810)
def test_live_mode_attaches_to_running_session_without_launching(disc, popen, wait_ready):
    popen.return_value = MagicMock(pid=1)
    handle, _ = _mgr().resolve(no_session=False, port=None, live_gui=True)
    # only the executor is spawned, no `withsession` launch
    assert popen.call_count == 1
    assert popen.call_args[0][0][1] == "remote"
    assert handle["owned"] is False


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
