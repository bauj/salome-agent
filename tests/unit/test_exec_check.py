from unittest.mock import MagicMock

from salome_agent.executor_client import SessionDied
from salome_agent.verify.exec_check import check_exec


def test_exec_ok_and_sanity_ok_returns_none():
    client = MagicMock()
    client.run.return_value = {"ok": True, "stdout": "", "exception": None}
    client.sanity.return_value = {"ok": True, "warnings": [], "failures": []}
    assert check_exec("geompy.MakeBoxDXDYDZ(1,1,1)", "geom", client, timeout=5) is None


def test_exec_failure_returns_stage_error():
    client = MagicMock()
    client.run.return_value = {"ok": False, "exception": "boom", "traceback": "...",
                                "offending_line": 3}
    err = check_exec("bad code", "geom", client, timeout=5)
    assert err["phase"] == "exec"
    assert err["offending_line"] == 3


def test_session_died_returns_sentinel():
    client = MagicMock()
    client.run.side_effect = SessionDied()
    assert check_exec("code", "geom", client, timeout=5) == "session_died"


def test_session_poisoned_returns_sentinel():
    from salome_agent.executor_client import SessionPoisoned
    client = MagicMock()
    client.run.side_effect = SessionPoisoned("boom")
    assert check_exec("code", "geom", client, timeout=5) == "session_poisoned"
