import pytest

pytestmark = pytest.mark.salome


def test_timeout_kills_infinite_loop(agent_client):
    # Own file on purpose: a `timeout after Ns` latches backend._poisoned for
    # the server's whole lifetime (the abandoned-watchdog hazard -- the
    # busy-loop thread can't be killed in-process), after which every request,
    # `hello` included, gets _FATAL. Sharing the module server with other tests
    # made them ERROR at fixture setup (issue #4). The module-scoped
    # `running_server` here is torn down straight after this one test.
    result = agent_client.run("while True: pass", timeout=1)
    assert result["ok"] is False
    assert "timeout" in result["exception"].lower()
