import json

import pytest


def send(f, cmd, **payload):
    f.write((json.dumps({"id": 1, "cmd": cmd, **payload}) + "\n").encode())
    f.flush()
    return json.loads(f.readline())


@pytest.mark.salome
def test_run_captures_stdout(executor_conn):
    result = send(executor_conn, "run", code="print('hi')", timeout=5)
    assert result["ok"] is True
    assert "hi" in result["stdout"]


@pytest.mark.salome
def test_run_syntax_error_reports_offending_line(executor_conn):
    result = send(executor_conn, "run", code="x = (\n", timeout=5)
    assert result["ok"] is False
    assert result["offending_line"] == 1


@pytest.mark.salome
def test_timeout_kills_infinite_loop(executor_conn):
    result = send(executor_conn, "run", code="while True: pass", timeout=1)
    assert result["ok"] is False
    assert "timeout" in result["exception"].lower()


@pytest.mark.salome
def test_geompy_call_actually_executes(executor_conn):
    result = send(executor_conn, "run",
                   code="geompy.addToStudy(geompy.MakeBoxDXDYDZ(1, 1, 1), 'box_1')",
                   timeout=10)
    assert result["ok"] is True, result
