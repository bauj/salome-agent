import pytest

pytestmark = pytest.mark.salome


def test_run_captures_stdout(agent_client):
    result = agent_client.run("print('hi')", timeout=5)
    assert result["ok"] is True
    assert "hi" in result["stdout"]


def test_run_syntax_error_reports_offending_line(agent_client):
    result = agent_client.run("x = (\n", timeout=5)
    assert result["ok"] is False
    assert result["offending_line"] == 1


def test_geompy_call_actually_executes(agent_client):
    result = agent_client.run(
        "geompy.addToStudy(geompy.MakeBoxDXDYDZ(1, 1, 1), 'box_1')", timeout=10)
    assert result["ok"] is True, result
