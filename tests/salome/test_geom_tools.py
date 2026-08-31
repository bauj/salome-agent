import pytest

pytestmark = pytest.mark.salome


def test_make_box_returns_call_record_and_publishes(agent_client):
    result = agent_client.call_tool("make_box", {"dx": 10, "dy": 10, "dz": 10})
    assert result["ok"] is True
    records = result["records"]
    assert len(records) == 1
    assert records[0]["fn"] == "MakeBoxDXDYDZ"
    assert records[0]["out_vars"] == ["box_1"]
    assert records[0]["publish"] == ["box_1"]


def test_make_box_coerces_string_args(agent_client):
    result = agent_client.call_tool("make_box", {"dx": "10", "dy": "10", "dz": "10"})
    assert result["ok"] is True
    assert result["records"][0]["args"] == [
        {"kind": "literal", "value": 10.0},
        {"kind": "literal", "value": 10.0},
        {"kind": "literal", "value": 10.0},
    ]


def test_fillet_edges_sanity_failure_does_not_publish(agent_client):
    agent_client.call_tool("make_box", {"dx": 1, "dy": 1, "dz": 1})
    result = agent_client.call_tool(
        "fillet_edges", {"shape": "box_1", "edges": "all", "radius": 50})
    assert result["ok"] is False
    followup = agent_client.call_tool("list_study_objects", {})
    assert "fillet_1" not in followup.get("names", [])


def test_check_shape_is_not_recordable(agent_client):
    agent_client.call_tool("make_box", {"dx": 1, "dy": 1, "dz": 1})
    result = agent_client.call_tool("check_shape", {"shape": "box_1"})
    assert result["ok"] is True
    assert result["records"] == []


def test_unknown_tool_returns_structured_error(agent_client):
    result = agent_client.call_tool("not_a_real_tool", {})
    assert result["ok"] is False
    assert "not_a_real_tool" in result["error"]


def test_bad_param_returns_structured_error_not_crash(agent_client):
    result = agent_client.call_tool(
        "fillet_edges", {"shape": "nonexistent", "edges": "all", "radius": 2})
    assert result["ok"] is False
    assert "error" in result
    followup = agent_client.call_tool("list_study_objects", {})
    assert followup["ok"] is True
