import json

import pytest


def send(f, cmd, **payload):
    f.write((json.dumps({"id": 1, "cmd": cmd, **payload}) + "\n").encode())
    f.flush()
    return json.loads(f.readline())


def call_tool(f, name, args):
    return send(f, "call_tool", name=name, args=args)


@pytest.mark.salome
def test_make_box_returns_call_record_and_publishes(executor_conn):
    result = call_tool(executor_conn, "make_box", {"dx": 10, "dy": 10, "dz": 10})
    assert result["ok"] is True
    records = result["records"]
    assert len(records) == 1
    assert records[0]["fn"] == "MakeBoxDXDYDZ"
    assert records[0]["out_vars"] == ["box_1"]
    assert records[0]["publish"] == ["box_1"]


@pytest.mark.salome
def test_fillet_edges_sanity_failure_does_not_publish(executor_conn):
    call_tool(executor_conn, "make_box", {"dx": 1, "dy": 1, "dz": 1})
    # A fillet radius larger than the box's edges is geometrically invalid —
    # the tool must not publish anything for a failed result.
    result = call_tool(executor_conn, "fillet_edges",
                        {"shape": "box_1", "edges": "all", "radius": 50})
    assert result["ok"] is False
    followup = call_tool(executor_conn, "list_study_objects", {})
    assert "fillet_1" not in followup.get("names", [])


@pytest.mark.salome
def test_check_shape_is_not_recordable(executor_conn):
    call_tool(executor_conn, "make_box", {"dx": 1, "dy": 1, "dz": 1})
    result = call_tool(executor_conn, "check_shape", {"shape": "box_1"})
    assert result["ok"] is True
    assert result["records"] == []


@pytest.mark.salome
def test_unknown_tool_returns_structured_error(executor_conn):
    result = call_tool(executor_conn, "not_a_real_tool", {})
    assert result["ok"] is False
    assert "not_a_real_tool" in result["error"]


@pytest.mark.salome
def test_bad_param_returns_structured_error_not_crash(executor_conn):
    result = call_tool(executor_conn, "fillet_edges",
                        {"shape": "nonexistent", "edges": "all", "radius": 2})
    assert result["ok"] is False
    assert "error" in result
    # The socket must still be alive after a bad call.
    followup = call_tool(executor_conn, "list_study_objects", {})
    assert followup["ok"] is True
