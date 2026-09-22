import pytest

from ._salome_root import SALOME_ROOT

pytestmark = [
    pytest.mark.salome,
    pytest.mark.skipif(SALOME_ROOT is None,
                       reason="set SALOME_ROOT or salome.root in config.json"),
]


def test_forget_then_pinned_rebuild_reuses_the_name(agent_client):
    r = agent_client.call_tool("make_box", {"dx": 10, "dy": 10, "dz": 10})
    name = r["records"][0]["out_vars"][0]

    f = agent_client.forget([name])
    assert f["ok"] and f["removed"] == [name]
    assert agent_client.forget([name])["removed"] == []      # gone from the study

    r2 = agent_client.call_tool("make_box", {"dx": 10, "dy": 10, "dz": 30}, pin=[name])
    assert r2["ok"], r2.get("error")
    assert r2["records"][0]["out_vars"] == [name]
    assert r2["shape_info"]["size"][2] == pytest.approx(30)
    assert agent_client.forget([name])["removed"] == [name]  # exactly one study object

    r3 = agent_client.call_tool("make_box", {"dx": 1, "dy": 1, "dz": 1})
    assert r3["records"][0]["out_vars"][0] != name


def test_pinned_multi_record_call_reuses_the_internal_vector_and_the_output(agent_client):
    box = agent_client.call_tool("make_box", {"dx": 10, "dy": 20, "dz": 5})
    box = box["records"][0]["out_vars"][0]
    r = agent_client.call_tool("rotate", {"shape": box, "axis": "OZ", "angle_deg": 30})
    assert r["ok"], r.get("error")
    out_vars = [v for rec in r["records"] for v in rec["out_vars"]]
    assert len(out_vars) == 2            # the internal "OZ" vector, then the rotated copy

    f = agent_client.forget(out_vars)
    assert out_vars[0] not in f["removed"]   # the vector was never published
    assert out_vars[1] in f["removed"]

    r2 = agent_client.call_tool("rotate", {"shape": box, "axis": "OZ", "angle_deg": 60},
                                pin=out_vars)
    assert r2["ok"], r2.get("error")
    assert [v for rec in r2["records"] for v in rec["out_vars"]] == out_vars
