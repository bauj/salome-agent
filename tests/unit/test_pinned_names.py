# tests/unit/test_pinned_names.py
from types import SimpleNamespace

from executor import geom
from executor.geom import helpers


def _ex(prefix=""):
    return SimpleNamespace(name_prefix=prefix)


def setup_function():
    geom.reset_counters()
    helpers._pinned.clear()


def test_unpinned_allocation_is_unchanged():
    ex = _ex()
    assert helpers._next_name(ex, "box") == "box_1"
    assert helpers._next_name(ex, "box") == "box_2"


def test_pinned_name_is_returned_and_counter_bumped_past_it():
    ex = _ex()
    helpers._pinned[:] = ["box_7"]
    assert helpers._next_name(ex, "box") == "box_7"
    assert helpers._pinned == []
    assert helpers._next_name(ex, "box") == "box_8"


def test_pin_matches_by_prefix_and_leaves_other_prefixes_queued():
    ex = _ex()
    helpers._pinned[:] = ["vec_3", "box_2"]
    assert helpers._next_name(ex, "box") == "box_2"
    assert helpers._pinned == ["vec_3"]
    assert helpers._next_name(ex, "vec") == "vec_3"


def test_pin_respects_the_session_name_prefix():
    ex = _ex("sa_r1_")
    helpers._pinned[:] = ["sa_r1_box_4"]
    assert helpers._next_name(ex, "box") == "sa_r1_box_4"


def test_pin_does_not_match_a_longer_prefix():
    ex = _ex()
    helpers._pinned[:] = ["box_two_1"]
    assert helpers._next_name(ex, "box") == "box_1"
    assert helpers._pinned == ["box_two_1"]


def test_call_tool_scopes_the_pin_to_one_call():
    seen = []

    def fake_tool(executor, args):
        seen.append(helpers._next_name(executor, "box"))
        return {"ok": False, "error": "stop"}

    geom.TOOLS["_fake_pin_tool"] = fake_tool
    try:
        geom.call_tool(_ex(), "_fake_pin_tool", {}, pin=["box_9"])
        geom.call_tool(_ex(), "_fake_pin_tool", {})
    finally:
        del geom.TOOLS["_fake_pin_tool"]
    assert seen == ["box_9", "box_10"]
    assert helpers._pinned == []


def test_core_forwards_pin(monkeypatch):
    from agent_server import core as core_mod
    from agent_server.core import Core

    got = {}

    def fake_call_tool(b, name, args, pin=None):
        got.update(name=name, args=args, pin=pin)
        return {"ok": False, "error": "x"}

    monkeypatch.setattr(core_mod.geom, "call_tool", fake_call_tool)
    backend = SimpleNamespace(poisoned=lambda: False, refresh=lambda m: None)
    Core(backend).handle({"id": 1, "cmd": "call_tool", "name": "make_box",
                          "args": {"dx": 1}, "pin": ["box_1"]})
    assert got == {"name": "make_box", "args": {"dx": 1}, "pin": ["box_1"]}


def test_replay_client_omits_pin_when_none():
    from tests.replay import ReplayExecutorClient

    c = ReplayExecutorClient({"exec": [
        {"cmd": "call_tool", "payload": {"name": "m", "args": {}}, "response": {"ok": True}},
        {"cmd": "call_tool", "payload": {"name": "m", "args": {}, "pin": ["a_1"]},
         "response": {"ok": True}}]})
    assert c.call_tool("m", {}) == {"ok": True}
    assert c.call_tool("m", {}, pin=["a_1"]) == {"ok": True}
