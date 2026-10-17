from agent_server.backend import HeadlessBackend


def _fake_ns():
    return {"geompy": object(), "smesh": object(), "find_object": lambda n: n}


def test_begin_session_headless_prefixes_by_run_and_resets_counters():
    # Alias keeps the assertions below unedited after the geom_tools -> geom package move.
    from executor import geom as geom_tools
    b = HeadlessBackend(build_namespace=lambda live: _fake_ns())
    geom_tools._counters["box"] = 7
    b.begin_session("abc123")
    assert b.name_prefix == "sa_abc123_"
    assert geom_tools._counters == {}
    b.begin_session(None)
    assert b.name_prefix == ""


def test_headless_find_object_tries_the_prefixed_name_first():
    ns = _fake_ns()
    ns["find_object"] = lambda n: {"sa_r1_box_1": "mine", "box_1": "old"}[n]
    b = HeadlessBackend(build_namespace=lambda live: ns)
    b.begin_session("r1")
    assert b.find_object("box_1") == "mine"
    assert b.find_object("sa_r1_box_1") == "mine"


def test_headless_find_object_falls_back_to_the_bare_name():
    ns = _fake_ns()
    ns["find_object"] = lambda n: {"cube": "user"}[n]
    b = HeadlessBackend(build_namespace=lambda live: ns)
    b.begin_session("r1")
    assert b.find_object("cube") == "user"


def test_run_code_ok():
    b = HeadlessBackend(build_namespace=lambda live: _fake_ns())
    b.begin_session(None)
    r = b.run_code("x = 1 + 1", timeout=5)
    assert r["ok"] is True


def test_run_code_exception_reports_offending_line():
    b = HeadlessBackend(build_namespace=lambda live: _fake_ns())
    b.begin_session(None)
    r = b.run_code("raise ValueError('boom')", timeout=5)
    assert r["ok"] is False
    assert r["offending_line"] == 1


def test_timeout_poisons_the_backend():
    b = HeadlessBackend(build_namespace=lambda live: _fake_ns())
    b.begin_session(None)
    assert b.poisoned() is False
    r = b.run_code("while True: pass", timeout=1)
    assert r["ok"] is False and "timeout" in r["exception"].lower()
    assert b.poisoned() is True


def test_run_tool_timeout_poisons_the_backend():
    import time
    import pytest

    b = HeadlessBackend(build_namespace=lambda live: _fake_ns())
    b.begin_session(None)
    assert b.poisoned() is False
    with pytest.raises(TimeoutError, match="exceeded"):
        b.run_tool(lambda: time.sleep(2.0), timeout=0.1)
    assert b.poisoned() is True


def test_counters_are_isolated_per_backend_instance():
    from executor.geom import helpers

    b1 = HeadlessBackend(build_namespace=lambda live: _fake_ns())
    b2 = HeadlessBackend(build_namespace=lambda live: _fake_ns())
    b1.begin_session("r1")
    b2.begin_session("r2")
    assert helpers._next_name(b1, "box") == "sa_r1_box_1"
    assert helpers._next_name(b1, "box") == "sa_r1_box_2"
    assert helpers._next_name(b2, "box") == "sa_r2_box_1"
    b2.begin_session("r2")
    assert helpers._next_name(b1, "box") == "sa_r1_box_3"


