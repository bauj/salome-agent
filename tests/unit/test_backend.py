from agent_server.backend import HeadlessBackend


def _fake_ns():
    return {"geompy": object(), "smesh": object(), "find_object": lambda n: n}


def test_begin_session_headless_has_empty_prefix_and_resets_counters():
    # Alias keeps the assertions below unedited after the geom_tools -> geom package move.
    from executor import geom as geom_tools
    b = HeadlessBackend(build_namespace=lambda live: _fake_ns())
    geom_tools._counters["box"] = 7
    b.begin_session("abc123")
    assert b.name_prefix == ""
    assert geom_tools._counters == {}


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


def test_reset_rebuilds_namespace_and_replays():
    made = []
    b = HeadlessBackend(build_namespace=lambda live: made.append(1) or _fake_ns())
    b.begin_session(None)
    made.clear()
    r = b.reset(["a = 1", "b = a + 1"])
    assert r == {"ok": True, "failed_index": None}
    assert made == [1]                       # rebuilt once


def test_reset_reports_failed_index():
    b = HeadlessBackend(build_namespace=lambda live: _fake_ns())
    b.begin_session(None)
    r = b.reset(["a = 1", "a = undefined_name"])
    assert r == {"ok": False, "failed_index": 1}
