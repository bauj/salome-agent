from agent_server.core import Core


class FakeBackend:
    mode = "headless"

    def __init__(self):
        self.name_prefix = ""
        self.namespace = {"salome": None}
        self.geom_bindings = {}
        self._poisoned = False
        self.began_with = None
        self.refreshed = []
        self.on_main = False

    def run_tool(self, fn, timeout):
        self.on_main = True
        try:
            return fn()
        finally:
            self.on_main = False

    def begin_session(self, run_id):
        self.began_with = run_id
        self.name_prefix = f"sa_{run_id}_" if run_id else ""

    def end_session(self):
        pass

    def run_code(self, code, timeout, module=None):
        return {"ok": True, "stdout": "", "exception": None}

    def refresh(self, module):
        self.refreshed.append(module)

    def forget(self, names):
        return {"ok": True, "removed": []}

    def find_object(self, name):
        return name

    def leftovers(self):
        return 0

    def poisoned(self):
        return self._poisoned

    def view_image(self):
        return {"ok": True, "image": "data:image/png;base64,AA"}


def test_hello_begins_session_and_returns_prefix():
    b = FakeBackend()
    resp = Core(b).handle({"id": 1, "cmd": "hello", "run_id": "deadbeef1234"})
    assert resp["ready"] is True
    assert resp["mode"] == "headless"
    assert resp["name_prefix"] == "sa_deadbeef1234_"
    assert b.began_with == "deadbeef1234"
    assert resp["id"] == 1


def test_run_ok_triggers_refresh():
    b = FakeBackend()
    Core(b, allow_run=True).handle({"id": 2, "cmd": "run", "code": "x=1", "timeout": 5,
                                    "module": "GEOM"})
    assert b.refreshed == ["GEOM"]


def test_run_is_refused_unless_allowed():
    b = FakeBackend()
    ran = []
    b.run_code = lambda *a, **k: ran.append(a) or {"ok": True}
    resp = Core(b).handle({"id": 2, "cmd": "run", "code": "x=1", "timeout": 5})
    assert resp["ok"] is False and "--allow-run" in resp["error"]
    assert ran == [] and b.refreshed == []


def test_poison_gate_refuses_every_command_including_hello():
    b = FakeBackend()
    b._poisoned = True
    core = Core(b)
    for cmd in ("hello", "run", "list_tools"):
        resp = core.handle({"id": 9, "cmd": cmd})
        assert resp["fatal"] is True
        assert resp["ok"] is False


def test_unknown_command():
    resp = Core(FakeBackend()).handle({"id": 3, "cmd": "frobnicate"})
    assert resp["ok"] is False
    assert "unknown" in resp["exception"]


def test_view_image_answers_with_the_backend_picture():
    resp = Core(FakeBackend()).handle({"id": 4, "cmd": "view_image"})
    assert resp == {"id": 4, "ok": True, "image": "data:image/png;base64,AA"}


def test_headless_backend_has_no_view():
    from agent_server.backend import HeadlessBackend
    r = HeadlessBackend.view_image(None)
    assert r["ok"] is False and "no viewer" in r["error"]


class FakeToolModule:
    def __init__(self, name):
        self.name = name
        self.calls = []
        self.resets = 0
        self.forgot = []
        self.off_main = []

    def _check(self, executor, what):
        if not executor.on_main:
            self.off_main.append(what)

    def list_tools(self):
        return [{"name": f"{self.name}_tool"}]

    def call_tool(self, executor, name, args, pin=None):
        self._check(executor, name)
        self.calls.append((name, args, pin))
        return {"ok": True, "records": []}

    def reset(self, executor):
        self.resets += 1

    def clear(self, executor):
        self._check(executor, "clear")
        self.resets += 1

    def forget(self, executor, names):
        self._check(executor, "forget")
        self.forgot.append(list(names))
        return [n for n in names if n.startswith("sketch")]


def _with_shaper(monkeypatch):
    import agent_server.core as core
    shaper = FakeToolModule("shaper")
    monkeypatch.setitem(core._TOOL_MODULES, "shaper", shaper)
    return shaper


def test_list_tools_defaults_to_geom(monkeypatch):
    _with_shaper(monkeypatch)
    resp = Core(FakeBackend()).handle({"id": 1, "cmd": "list_tools"})
    assert "make_box" in [t["name"] for t in resp["tools"]]


def test_list_tools_for_shaper(monkeypatch):
    _with_shaper(monkeypatch)
    resp = Core(FakeBackend()).handle({"id": 1, "cmd": "list_tools", "module": "shaper"})
    assert resp["tools"] == [{"name": "shaper_tool"}]
    assert resp["module"] == "shaper"


def test_call_tool_routes_to_shaper_and_refreshes_shaper(monkeypatch):
    shaper = _with_shaper(monkeypatch)
    b = FakeBackend()
    resp = Core(b).handle({"id": 1, "cmd": "call_tool", "module": "shaper",
                           "name": "create_sketch", "args": {"plane": "XOY"}})
    assert resp["ok"] is True
    assert shaper.calls == [("create_sketch", {"plane": "XOY"}, None)]
    assert b.refreshed == ["SHAPER"]


def test_unknown_tool_module_is_an_error(monkeypatch):
    _with_shaper(monkeypatch)
    resp = Core(FakeBackend()).handle({"id": 1, "cmd": "call_tool", "module": "paravis",
                                       "name": "x", "args": {}})
    assert resp["ok"] is False
    assert "paravis" in resp["error"]


def test_forget_reaches_shaper_objects_too(monkeypatch):
    shaper = _with_shaper(monkeypatch)
    resp = Core(FakeBackend()).handle({"id": 1, "cmd": "forget",
                                       "names": ["box_1", "sketch_1"]})
    assert shaper.forgot == [["box_1", "sketch_1"]]
    assert "sketch_1" in resp["removed"]


def test_hello_clears_shaper_state(monkeypatch):
    shaper = _with_shaper(monkeypatch)
    core = Core(FakeBackend())
    core.handle({"id": 1, "cmd": "hello", "run_id": "abc"})
    assert shaper.resets == 1


def test_shaper_calls_run_through_backend_run_tool(monkeypatch):
    # In the GUI, SHAPER fires Object Browser events on the calling thread:
    # a call off the Qt main thread crashes SALOME (task 308).
    shaper = _with_shaper(monkeypatch)
    b = FakeBackend()
    core = Core(b)
    core.handle({"id": 1, "cmd": "call_tool", "module": "shaper",
                 "name": "create_sketch", "args": {}})
    core.handle({"id": 2, "cmd": "forget", "names": ["sketch_1"]})
    b.clear_study = lambda: {"ok": True, "removed": 0}
    core.handle({"id": 3, "cmd": "clear_study"})
    assert shaper.calls and shaper.forgot and shaper.resets == 1
    assert shaper.off_main == []


def test_hello_keeps_the_run_dir_for_the_tools():
    b = FakeBackend()
    Core(b).handle({"id": 1, "cmd": "hello", "run_id": "r1", "run_dir": "/w/r1"})
    assert b.run_dir == "/w/r1"
    Core(b).handle({"id": 2, "cmd": "hello", "run_id": "r2"})
    assert b.run_dir is None


def test_a_command_that_raises_answers_with_an_error(monkeypatch):
    b = FakeBackend()

    def boom(names):
        raise KeyError("base_f")
    monkeypatch.setattr(b, "forget", boom)
    r = Core(b).handle({"id": 4, "cmd": "forget", "names": ["base"]})
    assert r == {"id": 4, "ok": False,
                 "error": "forget failed on the server: KeyError: 'base_f'"}
    assert Core(b).handle({"id": 5, "cmd": "list_study"})["id"] == 5


def test_a_timeout_that_poisoned_the_backend_still_answers_fatal(monkeypatch):
    b = FakeBackend()

    def stuck(names):
        b._poisoned = True
        raise TimeoutError("main-thread call exceeded 30s")
    monkeypatch.setattr(b, "forget", stuck)
    r = Core(b).handle({"id": 6, "cmd": "forget", "names": ["x"]})
    assert r["fatal"] is True and r["id"] == 6
