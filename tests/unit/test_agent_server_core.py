from agent_server.core import Core


class FakeBackend:
    mode = "headless"

    def __init__(self):
        self.name_prefix = ""
        self.namespace = {"salome": None}
        self.geom_bindings = {}
        self.accepted_scripts = []
        self._poisoned = False
        self.began_with = None
        self.refreshed = []

    def begin_session(self, run_id):
        self.began_with = run_id
        self.name_prefix = f"sa_{run_id}_" if run_id else ""

    def end_session(self):
        pass

    def run_code(self, code, timeout, module=None):
        return {"ok": True, "stdout": "", "exception": None}

    def reset(self, scripts):
        return {"ok": True, "failed_index": None}

    def refresh(self, module):
        self.refreshed.append(module)

    def find_object(self, name):
        return name

    def poisoned(self):
        return self._poisoned


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
    Core(b).handle({"id": 2, "cmd": "run", "code": "x=1", "timeout": 5,
                    "module": "GEOM"})
    assert b.refreshed == ["GEOM"]


def test_poison_gate_refuses_every_command_including_hello():
    b = FakeBackend()
    b._poisoned = True
    core = Core(b)
    for cmd in ("hello", "run", "list_tools", "sanity"):
        resp = core.handle({"id": 9, "cmd": cmd})
        assert resp["fatal"] is True
        assert resp["ok"] is False


def test_unknown_command():
    resp = Core(FakeBackend()).handle({"id": 3, "cmd": "frobnicate"})
    assert resp["ok"] is False
    assert "unknown" in resp["exception"]
