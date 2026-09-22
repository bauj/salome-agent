# tests/unit/test_forget.py
from types import SimpleNamespace

from agent_server.backend import HeadlessBackend, forget_names
from agent_server.core import Core


class FakeStudy:
    def __init__(self, names):
        self.objs = {n: SimpleNamespace(name=n) for n in names}
        self.removed = []

    def FindObjectByName(self, name, component):
        assert component == "GEOM"
        return [self.objs[name]] if name in self.objs else []

    def NewBuilder(self):
        study = self

        class B:
            def RemoveObjectWithChildren(self, so):
                study.removed.append(so.name)
                del study.objs[so.name]
        return B()


def _ns(study):
    return {"salome": SimpleNamespace(myStudy=study), "geompy": object(),
            "smesh": object(), "find_object": lambda n: n}


def test_forget_removes_study_objects_and_bindings():
    study = FakeStudy(["box_1", "fuse_1"])
    bindings = {"box_1": 1, "vec_1": 2, "fuse_1": 3}
    r = forget_names(_ns(study), bindings, ["box_1", "vec_1"])
    assert r == {"ok": True, "removed": ["box_1"]}
    assert study.removed == ["box_1"]
    assert bindings == {"fuse_1": 3}


def test_forget_unknown_name_is_not_an_error():
    r = forget_names(_ns(FakeStudy([])), {}, ["nope_1"])
    assert r == {"ok": True, "removed": []}


def test_headless_backend_forget_uses_its_namespace():
    study = FakeStudy(["box_1"])
    b = HeadlessBackend(build_namespace=lambda live: _ns(study))
    b.geom_bindings["box_1"] = object()
    assert b.forget(["box_1"])["removed"] == ["box_1"]
    assert "box_1" not in b.geom_bindings


def test_core_forget_command_refreshes_geom():
    refreshed = []
    backend = SimpleNamespace(poisoned=lambda: False,
                              forget=lambda names: {"ok": True, "removed": list(names)},
                              refresh=refreshed.append)
    r = Core(backend).handle({"id": 4, "cmd": "forget", "names": ["box_1"]})
    assert r == {"id": 4, "ok": True, "removed": ["box_1"]}
    assert refreshed == ["GEOM"]
