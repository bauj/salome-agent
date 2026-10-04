from types import SimpleNamespace

from agent_server.backend import HeadlessBackend, clear_study, study_objects
from agent_server.core import Core


class _Iter:
    def __init__(self, items):
        self._items, self._i = list(items), 0

    def More(self):
        return self._i < len(self._items)

    def Value(self):
        return self._items[self._i]

    def Next(self):
        self._i += 1


def _sobj(name):
    so = SimpleNamespace(name=name, attrs=["AttributeName"])
    so.GetAllAttributes = lambda: so.attrs
    return so


class FakeStudy:
    def __init__(self, components):
        self.tree = {c: [_sobj(n) for n in names] for c, names in components.items()}

    def FindComponent(self, name):
        return name if name in self.tree else None

    def NewChildIterator(self, comp):
        return _Iter(self.tree[comp])

    def NewBuilder(self):
        study = self

        class B:
            def RemoveObjectWithChildren(self, so):
                # SALOME empties the label but leaves it in the tree
                so.attrs = []
        return B()


def _ns(study):
    return {"salome": SimpleNamespace(myStudy=study), "geompy": object(),
            "smesh": object(), "find_object": lambda n: n}


def test_study_objects_lists_geom_and_smesh_children():
    study = FakeStudy({"GEOM": ["box_1", "fuse_1"], "SMESH": ["mesh_1"], "OTHER": ["x"]})
    assert [o.name for o in study_objects(_ns(study))] == ["box_1", "fuse_1", "mesh_1"]


def test_study_objects_skips_labels_emptied_by_an_earlier_removal():
    study = FakeStudy({"GEOM": ["box_1", "box_2"]})
    study.NewBuilder().RemoveObjectWithChildren(study.tree["GEOM"][0])
    assert [o.name for o in study_objects(_ns(study))] == ["box_2"]


def test_study_objects_with_no_components_is_empty():
    assert study_objects(_ns(FakeStudy({}))) == []


def test_clear_study_removes_them_all():
    study = FakeStudy({"GEOM": ["box_1", "fuse_1"], "SMESH": ["mesh_1"]})
    assert clear_study(_ns(study)) == 3
    assert study_objects(_ns(study)) == []


def test_headless_backend_reports_and_clears_leftovers():
    study = FakeStudy({"GEOM": ["box_1"]})
    b = HeadlessBackend(build_namespace=lambda live: _ns(study))
    b.begin_session("r1")
    assert b.leftovers() == 1
    assert b.clear_study() == {"ok": True, "removed": 1}
    assert b.leftovers() == 0


def test_hello_reports_leftovers():
    study = FakeStudy({"GEOM": ["box_1", "box_2"]})
    b = HeadlessBackend(build_namespace=lambda live: _ns(study))
    resp = Core(b).handle({"id": 0, "cmd": "hello", "run_id": "r1"})
    assert resp["leftovers"] == 2
    assert resp["name_prefix"] == "sa_r1_"


def test_core_clear_study_command_refreshes():
    refreshed = []
    backend = SimpleNamespace(poisoned=lambda: False,
                              clear_study=lambda: {"ok": True, "removed": 4},
                              refresh=refreshed.append,
                              run_tool=lambda fn, timeout: fn())
    r = Core(backend).handle({"id": 3, "cmd": "clear_study"})
    assert r == {"id": 3, "ok": True, "removed": 4}
    assert refreshed == ["GEOM"]
