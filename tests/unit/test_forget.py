# tests/unit/test_forget.py
from types import SimpleNamespace

from agent_server.backend import HeadlessBackend, display_names, forget_names, hide_names
from agent_server.core import Core


class FakeStudy:
    def __init__(self, names, shaper=()):
        self.objs = {n: SimpleNamespace(name=n, GetID=lambda n=n: f"0:1:{n}") for n in names}
        self.objs.update({n: SimpleNamespace(name=n, GetID=lambda n=n: f"0:2:{n}")
                          for n in shaper})
        self.comp = {**{n: "GEOM" for n in names}, **{n: "SHAPERSTUDY" for n in shaper}}
        self.removed = []

    def FindObjectByName(self, name, component):
        if name in self.objs and self.comp[name] == component:
            return [self.objs[name]]
        return []

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
                              refresh=refreshed.append,
                              run_tool=lambda fn, timeout: fn())
    r = Core(backend).handle({"id": 4, "cmd": "forget", "names": ["box_1"]})
    assert r == {"id": 4, "ok": True, "removed": ["box_1"]}
    assert refreshed == ["GEOM"]


def test_forget_erases_each_study_object_from_the_viewer_before_removing_it():
    study = FakeStudy(["box_1"])
    order = []
    orig = study.NewBuilder

    def builder():
        b = orig()
        remove = b.RemoveObjectWithChildren
        b.RemoveObjectWithChildren = lambda so: (order.append(("remove", so.name)), remove(so))
        return b

    study.NewBuilder = builder
    forget_names(_ns(study), {"box_1": 1, "vec_1": 2}, ["box_1", "vec_1"],
                 erase=lambda entry: order.append(("erase", entry)))
    assert order == [("erase", "0:1:box_1"), ("remove", "box_1")]


def test_a_failing_erase_does_not_stop_the_forget():
    study = FakeStudy(["box_1"])

    def boom(entry):
        raise RuntimeError("no viewer")

    r = forget_names(_ns(study), {}, ["box_1"], erase=boom)
    assert r["removed"] == ["box_1"] and "box_1" not in study.objs


def test_display_shows_each_found_study_object():
    study = FakeStudy(["fuse_1"])
    shown = []
    r = display_names(_ns(study), ["fuse_1", "nope_1"], shown.append)
    assert r == {"ok": True, "shown": ["fuse_1"]}
    assert shown == ["0:1:fuse_1"]


def test_headless_display_is_a_no_op():
    b = HeadlessBackend(build_namespace=lambda live: _ns(FakeStudy(["box_1"])))
    assert b.display(["box_1"]) == {"ok": True, "shown": []}


def test_core_display_command():
    backend = SimpleNamespace(poisoned=lambda: False, run_tool=lambda fn, timeout: fn(),
                              display=lambda names: {"ok": True, "shown": list(names)})
    assert Core(backend).handle({"id": 5, "cmd": "display", "names": ["a"]}) == {
        "id": 5, "ok": True, "shown": ["a"]}


def test_hide_erases_each_found_study_object_from_the_viewer_only():
    study = FakeStudy(["fuse_1"])
    erased = []
    r = hide_names(_ns(study), ["fuse_1", "nope_1"], erased.append)
    assert r == {"ok": True, "hidden": ["fuse_1"]}
    assert erased == ["0:1:fuse_1"] and "fuse_1" in study.objs


def test_headless_hide_is_a_no_op():
    b = HeadlessBackend(build_namespace=lambda live: _ns(FakeStudy(["box_1"])))
    assert b.hide(["box_1"]) == {"ok": True, "hidden": []}


def test_core_hide_command():
    backend = SimpleNamespace(poisoned=lambda: False, run_tool=lambda fn, timeout: fn(),
                              hide=lambda names: {"ok": True, "hidden": list(names)})
    assert Core(backend).handle({"id": 6, "cmd": "hide", "names": ["a"]}) == {
        "id": 6, "ok": True, "hidden": ["a"]}


def test_client_hide_sends_the_hide_command():
    from salome_agent.executor_client import ExecutorClient
    client = ExecutorClient.__new__(ExecutorClient)
    sent = []
    client._call = lambda cmd, _bound=None, **kw: sent.append((cmd, kw)) or {"ok": True}
    client.hide(["a"])
    assert sent == [("hide", {"names": ["a"]})]


def test_shaper_results_are_left_to_shaper_in_the_viewer_but_forgotten_from_the_study():
    # A presentation made from a SHAPERSTUDY entry has no eye in the GEOM or
    # SHAPER tree, so it stayed in the OCC view whatever the user hid.
    study = FakeStudy(["box_1"], shaper=["extrusion_1"])
    shown, erased = [], []
    assert display_names(_ns(study), ["extrusion_1"], shown.append)["shown"] == []
    assert hide_names(_ns(study), ["extrusion_1"], erased.append)["hidden"] == []
    assert shown == erased == []
    r = forget_names(_ns(study), {}, ["extrusion_1", "box_1"], erase=erased.append)
    assert r == {"ok": True, "removed": ["extrusion_1", "box_1"]}
    assert erased[0] == "0:2:extrusion_1"


def test_core_display_shows_shaper_results_through_shaper(monkeypatch):
    import agent_server.core as core
    seen = []
    pkg = SimpleNamespace(show=lambda b, names, shown: seen.append((names, shown)) or ["ext_1"])
    monkeypatch.setattr(core, "_side_modules", lambda: [pkg])
    backend = SimpleNamespace(poisoned=lambda: False, run_tool=lambda fn, timeout: fn(),
                              display=lambda names: {"ok": True, "shown": []},
                              hide=lambda names: {"ok": True, "hidden": []})
    assert Core(backend).handle({"id": 1, "cmd": "display", "names": ["ext_1"]})["shown"] == ["ext_1"]
    assert Core(backend).handle({"id": 2, "cmd": "hide", "names": ["ext_1"]})["hidden"] == ["ext_1"]
    assert seen == [(["ext_1"], True), (["ext_1"], False)]


def test_introspect_imports_shaper_model_on_the_gui_thread():
    calls = []
    backend = SimpleNamespace(poisoned=lambda: False, namespace={"geompy": object()},
                              run_tool=lambda fn, timeout: calls.append(1) or fn())
    import agent_server.core as core
    core._LAZY_APIS["model"] = "json"   # any importable module stands in for SHAPER's
    try:
        r = Core(backend).handle({"id": 1, "cmd": "introspect", "names": ["model", "nope"]})
    finally:
        core._LAZY_APIS["model"] = "salome.shaper.model"
    assert "dumps" in r["model"]["members"] and r["nope"]["members"] == []
    assert calls == [1]
