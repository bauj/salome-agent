from types import SimpleNamespace

import pytest

import executor.shaper.state as state_mod
from executor.shaper.sketch import exec_records


class FakeModel:
    def __init__(self):
        self.begins = self.ends = 0
        self.flushes = []

    def begin(self):
        self.begins += 1

    def end(self):
        self.ends += 1

    def do(self):
        pass


class FakeResult:
    def __init__(self, log):
        self.log = log

    def setDisplayed(self, shown):
        self.log.append(("displayed", shown))


class FakeFeature:
    def __init__(self, name, log=None, subs=()):
        self._name = name
        self.log = log if log is not None else []
        self.subs = list(subs)

    def name(self):
        return self._name

    def results(self):
        return [FakeResult(self.log)]


class FakeDoc:
    def __init__(self, fail_remove=False):
        self.features = []
        self.fail_remove = fail_remove

    def allFeatures(self):
        return list(self.features)

    def removeFeature(self, feat):
        feat.log.append(("removed", feat.name()))
        if self.fail_remove:
            raise RuntimeError("remove failed")
        self.features.remove(feat)


@pytest.fixture
def fake_model(monkeypatch):
    m = FakeModel()
    monkeypatch.setattr(state_mod, "model", lambda: m)
    monkeypatch.setattr(state_mod, "_subs", lambda feat: feat.subs)
    monkeypatch.setattr(state_mod, "_flush_redisplay", lambda: m.flushes.append(1))
    return m


def _executor(doc):
    st = state_mod.ShaperState({"shaper_doc": doc})
    return SimpleNamespace(shaper_state=st, geom_bindings={}, name_prefix="",
                           namespace={})


def test_clear_never_touches_shaper_on_a_geom_only_server(monkeypatch):
    monkeypatch.setattr(state_mod, "_USED", False)

    def boom():
        raise AssertionError("SHAPER loaded")
    monkeypatch.setattr(state_mod, "model", boom)
    state_mod.clear(SimpleNamespace(geom_bindings={}))


def test_transaction_is_closed_when_its_body_raises(fake_model):
    with pytest.raises(ValueError):
        with state_mod.transaction():
            raise ValueError("x")
    assert fake_model.begins == fake_model.ends == 1


def test_forget_closes_its_transaction_when_removal_fails(fake_model):
    doc = FakeDoc(fail_remove=True)
    ex = _executor(doc)
    feat = FakeFeature("f")
    ex.shaper_state.objects["ext_1"] = "solid"
    ex.shaper_state.ns["ext_1"] = SimpleNamespace(feature=lambda: feat)
    with pytest.raises(RuntimeError):
        state_mod.forget(ex, ["ext_1"])
    assert fake_model.begins == fake_model.ends


def test_a_failed_batch_removes_what_it_already_created(fake_model):
    doc = FakeDoc()
    doc.features.append(FakeFeature("old"))
    ex = _executor(doc)
    def add(n):
        feat = FakeFeature(n)
        doc.features.append(feat)
        return SimpleNamespace(feature=lambda: feat)
    ex.shaper_state.ns["add"] = add
    records = [{"code": "a = add('a')", "out_vars": ["a"]},
               {"code": "b = add('b')", "out_vars": ["b"]},
               {"code": "c = add('c')\n1 / 0", "out_vars": ["c"]}]
    with pytest.raises(ZeroDivisionError):
        exec_records(ex.shaper_state, records)
    assert [f.name() for f in doc.features] == ["old"]
    assert fake_model.begins == fake_model.ends


def test_shape_status_reads_a_shaper_measure():
    ok = {"valid": True, "faces": 3, "edges": 3, "area": 10.0}
    assert state_mod.shape_status(ok) == {"status": "ok"}
    assert state_mod.shape_status({**ok, "valid": False, "check": "bad"})["status"] == "invalid"
    assert state_mod.shape_status({**ok, "faces": 0, "edges": 0})["status"] == "empty"
    assert state_mod.shape_status({**ok, "area": 0.0})["detail"] == "its faces have zero area"


def _with_features(doc, ex, *names):
    for n in names:
        feat = FakeFeature(n)
        doc.features.append(feat)
        ex.shaper_state.ns[n] = SimpleNamespace(feature=lambda feat=feat: feat)


def test_forget_leaves_an_object_whose_name_starts_with_the_forgotten_one(fake_model,
                                                                          monkeypatch):
    monkeypatch.setattr(state_mod, "_remove_study_objects", lambda ex, names: None)
    doc = FakeDoc()
    ex = _executor(doc)
    _with_features(doc, ex, "base", "base_origin", "base_f", "base_f_origin")
    ex.shaper_state.objects.update({"base": "group", "base_f": "group"})
    assert state_mod.forget(ex, ["base"]) == ["base"]
    assert [f.name() for f in doc.features] == ["base_f", "base_f_origin"]
    assert {"base_f", "base_f_origin"} <= set(ex.shaper_state.ns)
    assert state_mod.forget(ex, ["base_f"]) == ["base_f"]
    assert doc.features == []


def test_forget_takes_both_when_the_longer_name_comes_second(fake_model, monkeypatch):
    monkeypatch.setattr(state_mod, "_remove_study_objects", lambda ex, names: None)
    doc = FakeDoc()
    ex = _executor(doc)
    _with_features(doc, ex, "base", "base_f")
    ex.shaper_state.objects.update({"base": "group", "base_f": "group"})
    assert state_mod.forget(ex, ["base", "base_f"]) == ["base", "base_f"]
    assert doc.features == []


def test_forget_hides_the_results_and_sketch_entities_before_removing(fake_model, monkeypatch):
    # With the SHAPER module active, a feature removed while still displayed
    # stayed drawn in the viewer.
    monkeypatch.setattr(state_mod, "_remove_study_objects", lambda ex, names: None)
    log = []
    monkeypatch.setattr(state_mod, "_flush_redisplay", lambda: log.append(("flush",)))
    doc = FakeDoc()
    ex = _executor(doc)
    entity = FakeFeature("SketchCircle_1", log)
    sketch = FakeFeature("sketch_1", log, subs=[entity])
    doc.features.append(sketch)
    ex.shaper_state.sketches["sketch_1"] = SimpleNamespace(
        api=SimpleNamespace(feature=lambda: sketch))
    assert state_mod.forget(ex, ["sketch_1"]) == ["sketch_1"]
    assert log == [("displayed", False), ("displayed", False), ("flush",),
                   ("removed", "sketch_1")]


def test_show_sets_the_displayed_flag_of_known_results_and_flushes(fake_model):
    log = []
    doc = FakeDoc()
    ex = _executor(doc)
    feat = FakeFeature("ext", log)
    ex.shaper_state.objects["ext_1"] = "solid"
    ex.shaper_state.ns["ext_1"] = SimpleNamespace(feature=lambda: feat)
    assert state_mod.show(ex, ["ext_1", "box_9"], True) == ["ext_1"]
    assert log == [("displayed", True)]
    assert fake_model.flushes == [1] and fake_model.begins == 0


@pytest.mark.parametrize("id_", ["L1", "L1_1", "M_outline", "K3"])
def test_sketch_ids_may_use_underscores(id_):
    state_mod.check_id(id_, "entity")


@pytest.mark.parametrize("id_, text", [("1L", "must start with a letter"),
                                       ("L-1", "must start with a letter"),
                                       ("P_op", "reserved for sketch_copy")])
def test_sketch_ids_that_could_clash_are_refused(id_, text):
    with pytest.raises(ValueError, match=text):
        state_mod.check_id(id_, "entity")
