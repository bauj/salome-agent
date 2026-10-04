from types import SimpleNamespace

import pytest

import executor.shaper.state as state_mod
from executor.shaper.sketch import exec_records


class FakeModel:
    def __init__(self):
        self.begins = self.ends = 0

    def begin(self):
        self.begins += 1

    def end(self):
        self.ends += 1

    def do(self):
        pass


class FakeFeature:
    def __init__(self, name):
        self._name = name

    def name(self):
        return self._name


class FakeDoc:
    def __init__(self, fail_remove=False):
        self.features = []
        self.fail_remove = fail_remove

    def allFeatures(self):
        return list(self.features)

    def removeFeature(self, feat):
        if self.fail_remove:
            raise RuntimeError("remove failed")
        self.features.remove(feat)


@pytest.fixture
def fake_model(monkeypatch):
    m = FakeModel()
    monkeypatch.setattr(state_mod, "model", lambda: m)
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
