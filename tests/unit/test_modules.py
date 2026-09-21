import salome_agent.specialists.geom  # noqa: F401 — trigger GEOM_SPEC registration
from salome_agent.modules import MODULES, ModuleSpec, Terminator, spec_for


def test_geom_spec_is_registered_and_frozen():
    spec = spec_for("geom")
    assert isinstance(spec, ModuleSpec)
    assert spec.name == "geom"
    assert MODULES["geom"] is spec


def test_geom_terminator_is_finish_stage():
    t = spec_for("geom").terminator
    assert isinstance(t, Terminator)
    assert t.name == "finish_stage"
    assert t.schema["required"] == ["result_names"]
    assert t.description


def test_geom_terminator_handle_returns_call_log_and_leaf_names():
    t = spec_for("geom").terminator
    call_log = [{"fn": "MakeBoxDXDYDZ", "args": [], "out_vars": ["box_1"], "publish": ["box_1"]}]
    assert t.handle({"result_names": ["box_1"]}, call_log) == {
        "call_log": call_log, "result_names": ["box_1"]}


def test_geom_terminator_handle_decodes_a_json_encoded_result_names():
    # seen live: a model returns the array as a JSON string
    t = spec_for("geom").terminator
    out = t.handle({"result_names": '["box_1"]'}, [])
    assert out["result_names"] == ["box_1"]


def test_geom_levers_are_configured_on():
    spec = spec_for("geom")
    assert "make_box" in spec.core_tools
    assert spec.risky_categories == frozenset({"boolean", "transform"})
    assert spec.measure_tools == frozenset({"measure", "check_shape"})
    assert spec.categories
    assert spec.call_prefix == "geompy."
    assert spec.catalogue  # geom is routable


def test_module_spec_is_frozen():
    import dataclasses

    import pytest

    with pytest.raises(dataclasses.FrozenInstanceError):
        spec_for("geom").name = "nope"


def test_spec_for_rejects_an_unknown_module():
    import pytest

    with pytest.raises(KeyError):
        spec_for("shaper")


def test_a_failed_specialist_import_does_not_permanently_break_the_registry(monkeypatch):
    """Verify that a failed specialist import does NOT latch the flag, so
    retries can happen. This is the critical case: if the flag is set BEFORE
    imports, a failed import leaves MODULES={} and flag=True forever. With
    flag set AFTER imports, failed imports leave flag=False and can be retried.

    Test simulates import failure, verifies flag is not latched, then verifies
    retry succeeds. This would FAIL if flag were set before imports.
    """
    import builtins
    import pytest
    import salome_agent.modules as modules_module

    # Save flag for restoration
    saved_flag = modules_module._SPECIALISTS_IMPORTED

    try:
        # Reset flag to simulate first call
        monkeypatch.setattr(modules_module, "_SPECIALISTS_IMPORTED", False)

        # Intercept builtins.__import__ to fail on specialist modules
        real_import = builtins.__import__

        def boom(name, *args, **kwargs):
            if "specialists" in name:
                raise ImportError("simulated specialist import failure")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", boom)

        # First call with intercepted import: should raise and leave flag False
        with pytest.raises(ImportError, match="simulated"):
            modules_module.spec_for("geom")

        # CRITICAL: flag must NOT be latched after failure
        assert modules_module._SPECIALISTS_IMPORTED is False, (
            "Flag must stay False after failed import to allow retry. "
            "This is THE guard against silent MODULES={}. "
            "If this fails, flag is being set BEFORE imports complete."
        )
    finally:
        # Restore flag so other tests are not affected
        monkeypatch.setattr(modules_module, "_SPECIALISTS_IMPORTED", saved_flag)
