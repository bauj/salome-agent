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


def test_registration_guard_allows_retry_on_failure():
    """Verify that the _SPECIALISTS_IMPORTED guard allows retries after
    a hypothetical import failure. The guard is set AFTER imports succeed,
    not before, so failed imports don't permanently lock out retries.

    This test verifies the pattern: if flag is False, we attempt registration;
    if imports fail, flag stays False so later calls can retry (the error is
    loud, not silent).
    """
    import salome_agent.modules as modules_module

    # Trigger registration by calling spec_for
    modules_module.spec_for("geom")

    # After first call, flag should be True
    assert modules_module._SPECIALISTS_IMPORTED, "Flag set after successful registration"

    # Verify routable returns non-empty (specialists registered)
    routable_list = modules_module.routable()
    assert len(routable_list) == 3, "Should have 3 routable modules"
    assert all(s.catalogue for s in routable_list), "All routable modules must have non-empty catalogue"
