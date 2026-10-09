from salome_agent.modules import spec_for


def test_smesh_spec_is_a_tool_module():
    spec = spec_for("smesh")
    assert spec.call_prefix == "smesh."
    assert spec.terminator.name == "finish_stage"
    assert spec.checks_results and spec.forgets and spec.calc and spec.needs_session
    assert spec.publish_line == ""


def test_prompt_wording_fills_the_limit_and_the_catalogue():
    text = spec_for("smesh").prompt("final")
    assert "compute refuses a mesh predicted above\n1 000 000 elements." in text
    assert "  mesh_tetra -- " in text
    assert "You mesh geometry in SMESH by calling tools." in text
    assert "{max_elements}" not in text and "{catalogue}" not in text


def test_core_tools_are_registered_tools():
    from executor.smesh import registry
    from salome_agent.specialists.smesh import CORE_TOOLS, PLANNER_TOOL_LINES
    assert CORE_TOOLS <= set(registry.names)
    assert set(PLANNER_TOOL_LINES) == set(registry.names)


def test_categories_cover_the_registry():
    from executor.smesh import registry
    assert set(registry.categories().values()) <= set(spec_for("smesh").categories)


def test_upstream_lists_a_consumed_shape_without_failing_the_rest():
    from salome_agent.specialists.smesh import _upstream

    class Client:
        def call_tool(self, name, args, module=None):
            return {"ok": True, "shapes": {
                "box_1": {"error": "no SHAPER result 'box_1' in the study"},
                "cut_1": {"type": "solid", "volume": 10.0, "area": 5.0, "faces": 7}}}

    text = _upstream(Client(), ["box_1", "cut_1"])
    assert "  box_1 — no longer in the study (used up by a later step)" in text
    assert "  cut_1 — solid, volume 10.0, area 5.0, 7 faces" in text
