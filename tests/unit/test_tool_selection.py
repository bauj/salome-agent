from salome_agent.specialists.tool_selection import (
    CORE_TOOLS, hint_tool_names, initial_offered, resolve_load,
)

ALL = {"make_box", "make_cylinder", "translate", "set_color", "fuse", "cut",
       "measure", "check_shape", "list_study_objects", "make_thick_solid"}
DEFS = {
    "make_box": {"name": "make_box", "category": "primitive"},
    "make_cylinder": {"name": "make_cylinder", "category": "primitive"},
    "translate": {"name": "translate", "category": "transform"},
    "set_color": {"name": "set_color", "category": "display"},
    "make_thick_solid": {"name": "make_thick_solid", "category": "swept"},
}


def test_hint_tool_names_reads_the_tools_key():
    assert hint_tool_names({"tools": ["make_box", "translate"]}) == ["make_box", "translate"]


def test_hint_tool_names_also_reads_the_operations_key_and_dict_items():
    assert hint_tool_names({"operations": [{"tool": "make_box"}, {"tool": "cut"}]}) == ["make_box", "cut"]


def test_hint_tool_names_ignores_malformed_entries_and_missing_keys():
    assert hint_tool_names({}) == []
    assert hint_tool_names({"tools": [1, None, {"nope": "x"}, "make_box"]}) == ["make_box"]


def test_initial_offered_is_core_union_valid_hints():
    got = initial_offered({"tools": ["make_cylinder", "not_a_tool"]}, ALL)
    assert got == set(CORE_TOOLS) | {"make_cylinder"}
    assert "not_a_tool" not in got


def test_initial_offered_with_no_hint_is_exactly_core():
    assert initial_offered({}, ALL) == set(CORE_TOOLS)


def test_resolve_load_expands_names_categories_and_both():
    assert resolve_load(["make_box"], None, DEFS) == {"make_box"}
    assert resolve_load(None, ["primitive"], DEFS) == {"make_box", "make_cylinder"}
    assert resolve_load(["translate"], ["swept"], DEFS) == {"translate", "make_thick_solid"}


def test_resolve_load_ignores_unknown_names_and_categories():
    assert resolve_load(["ghost"], ["nonsense"], DEFS) == set()
    assert resolve_load(None, None, DEFS) == set()
