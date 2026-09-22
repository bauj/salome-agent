from types import SimpleNamespace

# Use the alias to avoid rewriting test assertions (task 7 cleanup).
from executor import geom as geom_tools


def test_next_name_applies_prefix_and_increments():
    geom_tools.reset_counters()
    ex = SimpleNamespace(name_prefix="sa_deadbeef_")
    assert geom_tools._next_name(ex, "box") == "sa_deadbeef_box_1"
    assert geom_tools._next_name(ex, "box") == "sa_deadbeef_box_2"
    assert geom_tools._next_name(ex, "fillet") == "sa_deadbeef_fillet_1"


def test_empty_prefix_matches_legacy_names():
    geom_tools.reset_counters()
    ex = SimpleNamespace(name_prefix="")
    assert geom_tools._next_name(ex, "box") == "box_1"


def test_reset_counters_is_independent_per_session():
    ex = SimpleNamespace(name_prefix="")
    geom_tools.reset_counters()
    geom_tools._next_name(ex, "box")
    geom_tools.reset_counters()
    assert geom_tools._next_name(ex, "box") == "box_1"
