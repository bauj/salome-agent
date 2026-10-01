import pytest

from executor.registry import ToolRegistry

_CATEGORIES = {"primitive", "sketch", "swept", "boolean", "transform",
               "feature", "display", "query"}


def _reg():
    r = ToolRegistry(valid_categories=_CATEGORIES)

    @r.tool("make_thing", category="primitive", description="Make a thing.",
            schema={"type": "object", "properties": {"n": {"type": "number"}},
                    "required": ["n"]},
            arg_types={"n": float})
    def tool_make_thing(executor, args):
        return {"ok": True, "n": args["n"]}

    return r


def test_registration_fills_every_projection():
    r = _reg()
    assert list(r.handlers()) == ["make_thing"]
    assert r.categories() == {"make_thing": "primitive"}
    assert r.arg_types() == {"make_thing": {"n": float}}
    assert r.schemas()["make_thing"]["required"] == ["n"]
    assert r.descriptions()["make_thing"] == "Make a thing."


def test_the_decorator_returns_the_function_unchanged():
    r = ToolRegistry(valid_categories=_CATEGORIES)

    def handler(executor, args):
        return {"ok": True}

    assert r.tool("t", category="query", description="d", schema={})(handler) is handler


def test_list_tools_entry_shape():
    assert _reg().list_tools() == [{
        "name": "make_thing",
        "parameters": {"type": "object", "properties": {"n": {"type": "number"}},
                       "required": ["n"]},
        "description": "Make a thing.",
        "category": "primitive"}]


def test_projections_preserve_registration_order():
    r = ToolRegistry(valid_categories=_CATEGORIES)
    for name in ("z", "a", "m"):
        r.tool(name, category="query", description="d", schema={})(lambda e, a: {"ok": True})
    assert list(r.handlers()) == ["z", "a", "m"]
    assert [t["name"] for t in r.list_tools()] == ["z", "a", "m"]


def test_coerce_args_applies_the_declared_types():
    assert _reg().coerce_args("make_thing", {"n": "3"}) == {"n": 3.0}


def test_coerce_args_leaves_an_unknown_tool_alone():
    assert _reg().coerce_args("nope", {"n": "3"}) == {"n": "3"}


def test_an_unknown_category_is_rejected_at_registration():
    r = ToolRegistry(valid_categories=_CATEGORIES)
    with pytest.raises(ValueError, match="category"):
        r.tool("t", category="bogus", description="d", schema={})(lambda e, a: None)


def test_a_missing_description_is_rejected_at_registration():
    r = ToolRegistry(valid_categories=_CATEGORIES)
    with pytest.raises(ValueError, match="description"):
        r.tool("t", category="query", description="", schema={})(lambda e, a: None)


def test_a_duplicate_name_is_rejected():
    r = _reg()
    with pytest.raises(ValueError, match="already registered"):
        r.tool("make_thing", category="query", description="d", schema={})(lambda e, a: None)


def test_coerce_args_raises_on_a_failed_cast():
    """A cast that fails raises ValueError, not swallowed."""
    r = _reg()
    with pytest.raises(ValueError):
        r.coerce_args("make_thing", {"n": "not_a_number"})


def test_coerce_args_raises_on_a_none_value():
    """A None value raises TypeError when casting, not skipped."""
    r = _reg()
    with pytest.raises(TypeError):
        r.coerce_args("make_thing", {"n": None})


def test_coerce_args_leaves_missing_key_absent():
    """A key not in args is not added to the result."""
    r = _reg()
    result = r.coerce_args("make_thing", {})
    assert "n" not in result
    assert result == {}
