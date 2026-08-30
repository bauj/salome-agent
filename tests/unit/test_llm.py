import pytest

from salome_agent.llm import _parse_json_content


def test_plain_json():
    assert _parse_json_content('{"stages": []}') == {"stages": []}


def test_fenced_json():
    assert _parse_json_content('```json\n{"stages": [1]}\n```') == {"stages": [1]}


def test_json_with_prose_around_it():
    raw = 'Here is the plan:\n{"stages": [{"module": "geom"}]}\nHope that helps.'
    assert _parse_json_content(raw) == {"stages": [{"module": "geom"}]}


def test_unparseable_raises_with_raw_content():
    with pytest.raises(ValueError, match="parseable JSON"):
        _parse_json_content("sorry, I cannot do that")
