import io

from salome_agent import interactive
from salome_agent.interactive import prompt_plan, prompt_retry, normalize_plan, terminal_gate


def test_edit_preserves_the_tools_hint(monkeypatch):
    monkeypatch.setenv("EDITOR", "true")  # no-op editor, file left unchanged
    plan = [{"module": "geom", "intent": "x", "params": {}, "inputs": [],
             "tools": ["make_box"]}]
    out = interactive._edit_plan_in_editor(plan)
    assert out[0]["tools"] == ["make_box"]


def test_edit_defaults_missing_tools_to_empty_list(monkeypatch):
    monkeypatch.setenv("EDITOR", "true")
    plan = [{"module": "geom", "intent": "x", "params": {}, "inputs": []}]
    out = interactive._edit_plan_in_editor(plan)
    assert out[0]["tools"] == []


def _reader(answers):
    it = iter(answers)
    return lambda _prompt="": next(it)


def test_prompt_plan_approve():
    d = prompt_plan({"plan": [{"module": "geom", "intent": "box"}]},
                    read=_reader(["y"]), stream=io.StringIO())
    assert d == {"action": "approve"}


def test_prompt_plan_empty_is_approve():
    d = prompt_plan({"plan": []}, read=_reader([""]), stream=io.StringIO())
    assert d == {"action": "approve"}


def test_prompt_plan_abort():
    d = prompt_plan({"plan": []}, read=_reader(["n"]), stream=io.StringIO())
    assert d == {"action": "abort"}


def test_prompt_plan_reasks_on_garbage():
    d = prompt_plan({"plan": []}, read=_reader(["huh", "wat", "y"]), stream=io.StringIO())
    assert d == {"action": "approve"}


def test_prompt_plan_edit_returns_edited_plan():
    new = [{"module": "geom", "intent": "box", "params": {}, "inputs": []}]
    d = prompt_plan({"plan": [{"module": "smesh"}]}, read=_reader(["e"]),
                    stream=io.StringIO(), edit_fn=lambda _p: new)
    assert d == {"action": "edit", "plan": new}


def test_prompt_plan_edit_cancelled_falls_back_to_reask():
    d = prompt_plan({"plan": []}, read=_reader(["e", "y"]), stream=io.StringIO(),
                    edit_fn=lambda _p: None)
    assert d == {"action": "approve"}


def test_prompt_retry_hint():
    d = prompt_retry({"last_error": {"message": "boom"}},
                     read=_reader(["h", "use MakeBoxDXDYDZ"]), stream=io.StringIO())
    assert d == {"action": "hint", "hint": "use MakeBoxDXDYDZ"}


def test_prompt_retry_skip():
    d = prompt_retry({"last_error": {"message": "boom"}},
                     read=_reader(["s"]), stream=io.StringIO())
    assert d == {"action": "skip"}


def test_prompt_retry_abort():
    d = prompt_retry({"last_error": {"message": "boom"}},
                     read=_reader(["a"]), stream=io.StringIO())
    assert d == {"action": "abort"}


def test_prompt_retry_eof_aborts():
    def _raise(_p=""):
        raise EOFError

    d = prompt_retry({"last_error": {"message": "boom"}}, read=_raise, stream=io.StringIO())
    assert d == {"action": "abort"}


def test_normalize_plan_fills_defaults():
    out = normalize_plan([{"module": "geom"}])
    assert out == [{"module": "geom", "intent": "", "params": {}, "inputs": [], "tools": []}]


def test_normalize_plan_rejects_bad_shapes():
    assert normalize_plan([]) is None
    assert normalize_plan({"module": "geom"}) is None
    assert normalize_plan([{"module": "cad"}]) is None
    assert normalize_plan(["geom"]) is None


def test_terminal_gate_routes_plan_payload():
    d = terminal_gate({"plan": []}, read=_reader(["y"]), stream=io.StringIO())
    assert d == {"action": "approve"}


def test_terminal_gate_routes_retry_payload():
    d = terminal_gate({"last_error": {"message": "boom"}}, read=_reader(["s"]),
                      stream=io.StringIO())
    assert d == {"action": "skip"}


def test_terminal_gate_eof_aborts():
    def _raise(_p=""):
        raise EOFError

    assert terminal_gate({"plan": []}, read=_raise, stream=io.StringIO()) == {"action": "abort"}


def test_terminal_gate_unknown_payload_raises():
    import pytest

    with pytest.raises(RuntimeError):
        terminal_gate({"other": 1}, read=_reader([]), stream=io.StringIO())


