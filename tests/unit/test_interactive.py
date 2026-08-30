import io

from salome_agent.interactive import prompt_plan, prompt_retry, resume_interactive


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


class _FakeInterrupt:
    def __init__(self, value):
        self.value = value


class _FakeGraph:
    def __init__(self, results):
        self._results = list(results)
        self.resumes = []

    def invoke(self, command, config=None):
        self.resumes.append(command)
        return self._results.pop(0)


def test_resume_interactive_plan_then_retry_then_done():
    first = {"__interrupt__": (_FakeInterrupt({"plan": []}),)}
    graph = _FakeGraph([
        {"__interrupt__": (_FakeInterrupt({"last_error": {"message": "boom"}}),)},
        {"report": "ok"},
    ])
    final = resume_interactive(graph, first, {"configurable": {}},
                              read=_reader(["y", "s"]), stream=io.StringIO())
    assert final == {"report": "ok"}
    assert len(graph.resumes) == 2


def test_resume_interactive_no_interrupt_is_passthrough():
    graph = _FakeGraph([])
    final = resume_interactive(graph, {"report": "done"}, {}, read=_reader([]), stream=io.StringIO())
    assert final == {"report": "done"}
