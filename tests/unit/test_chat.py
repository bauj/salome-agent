import io
from types import SimpleNamespace

from salome_agent.chat import repl
from salome_agent.conversation import TurnResult


class FakeConv:
    def __init__(self):
        self.turns = []
        self.run_dir = "/w/r1"
        self.graph = SimpleNamespace(render=lambda: "n1  box_1 = make_box(dx=1)")
        self.ended = False
        self.undone = 0
        self.rebuilt = 0

    def undo(self):
        self.undone += 1
        return TurnResult("ok", "undone", {})

    def redo(self):
        self.redone = getattr(self, "redone", 0) + 1
        return TurnResult("ok", "redone", {})

    def rebuild(self):
        self.rebuilt += 1
        return TurnResult("ok", "rebuilt 1 node(s)", {})


def _run(lines, conv=None):
    conv = conv or FakeConv()
    feed = iter(lines)

    def read(prompt):
        try:
            return next(feed)
        except StopIteration:
            raise EOFError

    def run_turn(c, text, gate, stream):
        c.turns.append(text)
        return TurnResult("ok", f"did {text}", {})

    out = io.StringIO()
    code = repl(conv, "a house", read=read, stream=out, run_turn=run_turn)
    return conv, out.getvalue(), code


def test_first_request_then_bare_text_turns():
    conv, out, code = _run(["make the roof taller", "/quit"])
    assert conv.turns == ["a house", "make the roof taller"]
    assert code == 0 and "did make the roof taller" in out


def test_commands():
    conv, out, _ = _run(["/show", "/undo", "/rebuild", "/help", "/nope", ""])
    assert "n1  box_1 = make_box(dx=1)" in out
    assert conv.undone == 1 and conv.rebuilt == 1
    assert "/undo" in out                      # help text
    assert "unknown command /nope" in out
    assert conv.turns == ["a house"]           # blank line is not a turn


def test_eof_exits_cleanly():
    _, _, code = _run([])
    assert code == 0


def test_result_prints_path(tmp_path):
    conv = FakeConv()
    conv.run_dir = str(tmp_path)
    (tmp_path / "result.py").write_text("print('hi')\n")
    _, out, _ = _run(["/result"], conv)
    assert str(tmp_path / "result.py") in out and "print('hi')" in out


def test_ended_conversation_exits_1():
    conv = FakeConv()

    def run_turn(c, text, gate, stream):
        c.ended = True
        return TurnResult("ended", "server gone", {})

    code = repl(conv, "a house", read=lambda p: "/quit", stream=io.StringIO(),
                run_turn=run_turn)
    assert code == 1


def test_redo_command():
    conv, out, _ = _run(["/redo"])
    assert conv.redone == 1 and "/redo" in out
