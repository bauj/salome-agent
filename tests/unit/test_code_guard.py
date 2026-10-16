import math
import os
import types

import pytest

from executor.code_guard import check, refusal

_model = types.ModuleType("salome.shaper.model")
_model.os = os
_model.addBox = lambda *a: None
NS = {"model": _model, "shaper_doc": object(), "box_1": object(), "math": math,
      "_name": lambda *a: None}


@pytest.mark.parametrize("code", [
    "box_2 = model.addBox(shaper_doc, 10, 20, 30)\n_name(box_2, 'box_2')",
    "for _r in box_1.results():\n    _r.setColor(255, 0, 0)",
    "pts = [p for p in range(3)]\nn = len(pts)",
    "f = lambda a: a + 1\nf(2)",
])
def test_generated_code_passes(code):
    assert refusal(code, NS) is None


@pytest.mark.parametrize("code, why", [
    ("import os", "import"),
    ("from os import system", "import"),
    ("box_1.__class__", "__class__"),
    ("open('/tmp/x', 'w')", "'open'"),
    ("exec('1')", "'exec'"),
    ("eval('1')", "'eval'"),
    ("getattr(box_1, 'x')", "'getattr'"),
    ("math.pi", "module 'math'"),
    ("model.os.system('touch /tmp/x')", "module 'os'"),
    ("box_1)\nopen('/tmp/x')#", "does not parse"),
    ("box_1\nopen('/tmp/x')", "'open'"),
    ("_name(box_1, '')\n__import__('os').system('id')", "__import__"),
])
def test_break_outs_are_refused(code, why):
    r = refusal(code, NS)
    assert r is not None and why in r


def test_check_raises_and_eval_mode():
    with pytest.raises(ValueError):
        check("open('x')", NS)
    assert refusal("box_1", NS, "eval") is None
    assert refusal("box_1; open('x')", NS, "eval") is not None
