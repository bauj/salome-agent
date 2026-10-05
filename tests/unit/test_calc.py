from salome_agent.calc import CALC_DEF, run_calc


def test_prints_and_last_expression():
    out = run_calc("x = 3\nprint('x is', x)\nx * 2")
    assert out == "x is 3\n6"


def test_math_available_bare_and_as_module():
    assert run_calc("round(sin(radians(30)), 6)") == "0.5"
    assert run_calc("math.hypot(3, 4)") == "5.0"


def test_loops_and_defs():
    code = ("def corner(t):\n    return (round(cos(t), 3), round(sin(t), 3))\n"
            "[corner(k * pi / 2) for k in range(3)]")
    assert run_calc(code) == "[(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0)]"


def test_last_expression_none_shows_nothing():
    assert run_calc("print(1)\nNone") == "1"


def test_import_refused():
    assert run_calc("import os").startswith("error: imports are not allowed")
    assert run_calc("from math import sin").startswith("error: imports are not allowed")


def test_dunder_refused():
    assert "error:" in run_calc("().__class__")
    assert "error:" in run_calc("__builtins__")


def test_no_open_or_eval():
    assert run_calc("open('/etc/passwd')").startswith("error: NameError")
    assert run_calc("eval('1')").startswith("error: NameError")


def test_runtime_error_names_line():
    assert run_calc("a = 1\nb = a / 0") == "error: ZeroDivisionError: division by zero (line 2)"


def test_syntax_error():
    assert run_calc("x = (1").startswith("error: SyntaxError")


def test_timeout():
    assert run_calc("while True:\n    pass", timeout=0.5).startswith("error: timed out")


def test_long_output_cut():
    out = run_calc("for i in range(5000):\n    print(i)")
    assert len(out) < 4200
    assert out.endswith("(output cut)")


def test_empty_snippet():
    assert run_calc("   ").startswith("error: empty")


def test_def_schema():
    assert CALC_DEF["name"] == "calc"
    assert CALC_DEF["parameters"]["required"] == ["code"]
    assert CALC_DEF["description"]
