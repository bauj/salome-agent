import pytest

from executor.shaper.groups import group_sub_type, check_group_name


def test_group_sub_type_from_ids():
    assert group_sub_type(["F1", "F3"]) == "face"
    assert group_sub_type(["E2"]) == "edge"


@pytest.mark.parametrize("ids", [[], ["F1", "E2"], ["X1"], "F1", [1, 2]])
def test_group_sub_type_rejects_bad_ids(ids):
    with pytest.raises(ValueError):
        group_sub_type(ids)


def test_check_group_name():
    check_group_name("inlet", taken=set())
    with pytest.raises(ValueError, match="taken"):
        check_group_name("inlet", taken={"inlet"})
    with pytest.raises(ValueError, match="letter"):
        check_group_name("2nd face", taken=set())


@pytest.mark.parametrize("name", ["geompy", "smesh", "SMESH", "model", "shaper_doc",
                                  "salome", "class", "lambda", "list", "print"])
def test_check_group_name_refuses_reserved_names(name):
    with pytest.raises(ValueError, match="reserved"):
        check_group_name(name, taken=set())


def _module_level_names(source: str) -> set:
    import ast
    names = set()

    def walk(body):
        for node in body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names.update((a.asname or a.name).split(".")[0] for a in node.names)
            elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                names.add(node.name)
            elif isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign, ast.For)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for t in targets:
                    names.update(n.id for n in ast.walk(t) if isinstance(n, ast.Name))
            for part in ("body", "orelse", "finalbody"):
                if isinstance(node, (ast.For, ast.Try, ast.If, ast.With)):
                    walk(getattr(node, part, None) or [])
            for h in getattr(node, "handlers", []):
                walk(h.body)
    walk(ast.parse(source).body)
    return names


def test_every_name_result_py_binds_is_reserved():
    from executor.shaper.groups import RESERVED
    from salome_agent.modules import spec_for
    from salome_agent.nodes import assembler
    sources = [v for k, v in vars(assembler).items() if k.endswith("_INIT")]
    for m in ("geom", "shaper", "smesh"):
        sources += [spec_for(m).preamble, spec_for(m).postamble]
    bound = set().union(*(_module_level_names(s) for s in sources if s))
    assert {n for n in bound if not n.startswith("_")} <= RESERVED
