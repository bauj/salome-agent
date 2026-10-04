from pathlib import Path

import pytest

tomllib = pytest.importorskip("tomllib")

ROOT = Path(__file__).resolve().parents[2]


def test_every_package_is_listed_in_pyproject():
    listed = set(tomllib.loads((ROOT / "pyproject.toml").read_text())
                 ["tool"]["setuptools"]["packages"])
    found = {
        ".".join(init.parent.relative_to(ROOT).parts)
        for top in ("salome_agent", "agent_server", "executor")
        for init in (ROOT / top).rglob("__init__.py")
        if "__pycache__" not in init.parts
    }
    assert found - listed == set()
