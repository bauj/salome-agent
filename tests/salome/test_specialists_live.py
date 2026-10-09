"""Live-LLM smoke test for the smesh specialist (task 177).

Every other specialist test uses a MagicMock LLM, which cannot catch
message-shape / prompt-wiring / schema bugs — those only surface against a
real endpoint (see the geom-specialist bugs 19-23). This drives the real
smesh path against the configured LLM and a live SALOME session.

Spends tokens: deselected by default; run with `pytest -m llm`. Also skipped
without a real ./config.json.
"""
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "config.json"

pytestmark = [
    pytest.mark.salome,
    pytest.mark.llm,
    pytest.mark.skipif(not CONFIG_PATH.exists(),
                       reason="no config.json — needs a real LLM endpoint"),
]


@pytest.fixture(scope="module")
def config() -> dict:
    return json.loads(CONFIG_PATH.read_text())


def test_geom_to_smesh_multistage_live(config, tmp_path, running_server):
    """Full graph, real LLM: geom stage publishes a box, the smesh stage must
    mesh it by the ACTUAL published name (not the planner's guess). Exercises
    the cross-stage handoff, the SMESH tool loop and the assembled result.py."""
    import ast

    from salome_agent.cli import main

    cfg = dict(config)
    cfg["work_dir"] = str(tmp_path)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(cfg))

    rc = main(["make a 20mm cube and mesh it with 5mm tetrahedra", "--batch",
               "--work-dir", str(tmp_path), "--config", str(config_path)])
    assert rc == 0

    report = next(tmp_path.glob("*/report.md")).read_text()
    assert "(geom): ok" in report, report
    assert "(smesh): ok" in report, report

    src = next(tmp_path.glob("*/result.py")).read_text()
    ast.parse(src)
    assert "MakeBoxDXDYDZ" in src
    assert "find_object(" in src
    assert "Compute()" in src
