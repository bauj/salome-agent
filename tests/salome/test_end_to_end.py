import ast
import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from ._salome_root import SALOME_ROOT

pytestmark = [
    pytest.mark.salome,
    pytest.mark.skipif(SALOME_ROOT is None,
                       reason="set SALOME_ROOT or salome.root in config.json"),
]


def test_cube_and_mesh_produces_runnable_result(tmp_path, running_server):
    # No config.json is checked into the repo (config.example.json is the
    # template — real configs carry an api_key). build_llm() is patched out
    # below, so the llm.* values here only need to satisfy load_config's
    # required-key check, not point at a real endpoint.
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({
        "llm": {"base_url": "http://unused", "api_key": "", "model": "unused", "temperature": 0},
        "salome": {"root": SALOME_ROOT},
        "policy": {"static_retries": 3, "exec_retries": 2, "exec_timeout_s": 120,
                   "tool_iteration_cap": 12, "on_stage_failure": "abort",
                   "assembler_smoke_run": True},
        "work_dir": str(tmp_path),
    }))

    # Headless names are per-run (sa_<run_id>_); pin the run id so the
    # scripted model can name its box.
    box = "sa_e2e000000000_box_1"
    fake_llm = MagicMock()
    fake_llm.usage = None  # real LLMClient exposes a usage dict; report.py reads it
    fake_llm.chat.side_effect = [
        {"stages": [
            {"module": "geom", "intent": "20mm cube", "params": {"size": 20}, "inputs": []},
            {"module": "smesh", "intent": "mesh 5mm tets", "params": {"max_size": 5}, "inputs": [box]},
        ]},
        {"tool_calls": [{"name": "make_box", "args": {"dx": 20, "dy": 20, "dz": 20}}]},
        {"tool_calls": [{"name": "finish_stage", "args": {"result_names": [box]}}]},
        {"tool_calls": [{"name": "emit_script", "args": {"code": (
            # smesh.NETGEN_1D2D3D does not exist — the constant lives on
            # the smeshBuilder *module*, not the smeshBuilder.New() instance
            # bound to "smesh" (confirmed live).
            "from salome.smesh import smeshBuilder\n"
            f"shape = find_object('{box}')\n"
            "m = smesh.Mesh(shape)\n"
            "m.Tetrahedron(algo=smeshBuilder.NETGEN_1D2D3D).SetMaxSize(5)\n"
            "m.Compute()\n"
        )}}]},
    ]
    with patch("salome_agent.cli.build_llm", return_value=fake_llm), \
            patch("uuid.uuid4", return_value=SimpleNamespace(hex="e2e000000000" * 3)):
        from salome_agent.cli import main
        rc = main(["make a 20mm cube and mesh it with 5mm tetrahedra", "--batch",
                   "--work-dir", str(tmp_path), "--config", str(config_path)])
    assert rc == 0
    result_py = next(tmp_path.glob("*/result.py"))
    source = result_py.read_text()
    ast.parse(source)
    assert "MakeBoxDXDYDZ" in source
    assert "Compute" in source
