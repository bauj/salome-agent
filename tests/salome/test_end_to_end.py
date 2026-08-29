import ast
import json
from unittest.mock import MagicMock, patch

import pytest

SALOME_ROOT = "/storage/salome_platform/devel/SALOME-master-native-FD44"


@pytest.mark.salome
def test_cube_and_mesh_produces_runnable_result(tmp_path):
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

    fake_llm = MagicMock()
    fake_llm.chat.side_effect = [
        {"stages": [
            {"module": "geom", "intent": "20mm cube", "params": {"size": 20}, "inputs": []},
            {"module": "smesh", "intent": "mesh 5mm tets", "params": {"max_size": 5}, "inputs": ["box_1"]},
        ]},
        {"tool_call": "make_box", "args": {"dx": 20, "dy": 20, "dz": 20}},
        {"tool_call": "finish_stage", "args": {"result_names": ["box_1"]}},
        {"tool_call": "emit_script", "args": {"code": (
            # smesh.NETGEN_1D2D3D does not exist — the constant lives on
            # the smeshBuilder *module*, not the smeshBuilder.New() instance
            # bound to "smesh" (confirmed live against SALOME-master-native-FD44).
            "from salome.smesh import smeshBuilder\n"
            "shape = find_object('box_1')\n"
            "m = smesh.Mesh(shape)\n"
            "m.Tetrahedron(algo=smeshBuilder.NETGEN_1D2D3D).SetMaxSize(5)\n"
            "m.Compute()\n"
        )}},
    ]
    with patch("salome_agent.cli.build_llm", return_value=fake_llm):
        from salome_agent.cli import main
        rc = main(["make a 20mm cube and mesh it with 5mm tetrahedra", "--batch",
                   "--work-dir", str(tmp_path), "--config", str(config_path)])
    assert rc == 0
    result_py = next(tmp_path.glob("*/result.py"))
    source = result_py.read_text()
    ast.parse(source)
    assert "MakeBoxDXDYDZ" in source
    assert "Compute" in source
