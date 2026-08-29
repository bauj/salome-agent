import json

import pytest

from salome_agent.config import ConfigError, load_config


def test_loads_valid_config(tmp_path):
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps({
        "llm": {"base_url": "http://x", "api_key": "k", "model": "m", "temperature": 0},
        "salome": {"root": "/opt/salome"},
        "policy": {"static_retries": 3, "exec_retries": 2, "exec_timeout_s": 120,
                   "tool_iteration_cap": 12, "on_stage_failure": "abort",
                   "assembler_smoke_run": True},
        "work_dir": "~/.local/share/salome-agent/runs",
    }))
    config = load_config(str(cfg_path))
    assert config["llm"]["model"] == "m"


def test_missing_required_key_raises(tmp_path):
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps({"llm": {"base_url": "http://x"}}))
    with pytest.raises(ConfigError):
        load_config(str(cfg_path))
