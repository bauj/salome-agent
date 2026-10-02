import json

import pytest

from salome_agent.config import ConfigError, load_config


def test_loads_valid_config(tmp_path):
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps({
        "llm": {"base_url": "http://x", "api_key": "k", "model": "m", "temperature": 0},
        "salome": {"root": "/opt/salome"},
        "policy": {"static_retries": 3, "exec_retries": 2, "exec_timeout_s": 120,
                   "tool_iteration_cap": 12, "on_stage_failure": "abort"},
        "work_dir": "~/.local/share/salome-agent/runs",
    }))
    config = load_config(str(cfg_path))
    assert config["llm"]["model"] == "m"


def test_missing_required_key_raises(tmp_path):
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps({"llm": {"base_url": "http://x"}}))
    with pytest.raises(ConfigError):
        load_config(str(cfg_path))


def test_unknown_llm_key_warns(tmp_path, caplog):
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps({
        "llm": {"base_url": "http://x", "model": "m",
                "extra-body": {"reasoning_effort": "high"}},
        "salome": {"root": "/opt/salome"},
    }))
    with caplog.at_level("WARNING", logger="salome_agent"):
        load_config(str(cfg_path))
    assert "llm.extra-body" in caplog.text


def test_known_llm_keys_do_not_warn(tmp_path, caplog):
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps({
        "llm": {"base_url": "http://x", "model": "m", "extra_body": {},
                "extra_headers": {}, "json_retries": 2, "timeout": 60},
        "salome": {"root": "/opt/salome"},
    }))
    with caplog.at_level("WARNING", logger="salome_agent"):
        load_config(str(cfg_path))
    assert "unknown key" not in caplog.text


def test_salome_root_is_optional(tmp_path):
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps({"llm": {"base_url": "http://x", "model": "m"}}))
    assert load_config(str(cfg_path))["llm"]["model"] == "m"


def test_vision_fix_retries_is_a_known_llm_key():
    from salome_agent.config import LLM_KNOWN_KEYS
    assert "vision_fix_retries" in LLM_KNOWN_KEYS


def test_llm_modules_is_known_and_its_unknown_subkeys_warn(tmp_path, caplog):
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps({
        "llm": {"base_url": "http://x", "model": "m",
                "modules": {"vision": {"model": "big", "base-url": "http://y"}}},
    }))
    with caplog.at_level("WARNING", logger="salome_agent"):
        load_config(str(cfg_path))
    assert "unknown key llm.modules in config" not in caplog.text
    assert "llm.modules.vision.base-url" in caplog.text
    assert "llm.modules.vision.model" not in caplog.text
