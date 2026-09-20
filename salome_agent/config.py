import json

from salome_agent.logs import log

REQUIRED = [("llm", "base_url"), ("llm", "model"), ("salome", "root")]

# Keys the code actually reads out of the `llm` block (see cli.build_llm /
# llm.LLMClient). An unknown key here is almost always a typo -- a hyphen for
# an underscore, say -- and it fails silently: the knob just never takes
# effect. Warn loudly instead.
LLM_KNOWN_KEYS = {
    "base_url", "api_key", "ssl_cert_file", "model", "temperature", "timeout",
    "json_retries", "max_retries", "extra_headers", "extra_body", "module_extra_body",
}


class ConfigError(Exception):
    pass


def load_config(path: str) -> dict:
    with open(path) as f:
        config = json.load(f)
    for section, key in REQUIRED:
        if key not in config.get(section, {}):
            raise ConfigError(f"missing required config key: {section}.{key}")
    unknown = sorted(set(config.get("llm", {})) - LLM_KNOWN_KEYS)
    for key in unknown:
        log.warning("unknown key llm.%s in config -- ignored (typo? "
                    "known keys: %s)", key, ", ".join(sorted(LLM_KNOWN_KEYS)))
    return config
