import json

REQUIRED = [("llm", "base_url"), ("llm", "model"), ("salome", "root")]


class ConfigError(Exception):
    pass


def load_config(path: str) -> dict:
    with open(path) as f:
        config = json.load(f)
    for section, key in REQUIRED:
        if key not in config.get(section, {}):
            raise ConfigError(f"missing required config key: {section}.{key}")
    return config
