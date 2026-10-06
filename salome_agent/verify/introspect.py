import json
from pathlib import Path

FIXTURE_PATH = Path(__file__).parent / "toplevel_api.json"
# geompy and smesh live in the server namespace; model is SHAPER's API.
MODULES = ["geompy", "smesh", "model"]


def build_api_map(client=None) -> dict:
    if client is None:
        return json.loads(FIXTURE_PATH.read_text())
    return client.introspect(MODULES)
