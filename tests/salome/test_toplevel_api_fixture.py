import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.salome

FIXTURE = Path(__file__).resolve().parent.parent.parent / "salome_agent" / "verify" / "toplevel_api.json"
MODULES = ["geompy", "smesh", "pvsimple"]


def test_fixture_matches_live_introspect(agent_client):
    """verify/toplevel_api.json is the --no-session static-check API map and has
    no source but a live session. If this drifts, regenerate it:
        python3 scripts/regen-toplevel-api.py
    """
    raw = agent_client.introspect(MODULES)
    fixture = json.loads(FIXTURE.read_text())
    for name in MODULES:
        live = sorted(m for m in raw.get(name, {}).get("members", []) if not m.startswith("_"))
        if not live:  # module loaded lazily on this server — fixture keeps its own list
            continue
        assert fixture.get(name, {}).get("members") == live, (
            f"{name} member list is stale — run scripts/regen-toplevel-api.py")
