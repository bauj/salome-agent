from pathlib import Path

# The checked-in map of SALOME's top-level APIs (scripts/regen-toplevel-api.py).
FIXTURE_PATH = Path(__file__).parent / "toplevel_api.json"
# geompy and smesh live in the server namespace; model is SHAPER's API.
MODULES = ["geompy", "smesh", "model"]
