# Copyright (C) 2026  salome-agent authors
# LGPL 2.1 or later
from pathlib import Path

# The checked-in map of SALOME's top-level APIs (scripts/regen-toplevel-api.py).
FIXTURE_PATH = Path(__file__).parent / "toplevel_api.json"
# geompy and smesh live in the server namespace; model is SHAPER's API.
MODULES = ["geompy", "smesh", "model"]
