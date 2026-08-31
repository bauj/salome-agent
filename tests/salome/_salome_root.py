"""Resolve the SALOME install root for the live (`-m salome`) tests.

Order: the `SALOME_ROOT` env var, then `salome.root` in a repo-root
`config.json`. If neither is set the live tests cannot run — set one of
them (GitHub issue #1: no install path baked into the source).
"""
import json
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def salome_root() -> str | None:
    env = os.environ.get("SALOME_ROOT")
    if env:
        return env
    cfg = REPO_ROOT / "config.json"
    if cfg.exists():
        try:
            root = json.loads(cfg.read_text()).get("salome", {}).get("root")
            if root:
                return root
        except (json.JSONDecodeError, OSError):
            pass
    return None


SALOME_ROOT = salome_root()
