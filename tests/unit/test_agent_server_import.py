import importlib
import sys

import pytest


@pytest.mark.xfail(reason="core/backend land in later tasks", strict=False)
def test_core_and_backend_import_without_qt(monkeypatch):
    # Simulate the dev venv: no PySide2, no SalomePyQt.
    for mod in list(sys.modules):
        if mod.startswith(("PySide2", "SalomePyQt", "agent_server")):
            monkeypatch.delitem(sys.modules, mod, raising=False)
    monkeypatch.setitem(sys.modules, "PySide2", None)
    monkeypatch.setitem(sys.modules, "SalomePyQt", None)

    importlib.import_module("agent_server.core")
    importlib.import_module("agent_server.backend")
    importlib.import_module("agent_server.handshake")
