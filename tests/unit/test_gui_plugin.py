import sys
import types


def _install_fakes(monkeypatch):
    fake_gb = types.ModuleType("agent_server.gui_backend")

    class FakeInvoker:
        def __init__(self):
            pass

    class FakeGuiBackend:
        mode = "gui"

        def __init__(self, invoker, build_namespace=None):
            pass

    fake_gb.MainThreadInvoker = FakeInvoker
    fake_gb.GuiBackend = FakeGuiBackend
    monkeypatch.setitem(sys.modules, "agent_server.gui_backend", fake_gb)


def test_launch_starts_server_and_writes_handshake(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    _install_fakes(monkeypatch)
    started = {}

    import agent_server.wire as wire

    class FakeServer:
        def __init__(self, core, token):
            started["token"] = token

        def bind(self):
            return 55555

        def serve_forever(self):
            started["served"] = True

        def stop(self):
            started["stopped"] = True

    monkeypatch.setattr(wire, "Server", FakeServer)

    from agent_server import gui_plugin, handshake
    gui_plugin.launch(context=types.SimpleNamespace(sg=None))
    hs = handshake.read()
    assert hs["port"] == 55555
    assert hs["mode"] == "gui"
    gui_plugin.stop()
    assert started["stopped"] is True


def test_launch_refuses_when_another_process_owns_the_server(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    _install_fakes(monkeypatch)

    from agent_server import gui_plugin, handshake

    monkeypatch.setattr(handshake, "live_owner",
                        lambda: {"pid": 4242, "mode": "headless", "port": 9})
    msgs = []
    monkeypatch.setattr(gui_plugin, "_status", msgs.append)

    import agent_server.wire as wire

    def _boom(*a, **k):
        raise AssertionError("must not build a server while one is running")

    monkeypatch.setattr(wire, "Server", _boom)

    gui_plugin.launch(context=types.SimpleNamespace(sg=None))
    assert any("already running" in m for m in msgs)
    assert handshake.read() is None
