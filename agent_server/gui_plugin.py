import os
import secrets
import threading

from agent_server import handshake
from agent_server import wire as _wire
from agent_server.core import Core

_INVOKER = None
_SERVER = None
_SERVER_THREAD = None
_CORE = None


def _status(text):
    try:
        import SalomePyQt
        SalomePyQt.SalomePyQt().message(text, False)
    except Exception:
        print(f"[agent-server] {text}")


def stop() -> None:
    global _SERVER, _SERVER_THREAD
    if _SERVER is not None:
        _SERVER.stop()
    if _SERVER_THREAD is not None:
        _SERVER_THREAD.join(timeout=3)
    _SERVER = _SERVER_THREAD = None
    handshake.remove()


def launch(context) -> None:
    global _INVOKER, _SERVER, _SERVER_THREAD, _CORE
    from agent_server.gui_backend import GuiBackend, MainThreadInvoker

    stop()                                    # 2nd click == restart
    try:
        if _INVOKER is None:
            _INVOKER = MainThreadInvoker()
        backend = GuiBackend(_INVOKER)
        token = secrets.token_hex(16)
        _CORE = Core(backend)
        _SERVER = _wire.Server(_CORE, token)
        port = _SERVER.bind()
        _SERVER_THREAD = threading.Thread(target=_SERVER.serve_forever, daemon=True)
        _SERVER_THREAD.start()
        handshake.write(port=port, token=token, mode="gui",
                        log_path=os.path.expanduser("~/.salome_agent/server.log"))
        _register_about_to_quit()
        _status(f"SALOME agent server on 127.0.0.1:{port}")
    except Exception as e:                     # noqa: BLE001
        _status(f"SALOME agent server failed to start: {e}")
        raise


def _register_about_to_quit() -> None:
    try:
        try:
            from PySide2.QtWidgets import QApplication
        except ImportError:
            from PyQt5.QtWidgets import QApplication
        app = QApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(stop)
    except Exception:
        pass
