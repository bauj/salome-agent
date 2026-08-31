import gc
import threading
import time

import pytest

pytestmark = pytest.mark.salome

try:
    import SalomePyQt  # noqa: F401
    _HAVE_QT = True
except ImportError:
    _HAVE_QT = False


@pytest.fixture
def qapp():
    if not _HAVE_QT:
        pytest.skip("run under `salome shell -- pytest`")
    from PySide2.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app


def _pump_until(box, key, deadline=3):
    from PySide2.QtCore import QCoreApplication
    start = time.time()
    while key not in box and time.time() - start < deadline:
        QCoreApplication.processEvents()
        time.sleep(0.01)


def test_marshal_fires_from_bare_thread_onto_main(qapp):
    from agent_server.gui_backend import MainThreadInvoker
    inv = MainThreadInvoker()
    gc.collect()
    box = {}

    def worker():
        box["ran_on"] = inv.run_on_main(
            lambda: threading.current_thread().name, timeout=2)

    threading.Thread(target=worker, daemon=True).start()
    _pump_until(box, "ran_on")
    assert box.get("ran_on") == threading.main_thread().name


def test_slot_exception_surfaces_without_abort(qapp):
    from agent_server.gui_backend import MainThreadInvoker
    inv = MainThreadInvoker()
    box = {}

    def worker():
        try:
            inv.run_on_main(lambda: (_ for _ in ()).throw(ValueError("boom")),
                            timeout=2)
        except ValueError as e:
            box["err"] = str(e)

    threading.Thread(target=worker, daemon=True).start()
    _pump_until(box, "err")
    assert box["err"] == "boom"


def test_timeout_raises_and_latches_busy(qapp):
    from agent_server.gui_backend import MainThreadInvoker
    inv = MainThreadInvoker()
    # nothing pumps the event loop -> the emit never runs -> bounded wait fires
    with pytest.raises(TimeoutError):
        inv.run_on_main(lambda: 1, timeout=0.2)
    with pytest.raises(RuntimeError):
        inv.run_on_main(lambda: 1, timeout=0.2)   # still latched
