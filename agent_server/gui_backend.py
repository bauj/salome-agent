import threading

import SalomePyQt

if SalomePyQt.usePySide():
    from PySide2.QtCore import QObject, Signal as _Signal, Qt
else:
    from PyQt5.QtCore import QObject, pyqtSignal as _Signal, Qt

from agent_server.backend import display_names, forget_names, hide_names
from agent_server.limits import MAIN_THREAD_TIMEOUT, REFRESH_TIMEOUT
from executor import geom


class _Job:
    __slots__ = ("fn", "box", "ev")

    def __init__(self, fn):
        self.fn = fn
        self.box: dict = {}
        self.ev = threading.Event()


class MainThreadInvoker(QObject):
    _invoke = _Signal(object)

    def __init__(self):
        super().__init__()
        self._lock = threading.Lock()
        self._busy = False
        self._invoke.connect(self._on_invoke, Qt.QueuedConnection)

    def _on_invoke(self, job):
        # Runs on the Qt main thread. An escaped exception here reaches Qt as
        # an unhandled slot exception -> qFatal() -> the GUI aborts. The guard
        # must be total and ev.set() must always run.
        try:
            job.box["r"] = job.fn()
        except BaseException as e:  # noqa: BLE001
            job.box["e"] = e
        finally:
            job.ev.set()

    def run_on_main(self, fn, timeout):
        with self._lock:
            if self._busy:
                raise RuntimeError("a marshalled call is already in flight")
            self._busy = True
        job = _Job(fn)
        timed_out = False
        try:
            self._invoke.emit(job)
            if not job.ev.wait(timeout):
                timed_out = True
                raise TimeoutError(f"main-thread call exceeded {timeout}s")
            if "e" in job.box:
                raise job.box["e"]
            return job.box["r"]
        finally:
            if not timed_out:            # leave _busy LATCHED after a timeout
                with self._lock:
                    self._busy = False


class GuiBackend:
    mode = "gui"

    def __init__(self, invoker: MainThreadInvoker, build_namespace=None):
        if build_namespace is None:
            from agent_server.namespace import build_namespace as bn
            build_namespace = bn
        self._invoker = invoker
        # __init__ runs on the Qt main thread (the plugin's launch()), so build
        # the namespace directly — marshalling from the main thread would
        # deadlock (ev.wait blocks the thread that must dispatch the slot).
        self.namespace = build_namespace(True)
        self.name_prefix = ""
        self.geom_bindings: dict = {}
        self._poisoned = False

    def begin_session(self, run_id):
        self.name_prefix = f"sa_{run_id}_" if run_id else ""
        geom.reset_counters()
        self.geom_bindings.clear()

    def end_session(self):
        pass

    def run_code(self, code, timeout, module=None):
        try:
            self._invoker.run_on_main(
                lambda: exec(compile(code, "<stage>", "exec"), self.namespace),
                timeout,
            )
            return {"ok": True, "stdout": "", "stderr": "", "exception": None,
                    "traceback": None, "offending_line": None}
        except TimeoutError:
            self._poisoned = True
            return {"ok": False, "stdout": "", "stderr": "",
                    "exception": f"timeout after {timeout}s",
                    "traceback": None, "offending_line": None}
        except BaseException as e:  # noqa: BLE001
            import traceback
            return {"ok": False, "stdout": "", "stderr": "", "exception": str(e),
                    "traceback": traceback.format_exc(), "offending_line": None}

    def run_tool(self, fn, timeout):
        try:
            return self._invoker.run_on_main(fn, timeout)
        except TimeoutError:
            self._poisoned = True
            raise

    def refresh(self, module):
        # salome.sg does not exist on this build (Phase 0 spike, 2026-08-31);
        # SALOMEGUI_Swig().updateObjBrowser() does, and runs a direct inline
        # Execute() on the session thread — which is where this already is,
        # having been marshalled by run_on_main.
        def _do():
            import libSALOME_Swig
            libSALOME_Swig.SALOMEGUI_Swig().updateObjBrowser()
        try:
            self._invoker.run_on_main(_do, timeout=REFRESH_TIMEOUT)
        except Exception:
            pass

    def forget(self, names):
        def _do():
            import libSALOME_Swig
            sg = libSALOME_Swig.SALOMEGUI_Swig()
            r = forget_names(self.namespace, self.geom_bindings, names, erase=sg.Erase)
            try:
                sg.UpdateView()
            except Exception:
                pass
            return r
        return self._invoker.run_on_main(_do, timeout=MAIN_THREAD_TIMEOUT)

    def display(self, names):
        def _do():
            import libSALOME_Swig
            sg = libSALOME_Swig.SALOMEGUI_Swig()
            r = display_names(self.namespace, names, sg.Display)
            sg.UpdateView()
            return r
        return self._invoker.run_on_main(_do, timeout=MAIN_THREAD_TIMEOUT)

    def hide(self, names):
        def _do():
            import libSALOME_Swig
            sg = libSALOME_Swig.SALOMEGUI_Swig()
            r = hide_names(self.namespace, names, sg.Erase)
            sg.UpdateView()
            return r
        return self._invoker.run_on_main(_do, timeout=MAIN_THREAD_TIMEOUT)

    def view_image(self):
        """The active view as the user sees it, as a PNG data: URI."""
        def _do():
            import base64
            import os
            import tempfile
            fd, path = tempfile.mkstemp(suffix=".png", prefix="sa_view_")
            os.close(fd)
            try:
                if not SalomePyQt.SalomePyQt().dumpView(path):
                    return {"ok": False, "error": "SALOME could not take a picture of the view"}
                with open(path, "rb") as f:
                    data = f.read()
            finally:
                os.remove(path)
            return {"ok": True, "image": "data:image/png;base64,"
                                         + base64.b64encode(data).decode("ascii")}
        return self._invoker.run_on_main(_do, timeout=MAIN_THREAD_TIMEOUT)

    def find_object(self, name):
        def _do():
            fo = self.namespace["find_object"]
            try:
                return fo(self.name_prefix + name)
            except LookupError:
                return fo(name)
        return self._invoker.run_on_main(_do, timeout=MAIN_THREAD_TIMEOUT)

    def leftovers(self):
        # the live study is the user's own; never offer to wipe it
        return 0

    def clear_study(self):
        return {"ok": False, "error": "will not clear a live GUI study"}

    def poisoned(self):
        return self._poisoned
