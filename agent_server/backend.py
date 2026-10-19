import threading
import traceback as tb_module
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from typing import Protocol

from executor import geom


def _run_with_timeout(code: str, namespace: dict, timeout: float) -> dict:
    # Moved verbatim from executor/executor.py. The watchdog thread is a
    # best-effort bound only — a C-level Compute() may not respond. In the
    # persistent-server world a timeout must also poison the backend
    # (HeadlessBackend.run_code below) since the abandoned thread keeps
    # mutating `namespace`.
    out, err = StringIO(), StringIO()
    result = {"ok": False, "stdout": "", "stderr": "", "exception": None,
              "traceback": None, "offending_line": None}

    def target():
        try:
            with redirect_stdout(out), redirect_stderr(err):
                exec(compile(code, "<stage>", "exec"), namespace)
            result["ok"] = True
        except SyntaxError as e:
            result["exception"] = str(e)
            result["offending_line"] = e.lineno
        except BaseException as e:  # noqa: BLE001
            result["exception"] = str(e)
            result["traceback"] = "".join(tb_module.format_exc())
            for entry in tb_module.extract_tb(e.__traceback__):
                if entry.filename == "<stage>":
                    result["offending_line"] = entry.lineno

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        result["exception"] = f"timeout after {timeout}s"
    result["stdout"] = out.getvalue()
    result["stderr"] = err.getvalue()
    return result


# Where an agent-built object shows in the study tree and the viewer.
_SHAPE_COMPONENTS = ("GEOM", "SHAPERSTUDY")


def _find(study, name, components=_SHAPE_COMPONENTS):
    return [sobj for comp in components for sobj in study.FindObjectByName(name, comp)]


def forget_names(namespace: dict, geom_bindings: dict, names: list, erase=None) -> dict:
    """erase(entry), when given, takes the object out of the viewer first —
    removing the study object alone leaves its presentation on screen."""
    study = namespace["salome"].myStudy
    builder = study.NewBuilder()
    removed = []
    for name in names:
        geom_bindings.pop(name, None)
        for sobj in _find(study, name):
            if erase is not None:
                try:
                    erase(sobj.GetID())
                except Exception:
                    pass
            builder.RemoveObjectWithChildren(sobj)
            if name not in removed:
                removed.append(name)
    return {"ok": True, "removed": removed}


def display_names(namespace: dict, names: list, show) -> dict:
    study = namespace["salome"].myStudy
    shown = []
    for name in names:
        for sobj in _find(study, name, ("GEOM",)):
            show(sobj.GetID())
            shown.append(name)
    return {"ok": True, "shown": shown}


def hide_names(namespace: dict, names: list, erase) -> dict:
    """Takes objects out of the viewer; they stay in the study."""
    study = namespace["salome"].myStudy
    hidden = []
    for name in names:
        for sobj in _find(study, name, ("GEOM",)):
            erase(sobj.GetID())
            hidden.append(name)
    return {"ok": True, "hidden": hidden}


_AGENT_COMPONENTS = ("GEOM", "SMESH", "SHAPERSTUDY")


def study_objects(namespace: dict) -> list:
    study = namespace["salome"].myStudy
    out = []
    for comp_name in _AGENT_COMPONENTS:
        comp = study.FindComponent(comp_name)
        if comp is None:
            continue
        it = study.NewChildIterator(comp)
        while it.More():
            sobj = it.Value()
            # a removed object leaves its label behind, emptied of attributes
            if sobj.GetAllAttributes():
                out.append(sobj)
            it.Next()
    return out


def clear_study(namespace: dict) -> int:
    objs = study_objects(namespace)
    builder = namespace["salome"].myStudy.NewBuilder()
    for sobj in objs:
        builder.RemoveObjectWithChildren(sobj)
    return len(objs)


class Backend(Protocol):
    name_prefix: str
    namespace: dict
    geom_bindings: dict

    def begin_session(self, run_id: str | None) -> None: ...
    def end_session(self) -> None: ...
    def run_code(self, code: str, timeout: float, module: str | None = None) -> dict: ...
    def refresh(self, module: str | None) -> None: ...
    def run_tool(self, fn, timeout: float): ...
    def forget(self, names: list) -> dict: ...
    def display(self, names: list) -> dict: ...
    def hide(self, names: list) -> dict: ...
    def view_image(self) -> dict: ...
    def find_object(self, name: str): ...
    def leftovers(self) -> int: ...
    def clear_study(self) -> dict: ...
    def poisoned(self) -> bool: ...


class HeadlessBackend:
    mode = "headless"

    def __init__(self, build_namespace=None):
        if build_namespace is None:
            from agent_server.namespace import build_namespace as bn
            build_namespace = bn
        self._build = build_namespace
        self.name_prefix = ""
        self.namespace = self._build(False)
        self.geom_bindings: dict = {}
        self._poisoned = False

    def begin_session(self, run_id: str | None) -> None:
        # The study outlives the run on a --serve-forever server, so name by
        # run as GuiBackend does: forget-by-name must not reach another run's objects.
        self.name_prefix = f"sa_{run_id}_" if run_id else ""
        geom.reset_counters()
        self.geom_bindings.clear()

    def end_session(self) -> None:
        pass

    def run_code(self, code: str, timeout: float, module: str | None = None) -> dict:
        r = _run_with_timeout(code, self.namespace, timeout)
        if not r["ok"] and (r["exception"] or "").startswith("timeout after"):
            self._poisoned = True
        return r

    def refresh(self, module: str | None) -> None:
        pass

    def run_tool(self, fn, timeout: float):
        return fn()

    def forget(self, names: list) -> dict:
        return forget_names(self.namespace, self.geom_bindings, names)

    def display(self, names: list) -> dict:
        return {"ok": True, "shown": []}

    def hide(self, names: list) -> dict:
        return {"ok": True, "hidden": []}

    def view_image(self) -> dict:
        return {"ok": False, "error": "this SALOME session is headless: it has no viewer"}

    def find_object(self, name: str):
        fo = self.namespace["find_object"]
        try:
            return fo(self.name_prefix + name)
        except LookupError:
            return fo(name)

    def leftovers(self) -> int:
        return len(study_objects(self.namespace))

    def clear_study(self) -> dict:
        return {"ok": True, "removed": clear_study(self.namespace)}

    def poisoned(self) -> bool:
        return self._poisoned
