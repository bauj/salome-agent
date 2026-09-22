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


def forget_names(namespace: dict, geom_bindings: dict, names: list, erase=None) -> dict:
    """erase(entry), when given, takes the object out of the viewer first —
    removing the study object alone leaves its presentation on screen."""
    study = namespace["salome"].myStudy
    builder = study.NewBuilder()
    removed = []
    for name in names:
        geom_bindings.pop(name, None)
        for sobj in study.FindObjectByName(name, "GEOM"):
            if erase is not None:
                try:
                    erase(sobj.GetID())
                except Exception:
                    pass
            builder.RemoveObjectWithChildren(sobj)
            removed.append(name)
    return {"ok": True, "removed": removed}


class Backend(Protocol):
    name_prefix: str
    namespace: dict
    geom_bindings: dict
    accepted_scripts: list

    def begin_session(self, run_id: str | None) -> None: ...
    def end_session(self) -> None: ...
    def run_code(self, code: str, timeout: float, module: str | None = None) -> dict: ...
    def reset(self, scripts: list) -> dict: ...
    def refresh(self, module: str | None) -> None: ...
    def forget(self, names: list) -> dict: ...
    def find_object(self, name: str): ...
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
        self.accepted_scripts: list = []
        self._poisoned = False

    def begin_session(self, run_id: str | None) -> None:
        geom.reset_counters()
        self.geom_bindings.clear()
        self.accepted_scripts = []

    def end_session(self) -> None:
        pass

    def run_code(self, code: str, timeout: float, module: str | None = None) -> dict:
        r = _run_with_timeout(code, self.namespace, timeout)
        if not r["ok"] and (r["exception"] or "").startswith("timeout after"):
            self._poisoned = True
        return r

    def reset(self, scripts: list) -> dict:
        self.namespace = self._build(False)
        geom.reset_counters()
        self.geom_bindings.clear()
        for idx, s in enumerate(scripts):
            r = _run_with_timeout(s, self.namespace, 120)
            if not r["ok"]:
                return {"ok": False, "failed_index": idx}
        self.accepted_scripts = list(scripts)
        return {"ok": True, "failed_index": None}

    def refresh(self, module: str | None) -> None:
        pass

    def forget(self, names: list) -> dict:
        return forget_names(self.namespace, self.geom_bindings, names)

    def find_object(self, name: str):
        return self.namespace["find_object"](name)

    def poisoned(self) -> bool:
        return self._poisoned
