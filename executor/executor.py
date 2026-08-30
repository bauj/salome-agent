import argparse
import json
import os
import socket
import sys
import threading
import traceback as tb_module
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO

# So `import executor.geom_tools` resolves regardless of the cwd `salome
# shell` launches this script from.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Imported here, at module load time, before any SALOME import runs —
# confirmed 2026-08-29 that something in build_namespace()'s salome_init()/
# geomBuilder.New() chain mutates sys.path (KERNEL's salome/__init__.py
# defines an extend_path() for exactly this kind of namespace-package
# surgery), which silently drops our sys.path.insert above by the time a
# LAZY `from executor.geom_tools import ...` inside handle() would run.
# Importing early, before that mutation happens, sidesteps it entirely.
from executor import geom_tools


def _run_with_timeout(code: str, namespace: dict, timeout: float) -> dict:
    out, err = StringIO(), StringIO()
    result = {"ok": False, "stdout": "", "stderr": "", "exception": None,
              "traceback": None, "offending_line": None}

    def target():
        try:
            with redirect_stdout(out), redirect_stderr(err):
                compiled = compile(code, "<stage>", "exec")
                exec(compiled, namespace)
            result["ok"] = True
        except SyntaxError as e:
            result["exception"] = str(e)
            result["offending_line"] = e.lineno
        except BaseException as e:
            result["exception"] = str(e)
            result["traceback"] = "".join(tb_module.format_exc())
            for entry in tb_module.extract_tb(e.__traceback__):
                if entry.filename == "<stage>":
                    result["offending_line"] = entry.lineno

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        result["exception"] = f"timeout after {timeout}s"
        # Best-effort only — a C-level Compute() may not respond to this
        # (spec §12 risk: "may need to hard-exit and be relaunched").
    result["stdout"] = out.getvalue()
    result["stderr"] = err.getvalue()
    return result


def _find_object(name: str):
    # Whole-study search by name, not folder-scoped. study_ns folder
    # scoping (geompy.PutToFolder / attach-mode rollback, spec §4.3/§12)
    # is a separate, still-open piece of work — geompy.PutToFolder did not
    # actually nest objects under the folder in manual testing against
    # this SALOME build (2026-08-29), so folder-scoped lookup is deferred
    # rather than shipped half-working. This search-by-name is correct as
    # long as published names stay unique/deterministic (spec §5.2).
    from salome.kernel import salome as salome_kernel

    def walk(sobj):
        it = salome_kernel.myStudy.NewChildIterator(sobj)
        while it.More():
            child = it.Value()
            if child.GetName() == name:
                return child.GetObject()
            found = walk(child)
            if found is not None:
                return found
            it.Next()
        return None

    root = salome_kernel.myStudy.FindObjectID("0:1")
    result = walk(root)
    if result is None:
        raise LookupError(f"no study object named {name!r}")
    return result


def build_namespace(live: bool = False) -> dict:
    # Real import convention for this SALOME build, confirmed by manual
    # testing 2026-08-29 — NOT `import salome; import geompy; import smesh`
    # as earlier spec drafts assumed. `geompy`/`smesh`/`salome` are still
    # bound under those exact names below, so stage scripts and the static
    # checker (verify/static.py's TOPLEVEL_MODULES) see the same bare
    # top-level names either way.
    if live:
        # The one line that makes geomBuilder.New() etc. return the SAME
        # engine instance as the already-open GUI, instead of a private
        # disconnected one. Must run BEFORE salome_init(). Confirmed the
        # entire fix by the task-172 patch matrix (2026-08-30).
        from salome.kernel import KernelBasis
        KernelBasis.setSSLMode(False)
    from salome.kernel import salome

    # When the agent just launched the GUI itself, the naming service
    # registers before the ModuleCatalog / Container servers do — salome_init()
    # then dies with "'NoneType' object has no attribute '_narrow'" (a failed
    # Resolve). Retry through that startup window.
    import time as _t
    last_exc = None
    for _ in range(15):
        try:
            salome.salome_init()
            from salome.geom import geomBuilder
            geompy = geomBuilder.New()
            from salome.smesh import smeshBuilder
            smesh = smeshBuilder.New()
            break
        except (AttributeError, RuntimeError) as e:
            last_exc = e
            _t.sleep(2)
    else:
        raise RuntimeError(f"salome_init did not succeed after retries: {last_exc}")
    try:
        import pvsimple
    except ImportError:
        pvsimple = None
    study_ns = "salome_agent_run"
    return {
        "salome": salome, "geompy": geompy, "smesh": smesh, "pvsimple": pvsimple,
        "study_ns": study_ns,
        "find_object": _find_object,
    }


class Executor:
    def __init__(self, socket_path: str, live: bool = False):
        self.socket_path = socket_path
        self.live = live
        self.namespace = build_namespace(live)
        self.accepted_scripts: list[str] = []
        self.geom_bindings: dict[str, object] = {}  # filled in by geom_tools.py

    def _emit_refresh(self, module: str | None) -> None:
        # Push a live GUI update after something was published. No-op unless
        # attached to a real GUI (--live). Best-effort: a refresh failure
        # must never fail the stage. `register_module_in_study/<M>` loads the
        # module + connects it to the study so its tree/shapes become visible;
        # `updateObjBrowser` redraws. Both run on the GUI's own Qt thread
        # (LightApp_Application::onDesktopMessage).
        if not self.live:
            return
        try:
            salome = self.namespace["salome"]
            from salome.kernel import SALOME_CMOD
            sess = salome.naming_service.Resolve("/Kernel/Session")._narrow(SALOME_CMOD.Session)
            if module:
                sess.emitMessageOneWay(f"register_module_in_study/{module}")
            sess.emitMessageOneWay("updateObjBrowser")
        except Exception as e:
            print(f"[executor] live GUI refresh failed: {e}", file=sys.stderr)

    def handle(self, req: dict) -> dict:
        cmd = req["cmd"]
        if cmd == "run":
            r = _run_with_timeout(req["code"], self.namespace, req["timeout"])
            if r["ok"]:
                self._emit_refresh(req.get("module"))
            return {"id": req["id"], **r}
        if cmd == "reset_and_replay":
            self.namespace = build_namespace(self.live)
            self.geom_bindings.clear()
            for idx, script in enumerate(req["scripts"]):
                r = _run_with_timeout(script, self.namespace, req.get("timeout", 120))
                if not r["ok"]:
                    return {"id": req["id"], "ok": False, "failed_index": idx}
            self.accepted_scripts = list(req["scripts"])
            self._emit_refresh(None)
            return {"id": req["id"], "ok": True, "failed_index": None}
        if cmd == "introspect":
            out = {}
            for name in req["names"]:
                obj = self.namespace.get(name)
                out[name] = {"members": dir(obj) if obj is not None else []}
            return {"id": req["id"], **out}
        if cmd == "list_study":
            salome = self.namespace["salome"]
            objs = []
            root = salome.myStudy.FindObjectID("0:1")
            it = salome.myStudy.NewChildIterator(root)
            while it.More():
                comp = it.Value()
                objs.append({"name": comp.GetName(), "type": ""})
                it.Next()
            return {"id": req["id"], "objects": objs}
        if cmd == "sanity":
            return {"id": req["id"], "ok": True, "warnings": [], "failures": []}
        if cmd == "list_tools":
            return {"id": req["id"], "tools": geom_tools.list_tools()}
        if cmd == "call_tool":
            result = geom_tools.call_tool(self, req["name"], req["args"])
            if result.get("ok"):
                self._emit_refresh("GEOM")
            return {"id": req["id"], **result}
        if cmd == "shutdown":
            return {"id": req["id"], "ok": True}
        return {"id": req["id"], "ok": False, "exception": f"unknown command {cmd!r}"}

    def serve(self):
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        srv.bind(self.socket_path)
        srv.listen(1)
        conn, _ = srv.accept()
        f = conn.makefile("rwb")
        f.write((json.dumps({"ready": True}) + "\n").encode())
        f.flush()
        for line in f:
            req = json.loads(line)
            resp = self.handle(req)
            f.write((json.dumps(resp) + "\n").encode())
            f.flush()
            if req["cmd"] == "shutdown":
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket", required=True)
    parser.add_argument("--live", action="store_true",
                        help="attach to a running GUI session (setSSLMode(False)) and "
                             "push Object Browser refreshes after each publish")
    args = parser.parse_args()
    Executor(args.socket, live=args.live).serve()
