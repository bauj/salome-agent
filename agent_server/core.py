import traceback

from executor import geom, shaper, smesh  # top-level — salome_init mutates sys.path later
from agent_server import handshake
from agent_server.limits import SIDE_MODULES, TOOL_TIMEOUT

# Tool-calling modules: name -> (package, the SALOME module refreshed after a call).
_TOOL_MODULES = {"geom": geom, "shaper": shaper, "smesh": smesh}
_REFRESH = {"geom": "GEOM", "shaper": "SHAPER", "smesh": "SMESH"}
# Top-level APIs a script may call that are not in the server namespace: imported
# on demand (on the GUI thread) for introspect.
_LAZY_APIS = {"model": "salome.shaper.model"}

_FATAL = {"ok": False, "fatal": True,
          "error": "server poisoned by a prior timeout; restart the agent server"}


class Core:
    def __init__(self, backend, allow_run: bool = False):
        self.backend = backend
        # `run` executes any code sent with the token; only the live tests use it.
        self.allow_run = allow_run

    def poisoned(self) -> bool:
        return self.backend.poisoned()

    def handle(self, req: dict) -> dict:
        rid = req.get("id")
        if self.backend.poisoned():
            return {"id": rid, **_FATAL}
        try:
            return self._dispatch(req, rid)
        except Exception as e:  # noqa: BLE001
            if self.backend.poisoned():
                return {"id": rid, **_FATAL}
            # An answer, not a dropped connection: the session and the study
            # stay usable, and the caller sees what broke.
            traceback.print_exc()
            return {"id": rid, "ok": False,
                    "error": f"{req.get('cmd')} failed on the server: "
                             f"{type(e).__name__}: {e}"}

    def _dispatch(self, req: dict, rid) -> dict:
        cmd = req.get("cmd")
        b = self.backend

        if cmd == "hello":
            b.begin_session(req.get("run_id"))
            # where the tools write files (export_mesh)
            b.run_dir = req.get("run_dir")
            _reset_side_modules(b)
            return {"id": rid, "ready": True, "proto": handshake.PROTO,
                    "mode": b.mode, "name_prefix": b.name_prefix,
                    "leftovers": b.leftovers()}
        if cmd == "run":
            if not self.allow_run:
                return {"id": rid, "ok": False,
                        "error": "run is disabled on this server (start it with --allow-run)"}
            r = b.run_code(req["code"], req["timeout"], req.get("module"))
            if r["ok"]:
                b.refresh(req.get("module"))
            return {"id": rid, **r}
        if cmd == "introspect":
            out = {}
            for name in req["names"]:
                obj = b.namespace.get(name)
                if obj is None and name in _LAZY_APIS:
                    obj = b.run_tool(lambda: __import__(_LAZY_APIS[name], fromlist=["_"]),
                                     TOOL_TIMEOUT)
                out[name] = {"members": dir(obj) if obj is not None else []}
            return {"id": rid, **out}
        if cmd == "list_study":
            salome = b.namespace["salome"]
            objs = []
            root = salome.myStudy.FindObjectID("0:1")
            it = salome.myStudy.NewChildIterator(root)
            while it.More():
                objs.append({"name": it.Value().GetName(), "type": ""})
                it.Next()
            return {"id": rid, "objects": objs}
        if cmd in ("list_tools", "call_tool"):
            module = req.get("module") or "geom"
            pkg = _TOOL_MODULES.get(module)
            if pkg is None:
                return {"id": rid, "ok": False, "error": f"no tool module {module!r}"}
            if cmd == "list_tools":
                return {"id": rid, "tools": pkg.list_tools(), "module": module}
            call = lambda: pkg.call_tool(b, req["name"], req["args"], pin=req.get("pin"))
            limit = getattr(pkg, "timeout_for", lambda n: None)(req["name"]) or TOOL_TIMEOUT
            r = call() if module == "geom" else b.run_tool(call, limit)
            if r.get("ok"):
                b.refresh(_REFRESH[module])
            return {"id": rid, **r}
        if cmd == "check_results":
            module = req.get("module") or "geom"
            pkg = _TOOL_MODULES.get(module)
            if pkg is None:
                return {"id": rid, "ok": False, "error": f"no tool module {module!r}"}
            call = lambda: pkg.check_results(b, req["names"])
            r = call() if module == "geom" else b.run_tool(call, TOOL_TIMEOUT)
            return {"id": rid, "ok": True, "results": r}
        if cmd == "forget":
            r = b.forget(req["names"])
            gone = _forget_side_modules(b, req["names"])
            if gone:
                r = {**r, "removed": list(dict.fromkeys(list(r.get("removed") or []) + gone))}
            b.refresh("GEOM")
            return {"id": rid, **r}
        if cmd == "clear_study":
            for pkg in _side_modules():
                b.run_tool(lambda: pkg.clear(b), TOOL_TIMEOUT)
            r = b.clear_study()
            b.refresh("GEOM")
            return {"id": rid, **r}
        if cmd == "display":
            r = b.display(req["names"])
            return {"id": rid, **r, "shown": r["shown"] + _show_side_modules(b, req["names"], True)}
        if cmd == "hide":
            r = b.hide(req["names"])
            return {"id": rid, **r, "hidden": r["hidden"] + _show_side_modules(b, req["names"], False)}
        if cmd == "shutdown":
            b.end_session()
            return {"id": rid, "ok": True}
        return {"id": rid, "ok": False, "exception": f"unknown command {cmd!r}"}


# Side modules (SHAPER) live in the GUI process and notify its widgets on the
# calling thread, so every call into them goes through backend.run_tool.
def _side_modules():
    return [_TOOL_MODULES[name] for name in SIDE_MODULES]


def _reset_side_modules(b) -> None:
    for pkg in _side_modules():
        pkg.reset(b)


# SHAPER draws its results in the OCC viewer itself; their SHAPERSTUDY entries
# are for the other modules, and a presentation made from one has no eye in
# any tree there.
def _show_side_modules(b, names: list, shown: bool) -> list:
    return [n for pkg in _side_modules() if hasattr(pkg, "show")
            for n in b.run_tool(lambda: pkg.show(b, names, shown), TOOL_TIMEOUT)]


def _forget_side_modules(b, names: list) -> list:
    return [n for pkg in _side_modules()
            for n in b.run_tool(lambda: pkg.forget(b, names), TOOL_TIMEOUT)]
