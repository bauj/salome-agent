from executor import geom, shaper  # top-level — salome_init mutates sys.path later
from agent_server import handshake

# Tool-calling modules: name -> (package, the SALOME module refreshed after a call).
_TOOL_MODULES = {"geom": geom, "shaper": shaper}
_REFRESH = {"geom": "GEOM", "shaper": "SHAPER"}

_FATAL = {"ok": False, "fatal": True,
          "error": "server poisoned by a prior timeout; restart the agent server"}


class Core:
    def __init__(self, backend):
        self.backend = backend

    def poisoned(self) -> bool:
        return self.backend.poisoned()

    def handle(self, req: dict) -> dict:
        rid = req.get("id")
        if self.backend.poisoned():
            return {"id": rid, **_FATAL}
        cmd = req.get("cmd")
        b = self.backend

        if cmd == "hello":
            b.begin_session(req.get("run_id"))
            _reset_side_modules(b)
            return {"id": rid, "ready": True, "proto": handshake.PROTO,
                    "mode": b.mode, "name_prefix": b.name_prefix,
                    "leftovers": b.leftovers()}
        if cmd == "run":
            r = b.run_code(req["code"], req["timeout"], req.get("module"))
            if r["ok"]:
                b.refresh(req.get("module"))
            return {"id": rid, **r}
        if cmd == "reset_and_replay":
            _reset_side_modules(b)
            return {"id": rid, **b.reset(req["scripts"])}
        if cmd == "introspect":
            out = {}
            for name in req["names"]:
                obj = b.namespace.get(name)
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
        if cmd == "sanity":
            return {"id": rid, "ok": True, "warnings": [], "failures": []}
        if cmd in ("list_tools", "call_tool"):
            module = req.get("module") or "geom"
            pkg = _TOOL_MODULES.get(module)
            if pkg is None:
                return {"id": rid, "ok": False, "error": f"no tool module {module!r}"}
            if cmd == "list_tools":
                return {"id": rid, "tools": pkg.list_tools()}
            r = pkg.call_tool(b, req["name"], req["args"], pin=req.get("pin"))
            if r.get("ok"):
                b.refresh(_REFRESH[module])
            return {"id": rid, **r}
        if cmd == "forget":
            r = b.forget(req["names"])
            gone = _forget_side_modules(b, req["names"])
            if gone:
                r = {**r, "removed": list(r.get("removed") or []) + gone}
            b.refresh("GEOM")
            return {"id": rid, **r}
        if cmd == "clear_study":
            for pkg in _side_modules():
                pkg.clear(b)
            r = b.clear_study()
            b.refresh("GEOM")
            return {"id": rid, **r}
        if cmd == "display":
            return {"id": rid, **b.display(req["names"])}
        if cmd == "hide":
            return {"id": rid, **b.hide(req["names"])}
        if cmd == "shutdown":
            b.end_session()
            return {"id": rid, "ok": True}
        return {"id": rid, "ok": False, "exception": f"unknown command {cmd!r}"}


def _side_modules():
    return [pkg for name, pkg in _TOOL_MODULES.items() if name != "geom"]


def _reset_side_modules(b) -> None:
    for pkg in _side_modules():
        pkg.reset(b)


def _forget_side_modules(b, names: list) -> list:
    return [n for pkg in _side_modules() for n in pkg.forget(b, names)]
