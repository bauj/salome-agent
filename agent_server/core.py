from executor import geom  # top-level — salome_init mutates sys.path later
from agent_server import handshake

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
            return {"id": rid, "ready": True, "proto": handshake.PROTO,
                    "mode": b.mode, "name_prefix": b.name_prefix,
                    "leftovers": b.leftovers()}
        if cmd == "run":
            r = b.run_code(req["code"], req["timeout"], req.get("module"))
            if r["ok"]:
                b.refresh(req.get("module"))
            return {"id": rid, **r}
        if cmd == "reset_and_replay":
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
        if cmd == "list_tools":
            return {"id": rid, "tools": geom.list_tools()}
        if cmd == "call_tool":
            r = geom.call_tool(b, req["name"], req["args"], pin=req.get("pin"))
            if r.get("ok"):
                b.refresh("GEOM")
            return {"id": rid, **r}
        if cmd == "forget":
            r = b.forget(req["names"])
            b.refresh("GEOM")
            return {"id": rid, **r}
        if cmd == "clear_study":
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
