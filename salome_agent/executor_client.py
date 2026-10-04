import json
import socket


class SessionDied(Exception):
    pass


class SessionPoisoned(Exception):
    pass


class ServerBusy(Exception):
    pass


class ExecutorClient:
    def __init__(self, host: str, port: int, token: str, run_id: str,
                 record_path: str | None = None, hold: bool = False):
        self.record_path = record_path
        self._sock = socket.create_connection((host, port), timeout=10)
        self._f = self._sock.makefile("rwb")
        self._next_id = 1
        hello = {"id": 0, "cmd": "hello", "token": token, "run_id": run_id}
        if hold:
            hello["hold"] = True
        self._f.write((json.dumps(hello) + "\n").encode())
        self._f.flush()
        ready = self._recv()
        if ready.get("fatal"):
            raise SessionPoisoned(ready.get("error", "poisoned"))
        if ready.get("error") == "busy":
            raise ServerBusy("another agent run is in progress")
        assert ready.get("ready") is True
        self.mode = ready.get("mode")
        self.name_prefix = ready.get("name_prefix", "")
        self.leftovers = ready.get("leftovers", 0)

    def _recv(self) -> dict:
        try:
            line = self._f.readline()
        except (BrokenPipeError, ConnectionResetError, OSError) as e:
            raise SessionDied(f"server socket read failed: {e}") from e
        if not line:
            raise SessionDied("server socket closed")
        return json.loads(line)

    def _call(self, cmd: str, **payload) -> dict:
        req_id = self._next_id
        self._next_id += 1
        try:
            self._f.write((json.dumps({"id": req_id, "cmd": cmd, **payload}) + "\n").encode())
            self._f.flush()
        except (BrokenPipeError, ConnectionResetError, OSError) as e:
            raise SessionDied(f"server socket write failed: {e}") from e
        resp = self._recv()
        if resp.get("fatal"):
            raise SessionPoisoned(resp.get("error", "poisoned"))
        if self.record_path:
            self._record(cmd, payload, resp)
        return resp

    def _record(self, cmd: str, payload: dict, resp: dict) -> None:
        # Recorded so a run can be replayed offline against a golden result.py;
        # a failure here must never abort the run it is only observing.
        try:
            import os

            os.makedirs(os.path.dirname(self.record_path) or ".", exist_ok=True)
            with open(self.record_path, "a") as f:
                f.write(json.dumps({"cmd": cmd, "payload": payload, "response": resp},
                                   default=str, ensure_ascii=False) + "\n")
        except OSError:
            pass

    def run(self, code, timeout, module=None):
        return self._call("run", code=code, timeout=timeout, module=module)

    def reset_and_replay(self, scripts):
        return self._call("reset_and_replay", scripts=scripts)

    def introspect(self, names):
        return self._call("introspect", names=names)

    def list_study(self):
        return self._call("list_study")

    def sanity(self, module, expectations):
        return self._call("sanity", module=module, expectations=expectations)

    def list_tools(self, module=None):
        if module is None:
            return self._call("list_tools")
        return self._call("list_tools", module=module)

    def call_tool(self, name, args, pin=None, module=None):
        extra = {}
        if pin is not None:
            extra["pin"] = pin
        if module is not None:
            extra["module"] = module
        return self._call("call_tool", name=name, args=args, **extra)

    def forget(self, names):
        return self._call("forget", names=names)

    def clear_study(self):
        return self._call("clear_study")

    def display(self, names):
        return self._call("display", names=names)

    def hide(self, names):
        return self._call("hide", names=names)

    def shutdown(self):
        return self._call("shutdown")

    def close(self):
        """Drop the connection without stopping the server. The server is
        owned by whoever started it (the GUI menu, start-headless-server.sh,
        or `--once` self-exit) — a transient CLI run must not shut it down
        (GH #26)."""
        try:
            self._f.close()
        except OSError:
            pass
        try:
            self._sock.close()
        except OSError:
            pass
