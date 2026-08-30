import json
import socket


class SessionDied(Exception):
    pass


class ExecutorClient:
    def __init__(self, socket_path: str):
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock.connect(socket_path)
        self._f = self._sock.makefile("rwb")
        ready = self._recv()
        assert ready.get("ready") is True
        self._next_id = 1

    def _recv(self) -> dict:
        line = self._f.readline()
        if not line:
            raise SessionDied("executor socket closed")
        return json.loads(line)

    def _call(self, cmd: str, **payload) -> dict:
        req_id = self._next_id
        self._next_id += 1
        try:
            self._f.write((json.dumps({"id": req_id, "cmd": cmd, **payload}) + "\n").encode())
            self._f.flush()
        except (BrokenPipeError, ConnectionResetError, OSError) as e:
            raise SessionDied(f"executor socket write failed: {e}") from e
        return self._recv()

    def run(self, code: str, timeout: float, module: str | None = None) -> dict:
        return self._call("run", code=code, timeout=timeout, module=module)

    def reset_and_replay(self, scripts: list[str]) -> dict:
        return self._call("reset_and_replay", scripts=scripts)

    def introspect(self, names: list[str]) -> dict:
        return self._call("introspect", names=names)

    def list_study(self) -> dict:
        return self._call("list_study")

    def sanity(self, module: str, expectations: dict) -> dict:
        return self._call("sanity", module=module, expectations=expectations)

    def list_tools(self) -> dict:
        return self._call("list_tools")

    def call_tool(self, name: str, args: dict) -> dict:
        return self._call("call_tool", name=name, args=args)

    def shutdown(self) -> dict:
        return self._call("shutdown")
