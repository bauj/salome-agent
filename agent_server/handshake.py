import json
import os
import time
from pathlib import Path

PROTO = 1


def path() -> Path:
    return Path(os.path.expanduser("~/.salome_agent/server.json"))


def _pid_start(pid: int) -> int | None:
    try:
        with open(f"/proc/{pid}/stat") as f:
            return int(f.read().rsplit(")", 1)[1].split()[19])
    except (OSError, IndexError, ValueError):
        return None


def write(*, port: int, token: str, mode: str, log_path: str,
          salome_port: int | None = None) -> None:
    p = path()
    p.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(p.parent, 0o700)
    pid = os.getpid()
    data = {
        "proto": PROTO, "host": "127.0.0.1", "port": port, "token": token,
        "pid": pid, "pid_start": _pid_start(pid), "mode": mode,
        "salome_port": salome_port, "log_path": log_path,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.chmod(tmp, 0o600)
    os.replace(tmp, p)


def read() -> dict | None:
    try:
        data = json.loads(path().read_text())
    except (OSError, ValueError):
        return None
    if data.get("proto") != PROTO:
        return None
    return data


def remove() -> None:
    try:
        data = json.loads(path().read_text())
    except (OSError, ValueError):
        return
    if data.get("pid") == os.getpid():
        path().unlink(missing_ok=True)
