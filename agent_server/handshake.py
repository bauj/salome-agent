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


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except PermissionError:
        return True
    except OSError:
        return False


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


def live_owner() -> dict | None:
    """The handshake's server if its pid is still alive and is the same
    process that wrote it (pid_start guards against pid reuse). Else None --
    stale file, dead pid, or a recycled pid.
    """
    data = read()
    if data is None:
        return None
    pid = data.get("pid")
    if not isinstance(pid, int) or pid == os.getpid():
        return None
    if os.path.isdir("/proc"):
        live = _pid_start(pid)
        if live is None:
            return None                   # pid not running
        stored = data.get("pid_start")
        if stored is not None and stored != live:
            return None                   # pid was reused by an unrelated process
    elif not _pid_alive(pid):
        return None
    return data


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
