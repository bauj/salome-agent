import json
import os
import time
from pathlib import Path

PROTO = 1


_OLD_DIR = "~/.salome_agent"
_OLD_FILES = ("server.json", "server.json.tmp", "server.log", "salome-agent-gui.pid")


def state_dir() -> Path:
    return Path(os.path.expanduser("~/.local/share/salome-agent"))


def path() -> Path:
    return state_dir() / "server.json"


def log_path() -> str:
    return str(state_dir() / "server.log")


def old_path() -> Path:
    return Path(os.path.expanduser(_OLD_DIR)) / "server.json"


def _pid_start(pid: int) -> int | None:
    try:
        with open(f"/proc/{pid}/stat") as f:
            return int(f.read().rsplit(")", 1)[1].split()[19])
    except (OSError, IndexError, ValueError):
        return None


def write(*, port: int, token: str, mode: str, log_path: str,
          salome_port: int | None = None) -> None:
    p = path()
    p.parent.mkdir(parents=True, exist_ok=True)
    drop_old_dir()
    pid = os.getpid()
    data = {
        "proto": PROTO, "host": "127.0.0.1", "port": port, "token": token,
        "pid": pid, "pid_start": _pid_start(pid), "mode": mode,
        "salome_port": salome_port, "log_path": log_path,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    tmp = p.with_suffix(".json.tmp")
    tmp.unlink(missing_ok=True)
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(json.dumps(data, indent=2))
    os.replace(tmp, p)


def live_owner(p: Path | None = None) -> dict | None:
    """The handshake's server if its pid is still alive and is the same
    process that wrote it (pid_start guards against pid reuse). Else None --
    stale file, dead pid, or a recycled pid.
    """
    data = read(p)
    if data is None:
        return None
    pid = data.get("pid")
    if not isinstance(pid, int) or pid == os.getpid():
        return None
    live = _pid_start(pid)
    if live is None:
        return None                       # pid not running
    stored = data.get("pid_start")
    if stored is not None and stored != live:
        return None                       # pid was reused by an unrelated process
    return data


def read(p: Path | None = None) -> dict | None:
    try:
        data = json.loads((p or path()).read_text())
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


def drop_old_dir() -> None:
    """Remove what older versions left in ~/.salome_agent, unless a server
    started by an older install still runs from there."""
    old = old_path().parent
    if not old.is_dir() or live_owner(old_path()) is not None:
        return
    for name in _OLD_FILES:
        (old / name).unlink(missing_ok=True)
    try:
        old.rmdir()
    except OSError:
        pass
