import os
import subprocess
import time
import uuid

from salome_agent.executor_client import ExecutorClient
from salome_agent.session.discovery import discover_port
from salome_agent.state import SessionHandle


def _salome_subprocess_env() -> dict:
    # Two things confirmed by hands-on testing against a live SALOME
    # install (2026-08-29, see executor/executor.py's test fixture):
    # (1) subprocess.Popen's `env=` REPLACES the environment wholesale,
    # never merges — passing a partial dict here would strip PATH/HOME/
    # everything `salome shell` itself needs to run at all, not just add
    # SALOME_PORT. Always start from a full os.environ.copy().
    # (2) if this process's own venv is active, bare "python3" on PATH
    # resolves to the venv's python (missing psutil, no SALOME sys.path)
    # since venv/bin is prepended — must strip it out here.
    env = os.environ.copy()
    env["PATH"] = os.pathsep.join(
        p for p in env.get("PATH", "").split(os.pathsep) if ".venv" not in p
    )
    env.pop("VIRTUAL_ENV", None)
    return env


class SessionManager:
    def __init__(self, salome_root: str, socket_dir: str = "/tmp",
                 executor_log: str | None = None, gui_log: str | None = None):
        self.salome_bin = f"{salome_root}/salome"
        self.socket_dir = socket_dir
        self.executor_log = executor_log
        self.gui_log = gui_log
        self._proc: subprocess.Popen | None = None
        self._gui_proc: subprocess.Popen | None = None
        self._socket_path: str | None = None
        self._logfile = None
        self._gui_logfile = None

    def resolve(self, no_session: bool, port: int | None,
                live_gui: bool = False) -> tuple[SessionHandle, ExecutorClient | None]:
        if no_session:
            return {"mode": "none", "port": None, "executor_pid": None,
                    "alive": False, "owned": False}, None

        if live_gui:
            return self._resolve_live(port)

        resolved_port = port or discover_port()
        # A fixed socket path collides across concurrent/crashed runs (a
        # leftover file from a prior process makes Popen's bind fail with
        # "Address already in use") — one unique path per SessionManager
        # instance instead.
        socket_path = f"{self.socket_dir}/salome_agent_executor_{uuid.uuid4().hex}.sock"
        self._socket_path = socket_path
        env = _salome_subprocess_env()

        out = err = None
        if self.executor_log is not None:
            self._logfile = open(self.executor_log, "w")
            out = err = self._logfile

        if resolved_port is not None:
            env["SALOME_PORT"] = str(resolved_port)
            self._proc = subprocess.Popen(
                [self.salome_bin, "shell", "--", "python3", "executor/executor.py",
                 "--socket", socket_path],
                env=env, stdout=out, stderr=err,
            )
            client = self._wait_ready(socket_path)
            return {"mode": "attach", "port": resolved_port,
                    "executor_pid": self._proc.pid, "alive": True, "owned": False}, client

        self._proc = subprocess.Popen(
            [self.salome_bin, "shell", "--", "python3", "executor/executor.py", "--socket", socket_path],
            env=env, stdout=out, stderr=err,
        )
        client = self._wait_ready(socket_path)
        return {"mode": "launch", "port": None, "executor_pid": self._proc.pid,
                "alive": True, "owned": True}, client

    def _resolve_live(self, port: int | None) -> tuple[SessionHandle, ExecutorClient]:
        resolved_port = port or discover_port()
        owned_gui = False
        if resolved_port is None:
            resolved_port = self._launch_gui()
            owned_gui = True

        socket_path = f"{self.socket_dir}/salome_agent_executor_{uuid.uuid4().hex}.sock"
        self._socket_path = socket_path
        env = _salome_subprocess_env()

        out = err = None
        if self.executor_log is not None:
            self._logfile = open(self.executor_log, "w")
            out = err = self._logfile

        # `salome remote` (not `salome shell`) is the only launcher that
        # attaches to a live multi-process GUI session; the executor then
        # calls KernelBasis.setSSLMode(False) under --live so geomBuilder.New()
        # returns the GUI's own engine instead of a private one.
        self._proc = subprocess.Popen(
            [self.salome_bin, "remote", "-p", str(resolved_port), "--",
             "python3", "executor/executor.py", "--socket", socket_path, "--live"],
            env=env, stdout=out, stderr=err,
        )
        # Longer than the default: under --live the executor may retry
        # salome_init() for up to ~30s while a freshly launched GUI's
        # ModuleCatalog / Container servers finish registering.
        client = self._wait_ready(socket_path, timeout=90)
        return {"mode": "live", "port": resolved_port, "executor_pid": self._proc.pid,
                "alive": True, "owned": owned_gui}, client

    def _launch_gui(self, timeout: float = 120) -> int:
        env = _salome_subprocess_env()
        # SALOME's Qt UI does not run under a native Wayland session on this
        # box; the user sets this by hand. Match that when we launch for them.
        if os.environ.get("WAYLAND_DISPLAY") and not env.get("QT_QPA_PLATFORM"):
            env["QT_QPA_PLATFORM"] = "xcb"

        out = err = subprocess.DEVNULL
        if self.gui_log is not None:
            self._gui_logfile = open(self.gui_log, "w")
            out = err = self._gui_logfile

        # `withsession` (not the default single-process `start`) is the only
        # mode that spawns real CORBA servers an external client can attach
        # to. Detached (start_new_session) so it outlives this run — the user
        # keeps the window and its objects after the agent exits.
        self._gui_proc = subprocess.Popen(
            [self.salome_bin, "withsession", "-g"],
            env=env, stdout=out, stderr=err, start_new_session=True,
        )
        deadline = time.time() + timeout
        while time.time() < deadline:
            p = discover_port()
            if p is not None:
                return p
            if self._gui_proc.poll() is not None:
                raise RuntimeError(
                    f"`salome withsession` exited (code {self._gui_proc.returncode}) "
                    f"before a session came up"
                    + (f"; see {self.gui_log}" if self.gui_log else ""))
            time.sleep(1.0)
        raise TimeoutError(f"SALOME GUI did not come up within {timeout:.0f}s")

    def _wait_ready(self, socket_path: str, timeout: float = 30) -> ExecutorClient:
        deadline = time.time() + timeout
        last_err = None
        while time.time() < deadline:
            try:
                return ExecutorClient(socket_path)
            except (FileNotFoundError, ConnectionRefusedError) as e:
                last_err = e
            if self._proc is not None and self._proc.poll() is not None:
                hint = f"; see {self.executor_log}" if self.executor_log else ""
                raise RuntimeError(
                    f"executor exited (code {self._proc.returncode}) before it was "
                    f"ready{hint}")
            time.sleep(0.2)
        raise TimeoutError(f"executor did not become ready: {last_err}")

    def stop(self):
        if self._proc is not None:
            self._proc.terminate()
            self._proc.wait(timeout=10)
        if self._socket_path is not None:
            try:
                os.unlink(self._socket_path)
            except FileNotFoundError:
                pass
        if self._logfile is not None:
            self._logfile.close()
            self._logfile = None
        # A GUI we launched is deliberately left running (detached) so the
        # user keeps the window; only release our file handle to its log.
        if self._gui_logfile is not None:
            self._gui_logfile.close()
            self._gui_logfile = None
