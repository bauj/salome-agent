import hmac
import json
import socket
import threading
import time


class Server:
    PRE_HELLO_TIMEOUT = 5.0
    BUSY_IDLE_S = 30.0
    # A client that just closed may not be seen as gone yet: a hello waits this
    # long for the active connection to end before it is answered busy.
    CLAIM_WAIT_S = 1.0

    def __init__(self, core, token: str):
        self.core = core
        self.token = token
        self._srv: socket.socket | None = None
        self._stop = threading.Event()
        self._conn_lock = threading.Condition()
        self._active = False
        self._held = False
        self._last_activity = 0.0

    def bind(self) -> int:
        self._srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._srv.bind(("127.0.0.1", 0))
        self._srv.listen(8)
        self._srv.settimeout(0.5)
        return self._srv.getsockname()[1]

    def serve_forever(self) -> None:
        while not self._stop.is_set():
            # A timeout poisons the backend and abandons a running thread; the
            # process must exit so that thread dies and clients stop getting a
            # silent _FATAL from a still-listening but broken server. `_accept`
            # times out every 0.5s so this is noticed with no traffic.
            if self.core.poisoned():
                break
            conn = self._accept()
            if conn is None:
                continue
            threading.Thread(target=self._handle_conn, args=(conn,),
                             daemon=True).start()
        self._close_listener()

    def serve_forever_once(self) -> None:
        while not self._stop.is_set():
            conn = self._accept()
            if conn is None:
                continue
            self._handle_conn(conn)
            break
        self._close_listener()

    def stop(self) -> None:
        self._stop.set()

    def _accept(self):
        try:
            conn, _ = self._srv.accept()
            return conn
        except socket.timeout:
            return None
        except OSError:
            self._stop.set()
            return None

    def _close_listener(self) -> None:
        if self._srv is not None:
            try:
                self._srv.close()
            except OSError:
                pass

    def _claim(self, hold: bool) -> bool:
        def taken():
            return self._active and (self._held or
                                     (time.monotonic() - self._last_activity) < self.BUSY_IDLE_S)
        with self._conn_lock:
            if not self._conn_lock.wait_for(lambda: not taken(), self.CLAIM_WAIT_S):
                return False
            self._active = True
            self._held = hold
            self._last_activity = time.monotonic()
            return True

    def _handle_conn(self, conn) -> None:
        conn.settimeout(self.PRE_HELLO_TIMEOUT)
        f = conn.makefile("rwb")
        claimed = False
        try:
            first = f.readline()
            if not first:
                return
            req = json.loads(first)
            if not hmac.compare_digest(str(req.get("token", "")), self.token):
                return                       # drop, no response
            if not self._claim(bool(req.get("hold"))):
                f.write(b'{"ready": false, "error": "busy"}\n')
                f.flush()
                return
            claimed = True
            conn.settimeout(None)
            f.write((json.dumps(self.core.handle(req), default=str) + "\n").encode())
            f.flush()
            for line in f:
                with self._conn_lock:
                    self._last_activity = time.monotonic()
                parsed = json.loads(line)
                f.write((json.dumps(self.core.handle(parsed), default=str) + "\n").encode())
                f.flush()
                if parsed.get("cmd") == "shutdown":
                    break
        except (socket.timeout, ValueError, OSError):
            pass
        finally:
            if claimed:
                with self._conn_lock:
                    self._active = False
                    self._held = False
                    self._conn_lock.notify_all()
            try:
                conn.close()
            except OSError:
                pass
