import argparse
import os
import secrets
import sys

from agent_server import handshake
from agent_server.core import Core
from agent_server.wire import Server


def parse_args(argv):
    p = argparse.ArgumentParser(prog="agent_server")
    p.add_argument("--serve-forever", dest="once", action="store_false",
                   default=True,
                   help="keep serving after the first client disconnects "
                        "(accepts the abandoned-watchdog hazard)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    log_path = os.path.expanduser("~/.salome_agent/server.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logf = open(log_path, "w")
    os.dup2(logf.fileno(), sys.stdout.fileno())
    os.dup2(logf.fileno(), sys.stderr.fileno())

    from agent_server.backend import HeadlessBackend

    backend = HeadlessBackend()
    token = secrets.token_hex(16)
    srv = Server(Core(backend), token)
    port = srv.bind()
    handshake.write(port=port, token=token, mode="headless", log_path=log_path)
    try:
        srv.serve_forever_once() if args.once else srv.serve_forever()
    finally:
        srv.stop()
        handshake.remove()
    return 1 if backend.poisoned() else 0


if __name__ == "__main__":
    sys.exit(main())
