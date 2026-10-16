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
    p.add_argument("--allow-run", action="store_true",
                   help="answer the run command, which executes any code sent "
                        "with the token (live tests only)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])

    # Before redirecting stdio to the log: a second server would clobber the
    # first's handshake and leave it orphaned (issue #5). Bail while the user
    # can still see why on the terminal.
    owner = handshake.live_owner()
    if owner is not None:
        print(f"agent server already running (pid {owner['pid']}, "
              f"{owner.get('mode')} mode on port {owner.get('port')}); "
              f"stop it first", file=sys.stderr)
        return 2

    log_path = os.path.expanduser("~/.salome_agent/server.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logf = open(log_path, "w")
    os.dup2(logf.fileno(), sys.stdout.fileno())
    os.dup2(logf.fileno(), sys.stderr.fileno())

    from agent_server.backend import HeadlessBackend

    backend = HeadlessBackend()
    token = secrets.token_hex(16)
    srv = Server(Core(backend, allow_run=args.allow_run), token)
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
