# Copyright (C) 2026  salome-agent authors
# LGPL 2.1 or later
from agent_server import handshake
from salome_agent.executor_client import (ExecutorClient, ServerBusy,
                                          SessionDied, SessionPoisoned)

_HELP = ("Start it from Tools > Plugins > SALOME Agent, "
         "or run scripts/start-headless-server.sh")


class ServerUnavailable(Exception):
    pass


def read_handshake() -> dict | None:
    return handshake.read()


def _unavailable_text() -> str:
    if handshake.live_owner(handshake.old_path()) is not None:
        return (f"No SALOME agent server running from this version; one from an older "
                f"install still writes {handshake.old_path()}. Restart SALOME from the "
                f"updated install, or stop that server. {_HELP}")
    return f"No SALOME agent server running. {_HELP}"


def connect(run_id: str, record_path: str | None = None, hold: bool = False,
            run_dir: str | None = None) -> ExecutorClient:
    for attempt in (1, 2):
        hs = read_handshake()
        if hs is None:
            raise ServerUnavailable(_unavailable_text())
        try:
            return ExecutorClient(hs["host"], hs["port"], hs["token"], run_id,
                                  record_path=record_path, hold=hold, run_dir=run_dir)
        except ServerBusy as e:
            raise ServerUnavailable(f"{e}. {_HELP}") from e
        except SessionPoisoned as e:
            raise ServerUnavailable(
                f"The SALOME agent server is poisoned ({e}); restart it. {_HELP}") from e
        except (ConnectionError, SessionDied, OSError) as e:
            if attempt == 2:
                raise ServerUnavailable(
                    f"Could not connect to the SALOME agent server ({e}). {_HELP}") from e
            # fall through: re-read the handshake (token may have rotated), retry
    raise ServerUnavailable(f"Could not connect to the SALOME agent server. {_HELP}")
