import json
import os

from salome_agent.executor_client import (ExecutorClient, ServerBusy,
                                          SessionDied, SessionPoisoned)

_HELP = ("Start it from Tools > Plugins > Launch SALOME agent server, "
         "or run scripts/start-headless-server.sh")


class ServerUnavailable(Exception):
    pass


def read_handshake() -> dict | None:
    try:
        with open(os.path.expanduser("~/.salome_agent/server.json")) as f:
            data = json.loads(f.read())
    except (OSError, ValueError):
        return None
    if data.get("proto") != 1:
        return None
    return data


def connect(run_id: str, record_path: str | None = None) -> ExecutorClient:
    for attempt in (1, 2):
        hs = read_handshake()
        if hs is None:
            raise ServerUnavailable(f"No SALOME agent server running. {_HELP}")
        try:
            return ExecutorClient(hs["host"], hs["port"], hs["token"], run_id,
                                  record_path=record_path)
        except ServerBusy as e:
            raise ServerUnavailable(f"{e}. {_HELP}") from e
        except SessionPoisoned as e:
            raise ServerUnavailable(
                f"The SALOME agent server is poisoned ({e}); restart it. {_HELP}") from e
        except (ConnectionError, SessionDied, OSError, AssertionError):
            if attempt == 2:
                raise ServerUnavailable(
                    f"Could not connect to the SALOME agent server. {_HELP}")
            # fall through: re-read the handshake (token may have rotated), retry
    raise ServerUnavailable(f"Could not connect to the SALOME agent server. {_HELP}")
