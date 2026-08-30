import logging
import sys
import time

log = logging.getLogger("salome_agent")

_RESET = "\033[0m"
_DIM = "\033[2m"
_LEVEL_COLOR = {
    "DEBUG": "\033[2m",       # dim
    "INFO": "\033[36m",       # cyan
    "WARNING": "\033[33m",    # yellow
    "ERROR": "\033[31m",      # red
    "CRITICAL": "\033[1;31m",
}
_LEVEL_TAG = {
    "DEBUG": "",
    "INFO": "",
    "WARNING": "! ",
    "ERROR": "x ",
    "CRITICAL": "x ",
}


class _PrettyFormatter(logging.Formatter):
    def __init__(self, color: bool):
        super().__init__()
        self._start = time.time()
        self._color = color

    def format(self, record):
        elapsed = time.time() - self._start
        stamp = f"[{elapsed:5.1f}s]"
        tag = _LEVEL_TAG.get(record.levelname, "")
        msg = record.getMessage()
        if self._color:
            stamp = f"{_DIM}{stamp}{_RESET}"
            col = _LEVEL_COLOR.get(record.levelname)
            if col:
                msg = f"{col}{msg}{_RESET}"
        return f"{stamp} {tag}{msg}"


def setup_logging(verbose: bool = False):
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(_PrettyFormatter(color=sys.stderr.isatty()))
    log.handlers[:] = [handler]
    log.setLevel(logging.DEBUG if verbose else logging.INFO)
    log.propagate = False
