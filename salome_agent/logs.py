import logging
import sys
import time

log = logging.getLogger("salome_agent")


class _ElapsedFormatter(logging.Formatter):
    def __init__(self):
        super().__init__("%(elapsed)6.1fs  %(message)s")
        self._start = time.time()

    def format(self, record):
        record.elapsed = time.time() - self._start
        return super().format(record)


def setup_logging(verbose: bool = False):
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(_ElapsedFormatter())
    log.handlers[:] = [handler]
    log.setLevel(logging.DEBUG if verbose else logging.INFO)
    log.propagate = False
