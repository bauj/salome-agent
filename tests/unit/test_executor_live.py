from unittest.mock import MagicMock

from executor.executor import Executor


def _bare(live):
    ex = Executor.__new__(Executor)
    ex.live = live
    ex.namespace = {}
    return ex


def test_emit_refresh_noop_when_not_live():
    _bare(False)._emit_refresh("GEOM")  # must not raise, must not touch namespace


def test_emit_refresh_swallows_errors_when_live():
    ex = _bare(True)
    ex.namespace = {"salome": MagicMock(name="salome-without-real-corba")}
    # SALOME_CMOD import will fail in the unit env; _emit_refresh must swallow it.
    ex._emit_refresh("GEOM")
