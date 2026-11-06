# Copyright (C) 2026  salome-agent authors
# LGPL 2.1 or later
import os
import signal
import time

import pytest

from agent_server import handshake
from agent_server.__main__ import main, parse_args


def test_once_is_the_default():
    assert parse_args([]).once is True


def test_serve_forever_flag():
    assert parse_args(["--serve-forever"]).once is False


def test_main_refuses_when_another_server_is_live(monkeypatch, capsys):
    monkeypatch.setattr(handshake, "live_owner",
                        lambda: {"pid": 4242, "mode": "gui", "port": 9})
    rc = main([])
    assert rc == 2
    assert "already running" in capsys.readouterr().err


@pytest.mark.parametrize("sig", [signal.SIGINT, signal.SIGTERM, signal.SIGHUP],
                         ids=lambda s: s.name)
def test_stop_on_signals_raises_keyboard_interrupt(sig):
    from agent_server.__main__ import _stop_on_signals

    saved = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
    try:
        _stop_on_signals()
        with pytest.raises(KeyboardInterrupt):
            os.kill(os.getpid(), sig)
            time.sleep(1)
    finally:
        for s, handler in saved.items():
            signal.signal(s, handler)
