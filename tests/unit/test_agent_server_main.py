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
