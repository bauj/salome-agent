from agent_server.__main__ import parse_args


def test_once_is_the_default():
    assert parse_args([]).once is True


def test_serve_forever_flag():
    assert parse_args(["--serve-forever"]).once is False
