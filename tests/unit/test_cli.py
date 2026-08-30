import pytest

from salome_agent.cli import parse_args


def test_parses_batch_and_no_session():
    args = parse_args(["make a box", "--batch", "--no-session"])
    assert args.request == "make a box"
    assert args.batch is True
    assert args.no_session is True


def test_defaults_are_interactive_with_session():
    args = parse_args(["make a box"])
    assert args.batch is False
    assert args.no_session is False
    assert args.live_gui is False


def test_parses_live_gui():
    args = parse_args(["make a box", "--live-gui"])
    assert args.live_gui is True


def test_live_gui_and_no_session_are_mutually_exclusive():
    with pytest.raises(SystemExit):
        parse_args(["make a box", "--live-gui", "--no-session"])
