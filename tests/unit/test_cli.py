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


def test_verbose_counts():
    assert parse_args(["x"]).verbose == 0
    assert parse_args(["x", "-v"]).verbose == 1
    assert parse_args(["x", "-vv"]).verbose == 2


def test_tools_flag():
    assert parse_args(["x"]).tools is None
    assert parse_args(["x", "--tools", "all"]).tools == "all"
    assert parse_args(["x", "--tools", "hints"]).tools == "hints"


def test_no_live_gui_or_port():
    args = parse_args(["make a box"])
    assert not hasattr(args, "live_gui")
    assert not hasattr(args, "port")
