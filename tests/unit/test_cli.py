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


def test_main_no_session_delegates_to_run_request(tmp_path, monkeypatch, capsys):
    import json

    from salome_agent import cli
    from salome_agent.runner import RunResult

    cfg = tmp_path / "c.json"
    cfg.write_text(json.dumps({"llm": {"base_url": "http://x", "model": "m"},
                               "salome": {"root": "/x"},
                               "policy": {"on_stage_failure": "abort"},
                               "work_dir": str(tmp_path)}))
    seen = {}

    def fake_run(request, config, **kw):
        seen.update(kw, request=request, config=config)
        return RunResult(run_id=kw["run_id"], run_dir=str(tmp_path / kw["run_id"]),
                         final_state={"report": "REPORT-TEXT"}, exit_code=0)

    monkeypatch.setattr("salome_agent.runner.run_request", fake_run)
    rc = cli.main(["make a box", "--no-session", "--batch", "--config", str(cfg),
                   "--on-failure", "continue"])
    assert rc == 0
    assert seen["request"] == "make a box"
    assert seen["client"] is None and seen["batch"] is True
    assert seen["config"]["policy"]["on_stage_failure"] == "continue"
    assert "REPORT-TEXT" in capsys.readouterr().out


def test_main_returns_1_on_abort(tmp_path, monkeypatch):
    import json

    from salome_agent import cli
    from salome_agent.runner import RunResult

    cfg = tmp_path / "c.json"
    cfg.write_text(json.dumps({"llm": {"base_url": "http://x", "model": "m"},
                               "salome": {"root": "/x"}, "policy": {}}))
    monkeypatch.setattr(
        "salome_agent.runner.run_request",
        lambda request, config, **kw: RunResult(kw["run_id"], "/x", {"aborted": True}, 1))
    assert cli.main(["x", "--no-session", "--config", str(cfg)]) == 1


def test_chat_with_batch_is_an_error(capsys):
    from salome_agent.cli import main
    assert main(["a box", "--chat", "--batch"]) == 2


def test_chat_with_no_session_is_an_error(capsys):
    from salome_agent.cli import main
    assert main(["a box", "--chat", "--no-session"]) == 2
