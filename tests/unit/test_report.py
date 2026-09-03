from salome_agent.nodes.report import write_report


def _state(**over):
    base = {"run_id": "r1", "user_request": "make a box", "stages": {},
            "session": {"mode": "gui", "name_prefix": "sa_r1_"}, "fatal": None}
    base.update(over)
    return base


def test_report_notes_gui_debris():
    out = write_report(_state())
    assert "sa_" in out and ("GUI" in out or "live" in out.lower())


def test_report_renders_fatal():
    out = write_report(_state(fatal="server poisoned by a timeout"))
    assert "poisoned" in out


def test_report_headless_line():
    out = write_report(_state(session={"mode": "headless"}))
    assert "isolated study" in out


def test_report_includes_llm_usage():
    usage = {"calls": 12, "metered_calls": 12,
             "prompt_tokens": 98000, "completion_tokens": 4200}
    out = write_report(_state(), usage=usage)
    assert "## LLM usage" in out
    assert "calls: 12" in out
    assert "98,000" in out
    assert "102,200" in out


def test_report_flags_calls_without_usage_data():
    usage = {"calls": 10, "metered_calls": 7,
             "prompt_tokens": 5000, "completion_tokens": 500}
    out = write_report(_state(), usage=usage)
    assert "3 call(s) returned no usage data" in out


def test_report_omits_usage_section_when_absent():
    assert "## LLM usage" not in write_report(_state())
    assert "## LLM usage" not in write_report(_state(), usage={"calls": 0})
