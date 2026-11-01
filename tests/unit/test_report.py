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
    out = write_report(_state(session={"mode": "headless", "name_prefix": "sa_r1_"}))
    assert "isolated study" not in out
    assert "`sa_r1_`" in out


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


def test_report_estimates_output_of_incomplete_calls():
    usage = {"calls": 3, "metered_calls": 2, "prompt_tokens": 5000,
             "completion_tokens": 500, "incomplete_calls": 1,
             "approx_incomplete_output_tokens": 20123}
    out = write_report(_state(), usage=usage)
    assert "incomplete calls (cut, cancelled or dropped): 1" in out
    assert "20,123" in out
    assert "incomplete" not in write_report(_state(), usage={**usage, "incomplete_calls": 0})


def test_report_omits_usage_section_when_absent():
    assert "## LLM usage" not in write_report(_state())
    assert "## LLM usage" not in write_report(_state(), usage={"calls": 0})


def _timed_state():
    return _state(
        plan_timing={"wall": 12.0, "llm_s": 11.5, "core_s": 0.5, "turns": 1,
                     "in_tok": 2000, "out_tok": 400},
        stages={
            0: {"module": "geom", "status": "ok",
                "timing": {"wall": 40.0, "llm_s": 30.0, "core_s": 2.0, "exec_s": 8.0,
                           "turns": 5, "in_tok": 16000, "out_tok": 2100}},
        })


def test_report_stage_bullet_carries_timing():
    out = write_report(_timed_state())
    assert "5 turns" in out
    assert "llm 30.0s" in out
    assert "exec 8.0s" in out
    assert "16.0k in / 2.1k out" in out


def test_report_timing_section():
    out = write_report(_timed_state())
    assert "## Timing" in out
    assert "planner: 12.0s wall" in out
    assert "stage 1/1 (geom): 40.0s wall" in out
    # total wall 52, llm 41.5 -> ~80%
    assert "52.0s measured wall" in out and "80%" in out


def test_report_omits_timing_when_no_timing_data():
    assert "## Timing" not in write_report(_state())


def test_report_says_an_assumed_2d_target_was_built_flat():
    spec = {"dimensionality": {"value": "2d", "assumed": True}, "parts": [{"id": "P1"}]}
    out = write_report(_state(image_spec=spec, plan=[]))
    assert "## Reference image" in out
    assert 'Built as a flat face: the image gave no thickness -- say "extrude N mm" to make it solid.' in out


def test_report_lists_unclaimed_parts_and_conflicts():
    spec = {"dimensionality": {"value": "3d"},
            "parts": [{"id": "P1", "conflicts": ["arc off radius"]}, {"id": "P2"}],
            "conflicts": ["datum unclear"]}
    out = write_report(_state(image_spec=spec, plan=[{"module": "geom", "parts": ["P1"]}]))
    assert "Parts in the image spec no stage built: P2." in out
    assert "- datum unclear" in out and "- P1: arc off radius" in out


def test_report_has_no_image_section_when_nothing_to_say():
    spec = {"dimensionality": {"value": "3d"}, "parts": [{"id": "P1"}]}
    assert "## Reference image" not in write_report(_state(image_spec=spec, plan=[]))
    assert "## Reference image" not in write_report(_state())


def test_report_names_each_endpoint_and_model():
    out = write_report(_state(), endpoints=[(None, "https://api.x/v1", "small"),
                                            ("vision", "https://other/v1", "big")])
    assert "## LLM endpoint" in out
    assert "- shared: `small` at https://api.x/v1" in out
    assert "- vision: `big` at https://other/v1" in out


def test_report_warnings_and_built_lines():
    out = write_report(_state(stages={
        0: {"module": "shaper", "status": "ok", "built": "flange 120 mm"},
        1: {"module": "smesh", "status": "ok", "built": "body-fitted mesh, 2 mm",
            "request_conflict": "kept your 2 mm step; the volume check fails",
            "warnings": ["mesh_2: its volume 140550.7 is 4.10% off the shape's 146562.4"]},
        2: {"module": "smesh", "status": "failed", "request_conflict": "not shown"}}))
    assert "## Warnings" in out
    assert "- stage 2/3 (smesh): kept your 2 mm step; the volume check fails" in out
    assert "- stage 2/3 (smesh) missed a check: mesh_2: its volume 140550.7 is 4.10% off" in out
    assert "not shown" not in out
    assert out.index("## Warnings") < out.index("## Stages")
    assert "  - built: body-fitted mesh, 2 mm" in out


def test_report_has_no_warnings_section_without_conflicts():
    assert "## Warnings" not in write_report(_timed_state())


def test_report_timing_section_on_an_amend_turn():
    out = write_report(_state(
        turn_idx=1, stages={},
        amend_timing={"wall": 20.0, "llm_s": 15.0, "exec_s": 3.0, "core_s": 2.0, "turns": 3,
                      "in_tok": 9000, "out_tok": 700}))
    assert "## Timing" in out
    assert ("- amend (turn 2): 20.0s wall, 15.0s llm, 3.0s exec, 3 calls, "
            "tokens 9.0k in / 700 out") in out
    assert "20.0s measured wall — 15.0s llm (75%), 3.0s exec" in out


def test_report_timing_totals_an_amend_that_replanned():
    out = write_report(_timed_state() | {
        "turn_idx": 1,
        "amend_timing": {"wall": 8.0, "llm_s": 6.0, "core_s": 2.0, "turns": 1,
                         "in_tok": 1000, "out_tok": 100}})
    lines = out.splitlines()
    assert lines.index("- amend (turn 2): 8.0s wall, 6.0s llm, 0.0s exec, 1 calls, "
                       "tokens 1.0k in / 100 out") \
        < next(i for i, l in enumerate(lines) if l.startswith("- planner:"))
    assert "60.0s measured wall" in out
