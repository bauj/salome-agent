import ast

from salome_agent.codegen import generate


def test_drops_unreachable_call_keeps_transitive_dependency():
    call_log = [
        {"fn": "MakeBoxDXDYDZ", "args": [{"kind": "literal", "value": 10}] * 3,
         "out_vars": ["box_1"], "publish": ["box_1"]},
        {"fn": "MakeFilletEdges",
         "args": [{"kind": "ref", "value": "box_1"}, {"kind": "literal", "value": 2},
                   {"kind": "literal", "value": [1, 2]}],
         "out_vars": ["fillet_1"], "publish": ["fillet_1"]},
        # An abandoned exploratory call whose result finish_stage never named —
        # must be dropped even though it's a valid earlier call.
        {"fn": "MakeCylinderRH", "args": [{"kind": "literal", "value": 5}, {"kind": "literal", "value": 5}],
         "out_vars": ["cyl_1"], "publish": ["cyl_1"]},
    ]
    source = generate(call_log, result_names=["fillet_1"])
    ast.parse(source)  # syntactically valid
    assert "box_1 = geompy.MakeBoxDXDYDZ(10, 10, 10)" in source
    assert "fillet_1 = geompy.MakeFilletEdges(box_1, 2, [1, 2])" in source
    assert "cyl_1" not in source  # unreachable from result_names, pruned


def test_list_kind_arg_resolves_to_python_list_of_refs():
    call_log = [
        {"fn": "MakeBoxDXDYDZ", "args": [{"kind": "literal", "value": 1}] * 3,
         "out_vars": ["box_1"], "publish": ["box_1"]},
        {"fn": "MakeBoxDXDYDZ", "args": [{"kind": "literal", "value": 2}] * 3,
         "out_vars": ["box_2"], "publish": ["box_2"]},
        {"fn": "MakeFuseList",
         "args": [{"kind": "list", "value": [{"kind": "ref", "value": "box_1"},
                                              {"kind": "ref", "value": "box_2"}]}],
         "out_vars": ["fused_1"], "publish": ["fused_1"]},
    ]
    source = generate(call_log, result_names=["fused_1"])
    assert "fused_1 = geompy.MakeFuseList([box_1, box_2])" in source


def test_study_kind_arg_resolves_via_find_object():
    call_log = [
        {"fn": "MakeFilletEdges",
         "args": [{"kind": "study", "value": "stage0_box"}, {"kind": "literal", "value": 2},
                   {"kind": "literal", "value": "all"}],
         "out_vars": ["fillet_1"], "publish": ["fillet_1"]},
    ]
    source = generate(call_log, result_names=["fillet_1"])
    assert 'find_object("stage0_box")' in source


def test_method_call_emit_invokes_method_no_assign():
    """method_call emit generates receiver.fn(args) with no assignment."""
    call_log = [
        {"fn": "MakeBoxDXDYDZ", "args": [{"kind": "literal", "value": 10}] * 3,
         "out_vars": ["box_1"], "publish": ["box_1"]},
        {"fn": "SetColor",
         "args": [{"kind": "salomeds_color", "value": [255, 0, 0]}],
         "out_vars": [], "publish": None,
         "emit": "method_call", "recv": "box_1"},
    ]
    source = generate(call_log, result_names=["box_1"])
    assert "box_1.SetColor(SALOMEDS.Color(1.0000, 0.0000, 0.0000))" in source
    # No assignment for method_call
    assert "_ = box_1.SetColor" not in source


def test_gui_call_emit_invokes_gg_in_try_except():
    """gui_call emit generates gg.fn wrapped in try/except."""
    call_log = [
        {"fn": "MakeBoxDXDYDZ", "args": [{"kind": "literal", "value": 10}] * 3,
         "out_vars": ["box_1"], "publish": ["box_1"]},
        {"fn": "setTransparency",
         "args": [{"kind": "literal", "value": 0.5}],
         "out_vars": [], "publish": None,
         "emit": "gui_call", "recv": "box_1"},
    ]
    source = generate(call_log, result_names=["box_1"])
    assert "try:" in source
    assert "gg.setTransparency(salome.ObjectToID(box_1), 0.5)" in source
    assert "except Exception:" in source
    assert "pass" in source


def test_salomeds_color_arg_formats_correctly():
    """salomeds_color kind arg formats to SALOMEDS.Color(r/255, g/255, b/255)."""
    call_log = [
        {"fn": "MakeBoxDXDYDZ", "args": [{"kind": "literal", "value": 10}] * 3,
         "out_vars": ["box_1"], "publish": ["box_1"]},
        {"fn": "SetColor",
         "args": [{"kind": "salomeds_color", "value": [128, 64, 32]}],
         "out_vars": [], "publish": None,
         "emit": "method_call", "recv": "box_1"},
    ]
    source = generate(call_log, result_names=["box_1"])
    # Check that color components are divided by 255 with 4 decimal places
    assert "SALOMEDS.Color(0.5020, 0.2510, 0.1255)" in source


def test_display_record_dropped_when_recv_unreachable():
    """Display records (no out_vars) are dropped if their recv is unreachable."""
    call_log = [
        {"fn": "MakeBoxDXDYDZ", "args": [{"kind": "literal", "value": 10}] * 3,
         "out_vars": ["box_1"], "publish": ["box_1"]},
        {"fn": "MakeCylinderRH", "args": [{"kind": "literal", "value": 5}, {"kind": "literal", "value": 10}],
         "out_vars": ["cyl_1"], "publish": ["cyl_1"]},
        {"fn": "SetColor",
         "args": [{"kind": "salomeds_color", "value": [255, 0, 0]}],
         "out_vars": [], "publish": None,
         "emit": "method_call", "recv": "cyl_1"},
    ]
    # Only box_1 is in result_names, so cyl_1 and its display record are dropped
    source = generate(call_log, result_names=["box_1"])
    assert "cyl_1.SetColor" not in source


def test_display_record_kept_when_recv_reachable():
    """Display records are kept if their recv is reachable."""
    call_log = [
        {"fn": "MakeBoxDXDYDZ", "args": [{"kind": "literal", "value": 10}] * 3,
         "out_vars": ["box_1"], "publish": ["box_1"]},
        {"fn": "SetColor",
         "args": [{"kind": "salomeds_color", "value": [255, 0, 0]}],
         "out_vars": [], "publish": None,
         "emit": "method_call", "recv": "box_1"},
    ]
    source = generate(call_log, result_names=["box_1"])
    assert "box_1.SetColor" in source
