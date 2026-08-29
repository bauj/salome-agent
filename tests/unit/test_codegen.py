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
