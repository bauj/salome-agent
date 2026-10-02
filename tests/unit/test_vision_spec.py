import copy
import json

from salome_agent.nodes.vision_spec import (all_conflicts, amend_block, attach_conflicts,
                                            compact_spec, format_errors, parse_num, parse_point,
                                            part_ids, parts_block, stage_part_ids,
                                            unclaimed_parts, validate_spec)


def p1_spec():
    """The p1.png plate as a correct spec (clean benchmark: area 19380, 8 wires)."""
    return {
        "dimensionality": {"value": "2d", "assumed": False, "reason": "flat plate, request says 2D"},
        "summary": "flat plate: round top lobe over a narrower body with a round bottom end",
        "views": [{"image": 1, "view": "front", "shows": "full outline"}],
        "dimensions_read": [{"text": "R96", "means": "top lobe radius, centre O"}],
        "datum": "O = centre of top lobe; +X right, +Y up",
        "parts": [{
            "id": "P1", "name": "plate",
            "base": {"kind": "profile", "plane": "XY", "extrude": None, "revolve": None,
                     "profile": {
                         "start": "96,0,0",
                         "segments": [
                             {"to": "-96,0,0", "through": "0,96,0"},
                             {"to": "-54,0,0", "fillet": 20},
                             {"to": "-54,-80,0"},
                             {"to": "54,-80,0", "through": "0,-134,0"},
                             {"to": "54,0,0", "fillet": 20},
                             {"to": "96,0,0"}],
                         "holes": [
                             {"center": "0,0,0", "radius": 25},
                             {"center": "55.426,32,0", "radius": 15},
                             {"center": "0,64,0", "radius": 15},
                             {"center": "-55.426,32,0", "radius": 15},
                             {"center": "0,-80,0", "radius": 20},
                             {"slot": {"center": "0,-80,0", "radius": 35, "width": 18,
                                       "from_angle": -60, "to_angle": 60}},
                             {"slot": {"center": "0,-80,0", "radius": 35, "width": 18,
                                       "from_angle": 120, "to_angle": 240}}]}},
            "placement": {"origin": "0,0,0", "normal": "0,0,1"},
            "features": [],
            "relation": None}],
        "uncertainties": []}


def box_spec():
    return {
        "dimensionality": {"value": "3d", "assumed": False, "reason": "photo"},
        "summary": "a block with a hole", "views": [], "dimensions_read": [], "datum": "corner",
        "parts": [{"id": "P1", "name": "block",
                   "base": {"kind": "primitive", "shape": "box",
                            "dims": {"dx": "~40", "dy": "~30", "dz": 10}},
                   "placement": {"origin": "~0,0,0", "normal": "0,0,1"},
                   "features": [{"type": "hole", "center": "20,15,10", "diameter": 8,
                                 "depth": "through"},
                                {"type": "pattern", "of": 1, "kind": "linear", "count": 3}]},
                  {"id": "P2", "name": "lid",
                   "base": {"kind": "approx", "description": "rounded cap", "dims": {"d": "~30"}},
                   "placement": {"origin": "20,15,10", "normal": "0,0,1"},
                   "ops": [{"op": "fuse", "with": "P1"}]}],
        "uncertainties": []}


def msgs(errors):
    return [m for _, m in errors]


def test_parse_num_takes_numbers_and_estimates():
    assert parse_num(3) == 3.0
    assert parse_num("~40") == 40.0
    assert parse_num("~ -2.5") == -2.5
    assert parse_num("forty") is None
    assert parse_num(True) is None


def test_parse_point_takes_estimated_coordinates():
    assert parse_point("1,2,3") == (1.0, 2.0, 3.0)
    assert parse_point("~40, 0, 0") == (40.0, 0.0, 0.0)
    assert parse_point("1,2") is None
    assert parse_point("A") is None


def test_correct_specs_pass():
    assert validate_spec(p1_spec()) == []
    assert validate_spec(box_spec()) == []


def test_non_object_spec_is_one_error():
    assert validate_spec([1, 2]) == [(None, "the spec must be a JSON object")]


def test_missing_parts_and_bad_dimensionality():
    errs = validate_spec({"dimensionality": {"value": "flat"}, "parts": []})
    assert (None, 'dimensionality.value must be "2d" or "3d"') in errs
    assert (None, "parts must be a non-empty list") in errs


def test_duplicate_ids():
    spec = box_spec()
    spec["parts"][1]["id"] = "P1"
    assert ("P1", "duplicate part id") in validate_spec(spec)


def test_arc_end_off_its_radius():
    spec = p1_spec()
    spec["parts"][0]["base"]["profile"]["segments"][3] = {"to": "54,-78,0", "center": "0,-80,0"}
    (pid, msg), = validate_spec(spec)
    assert pid == "P1"
    assert msg.startswith("base.profile segment 4: arc start (-54, -80, 0) is 54 from centre (0, -80, 0)")
    assert "both ends must lie on one radius" in msg


def test_collinear_through_point():
    spec = p1_spec()
    spec["parts"][0]["base"]["profile"]["segments"][2] = {"to": "-54,-80,0", "through": "-54,-40,0"}
    (_, msg), = validate_spec(spec)
    assert "segment 3: through point (-54, -40, 0) is on the line" in msg


def test_degenerate_outline():
    spec = p1_spec()
    spec["parts"][0]["base"]["profile"] = {"start": "0,0,0",
                                           "segments": [{"to": "10,0,0"}, {"to": "20,0,0"}]}
    (_, msg), = validate_spec(spec)
    assert "degenerate" in msg


def test_point_off_the_profile_plane():
    spec = p1_spec()
    spec["parts"][0]["base"]["profile"]["segments"][2] = {"to": "-54,-80,5"}
    assert any("is not in the XY plane" in m for m in msgs(validate_spec(spec)))


def test_hole_outside_the_outline():
    spec = p1_spec()
    spec["parts"][0]["base"]["profile"]["holes"].append({"center": "200,0,0", "radius": 5})
    (_, msg), = validate_spec(spec)
    assert msg.startswith("base.profile.holes[8]: centre (200, 0) lies outside the outline")


def test_bad_estimate_string():
    spec = box_spec()
    spec["parts"][0]["base"]["dims"]["dx"] = "about 40"
    assert ("P1", 'base.dims.dx must be a number or "~N"') in validate_spec(spec)


def test_pattern_must_point_at_another_feature():
    spec = box_spec()
    spec["parts"][0]["features"][1]["of"] = 2
    assert ("P1", "features[2].of must be the 1-based index of another feature (1-2)") \
        in validate_spec(spec)


def test_ops_must_name_another_part():
    spec = box_spec()
    spec["parts"][1]["ops"] = [{"op": "fuse", "with": "P9"}]
    assert ("P2", "ops[1].with must name another part (P1, P2)") in validate_spec(spec)


def test_2d_target_has_no_extrusion_and_no_primitives():
    spec = p1_spec()
    spec["parts"][0]["base"]["extrude"] = {"height": 5}
    assert ("P1", "base: a 2d target has no extrude or revolve") in validate_spec(spec)
    spec = box_spec()
    spec["dimensionality"]["value"] = "2d"
    assert ("P1", "base: a 2d target is built from profile parts only") in validate_spec(spec)


def test_extrude_and_revolve_together():
    spec = p1_spec()
    spec["dimensionality"]["value"] = "3d"
    base = spec["parts"][0]["base"]
    base["extrude"] = {"height": 5}
    base["revolve"] = {"point": "0,0,0", "direction": "0,1,0", "angle": 360}
    assert ("P1", "base: give extrude or revolve, not both") in validate_spec(spec)


def test_spec_hole_cannot_name_a_wire():
    spec = p1_spec()
    spec["parts"][0]["base"]["profile"]["holes"] = [{"wire": "W1"}]
    assert ("P1", "base.profile.holes[1]: a spec hole cannot name a wire; give its outline") \
        in validate_spec(spec)


def test_validator_does_not_mutate_the_spec():
    spec = p1_spec()
    before = copy.deepcopy(spec)
    validate_spec(spec)
    assert spec == before
import json

from salome_agent.nodes.vision_spec import (all_conflicts, amend_block, attach_conflicts,
                                            compact_spec, format_errors, part_ids, parts_block,
                                            stage_part_ids, unclaimed_parts)


def test_format_errors_lists_each_with_its_part():
    assert format_errors([("P1", "bad arc"), (None, "no parts")]) == "- P1: bad arc\n- no parts"


def test_attach_conflicts_puts_errors_on_their_part():
    spec = box_spec()
    out = attach_conflicts(spec, [("P2", "lid too big"), (None, "datum unclear")])
    assert out["parts"][1]["conflicts"] == ["lid too big"]
    assert out["conflicts"] == ["datum unclear"]
    assert "conflicts" not in spec["parts"][1]
    assert all_conflicts(out) == ["datum unclear", "P2: lid too big"]


def test_attach_conflicts_wraps_a_non_object_answer():
    out = attach_conflicts(["a box"], [(None, "the spec must be a JSON object")])
    assert out == {"summary": '["a box"]', "parts": [],
                   "conflicts": ["the spec must be a JSON object"]}
    assert compact_spec(out).startswith("Reference image spec")


def test_compact_spec_summarises_each_part_without_its_numbers():
    text = compact_spec(p1_spec())
    assert text.startswith("Reference image spec")
    assert "Target: 2D in the XY plane" in text
    assert "- P1 plate: profile in XY, flat face, 192 x 230, 7 holes" in text
    assert "55.426" not in text


def test_compact_spec_flags_an_assumed_2d_target():
    spec = p1_spec()
    spec["dimensionality"] = {"value": "2d", "assumed": True, "reason": "single outline"}
    assert "Target: 2D in the XY plane (assumed: single outline)" in compact_spec(spec)


def test_compact_spec_names_primitive_approx_features_and_conflicts():
    spec = attach_conflicts(box_spec(), [("P1", "hole off the face")])
    text = compact_spec(spec)
    assert '- P1 block: box {"dx": "~40", "dy": "~30", "dz": 10}; features: 1 hole, 1 pattern; 1 conflict' in text
    assert '- P2 lid: approx. rounded cap {"d": "~30"}' in text


def test_parts_block_carries_exact_json_of_the_listed_parts_only():
    text = parts_block(box_spec(), ["P2"])
    assert '"id": "P2"' in text and '"id": "P1"' not in text
    assert "Target: 3D" in text
    data = json.loads(text.split("```json\n", 1)[1].split("\n```", 1)[0])
    assert data == [box_spec()["parts"][1]]


def test_parts_block_keeps_estimated_coordinates_verbatim():
    assert '"origin": "~0,0,0"' in parts_block(box_spec(), ["P1"])


def test_amend_block_has_summary_and_all_parts():
    text = amend_block(box_spec())
    assert "a block with a hole" in text
    assert '"id": "P1"' in text and '"id": "P2"' in text


def test_part_ids_reads_a_bare_string():
    assert part_ids({"parts": "P1"}) == ["P1"]
    assert part_ids({"parts": ["P1", "P2"]}) == ["P1", "P2"]
    assert part_ids({}) == []


def test_stage_part_ids_uses_the_stage_list():
    plan = [{"module": "geom", "parts": ["P1"]}, {"module": "geom", "parts": ["P2"]},
            {"module": "geom"}]
    assert stage_part_ids(plan, 1, box_spec()) == ["P2"]
    assert stage_part_ids(plan, 2, box_spec()) == []


def test_stage_part_ids_falls_back_to_every_part_when_no_stage_lists_any():
    plan = [{"module": "geom"}, {"module": "geom"}]
    assert stage_part_ids(plan, 1, box_spec()) == ["P1", "P2"]


def test_unclaimed_parts():
    assert unclaimed_parts([{"module": "geom", "parts": ["P1"]}], box_spec()) == ["P2"]
    assert unclaimed_parts([{"module": "geom"}], box_spec()) == []


def test_holes_that_overlap():
    spec = p1_spec()
    spec["parts"][0]["base"]["profile"]["holes"][6]["slot"]["center"] = "0,-26,0"
    errs = msgs(validate_spec(spec))
    assert any(m.startswith("base.profile.holes[1] and holes[7] overlap by ") for m in errs), errs


def test_hole_crossing_the_outline():
    spec = p1_spec()
    spec["parts"][0]["base"]["profile"]["holes"].append({"center": "85,40,0", "radius": 10})
    (_, msg), = validate_spec(spec)
    assert msg.startswith("base.profile.holes[8]: the hole crosses the outline by ")


def test_hole_inside_the_bounding_box_but_outside_the_outline():
    spec = p1_spec()
    spec["parts"][0]["base"]["profile"]["holes"].append({"center": "80,-50,0", "radius": 5})
    (_, msg), = validate_spec(spec)
    assert msg == "base.profile.holes[8]: centre (80, -50) lies outside the outline"


def test_slot_crossing_the_outline():
    spec = p1_spec()
    spec["parts"][0]["base"]["profile"]["holes"][5]["slot"]["radius"] = 48
    assert any("holes[6]: the hole crosses the outline by " in m for m in msgs(validate_spec(spec)))


def test_tangent_corner_arc_must_be_given_as_a_fillet():
    spec = p1_spec()
    segs = spec["parts"][0]["base"]["profile"]["segments"]
    segs[1:3] = [{"to": "-74,0,0"}, {"to": "-54,-20,0", "center": "-74,-20,0"},
                 {"to": "-54,-80,0"}]
    (_, msg), = validate_spec(spec)
    assert msg == ('base.profile segment 3: this arc rounds the corner at (-54, 0) between two '
                   'straight lines; end the line before it at that corner with "fillet": 20 '
                   'and drop the arc')


def test_arc_that_is_not_tangent_to_its_lines_is_kept():
    spec = p1_spec()
    segs = spec["parts"][0]["base"]["profile"]["segments"]
    segs[1:3] = [{"to": "-74,0,0"}, {"to": "-54,-20,0", "center": "-54,0,0"},
                 {"to": "-54,-80,0"}]
    assert validate_spec(spec) == []


def p1_with_callouts(*texts):
    spec = p1_spec()
    spec["dimensions_read"] += [{"text": t, "means": "callout"} for t in texts]
    return spec


def test_counted_callouts_that_match_the_spec_pass():
    assert validate_spec(p1_with_callouts("Ø30 - 3X", "R9 - 4X", "2X R20")) == []


def test_slot_ends_short_of_a_counted_radius_are_flagged():
    spec = p1_with_callouts("R9 - 4X")
    holes = spec["parts"][0]["base"]["profile"]["holes"]
    del holes[6]
    holes[5]["slot"].update(from_angle=150, to_angle=30)
    out = msgs(validate_spec(spec))
    assert len(out) == 1
    assert '"R9 - 4X"' in out[0] and "4 features of R9" in out[0]
    assert "2 slot ends" in out[0]


def test_round_holes_short_of_a_counted_diameter_are_flagged():
    spec = p1_with_callouts("3X Ø30")
    del spec["parts"][0]["base"]["profile"]["holes"][2]
    out = msgs(validate_spec(spec))
    assert len(out) == 1 and "3 features of Ø30" in out[0] and "2 round holes" in out[0]


def test_outline_fillets_and_arcs_count_toward_a_radius():
    spec = p1_with_callouts("R20 - 3X")
    out = msgs(validate_spec(spec))
    assert len(out) == 1 and "2 outline corners/arcs" in out[0]


def test_pattern_multiplies_a_hole_feature():
    spec = box_spec()
    spec["dimensions_read"] = [{"text": "Ø8 - 3X", "means": "holes"}]
    assert validate_spec(spec) == []
    spec["dimensions_read"] = [{"text": "Ø8 - 4X", "means": "holes"}]
    assert "4 features of Ø8" in msgs(validate_spec(spec))[0]


def test_counted_callouts_are_skipped_when_they_cannot_be_counted():
    spec = p1_with_callouts("R5 - 4X", "Ø30 - 3X THRU, Ø40 CBORE", "R20 TYP")
    assert validate_spec(spec) == []
    spec = p1_with_callouts("R20 - 6X")
    spec["parts"][0]["features"] = [{"type": "fillet", "radius": 20, "where": "top edges"}]
    assert validate_spec(spec) == []
