import copy

from salome_agent.nodes.vision_spec import parse_num, parse_point, validate_spec


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
