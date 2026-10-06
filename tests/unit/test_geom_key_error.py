from types import SimpleNamespace

from executor.geom.hints import key_error_message


def test_a_shaper_name_given_to_a_geom_tool_says_why_it_is_unknown():
    shaper = SimpleNamespace(objects={"extrusion_1": "solid"}, sketches={})
    ex = SimpleNamespace(geom_bindings={"box_1": 1}, shaper_state=shaper)
    msg = key_error_message(ex, "fuse", {"shapes": ["extrusion_1"]}, "extrusion_1")
    assert msg == ("fuse: no object named 'extrusion_1' — extrusion_1 was built in SHAPER, "
                   "and GEOM tools cannot use SHAPER objects.")


def test_an_unknown_name_without_shaper_keeps_the_usual_hint():
    ex = SimpleNamespace(geom_bindings={"box_1": 1})
    assert "did you mean box_1" in key_error_message(ex, "fuse", {}, "box_2")
