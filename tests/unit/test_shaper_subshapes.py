import math

import pytest

from executor.shaper.state import Sub
from executor.shaper.subshapes import select


def _subs():
    # three edges of a 10 x 10 x 10 box: bottom front, top front, a vertical one
    return [Sub("E1", None, "a", [5, 0, 0], 10), Sub("E2", None, "b", [0, 0, 5], 10),
            Sub("E3", None, "c", [5, 0, 10], 4)]


_PTS = {"E1": [[0, 0, 0], [10, 0, 0]], "E2": [[0, 0, 0], [0, 0, 10]],
        "E3": [[0, 0, 10], [10, 0, 10]]}
_points = lambda u: _PTS[u.id]


def test_all_and_sorted_by():
    assert [u.id for u in select(_subs(), "all", _points)] == ["E1", "E2", "E3"]
    top = select(_subs(), {"sorted_by": "z", "take": "max", "count": 1}, _points)
    assert [u.id for u in top] == ["E3"]


def test_by_size():
    assert [u.id for u in select(_subs(), {"by_size": "shortest", "count": 1}, _points)] == ["E3"]


def test_on_plane_keeps_only_what_lies_in_it():
    on = select(_subs(), {"on_plane": {"point": [0, 0, 10], "normal": [0, 0, 1]}}, _points)
    assert [u.id for u in on] == ["E3"]
    below = select(_subs(), {"on_plane": {"point": [0, 0, 10], "normal": [0, 0, 1]},
                             "state": "onin"}, _points)
    assert [u.id for u in below] == ["E1", "E2", "E3"]
    strictly = select(_subs(), {"on_plane": {"point": [0, 0, 10], "normal": [0, 0, 1]},
                                "state": "in"}, _points)
    assert [u.id for u in strictly] == ["E1"]


def test_a_json_string_predicate_is_read():
    got = select(_subs(), '{"sorted_by": "x", "take": "min", "count": 1}', _points)
    assert [u.id for u in got] == ["E2"]


@pytest.mark.parametrize("where", [
    {"sorted_by": "w", "count": 1}, {"by_size": "huge", "count": 1},
    {"sorted_by": "z", "count": 0}, {"on_plane": {"point": [0, 0], "normal": [0, 0, 1]}},
    {"on_plane": {"point": [0, 0, 0], "normal": [0, 0, 0]}}, {"nearest": [0, 0, 0]}, 7])
def test_bad_predicates_are_refused(where):
    with pytest.raises(ValueError):
        select(_subs(), where, _points)


@pytest.mark.parametrize("where", [
    {"sorted_by": "z", "count": float("inf")}, {"sorted_by": "z", "count": float("nan")},
    {"on_plane": {"point": {"x": 0}, "normal": [0, 0, 1]}},
    {"on_plane": {"point": [0, 0, 0], "normal": [0, 0, 1]}, "state": ["on"]}])
def test_review_fixes_refuse_odd_predicate_values(where):
    with pytest.raises(ValueError):
        select(_subs(), where, _points)


def test_fillets_meeting_across_a_face_are_refused():
    from executor.shaper.subshapes import fillets_meet
    rim = [("E4", "E5", 2.0)]
    assert fillets_meet(0.9, rim) is None
    for r in (0.95, 1.0, 3.0):
        msg = fillets_meet(r, rim)
        assert "E4 and E5 would meet" in msg and "below 0.95" in msg
    assert fillets_meet(5.0, [("E1", "E2", 0.0)]) is None   # edges that touch


def _car_top(crease_deg=3.47, back=False):
    # Top side edge E54 of run llm_trace_3: at vertex 1 it meets the slope edge S
    # (a crease) and the top/slope crease C; at vertex 2 the slope edge meets the
    # nose edge N and the slope/nose crease C2, both real creases.
    a = math.radians(crease_deg)
    t_s = [-math.cos(a), 0, math.sin(a)] if back else [math.cos(a), 0, -math.sin(a)]
    ends = {"E54": [(0, [1, 0, 0]), (1, [-1, 0, 0])], "S": [(1, t_s), (2, [-1, 0, 0])],
            "C": [(1, [0, 1, 0])], "N": [(2, [0, 0, -1])], "C2": [(2, [0, 1, 0])],
            "B0": [(0, [0, 1, 0])], "A0": [(0, [-1, 0, 0])]}
    at = {0: ["E54", "B0", "A0"], 1: ["E54", "S", "C"], 2: ["S", "N", "C2"]}
    smooth = {"E54": False, "S": False, "C": crease_deg < math.degrees(0.1), "N": False,
              "C2": False, "B0": False, "A0": False}
    return ends, at, smooth


def test_fillet_goes_on_along_an_edge_met_at_a_shallow_crease():
    from executor.shaper.subshapes import fillet_chain
    ends, at, smooth = _car_top()
    assert fillet_chain(["E54"], ends, at, smooth, lambda e, v: False) == (["S"], [])


def test_fillet_stops_at_a_crease_over_the_smoothness_limit():
    from executor.shaper.subshapes import fillet_chain
    ends, at, smooth = _car_top(crease_deg=6.0)
    assert fillet_chain(["E54"], ends, at, smooth, lambda e, v: False) == ([], [])


def test_fillet_does_not_turn_back():
    from executor.shaper.subshapes import fillet_chain
    ends, at, smooth = _car_top(back=True)
    assert fillet_chain(["E54"], ends, at, smooth, lambda e, v: False) == ([], [])


def test_fillet_on_faces_meeting_tangentially_needs_a_straight_run():
    from executor.shaper.subshapes import fillet_chain
    ends, at, smooth = _car_top()
    assert fillet_chain(["E54"], ends, at, smooth, lambda e, v: True) == ([], [])
    ends["S"][0] = (1, [1, 0, 0])
    assert fillet_chain(["E54"], ends, at, smooth, lambda e, v: True) == (["S"], [])


def test_an_undecided_edge_stops_the_walk_and_is_reported():
    from executor.shaper.subshapes import fillet_chain
    ends, at, smooth = _car_top()
    smooth["C"] = None
    assert fillet_chain(["E54"], ends, at, smooth, lambda e, v: False) == ([], [1])


def test_fillet_meeting_an_earlier_fillet_is_refused():
    from executor.shaper.subshapes import meets_earlier
    assert meets_earlier(0.2, [("E54", "F12", 0.25)]) is None
    msg = meets_earlier(3, [("E54", "F12", 0.25)])
    assert "earlier fillet F12" in msg and "below 0.2375" in msg
    assert meets_earlier(3, [("E54", "F12", 0.0)]) is None


def test_cut_face_refusal_names_the_narrowest_width():
    from executor.shaper.subshapes import cuts_face
    assert cuts_face(3, [], []) is None
    msg = cuts_face(3, [("F10", "E12", "E54", 0.75354), ("F3", "E20", "E7", 2.0)], [])
    assert "on E12 ran across F10 and cut it in two" in msg
    assert "between E12 and E54" in msg and "radius below 0.7535" in msg
    assert "also cut F3" in msg and "also rounds" not in msg
    msg = cuts_face(3, [("F10", "E60", "E12", 0.75354)], ["E60", "E61"])
    assert "SHAPER also rounds E60, E61" in msg
