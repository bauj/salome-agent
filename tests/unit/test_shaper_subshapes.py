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
