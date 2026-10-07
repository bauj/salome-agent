import math

import pytest

from executor.shaper.transforms import _count, align_delta, frame_rotation


def _rotate(axis, deg, v):
    """Rodrigues' rotation of v about the unit `axis`."""
    t = math.radians(deg)
    k = axis
    cross = [k[1] * v[2] - k[2] * v[1], k[2] * v[0] - k[0] * v[2], k[0] * v[1] - k[1] * v[0]]
    dot = sum(a * b for a, b in zip(k, v))
    return [v[i] * math.cos(t) + cross[i] * math.sin(t) + k[i] * dot * (1 - math.cos(t))
            for i in range(3)]


@pytest.mark.parametrize("x_dir, y_dir", [
    ([0, 1, 0], [-1, 0, 0]),       # quarter turn about Z
    ([-1, 0, 0], [0, -1, 0]),      # half turn about Z
    ([1, 0, 0], [0, -1, 0]),       # half turn about X
    ([0, 0, 1], [1, 0, 0]),
    ([1, 1, 0], [-1, 1, 0.5]),     # y_dir not perpendicular: made so
])
def test_frame_rotation_takes_x_and_y_to_the_target_axes(x_dir, y_dir):
    axis, deg = frame_rotation(x_dir, y_dir)
    n = math.sqrt(sum(c * c for c in x_dir))
    x = [c / n for c in x_dir]
    assert _rotate(axis, deg, [1, 0, 0]) == pytest.approx(x, abs=1e-9)
    y = _rotate(axis, deg, [0, 1, 0])
    assert sum(a * b for a, b in zip(y, x)) == pytest.approx(0, abs=1e-9)
    assert sum(a * b for a, b in zip(y, y_dir)) > 0


def test_frame_rotation_of_the_world_frame_is_none():
    assert frame_rotation([1, 0, 0], [0, 1, 0])[1] == 0


def test_frame_rotation_refuses_parallel_axes():
    with pytest.raises(ValueError, match="parallel"):
        frame_rotation([1, 0, 0], [2, 0, 0])


def test_align_puts_a_shape_on_top_of_another_and_centres_it():
    box = [0, 10, 0, 10, 0, 10]
    table = [-50, 50, -20, 20, 0, 30]
    d = align_delta(box, table, [{"axis": "z", "my": "min", "target": "max"},
                                 {"axis": "x", "my": "center", "target": "center", "gap": 2}])
    assert d == pytest.approx([-3, 0, 30])


def test_align_refuses_two_entries_on_one_axis_and_bad_anchors():
    with pytest.raises(ValueError, match="two match"):
        align_delta([0] * 6, [0] * 6, [{"axis": "x", "my": "min", "target": "max"},
                                       {"axis": "x", "my": "max", "target": "min"}])
    with pytest.raises(ValueError, match="min, max or center"):
        align_delta([0] * 6, [0] * 6, [{"axis": "x", "my": "top", "target": "max"}])


def test_count_wants_a_whole_number_of_at_least_two():
    assert _count(3.0, "count") == 3
    for bad in (1, 2.5, True, "3"):
        with pytest.raises(ValueError):
            _count(bad, "count")


def test_frame_rotation_sees_a_frame_equal_to_the_world_up_to_rounding():
    s = 1 / math.sqrt(3)
    assert frame_rotation([s * math.sqrt(3), 1e-17, 0], [0, 1, 0])[1] == 0


def test_frame_rotation_refuses_a_y_dir_parallel_up_to_noise():
    with pytest.raises(ValueError, match="parallel"):
        frame_rotation([1, 0, 0], [1e6, 1e-6, 0])


def test_count_refuses_non_finite_and_huge_values():
    for bad in (float("inf"), float("nan"), 10_000):
        with pytest.raises(ValueError):
            _count(bad, "count")


def test_align_refuses_an_unhashable_axis():
    with pytest.raises(ValueError, match="x, y or z"):
        align_delta([0] * 6, [0] * 6, [{"axis": ["x"], "my": "min", "target": "max"}])
