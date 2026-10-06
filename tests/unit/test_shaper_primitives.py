import pytest

from executor.shaper.primitives import _coords, _num, plane_coefficients


def test_coords_read_a_list_or_an_xyz_string():
    assert _coords([1, 2, 3], "p") == [1, 2, 3]
    assert _coords("1, 2.5, -3", "p") == [1.0, 2.5, -3.0]
    assert _coords("point_1", "p") is None


def test_coords_refuse_a_wrong_length_or_a_non_number():
    with pytest.raises(ValueError, match=r"\[x, y, z\]"):
        _coords([1, 2], "p")
    with pytest.raises(ValueError, match="must be a number"):
        _coords([1, "a", 3], "p")


def test_num_refuses_bools_and_non_positive_sizes():
    with pytest.raises(ValueError):
        _num(True, "radius")
    with pytest.raises(ValueError, match="greater than 0"):
        _num(0, "radius", positive=True)
    assert _num(-2, "x") == -2


def test_plane_through_a_point_with_a_normal():
    assert plane_coefficients([0, 0, 5], [0, 0, 1]) == [0, 0, 1, -5.0]
    a, b, c, d = plane_coefficients([1, 2, 3], [1, 1, 0])
    assert a * 1 + b * 2 + c * 3 + d == 0


def test_plane_refuses_a_zero_normal():
    with pytest.raises(ValueError, match="normal"):
        plane_coefficients([0, 0, 0], [0, 0, 0])
