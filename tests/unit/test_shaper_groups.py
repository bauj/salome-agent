import pytest

from executor.shaper.groups import group_sub_type, check_group_name


def test_group_sub_type_from_ids():
    assert group_sub_type(["F1", "F3"]) == "face"
    assert group_sub_type(["E2"]) == "edge"


@pytest.mark.parametrize("ids", [[], ["F1", "E2"], ["X1"], "F1", [1, 2]])
def test_group_sub_type_rejects_bad_ids(ids):
    with pytest.raises(ValueError):
        group_sub_type(ids)


def test_check_group_name():
    check_group_name("inlet", taken=set())
    with pytest.raises(ValueError, match="taken"):
        check_group_name("inlet", taken={"inlet"})
    with pytest.raises(ValueError, match="letter"):
        check_group_name("2nd face", taken=set())
