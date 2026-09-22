def test_the_geom_tools_shim_is_gone():
    import pytest

    with pytest.raises(ImportError):
        import executor.geom_tools  # noqa: F401
