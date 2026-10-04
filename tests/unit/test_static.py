from salome_agent.verify.static import check_static

API_MAP = {"geompy": {"members": ["MakeBoxDXDYDZ", "MakeFilletEdges"]}}


def test_syntax_error_reports_line():
    err = check_static("x = (\n", API_MAP)
    assert err is not None
    assert err["phase"] == "static"
    assert err["offending_line"] == 1


def test_unknown_toplevel_call_fails():
    err = check_static("geompy.NotARealMethod(1)", API_MAP)
    assert err is not None
    assert "NotARealMethod" in err["message"]


def test_known_toplevel_call_passes_even_with_wrong_arity():
    # Existence-only: static check does NOT catch wrong arg counts (spec §6.1).
    err = check_static("geompy.MakeBoxDXDYDZ(1, 2)", API_MAP)
    assert err is None


def test_derived_object_call_is_not_checked():
    err = check_static("mesh_obj.Compute()", API_MAP)  # mesh_obj isn't geompy/smesh
    assert err is None


def test_salome_init_call_is_banned():
    err = check_static("salome.salome_init()", API_MAP)
    assert err is not None
    assert "salome_init" in err["message"]
