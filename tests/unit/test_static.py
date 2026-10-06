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


def _fixture():
    from salome_agent.verify.introspect import build_api_map
    return build_api_map()


def test_every_model_call_the_shaper_tools_emit_is_in_the_api_map():
    import re
    from pathlib import Path
    import executor.shaper as shaper_pkg
    members = set(_fixture()["model"]["members"])
    src = "".join(p.read_text() for p in Path(shaper_pkg.__file__).parent.glob("*.py"))
    called = set(re.findall(r"model\.([A-Za-z_]+)\(", src)) - {"add"}
    called |= {f"add{op}" for op in ("Fuse", "Cut", "Common")}   # model.add{op}
    assert called and called <= members, sorted(called - members)


def test_shaper_result_script_frame_passes_and_a_bad_model_call_fails():
    from executor.shaper.state import POSTAMBLE, PREAMBLE
    api = _fixture()
    body = 'ext_1 = model.addExtrusion(shaper_doc, [model.selection("FACE", "f")], 5)\n'
    assert check_static(PREAMBLE + body + POSTAMBLE, api) is None
    err = check_static(PREAMBLE + "x = model.addNope(shaper_doc)\n" + POSTAMBLE, api)
    assert err["message"] == "model.addNope does not exist"
