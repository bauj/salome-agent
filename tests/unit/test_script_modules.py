from unittest.mock import MagicMock

from salome_agent.modules import spec_for
from salome_agent.specialists.script_modules import SMESH_SPEC
from salome_agent.specialists.tool_specialist import ToolSpecialist


def _resp(*calls):
    return {"tool_calls": [{"name": n, "args": a, "id": None} for n, a in calls],
            "content": None, "raw_message": None}


def test_smesh_script_module_is_registered():
    assert spec_for("smesh") is SMESH_SPEC


def test_smesh_system_prompt_is_unchanged():
    from salome_agent.specialists.smesh import SMESH_SYSTEM_PROMPT

    assert SMESH_SPEC.prompt("final") == SMESH_SYSTEM_PROMPT
    assert SMESH_SPEC.prompt("strict") == SMESH_SYSTEM_PROMPT


def test_emit_script_terminates_with_the_code():
    llm = MagicMock()
    llm.chat.side_effect = [_resp(("emit_script", {"code": "mesh.Compute()"}))]
    result = ToolSpecialist(SMESH_SPEC).run({"module": "smesh"}, llm, MagicMock(),
                                            iteration_cap=12)
    assert result == {"script": "mesh.Compute()", "warning": None}


def test_upstream_block_wording_is_unchanged():
    assert SMESH_SPEC.render_upstream(None, ["box_1"]) == (
        "Objects already in the study — call find_object() with "
        "exactly these names: ['box_1']")


def test_retry_block_wording_is_unchanged():
    assert SMESH_SPEC.render_retry({"last_error": {"message": "boom"}}) == (
        "Your previous attempt failed at execution with:\nboom\n"
        "Return a corrected script.")
    assert SMESH_SPEC.render_retry({"last_error": None}) is None


def test_api_introspect_executes_through_the_client():
    llm = MagicMock()
    llm.chat.side_effect = [_resp(("api_introspect", {"name": "geompy"})),
                            _resp(("emit_script", {"code": "x"}))]
    client = MagicMock()
    client.introspect.return_value = {"geompy": ["MakeBox"]}
    result = ToolSpecialist(SMESH_SPEC).run({"module": "smesh"}, llm, client,
                                            iteration_cap=12)
    client.introspect.assert_called_once_with(["geompy"])
    assert result["script"] == "x"
    second = llm.chat.call_args_list[1].kwargs["messages"]
    assert second[-1]["role"] == "tool"
    assert "MakeBox" in second[-1]["content"]


def test_no_client_offers_only_the_terminator():
    llm = MagicMock()
    llm.chat.side_effect = [_resp(("emit_script", {"code": "x"}))]
    ToolSpecialist(SMESH_SPEC).run({"module": "smesh"}, llm, None, iteration_cap=12)
    assert [t["name"] for t in llm.chat.call_args.kwargs["tools"]] == ["emit_script"]


def test_iteration_cap_warns_without_emit_script():
    llm = MagicMock()
    llm.chat.return_value = _resp(("list_study", {}))
    result = ToolSpecialist(SMESH_SPEC).run({"module": "smesh"}, llm, MagicMock(),
                                            iteration_cap=2)
    assert result == {"script": "", "warning": "iteration_cap reached without emit_script"}


def test_script_modules_are_never_nudged_to_measure():
    llm = MagicMock()
    llm.chat.side_effect = [_resp(("list_study", {})), _resp(("emit_script", {"code": "x"}))]
    result = ToolSpecialist(SMESH_SPEC).run({"module": "smesh"}, llm, MagicMock(),
                                            iteration_cap=12, verify_cadence="strict")
    assert result["script"] == "x"
