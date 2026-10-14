from unittest.mock import MagicMock

from salome_agent.llm import text_tool_calls
from salome_agent.specialists.geom import run_geom_specialist

NAMES = {"finish_stage", "measure", "get_subshapes"}

# gemma4:31b on Ollama, run 60a30b29b1aa stage 2
GEMMA = '''```json
[
  {
    "finish_stage": {
      "built": "The vase was already hollowed with a 2 mm wall in stage 1.",
      "result_names": [
        "sa_60a30b29b1aa_revolution_1"
      ]
    }
  }
]
```'''


def test_reads_a_fenced_list_of_tool_name_keyed_calls():
    assert text_tool_calls(GEMMA, NAMES) == [{"name": "finish_stage", "id": None, "args": {
        "built": "The vase was already hollowed with a 2 mm wall in stage 1.",
        "result_names": ["sa_60a30b29b1aa_revolution_1"]}}]


def test_reads_name_and_arguments_forms_with_prose_around():
    text = 'I will measure it.\n```json\n{"name": "measure", "arguments": "{\\"names\\": [\\"a\\"]}"}\n```'
    assert text_tool_calls(text, NAMES) == [{"name": "measure", "args": {"names": ["a"]}, "id": None}]
    assert text_tool_calls('{"tool": "measure", "args": {"names": ["b"]}}', NAMES)[0]["args"] == {
        "names": ["b"]}


def test_ignores_unknown_tools_plain_prose_and_json_that_is_not_a_call():
    assert text_tool_calls('[{"make_teapot": {}}]', NAMES) == []
    assert text_tool_calls("The vase is done.", NAMES) == []
    assert text_tool_calls('```json\n{"volume": 3}\n```', NAMES) == []
    assert text_tool_calls(None, NAMES) == []


def _resp(content=None, calls=()):
    return {"tool_calls": [{"name": n, "args": a, "id": None} for n, a in calls],
            "content": content, "raw_message": None}


def test_a_call_written_as_text_is_run():
    llm = MagicMock()
    llm.chat.side_effect = [
        _resp('```json\n[{"make_box": {"dx": 1, "dy": 1, "dz": 1}}]\n```'),
        _resp(calls=[("finish_stage", {"result_names": ["box_1"], "built": "a box"})]),
    ]
    client = MagicMock()
    client.list_tools.return_value = {"tools": [{"name": "make_box", "category": "primitive"}]}
    client.call_tool.return_value = {"ok": True, "records": [
        {"fn": "MakeBoxDXDYDZ", "args": [], "out_vars": ["box_1"], "publish": ["box_1"]}]}
    result = run_geom_specialist({"module": "geom", "intent": "x", "params": {}, "inputs": []},
                                 llm, client, iteration_cap=6, verify_cadence="none")
    assert client.call_tool.call_args.args[:2] == ("make_box", {"dx": 1, "dy": 1, "dz": 1})
    assert result["result_names"] == ["box_1"]


def test_a_call_written_as_text_that_does_not_parse_gets_the_text_call_nudge():
    llm = MagicMock()
    llm.chat.side_effect = [
        _resp('```json\n[{"make_box": {"dx": 1,}}]\n```'),
        _resp(calls=[("finish_stage", {"result_names": [], "built": "nothing"})]),
    ]
    client = MagicMock()
    client.list_tools.return_value = {"tools": [{"name": "make_box", "category": "primitive"}]}
    run_geom_specialist({"module": "geom", "intent": "x", "params": {}, "inputs": []},
                        llm, client, iteration_cap=6, verify_cadence="none")
    sent = llm.chat.call_args_list[1].kwargs["messages"]
    assert "You wrote a tool call as text" in sent[-1]["content"]
    client.call_tool.assert_not_called()


SCHEMAS = {
    "finish_turn": {"properties": {"summary": {"type": "string"}, "question": {"type": "string"}},
                    "required": ["summary"]},
    "extrude": {"properties": {"base": {"type": "string"}, "size": {"type": "number"}},
                "required": ["base", "size"]},
    "measure": {"properties": {"names": {"type": "array"}}, "required": ["names"]},
    "make_face": {"properties": {"regions": {"anyOf": [{"type": "array"},
                                                       {"type": "string", "enum": ["all"]}]}}},
}
TAGGED = set(SCHEMAS)

# gemma4:31b on Ollama, run ee92a610cbbf turn 4 (llm_trace call 50)
GEMMA_TAG = ('I do not have access to the SMESH module or any meshing tools; I can only create '
             'and modify geometry using the SHAPER tools.\n\n<finish_turn summary="No changes '
             'made as meshing tools are not available in the current toolset." />')


def test_reads_a_call_written_as_a_tag():
    assert text_tool_calls(GEMMA_TAG, TAGGED, SCHEMAS) == [{"name": "finish_turn", "id": None,
        "args": {"summary": "No changes made as meshing tools are not available in the "
                            "current toolset."}}]


def test_tag_values_take_the_type_their_schema_declares():
    calls = text_tool_calls('<extrude base="f1" size="5"/><measure names=\'["a", "b"]\'>'
                            '</measure>', TAGGED, SCHEMAS)
    assert [c["args"] for c in calls] == [{"base": "f1", "size": 5}, {"names": ["a", "b"]}]
    assert text_tool_calls('<make_face regions="all"/>', TAGGED, SCHEMAS)[0]["args"] == {
        "regions": "all"}
    assert text_tool_calls('<make_face regions="[0, 2]"/>', TAGGED, SCHEMAS)[0]["args"] == {
        "regions": [0, 2]}
    assert text_tool_calls('<finish_turn summary="2024 &quot;ok&quot;"/>', TAGGED, SCHEMAS)[0][
        "args"] == {"summary": '2024 "ok"'}


def test_a_tag_that_does_not_fit_its_tool_is_not_a_call():
    assert text_tool_calls('<extrude base="f1" size="big"/>', TAGGED, SCHEMAS) == []
    assert text_tool_calls("I will finish with <finish_turn> now.", TAGGED, SCHEMAS) == []
    assert text_tool_calls('<b>bold</b> <make_teapot spout="1"/>', TAGGED, SCHEMAS) == []
