import json

import pytest

from salome_agent.llm import LLMClient, _parse_json_content


class _FakeResp:
    def __init__(self, body):
        self._body = body

    def raise_for_status(self):
        pass

    def json(self):
        return self._body


def _stub_post(body):
    def post(url, json=None, headers=None, verify=None, timeout=None):
        return _FakeResp(body)
    return post


def test_plain_json():
    assert _parse_json_content('{"stages": []}') == {"stages": []}


def test_fenced_json():
    assert _parse_json_content('```json\n{"stages": [1]}\n```') == {"stages": [1]}


def test_json_with_prose_around_it():
    raw = 'Here is the plan:\n{"stages": [{"module": "geom"}]}\nHope that helps.'
    assert _parse_json_content(raw) == {"stages": [{"module": "geom"}]}


def test_unparseable_raises_with_raw_content():
    with pytest.raises(ValueError, match="parseable JSON"):
        _parse_json_content("sorry, I cannot do that")


def test_trace_writes_request_and_raw_response(tmp_path, monkeypatch):
    body = {"choices": [{"finish_reason": "stop", "message": {
        "role": "assistant", "content": '{"stages": []}',
        "reasoning_content": "the user wants nothing"}}]}
    monkeypatch.setattr("salome_agent.llm.requests.post", _stub_post(body))
    trace = tmp_path / "sub" / "llm_trace.jsonl"
    client = LLMClient("http://x/v1", "", "auto", trace_path=str(trace))

    client.chat(messages=[{"role": "user", "content": "do nothing"}])

    lines = trace.read_text().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["request"]["messages"][0]["content"] == "do nothing"
    assert entry["response"]["message"]["reasoning_content"] == "the user wants nothing"
    assert entry["response"]["finish_reason"] == "stop"


def test_chat_forwards_tool_description(monkeypatch):
    captured = {}

    def post(url, json=None, headers=None, verify=None, timeout=None):
        captured["payload"] = json
        return _FakeResp({"choices": [{"finish_reason": "tool_calls", "message": {
            "tool_calls": [{"id": "c1", "function": {"name": "make_box", "arguments": "{}"}}]}}]})

    monkeypatch.setattr("salome_agent.llm.requests.post", post)
    LLMClient("http://x/v1", "", "auto").chat(
        messages=[{"role": "user", "content": "x"}],
        tools=[{"name": "make_box", "parameters": {"type": "object"}, "description": "a solid box"}],
    )
    fn = captured["payload"]["tools"][0]["function"]
    assert fn["description"] == "a solid box"


def test_chat_uses_tool_choice_auto(monkeypatch):
    captured = {}

    def post(url, json=None, headers=None, verify=None, timeout=None):
        captured["payload"] = json
        return _FakeResp({"choices": [{"finish_reason": "tool_calls", "message": {
            "tool_calls": [{"id": "c1", "function": {"name": "mk", "arguments": "{}"}}]}}]})

    monkeypatch.setattr("salome_agent.llm.requests.post", post)
    LLMClient("http://x/v1", "", "auto").chat(
        messages=[{"role": "user", "content": "x"}],
        tools=[{"name": "mk", "parameters": {"type": "object"}}])
    assert captured["payload"]["tool_choice"] == "auto"


def test_chat_returns_none_toolcall_when_model_only_talks(monkeypatch):
    body = {"choices": [{"finish_reason": "stop",
                         "message": {"content": "let me think about this"}}]}
    monkeypatch.setattr("salome_agent.llm.requests.post", _stub_post(body))
    out = LLMClient("http://x/v1", "", "auto").chat(
        messages=[{"role": "user", "content": "x"}],
        tools=[{"name": "mk", "parameters": {"type": "object"}}])
    assert out["tool_call"] is None
    assert out["content"] == "let me think about this"


def test_chat_keeps_reasoning_content_alongside_the_tool_call(monkeypatch):
    body = {"choices": [{"finish_reason": "tool_calls", "message": {
        "content": "I'll make a box",
        "tool_calls": [{"id": "c1", "function": {"name": "mk", "arguments": '{"dx": 1}'}}]}}]}
    monkeypatch.setattr("salome_agent.llm.requests.post", _stub_post(body))
    out = LLMClient("http://x/v1", "", "auto").chat(
        messages=[{"role": "user", "content": "x"}],
        tools=[{"name": "mk", "parameters": {"type": "object"}}])
    assert out["tool_call"] == "mk"
    assert out["content"] == "I'll make a box"


def test_no_trace_path_writes_nothing(tmp_path, monkeypatch):
    body = {"choices": [{"finish_reason": "stop",
                         "message": {"content": '{"stages": []}'}}]}
    monkeypatch.setattr("salome_agent.llm.requests.post", _stub_post(body))
    LLMClient("http://x/v1", "", "auto").chat(messages=[{"role": "user", "content": "hi"}])
    assert list(tmp_path.iterdir()) == []
