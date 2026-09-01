import json

import pytest
import requests

from salome_agent.llm import LLMClient, _parse_json_content


class _FakeResp:
    def __init__(self, body, status_code=200, text=""):
        self._body = body
        self.status_code = status_code
        self.text = text

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def json(self):
        if self._body is None:
            raise ValueError("Expecting value: line 1 column 1 (char 0)")
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


def test_chat_retries_transient_gateway_errors(monkeypatch):
    body = {"choices": [{"finish_reason": "stop",
                         "message": {"role": "assistant", "content": '{"ok": 1}'}}]}
    calls = []

    def post(url, json=None, headers=None, verify=None, timeout=None):
        calls.append(1)
        if len(calls) < 3:
            return _FakeResp({}, status_code=503)
        return _FakeResp(body)

    monkeypatch.setattr("salome_agent.llm.requests.post", post)
    monkeypatch.setattr("salome_agent.llm.time.sleep", lambda s: None)
    client = LLMClient("http://x/v1", "", "auto")

    assert client.chat(messages=[{"role": "user", "content": "hi"}]) == {"ok": 1}
    assert len(calls) == 3


def test_chat_does_not_retry_a_slow_gateway_error(monkeypatch):
    # elapsed >= _GATEWAY_RETRY_MAX_ELAPSED => proxy upstream-timeout, not a blip.
    monkeypatch.setattr("salome_agent.llm._GATEWAY_RETRY_MAX_ELAPSED", 0.0)
    calls = []

    def post(url, json=None, headers=None, verify=None, timeout=None):
        calls.append(1)
        return _FakeResp({}, status_code=503)

    monkeypatch.setattr("salome_agent.llm.requests.post", post)
    monkeypatch.setattr("salome_agent.llm.time.sleep", lambda s: None)
    client = LLMClient("http://x/v1", "", "auto")

    with pytest.raises(requests.HTTPError):
        client.chat(messages=[{"role": "user", "content": "hi"}])
    assert len(calls) == 1


def test_chat_gives_up_after_persistent_gateway_errors(monkeypatch):
    monkeypatch.setattr("salome_agent.llm.requests.post",
                        lambda *a, **k: _FakeResp({}, status_code=502))
    monkeypatch.setattr("salome_agent.llm.time.sleep", lambda s: None)
    client = LLMClient("http://x/v1", "", "auto")

    with pytest.raises(requests.HTTPError):
        client.chat(messages=[{"role": "user", "content": "hi"}])


def test_chat_raises_clear_error_on_non_json_body(monkeypatch):
    monkeypatch.setattr("salome_agent.llm.requests.post",
                        lambda *a, **k: _FakeResp(None, status_code=200, text="<html>502</html>"))
    client = LLMClient("http://x/v1", "", "auto")

    with pytest.raises(RuntimeError, match="non-JSON body"):
        client.chat(messages=[{"role": "user", "content": "hi"}])


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


def test_prompt_is_logged_before_the_request_is_sent(tmp_path, monkeypatch, caplog):
    import logging

    import requests as _requests

    def post(url, json=None, headers=None, verify=None, timeout=None):
        raise _requests.ConnectionError("boom")

    monkeypatch.setattr("salome_agent.llm.requests.post", post)
    caplog.set_level(logging.DEBUG, logger="salome_agent")
    client = LLMClient("http://x/v1", "", "auto", trace_path=str(tmp_path / "t.jsonl"))
    with pytest.raises(_requests.ConnectionError):
        client.chat(messages=[{"role": "user", "content": "PROMPT-MARKER"}])
    assert any("PROMPT-MARKER" in r.message for r in caplog.records)


def test_no_trace_path_writes_nothing(tmp_path, monkeypatch):
    body = {"choices": [{"finish_reason": "stop",
                         "message": {"content": '{"stages": []}'}}]}
    monkeypatch.setattr("salome_agent.llm.requests.post", _stub_post(body))
    LLMClient("http://x/v1", "", "auto").chat(messages=[{"role": "user", "content": "hi"}])
    assert list(tmp_path.iterdir()) == []
