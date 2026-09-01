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


def test_no_trace_path_writes_nothing(tmp_path, monkeypatch):
    body = {"choices": [{"finish_reason": "stop",
                         "message": {"content": '{"stages": []}'}}]}
    monkeypatch.setattr("salome_agent.llm.requests.post", _stub_post(body))
    LLMClient("http://x/v1", "", "auto").chat(messages=[{"role": "user", "content": "hi"}])
    assert list(tmp_path.iterdir()) == []
