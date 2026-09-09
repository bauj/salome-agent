import json

import pytest
import requests

from salome_agent.llm import (LLMClient, _parse_json_content, assistant_tool_turn,
                              continuity_fields)


def _sse_frames(body):
    """Turn a canned non-streaming response body into the SSE `data:` lines a
    real endpoint would emit for it: one delta per choice carrying the whole
    message, an optional usage-only frame, then [DONE]."""
    lines = []
    for choice in body.get("choices", []):
        msg = choice.get("message", {})
        delta = {"role": "assistant"}
        if msg.get("content") is not None:
            delta["content"] = msg["content"]
        if msg.get("reasoning_content"):
            delta["reasoning_content"] = msg["reasoning_content"]
        if msg.get("extra_content"):
            delta["extra_content"] = msg["extra_content"]
        if msg.get("tool_calls"):
            delta["tool_calls"] = [
                {"index": i, "id": tc.get("id"), "type": "function",
                 "function": tc.get("function", {}),
                 **({"extra_content": tc["extra_content"]} if tc.get("extra_content") else {})}
                for i, tc in enumerate(msg["tool_calls"])
            ]
        lines.append("data: " + json.dumps(
            {"choices": [{"delta": delta, "finish_reason": choice.get("finish_reason")}]}))
    if body.get("usage"):
        lines.append("data: " + json.dumps({"choices": [], "usage": body["usage"]}))
    lines.append("data: [DONE]")
    return lines


class _FakeResp:
    def __init__(self, body, status_code=200, text="", lines=None, headers=None):
        self._body = body
        self.status_code = status_code
        self.text = text
        self.headers = headers or {}
        # explicit SSE lines override the synthesized ones (partial / broken streams)
        self._lines = lines

    def raise_for_status(self):
        if self.status_code >= 400:
            err = requests.HTTPError(f"HTTP {self.status_code}")
            err.response = self
            raise err

    def json(self):
        if self._body is None:
            raise ValueError("Expecting value: line 1 column 1 (char 0)")
        return self._body

    def iter_lines(self, chunk_size=512, decode_unicode=False):
        if self._lines is not None:
            yield from self._lines
            return
        if self._body is None:
            yield self.text or ""
            return
        yield from _sse_frames(self._body)

    def close(self):
        pass


def _stub_post(body):
    def post(url, json=None, headers=None, verify=None, timeout=None, stream=None):
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


def _msg(content):
    return {"choices": [{"finish_reason": "stop",
                         "message": {"role": "assistant", "content": content}}]}


def test_chat_retries_bad_json_with_parser_error_fed_back(monkeypatch):
    posts = []

    def post(url, json=None, headers=None, verify=None, timeout=None, stream=None):
        posts.append(json["messages"])
        if len(posts) == 1:
            return _FakeResp(_msg('{"stages": [1,]}'))   # trailing comma
        return _FakeResp(_msg('{"stages": [1]}'))

    monkeypatch.setattr("salome_agent.llm.requests.post", post)
    client = LLMClient("http://x/v1", "", "auto")

    assert client.chat(messages=[{"role": "user", "content": "plan"}]) == {"stages": [1]}
    assert len(posts) == 2
    # the retry carried the model's bad answer back plus a correction turn
    retry_msgs = posts[1]
    assert retry_msgs[-2] == {"role": "assistant", "content": '{"stages": [1,]}'}
    assert retry_msgs[-1]["role"] == "user"
    assert "not valid JSON" in retry_msgs[-1]["content"]


def test_chat_gives_up_after_json_retries_and_raises_raw_content(monkeypatch):
    posts = []

    def post(url, json=None, headers=None, verify=None, timeout=None, stream=None):
        posts.append(1)
        return _FakeResp(_msg("still not json"))

    monkeypatch.setattr("salome_agent.llm.requests.post", post)
    client = LLMClient("http://x/v1", "", "auto", json_retries=2)

    with pytest.raises(ValueError, match="still not json"):
        client.chat(messages=[{"role": "user", "content": "plan"}])
    assert len(posts) == 3   # 1 initial + 2 retries


def test_chat_json_retries_default_is_two(monkeypatch):
    posts = []
    monkeypatch.setattr("salome_agent.llm.requests.post",
                        lambda *a, **k: (posts.append(1), _FakeResp(_msg("nope")))[1])
    client = LLMClient("http://x/v1", "", "auto")
    with pytest.raises(ValueError):
        client.chat(messages=[{"role": "user", "content": "plan"}])
    assert len(posts) == 3


def test_chat_retries_transient_gateway_errors(monkeypatch):
    body = {"choices": [{"finish_reason": "stop",
                         "message": {"role": "assistant", "content": '{"ok": 1}'}}]}
    calls = []

    def post(url, json=None, headers=None, verify=None, timeout=None, stream=None):
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

    def post(url, json=None, headers=None, verify=None, timeout=None, stream=None):
        calls.append(1)
        return _FakeResp({}, status_code=503)

    monkeypatch.setattr("salome_agent.llm.requests.post", post)
    monkeypatch.setattr("salome_agent.llm.time.sleep", lambda s: None)
    client = LLMClient("http://x/v1", "", "auto")

    with pytest.raises(requests.HTTPError):
        client.chat(messages=[{"role": "user", "content": "hi"}])
    assert len(calls) == 1


def test_chat_retries_429_and_honours_retry_after(monkeypatch):
    body = {"choices": [{"finish_reason": "stop",
                         "message": {"role": "assistant", "content": '{"ok": 1}'}}]}
    calls = []
    slept = []

    def post(url, json=None, headers=None, verify=None, timeout=None, stream=None):
        calls.append(1)
        if len(calls) == 1:
            return _FakeResp({}, status_code=429, headers={"Retry-After": "5"})
        return _FakeResp(body)

    monkeypatch.setattr("salome_agent.llm.requests.post", post)
    monkeypatch.setattr("salome_agent.llm.time.sleep", lambda s: slept.append(s))
    assert LLMClient("http://x/v1", "", "auto").chat(
        messages=[{"role": "user", "content": "hi"}]) == {"ok": 1}
    assert len(calls) == 2
    assert slept == [5.0]


def test_chat_429_retry_after_is_capped(monkeypatch):
    body = {"choices": [{"finish_reason": "stop", "message": {"content": '{"ok": 1}'}}]}
    calls = []
    slept = []

    def post(url, json=None, headers=None, verify=None, timeout=None, stream=None):
        calls.append(1)
        if len(calls) == 1:
            return _FakeResp({}, status_code=429, headers={"Retry-After": "9999"})
        return _FakeResp(body)

    monkeypatch.setattr("salome_agent.llm.requests.post", post)
    monkeypatch.setattr("salome_agent.llm.time.sleep", lambda s: slept.append(s))
    LLMClient("http://x/v1", "", "auto").chat(messages=[{"role": "user", "content": "hi"}])
    assert slept == [60.0]


def test_chat_gives_up_after_persistent_429(monkeypatch):
    calls = []
    monkeypatch.setattr("salome_agent.llm.requests.post",
                        lambda *a, **k: calls.append(1) or _FakeResp({}, status_code=429))
    monkeypatch.setattr("salome_agent.llm.time.sleep", lambda s: None)
    with pytest.raises(requests.HTTPError):
        LLMClient("http://x/v1", "", "auto").chat(messages=[{"role": "user", "content": "hi"}])
    assert len(calls) == 3


def test_max_retries_config_extends_attempts_with_exponential_backoff(monkeypatch):
    calls = []
    slept = []
    monkeypatch.setattr("salome_agent.llm.requests.post",
                        lambda *a, **k: calls.append(1) or _FakeResp({}, status_code=503))
    monkeypatch.setattr("salome_agent.llm.time.sleep", lambda s: slept.append(s))
    with pytest.raises(requests.HTTPError):
        LLMClient("http://x/v1", "", "auto", max_retries=4).chat(
            messages=[{"role": "user", "content": "hi"}])
    assert len(calls) == 5              # 1 initial + 4 retries
    assert slept == [2.0, 4.0, 8.0, 16.0]   # exponential, before each retry


def test_backoff_is_capped(monkeypatch):
    from salome_agent.llm import _backoff_seconds
    assert _backoff_seconds(1) == 2.0
    assert _backoff_seconds(6) == 30.0     # 2*2**5 == 64 -> capped


def test_build_llm_reads_max_retries():
    from salome_agent.cli import build_llm

    cfg = {"llm": {"base_url": "http://x/v1", "model": "auto", "max_retries": 6}}
    assert build_llm(cfg).max_retries == 6


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

    def post(url, json=None, headers=None, verify=None, timeout=None, stream=None):
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


def test_chat_sends_extra_headers(monkeypatch):
    captured = {}

    def post(url, json=None, headers=None, verify=None, timeout=None, stream=None):
        captured["headers"] = headers
        return _FakeResp({"choices": [{"finish_reason": "stop",
                                       "message": {"content": '{"ok": 1}'}}]})

    monkeypatch.setattr("salome_agent.llm.requests.post", post)
    LLMClient("http://x/v1", "", "auto",
              extra_headers={"x-opencode-session": "run123"}).chat(
        messages=[{"role": "user", "content": "x"}])
    assert captured["headers"]["x-opencode-session"] == "run123"


def test_build_llm_substitutes_run_id_in_extra_headers():
    from salome_agent.cli import build_llm

    cfg = {"llm": {"base_url": "http://x/v1", "model": "auto",
                   "extra_headers": {"x-opencode-session": "{run_id}", "x-static": "keep"}}}
    client = build_llm(cfg, run_id="abc123")
    assert client.extra_headers == {"x-opencode-session": "abc123", "x-static": "keep"}


def test_chat_sends_extra_body(monkeypatch):
    captured = {}

    def post(url, json=None, headers=None, verify=None, timeout=None, stream=None):
        captured["payload"] = json
        return _FakeResp({"choices": [{"finish_reason": "stop",
                                       "message": {"content": '{"ok": 1}'}}]})

    monkeypatch.setattr("salome_agent.llm.requests.post", post)
    LLMClient("http://x/v1", "", "auto",
              extra_body={"reasoning_effort": "high"}).chat(
        messages=[{"role": "user", "content": "x"}])
    assert captured["payload"]["reasoning_effort"] == "high"


def test_build_llm_reads_extra_body():
    from salome_agent.cli import build_llm

    cfg = {"llm": {"base_url": "http://x/v1", "model": "auto",
                   "extra_body": {"reasoning_effort": "high"}}}
    client = build_llm(cfg)
    assert client.extra_body == {"reasoning_effort": "high"}


def _capture_payload(monkeypatch):
    seen = {}

    def post(url, json=None, headers=None, verify=None, timeout=None, stream=None):
        seen["payload"] = json
        return _FakeResp({"choices": [{"finish_reason": "tool_calls", "message": {
            "tool_calls": [{"id": "c1", "function": {"name": "mk", "arguments": "{}"}}]}}]})

    monkeypatch.setattr("salome_agent.llm.requests.post", post)
    return seen


def test_module_extra_body_replaces_extra_body_for_that_module(monkeypatch):
    seen = _capture_payload(monkeypatch)
    LLMClient("http://x/v1", "", "auto",
              extra_body={"reasoning_effort": "high"},
              module_extra_body={"geom": {}}).chat(
        messages=[{"role": "user", "content": "x"}],
        tools=[{"name": "mk", "parameters": {"type": "object"}}], module="geom")
    assert "reasoning_effort" not in seen["payload"]


def test_module_without_an_entry_falls_back_to_extra_body(monkeypatch):
    seen = _capture_payload(monkeypatch)
    LLMClient("http://x/v1", "", "auto",
              extra_body={"reasoning_effort": "high"},
              module_extra_body={"geom": {}}).chat(
        messages=[{"role": "user", "content": "x"}],
        tools=[{"name": "mk", "parameters": {"type": "object"}}], module="smesh")
    assert seen["payload"]["reasoning_effort"] == "high"


def test_no_module_uses_extra_body(monkeypatch):
    seen = _capture_payload(monkeypatch)
    LLMClient("http://x/v1", "", "auto",
              extra_body={"reasoning_effort": "high"},
              module_extra_body={"planner": {"reasoning_effort": "low"}}).chat(
        messages=[{"role": "user", "content": "x"}],
        tools=[{"name": "mk", "parameters": {"type": "object"}}])
    assert seen["payload"]["reasoning_effort"] == "high"


def test_trace_records_the_resolved_per_module_body(monkeypatch, tmp_path):
    body = {"choices": [{"finish_reason": "stop",
                         "message": {"role": "assistant", "content": '{"a": 1}'}}]}
    monkeypatch.setattr("salome_agent.llm.requests.post", _stub_post(body))
    trace = tmp_path / "t.jsonl"
    LLMClient("http://x/v1", "", "auto", trace_path=str(trace),
              extra_body={"reasoning_effort": "high"},
              module_extra_body={"geom": {"reasoning_effort": "none"}}).chat(
        messages=[{"role": "user", "content": "x"}], module="geom")
    entry = json.loads(trace.read_text().splitlines()[0])
    assert entry["request"]["extra_body"] == {"reasoning_effort": "none"}


def test_build_llm_reads_module_extra_body():
    from salome_agent.cli import build_llm

    cfg = {"llm": {"base_url": "http://x/v1", "model": "auto",
                   "module_extra_body": {"geom": {}}}}
    assert build_llm(cfg).module_extra_body == {"geom": {}}


def test_chat_uses_tool_choice_auto(monkeypatch):
    captured = {}

    def post(url, json=None, headers=None, verify=None, timeout=None, stream=None):
        captured["payload"] = json
        return _FakeResp({"choices": [{"finish_reason": "tool_calls", "message": {
            "tool_calls": [{"id": "c1", "function": {"name": "mk", "arguments": "{}"}}]}}]})

    monkeypatch.setattr("salome_agent.llm.requests.post", post)
    LLMClient("http://x/v1", "", "auto").chat(
        messages=[{"role": "user", "content": "x"}],
        tools=[{"name": "mk", "parameters": {"type": "object"}}])
    assert captured["payload"]["tool_choice"] == "auto"


def test_usage_accumulates_llm_seconds(monkeypatch):
    body = {"choices": [{"finish_reason": "stop", "message": {"content": "{}"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2}}
    monkeypatch.setattr("salome_agent.llm.requests.post", _stub_post(body))
    import itertools
    tick = itertools.count(0.0, 0.5)
    monkeypatch.setattr("salome_agent.llm.time.monotonic", lambda: next(tick))
    c = LLMClient("http://x/v1", "", "auto")
    c.chat(messages=[{"role": "user", "content": "x"}])
    before = c.usage["llm_seconds"]
    c.chat(messages=[{"role": "user", "content": "y"}])
    assert c.usage["calls"] == 2
    assert before > 0 and c.usage["llm_seconds"] > before


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


def test_continuity_fields_keeps_only_the_allowlist():
    msg = {"role": "assistant", "content": "x", "tool_calls": [],
           "extra_content": {"google": {"thought_signature": "abc"}},
           "reasoning_content": "a very long private chain of thought " * 50}
    assert continuity_fields(msg) == {"extra_content": {"google": {"thought_signature": "abc"}}}
    assert continuity_fields(None) == {}
    assert continuity_fields({"content": "x"}) == {}


def test_assistant_tool_turn_carries_per_call_and_message_continuity():
    parsed = [{"name": "make_box", "args": {"dx": 1}, "id": "c1"},
              {"name": "make_box", "args": {"dx": 2}, "id": "c2"}]
    raw = {"role": "assistant", "content": None,
           "extra_content": {"msg_level": 1},
           "reasoning_content": "long thoughts",
           "tool_calls": [
               {"id": "c1", "extra_content": {"google": {"thought_signature": "SIG1"}},
                "function": {"name": "make_box", "arguments": '{"dx": 1}'}},
               {"id": "c2", "function": {"name": "make_box", "arguments": '{"dx": 2}'}}]}
    turn = assistant_tool_turn(parsed, raw)
    assert turn["extra_content"] == {"msg_level": 1}
    assert "reasoning_content" not in turn
    assert turn["tool_calls"][0]["extra_content"] == {"google": {"thought_signature": "SIG1"}}
    assert "extra_content" not in turn["tool_calls"][1]


def test_assistant_tool_turn_without_raw_message_is_a_plain_turn():
    turn = assistant_tool_turn([{"name": "mk", "args": {}, "id": "c1"}], None)
    assert turn == {"role": "assistant", "content": None, "tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": "mk", "arguments": "{}"}}]}


def test_chat_returns_raw_message_for_echoing_back(monkeypatch):
    body = {"choices": [{"finish_reason": "tool_calls", "message": {
        "content": None, "extra_content": {"google": {"thought_signature": "sig"}},
        "tool_calls": [{"id": "c1", "function": {"name": "mk", "arguments": "{}"}}]}}]}
    monkeypatch.setattr("salome_agent.llm.requests.post", _stub_post(body))
    out = LLMClient("http://x/v1", "", "auto").chat(
        messages=[{"role": "user", "content": "x"}],
        tools=[{"name": "mk", "parameters": {"type": "object"}}])
    assert out["raw_message"]["extra_content"] == {"google": {"thought_signature": "sig"}}
    # also on the prose-only (no tool call) return
    body2 = {"choices": [{"finish_reason": "stop", "message": {
        "content": "hmm", "extra_content": {"google": {"thought_signature": "sig2"}}}}]}
    monkeypatch.setattr("salome_agent.llm.requests.post", _stub_post(body2))
    out2 = LLMClient("http://x/v1", "", "auto").chat(
        messages=[{"role": "user", "content": "x"}],
        tools=[{"name": "mk", "parameters": {"type": "object"}}])
    assert out2["raw_message"]["extra_content"] == {"google": {"thought_signature": "sig2"}}


def test_planner_retry_echoes_continuity_fields_but_not_reasoning(monkeypatch):
    sent = []

    def post(url, json=None, headers=None, verify=None, timeout=None, stream=None):
        sent.append(json["messages"])
        if len(sent) == 1:
            return _FakeResp({"choices": [{"finish_reason": "stop", "message": {
                "content": "not json", "extra_content": {"sig": "S"},
                "reasoning_content": "long thoughts"}}]})
        return _FakeResp(_msg('{"stages": []}'))

    monkeypatch.setattr("salome_agent.llm.requests.post", post)
    LLMClient("http://x/v1", "", "auto").chat(messages=[{"role": "user", "content": "plan"}])
    echoed = sent[1][-2]
    assert echoed["role"] == "assistant"
    assert echoed["extra_content"] == {"sig": "S"}
    assert "reasoning_content" not in echoed


def test_chat_returns_every_tool_call_in_the_turn(monkeypatch):
    body = {"choices": [{"finish_reason": "tool_calls", "message": {
        "content": "building four walls",
        "tool_calls": [
            {"id": "c1", "function": {"name": "make_box", "arguments": '{"dx": 1}'}},
            {"id": "c2", "function": {"name": "make_box", "arguments": '{"dx": 2}'}},
            {"id": "c3", "function": {"name": "translate", "arguments": '{"dx": 5}'}},
        ]}}]}
    monkeypatch.setattr("salome_agent.llm.requests.post", _stub_post(body))
    out = LLMClient("http://x/v1", "", "auto").chat(
        messages=[{"role": "user", "content": "x"}],
        tools=[{"name": "make_box", "parameters": {"type": "object"}}])
    assert [c["name"] for c in out["tool_calls"]] == ["make_box", "make_box", "translate"]
    assert [c["args"] for c in out["tool_calls"]] == [{"dx": 1}, {"dx": 2}, {"dx": 5}]
    assert [c["id"] for c in out["tool_calls"]] == ["c1", "c2", "c3"]
    # singular keys mirror the first call
    assert out["tool_call"] == "make_box" and out["args"] == {"dx": 1}


def test_chat_returns_empty_tool_calls_when_model_only_talks(monkeypatch):
    body = {"choices": [{"finish_reason": "stop",
                         "message": {"content": "thinking", "tool_calls": []}}]}
    monkeypatch.setattr("salome_agent.llm.requests.post", _stub_post(body))
    out = LLMClient("http://x/v1", "", "auto").chat(
        messages=[{"role": "user", "content": "x"}],
        tools=[{"name": "mk", "parameters": {"type": "object"}}])
    assert out["tool_calls"] == []
    assert out["tool_call"] is None


def test_prompt_is_logged_before_the_request_is_sent(tmp_path, monkeypatch, caplog):
    import logging

    import requests as _requests

    def post(url, json=None, headers=None, verify=None, timeout=None, stream=None):
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


def test_usage_accumulates_across_calls(monkeypatch):
    bodies = [
        {"choices": [{"finish_reason": "stop",
                      "message": {"role": "assistant", "content": '{"a": 1}'}}],
         "usage": {"prompt_tokens": 100, "completion_tokens": 10}},
        {"choices": [{"finish_reason": "stop",
                      "message": {"role": "assistant", "content": '{"a": 2}'}}],
         "usage": {"prompt_tokens": 250, "completion_tokens": 25}},
    ]
    it = iter(bodies)

    def post(url, json=None, headers=None, verify=None, timeout=None, stream=None):
        return _FakeResp(next(it))

    monkeypatch.setattr("salome_agent.llm.requests.post", post)
    client = LLMClient("http://x/v1", "", "auto")
    client.chat(messages=[{"role": "user", "content": "hi"}])
    client.chat(messages=[{"role": "user", "content": "hi"}])
    assert client.usage["calls"] == 2
    assert client.usage["metered_calls"] == 2
    assert client.usage["prompt_tokens"] == 350
    assert client.usage["completion_tokens"] == 35
    assert client.usage["approx_prompt_tokens"] > 0  # chars/4 of both requests


def test_usage_tolerates_a_missing_usage_block(monkeypatch):
    body = {"choices": [{"finish_reason": "stop",
                         "message": {"role": "assistant", "content": '{"a": 1}'}}]}
    monkeypatch.setattr("salome_agent.llm.requests.post", _stub_post(body))
    client = LLMClient("http://x/v1", "", "auto")
    client.chat(messages=[{"role": "user", "content": "hi"}])
    assert client.usage["calls"] == 1
    assert client.usage["metered_calls"] == 0
    assert client.usage["prompt_tokens"] == 0
    assert client.usage["approx_prompt_tokens"] > 0  # still counted with no usage block


def test_trace_line_carries_approx_input_tokens_including_tool_schemas(monkeypatch, tmp_path):
    body = {"choices": [{"finish_reason": "tool_calls", "message": {
        "tool_calls": [{"id": "c1", "function": {"name": "mk", "arguments": "{}"}}]}}]}
    monkeypatch.setattr("salome_agent.llm.requests.post", _stub_post(body))
    trace = tmp_path / "llm_trace.jsonl"
    client = LLMClient("http://x/v1", "", "auto", trace_path=str(trace))
    big_schema = {"type": "object", "properties": {
        k: {"type": "string", "description": "x" * 200} for k in range(20)}}
    client.chat(messages=[{"role": "user", "content": "hi"}],
                tools=[{"name": "mk", "parameters": big_schema}])
    entry = json.loads(trace.read_text().splitlines()[0])
    approx = entry["request"]["approx_input_tokens"]
    assert approx > 1000  # the fat tool schema dominates, not the 2-word message
    assert client.usage["approx_prompt_tokens"] == approx


def test_trace_line_records_extra_body(monkeypatch, tmp_path):
    body = {"choices": [{"finish_reason": "stop",
                         "message": {"role": "assistant", "content": '{"a": 1}'}}]}
    monkeypatch.setattr("salome_agent.llm.requests.post", _stub_post(body))
    trace = tmp_path / "llm_trace.jsonl"
    LLMClient("http://x/v1", "", "auto", trace_path=str(trace),
              extra_body={"reasoning_effort": "high"}).chat(
        messages=[{"role": "user", "content": "hi"}])
    entry = json.loads(trace.read_text().splitlines()[0])
    assert entry["request"]["extra_body"] == {"reasoning_effort": "high"}


def test_trace_line_omits_extra_body_when_unset(monkeypatch, tmp_path):
    body = {"choices": [{"finish_reason": "stop",
                         "message": {"role": "assistant", "content": '{"a": 1}'}}]}
    monkeypatch.setattr("salome_agent.llm.requests.post", _stub_post(body))
    trace = tmp_path / "llm_trace.jsonl"
    LLMClient("http://x/v1", "", "auto", trace_path=str(trace)).chat(
        messages=[{"role": "user", "content": "hi"}])
    entry = json.loads(trace.read_text().splitlines()[0])
    assert "extra_body" not in entry["request"]


# --- streaming (SSE) ---------------------------------------------------------

def _sse_post(lines):
    def post(url, json=None, headers=None, verify=None, timeout=None, stream=None):
        assert json["stream"] is True
        assert json["stream_options"] == {"include_usage": True}
        return _FakeResp(None, lines=lines)
    return post


def test_stream_reassembles_content_fragments(monkeypatch):
    lines = [
        'data: {"choices": [{"delta": {"role": "assistant", "content": "Hel"}}]}',
        'data: {"choices": [{"delta": {"content": "lo "}}]}',
        'data: {"choices": [{"delta": {"content": "world"}, "finish_reason": "stop"}]}',
        "data: [DONE]",
    ]
    monkeypatch.setattr("salome_agent.llm.requests.post", _sse_post(lines))
    out = LLMClient("http://x/v1", "", "auto").chat(
        messages=[{"role": "user", "content": "x"}],
        tools=[{"name": "mk", "parameters": {"type": "object"}}])
    assert out["content"] == "Hello world"
    assert out["tool_call"] is None


def test_stream_reassembles_tool_call_argument_fragments(monkeypatch):
    lines = [
        'data: {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c1", '
        '"function": {"name": "make_box", "arguments": "{\\"dx\\": "}}]}}]}',
        'data: {"choices": [{"delta": {"tool_calls": [{"index": 0, '
        '"function": {"arguments": "200}"}}]}}]}',
        'data: {"choices": [{"delta": {"tool_calls": [{"index": 1, "id": "c2", '
        '"function": {"name": "translate", "arguments": "{\\"dz\\": 5}"}}]}}]}',
        'data: {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]}',
        "data: [DONE]",
    ]
    monkeypatch.setattr("salome_agent.llm.requests.post", _sse_post(lines))
    out = LLMClient("http://x/v1", "", "auto").chat(
        messages=[{"role": "user", "content": "x"}],
        tools=[{"name": "make_box", "parameters": {"type": "object"}}])
    assert [c["name"] for c in out["tool_calls"]] == ["make_box", "translate"]
    assert [c["args"] for c in out["tool_calls"]] == [{"dx": 200}, {"dz": 5}]
    assert [c["id"] for c in out["tool_calls"]] == ["c1", "c2"]


def test_stream_gemini_whole_and_repeated_tool_args_do_not_concatenate(monkeypatch):
    # Gemini's OpenAI-compat shim sends the whole `arguments` object in one
    # delta and sometimes repeats it — appending would make "{...}{...}".
    lines = [
        'data: {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c1", '
        '"function": {"name": "make_box", "arguments": "{\\"dx\\": 10, \\"dy\\": 20}"}}]}}]}',
        'data: {"choices": [{"delta": {"tool_calls": [{"index": 0, '
        '"function": {"arguments": "{\\"dx\\": 10, \\"dy\\": 20}"}}]}}]}',
        'data: {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]}',
        "data: [DONE]",
    ]
    monkeypatch.setattr("salome_agent.llm.requests.post", _sse_post(lines))
    out = LLMClient("http://x/v1", "", "auto").chat(
        messages=[{"role": "user", "content": "x"}],
        tools=[{"name": "make_box", "parameters": {"type": "object"}}], module="geom")
    assert out["tool_calls"][0]["args"] == {"dx": 10, "dy": 20}


def test_loads_first_object_tolerates_a_trailing_repeat():
    from salome_agent.llm import _loads_first_object
    assert _loads_first_object('{"a": 1}{"a": 1}') == {"a": 1}
    assert _loads_first_object('{"a": 1}') == {"a": 1}


def test_stream_reassembles_reasoning_and_continuity_fragments(monkeypatch):
    lines = [
        'data: {"choices": [{"delta": {"reasoning_content": "first "}}]}',
        'data: {"choices": [{"delta": {"reasoning_content": "second"}}]}',
        'data: {"choices": [{"delta": {"content": "answer", '
        '"extra_content": {"google": {"thought_signature": "SIG"}}}, '
        '"finish_reason": "stop"}]}',
        "data: [DONE]",
    ]
    monkeypatch.setattr("salome_agent.llm.requests.post", _sse_post(lines))
    out = LLMClient("http://x/v1", "", "auto").chat(
        messages=[{"role": "user", "content": "x"}],
        tools=[{"name": "mk", "parameters": {"type": "object"}}])
    assert out["raw_message"]["reasoning_content"] == "first second"
    assert out["raw_message"]["extra_content"] == {"google": {"thought_signature": "SIG"}}


def test_stream_reads_usage_from_trailing_frame(monkeypatch):
    lines = [
        'data: {"choices": [{"delta": {"content": "hi"}, "finish_reason": "stop"}]}',
        'data: {"choices": [], "usage": {"prompt_tokens": 40, "completion_tokens": 7}}',
        "data: [DONE]",
    ]
    monkeypatch.setattr("salome_agent.llm.requests.post", _sse_post(lines))
    c = LLMClient("http://x/v1", "", "auto")
    c.chat(messages=[{"role": "user", "content": "x"}],
           tools=[{"name": "mk", "parameters": {"type": "object"}}])
    assert c.usage["prompt_tokens"] == 40
    assert c.usage["completion_tokens"] == 7
    assert c.usage["metered_calls"] == 1


def test_stream_error_frame_raises(monkeypatch):
    lines = [
        'data: {"choices": [{"delta": {"content": "par"}}]}',
        'data: {"error": {"message": "context length exceeded", "code": 400}}',
        "data: [DONE]",
    ]
    monkeypatch.setattr("salome_agent.llm.requests.post", _sse_post(lines))
    with pytest.raises(RuntimeError, match="error frame"):
        LLMClient("http://x/v1", "", "auto").chat(
            messages=[{"role": "user", "content": "x"}],
            tools=[{"name": "mk", "parameters": {"type": "object"}}])


def test_stream_empty_without_finish_reason_is_retried(monkeypatch):
    good = [
        'data: {"choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}]}',
        "data: [DONE]",
    ]
    # a role-only frame, no content / tool call / finish / [DONE] — a real cut
    empty = ['data: {"choices": [{"delta": {"role": "assistant"}}]}']
    calls = []

    def post(url, json=None, headers=None, verify=None, timeout=None, stream=None):
        calls.append(1)
        return _FakeResp(None, lines=empty if len(calls) == 1 else good)

    monkeypatch.setattr("salome_agent.llm.requests.post", post)
    monkeypatch.setattr("salome_agent.llm.time.sleep", lambda s: None)
    out = LLMClient("http://x/v1", "", "auto").chat(
        messages=[{"role": "user", "content": "x"}],
        tools=[{"name": "mk", "parameters": {"type": "object"}}])
    assert out["content"] == "ok"
    assert len(calls) == 2


def test_stream_with_output_but_no_terminator_is_kept(monkeypatch):
    # some OpenAI-compat shims omit [DONE] and the trailing finish_reason
    lines = ['data: {"choices": [{"delta": {"content": "the whole answer"}}]}']
    calls = []

    def post(url, json=None, headers=None, verify=None, timeout=None, stream=None):
        calls.append(1)
        return _FakeResp(None, lines=lines)

    monkeypatch.setattr("salome_agent.llm.requests.post", post)
    monkeypatch.setattr("salome_agent.llm.time.sleep", lambda s: None)
    out = LLMClient("http://x/v1", "", "auto").chat(
        messages=[{"role": "user", "content": "x"}],
        tools=[{"name": "mk", "parameters": {"type": "object"}}])
    assert out["content"] == "the whole answer"
    assert len(calls) == 1


def test_trace_records_ttft(monkeypatch, tmp_path):
    body = {"choices": [{"finish_reason": "stop",
                         "message": {"role": "assistant", "content": '{"a": 1}'}}]}
    monkeypatch.setattr("salome_agent.llm.requests.post", _stub_post(body))
    trace = tmp_path / "t.jsonl"
    LLMClient("http://x/v1", "", "auto", trace_path=str(trace)).chat(
        messages=[{"role": "user", "content": "hi"}])
    entry = json.loads(trace.read_text().splitlines()[0])
    assert "ttft_s" in entry["response"]
