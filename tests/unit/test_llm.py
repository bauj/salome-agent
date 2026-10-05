import json

import pytest
import requests

from salome_agent.llm import (LLMCancelled, LLMClient, _parse_json_content,
                              assistant_tool_turn, continuity_fields, with_images)


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


def test_with_images_none_or_empty_returns_plain_text():
    assert with_images("make a box", None) == "make a box"
    assert with_images("make a box", []) == "make a box"


def test_with_images_builds_multimodal_parts():
    content = with_images("make a box", ["data:image/png;base64,AAAA"])
    assert content == [
        {"type": "text", "text": "make a box"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
    ]


def test_with_images_supports_several_images():
    content = with_images("make a box", ["data:image/png;base64,AAAA",
                                         "data:image/bmp;base64,BBBB"])
    assert content == [
        {"type": "text", "text": "make a box"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        {"type": "image_url", "image_url": {"url": "data:image/bmp;base64,BBBB"}},
    ]


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


def test_chat_verifies_against_ssl_cert_file_with_home_expanded(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / "ca.pem").write_text("")
    captured = {}

    def post(url, json=None, headers=None, verify=None, timeout=None, stream=None):
        captured["verify"] = verify
        return _FakeResp({"choices": [{"finish_reason": "stop",
                                       "message": {"content": '{"ok": 1}'}}]})

    monkeypatch.setattr("salome_agent.llm.requests.post", post)
    LLMClient("http://x/v1", "", "auto", ssl_cert_file="~/ca.pem").chat(
        messages=[{"role": "user", "content": "x"}])
    assert captured["verify"] == str(tmp_path / "ca.pem")


def test_missing_ssl_cert_file_fails_at_construction(tmp_path):
    with pytest.raises(FileNotFoundError, match="SSL certificate file not found"):
        LLMClient("http://x/v1", "", "auto", ssl_cert_file=str(tmp_path / "nope.pem"))


def test_no_ssl_cert_file_verifies_against_system_store():
    assert LLMClient("http://x/v1", "", "auto").ssl_cert_file is True


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


def _capture_request(monkeypatch):
    seen = []

    def post(url, json=None, headers=None, verify=None, timeout=None, stream=None):
        seen.append({"url": url, "payload": json, "headers": headers, "timeout": timeout})
        return _FakeResp({"choices": [{"finish_reason": "stop", "message": {
            "role": "assistant", "content": '{"a": 1}'}}]})

    monkeypatch.setattr("salome_agent.llm.requests.post", post)
    return seen


def test_module_override_sends_its_calls_to_its_own_endpoint_and_model(monkeypatch):
    seen = _capture_request(monkeypatch)
    client = LLMClient("http://shared/v1", "k1", "small",
                       modules={"vision": {"base_url": "http://other/v1/",
                                           "api_key": "k2", "model": "big"}})
    client.chat(messages=[{"role": "user", "content": "x"}], module="vision")
    client.chat(messages=[{"role": "user", "content": "x"}], module="planner")
    vision, planner = seen
    assert vision["url"] == "http://other/v1/chat/completions"
    assert vision["payload"]["model"] == "big"
    assert vision["headers"]["Authorization"] == "Bearer k2"
    assert planner["url"] == "http://shared/v1/chat/completions"
    assert planner["payload"]["model"] == "small"
    assert planner["headers"]["Authorization"] == "Bearer k1"


def test_module_override_on_another_host_does_not_inherit_credentials(monkeypatch):
    seen = _capture_request(monkeypatch)
    LLMClient("http://shared/v1", "k1", "small", extra_headers={"X-Token": "t"},
              extra_body={"reasoning_effort": "high"},
              modules={"vision": {"base_url": "http://other/v1", "model": "big"}}).chat(
        messages=[{"role": "user", "content": "x"}], module="vision")
    req = seen[0]
    assert req["headers"] == {}
    assert "reasoning_effort" not in req["payload"]


def test_module_override_keeps_shared_values_it_leaves_out(monkeypatch):
    seen = _capture_request(monkeypatch)
    LLMClient("http://shared/v1", "k1", "small", temperature=0.3,
              extra_headers={"X-Run": "r"}, extra_body={"reasoning_effort": "high"},
              modules={"vision": {"model": "big"}}).chat(
        messages=[{"role": "user", "content": "x"}], module="vision")
    req = seen[0]
    assert req["url"] == "http://shared/v1/chat/completions"
    assert req["payload"]["model"] == "big"
    assert req["payload"]["temperature"] == 0.3
    assert req["payload"]["reasoning_effort"] == "high"
    assert req["headers"] == {"Authorization": "Bearer k1", "X-Run": "r"}


def test_module_override_extra_body_wins_over_module_extra_body(monkeypatch):
    seen = _capture_request(monkeypatch)
    LLMClient("http://x/v1", "", "auto",
              module_extra_body={"vision": {"reasoning_effort": "low"}},
              modules={"vision": {"extra_body": {"reasoning_effort": "high"}}}).chat(
        messages=[{"role": "user", "content": "x"}], module="vision")
    assert seen[0]["payload"]["reasoning_effort"] == "high"


def test_module_override_without_extra_body_uses_module_extra_body(monkeypatch):
    seen = _capture_request(monkeypatch)
    LLMClient("http://x/v1", "", "auto", extra_body={"reasoning_effort": "high"},
              module_extra_body={"vision": {"reasoning_effort": "low"}},
              modules={"vision": {"model": "big"}}).chat(
        messages=[{"role": "user", "content": "x"}], module="vision")
    assert seen[0]["payload"]["reasoning_effort"] == "low"


def test_module_override_timeout_sets_its_read_gap(monkeypatch):
    seen = _capture_request(monkeypatch)
    LLMClient("http://x/v1", "", "auto", request_timeout=600,
              modules={"vision": {"timeout": 20}}).chat(
        messages=[{"role": "user", "content": "x"}], module="vision")
    assert seen[0]["timeout"][1] == 20


def test_module_override_bad_cert_fails_at_startup(tmp_path):
    with pytest.raises(FileNotFoundError):
        LLMClient("http://x/v1", "", "auto",
                  modules={"vision": {"ssl_cert_file": str(tmp_path / "nope.pem")}})


def test_trace_records_the_module_model_and_base_url(monkeypatch, tmp_path):
    _capture_request(monkeypatch)
    trace = tmp_path / "t.jsonl"
    LLMClient("http://x/v1", "", "small", trace_path=str(trace),
              modules={"vision": {"base_url": "http://other/v1", "model": "big"}}).chat(
        messages=[{"role": "user", "content": "x"}], module="vision")
    req = json.loads(trace.read_text().splitlines()[0])["request"]
    assert req["model"] == "big"
    assert req["base_url"] == "http://other/v1"


def test_build_llm_reads_modules_and_fills_run_id_in_their_headers(monkeypatch):
    from salome_agent.cli import build_llm

    seen = _capture_request(monkeypatch)
    cfg = {"llm": {"base_url": "http://x/v1", "model": "small",
                   "modules": {"vision": {"model": "big",
                                          "extra_headers": {"X-Session": "{run_id}"}}}}}
    build_llm(cfg, run_id="abc").chat(
        messages=[{"role": "user", "content": "x"}], module="vision")
    assert seen[0]["payload"]["model"] == "big"
    assert seen[0]["headers"]["X-Session"] == "abc"


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


def test_stream_past_total_budget_raises_and_is_not_retried(monkeypatch):
    # a stream that keeps emitting frames forever (runaway reasoning) — the
    # per-read gap never trips, so only the total-budget deadline can stop it.
    never_ends = ['data: {"choices": [{"delta": {"reasoning_content": "hmm "}}]}'] * 50
    calls = []

    def post(url, json=None, headers=None, verify=None, timeout=None, stream=None):
        calls.append(1)
        return _FakeResp(None, lines=never_ends)

    # deterministic clock: every monotonic() call jumps 10s
    ticks = iter(range(0, 100_000, 10))
    monkeypatch.setattr("salome_agent.llm.time.monotonic", lambda: next(ticks))
    monkeypatch.setattr("salome_agent.llm.time.sleep", lambda s: None)
    monkeypatch.setattr("salome_agent.llm.requests.post", post)

    from salome_agent.llm import StreamBudgetExceeded
    client = LLMClient("http://x/v1", "", "auto", request_timeout=30)
    with pytest.raises(StreamBudgetExceeded):
        client.chat(messages=[{"role": "user", "content": "x"}],
                    tools=[{"name": "mk", "parameters": {"type": "object"}}])
    assert len(calls) == 1  # fatal — not retried


def test_stream_read_gap_capped_below_total_budget(monkeypatch):
    from salome_agent.llm import _STREAM_READ_GAP_CAP
    seen = {}

    def post(url, json=None, headers=None, verify=None, timeout=None, stream=None):
        seen["timeout"] = timeout
        return _FakeResp(None, lines=[
            'data: {"choices": [{"delta": {"content": "{\\"a\\": 1}"}, '
            '"finish_reason": "stop"}]}',
            "data: [DONE]"])

    monkeypatch.setattr("salome_agent.llm.requests.post", post)
    LLMClient("http://x/v1", "", "auto", request_timeout=600).chat(
        messages=[{"role": "user", "content": "x"}])
    assert seen["timeout"][1] == _STREAM_READ_GAP_CAP  # not 600


def test_trace_records_ttft(monkeypatch, tmp_path):
    body = {"choices": [{"finish_reason": "stop",
                         "message": {"role": "assistant", "content": '{"a": 1}'}}]}
    monkeypatch.setattr("salome_agent.llm.requests.post", _stub_post(body))
    trace = tmp_path / "t.jsonl"
    LLMClient("http://x/v1", "", "auto", trace_path=str(trace)).chat(
        messages=[{"role": "user", "content": "hi"}])
    entry = json.loads(trace.read_text().splitlines()[0])
    assert "ttft_s" in entry["response"]


def test_streamed_non_ascii_survives_decoding(monkeypatch):
    """An SSE body with no charset makes requests assume ISO-8859-1 (its text/*
    default); UTF-8 "×" then came back as "Ã\x97" and, echoed into later
    prompts, compounded (PR #44)."""
    import io

    # raw UTF-8 on the wire, as real providers send it — not \uXXXX escapes
    frame = json.dumps({"choices": [{"delta": {"role": "assistant",
                                               "content": "200×120 mm — 45°"},
                                     "finish_reason": "stop"}]}, ensure_ascii=False)
    body = f"data: {frame}\n\ndata: [DONE]\n\n".encode("utf-8")

    def post(url, json=None, headers=None, verify=None, timeout=None, stream=None):
        resp = requests.Response()
        resp.status_code = 200
        resp.headers["Content-Type"] = "text/event-stream"
        # what HTTPAdapter.build_response does
        resp.encoding = requests.utils.get_encoding_from_headers(resp.headers)
        resp.raw = io.BytesIO(body)
        return resp

    monkeypatch.setattr(requests, "post", post)
    out = LLMClient(base_url="http://x", api_key="k", model="m").chat(
        messages=[{"role": "user", "content": "hi"}],
        tools=[{"name": "t", "parameters": {"type": "object", "properties": {}}}])
    assert out["content"] == "200×120 mm — 45°"


def test_cancel_mid_stream_stops_reading_and_closes_the_response(monkeypatch):
    import threading
    cancel = threading.Event()
    read = []
    closed = threading.Event()

    class _Resp(_FakeResp):
        def iter_lines(self, chunk_size=512, decode_unicode=False):
            for i in range(1000):
                read.append(i)
                if i == 3:
                    cancel.set()
                yield "data: " + json.dumps(
                    {"choices": [{"delta": {"reasoning_content": "hmm "}}]})

        def close(self):
            closed.set()

    monkeypatch.setattr("salome_agent.llm.requests.post",
                        lambda *a, **k: _Resp(None))
    client = LLMClient("http://x/v1", "", "auto")
    client.cancel = cancel
    with pytest.raises(LLMCancelled):
        client.chat(messages=[{"role": "user", "content": "plan"}])
    assert len(read) == 4
    assert closed.wait(1)


def test_cancel_already_set_sends_no_request(monkeypatch):
    import threading
    posts = []
    monkeypatch.setattr("salome_agent.llm.requests.post",
                        lambda *a, **k: posts.append(1))
    client = LLMClient("http://x/v1", "", "auto")
    client.cancel = threading.Event()
    client.cancel.set()
    with pytest.raises(LLMCancelled):
        client.chat(messages=[{"role": "user", "content": "plan"}])
    assert posts == []


def test_cancel_while_waiting_for_response_headers_is_immediate(monkeypatch):
    import threading
    import time
    cancel, release = threading.Event(), threading.Event()

    def post(*a, **k):
        release.wait(10)
        return _FakeResp({})

    monkeypatch.setattr("salome_agent.llm.requests.post", post)
    client = LLMClient("http://x/v1", "", "auto")
    client.cancel = cancel
    threading.Timer(0.3, cancel.set).start()
    t0 = time.monotonic()
    try:
        with pytest.raises(LLMCancelled):
            client.chat(messages=[{"role": "user", "content": "plan"}])
        assert time.monotonic() - t0 < 1.0
    finally:
        release.set()


def test_cancel_during_a_silent_stream_gap_is_immediate(monkeypatch):
    import threading
    import time
    cancel, release = threading.Event(), threading.Event()
    closed = threading.Event()

    class _Resp(_FakeResp):
        def iter_lines(self, chunk_size=512, decode_unicode=False):
            yield "data: " + json.dumps(
                {"choices": [{"delta": {"reasoning_content": "hmm "}}]})
            release.wait(10)
            yield "data: [DONE]"

        def close(self):  # like http.client: waits for the parked read
            closed.set()
            release.wait(10)

    monkeypatch.setattr("salome_agent.llm.requests.post",
                        lambda *a, **k: _Resp(None))
    client = LLMClient("http://x/v1", "", "auto")
    client.cancel = cancel
    threading.Timer(0.3, cancel.set).start()
    t0 = time.monotonic()
    try:
        with pytest.raises(LLMCancelled):
            client.chat(messages=[{"role": "user", "content": "plan"}])
        assert time.monotonic() - t0 < 1.0
        assert closed.wait(1)
    finally:
        release.set()


def test_blocking_io_errors_still_reach_the_caller_when_cancellable(monkeypatch):
    import threading
    posts = []

    def post(*a, **k):
        posts.append(1)
        raise requests.ConnectionError("refused")

    monkeypatch.setattr("salome_agent.llm.requests.post", post)
    monkeypatch.setattr("salome_agent.llm._backoff_seconds", lambda attempt: 0)
    client = LLMClient("http://x/v1", "", "auto")
    client.cancel = threading.Event()
    with pytest.raises(requests.ConnectionError):
        client.chat(messages=[{"role": "user", "content": "plan"}])
    assert len(posts) == client.max_retries + 1


def test_cancel_during_a_retry_wait_gives_up_without_retrying(monkeypatch):
    import threading
    cancel = threading.Event()
    posts = []

    def post(*a, **k):
        posts.append(1)
        cancel.set()
        return _FakeResp({}, status_code=429, headers={"Retry-After": "30"})

    monkeypatch.setattr("salome_agent.llm.requests.post", post)
    client = LLMClient("http://x/v1", "", "auto")
    client.cancel = cancel
    with pytest.raises(LLMCancelled):
        client.chat(messages=[{"role": "user", "content": "plan"}])
    assert posts == [1]


def _hooked(monkeypatch, lines):
    monkeypatch.setattr("salome_agent.llm.requests.post",
                        lambda *a, **k: _FakeResp(None, lines=lines))
    client = LLMClient("http://x/v1", "", "auto")
    seen = []
    client.on_stream = seen.append
    return client, seen


def _delta(**d):
    return "data: " + json.dumps({"choices": [{"delta": d}]})


def test_stream_hook_gets_reasoning_and_prose_on_a_tool_call(monkeypatch):
    lines = [_delta(reasoning_content="think "), _delta(reasoning_content="more"),
             _delta(content="I will box."),
             "data: " + json.dumps({"choices": [{"delta": {}, "finish_reason": "stop"}]}),
             "data: [DONE]"]
    client, seen = _hooked(monkeypatch, lines)
    client.chat(messages=[{"role": "user", "content": "x"}],
                tools=[{"name": "make_box"}], module="geom")
    assert seen == [{"type": "start", "module": "geom"},
                    {"type": "delta", "channel": "reasoning", "text": "think "},
                    {"type": "delta", "channel": "reasoning", "text": "more"},
                    {"type": "delta", "channel": "prose", "text": "I will box."},
                    {"type": "end"}]


def test_stream_hook_skips_planner_content(monkeypatch):
    lines = [_delta(reasoning_content="plan it"), _delta(content='{"stages": []}'),
             "data: [DONE]"]
    client, seen = _hooked(monkeypatch, lines)
    client.chat(messages=[{"role": "user", "content": "x"}], module="planner")
    assert [e for e in seen if e["type"] == "delta"] == [
        {"type": "delta", "channel": "reasoning", "text": "plan it"}]
    assert seen[0] == {"type": "start", "module": "planner"}
    assert seen[-1] == {"type": "end"}


def test_stream_hook_ends_on_error_and_a_raising_hook_is_ignored(monkeypatch):
    monkeypatch.setattr("salome_agent.llm.requests.post",
                        lambda *a, **k: _FakeResp(None, lines=[_delta(reasoning_content="a"),
                                                               "data: [DONE]"]))
    client = LLMClient("http://x/v1", "", "auto", json_retries=0)
    seen = []

    def hook(ev):
        seen.append(ev["type"])
        raise RuntimeError("display broke")

    client.on_stream = hook
    with pytest.raises(ValueError):          # no JSON content -> planner parse error
        client.chat(messages=[{"role": "user", "content": "x"}], module="planner")
    assert seen == ["start", "delta", "end"]


def test_length_finish_reason_warns_about_truncation(monkeypatch, caplog):
    body = {"choices": [{"message": {"content": "{}"}, "finish_reason": "length"}],
            "usage": {"prompt_tokens": 2700, "completion_tokens": 1396}}
    monkeypatch.setattr("salome_agent.llm.requests.post", _stub_post(body))
    LLMClient("http://x/v1", "k", "m", ssl_cert_file="").chat(
        messages=[{"role": "user", "content": "hi"}])
    assert "cut off" in caplog.text and "2700" in caplog.text


def test_images_are_redacted_from_the_trace_and_the_token_estimate(monkeypatch, tmp_path):
    import json

    uri = "data:image/png;base64," + "A" * 40000
    body = {"choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}]}
    monkeypatch.setattr("salome_agent.llm.requests.post", _stub_post(body))
    trace = tmp_path / "llm_trace.jsonl"
    client = LLMClient("http://x/v1", "k", "m", ssl_cert_file="", trace_path=str(trace))
    client.chat(messages=[{"role": "user", "content": with_images("look", [uri])}])
    assert client.usage["approx_prompt_tokens"] < 1000
    entry = trace.read_text()
    assert "A" * 100 not in entry
    part = json.loads(entry)["request"]["messages"][0]["content"][1]
    assert part["image_url"]["url"] == f"<image, {len(uri)} chars>"


# --- task 278: interrupted calls keep their reasoning ------------------------

_THINK = 'data: {"choices": [{"delta": {"reasoning_content": "hmm "}}]}'


def _trace_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_budget_overrun_writes_an_incomplete_trace_entry(tmp_path, monkeypatch):
    ticks = iter(range(0, 100_000, 10))
    monkeypatch.setattr("salome_agent.llm.time.monotonic", lambda: next(ticks))
    monkeypatch.setattr("salome_agent.llm.requests.post",
                        lambda *a, **k: _FakeResp(None, lines=[_THINK] * 50))
    from salome_agent.llm import StreamBudgetExceeded
    trace = tmp_path / "llm_trace.jsonl"
    client = LLMClient("http://x/v1", "", "auto", request_timeout=30, trace_path=str(trace))
    with pytest.raises(StreamBudgetExceeded):
        client.chat(messages=[{"role": "user", "content": "x"}])
    [entry] = _trace_lines(trace)
    resp = entry["response"]
    assert "total budget" in resp["incomplete"]
    assert resp["message"]["reasoning_content"].startswith("hmm")
    assert resp["frames"] >= 1 and resp["elapsed_s"] > 0
    assert entry["request"]["messages"] == [{"role": "user", "content": "x"}]


def test_cancel_writes_an_incomplete_trace_entry(tmp_path, monkeypatch):
    import threading
    cancel = threading.Event()

    class _Resp(_FakeResp):
        def iter_lines(self, chunk_size=512, decode_unicode=False):
            for i in range(10):
                if i == 3:
                    cancel.set()
                yield _THINK

    monkeypatch.setattr("salome_agent.llm.requests.post", lambda *a, **k: _Resp(None))
    trace = tmp_path / "llm_trace.jsonl"
    client = LLMClient("http://x/v1", "", "auto", trace_path=str(trace))
    client.cancel = cancel
    with pytest.raises(LLMCancelled):
        client.chat(messages=[{"role": "user", "content": "x"}])
    [entry] = _trace_lines(trace)
    assert "cancelled" in entry["response"]["incomplete"]
    assert entry["response"]["message"]["reasoning_content"] == "hmm hmm hmm "


def test_dropped_attempt_is_traced_then_the_retry(tmp_path, monkeypatch):
    streams = iter([
        [_THINK, 'data: {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c1", '
                 '"function": {"name": "mk", "arguments": "{\\"a\\""}}]}}]}'],
        ['data: {"choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}]}',
         "data: [DONE]"],
    ])

    class _Resp(_FakeResp):
        def __init__(self, lines):
            super().__init__(None, lines=lines)

        def iter_lines(self, chunk_size=512, decode_unicode=False):
            yield from self._lines
            if "[DONE]" not in self._lines[-1]:
                raise requests.ConnectionError("connection reset")

    monkeypatch.setattr("salome_agent.llm.requests.post", lambda *a, **k: _Resp(next(streams)))
    monkeypatch.setattr("salome_agent.llm.time.sleep", lambda s: None)
    trace = tmp_path / "llm_trace.jsonl"
    out = LLMClient("http://x/v1", "", "auto", trace_path=str(trace)).chat(
        messages=[{"role": "user", "content": "x"}],
        tools=[{"name": "mk", "parameters": {"type": "object"}}])
    assert out["content"] == "ok"
    first, second = _trace_lines(trace)
    assert "connection reset" in first["response"]["incomplete"]
    assert first["response"]["attempt"] == 1
    partial = first["response"]["message"]
    assert partial["reasoning_content"] == "hmm "
    assert partial["tool_calls"][0]["function"] == {"name": "mk", "arguments": '{"a"'}
    assert "incomplete" not in second["response"]
    assert second["response"]["message"]["content"] == "ok"


def test_live_file_gets_reasoning_and_content_as_they_stream(tmp_path, monkeypatch):
    lines = [_THINK, _THINK,
             'data: {"choices": [{"delta": {"content": "{\\"stages\\": []}"}, '
             '"finish_reason": "stop"}]}',
             "data: [DONE]"]
    monkeypatch.setattr("salome_agent.llm.requests.post", _sse_post(lines))
    client = LLMClient("http://x/v1", "", "auto", trace_path=str(tmp_path / "llm_trace.jsonl"))
    client.chat(messages=[{"role": "user", "content": "x"}], module="planner")
    live = (tmp_path / "llm_live.txt").read_text()
    assert "planner" in live and "attempt 1" in live
    assert "hmm hmm " in live and '{"stages": []}' in live
    assert live.index("hmm") < live.index("stages")
    assert "finish=stop" in live


def test_live_file_creates_the_run_dir_for_the_first_call(tmp_path, monkeypatch):
    lines = [_THINK, 'data: {"choices": [{"delta": {"content": "{}"}, "finish_reason": "stop"}]}',
             "data: [DONE]"]
    monkeypatch.setattr("salome_agent.llm.requests.post", _sse_post(lines))
    run_dir = tmp_path / "run"
    client = LLMClient("http://x/v1", "", "auto", trace_path=str(run_dir / "llm_trace.jsonl"))
    client.chat(messages=[{"role": "user", "content": "x"}], module="planner")
    live = (run_dir / "llm_live.txt").read_text()
    assert "planner" in live and "hmm" in live


def test_live_file_marks_a_cut_call(tmp_path, monkeypatch):
    ticks = iter(range(0, 100_000, 10))
    monkeypatch.setattr("salome_agent.llm.time.monotonic", lambda: next(ticks))
    monkeypatch.setattr("salome_agent.llm.requests.post",
                        lambda *a, **k: _FakeResp(None, lines=[_THINK] * 50))
    from salome_agent.llm import StreamBudgetExceeded
    client = LLMClient("http://x/v1", "", "auto", request_timeout=30,
                       trace_path=str(tmp_path / "llm_trace.jsonl"))
    with pytest.raises(StreamBudgetExceeded):
        client.chat(messages=[{"role": "user", "content": "x"}])
    live = (tmp_path / "llm_live.txt").read_text()
    assert "hmm" in live and "incomplete" in live


def test_no_trace_path_writes_no_live_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("salome_agent.llm.requests.post",
                        lambda *a, **k: _FakeResp(None, lines=[_THINK] * 3))
    with pytest.raises(Exception):
        LLMClient("http://x/v1", "", "auto", max_retries=0).chat(
            messages=[{"role": "user", "content": "x"}])
    assert list(tmp_path.iterdir()) == []


def _reasoning_lines(chunks, then=None):
    """SSE lines streaming `chunks` as reasoning deltas, then the `then` delta."""
    lines = ["data: " + json.dumps({"choices": [{"delta": {"reasoning_content": c}}]})
             for c in chunks]
    if then is not None:
        lines.append("data: " + json.dumps({"choices": [{"delta": then,
                                                         "finish_reason": "stop"}]}))
    lines.append("data: [DONE]")
    return lines


_MK_TOOL = [{"name": "mk", "parameters": {"type": "object"}}]


def test_reasoning_budget_cuts_a_tool_call_that_only_deliberates(monkeypatch, tmp_path):
    from salome_agent.llm import ReasoningBudgetExceeded
    calls = []

    def post(url, json=None, headers=None, verify=None, timeout=None, stream=None):
        calls.append(1)
        return _FakeResp(None, lines=_reasoning_lines(["x" * 400] * 10))

    monkeypatch.setattr("salome_agent.llm.requests.post", post)
    trace = tmp_path / "t.jsonl"
    client = LLMClient("http://x/v1", "", "auto", reasoning_budget=500,
                       trace_path=str(trace))
    with pytest.raises(ReasoningBudgetExceeded) as e:
        client.chat(messages=[{"role": "user", "content": "x"}], tools=_MK_TOOL)
    assert e.value.tokens > 500
    assert e.value.reasoning.startswith("x" * 400)
    assert len(calls) == 1  # not retried as an endpoint failure
    assert "incomplete" in trace.read_text()


def test_reasoning_budget_stops_counting_once_a_tool_call_starts(monkeypatch):
    tc = {"tool_calls": [{"index": 0, "id": "c1", "type": "function",
                          "function": {"name": "mk", "arguments": "{}"}}]}
    lines = (_reasoning_lines(["x" * 400] * 4)[:-1]
             + ["data: " + json.dumps({"choices": [{"delta": tc}]})]
             + _reasoning_lines(["x" * 400] * 10, then={}))
    monkeypatch.setattr("salome_agent.llm.requests.post",
                        lambda *a, **k: _FakeResp(None, lines=lines))
    out = LLMClient("http://x/v1", "", "auto", reasoning_budget=500).chat(
        messages=[{"role": "user", "content": "x"}], tools=_MK_TOOL)
    assert out["tool_call"] == "mk"


def test_reasoning_budget_zero_is_off(monkeypatch):
    tc = {"tool_calls": [{"index": 0, "id": "c1", "type": "function",
                          "function": {"name": "mk", "arguments": "{}"}}]}
    monkeypatch.setattr("salome_agent.llm.requests.post",
                        lambda *a, **k: _FakeResp(None, lines=_reasoning_lines(
                            ["x" * 4000] * 10, then=tc)))
    out = LLMClient("http://x/v1", "", "auto", reasoning_budget=0).chat(
        messages=[{"role": "user", "content": "x"}], tools=_MK_TOOL)
    assert out["tool_call"] == "mk"


def test_reasoning_budget_does_not_apply_without_tools(monkeypatch):
    monkeypatch.setattr("salome_agent.llm.requests.post",
                        lambda *a, **k: _FakeResp(None, lines=_reasoning_lines(
                            ["x" * 4000] * 10, then={"content": '{"a": 1}'})))
    out = LLMClient("http://x/v1", "", "auto", reasoning_budget=500).chat(
        messages=[{"role": "user", "content": "plan"}])
    assert out == {"a": 1}


def test_a_cut_call_still_counts_in_usage_calls_and_seconds(monkeypatch):
    from salome_agent.llm import ReasoningBudgetExceeded
    ticks = iter(range(0, 100_000, 10))
    monkeypatch.setattr("salome_agent.llm.time.monotonic", lambda: next(ticks))
    monkeypatch.setattr("salome_agent.llm.requests.post",
                        lambda *a, **k: _FakeResp(None, lines=_reasoning_lines(["x" * 400] * 10)))
    client = LLMClient("http://x/v1", "", "auto", reasoning_budget=500, request_timeout=10_000)
    with pytest.raises(ReasoningBudgetExceeded):
        client.chat(messages=[{"role": "user", "content": "x"}], tools=_MK_TOOL)
    assert client.usage["calls"] == 1
    assert client.usage["metered_calls"] == 0
    assert client.usage["llm_seconds"] > 0
    assert client.usage["incomplete_calls"] == 1
    # cut once past 500 approx tokens: 6 chunks of 400 chars streamed
    assert client.usage["approx_incomplete_output_tokens"] == 600


def test_reasoning_seconds_cuts_a_call_still_reasoning_with_no_answer(monkeypatch):
    from salome_agent.llm import ReasoningTimeExceeded
    ticks = iter(range(0, 100_000, 10))
    monkeypatch.setattr("salome_agent.llm.time.monotonic", lambda: next(ticks))
    calls = []

    def post(*a, **k):
        calls.append(1)
        return _FakeResp(None, lines=_reasoning_lines(["x" * 400] * 50))

    monkeypatch.setattr("salome_agent.llm.requests.post", post)
    client = LLMClient("http://x/v1", "", "auto", request_timeout=10_000)
    with pytest.raises(ReasoningTimeExceeded) as e:
        client.chat(messages=[{"role": "user", "content": "x"}], reasoning_share=0.01)
    assert e.value.seconds == 100
    assert e.value.reasoning and set(e.value.reasoning) == {"x"}
    assert len(calls) == 1  # not retried as an endpoint failure
    assert client.usage["incomplete_calls"] == 1


def test_reasoning_seconds_stops_once_the_answer_starts(monkeypatch):
    ticks = iter(range(0, 100_000, 10))
    monkeypatch.setattr("salome_agent.llm.time.monotonic", lambda: next(ticks))
    lines = (_reasoning_lines(["x"] * 2)[:-1]
             + ["data: " + json.dumps({"choices": [{"delta": {"content": '{"a":'}}]})]
             + _reasoning_lines(["x"] * 50, then={"content": " 1}"}))
    monkeypatch.setattr("salome_agent.llm.requests.post",
                        lambda *a, **k: _FakeResp(None, lines=lines))
    out = LLMClient("http://x/v1", "", "auto", request_timeout=10_000).chat(
        messages=[{"role": "user", "content": "x"}], reasoning_share=0.01)
    assert out == {"a": 1}


def test_reasoning_seconds_zero_is_off(monkeypatch):
    ticks = iter(range(0, 100_000, 10))
    monkeypatch.setattr("salome_agent.llm.time.monotonic", lambda: next(ticks))
    monkeypatch.setattr("salome_agent.llm.requests.post",
                        lambda *a, **k: _FakeResp(None, lines=_reasoning_lines(
                            ["x"] * 50, then={"content": '{"a": 1}'})))
    out = LLMClient("http://x/v1", "", "auto", request_timeout=10_000).chat(
        messages=[{"role": "user", "content": "x"}])
    assert out == {"a": 1}


def _reasoned_tool_body():
    return {"choices": [{"finish_reason": "tool_calls", "message": {
        "content": None, "reasoning_content": "corners first",
        "tool_calls": [{"id": "c1", "function": {"name": "mk", "arguments": "{}"}}]}}]}


def test_chat_hands_back_reasoning_only_when_carrying(monkeypatch):
    monkeypatch.setattr("salome_agent.llm.requests.post", _stub_post(_reasoned_tool_body()))
    msgs = [{"role": "user", "content": "x"}]
    on = LLMClient("http://x/v1", "", "auto", carry_reasoning=True).chat(msgs, tools=_MK_TOOL)
    off = LLMClient("http://x/v1", "", "auto").chat(msgs, tools=_MK_TOOL)
    assert on["reasoning"] == "corners first"
    assert off["reasoning"] is None


def test_assistant_tool_turn_carries_reasoning_when_given():
    turn = assistant_tool_turn([{"name": "mk", "args": {}, "id": "c1"}], None, "why")
    assert turn["reasoning_content"] == "why"


def _ollama_lines(chunks, then):
    """Ollama/OpenRouter stream the thinking under `reasoning`, not `reasoning_content`."""
    lines = ["data: " + json.dumps({"choices": [{"delta": {"content": "", "reasoning": c}}]})
             for c in chunks]
    lines.append("data: " + json.dumps({"choices": [{"delta": then, "finish_reason": "tool_calls"}]}))
    lines.append("data: [DONE]")
    return lines


def test_reasoning_field_is_read_like_reasoning_content(monkeypatch):
    tc = {"tool_calls": [{"index": 0, "id": "c1", "type": "function",
                          "function": {"name": "mk", "arguments": "{}"}}]}
    monkeypatch.setattr("salome_agent.llm.requests.post",
                        lambda *a, **k: _FakeResp(None, lines=_ollama_lines(["We need", " a box"], tc)))
    out = LLMClient("http://x/v1", "", "auto").chat(
        messages=[{"role": "user", "content": "x"}], tools=_MK_TOOL)
    assert out["tool_call"] == "mk"
    assert out["raw_message"]["reasoning_content"] == "We need a box"


def test_reasoning_budget_counts_the_reasoning_field(monkeypatch):
    from salome_agent.llm import ReasoningBudgetExceeded
    monkeypatch.setattr("salome_agent.llm.requests.post",
                        lambda *a, **k: _FakeResp(None, lines=_ollama_lines(["x" * 400] * 10, {})))
    with pytest.raises(ReasoningBudgetExceeded):
        LLMClient("http://x/v1", "", "auto", reasoning_budget=500).chat(
            messages=[{"role": "user", "content": "x"}], tools=_MK_TOOL)


# --- repetition loop guard (task 317) ----------------------------------------

def _fixture(name):
    from pathlib import Path
    return (Path(__file__).parent.parent / "fixtures" / "repetition" / f"{name}.txt").read_text()


def _chunks(text, n=40):
    return [text[i:i + n] for i in range(0, len(text), n)]


def test_a_reasoning_loop_is_cut_and_not_retried(monkeypatch, tmp_path):
    from salome_agent.llm import RepetitionLoopDetected
    text = _fixture("loop_nemotron_shaper")
    calls = []

    def post(*a, **k):
        calls.append(1)
        return _FakeResp(None, lines=_ollama_lines(_chunks(text), {}))

    monkeypatch.setattr("salome_agent.llm.requests.post", post)
    trace = tmp_path / "t.jsonl"
    with pytest.raises(RepetitionLoopDetected) as e:
        LLMClient("http://x/v1", "", "auto", trace_path=str(trace)).chat(
            messages=[{"role": "user", "content": "x"}], tools=_MK_TOOL)
    assert len(calls) == 1
    assert e.value.channel == "reasoning"
    assert text.startswith(e.value.before)
    assert 0 < len(e.value.before) < len(text)
    assert e.value.sample in text[len(e.value.before):]
    assert "incomplete" in trace.read_text()


def test_the_planner_call_is_guarded_too(monkeypatch):
    from salome_agent.llm import RepetitionLoopDetected
    lines = _reasoning_lines(_chunks(_fixture("loop_gpt_oss_planner")), then={"content": "{}"})
    monkeypatch.setattr("salome_agent.llm.requests.post", lambda *a, **k: _FakeResp(None, lines=lines))
    with pytest.raises(RepetitionLoopDetected):
        LLMClient("http://x/v1", "", "auto").chat(messages=[{"role": "user", "content": "plan"}])


def test_a_content_loop_is_cut(monkeypatch):
    from salome_agent.llm import RepetitionLoopDetected
    text = " ".join(f"w{i}" for i in range(700)) + ". " + "I will now draw the ring and the bars. " * 200
    lines = ["data: " + json.dumps({"choices": [{"delta": {"content": c}}]}) for c in _chunks(text)]
    monkeypatch.setattr("salome_agent.llm.requests.post",
                        lambda *a, **k: _FakeResp(None, lines=lines + ["data: [DONE]"]))
    with pytest.raises(RepetitionLoopDetected) as e:
        LLMClient("http://x/v1", "", "auto").chat(
            messages=[{"role": "user", "content": "x"}], tools=_MK_TOOL)
    assert e.value.channel == "content"


def test_long_healthy_reasoning_is_not_cut(monkeypatch):
    tc = {"tool_calls": [{"index": 0, "id": "c1", "type": "function",
                          "function": {"name": "mk", "arguments": "{}"}}]}
    lines = _ollama_lines(_chunks(_fixture("ok_deepseek_shaper")), tc)
    monkeypatch.setattr("salome_agent.llm.requests.post", lambda *a, **k: _FakeResp(None, lines=lines))
    out = LLMClient("http://x/v1", "", "auto").chat(
        messages=[{"role": "user", "content": "x"}], tools=_MK_TOOL)
    assert out["tool_call"] == "mk"


def test_loop_retries_defaults_to_three():
    assert LLMClient("http://x/v1", "", "auto").loop_retries == 3
    assert LLMClient("http://x/v1", "", "auto", loop_retries=5).loop_retries == 5
