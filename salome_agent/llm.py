import json
import os
import re
import time

import requests

from salome_agent.logs import log

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)

# Retry a 502/503/504 only when it comes back fast (a real transient blip). A
# slow one is the proxy's upstream-timeout firing — retrying wastes another
# full timeout for the same slow backend.
_GATEWAY_RETRY_MAX_ELAPSED = 30.0

# Opaque provider-specific fields on an assistant message that MUST be echoed
# back verbatim in later requests for multi-turn continuity — e.g. Gemini 3
# carries a `thought_signature` inside `extra_content` and rejects the next
# request (HTTP 400) if it is missing. Deliberately excludes `reasoning_content`:
# it is bulky, not required for continuity, and gets re-billed as input if fed
# back.
_CONTINUITY_KEYS = ("extra_content",)


def continuity_fields(obj: dict | None) -> dict:
    """The subset of an assistant `message` (or of a single tool_call) that a
    caller must include when it appends that turn to its own history — see
    _CONTINUITY_KEYS. A provider may hang the field on the message, on each
    tool_call, or both (Gemini 3 puts `thought_signature` inside a tool_call's
    `extra_content`)."""
    return {k: obj[k] for k in _CONTINUITY_KEYS if obj and k in obj}


def assistant_tool_turn(parsed_calls: list[dict], raw_message: dict | None) -> dict:
    """Rebuild the assistant tool-call turn for history, carrying every
    provider continuity field (per-call and message-level) but not the bulky
    reasoning text. `parsed_calls` items are {name, args, id}; `raw_message` is
    the verbatim provider message, used only to recover continuity fields."""
    raw_calls = (raw_message or {}).get("tool_calls") or []
    raw_by_id = {c.get("id"): c for c in raw_calls if c.get("id") is not None}
    # Match a parsed call to its raw counterpart by id, falling back to position
    # (a provider that omits tool_call ids also gives us backfilled synthetic
    # ones that will not be in raw_by_id).
    def _raw_for(i: int, cid):
        if cid in raw_by_id:
            return raw_by_id[cid]
        return raw_calls[i] if i < len(raw_calls) else None
    return {
        "role": "assistant", "content": None,
        "tool_calls": [
            {"id": c["id"], "type": "function",
             "function": {"name": c["name"], "arguments": json.dumps(c["args"])},
             **continuity_fields(_raw_for(i, c["id"]))}
            for i, c in enumerate(parsed_calls)
        ],
        **continuity_fields(raw_message),
    }


def _parse_json_content(content: str):
    """Real LLMs often wrap the JSON in ```json fences or add a sentence of
    prose around it — try the raw string first, then a fenced block, then the
    first balanced {...} span."""
    content = (content or "").strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        first_err = e
    m = _FENCE_RE.search(content)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    start = content.find("{")
    end = content.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(content[start:end + 1])
        except json.JSONDecodeError:
            pass
    raise ValueError(
        f"LLM did not return parseable JSON ({first_err}). Raw content:\n{content}")


class LLMClient:
    """Minimal wrapper over any OpenAI-compatible tool-calling chat endpoint
    (spec §9/decision 10 — provider-agnostic, configured by endpoint+key+model).
    Retries transient failures (timeout, connection drop, HTTP 502/503/504)
    with backoff; no streaming."""

    def __init__(self, base_url: str, api_key: str, model: str, ssl_cert_file: str = "",
                 temperature: float = 0, request_timeout: float = 120,
                 trace_path: str | None = None, extra_headers: dict | None = None,
                 extra_body: dict | None = None, json_retries: int = 2):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.extra_headers = extra_headers or {}
        # extra_body: verbatim top-level request-body fields, merged in after
        # model/messages/temperature. Provider-specific params that belong in
        # the JSON body, not an HTTP header (e.g. a reasoning_effort knob),
        # go here — extra_headers can't reach the body (GH #28).
        self.extra_body = extra_body or {}
        # requests `verify`: a CA-bundle path, else True (default: verify against
        # the system trust store — never "" / False, which disables TLS checks).
        self.ssl_cert_file = ssl_cert_file or True
        self.temperature = temperature
        self.request_timeout = request_timeout
        # No-tools (planner) path only: on a response that will not parse as
        # JSON, feed the specific parser error back to the model and re-ask,
        # up to this many times, before failing the run (GH #6).
        self.json_retries = json_retries
        # -vv: append one JSON line per round-trip (request + raw response +
        # whatever reasoning key the provider returns) so tool-selection can be
        # inspected after the run.
        self.trace_path = trace_path
        # Running total of provider-reported token usage across every chat()
        # call on this client. `calls` counts all round-trips; `metered_calls`
        # counts the subset that came back with a `usage` block (some
        # providers/proxies omit it). Surfaced in report.md.
        # `approx_prompt_tokens` is a chars/4 estimate of the serialized
        # request (messages + FULL tool schemas) summed over every call —
        # the real input cost even when a provider strips its `usage` block,
        # and the only figure that reflects the re-sent tool schemas.
        self.usage = {"calls": 0, "metered_calls": 0,
                      "prompt_tokens": 0, "completion_tokens": 0,
                      "approx_prompt_tokens": 0}

    @staticmethod
    def _approx_tokens(payload: dict) -> int:
        return len(json.dumps(payload, default=str)) // 4

    def _record_usage(self, body: dict) -> None:
        self.usage["calls"] += 1
        u = body.get("usage") or {}
        if u.get("prompt_tokens") is None and u.get("completion_tokens") is None:
            return
        self.usage["metered_calls"] += 1
        self.usage["prompt_tokens"] += int(u.get("prompt_tokens") or 0)
        self.usage["completion_tokens"] += int(u.get("completion_tokens") or 0)

    def _trace_prompt(self, request: dict) -> None:
        # Printed BEFORE the POST so the prompt is visible while the call is
        # still in flight. The jsonl file gets the combined entry from _trace()
        # once the response is back.
        if not self.trace_path:
            return
        log.debug("LLM request:\n%s", json.dumps(request, indent=2, default=str, ensure_ascii=False))

    def _trace(self, request: dict, response: dict) -> None:
        if not self.trace_path:
            return
        entry = {"t": time.strftime("%H:%M:%S"), "request": request, "response": response}
        try:
            os.makedirs(os.path.dirname(self.trace_path) or ".", exist_ok=True)
            with open(self.trace_path, "a") as f:
                f.write(json.dumps(entry, default=str, ensure_ascii=False) + "\n")
        except OSError as e:
            log.warning("could not write LLM trace: %s", e)
        log.debug("LLM response:\n%s", json.dumps(response, indent=2, default=str, ensure_ascii=False))

    def _round_trip(self, payload: dict, tool_names: list[str] | None) -> dict:
        """One POST to /chat/completions (with transient-failure retries) —
        returns the raw provider `message` object."""
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        headers.update(self.extra_headers)
        log.debug("LLM request: model=%s, %d message(s), tools=%s, timeout=%ss",
                  self.model, len(payload["messages"]), tool_names or "-", self.request_timeout)
        approx_in = self._approx_tokens(payload)
        self.usage["approx_prompt_tokens"] += approx_in
        trace_request = {"model": self.model, "temperature": self.temperature,
                         "tools": tool_names or [], "approx_input_tokens": approx_in,
                         "messages": payload["messages"]}
        self._trace_prompt(trace_request)
        t0 = time.monotonic()
        resp = None
        attempts = 3
        for attempt in range(1, attempts + 1):
            try:
                resp = requests.post(f"{self.base_url}/chat/completions", json=payload,
                                     headers=headers, verify=self.ssl_cert_file,
                                     timeout=self.request_timeout)
                # A fast gateway error (proxy up, model backend down/overloaded)
                # crashes the planner/specialist node otherwise — retry with backoff.
                # A *slow* 502/503/504 means the proxy's own upstream timeout fired
                # (e.g. Manifest aborts at 180s): the backend is just slow, not
                # blipping — retrying only burns another full timeout, so give up.
                elapsed = time.monotonic() - t0
                if (resp.status_code in (502, 503, 504)
                        and attempt < attempts and elapsed < _GATEWAY_RETRY_MAX_ELAPSED):
                    log.debug("LLM request attempt %d got HTTP %d after %.1fs, retrying",
                              attempt, resp.status_code, elapsed)
                    time.sleep(2 * attempt)
                    t0 = time.monotonic()
                    continue
                break
            except (requests.Timeout, requests.ConnectionError) as e:
                log.debug("LLM request attempt %d failed after %.1fs: %s",
                          attempt, time.monotonic() - t0, e)
                if attempt == attempts:
                    raise
                time.sleep(2 * attempt)
                t0 = time.monotonic()
        resp.raise_for_status()
        try:
            body = resp.json()
        except ValueError:
            snippet = (resp.text or "").strip()[:500]
            raise RuntimeError(
                f"LLM endpoint returned HTTP {resp.status_code} with a non-JSON body "
                f"(is the model/proxy healthy?). Body:\n{snippet or '<empty>'}")
        self._record_usage(body)
        message = body["choices"][0]["message"]
        finish = body["choices"][0].get("finish_reason")
        self._trace(trace_request,
                    {"finish_reason": finish, "elapsed_s": round(time.monotonic() - t0, 2),
                     "usage": body.get("usage"), "message": message})
        log.debug("LLM response in %.1fs: finish=%s", time.monotonic() - t0, finish)
        return message

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        # tools: [{"name": str, "parameters": <JSON Schema object>}, ...] — a
        # real per-tool schema, not just a bare name, so the LLM knows each
        # arg's name/type/required-ness instead of guessing.
        payload = {"model": self.model, "messages": list(messages),
                   "temperature": self.temperature, **self.extra_body}
        if tools:
            payload["tools"] = [
                {"type": "function", "function": {
                    "name": t["name"],
                    **({"description": t["description"]} if t.get("description") else {}),
                    "parameters": t.get("parameters") or {"type": "object", "properties": {}},
                }}
                for t in tools
            ]
            # "auto", not "required": the model may narrate its reasoning in
            # `content` before the call, and some models only do so when not
            # forced. A turn with no tool call at all is handled by the caller.
            payload["tool_choice"] = "auto"
        tool_names = [t["name"] for t in tools] if tools else None

        if tools:
            message = self._round_trip(payload, tool_names)
            calls = message.get("tool_calls") or []
            content = message.get("content") or ""
            log.debug("LLM tool_calls=%s%s",
                      [c["function"]["name"] for c in calls] or "none",
                      f'\n    said: {content[:500]}' if content else "")
            parsed = [{"name": c["function"]["name"],
                       "args": json.loads(c["function"]["arguments"]),
                       "id": c.get("id")}
                      for c in calls]
            if not parsed:
                return {"tool_call": None, "args": None, "tool_call_id": None,
                        "tool_calls": [], "content": content, "raw_message": message}
            # tool_calls: every call the model emitted this turn (tool_choice=
            # "auto" allows several). The singular keys mirror the first call —
            # kept for callers that only ever expect one (specialists/base.py).
            first = parsed[0]
            return {"tool_call": first["name"], "args": first["args"],
                    "tool_call_id": first["id"], "tool_calls": parsed,
                    "content": content, "raw_message": message}

        # No-tools (planner) path: the response must be a JSON object. If it
        # will not parse, hand the model its own answer plus the exact parser
        # error and re-ask, up to json_retries times (GH #6).
        for attempt in range(self.json_retries + 1):
            message = self._round_trip(payload, None)
            content = message.get("content")
            try:
                return _parse_json_content(content)
            except ValueError as e:
                if attempt == self.json_retries:
                    raise
                reason = str(e).split(". Raw content:")[0]
                log.warning("LLM returned unparseable JSON (attempt %d/%d): %s",
                            attempt + 1, self.json_retries + 1, reason)
                payload["messages"] = payload["messages"] + [
                    {"role": "assistant", "content": content or "",
                     **continuity_fields(message)},
                    {"role": "user", "content":
                        f"Your previous response was not valid JSON: {reason}. "
                        "Common causes: a trailing comma, an unquoted or unescaped "
                        "value, prose around the object, or a code fence. Reply with "
                        "ONLY the JSON object -- nothing before or after it."},
                ]
