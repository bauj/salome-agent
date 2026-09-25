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

# HTTP 429 (rate limit / quota) is retried with backoff regardless of how slow
# it was — unlike a 5xx it is never an upstream-timeout. A Retry-After header
# is honoured up to this cap so one hostile value can't stall the run.
_RETRY_AFTER_CAP = 60.0

# Exponential backoff between retries: 2s, 4s, 8s, ... capped. A flaky free-tier
# endpoint (Gemini 503-storms) usually clears within a minute, so a couple of
# 2s/4s sleeps give up too early — `llm.max_retries` + this curve ride it out.
_BACKOFF_BASE = 2.0
_BACKOFF_CAP = 30.0


def _backoff_seconds(attempt: int) -> float:
    return min(_BACKOFF_BASE * (2 ** (attempt - 1)), _BACKOFF_CAP)


def _retry_after_seconds(resp, default: float) -> float:
    """Seconds to wait from a `Retry-After` header (delta-seconds form), capped;
    `default` when the header is absent or not a plain number."""
    try:
        return min(float((resp.headers.get("Retry-After") or "").strip()), _RETRY_AFTER_CAP)
    except ValueError:
        return default

# Connect phase gets its own short cap — a backend that won't even accept the
# socket is down, not slow.
_CONNECT_TIMEOUT = 10.0

# Socket read-gap cap: the longest silence tolerated *between* stream frames
# before the stream is declared stalled. `llm.timeout` is the total budget; the
# read gap is min(that, this) so a reasoning-ON backend that only emits `: ping`
# keep-alives every few seconds stays alive while a genuinely frozen socket
# still dies within a minute rather than waiting out the whole budget.
_STREAM_READ_GAP_CAP = 60.0


class StreamBudgetExceeded(requests.Timeout):
    """The streamed round-trip ran past `llm.timeout` total wall-clock. Fatal:
    the backend is stuck (a runaway reasoning loop, typically) and retrying just
    burns another full budget on the same stuck backend."""


class LLMCancelled(Exception):
    """The turn's cancel event was set while a call was in flight (Stop in the
    chat). Not a failure and never retried; deliberately not a requests or
    RuntimeError subclass so no endpoint-failure handler swallows it."""


# A dropped/half-read stream raises one of these; all are retried with backoff.
# StreamBudgetExceeded subclasses requests.Timeout but is caught first and NOT
# retried.
_STREAM_DROP = (requests.Timeout, requests.ConnectionError,
                requests.exceptions.ChunkedEncodingError)

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


def _is_json_object(s: str) -> bool:
    """True if `s` on its own is a complete JSON object. Used to tell a
    provider that streams whole tool-call argument objects (Gemini's shim)
    from one that streams partial fragments (OpenAI)."""
    try:
        return isinstance(json.loads(s), dict)
    except ValueError:
        return False


def _loads_first_object(s: str):
    """`json.loads` but tolerant of a trailing repeat — Gemini's shim
    sometimes emits a tool call's arguments twice, yielding "{...}{...}"."""
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        obj, _ = json.JSONDecoder().raw_decode(s.lstrip())
        return obj


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
    with backoff.

    Responses are consumed as an SSE stream (`stream: true`). A long
    generation — a big reasoning budget especially — then keeps bytes moving
    over the socket the whole time, so an idle-timeout in a proxy/load
    balancer never fires. `request_timeout` (`llm.timeout`) is enforced two
    ways: as the total wall-clock budget for one streamed round-trip (a
    deadline checked while reassembling the stream — a runaway reasoning loop
    that never stops emitting is still cut), and, capped at
    _STREAM_READ_GAP_CAP, as the socket read-gap so a frozen stream dies
    fast. The reassembled `message` has the same shape the non-streaming
    endpoint returned, so nothing downstream changes."""

    def __init__(self, base_url: str, api_key: str, model: str, ssl_cert_file: str = "",
                 temperature: float = 0, request_timeout: float = 120,
                 trace_path: str | None = None, extra_headers: dict | None = None,
                 extra_body: dict | None = None, json_retries: int = 2,
                 module_extra_body: dict | None = None, max_retries: int = 2):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.extra_headers = extra_headers or {}
        # extra_body: verbatim top-level request-body fields, merged in after
        # model/messages/temperature. Provider-specific params that belong in
        # the JSON body, not an HTTP header (e.g. a reasoning_effort knob),
        # go here — extra_headers can't reach the body (GH #28).
        self.extra_body = extra_body or {}
        # module_extra_body: {"<module>": {...}} — a call tagged with that
        # module (chat(module=...)) sends this body *instead of* extra_body
        # (full replacement, not a merge). Lets an expensive knob like
        # reasoning_effort sit on the planner only while the specialist loops
        # (geom/smesh/paravis) — many calls per stage — run without it.
        self.module_extra_body = module_extra_body or {}
        # requests `verify`: a CA-bundle path, else True (default: verify against
        # the system trust store — never "" / False, which disables TLS checks).
        self.ssl_cert_file = ssl_cert_file or True
        self.temperature = temperature
        # Total wall-clock budget for one streamed round-trip.
        self.request_timeout = request_timeout
        # Socket read-gap: silence between frames that means "stalled". Never
        # larger than the total budget.
        self.stream_read_timeout = min(request_timeout, _STREAM_READ_GAP_CAP)
        # No-tools (planner) path only: on a response that will not parse as
        # JSON, feed the specific parser error back to the model and re-ask,
        # up to this many times, before failing the run (GH #6).
        self.json_retries = json_retries
        # transient HTTP failures (connection drop, truncated stream, 429,
        # 502/503/504) get this many retries after the first attempt, with
        # exponential backoff. Bump it for a flaky free-tier endpoint.
        self.max_retries = max_retries
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
        # `llm_seconds` sums the wall-clock of each successful round-trip
        # (final attempt only; retry backoff sleeps excluded) — graph.py diffs
        # it at stage boundaries to split LLM wait from local/SALOME time.
        self.usage = {"calls": 0, "metered_calls": 0,
                      "prompt_tokens": 0, "completion_tokens": 0,
                      "approx_prompt_tokens": 0, "llm_seconds": 0.0}
        # The running turn's threading.Event, set by runner.drive(): a call
        # in flight stops at the next streamed line or retry wait (GH #47).
        self.cancel = None
        # Also set by runner.drive(): called with {"type": "start" | "delta" |
        # "end", ...} as a call streams, so a front end can show the reasoning live.
        self.on_stream = None

    def _check_cancel(self) -> None:
        if self.cancel is not None and self.cancel.is_set():
            raise LLMCancelled("LLM call cancelled")

    def _stream_event(self, **ev) -> None:
        if self.on_stream is None:
            return
        try:
            self.on_stream(ev)
        except Exception as e:  # noqa: BLE001 — a display problem must not fail the call
            log.warning("LLM stream hook failed: %s", e)

    def _pause(self, seconds: float) -> None:
        """A retry wait that a cancel cuts short."""
        if self.cancel is None:
            time.sleep(seconds)
        elif self.cancel.wait(seconds):
            raise LLMCancelled("LLM call cancelled")

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

    def _consume_stream(self, resp, t0: float,
                        prose: bool = False) -> tuple[dict, dict | None, str | None, float | None]:
        """Reassemble an SSE `data:` stream into one provider `message` dict
        (plus usage / finish_reason / time-to-first-token). `content` and
        `reasoning_content` fragments are concatenated; `tool_calls` fragments
        are gathered by `index` (the first fragment for an index carries its
        id and function name). OpenAI streams the JSON argument string in
        partial pieces (invalid alone) which we append; Gemini's OpenAI-compat
        shim sends the whole argument object in one delta and may repeat it,
        so a fragment that parses standalone replaces rather than extends.

        A stream that ends without ever yielding a parseable frame is a broken
        endpoint (an HTML error page, say) -> RuntimeError. One that yields
        no content/tool-call AND no finish_reason / [DONE] was cut mid-flight
        or blocked -> raised as a connection drop so the caller retries. Real
        output with a missing terminator is kept (some shims omit [DONE])."""
        content, reasoning = [], []
        tool_frags: dict[int, dict] = {}
        msg_extra: dict = {}
        finish = usage = ttft = None
        frames = 0
        done = False
        deadline = t0 + self.request_timeout
        # chunk_size=None: iter_lines() with a fixed size uses read(amt)
        # "read exactly amt bytes" semantics and blocks until the buffer fills
        # or the stream ends — a slow token trickle then stalls. None yields
        # whatever is on the socket as it arrives.
        # SSE is UTF-8 by spec; with no charset in the Content-Type, requests
        # falls back to ISO-8859-1 for text/* and garbles every non-ASCII char.
        resp.encoding = "utf-8"
        for raw in resp.iter_lines(chunk_size=None, decode_unicode=True):
            self._check_cancel()
            if time.monotonic() > deadline:
                raise StreamBudgetExceeded(
                    f"LLM stream ran past the {self.request_timeout:g}s total "
                    f"budget (llm.timeout) — backend still emitting after "
                    f"{frames} frame(s); treating as a runaway generation")
            if not raw:
                continue
            if ttft is None:
                ttft = round(time.monotonic() - t0, 2)
            if raw.startswith("data:"):
                raw = raw[5:].strip()
            if raw == "[DONE]":
                done = True
                break
            try:
                frame = json.loads(raw)
            except ValueError:
                continue  # SSE comment / keep-alive line
            frames += 1
            if frame.get("error"):
                raise RuntimeError(f"LLM stream returned an error frame: {frame['error']}")
            if frame.get("usage"):
                usage = frame["usage"]
            for choice in frame.get("choices") or []:
                delta = choice.get("delta") or {}
                if delta.get("content"):
                    content.append(delta["content"])
                    if prose:
                        self._stream_event(type="delta", channel="prose", text=delta["content"])
                if delta.get("reasoning_content"):
                    reasoning.append(delta["reasoning_content"])
                    self._stream_event(type="delta", channel="reasoning",
                                       text=delta["reasoning_content"])
                for key in _CONTINUITY_KEYS:
                    if delta.get(key):
                        msg_extra[key] = delta[key]
                for tc in delta.get("tool_calls") or []:
                    slot = tool_frags.setdefault(
                        tc.get("index", 0), {"id": None, "name": None, "args": []})
                    if tc.get("id"):
                        slot["id"] = tc["id"]
                    fn = tc.get("function") or {}
                    if fn.get("name"):
                        slot["name"] = fn["name"]
                    if fn.get("arguments"):
                        frag = fn["arguments"]
                        if _is_json_object(frag):
                            slot["args"] = [frag]      # whole object (Gemini shim)
                        else:
                            slot["args"].append(frag)  # OpenAI partial fragment
                    for key in _CONTINUITY_KEYS:
                        if tc.get(key):
                            slot.setdefault("extra", {})[key] = tc[key]
                if choice.get("finish_reason"):
                    finish = choice["finish_reason"]
        if frames == 0:
            raise RuntimeError(
                "LLM endpoint returned a non-SSE / non-JSON body over a streaming "
                "request (is the model/proxy healthy?).")
        if not done and finish is None and not (content or tool_frags):
            # nothing came through and no terminator — a real cut, or a
            # provider that blocked the response with no body
            raise requests.ConnectionError(
                "LLM stream ended with no content, tool call or finish_reason "
                "(truncated mid-response, or blocked)")
        if not done and finish is None:
            # a clean EOF with real output but no [DONE] / finish_reason — some
            # OpenAI-compat shims just omit the terminator; keep what we got
            log.warning("LLM stream ended without an explicit finish_reason / "
                        "[DONE]; using the %d frame(s) received", frames)
        message: dict = {"role": "assistant", "content": "".join(content) or None}
        if reasoning:
            message["reasoning_content"] = "".join(reasoning)
        if tool_frags:
            message["tool_calls"] = [
                {"id": slot["id"], "type": "function",
                 "function": {"name": slot["name"],
                              "arguments": "".join(slot["args"]) or "{}"},
                 **slot.get("extra", {})}
                for _, slot in sorted(tool_frags.items())
            ]
        message.update(msg_extra)
        return message, usage, finish, ttft

    def _round_trip(self, payload: dict, tool_names: list[str] | None,
                    extra_body: dict | None = None, module: str | None = None) -> dict:
        """One streamed POST to /chat/completions (with transient-failure
        retries) — returns the reassembled provider `message` object."""
        self._stream_event(type="start", module=module)
        try:
            return self._round_trip_attempts(payload, tool_names, extra_body)
        finally:
            self._stream_event(type="end")

    def _round_trip_attempts(self, payload: dict, tool_names: list[str] | None,
                             extra_body: dict | None) -> dict:
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        headers.update(self.extra_headers)
        payload = {**payload, "stream": True,
                   "stream_options": {"include_usage": True}}
        log.debug("LLM request: model=%s, %d message(s), tools=%s, timeout=%ss",
                  self.model, len(payload["messages"]), tool_names or "-", self.request_timeout)
        approx_in = self._approx_tokens(payload)
        self.usage["approx_prompt_tokens"] += approx_in
        trace_request = {"model": self.model, "temperature": self.temperature,
                         "tools": tool_names or [], "approx_input_tokens": approx_in,
                         "messages": payload["messages"]}
        if extra_body is None:
            extra_body = self.extra_body
        if extra_body:
            trace_request["extra_body"] = extra_body
        self._trace_prompt(trace_request)
        t0 = time.monotonic()
        attempts = self.max_retries + 1
        for attempt in range(1, attempts + 1):
            resp = None
            self._check_cancel()
            try:
                resp = requests.post(f"{self.base_url}/chat/completions", json=payload,
                                     headers=headers, verify=self.ssl_cert_file,
                                     timeout=(_CONNECT_TIMEOUT, self.stream_read_timeout),
                                     stream=True)
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
                    resp.close()
                    self._pause(_backoff_seconds(attempt))
                    t0 = time.monotonic()
                    continue
                if resp.status_code == 429 and attempt < attempts:
                    wait = _retry_after_seconds(resp, _backoff_seconds(attempt))
                    log.warning("LLM request attempt %d rate-limited (HTTP 429), "
                                "waiting %.0fs before retry", attempt, wait)
                    resp.close()
                    self._pause(wait)
                    t0 = time.monotonic()
                    continue
                resp.raise_for_status()
                message, usage, finish, ttft = self._consume_stream(
                    resp, t0, prose=bool(tool_names))
                break
            except LLMCancelled:
                log.info("LLM request attempt %d cancelled", attempt)
                if resp is not None:
                    resp.close()
                raise
            except StreamBudgetExceeded as e:
                log.warning("LLM request attempt %d: %s", attempt, e)
                if resp is not None:
                    resp.close()
                raise  # runaway generation — a retry burns another full budget
            except _STREAM_DROP as e:
                log.debug("LLM request attempt %d failed after %.1fs: %s",
                          attempt, time.monotonic() - t0, e)
                if resp is not None:
                    resp.close()
                if attempt == attempts:
                    raise
                self._pause(_backoff_seconds(attempt))
                t0 = time.monotonic()
        self.usage["llm_seconds"] += time.monotonic() - t0
        self._record_usage({"usage": usage})
        self._trace(trace_request,
                    {"finish_reason": finish, "elapsed_s": round(time.monotonic() - t0, 2),
                     "ttft_s": ttft, "usage": usage, "message": message})
        log.debug("LLM response in %.1fs (ttft %ss): finish=%s",
                  time.monotonic() - t0, ttft, finish)
        return message

    def chat(self, messages: list[dict], tools: list[dict] | None = None,
             module: str | None = None) -> dict:
        # tools: [{"name": str, "parameters": <JSON Schema object>}, ...] — a
        # real per-tool schema, not just a bare name, so the LLM knows each
        # arg's name/type/required-ness instead of guessing.
        # module: routes to module_extra_body[module] when present, else the
        # shared extra_body.
        extra_body = self.module_extra_body.get(module, self.extra_body)
        payload = {"model": self.model, "messages": list(messages),
                   "temperature": self.temperature, **extra_body}
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
            message = self._round_trip(payload, tool_names, extra_body, module)
            calls = message.get("tool_calls") or []
            content = message.get("content") or ""
            log.debug("LLM tool_calls=%s%s",
                      [c["function"]["name"] for c in calls] or "none",
                      f'\n    said: {content[:500]}' if content else "")
            parsed = [{"name": c["function"]["name"],
                       "args": _loads_first_object(c["function"]["arguments"]),
                       "id": c.get("id")}
                      for c in calls]
            if not parsed:
                return {"tool_call": None, "args": None, "tool_call_id": None,
                        "tool_calls": [], "content": content, "raw_message": message}
            # tool_calls: every call the model emitted this turn (tool_choice=
            # "auto" allows several). The singular keys mirror the first call,
            # kept for backward compatibility with callers that expect only one.
            first = parsed[0]
            return {"tool_call": first["name"], "args": first["args"],
                    "tool_call_id": first["id"], "tool_calls": parsed,
                    "content": content, "raw_message": message}

        # No-tools (planner) path: the response must be a JSON object. If it
        # will not parse, hand the model its own answer plus the exact parser
        # error and re-ask, up to json_retries times (GH #6).
        for attempt in range(self.json_retries + 1):
            message = self._round_trip(payload, None, extra_body, module)
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
