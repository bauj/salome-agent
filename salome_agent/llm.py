import json
import os
import re
import threading
import time

import requests

from salome_agent.logs import log
from salome_agent.repetition import RepetitionWatch

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
_CANCEL_POLL = 0.1
_EOF = object()


def redact_images(messages: list[dict]) -> list[dict]:
    """`messages` with each image_url data URI swapped for a short placeholder,
    for the token estimate and the trace (a base64 image is ~100k chars)."""
    out = []
    for m in messages:
        content = m.get("content")
        if isinstance(content, list):
            content = [{"type": "image_url", "image_url": {
                           "url": f"<image, {len(p['image_url'].get('url', ''))} chars>"}}
                       if p.get("type") == "image_url" else p
                       for p in content]
            m = {**m, "content": content}
        out.append(m)
    return out


def with_images(text: str, image_data_uris: list[str] | None):
    """Build a chat message `content` value: plain text, or an OpenAI-compatible
    multimodal parts list (text + one image_url part per reference image)."""
    if not image_data_uris:
        return text
    return [
        {"type": "text", "text": text},
        *({"type": "image_url", "image_url": {"url": uri}} for uri in image_data_uris),
    ]

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


class ReasoningBudgetExceeded(Exception):
    """A tool-calling round-trip reasoned past `llm.reasoning_budget` (approx
    tokens) before starting any tool call or content: the model is deliberating
    instead of acting. Never retried here; the specialist re-asks with a nudge."""

    def __init__(self, tokens: int, budget: int, reasoning: str = ""):
        super().__init__(f"model reasoned ~{tokens} tokens without acting "
                         f"(llm.reasoning_budget {budget})")
        self.tokens = tokens
        self.budget = budget
        self.reasoning = reasoning


class ReasoningTimeExceeded(Exception):
    """A call given `reasoning_seconds` was still reasoning, with no content
    yet, after that many seconds: a slow model going round in circles. Never
    retried here; the caller re-asks with a "decide now" nudge."""

    def __init__(self, seconds: float, tokens: int, reasoning: str = ""):
        super().__init__(f"model reasoned {seconds:g}s (~{tokens} tokens) without answering")
        self.seconds = seconds
        self.tokens = tokens
        self.reasoning = reasoning


class RepetitionLoopDetected(Exception):
    """The streamed reasoning or content started repeating itself (task 317):
    small models loop until the timeout. Never retried here; the caller
    re-asks with `before` (that channel's text up to the loop) and `sample`
    (one looped sentence), never the looped text itself."""

    def __init__(self, channel: str, words: int, before: str, sample: str):
        super().__init__(f"model {channel} looped (repeating itself after ~{words} words)")
        self.channel = channel
        self.words = words
        self.before = before
        self.sample = sample


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
# request (HTTP 400) if it is missing. Excludes `reasoning_content`: it is bulky
# and re-billed as input, so carrying it is opt-in (llm.carry_reasoning, passed
# to assistant_tool_turn as `reasoning`).
_CONTINUITY_KEYS = ("extra_content",)


def continuity_fields(obj: dict | None) -> dict:
    """The subset of an assistant `message` (or of a single tool_call) that a
    caller must include when it appends that turn to its own history — see
    _CONTINUITY_KEYS. A provider may hang the field on the message, on each
    tool_call, or both (Gemini 3 puts `thought_signature` inside a tool_call's
    `extra_content`)."""
    return {k: obj[k] for k in _CONTINUITY_KEYS if obj and k in obj}


def assistant_tool_turn(parsed_calls: list[dict], raw_message: dict | None,
                        reasoning: str | None = None) -> dict:
    """Rebuild the assistant tool-call turn for history, carrying every
    provider continuity field (per-call and message-level), and the turn's
    reasoning when given (llm.carry_reasoning). `parsed_calls` items are
    {name, args, id}; `raw_message` is the verbatim provider message, used only
    to recover continuity fields."""
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
        **({"reasoning_content": reasoning} if reasoning else {}),
    }


def _is_json_object(s: str) -> bool:
    """True if `s` on its own is a complete JSON object. Used to tell a
    provider that streams whole tool-call argument objects (Gemini's shim)
    from one that streams partial fragments (OpenAI)."""
    try:
        return isinstance(json.loads(s), dict)
    except ValueError:
        return False


def _assemble_message(partial: dict) -> dict:
    """The provider `message` from what _consume_stream gathered (complete,
    or as far as a stream that stopped early got)."""
    message: dict = {"role": "assistant", "content": "".join(partial.get("content", [])) or None}
    if partial.get("reasoning"):
        message["reasoning_content"] = "".join(partial["reasoning"])
    if partial.get("tool_frags"):
        message["tool_calls"] = [
            {"id": slot["id"], "type": "function",
             "function": {"name": slot["name"], "arguments": "".join(slot["args"]) or "{}"},
             **slot.get("extra", {})}
            for _, slot in sorted(partial["tool_frags"].items())
        ]
    message.update(partial.get("msg_extra", {}))
    return message


def _streamed_chars(partial: dict) -> int:
    return (sum(map(len, partial.get("content", [])))
            + sum(map(len, partial.get("reasoning", [])))
            + sum(len(f) for slot in partial.get("tool_frags", {}).values()
                  for f in slot["args"]))


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


def _cert_verify(ssl_cert_file: str):
    """requests `verify`: a CA-bundle path, else True (default: verify against
    the system trust store — never "" / False, which disables TLS checks).
    Checked here so a bad path fails at startup, not on the first request."""
    if not ssl_cert_file:
        return True
    cert_path = os.path.realpath(os.path.expanduser(ssl_cert_file))
    if not os.path.isfile(cert_path):
        raise FileNotFoundError(f"SSL certificate file not found: {cert_path}")
    return cert_path


class _Endpoint:
    """Where and how one call is sent: URL, key, model and the per-request knobs."""

    def __init__(self, base_url, api_key, model, verify, temperature,
                 request_timeout, extra_headers, extra_body):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.verify = verify
        self.temperature = temperature
        self.request_timeout = request_timeout
        self.stream_read_timeout = min(request_timeout, _STREAM_READ_GAP_CAP)
        self.extra_headers = extra_headers
        self.extra_body = extra_body


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
                 module_extra_body: dict | None = None, max_retries: int = 2,
                 modules: dict | None = None, reasoning_budget: int = 0,
                 carry_reasoning: bool = False):
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
        # (geom/smesh) — many calls per stage — run without it.
        self.module_extra_body = module_extra_body or {}
        self.ssl_cert_file = _cert_verify(ssl_cert_file)
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
        # Tool-calling calls only: approx reasoning tokens (chars/4) a model may
        # spend before its first tool call or content; 0 = no limit.
        self.reasoning_budget = reasoning_budget
        # Tool-calling calls only: hand back each turn's reasoning (chat()'s
        # "reasoning" key) so the loop can keep it in history and the model
        # resumes its own thinking after a tool result instead of starting over.
        self.carry_reasoning = carry_reasoning
        # modules: {"<module>": {<config.MODULE_LLM_KEYS subset>}} — that module's calls
        # go to its own endpoint/model (e.g. a better drawing reader for vision);
        # keys left out keep the shared value. Its extra_body, when given, wins
        # over module_extra_body.
        self._endpoints = {name: self._build_endpoint(name, override or {})
                           for name, override in (modules or {}).items()}
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
        # `incomplete_calls` / `approx_incomplete_output_tokens`: calls cut or
        # dropped mid-stream never get a usage frame, so their streamed output
        # (reasoning + content + tool args, chars/4) is estimated here.
        # `llm_seconds` sums the wall-clock of each streamed attempt, including
        # one cut or dropped mid-stream (retry backoff sleeps excluded) — graph.py diffs
        # it at stage boundaries to split LLM wait from local/SALOME time.
        self.usage = {"calls": 0, "metered_calls": 0,
                      "prompt_tokens": 0, "completion_tokens": 0,
                      "approx_prompt_tokens": 0, "llm_seconds": 0.0,
                      "incomplete_calls": 0, "approx_incomplete_output_tokens": 0}
        # The running turn's threading.Event, set by runner.drive(): a call
        # in flight stops within _CANCEL_POLL, even mid-wait (GH #47).
        self.cancel = None
        # Also set by runner.drive(): called with {"type": "start" | "delta" |
        # "end", ...} as a call streams, so a front end can show the reasoning live.
        self.on_stream = None
        self._live = None           # -vv llm_live.txt handle while a call streams
        self._live_channel = None

    def _check_cancel(self) -> None:
        if self.cancel is not None and self.cancel.is_set():
            raise LLMCancelled("LLM call cancelled")

    def _wait_on(self, fn):
        """Call blocking `fn` (a POST waiting for headers, a read waiting for
        the next line) on a daemon thread so a cancel stops the wait at once;
        the abandoned call ends at its next wake (data or read timeout)."""
        if self.cancel is None:
            return fn()
        box: list = []
        done, abandoned = threading.Event(), threading.Event()

        def run():
            try:
                box.append((True, fn()))
            except BaseException as e:  # noqa: BLE001 — re-raised in the caller
                box.append((False, e))
            done.set()
            if abandoned.is_set() and box[0][0] and hasattr(box[0][1], "close"):
                box[0][1].close()

        threading.Thread(target=run, daemon=True, name="llm-io").start()
        while not done.wait(_CANCEL_POLL):
            if self.cancel.is_set():
                abandoned.set()
                raise LLMCancelled("LLM call cancelled")
        ok, value = box[0]
        if ok:
            return value
        raise value

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

    def _trace_incomplete(self, request: dict, attempt: int, t0: float, partial: dict,
                          error: BaseException) -> None:
        """A call that did not finish (budget, cancel, dropped stream, Ctrl-C)
        still gets a trace line, with what it streamed before it stopped."""
        reason = f"{type(error).__name__}: {error}"
        self._live_write(f"\n=== incomplete after {time.monotonic() - t0:.1f}s: {reason} ===\n")
        self._trace(request, {"incomplete": reason, "attempt": attempt,
                              "elapsed_s": round(time.monotonic() - t0, 2),
                              "ttft_s": partial.get("ttft"), "frames": partial.get("frames", 0),
                              "message": _assemble_message(partial)})

    def _live_open(self, module: str | None, attempt: int) -> None:
        """-vv: llm_live.txt next to the trace gets each call's reasoning and
        content as they stream, flushed per delta, so `tail -f` shows a call
        in flight and a killed run keeps what it had."""
        self._live, self._live_channel = None, None
        if not self.trace_path:
            return
        path = os.path.join(os.path.dirname(self.trace_path) or ".", "llm_live.txt")
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self._live = open(path, "a", encoding="utf-8")
        except OSError as e:
            log.warning("could not open %s: %s", path, e)
            return
        self._live_write(f"\n=== {time.strftime('%H:%M:%S')} {module or 'call'} "
                         f"attempt {attempt} ===\n")

    def _live_write(self, text: str, channel: str | None = None) -> None:
        if self._live is None:
            return
        if channel and channel != self._live_channel:
            self._live_channel = channel
            text = f"\n--- {channel} ---\n{text}"
        try:
            self._live.write(text)
            self._live.flush()
        except (OSError, ValueError) as e:
            log.warning("could not write llm_live.txt: %s", e)
            self._live = None

    def _live_close(self) -> None:
        if self._live is not None:
            self._live.close()
            self._live = None

    def _consume_stream(self, resp, t0: float, prose: bool = False,
                        partial: dict | None = None, budget: float | None = None,
                        reasoning_budget: int = 0,
                        reasoning_seconds: float = 0) -> tuple[dict, dict | None, str | None, float | None]:
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
        output with a missing terminator is kept (some shims omit [DONE]).

        `partial` is filled as the stream arrives, so a caller can trace what
        came through if the stream stops early."""
        partial = {} if partial is None else partial
        content, reasoning = [], []
        tool_frags: dict[int, dict] = {}
        msg_extra: dict = {}
        partial.update(content=content, reasoning=reasoning, tool_frags=tool_frags,
                       msg_extra=msg_extra, frames=0, ttft=None)
        finish = usage = ttft = None
        frames = 0
        done = False
        reasoning_chars = 0
        watches = {"reasoning": RepetitionWatch(), "content": RepetitionWatch()}

        def watch(channel: str, text: str, gathered: list):
            loop = watches[channel].feed(text)
            if loop:
                raise RepetitionLoopDetected(channel, loop.words,
                                             "".join(gathered)[:loop.start], loop.sample)
        budget = self.request_timeout if budget is None else budget
        deadline = t0 + budget
        # chunk_size=None: iter_lines() with a fixed size uses read(amt)
        # "read exactly amt bytes" semantics and blocks until the buffer fills
        # or the stream ends — a slow token trickle then stalls. None yields
        # whatever is on the socket as it arrives.
        # SSE is UTF-8 by spec; with no charset in the Content-Type, requests
        # falls back to ISO-8859-1 for text/* and garbles every non-ASCII char.
        resp.encoding = "utf-8"
        lines = resp.iter_lines(chunk_size=None, decode_unicode=True)
        while (raw := self._wait_on(lambda: next(lines, _EOF))) is not _EOF:
            self._check_cancel()
            if time.monotonic() > deadline:
                raise StreamBudgetExceeded(
                    f"LLM stream ran past the {budget:g}s total "
                    f"budget (llm.timeout) — backend still emitting after "
                    f"{frames} frame(s); treating as a runaway generation")
            if not raw:
                continue
            if ttft is None:
                ttft = partial["ttft"] = round(time.monotonic() - t0, 2)
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
            partial["frames"] = frames
            if frame.get("error"):
                raise RuntimeError(f"LLM stream returned an error frame: {frame['error']}")
            if frame.get("usage"):
                usage = frame["usage"]
            for choice in frame.get("choices") or []:
                delta = choice.get("delta") or {}
                if delta.get("content"):
                    content.append(delta["content"])
                    watch("content", delta["content"], content)
                    self._live_write(delta["content"], "content")
                    if prose:
                        self._stream_event(type="delta", channel="prose", text=delta["content"])
                # `reasoning_content` (DeepSeek style) or `reasoning` (Ollama, OpenRouter)
                thought = delta.get("reasoning_content") or delta.get("reasoning")
                if thought:
                    reasoning.append(thought)
                    reasoning_chars += len(thought)
                    watch("reasoning", thought, reasoning)
                    if (reasoning_budget and not content and not tool_frags
                            and reasoning_chars // 4 > reasoning_budget):
                        raise ReasoningBudgetExceeded(reasoning_chars // 4, reasoning_budget,
                                                      "".join(reasoning))
                    if (reasoning_seconds and not content and not tool_frags
                            and time.monotonic() - t0 > reasoning_seconds):
                        raise ReasoningTimeExceeded(reasoning_seconds, reasoning_chars // 4,
                                                    "".join(reasoning))
                    self._live_write(thought, "reasoning")
                    self._stream_event(type="delta", channel="reasoning", text=thought)
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
        return _assemble_message(partial), usage, finish, ttft

    def _round_trip(self, payload: dict, tool_names: list[str] | None,
                    ep: _Endpoint | None = None, module: str | None = None,
                    reasoning_seconds: float = 0) -> dict:
        """One streamed POST to /chat/completions (with transient-failure
        retries) — returns the reassembled provider `message` object."""
        self._stream_event(type="start", module=module)
        try:
            return self._round_trip_attempts(payload, tool_names,
                                             ep or self.endpoint(module), module,
                                             reasoning_seconds)
        finally:
            self._stream_event(type="end")

    def _round_trip_attempts(self, payload: dict, tool_names: list[str] | None,
                             ep: _Endpoint, module: str | None = None,
                             reasoning_seconds: float = 0) -> dict:
        headers = {"Authorization": f"Bearer {ep.api_key}"} if ep.api_key else {}
        headers.update(ep.extra_headers)
        payload = {**payload, "stream": True,
                   "stream_options": {"include_usage": True}}
        log.debug("LLM request: model=%s, %d message(s), tools=%s, timeout=%ss",
                  ep.model, len(payload["messages"]), tool_names or "-", ep.request_timeout)
        messages = redact_images(payload["messages"])
        approx_in = self._approx_tokens({**payload, "messages": messages})
        self.usage["approx_prompt_tokens"] += approx_in
        trace_request = {"model": ep.model, "temperature": ep.temperature,
                         "tools": tool_names or [], "approx_input_tokens": approx_in,
                         "messages": messages}
        if ep.base_url != self.base_url:
            trace_request["base_url"] = ep.base_url
        if ep.extra_body:
            trace_request["extra_body"] = ep.extra_body
        self._trace_prompt(trace_request)
        t0 = time.monotonic()
        attempts = self.max_retries + 1
        for attempt in range(1, attempts + 1):
            resp = None
            self._check_cancel()
            try:
                resp = self._wait_on(lambda: requests.post(
                    f"{ep.base_url}/chat/completions", json=payload,
                    headers=headers, verify=ep.verify,
                    timeout=(_CONNECT_TIMEOUT, ep.stream_read_timeout),
                    stream=True))
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
                partial: dict = {}
                self._live_open(module, attempt)
                try:
                    message, usage, finish, ttft = self._consume_stream(
                        resp, t0, prose=bool(tool_names), partial=partial,
                        budget=ep.request_timeout,
                        reasoning_budget=self.reasoning_budget if tool_names else 0,
                        reasoning_seconds=reasoning_seconds)
                except BaseException as e:
                    # a cut or dropped call still cost a round-trip and its wait
                    self.usage["calls"] += 1
                    self.usage["llm_seconds"] += time.monotonic() - t0
                    self.usage["incomplete_calls"] += 1
                    self.usage["approx_incomplete_output_tokens"] += _streamed_chars(partial) // 4
                    self._trace_incomplete(trace_request, attempt, t0, partial, e)
                    self._live_close()
                    raise
                break
            except LLMCancelled:
                log.info("LLM request attempt %d cancelled", attempt)
                if resp is not None:
                    # close() waits for a read parked on the socket; let it
                    # finish when that read wakes instead of holding up Stop
                    threading.Thread(target=resp.close, daemon=True).start()
                raise
            except (ReasoningBudgetExceeded, ReasoningTimeExceeded, RepetitionLoopDetected) as e:
                log.warning("LLM request attempt %d: %s", attempt, e)
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
        self._live_write(f"\n=== finish={finish}, {time.monotonic() - t0:.1f}s ===\n")
        self._live_close()
        self.usage["llm_seconds"] += time.monotonic() - t0
        self._record_usage({"usage": usage})
        self._trace(trace_request,
                    {"finish_reason": finish, "elapsed_s": round(time.monotonic() - t0, 2),
                     "ttft_s": ttft, "usage": usage, "message": message})
        log.debug("LLM response in %.1fs (ttft %ss): finish=%s",
                  time.monotonic() - t0, ttft, finish)
        if finish == "length":
            u = usage or {}
            log.warning("LLM output was cut off (finish_reason=length; prompt %s + "
                        "completion %s tokens): the context window or max-tokens limit "
                        "was hit. Hidden reasoning counts against it -- raise the "
                        "server's context (Ollama defaults to 4096) or lower the "
                        "reasoning effort (llm.module_extra_body).",
                        u.get("prompt_tokens", "?"), u.get("completion_tokens", "?"))
        return message

    def _build_endpoint(self, module: str | None, override: dict) -> _Endpoint:
        def pick(key, shared):
            return override[key] if key in override else shared
        # Another host is another provider: never hand it the shared key, CA,
        # headers or body unless the override repeats them.
        own_host = "base_url" in override and override["base_url"].rstrip("/") != self.base_url
        extra_body = self.module_extra_body.get(module, {} if own_host else self.extra_body)
        verify = (_cert_verify(override["ssl_cert_file"]) if "ssl_cert_file" in override
                  else True if own_host else self.ssl_cert_file)
        return _Endpoint(pick("base_url", self.base_url),
                         pick("api_key", "" if own_host else self.api_key),
                         pick("model", self.model), verify,
                         pick("temperature", self.temperature),
                         pick("timeout", self.request_timeout),
                         pick("extra_headers", {} if own_host else self.extra_headers),
                         pick("extra_body", extra_body))

    def endpoint(self, module: str | None = None) -> _Endpoint:
        if module in self._endpoints:
            return self._endpoints[module]
        return self._build_endpoint(module, {})

    def chat(self, messages: list[dict], tools: list[dict] | None = None,
             module: str | None = None, reasoning_share: float = 0) -> dict:
        # tools: [{"name": str, "parameters": <JSON Schema object>}, ...] — a
        # real per-tool schema, not just a bare name, so the LLM knows each
        # arg's name/type/required-ness instead of guessing.
        # module: routes to llm.modules[module] (endpoint/model override) and
        # module_extra_body[module] when present, else the shared settings.
        # reasoning_share: cut a no-tools call still reasoning, with no answer
        # yet, after that share of llm.timeout (ReasoningTimeExceeded); 0 = off.
        ep = self.endpoint(module)
        reasoning_seconds = ep.request_timeout * reasoning_share
        payload = {"model": ep.model, "messages": list(messages),
                   "temperature": ep.temperature, **ep.extra_body}
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
            message = self._round_trip(payload, tool_names, ep, module)
            calls = message.get("tool_calls") or []
            content = message.get("content") or ""
            log.debug("LLM tool_calls=%s%s",
                      [c["function"]["name"] for c in calls] or "none",
                      f'\n    said: {content[:500]}' if content else "")
            parsed = [{"name": c["function"]["name"],
                       "args": _loads_first_object(c["function"]["arguments"]),
                       "id": c.get("id")}
                      for c in calls]
            reasoning = message.get("reasoning_content") if self.carry_reasoning else None
            if not parsed:
                return {"tool_call": None, "args": None, "tool_call_id": None,
                        "tool_calls": [], "content": content, "raw_message": message,
                        "reasoning": reasoning}
            # tool_calls: every call the model emitted this turn (tool_choice=
            # "auto" allows several). The singular keys mirror the first call,
            # kept for backward compatibility with callers that expect only one.
            first = parsed[0]
            return {"tool_call": first["name"], "args": first["args"],
                    "tool_call_id": first["id"], "tool_calls": parsed,
                    "content": content, "raw_message": message, "reasoning": reasoning}

        # No-tools (planner) path: the response must be a JSON object. If it
        # will not parse, hand the model its own answer plus the exact parser
        # error and re-ask, up to json_retries times (GH #6).
        for attempt in range(self.json_retries + 1):
            message = self._round_trip(payload, None, ep, module, reasoning_seconds)
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
