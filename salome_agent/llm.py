import json
import os
import re
import time

import requests

from salome_agent.logs import log

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def _parse_json_content(content: str):
    """Real LLMs often wrap the JSON in ```json fences or add a sentence of
    prose around it — try the raw string first, then a fenced block, then the
    first balanced {...} span."""
    content = (content or "").strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
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
    raise ValueError(f"LLM did not return parseable JSON. Raw content:\n{content}")


class LLMClient:
    """Minimal wrapper over any OpenAI-compatible tool-calling chat endpoint
    (spec §9/decision 10 — provider-agnostic, configured by endpoint+key+model).
    No retry/streaming — out of scope per the plan's Task 14 note."""

    def __init__(self, base_url: str, api_key: str, model: str, ssl_cert_file: str = "",
                 temperature: float = 0, request_timeout: float = 120,
                 trace_path: str | None = None):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        # requests `verify`: a CA-bundle path, else True (default: verify against
        # the system trust store — never "" / False, which disables TLS checks).
        self.ssl_cert_file = ssl_cert_file or True
        self.temperature = temperature
        self.request_timeout = request_timeout
        # -vv: append one JSON line per round-trip (request + raw response +
        # whatever reasoning key the provider returns) so tool-selection can be
        # inspected after the run.
        self.trace_path = trace_path

    def _trace(self, request: dict, response: dict) -> None:
        if not self.trace_path:
            return
        entry = {"t": time.strftime("%H:%M:%S"), "request": request, "response": response}
        try:
            os.makedirs(os.path.dirname(self.trace_path) or ".", exist_ok=True)
            with open(self.trace_path, "a") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        except OSError as e:
            log.warning("could not write LLM trace: %s", e)
        log.debug("LLM trace entry:\n%s", json.dumps(entry, indent=2, default=str))

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        # tools: [{"name": str, "parameters": <JSON Schema object>}, ...] — a
        # real per-tool schema, not just a bare name, so the LLM knows each
        # arg's name/type/required-ness instead of guessing.
        payload = {"model": self.model, "messages": messages, "temperature": self.temperature}
        if tools:
            payload["tools"] = [
                {"type": "function", "function": {
                    "name": t["name"],
                    "parameters": t.get("parameters") or {"type": "object", "properties": {}},
                }}
                for t in tools
            ]
            payload["tool_choice"] = "required"
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        tool_names = [t["name"] for t in tools] if tools else None
        log.debug("LLM request: model=%s, %d message(s), tools=%s, timeout=%ss",
                  self.model, len(messages), tool_names or "-", self.request_timeout)
        t0 = time.monotonic()
        resp = None
        for attempt in (1, 2):
            try:
                resp = requests.post(f"{self.base_url}/chat/completions", json=payload,
                                     headers=headers, verify=self.ssl_cert_file,
                                     timeout=self.request_timeout)
                break
            except (requests.Timeout, requests.ConnectionError) as e:
                log.debug("LLM request attempt %d failed after %.1fs: %s",
                          attempt, time.monotonic() - t0, e)
                if attempt == 2:
                    raise
                t0 = time.monotonic()
        resp.raise_for_status()
        body = resp.json()
        message = body["choices"][0]["message"]
        finish = body["choices"][0].get("finish_reason")
        self._trace(
            {"model": self.model, "temperature": self.temperature,
             "tools": tool_names or [], "messages": messages},
            {"finish_reason": finish, "elapsed_s": round(time.monotonic() - t0, 2),
             "message": message},
        )

        if tools:
            calls = message.get("tool_calls") or []
            log.debug("LLM response in %.1fs: finish=%s, tool_calls=%s",
                      time.monotonic() - t0, finish,
                      [c["function"]["name"] for c in calls] or "none")
            tool_call = message["tool_calls"][0]
            call = tool_call["function"]
            return {"tool_call": call["name"], "args": json.loads(call["arguments"]),
                    "tool_call_id": tool_call.get("id")}
        content = message["content"]
        log.debug("LLM response in %.1fs: finish=%s, %d chars of content",
                  time.monotonic() - t0, finish, len(content or ""))
        return _parse_json_content(content)
