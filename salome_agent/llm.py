import json
import re

import requests

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

    def __init__(self, base_url: str, api_key: str, model: str, ssl_cert_file: str = "", temperature: float = 0):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        # requests `verify`: a CA-bundle path, else True (default: verify against
        # the system trust store — never "" / False, which disables TLS checks).
        self.ssl_cert_file = ssl_cert_file or True
        self.temperature = temperature

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
        resp = requests.post(f"{self.base_url}/chat/completions", json=payload,
                              headers=headers, verify=self.ssl_cert_file, timeout=120)
        resp.raise_for_status()
        message = resp.json()["choices"][0]["message"]

        if tools:
            tool_call = message["tool_calls"][0]
            call = tool_call["function"]
            return {"tool_call": call["name"], "args": json.loads(call["arguments"]),
                    "tool_call_id": tool_call.get("id")}
        return _parse_json_content(message["content"])
