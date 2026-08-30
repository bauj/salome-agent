import json

import requests


class LLMClient:
    """Minimal wrapper over any OpenAI-compatible tool-calling chat endpoint
    (spec §9/decision 10 — provider-agnostic, configured by endpoint+key+model).
    No retry/streaming — out of scope per the plan's Task 14 note."""

    def __init__(self, base_url: str, api_key: str, model: str, temperature: float = 0):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
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
                              headers=headers, timeout=120)
        resp.raise_for_status()
        message = resp.json()["choices"][0]["message"]

        if tools:
            tool_call = message["tool_calls"][0]
            call = tool_call["function"]
            return {"tool_call": call["name"], "args": json.loads(call["arguments"]),
                    "tool_call_id": tool_call.get("id")}
        return json.loads(message["content"])
