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

    def chat(self, messages: list[dict], tools: list[str] | None = None) -> dict:
        payload = {"model": self.model, "messages": messages, "temperature": self.temperature}
        if tools:
            payload["tools"] = [
                {"type": "function", "function": {"name": name, "parameters": {"type": "object"}}}
                for name in tools
            ]
            payload["tool_choice"] = "required"
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        resp = requests.post(f"{self.base_url}/chat/completions", json=payload,
                              headers=headers, timeout=120)
        resp.raise_for_status()
        message = resp.json()["choices"][0]["message"]

        if tools:
            call = message["tool_calls"][0]["function"]
            return {"tool_call": call["name"], "args": json.loads(call["arguments"])}
        return json.loads(message["content"])
