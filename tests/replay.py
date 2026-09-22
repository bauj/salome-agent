"""Deterministic replay of a recorded run: no network, no SALOME.

A cassette is {"llm": [<provider message dict>, ...],
               "exec": [{"cmd", "payload", "response"}, ...]}.
Both streams replay strictly in order; any divergence from the recording is
an AssertionError, which is the whole point — a refactor that changes the
order of calls is exactly what this is meant to catch.
"""
import json
import os

from salome_agent.llm import LLMClient

_FIX = os.path.join(os.path.dirname(__file__), "fixtures", "cassettes")


def load_cassette(name: str) -> dict:
    with open(os.path.join(_FIX, f"{name}.json")) as f:
        return json.load(f)


class ReplayLLM(LLMClient):
    """An LLMClient whose provider round-trip is a recording.

    Subclassing (rather than faking `chat`) keeps every parsing rule in
    llm.py — tool-call argument decoding, the JSON-retry path, continuity
    fields — on the tested path.
    """

    def __init__(self, cassette: dict):
        super().__init__(base_url="http://replay.invalid", api_key="none",
                         model="replay", request_timeout=1)
        self._messages = list(cassette["llm"])
        self._i = 0

    @property
    def exhausted(self) -> bool:
        return self._i >= len(self._messages)

    def _round_trip(self, payload, tool_names, extra_body):
        assert not self.exhausted, "cassette ran out of LLM responses"
        message = self._messages[self._i]
        self._i += 1
        self.usage["calls"] += 1
        return message


class ReplayExecutorClient:
    """Replays recorded executor responses, asserting the call sequence."""

    mode = "none"
    name_prefix = ""

    def __init__(self, cassette: dict):
        self._calls = list(cassette["exec"])
        self._i = 0

    def _next(self, cmd: str, payload: dict) -> dict:
        assert self._i < len(self._calls), f"cassette ran out at {cmd}({payload})"
        rec = self._calls[self._i]
        self._i += 1
        assert rec["cmd"] == cmd, f"call {self._i}: recorded {rec['cmd']}, got {cmd}"
        assert rec["payload"] == payload, (
            f"call {self._i} ({cmd}): recorded {rec['payload']}, got {payload}")
        return rec["response"]

    def list_tools(self):
        return self._next("list_tools", {})

    def call_tool(self, name, args, pin=None):
        payload = {"name": name, "args": args}
        if pin is not None:
            payload["pin"] = pin
        return self._next("call_tool", payload)

    def forget(self, names):
        return self._next("forget", {"names": names})

    def introspect(self, names):
        return self._next("introspect", {"names": names})

    def list_study(self):
        return self._next("list_study", {})

    def run(self, code, timeout, module=None):
        return self._next("run", {"code": code, "timeout": timeout, "module": module})

    def sanity(self, module, expectations):
        return self._next("sanity", {"module": module, "expectations": expectations})

    def reset_and_replay(self, scripts):
        return self._next("reset_and_replay", {"scripts": scripts})

    def close(self):
        pass
