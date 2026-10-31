"""Minimal OpenAI-compatible chat-completions client with tool calling.

Credentials come from server-side settings only. The key is sent in the Authorization header
and never logged, stored, or returned by any API. Token usage is whatever the provider reports;
if the provider omits usage we record None - we never estimate or fabricate it.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import httpx

from relay.config import Settings


class LLMError(Exception):
    def __init__(self, message: str, code: str = "LLM_UNAVAILABLE"):
        super().__init__(message)
        self.code = code


@dataclass
class LLMResponse:
    message: dict
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    latency_ms: int
    model: str


class LLMClient:
    def __init__(self, settings: Settings, transport: httpx.BaseTransport | None = None):
        self.model = settings.llm_model
        self._base = settings.llm_base_url.rstrip("/")
        self._key = settings.llm_api_key
        self._timeout = settings.llm_timeout_s
        self._client = httpx.Client(transport=transport)

    def chat(self, messages: list[dict], tools: list[dict], tool_choice: dict | str = "auto",
             max_attempts: int = 3) -> LLMResponse:
        body = {"model": self.model, "messages": messages, "tools": tools, "tool_choice": tool_choice,
                "temperature": 0}
        return self._call(body, max_attempts)

    def chat_text(self, messages: list[dict], max_attempts: int = 3) -> LLMResponse:
        """Plain text completion. Used for customer-facing phrasing; never given tools."""
        body = {"model": self.model, "messages": messages, "temperature": 0}
        return self._call(body, max_attempts)

    def _call(self, body: dict, max_attempts: int) -> LLMResponse:
        headers = {"Authorization": f"Bearer {self._key}", "Content-Type": "application/json"}
        last = "unknown error"
        for attempt in range(1, max_attempts + 1):
            started = time.perf_counter()
            try:
                r = self._client.post(f"{self._base}/chat/completions", json=body, headers=headers,
                                      timeout=self._timeout)
            except httpx.HTTPError as e:
                last = f"transport error: {type(e).__name__}"
            else:
                if r.status_code == 200:
                    return self._parse(r, started)
                last = f"provider returned HTTP {r.status_code}"
                if r.status_code not in (408, 409, 429) and r.status_code < 500:
                    raise LLMError(last)  # 4xx other than rate limits: retrying will not help
            if attempt < max_attempts:
                time.sleep(min(0.5 * 2 ** (attempt - 1), 4))
        raise LLMError(last)

    def _parse(self, r: httpx.Response, started: float) -> LLMResponse:
        try:
            data = r.json()
            message = data["choices"][0]["message"]
        except (ValueError, KeyError, IndexError, TypeError) as e:
            raise LLMError("malformed provider response", "LLM_INVALID_OUTPUT") from e
        usage = data.get("usage") or {}
        return LLMResponse(message=message, prompt_tokens=usage.get("prompt_tokens"),
                           completion_tokens=usage.get("completion_tokens"), total_tokens=usage.get("total_tokens"),
                           latency_ms=int((time.perf_counter() - started) * 1000), model=data.get("model", self.model))
