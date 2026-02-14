"""Ollama provider (local LLMs via HTTP API)."""

from __future__ import annotations

import json
from typing import AsyncIterator

import httpx

from mars.config import AppConfig
from mars.models import Message, TokenUsage


class OllamaProvider:
    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._base_url = config.ollama_base_url.rstrip("/")
        self._last_usage = TokenUsage()

    @property
    def name(self) -> str:
        return "ollama"

    @property
    def default_model(self) -> str:
        return self._config.get_default_model("ollama")

    @property
    def last_usage(self) -> TokenUsage:
        return self._last_usage

    async def generate(
        self, messages: list[Message], *, model: str | None = None, max_tokens: int = 8192, temperature: float | None = None
    ) -> tuple[str, TokenUsage]:
        options: dict = {"num_predict": max_tokens}
        if temperature is not None:
            options["temperature"] = temperature
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{self._base_url}/api/chat",
                json={
                    "model": model or self.default_model,
                    "messages": [
                        {"role": m.role, "content": m.content} for m in messages
                    ],
                    "stream": False,
                    "options": options,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            usage = TokenUsage(
                input_tokens=data.get("prompt_eval_count", 0) or 0,
                output_tokens=data.get("eval_count", 0) or 0,
            )
            self._last_usage = usage
            return data["message"]["content"], usage

    async def stream(
        self, messages: list[Message], *, model: str | None = None, max_tokens: int = 8192, temperature: float | None = None
    ) -> AsyncIterator[str]:
        options: dict = {"num_predict": max_tokens}
        if temperature is not None:
            options["temperature"] = temperature
        self._last_usage = TokenUsage()
        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream(
                "POST",
                f"{self._base_url}/api/chat",
                json={
                    "model": model or self.default_model,
                    "messages": [
                        {"role": m.role, "content": m.content} for m in messages
                    ],
                    "stream": True,
                    "options": options,
                },
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    data = json.loads(line)
                    if data.get("done"):
                        self._last_usage = TokenUsage(
                            input_tokens=data.get("prompt_eval_count", 0) or 0,
                            output_tokens=data.get("eval_count", 0) or 0,
                        )
                    content = data.get("message", {}).get("content", "")
                    if content:
                        yield content
