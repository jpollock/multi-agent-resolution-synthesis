"""OpenAI provider."""

from __future__ import annotations

from typing import AsyncIterator

from openai import AsyncOpenAI

from mar.config import AppConfig
from mar.models import Message, TokenUsage


class OpenAIProvider:
    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._client = AsyncOpenAI(api_key=config.get_api_key("openai"))
        self._last_usage = TokenUsage()

    @property
    def name(self) -> str:
        return "openai"

    @property
    def default_model(self) -> str:
        return self._config.get_default_model("openai")

    @property
    def last_usage(self) -> TokenUsage:
        return self._last_usage

    async def generate(
        self, messages: list[Message], *, model: str | None = None
    ) -> tuple[str, TokenUsage]:
        resp = await self._client.chat.completions.create(
            model=model or self.default_model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
        )
        usage = TokenUsage()
        if resp.usage:
            usage = TokenUsage(
                input_tokens=resp.usage.prompt_tokens or 0,
                output_tokens=resp.usage.completion_tokens or 0,
            )
        self._last_usage = usage
        return resp.choices[0].message.content or "", usage

    async def stream(
        self, messages: list[Message], *, model: str | None = None
    ) -> AsyncIterator[str]:
        resp = await self._client.chat.completions.create(
            model=model or self.default_model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            stream=True,
            stream_options={"include_usage": True},
        )
        self._last_usage = TokenUsage()
        async for chunk in resp:
            if chunk.usage:
                self._last_usage = TokenUsage(
                    input_tokens=chunk.usage.prompt_tokens or 0,
                    output_tokens=chunk.usage.completion_tokens or 0,
                )
            if chunk.choices:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
