"""Anthropic provider."""

from __future__ import annotations

from typing import AsyncIterator

import anthropic

from mar.config import AppConfig
from mar.models import Message, TokenUsage


class AnthropicProvider:
    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._client = anthropic.AsyncAnthropic(
            api_key=config.get_api_key("anthropic")
        )
        self._last_usage = TokenUsage()

    @property
    def name(self) -> str:
        return "anthropic"

    @property
    def default_model(self) -> str:
        return self._config.get_default_model("anthropic")

    @property
    def last_usage(self) -> TokenUsage:
        return self._last_usage

    def _split_system(
        self, messages: list[Message]
    ) -> tuple[str, list[dict[str, str]]]:
        system = ""
        msgs: list[dict[str, str]] = []
        for m in messages:
            if m.role == "system":
                system = m.content
            else:
                msgs.append({"role": m.role, "content": m.content})
        return system, msgs

    async def generate(
        self, messages: list[Message], *, model: str | None = None
    ) -> tuple[str, TokenUsage]:
        system, msgs = self._split_system(messages)
        resp = await self._client.messages.create(
            model=model or self.default_model,
            max_tokens=4096,
            system=system,
            messages=msgs,
        )
        usage = TokenUsage(
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
        )
        self._last_usage = usage
        return resp.content[0].text, usage

    async def stream(
        self, messages: list[Message], *, model: str | None = None
    ) -> AsyncIterator[str]:
        system, msgs = self._split_system(messages)
        self._last_usage = TokenUsage()
        async with self._client.messages.stream(
            model=model or self.default_model,
            max_tokens=4096,
            system=system,
            messages=msgs,
        ) as stream:
            async for text in stream.text_stream:
                yield text
            final = await stream.get_final_message()
            self._last_usage = TokenUsage(
                input_tokens=final.usage.input_tokens,
                output_tokens=final.usage.output_tokens,
            )
