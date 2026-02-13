"""LLM provider protocol."""

from __future__ import annotations

from typing import AsyncIterator, Protocol, runtime_checkable

from mar.models import Message, TokenUsage


@runtime_checkable
class LLMProvider(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def default_model(self) -> str: ...

    @property
    def last_usage(self) -> TokenUsage: ...

    async def generate(
        self, messages: list[Message], *, model: str | None = None, max_tokens: int = 8192, temperature: float | None = None
    ) -> tuple[str, TokenUsage]: ...

    async def stream(
        self, messages: list[Message], *, model: str | None = None, max_tokens: int = 8192, temperature: float | None = None
    ) -> AsyncIterator[str]: ...
