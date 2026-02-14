"""Google Gemini provider."""

from __future__ import annotations

from typing import AsyncIterator

from google import genai
from google.genai.types import Content, GenerateContentConfig, Part

from mars.config import AppConfig
from mars.models import Message, TokenUsage


class GoogleProvider:
    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._client = genai.Client(api_key=config.get_api_key("google"))
        self._last_usage = TokenUsage()

    @property
    def name(self) -> str:
        return "google"

    @property
    def default_model(self) -> str:
        return self._config.get_default_model("google")

    @property
    def last_usage(self) -> TokenUsage:
        return self._last_usage

    def _build_contents(
        self, messages: list[Message]
    ) -> tuple[str | None, list[Content]]:
        system: str | None = None
        contents: list[Content] = []
        for m in messages:
            if m.role == "system":
                system = m.content
            else:
                role = "model" if m.role == "assistant" else "user"
                contents.append(Content(role=role, parts=[Part(text=m.content)]))
        return system, contents

    async def generate(
        self, messages: list[Message], *, model: str | None = None, max_tokens: int = 8192, temperature: float | None = None
    ) -> tuple[str, TokenUsage]:
        system, contents = self._build_contents(messages)
        cfg_kwargs: dict = {"max_output_tokens": max_tokens}
        if system:
            cfg_kwargs["system_instruction"] = system
        if temperature is not None:
            cfg_kwargs["temperature"] = temperature
        config = GenerateContentConfig(**cfg_kwargs)
        resp = await self._client.aio.models.generate_content(
            model=model or self.default_model,
            contents=contents,
            config=config,
        )
        usage = TokenUsage()
        if resp.usage_metadata:
            usage = TokenUsage(
                input_tokens=resp.usage_metadata.prompt_token_count or 0,
                output_tokens=resp.usage_metadata.candidates_token_count or 0,
            )
        self._last_usage = usage
        return resp.text or "", usage

    async def stream(
        self, messages: list[Message], *, model: str | None = None, max_tokens: int = 8192, temperature: float | None = None
    ) -> AsyncIterator[str]:
        system, contents = self._build_contents(messages)
        cfg_kwargs: dict = {"max_output_tokens": max_tokens}
        if system:
            cfg_kwargs["system_instruction"] = system
        if temperature is not None:
            cfg_kwargs["temperature"] = temperature
        config = GenerateContentConfig(**cfg_kwargs)
        self._last_usage = TokenUsage()
        response = await self._client.aio.models.generate_content_stream(
            model=model or self.default_model,
            contents=contents,
            config=config,
        )
        async for chunk in response:
            if chunk.usage_metadata:
                self._last_usage = TokenUsage(
                    input_tokens=chunk.usage_metadata.prompt_token_count or 0,
                    output_tokens=chunk.usage_metadata.candidates_token_count or 0,
                )
            if chunk.text:
                yield chunk.text
