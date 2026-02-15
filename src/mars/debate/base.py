"""Abstract debate strategy with shared LLM interaction methods."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from mars.models import LLMResponse, Message, TokenUsage, Verbosity
from mars.providers.base import retry_with_backoff

if TYPE_CHECKING:
    from mars.display.renderer import Renderer
    from mars.models import DebateConfig, DebateResult
    from mars.output.writer import OutputWriter
    from mars.providers.base import LLMProvider

FINAL_ANSWER_HEADING = "## Final Answer"


class DebateStrategy(ABC):
    def __init__(
        self,
        providers: dict[str, LLMProvider],
        config: DebateConfig,
        renderer: Renderer,
        writer: OutputWriter,
    ) -> None:
        self.providers = providers
        self.config = config
        self.renderer = renderer
        self.writer = writer

    @abstractmethod
    async def run(self) -> DebateResult: ...

    def _full_prompt_with_context(self) -> str:
        """Build the complete original prompt including any context."""
        parts = []
        if self.config.context:
            parts.append("=== CONTEXT ===")
            for i, ctx in enumerate(self.config.context, 1):
                if len(self.config.context) > 1:
                    parts.append(f"\n--- Context {i} ---")
                parts.append(ctx)
            parts.append("\n=== END CONTEXT ===\n")
        parts.append(f"ORIGINAL PROMPT: {self.config.prompt}")
        return "\n".join(parts)

    def _build_system(self) -> Message | None:
        """Build system message with context, or None if no context."""
        if not self.config.context:
            return None
        ctx = "\n\n---\n\n".join(self.config.context)
        return Message(
            role="system",
            content=(
                "You are participating in a structured debate. The user's prompt "
                "includes context that is essential to the task. Treat the context "
                "as primary source material - reference it directly, address its "
                "specifics, and ensure your answer covers every requirement stated "
                "in both the context and prompt.\n\n"
                f"CONTEXT:\n{ctx}"
            ),
        )

    async def _get_response(
        self, provider: LLMProvider, messages: list[Message], model: str | None
    ) -> LLMResponse:
        """Get a response from a provider with verbose/quiet handling."""
        actual_model = model or provider.default_model
        usage = TokenUsage()
        max_tokens = self.config.max_tokens
        temperature = self.config.temperature

        if self.config.verbosity == Verbosity.VERBOSE:
            self.renderer.start_provider_stream(provider.name)
            content = ""
            async for chunk in provider.stream(  # type: ignore[attr-defined]
                messages,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
            ):
                self.renderer.stream_chunk(chunk)
                content += chunk
            self.renderer.end_provider_stream()
            usage = provider.last_usage
        else:
            content, usage = await retry_with_backoff(
                provider.generate,
                messages,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            self.renderer.show_response(provider.name, content)

        return LLMResponse(
            provider=provider.name,
            model=actual_model,
            content=content,
            usage=usage,
        )

    async def _gather_responses(
        self,
        providers: list[tuple[str, LLMProvider]],
        messages: list[Message],
        phase: str = "Generating",
    ) -> list[LLMResponse]:
        """Run providers concurrently in quiet mode, sequentially in verbose."""
        responses: list[LLMResponse] = []

        if self.config.verbosity == Verbosity.VERBOSE:
            for name, provider in providers:
                model = self.config.model_overrides.get(name)
                try:
                    resp = await self._get_response(provider, messages, model)
                    responses.append(resp)
                except Exception as e:
                    self.renderer.show_error(name, str(e))
        else:
            names = [n for n, _ in providers]
            self.renderer.start_work(names, phase)
            tasks = []
            provider_names = []
            for name, provider in providers:
                provider_names.append(name)
                model = self.config.model_overrides.get(name)
                tasks.append(self._get_response(provider, messages, model))

            results = await asyncio.gather(*tasks, return_exceptions=True)
            self.renderer.stop_work()
            for name, r in zip(provider_names, results, strict=True):
                if isinstance(r, Exception):
                    self.renderer.show_error(name, str(r))
                    continue
                responses.append(r)  # type: ignore[arg-type]

        return responses

    def _parse_final_answer(self, content: str) -> tuple[str, str]:
        """Split content on FINAL_ANSWER_HEADING into (final_answer, resolution)."""
        if FINAL_ANSWER_HEADING in content:
            parts = content.split(FINAL_ANSWER_HEADING, 1)
            return parts[1].strip(), parts[0].strip()
        return content, ""
