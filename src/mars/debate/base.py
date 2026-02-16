"""Abstract debate strategy with shared LLM interaction methods."""

from __future__ import annotations

import asyncio
import re
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from mars.debate.prompts import SYSTEM_CONTEXT_TEMPLATE
from mars.models import LLMResponse, Message, TokenUsage, Verbosity
from mars.providers.base import retry_with_backoff

if TYPE_CHECKING:
    from mars.display.renderer import Renderer
    from mars.models import DebateConfig, DebateResult
    from mars.output.writer import OutputWriter
    from mars.providers.base import LLMProvider

FINAL_ANSWER_HEADING = "## Final Answer"


def _sanitize_secrets(text: str) -> str:
    """Strip API keys and tokens from error text."""
    # Common key patterns: sk-..., key-..., AIza..., ya29..., Bearer tokens
    text = re.sub(r"(sk-[A-Za-z0-9_-]{8})[A-Za-z0-9_-]+", r"\1...", text)
    text = re.sub(r"(key-[A-Za-z0-9]{8})[A-Za-z0-9]+", r"\1...", text)
    text = re.sub(r"(AIza[A-Za-z0-9_-]{8})[A-Za-z0-9_-]+", r"\1...", text)
    text = re.sub(r"(ya29\.)[A-Za-z0-9_.-]+", r"\1...", text)
    text = re.sub(r"(Bearer\s+)[A-Za-z0-9_./+-]+", r"\1[REDACTED]", text)
    return text


def _format_provider_error(error: Exception) -> str:
    """Turn raw SDK exceptions into concise, actionable messages."""
    msg = _sanitize_secrets(str(error))
    msg_lower = msg.lower()

    # 404 / model not found
    if "404" in msg or "not_found" in msg_lower or "does not exist" in msg_lower:
        # Try to extract model name from various SDK formats
        # OpenAI: "The model `gpt-oss-120b` does not exist"
        # Anthropic: "model: claude-4-6-opus@20260215"
        # Google/Vertex: "models/gemini-3-pro` was not found"
        model_name = None
        for pattern in [
            r"The model `([^`]+)`",
            r"model:\s*(\S+)",
            r"models/([^\s`'\"]+)",
        ]:
            m = re.search(pattern, msg)
            if m:
                model_name = m.group(1).rstrip("`")
                break
        if model_name:
            return f"Model '{model_name}' not found. Check the model name with your provider."
        return "Model not found. Check the model name with your provider."

    # 401 / authentication
    if "401" in msg or "unauthorized" in msg_lower or "invalid.*api.?key" in msg_lower:
        return "Authentication failed. Run 'mars configure' to check your API key."

    # 403 / permission denied
    if "403" in msg or "forbidden" in msg_lower or "permission" in msg_lower:
        return "Permission denied. Check your API key permissions or account access."

    # Quota / billing
    if "quota" in msg_lower or "billing" in msg_lower or "insufficient" in msg_lower:
        return "Quota exceeded or billing issue. Check your account at your provider's console."

    # Connection errors
    if "connection" in msg_lower or ("connect" in msg_lower and "refused" in msg_lower):
        return "Connection failed. Check that the service is reachable."

    # Fall back: strip JSON noise, keep first meaningful line
    # Many SDK errors start with a status line then dump JSON
    lines = msg.strip().split("\n")
    first = lines[0].strip()
    # Remove trailing JSON blob from single-line errors
    json_start = first.find("{'")
    if json_start > 20:
        first = first[:json_start].strip().rstrip(".-")
    if len(first) > 200:
        first = first[:200] + "..."
    return first


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
            content=SYSTEM_CONTEXT_TEMPLATE.format(context=ctx),
        )

    async def _get_response(
        self,
        provider: LLMProvider,
        messages: list[Message],
        model: str | None,
        *,
        participant_id: str | None = None,
    ) -> LLMResponse:
        """Get a response from a provider with verbose/quiet handling."""
        display_name = participant_id or provider.name
        actual_model = model or provider.default_model
        usage = TokenUsage()
        max_tokens = self.config.max_tokens
        temperature = self.config.temperature

        if self.config.verbosity == Verbosity.VERBOSE:
            self.renderer.start_provider_stream(display_name)
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
            self.renderer.show_response(display_name, content)

        return LLMResponse(
            provider=display_name,
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
                    resp = await self._get_response(provider, messages, model, participant_id=name)
                    responses.append(resp)
                except Exception as e:
                    self.renderer.show_error(name, _format_provider_error(e))
        else:
            names = [n for n, _ in providers]
            self.renderer.start_work(names, phase)
            tasks = []
            provider_names = []
            for name, provider in providers:
                provider_names.append(name)
                model = self.config.model_overrides.get(name)
                tasks.append(self._get_response(provider, messages, model, participant_id=name))

            results = await asyncio.gather(*tasks, return_exceptions=True)
            self.renderer.stop_work()
            for name, r in zip(provider_names, results, strict=True):
                if isinstance(r, Exception):
                    self.renderer.show_error(name, _format_provider_error(r))
                    continue
                responses.append(r)  # type: ignore[arg-type]

        return responses

    def _parse_final_answer(self, content: str) -> tuple[str, str]:
        """Split content on FINAL_ANSWER_HEADING into (final_answer, resolution)."""
        if FINAL_ANSWER_HEADING in content:
            parts = content.split(FINAL_ANSWER_HEADING, 1)
            return parts[1].strip(), parts[0].strip()
        return content, ""
