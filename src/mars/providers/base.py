"""LLM provider protocol and retry utilities."""

from __future__ import annotations

import asyncio
import logging
import random
import re
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any, Protocol, runtime_checkable

from mars.models import Message, TokenUsage

logger = logging.getLogger(__name__)

_RETRYABLE_NAMES = (
    "timeout",
    "ratelimit",
    "rate_limit",
    "connection",
    "internalserver",
    "server_error",
    "503",
    "529",
)


def _is_retryable(exc: Exception) -> bool:
    """Check if an exception is transient and worth retrying."""
    name = type(exc).__name__.lower()
    msg = str(exc).lower()
    return any(r in name or r in msg for r in _RETRYABLE_NAMES)


def _sanitize_log_message(exc: Exception) -> str:
    """Sanitize exception text for logging (strip API keys/tokens)."""
    text = str(exc)
    text = re.sub(r"(sk-[A-Za-z0-9_-]{8})[A-Za-z0-9_-]+", r"\1...", text)
    text = re.sub(r"(key-[A-Za-z0-9]{8})[A-Za-z0-9]+", r"\1...", text)
    text = re.sub(r"(AIza[A-Za-z0-9_-]{8})[A-Za-z0-9_-]+", r"\1...", text)
    text = re.sub(r"(ya29\.)[A-Za-z0-9_.-]+", r"\1...", text)
    text = re.sub(r"(Bearer\s+)[A-Za-z0-9_./+-]+", r"\1[REDACTED]", text)
    return text


async def retry_with_backoff(
    fn: Callable[..., Awaitable[Any]],
    *args: Any,
    max_retries: int = 3,
    base_delay: float = 1.0,
    **kwargs: Any,
) -> Any:
    """Call an async function with exponential backoff on transient errors."""
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return await fn(*args, **kwargs)
        except (TimeoutError, ConnectionError, OSError) as e:
            last_exc = e
        except Exception as e:
            if _is_retryable(e):
                last_exc = e
            else:
                raise
        if attempt < max_retries:
            delay = base_delay * (2**attempt) + random.uniform(0, base_delay)
            logger.warning(
                "Retry %d/%d after %.1fs: %s",
                attempt + 1, max_retries, delay, _sanitize_log_message(last_exc),
            )
            await asyncio.sleep(delay)
    raise last_exc  # type: ignore[misc]


@runtime_checkable
class LLMProvider(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def default_model(self) -> str: ...

    @property
    def last_usage(self) -> TokenUsage: ...

    async def generate(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
        max_tokens: int = 8192,
        temperature: float | None = None,
    ) -> tuple[str, TokenUsage]: ...

    async def stream(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
        max_tokens: int = 8192,
        temperature: float | None = None,
    ) -> AsyncIterator[str]: ...
