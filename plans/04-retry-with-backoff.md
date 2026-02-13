# Plan 04: Retry with Exponential Backoff

## Branch: `feature/retry-backoff`

## Problem
Transient API errors (rate limits, timeouts, 500s) crash the entire run. Providers should retry automatically for recoverable errors.

## Changes

### 1. `src/mar/providers/base.py` — Add retry decorator/utility

Create a reusable retry wrapper:
```python
import asyncio
from functools import wraps

RETRYABLE_EXCEPTIONS = (
    TimeoutError,
    ConnectionError,
)

async def retry_with_backoff(
    coro_fn,
    *args,
    max_retries: int = 3,
    base_delay: float = 1.0,
    **kwargs,
):
    """Call coro_fn with exponential backoff on transient errors."""
    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            return await coro_fn(*args, **kwargs)
        except RETRYABLE_EXCEPTIONS as e:
            last_exc = e
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt)
                await asyncio.sleep(delay)
            continue
        except Exception as e:
            # Check for retryable status codes in provider-specific exceptions
            if _is_retryable(e):
                last_exc = e
                if attempt < max_retries:
                    delay = base_delay * (2 ** attempt)
                    await asyncio.sleep(delay)
                continue
            raise
    raise last_exc


def _is_retryable(exc: Exception) -> bool:
    """Check if a provider-specific exception is retryable."""
    exc_str = str(type(exc).__name__).lower()
    # OpenAI: APITimeoutError, RateLimitError, APIConnectionError, InternalServerError
    # Anthropic: APITimeoutError, RateLimitError, APIConnectionError, InternalServerError
    # Google: similar patterns
    retryable_names = ("timeout", "ratelimit", "rate_limit", "connection", "internalserver", "503", "529")
    return any(name in exc_str for name in retryable_names)
```

### 2. Update all 4 providers — Wrap `generate()` calls

**`openai.py`** `generate()`:
```python
async def generate(self, messages, *, model=None, max_tokens=8192):
    async def _call():
        return await self._client.chat.completions.create(...)
    resp = await retry_with_backoff(_call)
    ...
```

Same pattern for `anthropic.py`, `google.py`, `ollama.py`.

Note: Streaming is harder to retry (partially consumed stream). For streaming, retry only on connection-level errors before any chunks arrive. If chunks have started, don't retry.

### 3. `src/mar/models.py` — Add retry config (optional)
```python
class DebateConfig(BaseModel):
    ...
    max_retries: int = 3  # NEW
```

### 4. `src/mar/cli.py` — Add `--max-retries` option
```python
@click.option("--max-retries", type=int, default=3, help="Max retries per API call.")
```

## Verification

1. **Rate limit simulation**: Hard to test directly. Verify by temporarily using an invalid base URL that returns 503 for ollama, confirm retry log messages appear and it retries 3 times before failing.
2. **Timeout retry**: Set a very short timeout on httpx client for ollama, verify it retries.
3. **Non-retryable errors**: Use an invalid API key — verify it does NOT retry (auth errors are not retryable).
4. **No regression**: Normal run with valid keys should produce identical results, no unnecessary delays.
5. **Logging**: Each retry should print a warning via renderer showing attempt number and delay.
