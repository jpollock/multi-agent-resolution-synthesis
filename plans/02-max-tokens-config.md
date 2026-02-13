# Plan 02: Configurable Max Tokens

## Branch: `feature/max-tokens-config`

## Problem
Anthropic provider hardcodes `max_tokens=4096` in both `generate()` and `stream()` (`anthropic.py:51,68`). Synthesis of large debates can be truncated. Other providers use API defaults which vary.

## Changes

### 1. `src/mar/models.py` — Add `max_tokens` to `DebateConfig`
```python
class DebateConfig(BaseModel):
    ...
    max_tokens: int = 8192  # NEW — reasonable default for synthesis
```

### 2. `src/mar/cli.py` — Add `--max-tokens` option
```python
@click.option("--max-tokens", type=int, default=8192, help="Max output tokens per LLM call.")
```

### 3. `src/mar/debate/base.py` — Strategy has access to config already
No changes needed — strategies already have `self.config`.

### 4. `src/mar/providers/base.py` — Update Protocol
Add `max_tokens` parameter to `generate()` and `stream()`:
```python
async def generate(self, messages: list[Message], *, model: str | None = None, max_tokens: int = 8192) -> tuple[str, TokenUsage]: ...
async def stream(self, messages: list[Message], *, model: str | None = None, max_tokens: int = 8192) -> AsyncIterator[str]: ...
```

### 5. Update all 4 providers

**`anthropic.py`**: Replace hardcoded `max_tokens=4096` with parameter.
```python
async def generate(self, messages, *, model=None, max_tokens=8192):
    ...
    resp = await self._client.messages.create(
        model=model or self.default_model,
        max_tokens=max_tokens,
        ...
    )
```
Same for `stream()`.

**`openai.py`**: Add `max_tokens` parameter, pass as `max_completion_tokens`.
```python
async def generate(self, messages, *, model=None, max_tokens=8192):
    resp = await self._client.chat.completions.create(
        model=model or self.default_model,
        messages=...,
        max_completion_tokens=max_tokens,
    )
```

**`google.py`**: Add `max_tokens` to `GenerateContentConfig`:
```python
config = GenerateContentConfig(
    system_instruction=system,
    max_output_tokens=max_tokens,
)
```

**`ollama.py`**: Add to options in request body:
```python
"options": {"num_predict": max_tokens}
```

### 6. Update `_get_response()` in both strategies
Pass `self.config.max_tokens` through to provider calls:
```python
content, usage = await provider.generate(messages, model=model, max_tokens=self.config.max_tokens)
# and for streaming:
async for chunk in provider.stream(messages, model=model, max_tokens=self.config.max_tokens):
```

## Verification

1. **Truncation test**: Run with `--max-tokens 200` and verify responses are short/truncated.
2. **Large synthesis**: Run with `--max-tokens 16384` and verify synthesis produces longer output than default 4096.
3. **Per-provider**: Verify each provider respects the setting by checking output token counts in cost report.
4. **Default behavior**: Run without `--max-tokens` and verify 8192 is used (check that anthropic no longer truncates at 4096).
