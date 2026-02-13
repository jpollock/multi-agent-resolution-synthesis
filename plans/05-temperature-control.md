# Plan 05: Temperature Control

## Branch: `feature/temperature-control`

## Problem
No way to set temperature per provider or globally. Lower temperature for synthesis produces more focused output; higher temperature for initial brainstorming produces more diverse responses.

## Changes

### 1. `src/mar/models.py` — Add `temperature` to `DebateConfig`
```python
class DebateConfig(BaseModel):
    ...
    temperature: float | None = None  # NEW — None means use provider default
```

### 2. `src/mar/cli.py` — Add `--temperature` / `-t` option
```python
@click.option("-t", "--temperature", type=float, default=None, help="Temperature (0.0-2.0). Default: provider default.")
```

### 3. `src/mar/providers/base.py` — Add temperature to Protocol
```python
async def generate(self, messages, *, model=None, max_tokens=8192, temperature: float | None = None) -> tuple[str, TokenUsage]: ...
async def stream(self, messages, *, model=None, max_tokens=8192, temperature: float | None = None) -> AsyncIterator[str]: ...
```

### 4. Update all 4 providers

**`openai.py`**: Pass `temperature=temperature` to `chat.completions.create()` (only if not None).
**`anthropic.py`**: Pass `temperature=temperature` to `messages.create()` (only if not None).
**`google.py`**: Add `temperature=temperature` to `GenerateContentConfig` (only if not None).
**`ollama.py`**: Add `"temperature": temperature` to options (only if not None).

For each provider, conditionally include the parameter:
```python
kwargs = {}
if temperature is not None:
    kwargs["temperature"] = temperature
resp = await self._client.chat.completions.create(
    model=model or self.default_model,
    messages=...,
    **kwargs,
)
```

### 5. Update `_get_response()` in both strategies
Pass `temperature=self.config.temperature` to provider calls.

## Verification

1. **Low temperature**: Run `mar debate "1+1=" -t 0.0 -p openai -p anthropic` — verify consistent, deterministic-ish output.
2. **High temperature**: Run same with `-t 1.5` — verify more varied output.
3. **Default (None)**: Run without `-t` — verify providers use their own defaults (no temperature key sent).
4. **Check API calls**: In verbose mode, verify no errors about invalid temperature values.
