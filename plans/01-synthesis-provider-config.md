# Plan 01: Configurable Synthesis Provider

## Branch: `feature/synthesis-provider-config`

## Problem
`_synthesize()` in `round_robin.py:278` picks `next(iter(self.providers.values()))` — whichever provider is listed first. This is arbitrary and often suboptimal (e.g., Gemini Flash might synthesize a debate where Claude Sonnet participated).

## Changes

### 1. `src/mar/models.py` — Add `synthesis_provider` to `DebateConfig`
```python
class DebateConfig(BaseModel):
    ...
    synthesis_provider: str | None = None  # NEW — defaults to None (auto-select)
```

### 2. `src/mar/cli.py` — Add `--synthesis-provider` / `-s` option
```python
@click.option("-s", "--synthesis-provider", help="Provider for final synthesis (default: auto).")
```
Pass through to `DebateConfig(synthesis_provider=synthesis_provider)`.

### 3. `src/mar/debate/round_robin.py` — Use config in `_synthesize()`
Replace line 278:
```python
# Before:
synth_provider = next(iter(self.providers.values()))

# After:
synth_name = self.config.synthesis_provider
if synth_name:
    if synth_name not in self.providers:
        raise ValueError(f"Synthesis provider '{synth_name}' not in selected providers.")
    synth_provider = self.providers[synth_name]
else:
    # Auto-select: prefer anthropic > openai > first available
    for preferred in ("anthropic", "openai"):
        if preferred in self.providers:
            synth_provider = self.providers[preferred]
            break
    else:
        synth_provider = next(iter(self.providers.values()))
```

### 4. `src/mar/debate/engine.py` — Log which provider is synthesizing
The renderer already shows "Round 0" for synthesis. No engine changes needed.

## Verification

1. **Explicit flag**: Run `mar debate "test" -p openai -p anthropic -p google -s anthropic` — verify synthesis round uses anthropic (check audit file for synthesis round, provider name should be anthropic).
2. **Auto-select**: Run `mar debate "test" -p google -p anthropic` without `-s` — verify anthropic is chosen for synthesis (not google).
3. **Auto-select fallback**: Run `mar debate "test" -p google -p ollama` without `-s` — verify google is chosen (first available when no preferred match).
4. **Invalid provider**: Run `mar debate "test" -p openai -s nonexistent` — verify ValueError with clear message.
5. **Unit test**: Test the selection logic in isolation by mocking providers dict and checking which is picked.
