# Plan 03: Synthesis Fallback on Provider Failure

## Branch: `feature/synthesis-fallback`

## Problem
If the synthesis provider fails (timeout, rate limit, network error), the entire debate is lost — all rounds completed but no final answer produced. This happened in practice with an OpenAI ConnectTimeout.

## Changes

### 1. `src/mar/debate/round_robin.py` — Add fallback logic in `_synthesize()`

After selecting `synth_provider`, wrap the call in a try/except that falls back to other providers:

```python
async def _synthesize(self, latest: dict[str, LLMResponse]) -> tuple[str, str]:
    # ... build messages (existing code) ...

    # Determine provider order: preferred first, then others
    synth_name = self.config.synthesis_provider
    if synth_name and synth_name in self.providers:
        ordered = [synth_name] + [n for n in self.providers if n != synth_name]
    else:
        # Auto-select order (from plan 01)
        ordered = self._synthesis_provider_order()

    last_error: Exception | None = None
    for name in ordered:
        provider = self.providers[name]
        model = self.config.model_overrides.get(name)
        try:
            self.renderer.start_round(0)
            resp = await self._get_response(provider, messages, model)
            # ... parse response (existing code) ...
            return final, resolution
        except Exception as e:
            last_error = e
            self.renderer.show_error(name, f"Synthesis failed: {e}")
            continue

    # All providers failed
    raise RuntimeError(f"All providers failed during synthesis. Last error: {last_error}")
```

### 2. `src/mar/debate/judge.py` — Same pattern for judge evaluation

The `_judge()` method also has no fallback. Less critical (there's only one judge), but should at least surface the error clearly. Add a try/except with a clear error message.

### 3. `src/mar/display/renderer.py` — Add synthesis fallback message

```python
def show_fallback(self, failed: str, trying: str) -> None:
    self.console.print(
        f"[yellow]Synthesis failed with {failed}, trying {trying}...[/yellow]"
    )
```

## Verification

1. **Simulated failure**: Temporarily set an invalid API key for the first provider and run with 2+ providers. Verify it falls back to the second provider and completes.
2. **All-fail case**: Set invalid keys for all providers. Verify RuntimeError with clear message.
3. **Success path**: Normal run should behave identically to before (no regression).
4. **Audit trail**: Check that the synthesis round in audit files shows which provider actually did the synthesis.
