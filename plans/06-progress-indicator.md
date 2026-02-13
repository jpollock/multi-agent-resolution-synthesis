# Plan 06: Progress Indicator in Quiet Mode

## Branch: `feature/progress-indicator`

## Problem
In quiet mode, there's no feedback while waiting for providers. A debate can take 30-60+ seconds per round, and the user sees nothing until results appear.

## Changes

### 1. `src/mar/display/renderer.py` — Add Rich Status/Spinner support

```python
from rich.status import Status

class Renderer:
    def __init__(self, verbosity: Verbosity) -> None:
        ...
        self._status: Status | None = None

    def start_provider_work(self, providers: list[str], phase: str = "Generating") -> None:
        """Show a spinner with provider names in quiet mode."""
        if not self.verbose:
            label = f"[bold blue]{phase}[/bold blue]: {', '.join(providers)}"
            self._status = self.console.status(label)
            self._status.start()

    def stop_provider_work(self) -> None:
        """Stop the spinner."""
        if self._status:
            self._status.stop()
            self._status = None

    def update_provider_work(self, message: str) -> None:
        """Update spinner message."""
        if self._status:
            self._status.update(message)
```

### 2. `src/mar/debate/round_robin.py` — Add spinner calls

In `_initial_round()`:
```python
async def _initial_round(self):
    ...
    self.renderer.start_provider_work(list(self.providers.keys()), "Round 1")
    responses = await self._gather_responses(...)
    self.renderer.stop_provider_work()
    return responses
```

In `_critique_round()`:
```python
self.renderer.start_provider_work(list(self.providers.keys()), f"Round {round_num} critiques")
# ... gather ...
self.renderer.stop_provider_work()
```

In `_synthesize()`:
```python
self.renderer.start_provider_work([synth_provider.name], "Synthesizing")
resp = await self._get_response(...)
self.renderer.stop_provider_work()
```

### 3. `src/mar/debate/judge.py` — Same pattern

In `_initial_round()` and `_judge()`, add start/stop spinner calls.

## Verification

1. **Visual check**: Run in quiet mode and verify a spinner appears during each round, showing provider names.
2. **Verbose mode**: Run with `-v` and verify no spinner appears (spinners only in quiet mode).
3. **Phase labels**: Verify spinner shows "Round 1: openai, anthropic, google" then "Round 2 critiques: ..." then "Synthesizing: anthropic".
4. **No interference**: Verify spinner stops cleanly before Rich panels are printed (no garbled output).
