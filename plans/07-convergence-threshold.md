# Plan 07: Configurable Convergence Threshold

## Branch: `feature/convergence-threshold`

## Problem
Convergence threshold is hardcoded to 0.9 in `round_robin.py:61` ("similarity threshold 0.9 reached"). This is very strict and almost never triggers, meaning debates always hit max rounds.

## Changes

### 1. `src/mar/models.py` — Add `convergence_threshold` to `DebateConfig`
```python
class DebateConfig(BaseModel):
    ...
    convergence_threshold: float = 0.85  # NEW — slightly relaxed default
```

### 2. `src/mar/cli.py` — Add `--threshold` option
```python
@click.option("--threshold", type=float, default=0.85, help="Convergence similarity threshold (0.0-1.0).")
```
Pass as `convergence_threshold=threshold`.

### 3. `src/mar/debate/round_robin.py` — Use config threshold

In `_has_converged()`, change from hardcoded 0.9 to instance method that reads config:

```python
# Change _has_converged from @staticmethod to instance method
def _has_converged(self, prev: dict[str, LLMResponse], curr: dict[str, LLMResponse]) -> bool:
    common = set(prev) & set(curr)
    if not common:
        return False
    threshold = self.config.convergence_threshold
    for name in common:
        ratio = difflib.SequenceMatcher(
            None, prev[name].content, curr[name].content
        ).ratio()
        if ratio < threshold:
            return False
    return True
```

Update the convergence reason message to use the actual threshold:
```python
reason = (
    f"Answers converged after round {round_num} "
    f"(similarity threshold {self.config.convergence_threshold} reached)."
)
```

## Verification

1. **Low threshold**: Run `mar debate "test" --threshold 0.5 -r 5` — verify convergence happens earlier (rounds 2-3).
2. **High threshold**: Run with `--threshold 0.99` — verify it always hits max rounds.
3. **Default**: Run without `--threshold` — verify 0.85 is used and convergence message shows 0.85.
4. **Convergence message**: Check audit/convergence.md shows the actual threshold used.
5. **Boundary check**: `--threshold 0.0` should converge immediately after round 2. `--threshold 1.0` should never converge.
