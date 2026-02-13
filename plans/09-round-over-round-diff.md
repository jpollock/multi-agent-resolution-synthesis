# Plan 09: Round-over-Round Diff Tracking

## Branch: `feature/round-diff-tracking`

## Problem
No visibility into what changed between rounds per provider. Hard to know if the debate actually improved answers or just shuffled words.

## Changes

### 1. `src/mar/models.py` — Add `RoundDiff` model
```python
class RoundDiff(BaseModel):
    provider: str
    from_round: int
    to_round: int
    similarity: float = 0.0  # Overall text similarity
    sentences_added: int = 0
    sentences_removed: int = 0
    sentences_unchanged: int = 0

class AttributionReport(BaseModel):
    ...
    round_diffs: list[RoundDiff] = Field(default_factory=list)  # NEW
```

### 2. `src/mar/analysis/attribution.py` — Compute round diffs

Add method to `AttributionAnalyzer`:
```python
def _compute_round_diffs(self, provider_data: dict[str, _ProviderText]) -> list[RoundDiff]:
    diffs = []
    for name, data in provider_data.items():
        rounds_sorted = sorted(data.round_sentences.keys())
        for i in range(len(rounds_sorted) - 1):
            r1 = rounds_sorted[i]
            r2 = rounds_sorted[i + 1]
            s1 = data.round_sentences[r1]
            s2 = data.round_sentences[r2]

            # Count unchanged (matched above threshold)
            matched_s2 = set()
            unchanged = 0
            for sent in s1:
                idx, score = _best_match(sent, s2)
                if score >= self.threshold:
                    unchanged += 1
                    matched_s2.add(idx)

            removed = len(s1) - unchanged
            added = len(s2) - len(matched_s2)
            similarity = difflib.SequenceMatcher(
                None, "\n".join(s1), "\n".join(s2)
            ).ratio()

            diffs.append(RoundDiff(
                provider=name,
                from_round=r1,
                to_round=r2,
                similarity=round(similarity, 3),
                sentences_added=added,
                sentences_removed=removed,
                sentences_unchanged=unchanged,
            ))
    return diffs
```

Call in `analyze()` and attach to report:
```python
report.round_diffs = self._compute_round_diffs(provider_data)
```

### 3. `src/mar/display/renderer.py` — Show round diffs

Add method:
```python
def show_round_diffs(self, diffs: list[RoundDiff]) -> None:
    if not diffs:
        return
    self.console.print()
    self.console.rule("[bold cyan]Round-over-Round Changes[/bold cyan]")
    table = Table(show_header=True, header_style="bold")
    table.add_column("Provider", style="bold green")
    table.add_column("Rounds", justify="center")
    table.add_column("Similarity", justify="right")
    table.add_column("Added", justify="right", style="green")
    table.add_column("Removed", justify="right", style="red")
    table.add_column("Unchanged", justify="right")

    for d in diffs:
        table.add_row(
            d.provider,
            f"{d.from_round} -> {d.to_round}",
            f"{d.similarity:.1%}",
            str(d.sentences_added),
            str(d.sentences_removed),
            str(d.sentences_unchanged),
        )
    self.console.print(table)
```

### 4. `src/mar/output/writer.py` — Write round diffs to audit

Add to `write_attribution()` or new `write_round_diffs()`:
```python
def write_round_diffs(self, diffs: list[RoundDiff]) -> None:
    lines = ["# Round-over-Round Changes\n"]
    lines.append("| Provider | Rounds | Similarity | Added | Removed | Unchanged |")
    lines.append("|----------|--------|-----------|-------|---------|-----------|")
    for d in diffs:
        lines.append(
            f"| {d.provider} | {d.from_round}->{d.to_round} "
            f"| {d.similarity:.1%} | +{d.sentences_added} | -{d.sentences_removed} | {d.sentences_unchanged} |"
        )
    self._write(self._audit / "round-diffs.md", "\n".join(lines))
```

### 5. `src/mar/debate/engine.py` — Wire in

After attribution analysis:
```python
renderer.show_round_diffs(attribution.round_diffs)
writer.write_round_diffs(attribution.round_diffs)
```

## Verification

1. **Multi-round run**: Run a 3-round debate. Verify diffs show for rounds 1->2 and 2->3 for each provider.
2. **Sentence counts**: For each diff, `unchanged + removed` should equal the source round sentence count. `unchanged + added` should equal the target round sentence count.
3. **Similarity range**: All similarity values should be between 0.0 and 1.0.
4. **Display**: Verify table appears in terminal and `audit/round-diffs.md` is written.
5. **Judge mode**: Run in judge mode (only 2 rounds). Verify diffs are empty or minimal (providers only have 1 round of responses).
