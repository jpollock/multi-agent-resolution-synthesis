# Implementation Order & Dependencies

Plans are in `/plans/` directory. Each plan corresponds to a feature branch.

## Order of Implementation

Dependencies flow downward — implement in this order:

### Phase 1: Core Provider Improvements (no interdependencies)
1. **[Plan 02](02-max-tokens-config.md)** `feature/max-tokens-config` — Configurable max tokens (unblocks longer synthesis)
2. **[Plan 05](05-temperature-control.md)** `feature/temperature-control` — Temperature control
3. **[Plan 10](10-inline-model-selection.md)** `feature/inline-model-selection` — Inline provider:model syntax

### Phase 2: Reliability (builds on Phase 1 provider changes)
4. **[Plan 04](04-retry-with-backoff.md)** `feature/retry-backoff` — Retry with exponential backoff
5. **[Plan 01](01-synthesis-provider-config.md)** `feature/synthesis-provider-config` — Configurable synthesis provider
6. **[Plan 03](03-synthesis-fallback.md)** `feature/synthesis-fallback` — Synthesis fallback (depends on Plan 01 for provider ordering)

### Phase 3: UX Improvements
7. **[Plan 06](06-progress-indicator.md)** `feature/progress-indicator` — Progress spinner
8. **[Plan 07](07-convergence-threshold.md)** `feature/convergence-threshold` — Configurable convergence threshold

### Phase 4: Analysis Improvements
9. **[Plan 08](08-attribution-novel-content.md)** `feature/attribution-novel-content` — Track synthesizer novel content
10. **[Plan 09](09-round-over-round-diff.md)** `feature/round-diff-tracking` — Round-over-round diff tracking

## Branch Strategy

- Each feature gets its own branch off `main`
- After implementing and verifying, merge to `main` before starting next feature
- This keeps each branch small and testable

## Context Compaction Note

If context is compacted during implementation, read this file first:
`/Users/jeremy.pollock/development/ai/multi-agent-resolution/plans/00-implementation-order.md`

Then read the specific plan file for the current feature being worked on. The plan files contain all details needed to implement without prior context.
