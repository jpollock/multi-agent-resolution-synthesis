# Plan 10: Inline Provider:Model Selection

## Branch: `feature/inline-model-selection`

## Problem
Specifying models requires separate `--model` flags: `-p openai -p anthropic --model openai:gpt-4.1 --model anthropic:claude-sonnet-4`. Verbose and error-prone. Should support `-p openai:gpt-4.1 -p anthropic:claude-sonnet-4`.

## Changes

### 1. `src/mar/cli.py` — Parse provider:model in `-p` flag

Update the provider parsing logic:
```python
providers = []
model_overrides: dict[str, str] = {}

for p in provider:
    if ":" in p:
        prov, mod = p.split(":", 1)
        providers.append(prov)
        model_overrides[prov] = mod
    else:
        providers.append(p)

if not providers:
    providers = ["openai", "anthropic"]

# Also handle explicit --model overrides (these take precedence)
for m in model:
    if ":" not in m:
        raise click.BadParameter(...)
    prov, mod = m.split(":", 1)
    model_overrides[prov] = mod
```

### 2. Update help text
```python
@click.option(
    "-p", "--provider", multiple=True,
    help="Provider or provider:model (e.g. openai:gpt-4.1). Repeatable.",
)
```

### 3. Validation
After parsing, validate that all provider names (without model suffix) are in `AVAILABLE_PROVIDERS`.

## Verification

1. **Inline syntax**: Run `mar debate "test" -p openai:gpt-4o-mini -p anthropic:claude-haiku-3` — verify models are used (check cost report for model names).
2. **Mixed syntax**: Run `mar debate "test" -p openai -p anthropic --model openai:gpt-4o-mini` — verify `--model` override works.
3. **Override precedence**: Run `-p openai:gpt-4o -model openai:gpt-4o-mini` — verify `--model` wins.
4. **Plain providers**: Run `-p openai -p anthropic` (no model suffix) — verify default models are used.
5. **Invalid provider**: Run `-p nonexistent:model` — verify error message.
6. **Backward compatibility**: All existing command invocations should still work.
