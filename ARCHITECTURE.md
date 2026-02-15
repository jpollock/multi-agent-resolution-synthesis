# Architecture

## Project Layout

```
src/mars/
├── __init__.py              # Package marker
├── __main__.py              # python -m mars entry point
├── cli.py                   # Click CLI: argument parsing, config assembly
├── config.py                # AppConfig (pydantic-settings): env vars with MARS_ prefix
├── models.py                # All Pydantic data models
├── debate/
│   ├── engine.py            # DebateEngine: top-level orchestrator
│   ├── base.py              # DebateStrategy ABC + shared methods
│   ├── prompts.py           # Prompt templates for debate strategies
│   ├── round_robin.py       # RoundRobinStrategy: critique loop + synthesis
│   └── judge.py             # JudgeStrategy: single-judge evaluation
├── providers/
│   ├── base.py              # LLMProvider Protocol + retry_with_backoff
│   ├── registry.py          # Provider factory: name → class mapping
│   ├── openai.py            # OpenAI (Chat Completions API)
│   ├── anthropic.py         # Anthropic (Messages API)
│   ├── google.py            # Google Gemini (GenAI SDK)
│   └── ollama.py            # Ollama (HTTP /api/chat)
├── analysis/
│   ├── attribution.py       # Sentence-level attribution analysis
│   └── costs.py             # Token usage aggregation + cost estimation
├── display/
│   └── renderer.py          # Rich terminal output (tables, panels, spinners)
└── output/
    └── writer.py            # Markdown file writer for audit trail
```

## Data Flow

```
CLI (cli.py)
  │  parse args → DebateConfig + AppConfig
  ▼
DebateEngine (debate/engine.py)
  │  build providers from registry
  │  create Renderer + OutputWriter
  ▼
DebateStrategy.run() (round_robin.py or judge.py)
  │  orchestrate rounds: initial → critique → convergence check → synthesis
  │  renderer displays each step; writer persists each step
  ▼
Post-debate analysis (engine.py)
  │  AttributionAnalyzer.analyze(result)
  │  compute_costs(result)
  ▼
Output
  │  renderer: tables + panels to terminal
  │  writer: markdown files to mars-output/
```

## Key Abstractions

### LLMProvider Protocol (`providers/base.py`)

```python
class LLMProvider(Protocol):
    name: str                    # "openai", "anthropic", etc.
    default_model: str           # e.g. "gpt-4o"
    last_usage: TokenUsage       # populated after generate() or stream()

    async def generate(messages, *, model, max_tokens, temperature) -> (str, TokenUsage)
    async def stream(messages, *, model, max_tokens, temperature) -> AsyncIterator[str]
```

All four providers implement this interface. The protocol uses `runtime_checkable` for defensive checks.

`retry_with_backoff()` wraps `generate()` calls with 3 retries and exponential backoff on transient errors (timeouts, rate limits, 5xx).

### DebateStrategy (`debate/base.py`)

Abstract base with shared state: `providers`, `config`, `renderer`, `writer`. Subclasses implement `run() -> DebateResult`.

**RoundRobinStrategy**: Runs N rounds. Round 1 gathers independent answers. Rounds 2+ build critique prompts (your answer + others' answers → improved answer). Checks convergence via `difflib.SequenceMatcher` ratio against `threshold`. Final synthesis asks one provider to merge all answers.

**JudgeStrategy**: One round of independent answers, then a designated judge evaluates and produces a final ruling.

### Models (`models.py`)

All data flows through Pydantic models:

- `DebateConfig` — per-run settings from CLI (prompt, providers, mode, thresholds)
- `DebateResult` — accumulates rounds, final answer, convergence reason
- `DebateRound` — responses + critiques for one round
- `LLMResponse` — provider name, model, content, token usage
- `AttributionReport` / `ProviderAttribution` — per-provider attribution metrics
- `CostReport` / `ProviderCost` — per-provider cost breakdown

### AppConfig (`config.py`)

Uses `pydantic-settings` with `env_prefix = "MARS_"`. Fields map directly to env vars:

- `MARS_OPENAI_API_KEY` → `openai_api_key`
- `MARS_ANTHROPIC_API_KEY` → `anthropic_api_key`
- `MARS_GOOGLE_API_KEY` → `google_api_key`
- `MARS_OLLAMA_BASE_URL` → `ollama_base_url`

## Adding a New Provider

1. Create `src/mars/providers/newprovider.py`
2. Implement a class satisfying the `LLMProvider` protocol:
   - `name` property returning the provider key
   - `default_model` property
   - `last_usage` property returning `TokenUsage`
   - `generate()` async method
   - `stream()` async generator
3. Register in `src/mars/providers/registry.py`:
   ```python
   from mars.providers.newprovider import NewProvider
   _PROVIDERS["newprovider"] = NewProvider
   ```
4. Add default model to `_DEFAULT_MODELS` in `config.py`
5. Add API key field to `AppConfig` and update `get_api_key()`

## Analysis Pipeline

### Attribution (`analysis/attribution.py`)

Operates on sentences extracted via regex splitting (min 20 chars).

- **Contribution**: For each final-answer sentence, find the best-matching sentence across all providers (all rounds). Attribute to the provider with the highest match above threshold.
- **Survival**: For each provider's round-1 sentences, check if any final-answer sentence matches above threshold.
- **Influence**: For each round, check if other providers' next-round answers contain sentences from this provider that weren't in their current-round answers (adoption detection).
- **Round diffs**: Compare consecutive rounds per provider — count added, removed, unchanged sentences and overall similarity.

Similarity uses `difflib.SequenceMatcher` with a default threshold of 0.6.

### Cost Computation (`analysis/costs.py`)

Sums `TokenUsage` (input + output) per provider across all rounds. Looks up model pricing from a static table using prefix matching (e.g., `claude-sonnet-4-20250514` matches the `claude-sonnet-4` entry). Computes per-provider cost and share-of-total.

## Configuration Precedence

Model selection priority (highest wins):

1. `--model provider:model` flag
2. `-p provider:model` flag
3. Default model from `config.py`

Synthesis provider selection:

1. `-s provider` flag
2. Auto: prefer Anthropic, then OpenAI, then others
3. Falls back through all providers on failure

## Shared Strategy Methods

`DebateStrategy` (debate/base.py) contains shared methods used by both strategies:

- `_full_prompt_with_context()` — builds the user prompt with optional context blocks
- `_build_system()` — creates a system message from context (or `None` if no context)
- `_get_response()` — calls a provider with verbose (streaming) or quiet (generate + retry) handling
- `_gather_responses()` — runs multiple providers concurrently (quiet) or sequentially (verbose)
- `_parse_final_answer()` — splits response on `## Final Answer` heading into `(final, resolution)`

The `FINAL_ANSWER_HEADING` constant is defined in `debate/base.py`.

## Prompt Templates

`debate/prompts.py` centralises all prompt text used by the debate strategies:

- `SYSTEM_CONTEXT_TEMPLATE` — system message template with `{context}` placeholder
- `CRITIQUE_INSTRUCTIONS` — instructions for the critique phase (round-robin only)
- `EVALUATION_RULES` — shared "CRITICAL RULES" block used by both synthesis and judge prompts
- `SYNTHESIS_PREAMBLE` — synthesis-specific intro (round-robin)
- `JUDGE_PREAMBLE` — judge-specific intro

## Async Patterns

All LLM calls are async. In **quiet mode**, providers run concurrently via `asyncio.gather(*tasks, return_exceptions=True)`. Failed providers are logged and skipped without aborting the debate. In **verbose mode**, providers run sequentially to avoid interleaved streaming output. The `_gather_responses()` method in the base class encapsulates this pattern.

## Error Handling and Retry

`retry_with_backoff()` in `providers/base.py` wraps async calls with exponential backoff:

- **Max retries**: 3
- **Delay**: 1s, 2s, 4s (exponential)
- **Retryable errors**: detected by exception class name and message substring matching against known transient indicators (`timeout`, `rate_limit`, `connection`, `503`, `529`, etc.)
- **Non-retryable errors**: re-raised immediately

Provider-level errors during debate rounds are caught in `_gather_responses()` and displayed via `renderer.show_error()` without aborting.

## Convergence Algorithm

After each critique round (round 2+), convergence is checked by comparing each provider's current answer to their previous answer:

1. Compute `difflib.SequenceMatcher(None, prev_content, curr_content).ratio()`
2. **All** providers must exceed the threshold for convergence to trigger
3. Default threshold: `0.85` (configurable via `--threshold`)

This is character-level similarity, not semantic similarity.

## Renderer and Writer

**Renderer** (`display/renderer.py`) handles all terminal output via Rich: config tables, response panels, attribution/cost tables, progress spinners, and streaming output. It is stateful (tracks the current `Status` spinner).

**OutputWriter** (`output/writer.py`) produces a timestamped directory of Markdown files:

- `00-prompt.md` — original prompt and context
- `01-round-N-responses.md` — responses per round
- `02-round-N-critiques.md` — critiques per round
- `convergence.md` — convergence reason
- `resolution.md` — resolution analysis
- `final-answer.md` — the synthesised answer
- `attribution.md` — per-provider attribution metrics
- `round-diffs.md` — round-over-round diff analysis
- `costs.md` — token usage and cost breakdown

Each file is written as its step completes, providing crash-resilient output.
