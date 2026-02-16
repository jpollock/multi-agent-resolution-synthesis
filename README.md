# MARS — Multi-Agent Resolution Synthesis

Multiple LLMs debate your question through structured rounds of argumentation, critique, and synthesis to produce a single, well-reasoned answer.

## Installation

Requires Python 3.11+.

```sh
git clone https://github.com/jpollock/multi-agent-resolution-synthesis.git
cd multi-agent-resolution-synthesis
pip install -e .
```

Copy `.env.example` to `.env` and add your API keys:

```sh
cp .env.example .env
```

```
MARS_OPENAI_API_KEY=sk-...
MARS_ANTHROPIC_API_KEY=sk-ant-...
MARS_GOOGLE_API_KEY=AIza...
MARS_OLLAMA_BASE_URL=http://localhost:11434
```

## Quick Start

```sh
# Two providers debate (default: openai + anthropic)
mars debate "What is the best sorting algorithm for nearly-sorted data?"

# Pick specific providers
mars debate "Explain CAP theorem" -p openai -p google

# Include context from files
mars debate @prompt.md -c @context.txt

# Check which providers are configured
mars providers
```

## CLI Reference

### `mars debate PROMPT [OPTIONS]`

Run a multi-LLM debate on PROMPT. PROMPT can be plain text or `@file` to read from a file.

| Option | Default | Description |
|--------|---------|-------------|
| `PROMPT` | *(required)* | Question or `@file` path |
| `-c, --context` | | Context text or `@file` (repeatable) |
| `-p, --provider` | `openai anthropic` | Provider name or `provider:model` (repeatable) |
| `-m, --mode` | `round-robin` | Debate mode: `round-robin` or `judge` |
| `-r, --rounds` | `3` | Maximum debate rounds |
| `-j, --judge-provider` | | Provider to act as judge (judge mode) |
| `-s, --synthesis-provider` | | Provider for final synthesis (auto if omitted) |
| `--model` | | `provider:model` override (repeatable) |
| `--threshold` | `0.85` | Convergence similarity threshold (0.0-1.0) |
| `--max-tokens` | `8192` | Max output tokens per LLM call |
| `-t, --temperature` | *(provider default)* | Temperature (0.0-2.0) |
| `-v, --verbose` | off | Stream responses in real-time |
| `-o, --output-dir` | `./mars-output` | Output directory |

### `mars configure`

Set up MARS integration with [Claude Code](https://docs.anthropic.com/en/docs/claude-code). Installs the `/mars:debate` slash command so you can run debates from any Claude Code session.

```sh
mars configure
```

Then in Claude Code:

```
/mars:debate Should we use Redis or Postgres for caching?
```

### `mars providers`

List configured providers with their default models and configuration status.

## Providers

| Provider | Env Var | Default Model |
|----------|---------|---------------|
| `openai` | `MARS_OPENAI_API_KEY` | `gpt-4o` |
| `anthropic` | `MARS_ANTHROPIC_API_KEY` | `claude-sonnet-4-20250514` |
| `google` | `MARS_GOOGLE_API_KEY` | `gemini-2.0-flash` |
| `ollama` | `MARS_OLLAMA_BASE_URL` | `llama3.2` |

Override models per-run with `-p provider:model` or `--model provider:model`.

## Debate Modes

### Round-Robin (default)

All providers answer the prompt independently. Each provider then critiques the others' answers and produces an improved response. This repeats until answers converge (similarity exceeds `--threshold`) or max rounds are reached. A final synthesis step merges the best points into one answer.

### Judge

All providers answer independently. A designated judge provider (`-j`) evaluates every response and produces a final ruling with resolution reasoning.

```sh
mars debate "Is Rust better than Go for CLI tools?" \
  -p openai -p anthropic -p google \
  -m judge -j anthropic
```

## Examples

Basic two-provider debate:

```sh
mars debate "What are the trade-offs between microservices and monoliths?"
```

Three providers with model overrides:

```sh
mars debate "Design a rate limiter" \
  -p openai -p anthropic -p google \
  --model openai:gpt-4.1 --model anthropic:claude-opus-4-20250514
```

Using context files:

```sh
mars debate @question.md -c @codebase-summary.txt -c @requirements.txt
```

Tuning convergence and temperature:

```sh
mars debate "Optimal database indexing strategy" \
  -p openai -p anthropic \
  --threshold 0.70 -t 0.3 -r 5
```

## Output Structure

Each debate produces a timestamped directory:

```
mars-output/<timestamp>_<slug>/
├── final-answer.md
└── audit/
    ├── 00-prompt-and-context.md
    ├── 01-round-1-responses.md
    ├── 02-round-2-critiques.md
    ├── 03-round-3-critiques.md
    ├── attribution.md
    ├── costs.md
    ├── round-diffs.md
    ├── convergence.md
    └── resolution.md
```

| File | Contents |
|------|----------|
| `final-answer.md` | The synthesized final answer |
| `00-prompt-and-context.md` | Original prompt and all context |
| `NN-round-N-responses.md` | Each provider's response for that round |
| `NN-round-N-critiques.md` | Cross-critiques and improved answers |
| `attribution.md` | Per-provider contribution, survival, and influence metrics |
| `costs.md` | Token counts and estimated cost per provider |
| `round-diffs.md` | How each provider's answer changed between rounds |
| `convergence.md` | Why the debate stopped (converged or max rounds) |
| `resolution.md` | Synthesis reasoning: which points were accepted/rejected |

## Analysis Output

### Attribution

Three metrics per provider, computed via sentence-level similarity:

- **Contribution** — percentage of final answer sentences traced to this provider (best-match attribution above threshold).
- **Survival** — percentage of this provider's round-1 sentences that appear in the final answer.
- **Influence** — rate at which other providers adopted this provider's sentences in subsequent rounds.

### Cost Tracking

Token counts (input + output) and estimated USD cost per provider. Pricing uses prefix-matched model lookup (e.g., `claude-sonnet-4-20250514` matches `claude-sonnet-4` pricing). Ollama models show zero cost.

## Claude Code Integration

MARS can be used as a slash command inside [Claude Code](https://docs.anthropic.com/en/docs/claude-code).

```sh
# One-time setup after installing MARS
mars configure
```

This installs `/mars:debate` into `~/.claude/commands/`, making it available in every Claude Code session. Usage:

```
/mars:debate What is the best approach to database sharding?
/mars:debate Compare Kubernetes vs Docker Swarm for container orchestration
```

Claude Code will check your configured providers, run the debate with streaming output, and summarize the result.

## Configuration Tips

**Temperature**: `0.0` for deterministic/factual answers, `0.7` for creative tasks, `1.0+` for experimental diversity. Each provider uses its own default when `-t` is omitted.

**Convergence threshold**: Lower values (e.g., `0.70`) stop debate sooner when answers are roughly similar. Higher values (e.g., `0.95`) force more rounds of refinement. Default `0.85` balances quality and cost.

**Synthesis provider**: By default, MARS prefers Anthropic then OpenAI for synthesis. Use `-s` to override.

**Retries**: All provider calls retry up to 3 times with exponential backoff on transient errors (timeouts, rate limits, 503s).

## License

MIT
