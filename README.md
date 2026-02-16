# MARS — Multi-Agent Resolution Synthesis

Multiple LLMs debate your question through structured rounds of argumentation, critique, and synthesis to produce a single, well-reasoned answer.

## Installation

Requires Python 3.11+.

```sh
pip install mars-llm
```

## Quick Start

1. Configure your API keys (one-time setup):

```sh
mars configure
```

This walks you through setting up API keys for each provider interactively.
You need at least two providers configured to run a debate.

2. Run a debate:

```sh
mars debate "What is the best sorting algorithm for nearly-sorted data?"
```

3. View the results:

```sh
mars show
```

4. Check provider status anytime:

```sh
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

Interactive setup for API keys and integrations. Prompts for each provider's
API key, validates it, and stores keys in `~/.mars/config`. Optionally sets
up [Claude Code](https://docs.anthropic.com/en/docs/claude-code) integration.

### `mars providers`

List configured providers with their default models and configuration status.

### `mars show [SUBCOMMAND]`

View results of a completed debate. With no subcommand, shows a compact summary.

| Subcommand | Description |
|------------|-------------|
| *(none)* | Compact summary: prompt, providers, cost, attribution, answer |
| `answer` | Final synthesized answer only |
| `costs` | Token usage and cost breakdown |
| `attribution` | Per-provider contribution and influence metrics |
| `rounds` | Round-by-round responses and diffs |

| Option | Default | Description |
|--------|---------|-------------|
| `--debate` | *(most recent)* | Path to a specific debate directory |
| `-o, --output-dir` | `./mars-output` | Output directory |

### `mars history`

List past debates with timestamps, providers, rounds, and costs.

| Option | Default | Description |
|--------|---------|-------------|
| `-n, --limit` | *(all)* | Show only the last N debates |
| `-o, --output-dir` | `./mars-output` | Output directory |

### `mars copy`

Copy the final answer to the system clipboard.

| Option | Default | Description |
|--------|---------|-------------|
| `--full` | off | Include prompt, answer, and attribution |
| `--debate` | *(most recent)* | Path to a specific debate directory |
| `-o, --output-dir` | `./mars-output` | Output directory |

## Configuration

MARS looks for API keys in three places (highest priority wins):

| Source | Example | Priority |
|--------|---------|----------|
| Environment variables | `export MARS_OPENAI_API_KEY=sk-...` | Highest |
| Local `.env` file | `MARS_OPENAI_API_KEY=sk-...` in `.env` | Medium |
| Global config | `~/.mars/config` (set by `mars configure`) | Lowest |

This means you can set keys globally with `mars configure` and override
them per-project with a local `.env` file if needed.

### Providers

| Provider | Config Variable | Default Model |
|----------|----------------|---------------|
| `openai` | `MARS_OPENAI_API_KEY` | `gpt-4o` |
| `anthropic` | `MARS_ANTHROPIC_API_KEY` | `claude-sonnet-4-20250514` |
| `google` | `MARS_GOOGLE_API_KEY` | `gemini-2.0-flash` |
| `vertex` | `MARS_VERTEX_PROJECT_ID` | `claude-opus-4-6` |
| `ollama` | `MARS_OLLAMA_BASE_URL` | `llama3.2` |

Override models per-run with `-p provider:model` or `--model provider:model`.

### Vertex AI (Google Cloud)

Vertex AI acts as a gateway to both Claude and Gemini models through a single
authentication mechanism (Application Default Credentials).

Setup:

```sh
gcloud auth application-default login
mars configure   # enter your GCP project ID and region
```

Use `-p vertex:model` to specify models. The same `vertex` provider routes
to Claude or Gemini based on the model name:

```sh
# Claude via Vertex
mars debate "Question" -p vertex:claude-sonnet-4 -p openai

# Gemini via Vertex
mars debate "Question" -p vertex:gemini-2.5-flash -p openai

# Both Claude and Gemini via Vertex
mars debate "Question" \
  -p vertex:claude-sonnet-4 \
  -p vertex:gemini-2.5-flash
```

Vertex AI config variables:

| Variable | Description |
|----------|-------------|
| `MARS_VERTEX_PROJECT_ID` | GCP project ID |
| `MARS_VERTEX_REGION` | GCP region (default: `us-central1`) |

Auto-detected from `ANTHROPIC_VERTEX_PROJECT_ID`, `GOOGLE_CLOUD_PROJECT`,
and `CLOUD_ML_REGION` if set.

### Default Providers

Set default providers so you don't need `-p` every time:

```sh
mars configure   # prompted at the end for default providers
```

Or set `MARS_DEFAULT_PROVIDERS` directly:

```sh
export MARS_DEFAULT_PROVIDERS="vertex:claude-opus-4-6,vertex:gemini-2.5-flash"
```

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

Vertex AI — Claude vs Gemini:

```sh
mars debate "Best practices for API versioning" \
  -p vertex:claude-sonnet-4 \
  -p vertex:gemini-2.5-flash -v
```

Tuning convergence and temperature:

```sh
mars debate "Optimal database indexing strategy" \
  -p openai -p anthropic \
  --threshold 0.70 -t 0.3 -r 5
```

Reviewing results after a debate:

```sh
mars show                    # summary of most recent debate
mars show answer             # just the final answer
mars show costs              # cost breakdown
mars history                 # list all past debates
mars history -n 5            # last 5 debates
mars copy                    # copy final answer to clipboard
mars copy --full             # copy prompt + answer + attribution
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

`mars configure` detects [Claude Code](https://docs.anthropic.com/en/docs/claude-code)
and offers to install `/mars:debate` as a slash command. Once installed, you
can run debates from any Claude Code session:

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
