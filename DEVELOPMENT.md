# Development Guide

## Prerequisites

- Python 3.11+
- API keys for at least one LLM provider (OpenAI, Anthropic, or Google)

## Setup

Clone the repository:

```sh
git clone https://github.com/jpollock/multi-agent-resolution-synthesis.git
cd multi-agent-resolution-synthesis
```

Create and activate a virtual environment:

```sh
python -m venv .venv
source .venv/bin/activate
```

Install in editable mode with dev dependencies:

```sh
pip install -e ".[dev]"
```

Configure environment variables:

```sh
cp .env.example .env
# Edit .env with your API keys
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `MARS_OPENAI_API_KEY` | For OpenAI provider | OpenAI API key |
| `MARS_ANTHROPIC_API_KEY` | For Anthropic provider | Anthropic API key |
| `MARS_GOOGLE_API_KEY` | For Google provider | Google Gemini API key |
| `MARS_OLLAMA_BASE_URL` | For Ollama provider | Ollama server URL (default: `http://localhost:11434`) |

## Running

```sh
mars debate "Your question here" -p openai -p anthropic
python -m mars debate "Your question here"
```

## Linting and Type Checking

```sh
ruff check src/              # Lint
ruff check src/ --fix         # Auto-fix lint issues
ruff format src/              # Format
ruff format --check src/      # Check formatting
mypy                          # Type check
```

## Project Structure

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed documentation.
