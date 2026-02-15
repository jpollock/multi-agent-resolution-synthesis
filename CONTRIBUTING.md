# Contributing to MARS

## Getting Started

1. Fork the repository
2. Clone your fork
3. Set up the development environment (see [DEVELOPMENT.md](DEVELOPMENT.md))

## Branch Naming

Use feature branches with descriptive names:

- `feature/short-description` — new features or enhancements
- `fix/short-description` — bug fixes
- `docs/short-description` — documentation changes

## Pull Request Workflow

1. Create a feature branch from `main`:
   ```sh
   git checkout main && git pull origin main
   git checkout -b feature/my-change
   ```
2. Make your changes with clear, focused commits
3. Push your branch and open a PR to `main`
4. Ensure CI passes (ruff lint + mypy type check)
5. Request review

## Code Style

- **Formatter**: ruff format (line length 100)
- **Linter**: ruff check (rules: E, W, F, I, UP, B, SIM, RUF)
- **Type checking**: mypy with `check_untyped_defs = true`
- Run locally before pushing:
  ```sh
  ruff check src/ && ruff format --check src/ && mypy
  ```

## Code Conventions

- Use `from __future__ import annotations` in every module
- Pydantic v2 style: `@field_validator` with `@classmethod`
- Prompt templates go in `src/mars/debate/prompts.py`
- Shared strategy logic goes in `DebateStrategy` base class (`debate/base.py`)
- Provider implementations follow the `LLMProvider` Protocol

## Adding a New Provider

See the "Adding a New Provider" section in [ARCHITECTURE.md](ARCHITECTURE.md).
