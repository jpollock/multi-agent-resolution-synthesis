# Changelog

All notable changes to MARS will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

## [0.0.1] - 2026-02-13

### Added

- Multi-agent debate CLI with `mars debate` and `mars providers` commands
- Round-robin and judge debate modes
- OpenAI, Anthropic, Google Gemini, and Ollama provider support
- Configurable max tokens (`--max-tokens`) and temperature (`-t`)
- Inline provider:model syntax (`-p openai:gpt-4.1`)
- Retry with exponential backoff for transient API errors
- Configurable synthesis provider (`-s`) with automatic fallback
- Progress spinner in quiet mode
- Configurable convergence threshold (`--threshold`)
- Sentence-level attribution analysis (contribution, survival, influence)
- Synthesizer novel content tracking in attribution
- Round-over-round diff tracking
- Token usage and cost estimation per provider
- Markdown audit trail output
- Rich terminal output with tables, panels, and streaming
- PyPI publish workflow via GitHub Actions (OIDC trusted publishing)
