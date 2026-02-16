# Roadmap

Where MARS is headed. No timelines — just direction.

## Next

Things actively planned.

- **Persona assignment** — assign reasoning styles per provider (`--persona contrarian,empirical`) to increase argument diversity, especially useful when running multiple instances of the same model family via Vertex
- **Python API** — use MARS as a library, not just a CLI (`from mars import debate`)
- **More providers** — AWS Bedrock, Mistral, Cohere

## Later

Designed but not yet started.

- **Structured output mode** — constrain debate output to JSON or a user-defined schema
- **Debate templates** — presets for common use cases (code review, architecture decisions, trade-off analysis) with tuned prompts and convergence settings
- **Export formats** — HTML report, PDF summary
- **Input validation bounds** — enforce ranges on temperature, max-tokens, and other numeric parameters

## Exploring

Ideas under consideration. May not happen.

- **RAG integration** — ground debates in a knowledge base so providers argue over your data, not just their training
- **Web dashboard** — browse debate history, compare runs, visualize attribution
- **MCP server** — expose MARS as a tool that AI agents can call
- **Agent scaling** — the [ICML 2024 multiagent debate paper](https://composable-models.github.io/llm_debate/) found that more agents and more rounds improve results; experiment with 4-6 provider debates

## Not planned

Things MARS intentionally avoids.

- **Framework dependencies** — no LangChain, LangGraph, or orchestration frameworks. Direct SDK calls give full control over streaming, retries, and error handling without abstraction overhead.
- **Predetermined agent roles** — MARS gets diversity from different model architectures, not from role-playing prompts. Personas (above) are optional seasoning, not a core mechanism.
- **GUI-first design** — MARS is a CLI tool. Any web interface would be a viewer, not the primary interface.
