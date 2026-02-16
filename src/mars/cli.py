"""Click CLI for MARS — Multi-Agent Resolution Synthesis."""

from __future__ import annotations

import asyncio
from pathlib import Path

import click
from dotenv import load_dotenv

from mars.config import AppConfig
from mars.debate.engine import DebateEngine
from mars.models import DebateConfig, DebateMode, Verbosity
from mars.providers.registry import AVAILABLE_PROVIDERS


def _resolve_value(value: str) -> str:
    """If value starts with @, read the file; otherwise return as-is."""
    if value.startswith("@"):
        path = Path(value[1:])
        if not path.is_file():
            raise click.BadParameter(f"File not found: {path}")
        return path.read_text(encoding="utf-8").strip()
    return value


@click.group(epilog="""\b
Examples:
  mars debate "Is Python better than Rust?" -p openai -p anthropic
  mars debate @question.txt -c @data.csv -p openai -p google -v
  mars debate "Explain X" -m judge -j anthropic
  mars providers
""")
def main() -> None:
    """MARS — Multi-Agent Resolution Synthesis.

    Multiple LLMs debate a prompt, critique each other's answers,
    and converge on a synthesized best answer. Supports round-robin
    (iterative critique) and judge (single evaluator) modes.
    """
    load_dotenv()


@main.command(epilog="""\b
Examples:
  mars debate "Compare React vs Vue" -p openai -p anthropic
  mars debate @prompt.txt -c @data.csv -p openai -p google
  mars debate "Best algo?" -p openai -p google -m judge -j anthropic
  mars debate "Explain monads" -p openai:gpt-4.1 -p anthropic:claude-sonnet-4-20250514 -v
  mars debate "Topic" -p openai -p anthropic -r 5 --threshold 0.9
""")
@click.argument("prompt")
@click.option(
    "-c",
    "--context",
    multiple=True,
    help="Context text or @file path (repeatable).",
)
@click.option(
    "-p",
    "--provider",
    multiple=True,
    help=(
        "Provider or provider:model (e.g. openai:gpt-4.1). "
        f"Available: {', '.join(AVAILABLE_PROVIDERS)}"
    ),
)
@click.option(
    "-m",
    "--mode",
    type=click.Choice(["round-robin", "judge"]),
    default="round-robin",
    show_default=True,
    help="Debate mode.",
)
@click.option("-r", "--rounds", type=int, default=3, show_default=True, help="Max debate rounds.")
@click.option("-j", "--judge-provider", help="Provider to act as judge (judge mode).")
@click.option("-s", "--synthesis-provider", help="Provider for final synthesis (default: auto).")
@click.option(
    "--model",
    multiple=True,
    help="Provider:model override (e.g. openai:gpt-4o-mini). Repeatable.",
)
@click.option(
    "--threshold", type=float, default=0.85, show_default=True,
    help="Convergence similarity threshold (0.0-1.0).",
)
@click.option("--max-tokens", type=int, default=8192, show_default=True, help="Max output tokens per LLM call.")
@click.option(
    "-t",
    "--temperature",
    type=float,
    default=None,
    help="Temperature (0.0-2.0). Default: provider default.",
)
@click.option("-v", "--verbose", is_flag=True, help="Stream responses in real-time.")
@click.option("-o", "--output-dir", default="./mars-output", show_default=True, help="Output directory.")
def debate(
    prompt: str,
    context: tuple[str, ...],
    provider: tuple[str, ...],
    mode: str,
    rounds: int,
    judge_provider: str | None,
    synthesis_provider: str | None,
    model: tuple[str, ...],
    threshold: float,
    max_tokens: int,
    temperature: float | None,
    verbose: bool,
    output_dir: str,
) -> None:
    """Run a multi-LLM debate on PROMPT.

    PROMPT is the question or task for the LLMs to debate. Use @file
    to read the prompt from a file (e.g. mars debate @question.txt).

    In round-robin mode (default), providers answer independently, then
    critique each other's answers for up to N rounds until convergence.
    A final synthesis merges the best points from all providers.

    In judge mode, all providers answer once, then a designated judge
    provider evaluates and produces the final answer.

    Output is saved to mars-output/ as timestamped Markdown files.
    """
    resolved_prompt = _resolve_value(prompt)
    resolved_context = [_resolve_value(c) for c in context]

    providers: list[str] = []
    model_overrides: dict[str, str] = {}

    # Parse -p flags: support both "openai" and "openai:gpt-4.1"
    for p in provider:
        if ":" in p:
            prov, mod = p.split(":", 1)
            providers.append(prov)
            model_overrides[prov] = mod
        else:
            providers.append(p)

    if not providers:
        providers = ["openai", "anthropic"]

    for p in providers:
        if p not in AVAILABLE_PROVIDERS:
            raise click.BadParameter(
                f"Unknown provider '{p}'. Available: {', '.join(AVAILABLE_PROVIDERS)}"
            )

    # Explicit --model overrides take precedence
    for m in model:
        if ":" not in m:
            raise click.BadParameter(f"Invalid --model format '{m}'. Expected provider:model.")
        prov, mod = m.split(":", 1)
        model_overrides[prov] = mod

    config = DebateConfig(
        prompt=resolved_prompt,
        context=resolved_context,
        providers=providers,
        model_overrides=model_overrides,
        mode=DebateMode(mode),
        max_rounds=rounds,
        max_tokens=max_tokens,
        temperature=temperature,
        judge_provider=judge_provider,
        synthesis_provider=synthesis_provider,
        convergence_threshold=threshold,
        verbosity=Verbosity.VERBOSE if verbose else Verbosity.QUIET,
        output_dir=output_dir,
    )

    app_config = AppConfig()
    engine = DebateEngine(config, app_config)
    asyncio.run(engine.run())


_DEBATE_COMMAND = """\
Run a multi-LLM debate using the MARS CLI.

The user wants to debate: $ARGUMENTS

## Instructions

1. Run `mars providers` to check which providers are configured.
2. Run `mars debate --help` to see available options.
3. Construct and execute the appropriate `mars debate` command based on the
   user's request.
4. If the user didn't specify providers, default to two configured providers
   (e.g. `-p openai -p anthropic`).
5. Use `-v` to stream output so the user can watch the debate in real time.
6. After the debate completes, read the output file and summarize the result.

## Examples

```sh
mars debate "Is Python better than Rust?" -p openai -p anthropic -v
mars debate "Compare SQL vs NoSQL" -p openai -p google -m judge -j anthropic -v
mars debate @prompt.txt -c @context.md -p openai -p anthropic -v
```
"""


@main.command()
def configure() -> None:
    """Set up MARS integration with Claude Code.

    Installs the /mars:debate slash command into ~/.claude/commands/
    so you can use MARS from any Claude Code session.
    """
    commands_dir = Path.home() / ".claude" / "commands" / "mars"
    commands_dir.mkdir(parents=True, exist_ok=True)

    debate_cmd = commands_dir / "debate.md"
    debate_cmd.write_text(_DEBATE_COMMAND, encoding="utf-8")
    click.echo(f"Installed /mars:debate -> {debate_cmd}")
    click.echo()
    click.echo("You can now use /mars:debate in any Claude Code session.")
    click.echo('Example: /mars:debate Should we use Redis or Postgres for caching?')


@main.command("providers")
def list_providers() -> None:
    """List available LLM providers and their configuration status.

    Shows each provider's default model, whether an API key is
    configured (via MARS_*_API_KEY env vars), and the Ollama base URL.
    """
    app_config = AppConfig()
    for name in AVAILABLE_PROVIDERS:
        key = app_config.get_api_key(name)
        status = "configured" if key else "not configured"
        if name == "ollama":
            status = f"url: {app_config.ollama_base_url}"
        model = app_config.get_default_model(name)
        click.echo(f"  {name:12s}  model: {model:30s}  ({status})")
