"""Click CLI for Multi-Agent Resolution."""

from __future__ import annotations

import asyncio
from pathlib import Path

import click
from dotenv import load_dotenv

from mar.config import AppConfig
from mar.debate.engine import DebateEngine
from mar.models import DebateConfig, DebateMode, Verbosity
from mar.providers.registry import AVAILABLE_PROVIDERS


def _resolve_value(value: str) -> str:
    """If value starts with @, read the file; otherwise return as-is."""
    if value.startswith("@"):
        path = Path(value[1:])
        if not path.is_file():
            raise click.BadParameter(f"File not found: {path}")
        return path.read_text(encoding="utf-8").strip()
    return value


@click.group()
def main() -> None:
    """MAR - Multi-Agent Resolution: LLMs debate to find the best answer."""
    load_dotenv()


@main.command()
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
    help=f"Provider or provider:model (e.g. openai:gpt-4.1). Available: {', '.join(AVAILABLE_PROVIDERS)}",
)
@click.option(
    "-m",
    "--mode",
    type=click.Choice(["round-robin", "judge"]),
    default="round-robin",
    help="Debate mode.",
)
@click.option("-r", "--rounds", type=int, default=3, help="Max debate rounds.")
@click.option("-j", "--judge-provider", help="Provider to act as judge (judge mode).")
@click.option("-s", "--synthesis-provider", help="Provider for final synthesis (default: auto).")
@click.option(
    "--model",
    multiple=True,
    help="Provider:model override (e.g. openai:gpt-4o-mini). Repeatable.",
)
@click.option("--threshold", type=float, default=0.85, help="Convergence similarity threshold (0.0-1.0).")
@click.option("--max-tokens", type=int, default=8192, help="Max output tokens per LLM call.")
@click.option("-t", "--temperature", type=float, default=None, help="Temperature (0.0-2.0). Default: provider default.")
@click.option("-v", "--verbose", is_flag=True, help="Stream responses in real-time.")
@click.option(
    "-o", "--output-dir", default="./mar-output", help="Output directory."
)
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

    PROMPT can be plain text or @file to read from a file.
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
            raise click.BadParameter(
                f"Invalid --model format '{m}'. Expected provider:model."
            )
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


@main.command("providers")
def list_providers() -> None:
    """List available LLM providers."""
    app_config = AppConfig()
    for name in AVAILABLE_PROVIDERS:
        key = app_config.get_api_key(name)
        status = "configured" if key else "not configured"
        if name == "ollama":
            status = f"url: {app_config.ollama_base_url}"
        model = app_config.get_default_model(name)
        click.echo(f"  {name:12s}  model: {model:30s}  ({status})")
