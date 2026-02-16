"""Click CLI for MARS — Multi-Agent Resolution Synthesis."""

from __future__ import annotations

import asyncio
from pathlib import Path

import click

from mars.config import AppConfig, MARS_CONFIG_DIR, MARS_CONFIG_FILE, load_mars_config
from mars.debate.engine import DebateEngine
from mars.models import DebateConfig, DebateMode, Verbosity, provider_base_name
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
  mars show                  # view most recent debate
  mars history               # list past debates
  mars copy                  # copy final answer to clipboard
""")
def main() -> None:
    """MARS — Multi-Agent Resolution Synthesis.

    Multiple LLMs debate a prompt, critique each other's answers,
    and converge on a synthesized best answer. Supports round-robin
    (iterative critique) and judge (single evaluator) modes.
    """
    load_mars_config()


@main.command(epilog="""\b
Examples:
  mars debate "Compare React vs Vue" -p openai -p anthropic
  mars debate @prompt.txt -c @data.csv -p openai -p google
  mars debate "Best algo?" -p openai -p google -m judge -j anthropic
  mars debate "Explain monads" -p vertex:claude-opus-4-6 -p vertex:gemini-2.5-flash -v
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

    # Parse -p flags: support "openai", "openai:gpt-4.1", "vertex:claude-sonnet-4"
    # When ":" is used, the full string becomes the participant ID
    for p in provider:
        if ":" in p:
            prov, mod = p.split(":", 1)
            providers.append(p)  # Full string as participant ID
            model_overrides[p] = mod  # Keyed by participant ID
        else:
            providers.append(p)

    if not providers:
        providers = AppConfig().get_default_providers()

    # Validate base provider names
    for p in providers:
        base = provider_base_name(p)
        if base not in AVAILABLE_PROVIDERS:
            raise click.BadParameter(
                f"Unknown provider '{base}'. Available: {', '.join(AVAILABLE_PROVIDERS)}"
            )

    # Explicit --model overrides — match by base name against participant IDs
    for m in model:
        if ":" not in m:
            raise click.BadParameter(f"Invalid --model format '{m}'. Expected provider:model.")
        prov, mod = m.split(":", 1)
        matched = False
        for pid in providers:
            if provider_base_name(pid) == prov:
                model_overrides[pid] = mod
                matched = True
        if not matched:
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


def _mask_key(key: str) -> str:
    """Show first 4 and last 4 chars, mask the rest."""
    if len(key) <= 8:
        return "****"
    return key[:4] + "..." + key[-4:]


def _read_existing_config() -> dict[str, str]:
    """Read existing key=value pairs from ~/.mars/config."""
    config: dict[str, str] = {}
    if MARS_CONFIG_FILE.is_file():
        for line in MARS_CONFIG_FILE.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, v = line.split("=", 1)
                config[k.strip()] = v.strip()
    return config


def _write_config(values: dict[str, str]) -> None:
    """Write key=value pairs to ~/.mars/config in dotenv format."""
    MARS_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    lines = ["# MARS configuration — generated by `mars configure`\n"]
    for key, value in values.items():
        lines.append(f"{key}={value}\n")
    MARS_CONFIG_FILE.write_text("".join(lines))


def _validate_openai_key(api_key: str) -> tuple[bool, str]:
    """Validate OpenAI key by listing models."""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        client.models.list()
        return True, ""
    except Exception as e:
        return False, str(e)


def _validate_anthropic_key(api_key: str) -> tuple[bool, str]:
    """Validate Anthropic key with a minimal API call."""
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1,
            messages=[{"role": "user", "content": "hi"}],
        )
        return True, ""
    except Exception as e:
        if "auth" in str(e).lower() or "401" in str(e):
            return False, str(e)
        # Non-auth errors (rate limit, etc.) mean the key is valid
        return True, ""


def _validate_google_key(api_key: str) -> tuple[bool, str]:
    """Validate Google key by listing models."""
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        list(client.models.list())
        return True, ""
    except Exception as e:
        msg = str(e).lower()
        if "api key" in msg or "invalid" in msg or "401" in msg or "403" in msg:
            return False, str(e)
        return True, ""


def _validate_ollama_url(base_url: str) -> tuple[bool, str]:
    """Validate Ollama by hitting the /api/tags endpoint."""
    try:
        import httpx
        resp = httpx.get(f"{base_url.rstrip('/')}/api/tags", timeout=5.0)
        resp.raise_for_status()
        return True, ""
    except Exception as e:
        return False, str(e)


def _gcloud_installed() -> bool:
    """Check if the gcloud CLI is available."""
    import shutil
    return shutil.which("gcloud") is not None


def _validate_vertex(project_id: str) -> tuple[bool, str]:
    """Validate Vertex AI access via Application Default Credentials."""
    if not _gcloud_installed():
        return False, (
            "gcloud CLI not found. Install it from https://cloud.google.com/sdk/docs/install "
            "then run: gcloud auth application-default login"
        )
    try:
        import google.auth
        credentials, project = google.auth.default()
        return True, ""
    except Exception as e:
        return False, (
            "Application Default Credentials not found. Run:\n"
            "    gcloud auth application-default login"
        )


_VALIDATORS: dict[str, object] = {
    "openai": _validate_openai_key,
    "anthropic": _validate_anthropic_key,
    "google": _validate_google_key,
    "ollama": _validate_ollama_url,
    "vertex": _validate_vertex,
}

_PROVIDER_CONFIG = [
    {
        "name": "vertex",
        "display": "Vertex AI (Google Cloud)",
        "hint": "uses ADC — run: gcloud auth application-default login",
        "is_vertex": True,
    },
    {
        "name": "openai",
        "env_var": "MARS_OPENAI_API_KEY",
        "display": "OpenAI",
        "hint": "from https://platform.openai.com/api-keys",
        "is_url": False,
    },
    {
        "name": "anthropic",
        "env_var": "MARS_ANTHROPIC_API_KEY",
        "display": "Anthropic",
        "hint": "from https://console.anthropic.com/settings/keys",
        "is_url": False,
    },
    {
        "name": "google",
        "env_var": "MARS_GOOGLE_API_KEY",
        "display": "Google AI",
        "hint": "from https://aistudio.google.com/apikey",
        "is_url": False,
    },
    {
        "name": "ollama",
        "env_var": "MARS_OLLAMA_BASE_URL",
        "display": "Ollama (local)",
        "hint": "default: http://localhost:11434",
        "is_url": True,
    },
]


def _install_claude_command() -> None:
    """Install the MARS slash command into ~/.claude/commands/."""
    commands_dir = Path.home() / ".claude" / "commands" / "mars"
    commands_dir.mkdir(parents=True, exist_ok=True)
    debate_cmd = commands_dir / "debate.md"
    debate_cmd.write_text(_DEBATE_COMMAND, encoding="utf-8")
    click.echo(f"  Installed /mars:debate -> {debate_cmd}")


@main.command()
def configure() -> None:
    """Set up MARS API keys and integrations.

    Interactive setup: configure API keys for LLM providers,
    validate them, and optionally install the Claude Code slash command.
    Keys are stored in ~/.mars/config.
    """
    click.echo()
    click.echo("Welcome to MARS — Multi-Agent Resolution Synthesis")
    click.echo("=" * 52)
    click.echo()
    click.echo("This will configure your LLM provider API keys.")
    click.echo(f"Keys are stored in {MARS_CONFIG_FILE}")
    click.echo("Press Enter to skip any provider you don't need.")
    click.echo()

    existing = _read_existing_config()
    new_config: dict[str, str] = {}
    configured: list[str] = []
    skipped: list[str] = []

    for prov in _PROVIDER_CONFIG:
        is_vertex = prov.get("is_vertex", False)
        is_url = prov.get("is_url", False)

        if is_vertex:
            # Vertex AI: two fields (project ID + region)
            current_project = existing.get("MARS_VERTEX_PROJECT_ID", "")
            current_region = existing.get("MARS_VERTEX_REGION", "")
            project_source = ""

            # Auto-detect from common env vars
            import os
            if not current_project:
                for env_var_name in ("ANTHROPIC_VERTEX_PROJECT_ID", "GOOGLE_CLOUD_PROJECT"):
                    val = os.environ.get(env_var_name, "")
                    if val:
                        current_project = val
                        project_source = env_var_name
                        break
            if not current_region:
                current_region = os.environ.get("CLOUD_ML_REGION", "us-central1")

            # Show current status
            if current_project:
                source_note = f" (from ${project_source})" if project_source else ""
                click.echo(f"  {prov['display']}: project={current_project}{source_note}, region={current_region}")
            else:
                click.echo(f"  {prov['display']}: not configured")

            # Check gcloud and ADC status
            if not _gcloud_installed():
                click.echo("  gcloud CLI not found — install from https://cloud.google.com/sdk/docs/install")
            else:
                try:
                    import google.auth
                    credentials, _ = google.auth.default()
                    click.echo("  ADC credentials: found")
                except Exception:
                    click.echo("  ADC credentials: not found — run: gcloud auth application-default login")

            project_id = click.prompt(
                "  GCP Project ID",
                default=current_project or "",
                show_default=False,
                prompt_suffix=": ",
            )

            if not project_id:
                if current_project:
                    new_config["MARS_VERTEX_PROJECT_ID"] = current_project
                    new_config["MARS_VERTEX_REGION"] = current_region
                    click.echo("  Kept existing config.")
                    configured.append(prov["display"])
                else:
                    click.echo("  Skipped.")
                    skipped.append(prov["display"])
                click.echo()
                continue

            region = click.prompt(
                "  GCP Region",
                default=current_region or "us-central1",
            )

            # Validate ADC
            click.echo("  Validating... ", nl=False)
            validate_fn = _VALIDATORS["vertex"]
            ok, err = validate_fn(project_id)
            if ok:
                click.echo("valid!")
                new_config["MARS_VERTEX_PROJECT_ID"] = project_id
                new_config["MARS_VERTEX_REGION"] = region
                configured.append(prov["display"])
            else:
                click.echo(f"failed: {err}")
                if click.confirm("  Save anyway?", default=False):
                    new_config["MARS_VERTEX_PROJECT_ID"] = project_id
                    new_config["MARS_VERTEX_REGION"] = region
                    configured.append(prov["display"])
                elif current_project:
                    new_config["MARS_VERTEX_PROJECT_ID"] = current_project
                    new_config["MARS_VERTEX_REGION"] = current_region
                    click.echo("  Kept existing config.")
                    configured.append(prov["display"])
                else:
                    skipped.append(prov["display"])
            click.echo()
            continue

        env_var = prov["env_var"]
        current = existing.get(env_var, "")

        # Show current status
        if current:
            if is_url:
                click.echo(f"  {prov['display']}: currently {current}")
            else:
                click.echo(f"  {prov['display']}: currently {_mask_key(current)}")
        else:
            click.echo(f"  {prov['display']}: not configured")

        # Prompt for value
        if is_url:
            value = click.prompt(
                f"  {prov['display']} URL ({prov['hint']})",
                default=current or "http://localhost:11434",
                show_default=False,
            )
        else:
            value = click.prompt(
                f"  {prov['display']} API key ({prov['hint']})",
                default="",
                hide_input=True,
                show_default=False,
                prompt_suffix=": ",
            )

        # Handle skip
        if not value and not is_url:
            if current:
                new_config[env_var] = current
                click.echo("  Kept existing config.")
                configured.append(prov["display"])
            else:
                click.echo("  Skipped.")
                skipped.append(prov["display"])
            click.echo()
            continue

        # Validate
        click.echo("  Validating... ", nl=False)
        validate_fn = _VALIDATORS[prov["name"]]
        ok, err = validate_fn(value)
        if ok:
            click.echo("valid!")
            new_config[env_var] = value
            configured.append(prov["display"])
        else:
            click.echo(f"failed: {err}")
            if click.confirm("  Save anyway?", default=False):
                new_config[env_var] = value
                configured.append(prov["display"])
            elif current:
                new_config[env_var] = current
                click.echo("  Kept existing config.")
                configured.append(prov["display"])
            else:
                skipped.append(prov["display"])
        click.echo()

    # Default providers
    current_defaults = existing.get("MARS_DEFAULT_PROVIDERS", "")
    if configured:
        click.echo("Default providers for debates (comma-separated, e.g. openai,anthropic):")
        if current_defaults:
            click.echo(f"  Currently: {current_defaults}")
        defaults_input = click.prompt(
            "  Default providers",
            default=current_defaults or "",
            show_default=False,
            prompt_suffix=": ",
        )
        if defaults_input:
            new_config["MARS_DEFAULT_PROVIDERS"] = defaults_input
        elif current_defaults:
            new_config["MARS_DEFAULT_PROVIDERS"] = current_defaults
        click.echo()

    # Write config
    if new_config:
        _write_config(new_config)
        click.echo(f"Saved to {MARS_CONFIG_FILE}")
        click.echo()

    # Claude Code integration
    claude_dir = Path.home() / ".claude"
    if claude_dir.is_dir():
        if click.confirm(
            "Set up Claude Code integration? (installs /mars:debate slash command)",
            default=True,
        ):
            _install_claude_command()
            click.echo()

    # Summary
    click.echo("Setup complete!")
    click.echo()
    if configured:
        click.echo(f"  Configured: {', '.join(configured)}")
    if skipped:
        click.echo(f"  Skipped:    {', '.join(skipped)}")
    click.echo()
    click.echo("Next steps:")
    click.echo('  mars providers          — verify provider status')
    click.echo('  mars debate "Question"  — start a debate')
    click.echo()


@main.command("providers")
def list_providers() -> None:
    """List available LLM providers and their configuration status.

    Shows each provider's default model and whether an API key is
    configured (via ~/.mars/config, .env, or MARS_*_API_KEY env vars).
    """
    app_config = AppConfig()
    for name in AVAILABLE_PROVIDERS:
        key = app_config.get_api_key(name)
        status = "configured" if key else "not configured"
        if name == "ollama":
            status = f"url: {app_config.ollama_base_url}"
        elif name == "vertex":
            if app_config.vertex_project_id:
                status = f"project: {app_config.vertex_project_id}, region: {app_config.vertex_region}"
            else:
                status = "not configured"
        model = app_config.get_default_model(name)
        click.echo(f"  {name:15s}  model: {model:30s}  ({status})")


# ---------------------------------------------------------------------------
# Post-debate commands: show, history, copy
# ---------------------------------------------------------------------------

from mars.output.reader import (
    find_debates,
    resolve_debate,
    read_file,
    extract_timestamp,
    extract_prompt_from_dirname,
    parse_providers,
    count_rounds,
    parse_costs_total,
)


_DEBATE_OPTION = click.option(
    "--debate", default=None, help="Path to a specific debate directory.",
)
_OUTPUT_DIR_OPTION = click.option(
    "-o", "--output-dir", default="./mars-output", show_default=True,
    help="Output directory.",
)


@main.group(invoke_without_command=True, epilog="""\b
Examples:
  mars show                  # summary of most recent debate
  mars show answer           # just the final answer
  mars show costs            # cost breakdown
  mars show attribution      # provider contributions
  mars show rounds           # round-by-round responses
  mars show --debate ./mars-output/2026-02-16T08-06-45_my-topic
""")
@_DEBATE_OPTION
@_OUTPUT_DIR_OPTION
@click.pass_context
def show(ctx: click.Context, debate: str | None, output_dir: str) -> None:
    """View results of a completed debate.

    With no subcommand, shows a compact summary including the prompt,
    providers, cost, attribution highlights, and the final answer.

    Use a subcommand (answer, costs, attribution, rounds) to view
    a specific section.
    """
    ctx.ensure_object(dict)
    ctx.obj["debate_dir"] = resolve_debate(debate, output_dir)
    if ctx.invoked_subcommand is None:
        _show_summary(ctx.obj["debate_dir"])


def _show_summary(debate_dir: Path) -> None:
    """Render a compact summary of the debate."""
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel

    console = Console()
    dirname = debate_dir.name
    ts = extract_timestamp(dirname)
    prompt_slug = extract_prompt_from_dirname(dirname)
    providers = parse_providers(debate_dir)
    rounds = count_rounds(debate_dir)

    # Header
    console.print(Panel(
        f"[bold]{prompt_slug}[/bold]\n\n"
        f"Timestamp:  {ts}\n"
        f"Providers:  {', '.join(providers) or 'unknown'}\n"
        f"Rounds:     {rounds}",
        title="MARS Debate Summary",
    ))

    # Costs
    costs_content = read_file(debate_dir, "audit/costs.md")
    if costs_content:
        total = parse_costs_total(costs_content)
        console.print(f"\n[bold]Cost:[/bold] {total}")

    # Attribution highlights
    attr_content = read_file(debate_dir, "audit/attribution.md")
    if attr_content:
        console.print(Panel(Markdown(attr_content), title="Attribution"))

    # Final answer
    answer = read_file(debate_dir, "final-answer.md")
    if answer:
        console.print(Panel(Markdown(answer), title="Final Answer"))
    else:
        console.print("\n[yellow]No final answer yet (debate may be incomplete).[/yellow]")


@show.command("answer")
@click.pass_context
def show_answer(ctx: click.Context) -> None:
    """Show only the final synthesized answer."""
    from rich.console import Console
    from rich.markdown import Markdown

    debate_dir = ctx.obj["debate_dir"]
    console = Console()
    answer = read_file(debate_dir, "final-answer.md")
    if answer:
        console.print(Markdown(answer))
    else:
        raise click.ClickException("No final answer found (debate may be incomplete).")


@show.command("costs")
@click.pass_context
def show_costs(ctx: click.Context) -> None:
    """Show token usage and cost breakdown."""
    from rich.console import Console
    from rich.markdown import Markdown

    debate_dir = ctx.obj["debate_dir"]
    console = Console()
    content = read_file(debate_dir, "audit/costs.md")
    if content:
        console.print(Markdown(content))
    else:
        raise click.ClickException("No cost data found.")


@show.command("attribution")
@click.pass_context
def show_attribution(ctx: click.Context) -> None:
    """Show per-provider contribution and influence metrics."""
    from rich.console import Console
    from rich.markdown import Markdown

    debate_dir = ctx.obj["debate_dir"]
    console = Console()
    content = read_file(debate_dir, "audit/attribution.md")
    if content:
        console.print(Markdown(content))
    else:
        raise click.ClickException("No attribution data found.")


@show.command("rounds")
@click.pass_context
def show_rounds(ctx: click.Context) -> None:
    """Show round-by-round responses and diffs."""
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel

    debate_dir = ctx.obj["debate_dir"]
    console = Console()
    audit = debate_dir / "audit"
    if not audit.is_dir():
        raise click.ClickException("No audit directory found.")

    round_files = sorted(
        f for f in audit.iterdir()
        if f.name.startswith(("0", "1")) and "round" in f.name
    )
    if not round_files:
        raise click.ClickException("No round files found.")

    for rf in round_files:
        content = rf.read_text(encoding="utf-8")
        label = rf.stem.split("-", 1)[1] if "-" in rf.stem else rf.stem
        console.print(Panel(Markdown(content), title=label))

    # Show diffs if available
    diffs = read_file(debate_dir, "audit/round-diffs.md")
    if diffs:
        console.print(Panel(Markdown(diffs), title="Round Diffs"))


@main.command(epilog="""\b
Examples:
  mars history        # list all past debates
  mars history -n 5   # last 5 debates
""")
@click.option("-n", "--limit", type=int, default=None, help="Show only the last N debates.")
@_OUTPUT_DIR_OPTION
def history(limit: int | None, output_dir: str) -> None:
    """List past debates with timestamps, providers, and costs.

    Shows all debates found in the output directory, most recent first.
    """
    from rich.console import Console
    from rich.table import Table

    debates = find_debates(output_dir)
    if not debates:
        raise click.ClickException(f"No debates found in {output_dir}")

    if limit is not None:
        debates = debates[:limit]

    console = Console()
    table = Table(title="MARS Debate History")
    table.add_column("#", style="dim", width=4)
    table.add_column("Timestamp")
    table.add_column("Prompt", max_width=40)
    table.add_column("Providers")
    table.add_column("Rounds", justify="right")
    table.add_column("Cost", justify="right")

    for i, d in enumerate(debates, 1):
        ts = extract_timestamp(d.name)
        prompt_slug = extract_prompt_from_dirname(d.name)
        if len(prompt_slug) > 40:
            prompt_slug = prompt_slug[:37] + "..."
        providers = ", ".join(parse_providers(d))
        rounds = str(count_rounds(d))
        costs_content = read_file(d, "audit/costs.md")
        cost = parse_costs_total(costs_content) if costs_content else "n/a"
        table.add_row(str(i), ts, prompt_slug, providers, rounds, cost)

    console.print(table)


def _copy_to_clipboard(text: str) -> bool:
    """Copy text to system clipboard. Returns True on success."""
    import platform
    import subprocess

    system = platform.system()
    try:
        if system == "Darwin":
            subprocess.run(["pbcopy"], input=text.encode(), check=True)
        elif system == "Linux":
            try:
                subprocess.run(["xclip", "-selection", "clipboard"],
                               input=text.encode(), check=True)
            except FileNotFoundError:
                subprocess.run(["xsel", "--clipboard", "--input"],
                               input=text.encode(), check=True)
        elif system == "Windows":
            subprocess.run(["clip"], input=text.encode(), check=True)
        else:
            return False
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        return False


@main.command(epilog="""\b
Examples:
  mars copy              # copy final answer to clipboard
  mars copy --full       # copy prompt + answer + attribution
""")
@click.option("--full", is_flag=True, help="Include prompt, answer, and attribution.")
@_DEBATE_OPTION
@_OUTPUT_DIR_OPTION
def copy(full: bool, debate: str | None, output_dir: str) -> None:
    """Copy the final answer to the clipboard.

    By default copies just the final answer. Use --full to include
    the original prompt and attribution analysis.
    """
    debate_dir = resolve_debate(debate, output_dir)
    answer = read_file(debate_dir, "final-answer.md")
    if not answer:
        raise click.ClickException("No final answer found (debate may be incomplete).")

    if full:
        parts = []
        prompt_content = read_file(debate_dir, "audit/00-prompt-and-context.md")
        if prompt_content:
            parts.append(prompt_content)
        parts.append(answer)
        attr_content = read_file(debate_dir, "audit/attribution.md")
        if attr_content:
            parts.append(attr_content)
        text = "\n\n---\n\n".join(parts)
    else:
        text = answer

    if _copy_to_clipboard(text):
        click.echo("Copied to clipboard.")
    else:
        click.echo("Clipboard not available. Output printed below:\n")
        click.echo(text)
