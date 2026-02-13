"""Rich terminal output renderer."""

from __future__ import annotations

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.status import Status
from rich.table import Table

from mar.models import AttributionReport, CostReport, DebateResult, Verbosity


class Renderer:
    def __init__(self, verbosity: Verbosity) -> None:
        self.console = Console()
        self.verbose = verbosity == Verbosity.VERBOSE
        self._current_text = ""
        self._status: Status | None = None

    def start_debate(self, prompt: str, providers: list[str], mode: str) -> None:
        table = Table(title="Debate Configuration", show_header=False)
        table.add_column("Key", style="bold cyan")
        table.add_column("Value")
        table.add_row("Prompt", prompt[:120] + ("..." if len(prompt) > 120 else ""))
        table.add_row("Mode", mode)
        table.add_row("Providers", ", ".join(providers))
        self.console.print(table)
        self.console.print()

    def start_round(self, round_number: int) -> None:
        self.console.rule(f"[bold blue]Round {round_number}[/bold blue]")

    def start_provider_stream(self, provider: str) -> None:
        if self.verbose:
            self.console.print(f"\n[bold green]{provider}[/bold green]:")
            self._current_text = ""

    def stream_chunk(self, chunk: str) -> None:
        if self.verbose:
            self.console.print(chunk, end="", highlight=False)
            self._current_text += chunk

    def end_provider_stream(self) -> None:
        if self.verbose:
            self.console.print()

    def start_work(self, providers: list[str], phase: str = "Generating") -> None:
        """Show a spinner in quiet mode."""
        if not self.verbose:
            label = f"[bold blue]{phase}[/bold blue]: {', '.join(providers)}"
            self._status = self.console.status(label, spinner="dots")
            self._status.start()

    def stop_work(self) -> None:
        """Stop the spinner."""
        if self._status:
            self._status.stop()
            self._status = None

    def show_response(self, provider: str, content: str) -> None:
        if not self.verbose:
            self.console.print(
                Panel(
                    Markdown(content),
                    title=f"[bold]{provider}[/bold]",
                    border_style="green",
                )
            )

    def show_convergence(self, reason: str) -> None:
        self.console.print(
            Panel(reason, title="Convergence", border_style="yellow")
        )

    def show_final_answer(self, result: DebateResult) -> None:
        self.console.print()
        self.console.rule("[bold magenta]Final Answer[/bold magenta]")
        self.console.print(
            Panel(
                Markdown(result.final_answer),
                title="Resolved Answer",
                border_style="magenta",
            )
        )

    def show_attribution(self, report: AttributionReport) -> None:
        self.console.print()
        self.console.rule("[bold cyan]Attribution Analysis[/bold cyan]")
        table = Table(show_header=True, header_style="bold")
        table.add_column("Provider", style="bold green")
        table.add_column("Model", style="dim")
        table.add_column("Contribution", justify="right")
        table.add_column("Survival", justify="right")
        table.add_column("Influence", justify="right")

        for pa in report.providers:
            table.add_row(
                pa.provider,
                pa.model,
                f"{pa.contribution_pct:.1f}% ({pa.contributed_sentences}/{pa.total_final_sentences})",
                f"{pa.survival_rate:.1f}% ({pa.survived_sentences}/{pa.initial_sentences})",
                f"{pa.influence_score:.1f}%",
            )

        self.console.print(table)
        self.console.print(
            f"[dim]Similarity threshold: {report.similarity_threshold}  |  "
            f"Final answer sentences: {report.sentence_count_final}[/dim]"
        )

    def show_costs(self, report: CostReport) -> None:
        self.console.print()
        self.console.rule("[bold cyan]Cost Summary[/bold cyan]")
        table = Table(show_header=True, header_style="bold")
        table.add_column("Provider", style="bold green")
        table.add_column("Model", style="dim")
        table.add_column("Input", justify="right")
        table.add_column("Output", justify="right")
        table.add_column("Cost", justify="right")
        table.add_column("Share", justify="right")

        for pc in report.providers:
            table.add_row(
                pc.provider,
                pc.model,
                f"{pc.input_tokens:,}",
                f"{pc.output_tokens:,}",
                f"${pc.total_cost:.4f}",
                f"{pc.share_of_total:.1f}%",
            )

        self.console.print(table)
        self.console.print(
            f"[dim]Total: {report.total_input_tokens + report.total_output_tokens:,} tokens  |  "
            f"${report.total_cost:.4f}[/dim]"
        )

    def show_output_path(self, path: str) -> None:
        self.console.print(f"\n[dim]Output written to: {path}[/dim]")

    def show_error(self, provider: str, error: str) -> None:
        self.console.print(
            f"[bold red]Error from {provider}:[/bold red] {error}"
        )
