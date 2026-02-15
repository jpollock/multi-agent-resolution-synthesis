"""Markdown file output writer."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from slugify import slugify

from mars.models import AttributionReport, CostReport, Critique, LLMResponse, RoundDiff


class OutputWriter:
    def __init__(self, output_dir: str, prompt: str) -> None:
        slug = slugify(prompt[:60])
        timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        self._base = Path(output_dir) / f"{timestamp}_{slug}"
        self._audit = self._base / "audit"
        self._audit.mkdir(parents=True, exist_ok=True)

    @property
    def base_path(self) -> str:
        return str(self._base)

    def write_prompt(self, prompt: str, context: list[str]) -> None:
        lines = [f"# Prompt\n\n{prompt}\n"]
        if context:
            lines.append("\n# Context\n")
            for i, ctx in enumerate(context, 1):
                lines.append(f"\n## Context {i}\n\n{ctx}\n")
        self._write(self._audit / "00-prompt-and-context.md", "\n".join(lines))

    def write_round(
        self,
        round_num: int,
        responses: list[LLMResponse],
        critiques: list[Critique] | None = None,
    ) -> None:
        parts: list[str] = []
        if critiques:
            parts.append(f"# Round {round_num} - Critiques & Improved Answers\n")
            for c in critiques:
                parts.append(f"\n## {c.author} critiques {c.target}\n\n{c.content}\n")
            parts.append("\n---\n\n# Improved Answers\n")
        else:
            parts.append(f"# Round {round_num} - Initial Responses\n")

        for r in responses:
            parts.append(f"\n## {r.provider} ({r.model})\n\n{r.content}\n")

        idx = str(round_num).zfill(2)
        label = "critiques" if critiques else "responses"
        self._write(
            self._audit / f"{idx}-round-{round_num}-{label}.md",
            "\n".join(parts),
        )

    def write_convergence(self, reason: str) -> None:
        self._write(
            self._audit / "convergence.md",
            f"# Convergence\n\n{reason}\n",
        )

    def write_resolution(self, reasoning: str) -> None:
        self._write(
            self._audit / "resolution.md",
            f"# Resolution\n\n{reasoning}\n",
        )

    def write_final(self, answer: str) -> None:
        self._write(self._base / "final-answer.md", answer)

    def write_attribution(self, report: AttributionReport) -> None:
        lines = ["# Attribution Analysis\n"]
        lines.append(
            f"Similarity threshold: {report.similarity_threshold}  \n"
            f"Final answer sentences: {report.sentence_count_final}\n"
        )
        lines.append("\n## Summary\n")
        lines.append("| Provider | Model | Contribution | Survival | Influence |")
        lines.append("|----------|-------|-------------|----------|-----------|")
        for pa in report.providers:
            lines.append(
                f"| {pa.provider} | {pa.model} "
                f"| {pa.contribution_pct:.1f}%"
                f" ({pa.contributed_sentences}/{pa.total_final_sentences}) "
                f"| {pa.survival_rate:.1f}%"
                f" ({pa.survived_sentences}/{pa.initial_sentences}) "
                f"| {pa.influence_score:.1f}% |"
            )
        if report.novel_sentences > 0:
            lines.append(
                f"| *Synthesizer (novel)* | - "
                f"| {report.novel_pct:.1f}%"
                f" ({report.novel_sentences}/{report.sentence_count_final}) "
                f"| - | - |"
            )
        lines.append("\n## Metric Definitions\n")
        lines.append(
            "- **Contribution**: percentage of final answer sentences whose "
            "best match (above threshold) traces to this provider."
        )
        lines.append(
            "- **Survival rate**: percentage of this provider's round-1 "
            "sentences that appear (above threshold) in the final answer."
        )
        lines.append(
            "- **Influence**: average rate at which other providers adopted "
            "this provider's sentences in subsequent rounds."
        )
        for pa in report.providers:
            if pa.influence_details:
                lines.append(f"\n### {pa.provider} Influence Breakdown\n")
                for target, rate in pa.influence_details.items():
                    lines.append(f"- Adopted by **{target}**: {rate:.1f}%")
        self._write(self._audit / "attribution.md", "\n".join(lines))

    def write_costs(self, report: CostReport) -> None:
        lines = ["# Cost Summary\n"]
        lines.append("| Provider | Model | Input Tokens | Output Tokens | Cost | Share |")
        lines.append("|----------|-------|-------------|--------------|------|-------|")
        for pc in report.providers:
            lines.append(
                f"| {pc.provider} | {pc.model} "
                f"| {pc.input_tokens:,} | {pc.output_tokens:,} "
                f"| ${pc.total_cost:.4f} | {pc.share_of_total:.1f}% |"
            )
        lines.append(
            f"\n**Total**: {report.total_input_tokens + report.total_output_tokens:,} "
            f"tokens | ${report.total_cost:.4f}"
        )
        self._write(self._audit / "costs.md", "\n".join(lines))

    def write_round_diffs(self, diffs: list[RoundDiff]) -> None:
        if not diffs:
            return
        lines = ["# Round-over-Round Changes\n"]
        lines.append("| Provider | Rounds | Similarity | Added | Removed | Unchanged |")
        lines.append("|----------|--------|-----------|-------|---------|-----------|")
        for d in diffs:
            lines.append(
                f"| {d.provider} | {d.from_round}->{d.to_round} "
                f"| {d.similarity:.1%} | +{d.sentences_added} "
                f"| -{d.sentences_removed} | {d.sentences_unchanged} |"
            )
        self._write(self._audit / "round-diffs.md", "\n".join(lines))

    @staticmethod
    def _write(path: Path, content: str) -> None:
        path.write_text(content, encoding="utf-8")
