"""Read and query debate output directories."""

from __future__ import annotations

import re
from pathlib import Path

import click

_TIMESTAMP_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}_")
_ROUND_FILE = re.compile(r"^\d{2}-round-\d+-")


def find_debates(output_dir: str = "./mars-output") -> list[Path]:
    """Return debate directories sorted by timestamp descending (most recent first)."""
    base = Path(output_dir)
    if not base.is_dir():
        return []
    dirs = [d for d in base.iterdir() if d.is_dir() and _TIMESTAMP_PATTERN.match(d.name)]
    return sorted(dirs, key=lambda d: d.name, reverse=True)


def resolve_debate(debate: str | None, output_dir: str = "./mars-output") -> Path:
    """Resolve which debate directory to use.

    If *debate* is given, treat it as a path. Otherwise return the most recent.
    """
    if debate:
        p = Path(debate)
        if not p.is_dir():
            raise click.ClickException(f"Debate directory not found: {debate}")
        return p
    debates = find_debates(output_dir)
    if not debates:
        raise click.ClickException(f"No debates found in {output_dir}")
    return debates[0]


def read_file(debate_dir: Path, filename: str) -> str | None:
    """Read a file from the debate directory, returning None if missing."""
    path = debate_dir / filename
    if not path.is_file():
        return None
    return path.read_text(encoding="utf-8")


def extract_timestamp(dirname: str) -> str:
    """Extract a human-readable timestamp from a debate directory name."""
    ts = dirname.split("_", 1)[0]
    return ts.replace("T", " ")


def extract_prompt_from_dirname(dirname: str) -> str:
    """Extract the prompt slug from a debate directory name."""
    parts = dirname.split("_", 1)
    if len(parts) < 2:
        return dirname
    return parts[1].replace("-", " ")


def parse_providers(debate_dir: Path) -> list[str]:
    """Extract provider names from round-1 response headers."""
    content = read_file(debate_dir, "audit/01-round-1-responses.md")
    if not content:
        return []
    providers: list[str] = []
    for line in content.splitlines():
        if line.startswith("## ") and "(" in line:
            provider = line[3:].split("(")[0].strip()
            if provider and provider not in providers:
                providers.append(provider)
    return providers


def count_rounds(debate_dir: Path) -> int:
    """Count the number of round files in the audit directory."""
    audit = debate_dir / "audit"
    if not audit.is_dir():
        return 0
    return len([f for f in audit.iterdir() if _ROUND_FILE.match(f.name)])


def parse_costs_total(content: str) -> str:
    """Extract the total cost string from costs.md content."""
    for line in content.splitlines():
        if line.startswith("**Total**") and "$" in line:
            return "$" + line.split("$", 1)[1].strip()
    return "n/a"
