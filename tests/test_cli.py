"""Tests for the MARS CLI."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

import mars.cli
import mars.config
from mars.cli import main
from mars.output.reader import (
    count_rounds,
    extract_prompt_from_dirname,
    extract_timestamp,
    find_debates,
    parse_costs_total,
    parse_providers,
)

runner = CliRunner()


@pytest.fixture()
def isolated_home(tmp_path, monkeypatch):
    """Redirect ~/.mars/config and ~/.claude to tmp_path and mock validators."""
    config_dir = tmp_path / ".mars"
    config_file = config_dir / "config"
    monkeypatch.setattr(mars.config, "MARS_CONFIG_DIR", config_dir)
    monkeypatch.setattr(mars.config, "MARS_CONFIG_FILE", config_file)
    monkeypatch.setattr(mars.cli, "MARS_CONFIG_DIR", config_dir)
    monkeypatch.setattr(mars.cli, "MARS_CONFIG_FILE", config_file)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    # Mock validators to avoid real API calls
    def noop(v):
        return (True, "")
    monkeypatch.setitem(mars.cli._VALIDATORS, "openai", noop)
    monkeypatch.setitem(mars.cli._VALIDATORS, "anthropic", noop)
    monkeypatch.setitem(mars.cli._VALIDATORS, "google", noop)
    monkeypatch.setitem(mars.cli._VALIDATORS, "ollama", noop)
    monkeypatch.setitem(mars.cli._VALIDATORS, "vertex", noop)
    # Clear env vars that vertex auto-detects so prompt count is deterministic
    monkeypatch.delenv("ANTHROPIC_VERTEX_PROJECT_ID", raising=False)
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    monkeypatch.delenv("CLOUD_ML_REGION", raising=False)
    return tmp_path


class TestMainHelp:
    def test_shows_description(self):
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "Multi-Agent Resolution Synthesis" in result.output

    def test_shows_examples(self):
        result = runner.invoke(main, ["--help"])
        assert "Examples:" in result.output
        assert "mars debate" in result.output
        assert "mars providers" in result.output

    def test_lists_commands(self):
        result = runner.invoke(main, ["--help"])
        assert "debate" in result.output
        assert "providers" in result.output
        assert "configure" in result.output


class TestDebateHelp:
    def test_shows_description(self):
        result = runner.invoke(main, ["debate", "--help"])
        assert result.exit_code == 0
        assert "Run a multi-LLM debate on PROMPT" in result.output

    def test_shows_mode_explanation(self):
        result = runner.invoke(main, ["debate", "--help"])
        assert "round-robin mode" in result.output
        assert "judge mode" in result.output

    def test_shows_file_input_syntax(self):
        result = runner.invoke(main, ["debate", "--help"])
        assert "@file" in result.output

    def test_shows_defaults(self):
        result = runner.invoke(main, ["debate", "--help"])
        # Normalize whitespace since Click wraps long lines
        normalized = " ".join(result.output.split())
        assert "[default: round-robin]" in normalized
        assert "[default: 3]" in normalized
        assert "[default: 0.85]" in normalized
        assert "[default: 8192]" in normalized
        assert "[default: ./mars-output]" in normalized

    def test_shows_examples(self):
        result = runner.invoke(main, ["debate", "--help"])
        assert "Examples:" in result.output

    def test_shows_output_location(self):
        result = runner.invoke(main, ["debate", "--help"])
        assert "mars-output/" in result.output


class TestProvidersHelp:
    def test_shows_description(self):
        result = runner.invoke(main, ["providers", "--help"])
        assert result.exit_code == 0
        assert "configuration status" in result.output

    def test_mentions_config_sources(self):
        result = runner.invoke(main, ["providers", "--help"])
        assert "~/.mars/config" in result.output


class TestConfigure:
    # Configure prompts: Vertex (project ID) -> OpenAI -> Anthropic -> Google ->
    # Ollama -> Default providers -> (optional) Claude Code.
    # Ollama always gets configured (has default URL), so default providers
    # prompt always appears. Skipping everything = 6 newlines.

    def test_welcome_message(self, isolated_home):
        result = runner.invoke(main, ["configure"], input="\n\n\n\n\n\n")
        assert result.exit_code == 0
        assert "Welcome to MARS" in result.output
        assert "Setup complete" in result.output

    def test_skip_all_providers(self, isolated_home):
        result = runner.invoke(main, ["configure"], input="\n\n\n\n\n\n")
        assert result.exit_code == 0
        assert "Skipped" in result.output

    def test_saves_api_key(self, isolated_home):
        # Skip vertex, provide OpenAI key, skip anthropic/google, accept ollama, skip defaults
        result = runner.invoke(
            main, ["configure"],
            input="\nsk-test1234567890abcdef\n\n\n\n\n",
        )
        assert result.exit_code == 0
        config_file = isolated_home / ".mars" / "config"
        assert config_file.exists()
        content = config_file.read_text()
        assert "MARS_OPENAI_API_KEY=sk-test1234567890abcdef" in content

    def test_masks_existing_key(self, isolated_home):
        config_dir = isolated_home / ".mars"
        config_dir.mkdir()
        (config_dir / "config").write_text(
            "MARS_OPENAI_API_KEY=sk-abcdefghijklmnop\n"
        )
        result = runner.invoke(main, ["configure"], input="\n\n\n\n\n\n")
        assert "sk-a...mnop" in result.output
        assert "Kept existing config" in result.output

    def test_idempotent(self, isolated_home):
        runner.invoke(main, ["configure"], input="\n\n\n\n\n\n")
        result = runner.invoke(main, ["configure"], input="\n\n\n\n\n\n")
        assert result.exit_code == 0

    def test_claude_code_when_dir_exists(self, isolated_home):
        (isolated_home / ".claude").mkdir()
        # Skip vertex, skip all keys, accept ollama, skip defaults, accept Claude Code (y)
        result = runner.invoke(main, ["configure"], input="\n\n\n\n\n\ny\n")
        assert result.exit_code == 0
        cmd_file = isolated_home / ".claude" / "commands" / "mars" / "debate.md"
        assert cmd_file.exists()
        content = cmd_file.read_text()
        assert "$ARGUMENTS" in content

    def test_no_claude_prompt_without_dir(self, isolated_home):
        result = runner.invoke(main, ["configure"], input="\n\n\n\n\n\n")
        assert "Claude Code" not in result.output


# ---------------------------------------------------------------------------
# Fixtures for post-debate commands
# ---------------------------------------------------------------------------

_ROUND1_RESPONSES = """\
# Round 1 - Initial Responses


## openai (gpt-4o)

OpenAI's answer here.

## anthropic (claude-sonnet-4-20250514)

Anthropic's answer here.
"""

_COSTS_MD = """\
# Cost Summary

| Provider | Model | Input Tokens | Output Tokens | Cost | Share |
|----------|-------|-------------|--------------|------|-------|
| openai | gpt-4o | 3,028 | 1,961 | $0.0272 | 37.4% |
| anthropic | claude-sonnet-4-20250514 | 3,428 | 2,342 | $0.0454 | 62.6% |

**Total**: 10,759 tokens | $0.0726
"""

_ATTRIBUTION_MD = """\
# Attribution

| Provider | Contribution | Survival | Influence |
|----------|-------------|----------|-----------|
| openai | 55% | 60% | 45% |
| anthropic | 45% | 50% | 55% |
"""

_FINAL_ANSWER = "# Final Answer\n\nThe synthesized answer is here."


@pytest.fixture()
def fake_debate(tmp_path):
    """Create a complete fake debate directory."""
    debate_dir = tmp_path / "mars-output" / "2026-02-16T08-06-45_test-debate-topic"
    audit = debate_dir / "audit"
    audit.mkdir(parents=True)

    (debate_dir / "final-answer.md").write_text(_FINAL_ANSWER)
    (audit / "00-prompt-and-context.md").write_text("# Prompt\n\nTest debate topic")
    (audit / "01-round-1-responses.md").write_text(_ROUND1_RESPONSES)
    (audit / "02-round-2-critiques.md").write_text("# Round 2\n\nCritiques here.")
    (audit / "costs.md").write_text(_COSTS_MD)
    (audit / "attribution.md").write_text(_ATTRIBUTION_MD)
    (audit / "convergence.md").write_text("Converged at round 2.")
    (audit / "resolution.md").write_text("# Resolution\n\nSynthesis reasoning.")
    (audit / "round-diffs.md").write_text("# Round Diffs\n\nChanges between rounds.")

    return tmp_path


@pytest.fixture()
def fake_debates(tmp_path):
    """Create multiple debate directories for history tests."""
    output = tmp_path / "mars-output"
    output.mkdir()

    for ts, slug in [
        ("2026-02-14T10-00-00", "first-topic"),
        ("2026-02-15T12-00-00", "second-topic"),
        ("2026-02-16T08-06-45", "third-topic"),
    ]:
        d = output / f"{ts}_{slug}"
        audit = d / "audit"
        audit.mkdir(parents=True)
        (d / "final-answer.md").write_text("Answer.")
        (audit / "01-round-1-responses.md").write_text(
            "## openai (gpt-4o)\n\nAnswer.\n"
        )
        (audit / "02-round-2-critiques.md").write_text("Round 2.")
        (audit / "costs.md").write_text(
            "**Total**: 1,000 tokens | $0.01"
        )

    # Also create a non-debate dir that should be filtered out
    (output / "random-dir").mkdir()

    return tmp_path


# ---------------------------------------------------------------------------
# Reader tests
# ---------------------------------------------------------------------------


class TestReader:
    def test_find_debates_sorted_recent_first(self, fake_debates):
        debates = find_debates(str(fake_debates / "mars-output"))
        names = [d.name for d in debates]
        assert names[0].startswith("2026-02-16")
        assert names[-1].startswith("2026-02-14")

    def test_find_debates_filters_non_debate_dirs(self, fake_debates):
        debates = find_debates(str(fake_debates / "mars-output"))
        names = [d.name for d in debates]
        assert "random-dir" not in names

    def test_find_debates_empty_dir(self, tmp_path):
        (tmp_path / "empty").mkdir()
        assert find_debates(str(tmp_path / "empty")) == []

    def test_find_debates_missing_dir(self, tmp_path):
        assert find_debates(str(tmp_path / "nonexistent")) == []

    def test_extract_timestamp(self):
        assert extract_timestamp("2026-02-16T08-06-45_topic") == "2026-02-16 08-06-45"

    def test_extract_prompt_from_dirname(self):
        result = extract_prompt_from_dirname("2026-02-16T08-06-45_my-cool-topic")
        assert result == "my cool topic"

    def test_extract_prompt_no_underscore(self):
        assert extract_prompt_from_dirname("no-underscore") == "no-underscore"

    def test_parse_costs_total(self):
        assert parse_costs_total(_COSTS_MD) == "$0.0726"

    def test_parse_costs_total_missing(self):
        assert parse_costs_total("no cost data") == "n/a"

    def test_parse_providers(self, fake_debate):
        debate_dir = fake_debate / "mars-output" / "2026-02-16T08-06-45_test-debate-topic"
        providers = parse_providers(debate_dir)
        assert providers == ["openai", "anthropic"]

    def test_count_rounds(self, fake_debate):
        debate_dir = fake_debate / "mars-output" / "2026-02-16T08-06-45_test-debate-topic"
        assert count_rounds(debate_dir) == 2


# ---------------------------------------------------------------------------
# Show tests
# ---------------------------------------------------------------------------


class TestShow:
    def test_summary_view(self, fake_debate):
        output_dir = str(fake_debate / "mars-output")
        result = runner.invoke(main, ["show", "-o", output_dir])
        assert result.exit_code == 0
        assert "test debate topic" in result.output
        assert "Final Answer" in result.output

    def test_show_answer(self, fake_debate):
        output_dir = str(fake_debate / "mars-output")
        result = runner.invoke(main, ["show", "-o", output_dir, "answer"])
        assert result.exit_code == 0
        assert "synthesized answer" in result.output

    def test_show_costs(self, fake_debate):
        output_dir = str(fake_debate / "mars-output")
        result = runner.invoke(main, ["show", "-o", output_dir, "costs"])
        assert result.exit_code == 0
        assert "$0.0272" in result.output

    def test_show_attribution(self, fake_debate):
        output_dir = str(fake_debate / "mars-output")
        result = runner.invoke(main, ["show", "-o", output_dir, "attribution"])
        assert result.exit_code == 0
        assert "Contribution" in result.output

    def test_show_rounds(self, fake_debate):
        output_dir = str(fake_debate / "mars-output")
        result = runner.invoke(main, ["show", "-o", output_dir, "rounds"])
        assert result.exit_code == 0
        assert "round" in result.output.lower()

    def test_show_specific_debate(self, fake_debate):
        debate_dir = str(
            fake_debate / "mars-output" / "2026-02-16T08-06-45_test-debate-topic"
        )
        result = runner.invoke(main, ["show", "--debate", debate_dir])
        assert result.exit_code == 0
        assert "test debate topic" in result.output

    def test_show_no_debates(self, tmp_path):
        result = runner.invoke(main, ["show", "-o", str(tmp_path / "empty")])
        assert result.exit_code != 0
        assert "No debates found" in result.output

    def test_show_incomplete_debate(self, fake_debate):
        # Remove the final answer
        debate_dir = (
            fake_debate / "mars-output" / "2026-02-16T08-06-45_test-debate-topic"
        )
        (debate_dir / "final-answer.md").unlink()
        output_dir = str(fake_debate / "mars-output")
        result = runner.invoke(main, ["show", "-o", output_dir])
        assert result.exit_code == 0
        assert "incomplete" in result.output.lower()


# ---------------------------------------------------------------------------
# History tests
# ---------------------------------------------------------------------------


class TestHistory:
    def test_lists_debates(self, fake_debates):
        output_dir = str(fake_debates / "mars-output")
        result = runner.invoke(main, ["history", "-o", output_dir])
        assert result.exit_code == 0
        assert "first topic" in result.output
        assert "second topic" in result.output
        assert "third topic" in result.output

    def test_limit(self, fake_debates):
        output_dir = str(fake_debates / "mars-output")
        result = runner.invoke(main, ["history", "-n", "1", "-o", output_dir])
        assert result.exit_code == 0
        assert "third topic" in result.output
        assert "first topic" not in result.output

    def test_empty_directory(self, tmp_path):
        (tmp_path / "empty").mkdir()
        result = runner.invoke(main, ["history", "-o", str(tmp_path / "empty")])
        assert result.exit_code != 0
        assert "No debates found" in result.output

    def test_sorted_most_recent_first(self, fake_debates):
        output_dir = str(fake_debates / "mars-output")
        result = runner.invoke(main, ["history", "-o", output_dir])
        # Row 1 should be the most recent
        lines = result.output.splitlines()
        # Find the data rows (contain topic names)
        data_lines = [line for line in lines if "topic" in line]
        assert "third" in data_lines[0]
        assert "first" in data_lines[-1]


# ---------------------------------------------------------------------------
# Copy tests
# ---------------------------------------------------------------------------


class TestCopy:
    def test_prints_when_clipboard_unavailable(self, fake_debate):
        output_dir = str(fake_debate / "mars-output")
        with patch("mars.cli._copy_to_clipboard", return_value=False):
            result = runner.invoke(main, ["copy", "-o", output_dir])
        assert result.exit_code == 0
        assert "synthesized answer" in result.output
        assert "Clipboard not available" in result.output

    def test_clipboard_success_message(self, fake_debate):
        output_dir = str(fake_debate / "mars-output")
        with patch("mars.cli._copy_to_clipboard", return_value=True):
            result = runner.invoke(main, ["copy", "-o", output_dir])
        assert result.exit_code == 0
        assert "Copied to clipboard" in result.output

    def test_full_includes_prompt_and_attribution(self, fake_debate):
        output_dir = str(fake_debate / "mars-output")
        with patch("mars.cli._copy_to_clipboard", return_value=False):
            result = runner.invoke(main, ["copy", "--full", "-o", output_dir])
        assert result.exit_code == 0
        assert "Prompt" in result.output
        assert "Attribution" in result.output
        assert "synthesized answer" in result.output

    def test_no_answer_error(self, fake_debate):
        debate_dir = (
            fake_debate / "mars-output" / "2026-02-16T08-06-45_test-debate-topic"
        )
        (debate_dir / "final-answer.md").unlink()
        output_dir = str(fake_debate / "mars-output")
        result = runner.invoke(main, ["copy", "-o", output_dir])
        assert result.exit_code != 0
        assert "No final answer" in result.output

    def test_specific_debate(self, fake_debate):
        debate_dir = str(
            fake_debate / "mars-output" / "2026-02-16T08-06-45_test-debate-topic"
        )
        with patch("mars.cli._copy_to_clipboard", return_value=True):
            result = runner.invoke(main, ["copy", "--debate", debate_dir])
        assert result.exit_code == 0
        assert "Copied to clipboard" in result.output


# ---------------------------------------------------------------------------
# Vertex & participant ID tests
# ---------------------------------------------------------------------------


class TestVertexConfigure:
    def test_saves_project_and_region(self, isolated_home):
        # Provide vertex project + region, skip OpenAI/Anthropic/Google,
        # accept Ollama default, skip defaults
        result = runner.invoke(
            main, ["configure"],
            input="my-project\nus-east1\n\n\n\n\n\n",
        )
        assert result.exit_code == 0
        config_file = isolated_home / ".mars" / "config"
        assert config_file.exists()
        content = config_file.read_text()
        assert "MARS_VERTEX_PROJECT_ID=my-project" in content
        assert "MARS_VERTEX_REGION=us-east1" in content

    def test_skip_vertex(self, isolated_home):
        # Skip vertex (empty project ID), skip rest (6 newlines total)
        result = runner.invoke(main, ["configure"], input="\n\n\n\n\n\n")
        assert result.exit_code == 0
        assert "Skipped" in result.output

    def test_vertex_in_providers_list(self):
        result = runner.invoke(main, ["providers"])
        assert result.exit_code == 0
        assert "vertex" in result.output


class TestConfigurableDefaults:
    def test_get_default_providers_from_config(self):
        from mars.config import AppConfig
        config = AppConfig(default_providers="vertex:claude-opus-4-6,openai")
        assert config.get_default_providers() == ["vertex:claude-opus-4-6", "openai"]

    def test_get_default_providers_fallback(self):
        from mars.config import AppConfig
        config = AppConfig(default_providers="")
        assert config.get_default_providers() == ["openai", "anthropic"]


class TestParticipantId:
    def test_provider_base_name(self):
        from mars.models import provider_base_name
        assert provider_base_name("vertex:claude-opus-4-6") == "vertex"
        assert provider_base_name("openai") == "openai"
        assert provider_base_name("vertex:gemini-2.5-flash") == "vertex"

    def test_vertex_in_available_providers(self):
        from mars.providers.registry import AVAILABLE_PROVIDERS
        assert "vertex" in AVAILABLE_PROVIDERS

    def test_registry_routing(self):
        from mars.config import AppConfig
        from mars.providers.registry import get_provider
        from mars.providers.vertex import VertexClaudeProvider, VertexGeminiProvider
        config = AppConfig(vertex_project_id="test-project", vertex_region="us-central1")
        p1 = get_provider("vertex", config, model="claude-opus-4-6")
        assert isinstance(p1, VertexClaudeProvider)
        p2 = get_provider("vertex", config, model="gemini-2.5-flash")
        assert isinstance(p2, VertexGeminiProvider)
        # Default (no model) -> Claude
        p3 = get_provider("vertex", config)
        assert isinstance(p3, VertexClaudeProvider)


class TestErrorFormatting:
    def test_openai_404(self):
        from mars.debate.base import _format_provider_error
        err = Exception(
            "Error code: 404 - {'error': {'message': \"The model `gpt-oss-120b` does not exist "
            "or you do not have access to it.\", 'type': 'invalid_request_error'}}"
        )
        result = _format_provider_error(err)
        assert "gpt-oss-120b" in result
        assert "not found" in result.lower()
        assert "{" not in result

    def test_anthropic_404(self):
        from mars.debate.base import _format_provider_error
        err = Exception(
            "Error code: 404 - {'type': 'error', 'error': {'type': 'not_found_error', "
            "'message': 'model: claude-4-6-opus@20260215'}}"
        )
        result = _format_provider_error(err)
        assert "claude-4-6-opus@20260215" in result
        assert "not found" in result.lower()

    def test_google_404(self):
        from mars.debate.base import _format_provider_error
        err = Exception(
            "404 NOT_FOUND. {'error': {'code': 404, 'message': 'Publisher Model "
            "`projects/my-project/locations/us-central1/publishers/google/models/gemini-3-pro` "
            "was not found.'}}"
        )
        result = _format_provider_error(err)
        assert "gemini-3-pro" in result
        assert "not found" in result.lower()

    def test_auth_error(self):
        from mars.debate.base import _format_provider_error
        err = Exception("Error code: 401 - Unauthorized")
        result = _format_provider_error(err)
        assert "mars configure" in result.lower()

    def test_permission_error(self):
        from mars.debate.base import _format_provider_error
        err = Exception("403 Forbidden: permission denied for resource")
        result = _format_provider_error(err)
        assert "permission" in result.lower()

    def test_unknown_error_strips_json(self):
        from mars.debate.base import _format_provider_error
        err = Exception("500 Internal Server Error. {'error': {'code': 500, 'message': 'oops'}}")
        result = _format_provider_error(err)
        assert "{" not in result
