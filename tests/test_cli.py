"""Tests for the MARS CLI."""

from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from mars.cli import main


runner = CliRunner()


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

    def test_mentions_env_vars(self):
        result = runner.invoke(main, ["providers", "--help"])
        assert "MARS_*_API_KEY" in result.output


class TestConfigure:
    def test_creates_command_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        result = runner.invoke(main, ["configure"])
        assert result.exit_code == 0

        cmd_file = tmp_path / ".claude" / "commands" / "mars" / "debate.md"
        assert cmd_file.exists()

        content = cmd_file.read_text()
        assert "$ARGUMENTS" in content
        assert "mars debate" in content
        assert "mars providers" in content

    def test_prints_confirmation(self, tmp_path, monkeypatch):
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        result = runner.invoke(main, ["configure"])
        assert "/mars:debate" in result.output

    def test_idempotent(self, tmp_path, monkeypatch):
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        runner.invoke(main, ["configure"])
        result = runner.invoke(main, ["configure"])
        assert result.exit_code == 0

        cmd_file = tmp_path / ".claude" / "commands" / "mars" / "debate.md"
        assert cmd_file.exists()
