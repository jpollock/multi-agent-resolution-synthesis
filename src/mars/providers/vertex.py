"""Vertex AI providers (Claude via Anthropic SDK, Gemini via Google GenAI SDK)."""

from __future__ import annotations

from mars.config import AppConfig
from mars.models import TokenUsage
from mars.providers.anthropic import AnthropicProvider
from mars.providers.google import GoogleProvider


class VertexClaudeProvider(AnthropicProvider):
    """Claude models accessed through Vertex AI using the Anthropic SDK."""

    def __init__(self, config: AppConfig) -> None:
        from anthropic import AsyncAnthropicVertex

        self._config = config
        self._last_usage = TokenUsage()
        self._client = AsyncAnthropicVertex(
            project_id=config.vertex_project_id,
            region=config.vertex_region,
        )

    @property
    def name(self) -> str:
        return "vertex"

    @property
    def default_model(self) -> str:
        return self._config.get_default_model("vertex")


class VertexGeminiProvider(GoogleProvider):
    """Gemini models accessed through Vertex AI using the Google GenAI SDK."""

    def __init__(self, config: AppConfig) -> None:
        from google import genai

        self._config = config
        self._last_usage = TokenUsage()
        # Anthropic SDK accepts "global" but Google GenAI SDK needs a real region
        region = config.vertex_region
        if region == "global":
            region = "us-central1"
        self._client = genai.Client(
            vertexai=True,
            project=config.vertex_project_id,
            location=region,
        )

    @property
    def name(self) -> str:
        return "vertex"

    @property
    def default_model(self) -> str:
        return "gemini-2.5-flash"
