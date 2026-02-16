"""Provider factory and registry."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mars.providers.anthropic import AnthropicProvider
from mars.providers.google import GoogleProvider
from mars.providers.ollama import OllamaProvider
from mars.providers.openai import OpenAIProvider
from mars.providers.vertex import VertexClaudeProvider

if TYPE_CHECKING:
    from mars.config import AppConfig
    from mars.providers.base import LLMProvider

_PROVIDERS: dict[str, type] = {
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "google": GoogleProvider,
    "ollama": OllamaProvider,
    "vertex": VertexClaudeProvider,
}

AVAILABLE_PROVIDERS = list(_PROVIDERS.keys())


def get_provider(name: str, config: AppConfig, *, model: str | None = None) -> LLMProvider:
    if name == "vertex" and model and model.startswith("gemini"):
        from mars.providers.vertex import VertexGeminiProvider

        return VertexGeminiProvider(config)
    cls = _PROVIDERS.get(name)
    if cls is None:
        raise ValueError(f"Unknown provider '{name}'. Available: {', '.join(AVAILABLE_PROVIDERS)}")
    return cls(config)  # type: ignore[no-any-return]
