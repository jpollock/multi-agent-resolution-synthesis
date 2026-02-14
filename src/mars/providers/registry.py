"""Provider factory and registry."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mars.providers.anthropic import AnthropicProvider
from mars.providers.google import GoogleProvider
from mars.providers.ollama import OllamaProvider
from mars.providers.openai import OpenAIProvider

if TYPE_CHECKING:
    from mars.config import AppConfig
    from mars.providers.base import LLMProvider

_PROVIDERS: dict[str, type] = {
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "google": GoogleProvider,
    "ollama": OllamaProvider,
}

AVAILABLE_PROVIDERS = list(_PROVIDERS.keys())


def get_provider(name: str, config: AppConfig) -> LLMProvider:
    cls = _PROVIDERS.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown provider '{name}'. Available: {', '.join(AVAILABLE_PROVIDERS)}"
        )
    return cls(config)
