"""Provider factory and registry."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mar.providers.anthropic import AnthropicProvider
from mar.providers.google import GoogleProvider
from mar.providers.ollama import OllamaProvider
from mar.providers.openai import OpenAIProvider

if TYPE_CHECKING:
    from mar.config import AppConfig
    from mar.providers.base import LLMProvider

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
