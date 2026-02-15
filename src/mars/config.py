"""Application configuration via environment variables."""

from __future__ import annotations

from pydantic_settings import BaseSettings

_DEFAULT_MODELS: dict[str, str] = {
    "openai": "gpt-4o",
    "anthropic": "claude-sonnet-4-20250514",
    "google": "gemini-2.0-flash",
    "ollama": "llama3.2",
}


class AppConfig(BaseSettings):
    model_config = {"env_prefix": "MARS_"}

    openai_api_key: str = ""
    anthropic_api_key: str = ""
    google_api_key: str = ""
    ollama_base_url: str = "http://localhost:11434"

    def get_api_key(self, provider: str) -> str:
        key_map: dict[str, str] = {
            "openai": self.openai_api_key,
            "anthropic": self.anthropic_api_key,
            "google": self.google_api_key,
        }
        return key_map.get(provider, "")

    def get_default_model(self, provider: str) -> str:
        return _DEFAULT_MODELS.get(provider, "")
