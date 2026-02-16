"""Application configuration via environment variables."""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

MARS_CONFIG_DIR = Path.home() / ".mars"
MARS_CONFIG_FILE = MARS_CONFIG_DIR / "config"


def load_mars_config() -> None:
    """Load MARS configuration from all sources.

    Priority (highest wins): env vars > .env (local) > ~/.mars/config (global).

    Each load_dotenv call with override=False only sets vars not already
    present, so we load highest-priority sources first.
    """
    load_dotenv(override=False)
    if MARS_CONFIG_FILE.is_file():
        load_dotenv(MARS_CONFIG_FILE, override=False)

_DEFAULT_MODELS: dict[str, str] = {
    "openai": "gpt-4o",
    "anthropic": "claude-sonnet-4-20250514",
    "google": "gemini-2.0-flash",
    "ollama": "llama3.2",
    "vertex": "claude-opus-4-6",
}


class AppConfig(BaseSettings):
    model_config = {"env_prefix": "MARS_"}

    openai_api_key: str = ""
    anthropic_api_key: str = ""
    google_api_key: str = ""
    ollama_base_url: str = "http://localhost:11434"
    vertex_project_id: str = ""
    vertex_region: str = "us-central1"
    default_providers: str = ""

    def get_api_key(self, provider: str) -> str:
        key_map: dict[str, str] = {
            "openai": self.openai_api_key,
            "anthropic": self.anthropic_api_key,
            "google": self.google_api_key,
            "vertex": self.vertex_project_id,
        }
        return key_map.get(provider, "")

    def get_default_providers(self) -> list[str]:
        if self.default_providers:
            return [p.strip() for p in self.default_providers.split(",") if p.strip()]
        return ["openai", "anthropic"]

    def get_default_model(self, provider: str) -> str:
        return _DEFAULT_MODELS.get(provider, "")
