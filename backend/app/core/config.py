"""
Centralised application configuration.

All values are read from environment variables so that the same code can run
locally, in Docker, and on Render without any source changes. See
`backend/.env.example` for the full list of supported variables.
"""
from functools import lru_cache
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", extra="ignore", protected_namespaces=("settings_",)
    )

    # --- General ---
    app_name: str = "AI-Generated Text Detection API"
    app_version: str = "1.0.0"

    # --- CORS ---
    # Comma-separated origins, e.g. "https://app.vercel.app,http://localhost:5173"
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    # --- Input limits ---
    max_text_length: int = 20000
    min_text_length: int = 50

    # --- Perplexity model ---
    # XLM-RoBERTa is a masked language model. We therefore compute
    # pseudo-perplexity, not standard autoregressive perplexity.
    perplexity_model_name: str = "xlm-roberta-base"
    inference_device: str = "cpu"

    # --- Model artifacts ---
    model_dir: str = "models"

    # --- Supported languages ---
    supported_languages: List[str] = ["en", "te", "hi"]

    @property
    def cors_origin_list(self) -> List[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    """Settings are cached so environment variables are parsed only once per process."""
    return Settings()
