"""
Application settings — loaded from environment variables / .env file.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # LLM
    anthropic_api_key: str = Field(..., description="Anthropic API key")
    openai_api_key: str = Field("", description="OpenAI API key (optional fallback)")
    primary_model: str = "claude-sonnet-4-20250514"
    embedding_model: str = "all-MiniLM-L6-v2"
    max_tokens: int = 4096
    temperature: float = 0.1

    # News APIs
    newsapi_key: str = Field("", description="NewsAPI key")
    gnews_api_key: str = Field("", description="GNews API key")
    mediastack_api_key: str = Field("", description="Mediastack API key")
    twitter_bearer_token: str = Field("", description="Twitter Bearer token")

    # Cache
    redis_url: str = "redis://localhost:6379/0"
    cache_ttl_seconds: int = 3600

    # Vector DB
    faiss_index_path: Path = Path("./data/faiss_index")
    faiss_dimension: int = 384

    # API Server
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 4
    api_reload: bool = False

    # Rate limiting
    max_requests_per_minute: int = 60
    scraper_concurrency: int = 5
    retrieval_top_k: int = 10

    # Config files
    trusted_sources_config: Path = Path("./config/trusted_sources.yaml")

    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_format: Literal["json", "console"] = "json"

    # Monitoring
    enable_metrics: bool = True
    metrics_port: int = 9090

    @field_validator("faiss_index_path", "trusted_sources_config", mode="before")
    @classmethod
    def make_path(cls, v: str) -> Path:
        return Path(v)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
