"""Application configuration using Pydantic Settings.

Supports environment variable overrides following the 12-factor app pattern.
All settings can be overridden via environment variables prefixed with ``SHIELDAI_``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelConfig(BaseSettings):
    """Configuration for ML model loading and inference."""

    model_config = SettingsConfigDict(env_prefix="SHIELDAI_MODEL_")

    text_model_name: str = Field(
        default="unitary/toxic-bert",
        description="HuggingFace model ID for text toxicity classification",
    )
    image_model_name: str = Field(
        default="openai/clip-vit-base-patch32",
        description="HuggingFace model ID for image safety classification",
    )
    device: str = Field(
        default="cpu",
        description="Device to load models on: 'cpu', 'cuda', or 'mps'",
    )
    max_text_length: int = Field(
        default=512,
        description="Maximum token length for text inputs",
    )
    batch_size: int = Field(
        default=16,
        description="Batch size for inference",
    )
    model_cache_dir: Path = Field(
        default=Path.home() / ".cache" / "shieldai" / "models",
        description="Directory to cache downloaded models",
    )


class ThresholdConfig(BaseSettings):
    """Confidence thresholds for moderation verdicts."""

    model_config = SettingsConfigDict(env_prefix="SHIELDAI_THRESHOLD_")

    toxic: float = Field(default=0.7, description="Threshold for toxic content")
    hate_speech: float = Field(default=0.7, description="Threshold for hate speech")
    spam: float = Field(default=0.6, description="Threshold for spam detection")
    nsfw: float = Field(default=0.7, description="Threshold for NSFW content")
    needs_review: float = Field(
        default=0.4,
        description="Below reject threshold but above this → NEEDS_REVIEW",
    )


class APIConfig(BaseSettings):
    """Configuration for the FastAPI server."""

    model_config = SettingsConfigDict(env_prefix="SHIELDAI_API_")

    host: str = Field(default="0.0.0.0", description="API server host")
    port: int = Field(default=8000, description="API server port")
    workers: int = Field(default=1, description="Number of uvicorn workers")
    cors_origins: list[str] = Field(
        default=["*"],
        description="Allowed CORS origins",
    )
    max_request_size_mb: int = Field(
        default=10,
        description="Maximum request body size in megabytes",
    )
    rate_limit_per_minute: int = Field(
        default=60,
        description="Maximum requests per minute per client",
    )


class QueueConfig(BaseSettings):
    """Configuration for the async task queue."""

    model_config = SettingsConfigDict(env_prefix="SHIELDAI_QUEUE_")

    max_workers: int = Field(
        default=4,
        description="Maximum concurrent task workers",
    )
    max_queue_size: int = Field(
        default=1000,
        description="Maximum number of pending tasks",
    )
    task_timeout_seconds: int = Field(
        default=300,
        description="Timeout for individual tasks in seconds",
    )


class StorageConfig(BaseSettings):
    """Configuration for result storage."""

    model_config = SettingsConfigDict(env_prefix="SHIELDAI_STORAGE_")

    database_path: Path = Field(
        default=Path("data/shieldai.db"),
        description="Path to SQLite database file",
    )
    result_ttl_hours: int = Field(
        default=24,
        description="Hours to retain results before auto-cleanup",
    )


class Settings(BaseSettings):
    """Root settings aggregating all configuration sections."""

    model_config = SettingsConfigDict(
        env_prefix="SHIELDAI_",
        env_nested_delimiter="__",
    )

    app_name: str = "ShieldAI"
    version: str = "1.0.0"
    environment: Literal["development", "staging", "production"] = "development"
    debug: bool = Field(default=False, description="Enable debug mode")
    log_level: str = Field(default="INFO", description="Logging level")

    # Sub-configurations
    model: ModelConfig = Field(default_factory=ModelConfig)
    thresholds: ThresholdConfig = Field(default_factory=ThresholdConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    queue: QueueConfig = Field(default_factory=QueueConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)


# Singleton instance — import this throughout the application
_settings: Settings | None = None


def get_settings() -> Settings:
    """Return the application settings singleton."""
    global _settings  # noqa: PLW0603
    if _settings is None:
        _settings = Settings()
    return _settings


def reset_settings() -> None:
    """Reset settings singleton (useful for testing)."""
    global _settings  # noqa: PLW0603
    _settings = None
