"""
Centralized application configuration using pydantic-settings.

All settings are loaded from environment variables (or .env file).
Defaults match the local Docker Compose credentials in infra/docker-compose.yaml.
"""

from functools import lru_cache
from typing import Optional

from pydantic import computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseSettings(BaseSettings):
    """PostgreSQL connection settings."""

    model_config = SettingsConfigDict(env_prefix="POSTGRES_", env_file=".env", extra="ignore")

    host: str = "localhost"
    port: int = 5432
    user: str = "postgres"
    password: str = "localpassword"
    db: str = "invoice_extraction"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def dsn(self) -> str:
        """Async-compatible PostgreSQL DSN for SQLAlchemy."""
        return (
            f"postgresql+asyncpg://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.db}"
        )


class RedisSettings(BaseSettings):
    """Redis connection settings."""

    model_config = SettingsConfigDict(env_prefix="REDIS_", env_file=".env", extra="ignore")

    host: str = "localhost"
    port: int = 6379
    db: int = 0

    @computed_field  # type: ignore[prop-decorator]
    @property
    def url(self) -> str:
        """Redis connection URL."""
        return f"redis://{self.host}:{self.port}/{self.db}"


class MinioSettings(BaseSettings):
    """MinIO (S3-compatible) connection settings."""

    model_config = SettingsConfigDict(env_prefix="MINIO_", env_file=".env", extra="ignore")

    endpoint: str = "http://localhost:9000"
    access_key: str = "minioadmin"
    secret_key: str = "minioadminpassword"
    secure: bool = False


class OllamaSettings(BaseSettings):
    """Ollama self-hosted LLM settings."""

    model_config = SettingsConfigDict(env_prefix="OLLAMA_", env_file=".env", extra="ignore")

    host: str = "http://localhost:11434"
    model: str = "qwen2.5:7b"


class ProcessingSettings(BaseSettings):
    """Processing worker settings."""

    model_config = SettingsConfigDict(env_prefix="PROCESSING_", env_file=".env", extra="ignore")

    ocr_engine: str = "paddleocr"  # "paddleocr" or "docling"
    preprocess: bool = True  # Enable image preprocessing (CLAHE, de-skew, etc.)
    layout_reconstruction: bool = True  # Enable layout-aware text reconstruction
    ocr_version: str = "PP-OCRv5"
    ocr_max_workers: int = 2         # Thread pool size for OCR
    max_resolution: int = 4000       # Maximum image resolution (long side)
    fallback_enabled: bool = True    # Retry without preprocessing on 0 blocks
    det_limit_side_len: int = 960      # PaddleOCR detection max side length


class GeminiSettings(BaseSettings):
    """Google Gemini API settings."""

    model_config = SettingsConfigDict(env_prefix="GEMINI_", env_file=".env", extra="ignore")

    api_key: Optional[str] = None
    model: str = "gemini-2.5-flash"


class AppSettings(BaseSettings):
    """Root application settings aggregating all sub-settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_env: str = "local"
    log_level: str = "DEBUG"
    llm_provider: str = "ollama"  # "ollama" or "gemini"

    database: DatabaseSettings = DatabaseSettings()
    redis: RedisSettings = RedisSettings()
    minio: MinioSettings = MinioSettings()
    ollama: OllamaSettings = OllamaSettings()
    gemini: GeminiSettings = GeminiSettings()
    processing: ProcessingSettings = ProcessingSettings()


@lru_cache
def get_settings() -> AppSettings:
    """Return a cached singleton of application settings."""
    return AppSettings()
