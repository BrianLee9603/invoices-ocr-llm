"""
Centralized application configuration using pydantic-settings.

All settings are loaded from environment variables (or .env file).
Defaults match the local Docker Compose credentials in infra/docker-compose.yaml.
"""

from functools import lru_cache

from pydantic import computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseSettings(BaseSettings):
    """PostgreSQL connection settings."""

    model_config = SettingsConfigDict(env_prefix="POSTGRES_")

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

    model_config = SettingsConfigDict(env_prefix="REDIS_")

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

    model_config = SettingsConfigDict(env_prefix="MINIO_")

    endpoint: str = "http://localhost:9000"
    access_key: str = "minioadmin"
    secret_key: str = "minioadminpassword"
    secure: bool = False


class AppSettings(BaseSettings):
    """Root application settings aggregating all sub-settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_env: str = "local"
    log_level: str = "DEBUG"

    database: DatabaseSettings = DatabaseSettings()
    redis: RedisSettings = RedisSettings()
    minio: MinioSettings = MinioSettings()


@lru_cache
def get_settings() -> AppSettings:
    """Return a cached singleton of application settings."""
    return AppSettings()
