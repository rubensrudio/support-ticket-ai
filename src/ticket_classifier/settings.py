"""Runtime settings loaded from ``TICKET_*`` environment variables."""

from pathlib import Path

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Service configuration.

    ``api_key`` is kept as a ``SecretStr`` so it never appears in ``repr``/``str``.
    An empty or whitespace-only ``TICKET_API_KEY`` is treated as not configured.
    """

    model_config = SettingsConfigDict(env_prefix="TICKET_")

    api_key: SecretStr | None = None
    review_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    db_path: Path = Path("var/tickets.db")
    artifacts_dir: Path = Path("artifacts")
    retention_days: int = Field(default=90, ge=1)
    purge_interval_seconds: int = Field(default=86400, ge=1)

    @field_validator("api_key", mode="before")
    @classmethod
    def _blank_api_key_is_none(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        if isinstance(value, SecretStr) and not value.get_secret_value().strip():
            return None
        return value


def get_settings() -> Settings:
    """Build a fresh ``Settings`` instance from the current environment (no caching)."""
    return Settings()
