"""Runtime settings for the MukhaSpectra API.

Every field can be set with an environment variable prefixed ``MUKHA_`` (for example
``MUKHA_DATABASE_URL`` or ``MUKHA_ENABLE_DEMO=false``). Tests build a ``Settings`` object directly
and pass it to ``create_app`` so each test gets its own database and storage directory.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent.parent
VAR_DIR = BACKEND_DIR / "var"

DEFAULT_CORS_ORIGINS: tuple[str, ...] = (
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:4173",
    "http://127.0.0.1:4173",
)

_MB = 1024 * 1024


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MUKHA_", extra="ignore")

    database_url: str = Field(
        default=f"sqlite:///{(VAR_DIR / 'mukha.db').as_posix()}",
        description="SQLAlchemy URL. SQLite in development, PostgreSQL in production.",
    )
    storage_dir: Path = Field(
        default=VAR_DIR / "storage",
        description="Root of the local object store (captures, artifacts, reports).",
    )
    public_base_url: str = Field(
        default="http://localhost:5173",
        description="Base URL of the web console; used for the secure-record link and QR code.",
    )
    cors_origins: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: list(DEFAULT_CORS_ORIGINS),
        description="Allowed browser origins (JSON list or comma-separated).",
    )
    wl_model_dir: Path | None = Field(
        default=None,
        description="Folder with model.onnx + model_card.md for the white-light classifier. "
        "When unset the classifier reports 'model_unavailable' and is not used in triage.",
    )
    facilities_file: Path | None = Field(
        default=None, description="YAML/JSON list of referral facilities for the report."
    )
    max_upload_mb: int = Field(default=40, ge=1, le=2048, description="Per-file upload limit.")
    max_request_mb: int = Field(
        default=400, ge=1, le=8192, description="Limit for one whole multipart upload request."
    )
    enable_demo: bool = Field(default=True, description="Expose the SYNTHETIC demo endpoints.")
    log_level: str = Field(default="INFO")
    auto_migrate: bool = Field(
        default=True, description="Run Alembic migrations to head when the app starts."
    )

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_origins(cls, value: Any) -> Any:
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return []
            if text.startswith("["):
                return json.loads(text)
            return [part.strip() for part in text.split(",") if part.strip()]
        return value

    @field_validator("public_base_url")
    @classmethod
    def _strip_trailing_slash(cls, value: str) -> str:
        return value.rstrip("/")

    @field_validator("storage_dir")
    @classmethod
    def _absolute_storage(cls, value: Path) -> Path:
        return Path(value).expanduser().resolve()

    @field_validator("log_level")
    @classmethod
    def _upper_level(cls, value: str) -> str:
        return value.upper()

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * _MB

    @property
    def max_request_bytes(self) -> int:
        return self.max_request_mb * _MB


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings read from the environment (cached)."""
    return Settings()
