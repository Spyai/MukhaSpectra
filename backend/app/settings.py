"""Runtime settings for the MukhaSpectra API.

Every field can be set with an environment variable prefixed ``MUKHA_`` (for example
``MUKHA_DATABASE_URL``, ``MUKHA_API_KEY`` or ``MUKHA_ENABLE_DEMO=false``). Tests build a
``Settings`` object directly and pass it to ``create_app`` so each test gets its own database and
storage directory.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, timedelta, timezone, tzinfo
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import Field, SecretStr, field_validator
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
MIN_API_KEY_LENGTH = 16
_BUILD_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
# India observes no daylight saving time, so this fallback is exact when the system has no tz data.
_FIXED_ZONES: dict[str, tzinfo] = {
    "Asia/Kolkata": timezone(timedelta(hours=5, minutes=30), "IST"),
    "Asia/Calcutta": timezone(timedelta(hours=5, minutes=30), "IST"),
    "UTC": UTC,
}


def resolve_timezone(name: str) -> tzinfo:
    """The tzinfo for an IANA name (fixed-offset fallback for India and UTC)."""
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        if name in _FIXED_ZONES:
            return _FIXED_ZONES[name]
        raise ValueError(f"Unknown time zone '{name}'.") from exc


def _split_list(value: Any) -> Any:
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith("["):
            return json.loads(text)
        return [part.strip() for part in text.split(",") if part.strip()]
    return value


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
        description="Base URL of the web console; used for the record link and QR code.",
    )
    cors_origins: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: list(DEFAULT_CORS_ORIGINS),
        description="Allowed browser origins (JSON list or comma-separated).",
    )
    api_key: SecretStr | None = Field(
        default=None,
        description="Shared API key (env MUKHA_API_KEY). When set, every /api/v1 route except "
        "/health needs it in the X-API-Key header. Unset: no authentication at all, so the "
        "server must not be reachable beyond localhost. Per-user accounts and roles are Phase 5.",
    )
    allowed_hosts: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["*"],
        description="Host header values the API answers to (DNS-rebinding guard), JSON list or "
        "comma-separated; '*' allows any.",
    )
    build_id: str | None = Field(
        default=None,
        description="Build identifier recorded in every result (env MUKHA_BUILD_ID). Defaults to "
        "the git commit when available, else 'unknown'.",
    )
    default_timezone: str = Field(
        default="Asia/Kolkata",
        description="Time zone for timestamps sent without an offset (camp-local time).",
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
        default=200,
        ge=1,
        le=8192,
        description="Limit for one whole request body (keep deploy/nginx.conf in step).",
    )
    max_decoded_megapixels: int = Field(
        default=120,
        ge=1,
        le=4000,
        description="Budget for all images of one upload after decoding (each is decoded at "
        "most 2048 px on its longest side). Larger uploads get 413.",
    )
    max_concurrent_analyses: int = Field(
        default=2,
        ge=1,
        le=64,
        description="Uploads, demos and re-analyses running at once; more get 503.",
    )
    enable_demo: bool = Field(default=True, description="Expose the SYNTHETIC demo endpoints.")
    log_level: str = Field(default="INFO")
    auto_migrate: bool = Field(
        default=True, description="Run Alembic migrations to head when the app starts."
    )

    @field_validator("cors_origins", "allowed_hosts", mode="before")
    @classmethod
    def _split_lists(cls, value: Any) -> Any:
        return _split_list(value)

    @field_validator("api_key", mode="before")
    @classmethod
    def _blank_key_is_none(cls, value: Any) -> Any:
        if isinstance(value, SecretStr):
            value = value.get_secret_value()
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return None
            if len(value) < MIN_API_KEY_LENGTH:
                raise ValueError(f"The API key must be at least {MIN_API_KEY_LENGTH} characters.")
        return value

    @field_validator("build_id", mode="before")
    @classmethod
    def _check_build_id(cls, value: Any) -> Any:
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return None
            if not _BUILD_ID.fullmatch(value):
                raise ValueError("build_id may only contain letters, digits, '.', '_' and '-'.")
        return value

    @field_validator("default_timezone")
    @classmethod
    def _check_timezone(cls, value: str) -> str:
        resolve_timezone(value)
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

    @property
    def max_decoded_pixels(self) -> int:
        return self.max_decoded_megapixels * 1_000_000

    @property
    def tz(self) -> tzinfo:
        return resolve_timezone(self.default_timezone)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings read from the environment (cached)."""
    return Settings()
