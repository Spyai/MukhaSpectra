"""Programmatic Alembic migrations (run to head on app startup)."""

from __future__ import annotations

from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy.engine import Engine, make_url

from app.settings import BACKEND_DIR

ALEMBIC_DIR = BACKEND_DIR / "alembic"
ALEMBIC_INI = BACKEND_DIR / "alembic.ini"


def alembic_config(database_url: str | None = None) -> Config:
    """An Alembic config pointing at backend/alembic, without reading alembic.ini logging."""
    config = Config()
    config.set_main_option("script_location", str(ALEMBIC_DIR))
    if database_url:
        config.set_main_option("sqlalchemy.url", database_url.replace("%", "%%"))
    return config


def prepare_database(url: str) -> None:
    """Create the folder of a file-based SQLite database."""
    parsed = make_url(url)
    if parsed.get_backend_name() == "sqlite" and parsed.database not in (None, "", ":memory:"):
        Path(parsed.database).expanduser().parent.mkdir(parents=True, exist_ok=True)


def upgrade_to_head(engine: Engine) -> None:
    """Apply all migrations on ``engine`` (idempotent)."""
    prepare_database(engine.url.render_as_string(hide_password=False))
    config = alembic_config()
    with engine.begin() as connection:
        config.attributes["connection"] = connection
        command.upgrade(config, "head")
