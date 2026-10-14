"""Programmatic Alembic migrations (run to head on app startup)."""

from __future__ import annotations

from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy.engine import Engine, make_url

from app.settings import BACKEND_DIR


def find_alembic_dir() -> Path:
    """The migration scripts folder (``backend/alembic``).

    Next to the source tree in development and in the Docker image (uvicorn runs from
    ``/app/backend``); the working directory is also searched so a non-editable install still
    finds the scripts when started from the backend folder.
    """
    for base in (BACKEND_DIR, Path.cwd(), Path.cwd() / "backend"):
        candidate = base / "alembic"
        if (candidate / "env.py").is_file() and (candidate / "versions").is_dir():
            return candidate
    return BACKEND_DIR / "alembic"


ALEMBIC_DIR = find_alembic_dir()
ALEMBIC_INI = ALEMBIC_DIR.parent / "alembic.ini"


def alembic_config(database_url: str | None = None) -> Config:
    """An Alembic config pointing at backend/alembic, without reading alembic.ini logging."""
    if not (ALEMBIC_DIR / "env.py").is_file():
        raise RuntimeError(
            f"Alembic migration scripts were not found at {ALEMBIC_DIR}. Run the API from the "
            "backend folder or install it in editable mode."
        )
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
