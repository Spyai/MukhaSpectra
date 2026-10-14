"""Alembic environment for the MukhaSpectra API.

* The URL comes from ``sqlalchemy.url`` when set (programmatic use) or from the app settings
  (``MUKHA_DATABASE_URL``).
* ``app.migrations.upgrade_to_head`` passes an open connection in ``config.attributes``.
* SQLite uses batch mode so ALTER-style migrations work.
"""

from __future__ import annotations

from logging.config import fileConfig

from alembic import context
from sqlalchemy.engine import Connection

import app.models  # noqa: F401 - registers every table on Base.metadata
from app.db import Base, create_db_engine, is_sqlite
from app.migrations import prepare_database
from app.settings import get_settings

config = context.config

_configure_logging = config.attributes.get("configure_logger", True)
if (
    config.config_file_name is not None
    and config.attributes.get("connection") is None
    and _configure_logging
):
    fileConfig(config.config_file_name, disable_existing_loggers=False)

target_metadata = Base.metadata


def _database_url() -> str:
    return config.get_main_option("sqlalchemy.url") or get_settings().database_url


def _configure(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        render_as_batch=connection.dialect.name == "sqlite",
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_offline() -> None:
    url = _database_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        render_as_batch=is_sqlite(url),
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connection = config.attributes.get("connection")
    if connection is not None:
        _configure(connection)
        return
    url = _database_url()
    prepare_database(url)
    engine = create_db_engine(url)
    try:
        with engine.connect() as conn:
            _configure(conn)
    finally:
        engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
