"""Database engine, declarative base and session helpers (SQLAlchemy 2)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from fastapi import Request
from sqlalchemy import MetaData, create_engine, event
from sqlalchemy.engine import Engine, make_url
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from sqlalchemy.pool import StaticPool

# Deterministic constraint names: needed for Alembic batch migrations on SQLite and keeps
# PostgreSQL constraint names stable across environments.
NAMING_CONVENTION = {
    "ix": "ix_%(table_name)s_%(column_0_N_name)s",
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)


def is_sqlite(url: str) -> bool:
    return make_url(url).get_backend_name() == "sqlite"


def _sqlite_on_connect(dbapi_connection: Any, _record: Any) -> None:
    cursor = dbapi_connection.cursor()
    try:
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA busy_timeout=5000")
    finally:
        cursor.close()


def create_db_engine(url: str) -> Engine:
    """Create an engine for ``url``. SQLite gets foreign keys ON and cross-thread access."""
    parsed = make_url(url)
    if parsed.get_backend_name() == "sqlite":
        database = parsed.database
        kwargs: dict[str, Any] = {"connect_args": {"check_same_thread": False}}
        if database in (None, "", ":memory:"):
            kwargs["poolclass"] = StaticPool
        else:
            Path(database).expanduser().parent.mkdir(parents=True, exist_ok=True)
        engine = create_engine(url, **kwargs)
        event.listen(engine, "connect", _sqlite_on_connect)
        return engine
    return create_engine(url, pool_pre_ping=True)


def create_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, expire_on_commit=False, autoflush=True)


def get_session(request: Request) -> Iterator[Session]:
    """FastAPI dependency: one session per request, rolled back on error, always closed."""
    session: Session = request.app.state.services.session_factory()
    try:
        yield session
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
