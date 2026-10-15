"""Alembic migrations match the models; startup migration; SQLite foreign keys are enforced."""

from __future__ import annotations

import pytest
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import inspect
from sqlalchemy.exc import IntegrityError

from app.db import Base, create_db_engine, create_session_factory
from app.migrations import ALEMBIC_INI, alembic_config, upgrade_to_head
from app.models import Capture
from app.settings import get_settings

EXPECTED_TABLES = {
    "alembic_version",
    "patients",
    "patient_identities",
    "screenings",
    "captures",
    "analysis_results",
    "reports",
    "calibration_records",
    "audit_events",
}


def _head() -> str:
    return ScriptDirectory.from_config(alembic_config()).get_current_head()


def test_upgrade_head_matches_models(tmp_path):
    engine = create_db_engine(f"sqlite:///{tmp_path / 'fresh.db'}")
    upgrade_to_head(engine)
    upgrade_to_head(engine)  # idempotent
    with engine.connect() as conn:
        assert set(inspect(conn).get_table_names()) == EXPECTED_TABLES
        ctx = MigrationContext.configure(conn, opts={"compare_type": True})
        assert ctx.get_current_revision() == _head()
        assert compare_metadata(ctx, Base.metadata) == []
    engine.dispose()


def test_downgrade_to_base_and_back(tmp_path):
    url = f"sqlite:///{tmp_path / 'roundtrip.db'}"
    config = alembic_config(url)
    command.upgrade(config, "head")
    command.downgrade(config, "base")
    engine = create_db_engine(url)
    with engine.connect() as conn:
        assert set(inspect(conn).get_table_names()) == {"alembic_version"}
    command.upgrade(config, "head")
    with engine.connect() as conn:
        assert set(inspect(conn).get_table_names()) == EXPECTED_TABLES
    engine.dispose()


def test_cli_config_reads_the_url_from_settings(tmp_path, monkeypatch):
    db_path = tmp_path / "nested" / "cli.db"
    monkeypatch.setenv("MUKHA_DATABASE_URL", f"sqlite:///{db_path}")
    get_settings.cache_clear()
    try:
        config = Config(str(ALEMBIC_INI))
        config.attributes["configure_logger"] = False  # keep pytest's logging untouched
        command.upgrade(config, "head")
    finally:
        get_settings.cache_clear()
    assert db_path.exists()
    engine = create_db_engine(f"sqlite:///{db_path}")
    with engine.connect() as conn:
        assert MigrationContext.configure(conn).get_current_revision() == _head()
    engine.dispose()


def test_offline_sql_generation(tmp_path, capsys):
    config = alembic_config(f"sqlite:///{tmp_path / 'offline.db'}")
    command.upgrade(config, "head", sql=True)
    sql = capsys.readouterr().out
    assert "CREATE TABLE patient_identities" in sql
    assert "CREATE TABLE audit_events" in sql


def test_app_startup_migrates_its_own_database(client, api_settings):
    engine = client.app.state.engine
    with engine.connect() as conn:
        assert set(inspect(conn).get_table_names()) == EXPECTED_TABLES


def test_auto_migrate_can_be_disabled(make_client, api_settings):
    client = make_client(
        database_url=f"sqlite:///{api_settings.storage_dir.parent / 'no.db'}", auto_migrate=False
    )
    with client.app.state.engine.connect() as conn:
        assert inspect(conn).get_table_names() == []


def test_sqlite_foreign_keys_are_enforced(tmp_path):
    engine = create_db_engine(f"sqlite:///{tmp_path / 'fk.db'}")
    upgrade_to_head(engine)
    with create_session_factory(engine)() as session:
        session.add(
            Capture(
                screening_id="missing",
                kind="white_light",
                storage_key="k",
                sha256="0" * 64,
                size_bytes=1,
                meta={},
            )
        )
        with pytest.raises(IntegrityError):
            session.commit()
    engine.dispose()


def test_in_memory_sqlite_is_supported():
    engine = create_db_engine("sqlite://")
    upgrade_to_head(engine)
    with engine.connect() as conn:
        assert "screenings" in inspect(conn).get_table_names()
    engine.dispose()
