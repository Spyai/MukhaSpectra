"""Backend API test fixtures: a fresh SQLite DB and storage root per test, fake pipeline and
fake report renderer injected through ``create_app``."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from api_fakes import FakePipeline, FakeRenderer
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.main import create_app
from app.services import Services
from app.settings import Settings


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "real_pipeline: end-to-end test against the real pipeline (skips if missing)"
    )


@pytest.fixture
def api_settings(tmp_path: Path) -> Settings:
    return Settings(
        database_url=f"sqlite:///{(tmp_path / 'db' / 'test.db').as_posix()}",
        storage_dir=tmp_path / "storage",
        public_base_url="https://console.example.test/",
        wl_model_dir=None,
        facilities_file=None,
        enable_demo=True,
        max_upload_mb=5,
        max_request_mb=50,
        log_level="WARNING",
    )


@pytest.fixture
def fake_pipeline() -> FakePipeline:
    return FakePipeline()


@pytest.fixture
def fake_renderer() -> FakeRenderer:
    return FakeRenderer()


@pytest.fixture
def make_client(
    api_settings: Settings, fake_pipeline: FakePipeline, fake_renderer: FakeRenderer
) -> Iterator[Callable[..., TestClient]]:
    """Factory: ``make_client(enable_demo=False)`` builds a client with changed settings."""
    clients: list[TestClient] = []

    def factory(*, fakes: bool = True, **setting_overrides: Any) -> TestClient:
        settings = api_settings.model_copy(update=setting_overrides)
        app = create_app(
            settings,
            bridge=fake_pipeline.bridge() if fakes else None,
            report_renderer=fake_renderer,
        )
        client = TestClient(app)
        client.__enter__()  # run the lifespan (migrations)
        clients.append(client)
        return client

    yield factory
    for client in clients:
        client.__exit__(None, None, None)


@pytest.fixture
def client(make_client: Callable[..., TestClient]) -> TestClient:
    return make_client()


@pytest.fixture
def services(client: TestClient) -> Services:
    return client.app.state.services  # type: ignore[attr-defined]


@pytest.fixture
def db(services: Services) -> Iterator[Session]:
    session = services.session_factory()
    try:
        yield session
    finally:
        session.close()
