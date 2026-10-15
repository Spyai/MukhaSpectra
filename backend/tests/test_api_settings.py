"""Settings: defaults, MUKHA_ environment overrides and caching."""

from __future__ import annotations

from pathlib import Path

from app.settings import BACKEND_DIR, DEFAULT_CORS_ORIGINS, Settings, get_settings


def test_defaults(monkeypatch):
    for name in ("MUKHA_DATABASE_URL", "MUKHA_STORAGE_DIR", "MUKHA_CORS_ORIGINS"):
        monkeypatch.delenv(name, raising=False)
    s = Settings()
    expected_db = (BACKEND_DIR / "var" / "mukha.db").as_posix()
    assert s.database_url == f"sqlite:///{expected_db}"
    assert Path(expected_db).is_absolute()
    assert s.storage_dir == (BACKEND_DIR / "var" / "storage").resolve()
    assert s.public_base_url == "http://localhost:5173"
    assert s.cors_origins == list(DEFAULT_CORS_ORIGINS)
    assert s.wl_model_dir is None and s.facilities_file is None
    assert s.max_upload_mb == 40 and s.max_upload_bytes == 40 * 1024 * 1024
    assert s.enable_demo is True
    assert s.auto_migrate is True


def test_environment_overrides(monkeypatch, tmp_path):
    monkeypatch.setenv("MUKHA_DATABASE_URL", "postgresql+psycopg://u:p@db/mukha")
    monkeypatch.setenv("MUKHA_STORAGE_DIR", str(tmp_path / "objects"))
    monkeypatch.setenv("MUKHA_ENABLE_DEMO", "false")
    monkeypatch.setenv("MUKHA_MAX_UPLOAD_MB", "7")
    monkeypatch.setenv("MUKHA_PUBLIC_BASE_URL", "https://mukha.example/")
    monkeypatch.setenv("MUKHA_CORS_ORIGINS", "https://a.example, https://b.example")
    monkeypatch.setenv("MUKHA_LOG_LEVEL", "debug")
    monkeypatch.setenv("MUKHA_WL_MODEL_DIR", str(tmp_path / "model"))
    s = Settings()
    assert s.database_url.startswith("postgresql+psycopg://")
    assert s.storage_dir == (tmp_path / "objects").resolve()
    assert s.enable_demo is False
    assert s.max_upload_bytes == 7 * 1024 * 1024
    assert s.public_base_url == "https://mukha.example"
    assert s.cors_origins == ["https://a.example", "https://b.example"]
    assert s.log_level == "DEBUG"
    assert s.wl_model_dir == tmp_path / "model"


def test_cors_origins_accept_json_and_empty(monkeypatch):
    monkeypatch.setenv("MUKHA_CORS_ORIGINS", '["https://only.example"]')
    assert Settings().cors_origins == ["https://only.example"]
    monkeypatch.setenv("MUKHA_CORS_ORIGINS", "")
    assert Settings().cors_origins == []


def test_get_settings_is_cached(monkeypatch):
    get_settings.cache_clear()
    try:
        monkeypatch.setenv("MUKHA_MAX_UPLOAD_MB", "11")
        first = get_settings()
        monkeypatch.setenv("MUKHA_MAX_UPLOAD_MB", "12")
        assert get_settings() is first
        assert first.max_upload_mb == 11
    finally:
        get_settings.cache_clear()
