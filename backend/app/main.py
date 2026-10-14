"""FastAPI application factory. ``app.main:app`` is the module-level instance for uvicorn:

    uvicorn app.main:app --reload --app-dir backend

Importing this module has no filesystem side effects; the database folder, migrations and the
storage root are prepared when the app starts (lifespan).
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app import __version__
from app.db import create_db_engine, create_session_factory
from app.errors import install_error_handlers
from app.migrations import prepare_database, upgrade_to_head
from app.routers import API_PREFIX, api_router
from app.schemas import UploadManifest
from app.services import build_services
from app.services.pipeline_bridge import PipelineBridge
from app.services.reports import ReportRenderer
from app.settings import Settings, get_settings
from app.storage import Storage
from mukha_pipeline.schema.models import DISCLAIMER, PROTOTYPE_BANNER

logger = logging.getLogger("app")

API_TITLE = "MukhaSpectra API"
API_DESCRIPTION = f"""**{PROTOTYPE_BANNER}.**

MukhaSpectra is a point-of-care oral **screening and triage aid**. It flags suspicious findings
and assigns a risk level (LOW, MODERATE, HIGH or INCONCLUSIVE) with the reasons behind it, and
recommends an action such as re-examination or referral for clinical examination / biopsy.

> {DISCLAIMER}

Model status, stated honestly: the rule-based triage is a baseline heuristic (unvalidated); the
spectral classifier is a placeholder heuristic (not validated); the white-light classifier is not
used in triage until trained weights and a model card are installed. No accuracy figure is
claimed. Results from synthetic data are labelled SYNTHETIC.

Every result records the software, pipeline, schema, model and rule-config versions (with the
SHA-256 of each config file). Re-analysis never overwrites a result: it adds a new version.
"""

OPENAPI_TAGS = [
    {"name": "system", "description": "Health, versions, schema and site catalogue."},
    {"name": "screenings", "description": "Upload, results, re-analysis, artifacts, reports."},
    {"name": "demo", "description": "SYNTHETIC demo patients (labelled synthetic everywhere)."},
]


class SecurityHeadersMiddleware:
    """Conservative headers on every response (pure ASGI, streaming-safe)."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                headers.setdefault("X-Content-Type-Options", "nosniff")
                headers.setdefault("Referrer-Policy", "no-referrer")
                headers.setdefault("X-Frame-Options", "DENY")
                headers.setdefault(
                    "X-MukhaSpectra-Notice", "Research prototype - not for clinical use"
                )
            await send(message)

        await self.app(scope, receive, send_with_headers)


def _install_openapi(app: FastAPI) -> None:
    """Add the UploadManifest schema (the upload form is parsed by hand, so FastAPI cannot
    discover it) to the generated OpenAPI document."""

    def openapi() -> dict[str, Any]:
        if app.openapi_schema:
            return app.openapi_schema
        schema = get_openapi(
            title=app.title,
            version=app.version,
            description=app.description,
            routes=app.routes,
            tags=app.openapi_tags,
        )
        manifest = UploadManifest.model_json_schema(ref_template="#/components/schemas/{model}")
        components = schema.setdefault("components", {}).setdefault("schemas", {})
        for name, definition in manifest.pop("$defs", {}).items():
            components.setdefault(name, definition)
        components["UploadManifest"] = manifest
        app.openapi_schema = schema
        return schema

    app.openapi = openapi  # type: ignore[method-assign]


def create_app(
    settings: Settings | None = None,
    *,
    bridge: PipelineBridge | None = None,
    report_renderer: ReportRenderer | None = None,
    storage: Storage | None = None,
) -> FastAPI:
    """Build an app instance. Tests pass their own settings (temp DB + storage) and fakes."""
    settings = settings or get_settings()
    logging.getLogger("app").setLevel(settings.log_level)

    engine = create_db_engine(settings.database_url)
    services = build_services(
        settings,
        create_session_factory(engine),
        storage=storage,
        bridge=bridge,
        report_renderer=report_renderer,
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        prepare_database(settings.database_url)
        if settings.auto_migrate:
            await run_in_threadpool(upgrade_to_head, engine)
        logger.info("MukhaSpectra API %s ready (%s)", __version__, PROTOTYPE_BANNER)
        try:
            yield
        finally:
            engine.dispose()

    app = FastAPI(
        title=API_TITLE,
        version=__version__,
        description=API_DESCRIPTION,
        openapi_tags=OPENAPI_TAGS,
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.engine = engine
    app.state.services = services

    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["Content-Disposition", "X-Total-Count"],
    )
    install_error_handlers(app)
    app.include_router(api_router, prefix=API_PREFIX)
    _install_openapi(app)
    return app


app = create_app()
