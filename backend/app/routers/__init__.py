"""API routers, mounted under ``/api/v1``.

Only ``/health`` is open. Every other route depends on :func:`app.security.require_api_key`,
which enforces ``MUKHA_API_KEY`` when it is configured (Phase 5 replaces it with user accounts).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.routers import demo, meta, patients, screenings
from app.security import require_api_key

API_PREFIX = "/api/v1"

api_router = APIRouter()
api_router.include_router(meta.public_router)

protected_router = APIRouter(dependencies=[Depends(require_api_key)])
protected_router.include_router(meta.router)
protected_router.include_router(screenings.router)
protected_router.include_router(patients.router)
protected_router.include_router(demo.router)
api_router.include_router(protected_router)

__all__ = ["API_PREFIX", "api_router"]
