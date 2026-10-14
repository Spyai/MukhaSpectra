"""API routers, mounted under ``/api/v1``."""

from __future__ import annotations

from fastapi import APIRouter

from app.routers import demo, meta, screenings

API_PREFIX = "/api/v1"

api_router = APIRouter()
api_router.include_router(meta.router)
api_router.include_router(screenings.router)
api_router.include_router(demo.router)

__all__ = ["API_PREFIX", "api_router"]
