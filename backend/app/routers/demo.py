"""SYNTHETIC demo patients for demonstrations and end-to-end checks."""

from __future__ import annotations

from fastapi import APIRouter

from app.deps import ServicesDep, SessionDep
from app.errors import NotFoundError
from app.schemas import ErrorResponse, ScreeningCreated

router = APIRouter(tags=["demo"])

DEMO_DISABLED = "Demo screenings are disabled on this server."


@router.post(
    "/demo/screenings/{kind}",
    status_code=201,
    response_model=ScreeningCreated,
    responses={
        404: {"model": ErrorResponse, "description": "Demo endpoints are disabled"},
        422: {"model": ErrorResponse, "description": "Unknown demo kind"},
        503: {"model": ErrorResponse, "description": "Server busy, try again shortly"},
    },
    summary="Create and analyse a SYNTHETIC demo patient (low | moderate | high)",
)
def create_demo_screening(
    kind: str, services: ServicesDep, session: SessionDep
) -> ScreeningCreated:
    if not services.settings.enable_demo:
        raise NotFoundError(DEMO_DISABLED)
    screening_id, row, result = services.demo.create(session, kind)
    return ScreeningCreated(screening_id=screening_id, result_version=row.version, result=result)
