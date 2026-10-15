"""Patients: erasure of everything stored about one patient (DPDP Act 2023)."""

from __future__ import annotations

from fastapi import APIRouter
from starlette.concurrency import run_in_threadpool

from app.deps import ServicesDep, SessionDep
from app.schemas import ErasureResponse, ErrorResponse

router = APIRouter(tags=["patients"])


@router.delete(
    "/patients/{pseudonymous_id}",
    response_model=ErasureResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Unknown patient"},
        409: {"model": ErrorResponse, "description": "An analysis is running; try again"},
    },
    summary="Erase a patient: every screening, stored file, cached report and the identity",
)
async def erase_patient(
    pseudonymous_id: str, services: ServicesDep, session: SessionDep
) -> ErasureResponse:
    return await run_in_threadpool(services.erasure.erase_patient, session, pseudonymous_id)
