"""Database helpers shared by the services and routers (creation, lookup, summaries)."""

from __future__ import annotations

import secrets
from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload

from app.errors import NotFoundError
from app.models import (
    AnalysisResult,
    Patient,
    PatientIdentity,
    Screening,
    ScreeningStatus,
    as_utc,
    utcnow,
)
from app.schemas import ResultVersion, ScreeningSummary
from mukha_pipeline.schema.models import DataProvenance, Sex

if TYPE_CHECKING:  # pragma: no cover - typing only
    from mukha_pipeline.types import ScreeningInput, SiteInput

SCREENING_NOT_FOUND = "Screening not found."


def new_pseudonymous_id(*, synthetic: bool = False) -> str:
    """Random, non-derivable patient key (never built from name, phone or external id)."""
    prefix = "SYN" if synthetic else "P"
    return f"{prefix}-{secrets.token_hex(6).upper()}"


def site_input_entry(site_input: SiteInput, refs: dict[str, str | None]) -> dict[str, Any]:
    """What Screening.site_inputs stores for one site (enough to rebuild a SiteInput)."""
    return {
        "site": site_input.site.value,
        "role": site_input.role.value,
        "visual_flag": site_input.visual_flag.model_dump(mode="json"),
        "diascopy": site_input.diascopy.value,
        "px_per_mm": site_input.px_per_mm,
        "device_id": site_input.wl.meta.device_id,
        "firmware_version": site_input.wl.meta.firmware_version,
        "captures": refs,
    }


def create_screening_record(
    session: Session,
    *,
    inp: ScreeningInput,
    source: str,
    site_inputs: list[dict[str, Any]],
    calibration_ids: list[str],
    identity_name: str | None,
    identity_external_id: str | None,
    created_at: datetime | None = None,
) -> Screening:
    """Add Patient (+ separate PatientIdentity when PII is given) and Screening rows."""
    created = created_at or utcnow()
    patient = Patient(
        pseudonymous_id=inp.patient.pseudonymous_id,
        created_at=created,
        age_years=inp.patient.age_years,
        sex=Sex(inp.patient.sex).value,
    )
    session.add(patient)
    if identity_name or identity_external_id:
        session.add(
            PatientIdentity(
                pseudonymous_id=patient.pseudonymous_id,
                name=identity_name,
                external_id=identity_external_id,
                created_at=created,
            )
        )
    context = inp.context
    provenance = DataProvenance(inp.data_provenance)
    screening = Screening(
        id=inp.screening_id,
        pseudonymous_id=patient.pseudonymous_id,
        created_at=created,
        captured_at=as_utc(inp.now) or created,
        source=source,
        data_provenance=provenance.value,
        synthetic=provenance == DataProvenance.SYNTHETIC,
        screening_site_name=context.screening_site_name,
        camp_id=context.camp_id,
        health_worker_id=context.health_worker_id,
        health_worker_name=context.health_worker_name,
        language=context.language.value,
        consent=inp.consent.model_dump(mode="json"),
        risk_factors=inp.risk_factors.model_dump(mode="json"),
        context=context.model_dump(mode="json"),
        site_inputs=site_inputs,
        calibration_ids=calibration_ids,
        status=ScreeningStatus.RECEIVED,
    )
    screening.patient = patient
    session.add(screening)
    return screening


def get_screening(session: Session, screening_id: str) -> Screening:
    screening = session.get(Screening, screening_id)
    if screening is None:
        raise NotFoundError(SCREENING_NOT_FOUND)
    return screening


def result_rows(session: Session, screening_id: str) -> list[AnalysisResult]:
    return list(
        session.scalars(
            select(AnalysisResult)
            .where(AnalysisResult.screening_id == screening_id)
            .order_by(AnalysisResult.version)
        )
    )


def latest_result_row(session: Session, screening_id: str) -> AnalysisResult | None:
    return session.scalars(
        select(AnalysisResult)
        .where(AnalysisResult.screening_id == screening_id)
        .order_by(AnalysisResult.version.desc())
        .limit(1)
    ).first()


def get_result_row(session: Session, screening_id: str, version: int | None) -> AnalysisResult:
    """A specific result version, or the latest when ``version`` is None (404 if absent)."""
    if version is None:
        row = latest_result_row(session, screening_id)
        if row is None:
            raise NotFoundError("This screening has no analysis result yet.")
        return row
    row = session.scalars(
        select(AnalysisResult).where(
            AnalysisResult.screening_id == screening_id, AnalysisResult.version == version
        )
    ).first()
    if row is None:
        raise NotFoundError(f"Result version {version} was not found for this screening.")
    return row


def version_summary(row: AnalysisResult) -> ResultVersion:
    return ResultVersion(
        version=row.version,
        result_id=row.id,
        created_at=as_utc(row.created_at),  # type: ignore[arg-type]
        overall_risk_level=row.overall_risk_level,  # type: ignore[arg-type]
        pipeline_version=row.pipeline_version,
        rule_config_version=row.rule_config_version,
        rule_config_sha256=row.rule_config_sha256,
        synthetic=row.synthetic,
        supersedes_result_id=row.result.get("supersedes_result_id"),
    )


def list_summaries(
    session: Session, *, limit: int, offset: int
) -> tuple[list[ScreeningSummary], int]:
    """Screenings newest first, with their latest result's headline (no PII)."""
    total = session.scalar(select(func.count()).select_from(Screening)) or 0
    screenings = list(
        session.scalars(
            select(Screening)
            .options(selectinload(Screening.patient))
            .order_by(Screening.created_at.desc(), Screening.id.desc())
            .limit(limit)
            .offset(offset)
        )
    )
    latest: dict[str, AnalysisResult] = {}
    if screenings:
        ids = [s.id for s in screenings]
        newest = (
            select(AnalysisResult.screening_id, func.max(AnalysisResult.version).label("v"))
            .where(AnalysisResult.screening_id.in_(ids))
            .group_by(AnalysisResult.screening_id)
            .subquery()
        )
        rows = session.scalars(
            select(AnalysisResult).join(
                newest,
                (AnalysisResult.screening_id == newest.c.screening_id)
                & (AnalysisResult.version == newest.c.v),
            )
        )
        latest = {row.screening_id: row for row in rows}

    items: list[ScreeningSummary] = []
    for s in screenings:
        row = latest.get(s.id)
        result = row.result if row is not None else {}
        items.append(
            ScreeningSummary(
                screening_id=s.id,
                created_at=as_utc(s.created_at),  # type: ignore[arg-type]
                status=s.status,
                source=s.source,
                data_provenance=DataProvenance(s.data_provenance),
                synthetic=s.synthetic,
                pseudonymous_id=s.pseudonymous_id,
                age_years=s.patient.age_years,
                sex=Sex(s.patient.sex),
                screening_site_name=s.screening_site_name,
                camp_id=s.camp_id,
                health_worker_id=s.health_worker_id,
                site_count=len(s.site_inputs or []),
                latest_result_version=row.version if row is not None else None,
                latest_result_id=row.id if row is not None else None,
                overall_risk_level=row.overall_risk_level if row is not None else None,  # type: ignore[arg-type]
                lesion_count=len(result.get("lesions", [])) if row is not None else None,
                review_status=(result.get("triage") or {}).get("review_status"),
            )
        )
    return items, int(total)
