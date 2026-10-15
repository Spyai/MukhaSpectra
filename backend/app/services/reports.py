"""Clinician report generation, caching and the oral site map.

The report engine itself lives in ``app.reports`` (owned separately). This service joins the
PII (``patient_identities``) only at render time, reads artifacts of the exact result version
from storage, and stores each clinician PDF once per result version under
``screenings/{id}/reports/v{n}/clinician.pdf``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.errors import ApiError, ServiceUnavailableError
from app.models import AnalysisResult, PatientIdentity, Report, Screening, utcnow
from app.services.audit import AuditAction, record_event
from app.services.inputs import sha256_hex
from app.settings import Settings
from app.storage import ObjectNotFoundError, Storage, join_key, validate_key
from mukha_pipeline.schema.models import ScreeningResult

logger = logging.getLogger("app.reports_service")

CLINICIAN_REPORT = "clinician"
REPORT_ENGINE_UNAVAILABLE = "The report engine is not available on this server."
REPORT_FAILED = "The report could not be generated. The analysis result is unaffected."


class ReportRenderer(Protocol):
    """What the API needs from a report engine (see ``AppReportsRenderer``)."""

    def render_pdf(
        self,
        result: ScreeningResult,
        artifacts: Callable[[str], bytes | None],
        *,
        identity: Any,
        options: Any,
    ) -> bytes: ...

    def render_html(
        self,
        result: ScreeningResult,
        artifacts: Callable[[str], bytes | None],
        *,
        identity: Any,
        options: Any,
    ) -> str: ...

    def render_site_map_svg(
        self, result: ScreeningResult, *, width: int, show_legend: bool
    ) -> str: ...

    def load_facilities(self, path: Path | None) -> list[Any]: ...

    def make_identity(self, *, name: str | None, external_id: str | None) -> Any: ...

    def make_options(
        self,
        *,
        report_id: str,
        generated_at: datetime,
        record_url: str,
        facilities: list[Any],
        report_version: int,
    ) -> Any: ...


class AppReportsRenderer:
    """Default renderer: delegates to ``app.reports`` (imported lazily)."""

    def render_pdf(
        self,
        result: ScreeningResult,
        artifacts: Callable[[str], bytes | None],
        *,
        identity: Any,
        options: Any,
    ) -> bytes:
        from app.reports import render_clinician_report_pdf

        return render_clinician_report_pdf(result, artifacts, identity=identity, options=options)

    def render_html(
        self,
        result: ScreeningResult,
        artifacts: Callable[[str], bytes | None],
        *,
        identity: Any,
        options: Any,
    ) -> str:
        from app.reports import render_clinician_report_html

        return render_clinician_report_html(result, artifacts, identity=identity, options=options)

    def render_site_map_svg(self, result: ScreeningResult, *, width: int, show_legend: bool) -> str:
        from app.reports import render_site_map_svg

        return render_site_map_svg(result, width=width, show_legend=show_legend)

    def load_facilities(self, path: Path | None) -> list[Any]:
        from app.reports import load_facilities

        return list(load_facilities(path))

    def make_identity(self, *, name: str | None, external_id: str | None) -> Any:
        from app.reports import PatientIdentity as ReportIdentity

        return ReportIdentity(name=name, external_id=external_id)

    def make_options(
        self,
        *,
        report_id: str,
        generated_at: datetime,
        record_url: str,
        facilities: list[Any],
        report_version: int,
    ) -> Any:
        from app.reports import ReportOptions

        return ReportOptions(
            report_id=report_id,
            generated_at=generated_at,
            record_url=record_url,
            facilities=facilities,
            report_version=report_version,
        )


class ArtifactReader:
    """``Callable[[str], bytes | None]`` over one result version's artifacts in storage."""

    def __init__(self, storage: Storage, screening_id: str, version: int) -> None:
        self.storage = storage
        self.prefix = join_key("screenings", screening_id, "results", f"v{version}") + "/"

    def key_for(self, key: str) -> str:
        """Full storage key for an artifact key (raises ValueError for unsafe keys)."""
        validate_key(key)
        return self.prefix + key

    def __call__(self, key: str) -> bytes | None:
        try:
            return self.storage.get(self.key_for(key))
        except (ValueError, ObjectNotFoundError):
            return None


def report_display_id(screening_id: str, version: int) -> str:
    return f"MSR-{screening_id.replace('-', '')[:10].upper()}-V{version}"


def report_file_name(screening_id: str, version: int, *, synthetic: bool = False) -> str:
    """Download name of the clinician PDF; synthetic reports are labelled in the name too."""
    prefix = "SYNTHETIC_" if synthetic else ""
    return f"{prefix}MukhaSpectra_{screening_id}_v{version}.pdf"


class ReportService:
    def __init__(self, settings: Settings, storage: Storage, renderer: ReportRenderer) -> None:
        self.settings = settings
        self.storage = storage
        self.renderer = renderer

    def record_url(self, screening_id: str) -> str:
        return f"{self.settings.public_base_url}/screenings/{screening_id}"

    def _render_context(
        self, session: Session, screening: Screening, row: AnalysisResult, generated_at: datetime
    ) -> tuple[ScreeningResult, ArtifactReader, Any, Any]:
        result = ScreeningResult.model_validate(row.result)
        identity_row = session.scalar(
            select(PatientIdentity).where(
                PatientIdentity.pseudonymous_id == screening.pseudonymous_id
            )
        )
        identity = self.renderer.make_identity(
            name=identity_row.name if identity_row else None,
            external_id=identity_row.external_id if identity_row else None,
        )
        options = self.renderer.make_options(
            report_id=report_display_id(screening.id, row.version),
            generated_at=generated_at,
            record_url=self.record_url(screening.id),
            facilities=self.renderer.load_facilities(self.settings.facilities_file),
            report_version=row.version,
        )
        return result, ArtifactReader(self.storage, screening.id, row.version), identity, options

    def _guarded(self, fn: Callable[[], Any]) -> Any:
        try:
            return fn()
        except ImportError as exc:
            logger.exception("Report engine import failed")
            raise ServiceUnavailableError(REPORT_ENGINE_UNAVAILABLE) from exc
        except ApiError:
            raise
        except Exception as exc:
            logger.exception("Report rendering failed")
            raise ApiError(REPORT_FAILED, status_code=500) from exc

    def clinician_pdf(self, session: Session, screening: Screening, row: AnalysisResult) -> bytes:
        """The clinician PDF for one result version, rendered once and then served from storage."""
        existing = session.scalar(
            select(Report).where(
                Report.analysis_result_id == row.id, Report.kind == CLINICIAN_REPORT
            )
        )
        if existing is not None and self.storage.exists(existing.storage_key):
            pdf = self.storage.get(existing.storage_key)
            record_event(
                session,
                AuditAction.REPORT_VIEWED,
                "report",
                existing.id,
                {"screening_id": screening.id, "result_version": row.version, "format": "pdf"},
            )
            session.commit()
            return pdf

        generated_at = utcnow()

        def render() -> bytes:
            result, reader, identity, options = self._render_context(
                session, screening, row, generated_at
            )
            return bytes(
                self.renderer.render_pdf(result, reader, identity=identity, options=options)
            )

        pdf = self._guarded(render)
        key = join_key("screenings", screening.id, "reports", f"v{row.version}", "clinician.pdf")
        self.storage.put(key, pdf)
        report = existing or Report(
            screening_id=screening.id,
            analysis_result_id=row.id,
            version=row.version,
            kind=CLINICIAN_REPORT,
        )
        report.storage_key = key
        report.sha256 = sha256_hex(pdf)
        report.size_bytes = len(pdf)
        report.created_at = generated_at
        session.add(report)
        try:
            session.flush()
        except IntegrityError:
            # A concurrent request stored the same report first; our file replaced theirs, so
            # point their row at the bytes now in storage.
            session.rollback()
            report = session.scalars(
                select(Report).where(
                    Report.analysis_result_id == row.id, Report.kind == CLINICIAN_REPORT
                )
            ).one()
            report.sha256 = sha256_hex(pdf)
            report.size_bytes = len(pdf)
        detail = {"screening_id": screening.id, "result_version": row.version}
        record_event(
            session,
            AuditAction.REPORT_GENERATED,
            "report",
            report.id,
            {**detail, "kind": CLINICIAN_REPORT, "sha256": report.sha256},
        )
        record_event(
            session, AuditAction.REPORT_VIEWED, "report", report.id, {**detail, "format": "pdf"}
        )
        session.commit()
        return pdf

    def clinician_html(self, session: Session, screening: Screening, row: AnalysisResult) -> str:
        """Same report as HTML for preview/debugging (rendered on demand, not stored)."""

        def render() -> str:
            result, reader, identity, options = self._render_context(
                session, screening, row, utcnow()
            )
            return str(
                self.renderer.render_html(result, reader, identity=identity, options=options)
            )

        html = self._guarded(render)
        record_event(
            session,
            AuditAction.REPORT_VIEWED,
            "screening",
            screening.id,
            {"result_version": row.version, "format": "html", "kind": CLINICIAN_REPORT},
        )
        session.commit()
        return html

    def site_map_svg(self, row: AnalysisResult, *, width: int, show_legend: bool) -> str:
        result = ScreeningResult.model_validate(row.result)
        return str(
            self._guarded(
                lambda: self.renderer.render_site_map_svg(
                    result, width=width, show_legend=show_legend
                )
            )
        )
