"""Erasure of a screening or of a whole patient (DPDP Act 2023: withdrawal of consent).

Erasing a screening removes, in this order:
1. its database rows: reports, analysis results, captures and the screening itself, plus
   calibration records that only this screening used (demo calibrations);
2. every stored object under ``screenings/{id}/`` (inputs, result artifacts and the cached
   clinician PDFs, which contain the patient's name) and those calibration files;
3. the patient row and the identity (name, external id) when no other screening uses them.

A ``screening_erased`` audit event (and ``patient_erased`` for a patient request) records counts
only, never a name, an external id or a file name. Earlier audit events stay: they hold no PII.

The per-screening analysis lock is held while a screening is erased, so an analysis cannot write
new artifacts for it at the same time, and an analysis waiting for the lock finds the screening
gone. Each screening is committed separately; a failed request can simply be repeated.
"""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.errors import ConflictError, NotFoundError
from app.models import (
    AnalysisResult,
    AuditEvent,
    CalibrationRecord,
    Capture,
    Patient,
    PatientIdentity,
    Report,
    Screening,
)
from app.schemas import ErasureResponse
from app.services.analysis import AnalysisService
from app.services.audit import AuditAction, record_event
from app.services.records import SCREENING_NOT_FOUND
from app.storage import Storage, join_key

logger = logging.getLogger("app.erasure")

PATIENT_NOT_FOUND = "Patient not found."
ERASE_LOCK_TIMEOUT_S = 120
ANALYSIS_RUNNING = "This screening is being analysed; try the erasure again when it has finished."


class ErasureService:
    def __init__(self, storage: Storage, analysis: AnalysisService) -> None:
        self.storage = storage
        self.analysis = analysis

    # -- public --------------------------------------------------------------------------------

    def erase_screening(self, session: Session, screening_id: str) -> ErasureResponse:
        screening = session.get(Screening, screening_id)
        if screening is None:
            raise NotFoundError(SCREENING_NOT_FOUND)
        pseudonymous_id = screening.pseudonymous_id
        objects, event = self._erase_one(session, screening)
        patient_erased = self._erase_patient_if_unused(session, pseudonymous_id)
        event.detail = {**event.detail, "patient_record_erased": patient_erased}
        session.commit()
        return ErasureResponse(
            erased_screening_ids=[screening_id],
            patient_record_erased=patient_erased,
            stored_objects_deleted=objects,
        )

    def erase_patient(self, session: Session, pseudonymous_id: str) -> ErasureResponse:
        patient = session.get(Patient, pseudonymous_id)
        if patient is None:
            raise NotFoundError(PATIENT_NOT_FOUND)
        ids = list(
            session.scalars(
                select(Screening.id)
                .where(Screening.pseudonymous_id == pseudonymous_id)
                .order_by(Screening.created_at, Screening.id)
            )
        )
        objects = 0
        for screening_id in ids:
            screening = session.get(Screening, screening_id)
            if screening is not None:
                erased, _event = self._erase_one(session, screening)
                objects += erased
                session.commit()
        if not self._erase_patient_if_unused(session, pseudonymous_id):  # pragma: no cover
            raise ConflictError("A new screening was added for this patient; try again.")
        record_event(
            session,
            AuditAction.PATIENT_ERASED,
            "patient",
            pseudonymous_id,
            {"screenings": len(ids), "stored_objects_deleted": objects},
        )
        session.commit()
        return ErasureResponse(
            erased_screening_ids=ids, patient_record_erased=True, stored_objects_deleted=objects
        )

    # -- helpers -------------------------------------------------------------------------------

    def _erase_one(self, session: Session, screening: Screening) -> tuple[int, AuditEvent]:
        """Delete one screening's rows and files (no commit).

        Returns the number of stored objects deleted and the (uncommitted) audit event.
        """
        sid = screening.id
        lock = self.analysis.lock_for(sid)
        if not lock.acquire(timeout=ERASE_LOCK_TIMEOUT_S):
            raise ConflictError(ANALYSIS_RUNNING)
        try:
            session.refresh(screening)
            counts = {
                "captures": _count(session, Capture, Capture.screening_id == sid),
                "result_versions": _count(
                    session, AnalysisResult, AnalysisResult.screening_id == sid
                ),
                "reports": _count(session, Report, Report.screening_id == sid),
            }
            calibrations = self._exclusive_calibrations(session, screening)
            for report in session.scalars(select(Report).where(Report.screening_id == sid)):
                session.delete(report)
            session.flush()
            session.delete(screening)  # captures and results cascade (ORM + ON DELETE CASCADE)
            for record in calibrations:
                session.delete(record)
            session.flush()

            objects = self.storage.delete_prefix(join_key("screenings", sid) + "/")
            for record in calibrations:
                if record.storage_key:
                    objects += self.storage.delete_prefix(record.storage_key)
            detail: dict[str, Any] = {
                **counts,
                "calibration_records": len(calibrations),
                "stored_objects_deleted": objects,
                "source": screening.source,
                "synthetic": bool(screening.synthetic),
            }
            event = record_event(session, AuditAction.SCREENING_ERASED, "screening", sid, detail)
            session.flush()
        finally:
            lock.release()
        self.analysis.forget_lock(sid)
        logger.info("Screening %s erased (%d stored objects)", sid, objects)
        return objects, event

    @staticmethod
    def _exclusive_calibrations(session: Session, screening: Screening) -> list[CalibrationRecord]:
        """Calibration records referenced by this screening and by no other one."""
        own = set(screening.calibration_ids or [])
        if not own:
            return []
        others = session.scalars(
            select(Screening.calibration_ids).where(Screening.id != screening.id)
        )
        for ids in others:
            own -= set(ids or [])
            if not own:
                return []
        return list(session.scalars(select(CalibrationRecord).where(CalibrationRecord.id.in_(own))))

    @staticmethod
    def _erase_patient_if_unused(session: Session, pseudonymous_id: str) -> bool:
        if _count(session, Screening, Screening.pseudonymous_id == pseudonymous_id):
            return False
        for identity in session.scalars(
            select(PatientIdentity).where(PatientIdentity.pseudonymous_id == pseudonymous_id)
        ):
            session.delete(identity)
        patient = session.get(Patient, pseudonymous_id)
        if patient is not None:
            session.delete(patient)
        session.flush()
        return True


def _count(session: Session, model: Any, condition: Any) -> int:
    return int(session.scalar(select(func.count()).select_from(model).where(condition)) or 0)
