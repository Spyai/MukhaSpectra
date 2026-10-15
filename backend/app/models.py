"""Database models (Phase 1 subset of the data model in docs/architecture.md §6).

Privacy (India DPDP Act aware): clinical and imaging tables key patients by a random
``pseudonymous_id`` only. Name and external identifiers live in ``patient_identities`` and are
read only when a clinician report is rendered. Nothing in ``screenings``, ``captures``,
``analysis_results`` or ``audit_events`` may hold a patient's name or external id.

Column types are generic (``JSON``, ``String``, ``DateTime(timezone=True)``) so the same models
run on SQLite in development and PostgreSQL in production.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base


def utcnow() -> datetime:
    return datetime.now(UTC)


def new_id() -> str:
    return str(uuid.uuid4())


def as_utc(value: datetime | None) -> datetime | None:
    """SQLite returns naive datetimes; every timestamp we store is UTC."""
    if value is None:
        return None
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


class ScreeningStatus:
    RECEIVED = "received"  # captures stored, no analysis result yet
    ANALYSED = "analysed"
    ANALYSIS_FAILED = "analysis_failed"


class Patient(Base):
    __tablename__ = "patients"

    pseudonymous_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    age_years: Mapped[int | None] = mapped_column(Integer, nullable=True)
    sex: Mapped[str] = mapped_column(String(16), default="undisclosed")

    identity: Mapped[PatientIdentity | None] = relationship(
        back_populates="patient", uselist=False, cascade="all, delete-orphan"
    )
    screenings: Mapped[list[Screening]] = relationship(back_populates="patient")


class PatientIdentity(Base):
    """PII, kept apart from clinical data and joined only at report-render time."""

    __tablename__ = "patient_identities"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    pseudonymous_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("patients.pseudonymous_id", ondelete="CASCADE"),
        unique=True,
    )
    name: Mapped[str | None] = mapped_column(String(200), nullable=True)
    external_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    patient: Mapped[Patient] = relationship(back_populates="identity")


class Screening(Base):
    __tablename__ = "screenings"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    pseudonymous_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("patients.pseudonymous_id"), index=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, index=True
    )
    # Clock used to judge calibration validity; re-analysis reuses it so a calibration that was
    # valid when the captures were taken is still treated as valid later.
    captured_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    source: Mapped[str] = mapped_column(String(16), default="upload")  # upload | demo
    data_provenance: Mapped[str] = mapped_column(String(16))
    synthetic: Mapped[bool] = mapped_column(Boolean, default=False)
    screening_site_name: Mapped[str | None] = mapped_column(String(200), nullable=True)
    camp_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    health_worker_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    health_worker_name: Mapped[str | None] = mapped_column(String(200), nullable=True)
    language: Mapped[str] = mapped_column(String(8), default="en")
    consent: Mapped[dict[str, Any]] = mapped_column(JSON)
    risk_factors: Mapped[dict[str, Any]] = mapped_column(JSON)
    context: Mapped[dict[str, Any]] = mapped_column(JSON)
    # Per-site inputs needed to rebuild a ScreeningInput for re-analysis:
    # [{site, role, visual_flag, diascopy, px_per_mm, device_id, firmware_version,
    #   captures: {wl, af, dark}}] where captures hold Capture ids.
    site_inputs: Mapped[list[dict[str, Any]]] = mapped_column(JSON)
    calibration_ids: Mapped[list[str]] = mapped_column(JSON, default=list)
    status: Mapped[str] = mapped_column(String(32), default=ScreeningStatus.RECEIVED)

    patient: Mapped[Patient] = relationship(back_populates="screenings")
    captures: Mapped[list[Capture]] = relationship(
        back_populates="screening", cascade="all, delete-orphan", order_by="Capture.created_at"
    )
    results: Mapped[list[AnalysisResult]] = relationship(
        back_populates="screening",
        cascade="all, delete-orphan",
        order_by="AnalysisResult.version",
    )


class Capture(Base):
    __tablename__ = "captures"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    screening_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("screenings.id", ondelete="CASCADE"), index=True
    )
    site: Mapped[str | None] = mapped_column(String(48), nullable=True)  # None for spectral files
    kind: Mapped[str] = mapped_column(String(24))  # CaptureKind value
    storage_key: Mapped[str] = mapped_column(String(512), unique=True)
    sha256: Mapped[str] = mapped_column(String(64))
    size_bytes: Mapped[int] = mapped_column(Integer)
    mime_type: Mapped[str | None] = mapped_column(String(100), nullable=True)
    # Neutral, generated name (e.g. "buccal_mucosa_left-white_light.jpg"); never the uploaded
    # file name, which may contain the patient's name.
    file_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    meta: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    screening: Mapped[Screening] = relationship(back_populates="captures")


class AnalysisResult(Base):
    """One immutable ScreeningResult version. Re-analysis adds version n+1."""

    __tablename__ = "analysis_results"
    __table_args__ = (UniqueConstraint("screening_id", "version"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True)  # = ScreeningResult.result_id
    screening_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("screenings.id", ondelete="CASCADE"), index=True
    )
    version: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    overall_risk_level: Mapped[str] = mapped_column(String(16))
    pipeline_version: Mapped[str] = mapped_column(String(32))
    rule_config_version: Mapped[str] = mapped_column(String(32))
    rule_config_sha256: Mapped[str] = mapped_column(String(64))
    synthetic: Mapped[bool] = mapped_column(Boolean, default=False)
    result: Mapped[dict[str, Any]] = mapped_column(JSON)

    screening: Mapped[Screening] = relationship(back_populates="results")


class Report(Base):
    __tablename__ = "reports"
    __table_args__ = (UniqueConstraint("analysis_result_id", "kind"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    screening_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("screenings.id", ondelete="CASCADE"), index=True
    )
    analysis_result_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("analysis_results.id", ondelete="CASCADE")
    )
    version: Mapped[int] = mapped_column(Integer)
    kind: Mapped[str] = mapped_column(String(32))  # "clinician" (patient report: Phase 2)
    storage_key: Mapped[str] = mapped_column(String(512))
    sha256: Mapped[str] = mapped_column(String(64))
    size_bytes: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class CalibrationRecord(Base):
    __tablename__ = "calibration_records"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    device_id: Mapped[str] = mapped_column(String(100), index=True)
    kind: Mapped[str] = mapped_column(String(16))  # "image" | "spectral"
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    synthetic: Mapped[bool] = mapped_column(Boolean, default=False)
    data: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    storage_key: Mapped[str | None] = mapped_column(String(512), nullable=True)


class AuditEvent(Base):
    """Append-only log of who did what. ``detail`` must never contain PII."""

    __tablename__ = "audit_events"
    __table_args__ = (Index("ix_audit_events_entity", "entity_type", "entity_id"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    actor: Mapped[str] = mapped_column(String(100))
    action: Mapped[str] = mapped_column(String(64), index=True)
    entity_type: Mapped[str] = mapped_column(String(32))
    entity_id: Mapped[str] = mapped_column(String(64))
    detail: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
