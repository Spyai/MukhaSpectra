"""Public input types of the report engine.

The ``ScreeningResult`` is the only clinical input. Identity (name, external id) is passed
separately at render time so the stored result stays pseudonymous (India DPDP Act: collect and
show only what the purpose needs).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime

ArtifactReader = Callable[[str], bytes | None]
"""Returns the bytes stored under an artifact key (e.g. ``sites/<site>/heatmap_overlay.png``),
or ``None`` when the artifact does not exist. Exceptions are caught and treated as missing."""


@dataclass
class PatientIdentity:
    """Personally identifying details, joined from the separate PII store only for rendering.

    ``external_id`` (e.g. an ABHA or Aadhaar-linked number) is always printed masked, keeping
    only its last 4 characters."""

    name: str | None = None
    external_id: str | None = None


@dataclass
class Facility:
    """A referral facility shown on the report (configured in ``app/config/facilities.yaml``).

    ``levels`` limits the facility to results at those risk levels (``HIGH``, ``MODERATE``,
    ``LOW``, ``INCONCLUSIVE``); empty means every level. ``capabilities`` are short tags printed
    with the facility (e.g. "biopsy", "oral medicine")."""

    name: str
    address: str
    phone: str | None = None
    hours: str | None = None
    notes: str | None = None
    example: bool = False
    levels: list[str] = field(default_factory=list)
    capabilities: list[str] = field(default_factory=list)


@dataclass
class ProgrammeContact:
    """Who a patient or clinician calls about the screening programme."""

    name: str
    phone: str | None = None


@dataclass
class ReportOptions:
    report_id: str
    generated_at: datetime
    record_url: str | None = None
    facilities: list[Facility] = field(default_factory=list)
    report_version: int = 1
    display_tz: str = "Asia/Kolkata"
    """IANA time zone for every printed date and time (the reader's local time)."""
    programme_contact: ProgrammeContact | None = None
