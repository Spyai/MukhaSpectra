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
    """Personally identifying details, joined from the separate PII store only for rendering."""

    name: str | None = None
    external_id: str | None = None


@dataclass
class Facility:
    """A referral facility shown on the report (configured in ``app/config/facilities.yaml``)."""

    name: str
    address: str
    phone: str | None = None
    hours: str | None = None
    notes: str | None = None
    example: bool = False


@dataclass
class ReportOptions:
    report_id: str
    generated_at: datetime
    record_url: str | None = None
    facilities: list[Facility] = field(default_factory=list)
    report_version: int = 1
