"""Audit log: who did what to which record, and when.

Phase 1 has no authentication, so the actor is "anonymous"; Phase 5 adds users and roles.
``detail`` must never contain PII (names, external ids, uploaded file names).
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from sqlalchemy.orm import Session

from app.models import AuditEvent

ANONYMOUS_ACTOR = "anonymous"


class AuditAction(StrEnum):
    SCREENING_CREATED = "screening_created"
    ANALYSIS_COMPLETED = "analysis_completed"
    REANALYSIS_COMPLETED = "reanalysis_completed"
    ANALYSIS_FAILED = "analysis_failed"
    RESULT_VIEWED = "result_viewed"
    REPORT_GENERATED = "report_generated"
    REPORT_VIEWED = "report_viewed"


def record_event(
    session: Session,
    action: AuditAction | str,
    entity_type: str,
    entity_id: str,
    detail: dict[str, Any] | None = None,
    *,
    actor: str = ANONYMOUS_ACTOR,
) -> AuditEvent:
    """Add an audit event to the session. The caller's commit persists it."""
    event = AuditEvent(
        actor=actor,
        action=str(action),
        entity_type=entity_type,
        entity_id=entity_id,
        detail=detail or {},
    )
    session.add(event)
    return event
