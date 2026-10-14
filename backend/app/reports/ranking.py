"""Lesion ordering shared by the report body and the site map, so lesion numbers match."""

from __future__ import annotations

from app.reports.labels import SEVERITY_RANK
from mukha_pipeline.schema.models import LesionAssessment, RiskLevel, ScreeningResult


def severity_rank(level: RiskLevel | str) -> float:
    return SEVERITY_RANK.get(RiskLevel(level), 0.0)


def ordered_lesions(result: ScreeningResult) -> list[LesionAssessment]:
    """Lesions by severity (HIGH first), then rule score (highest first), then id."""
    return sorted(
        result.lesions,
        key=lambda les: (-severity_rank(les.risk_level), -les.score, les.lesion_id),
    )


def lesion_numbers(result: ScreeningResult) -> dict[str, int]:
    """lesion_id -> 1-based display number."""
    return {les.lesion_id: i for i, les in enumerate(ordered_lesions(result), start=1)}
