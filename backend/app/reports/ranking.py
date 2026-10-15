"""Lesion ordering and per-site flags shared by the report body and the site map, so they match."""

from __future__ import annotations

from app.reports.labels import SEVERITY_RANK
from mukha_pipeline.schema.models import (
    LesionAssessment,
    RiskLevel,
    ScreeningResult,
    SiteResult,
    SiteRole,
    VisualFlag,
)
from mukha_pipeline.sites import OralSite


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


def visual_seen(flag: VisualFlag) -> bool:
    """The health worker recorded something at the site: the checkbox or any appearance.

    Same rule as ``mukha_pipeline.fusion.lesions.visual_seen`` (a recorded appearance counts
    even if the checkbox was missed); kept here so the report engine does not import fusion.
    """
    return flag.lesion_seen or bool(flag.appearance)


def flagged_reference_sites(result: ScreeningResult) -> set[OralSite]:
    """Reference-only sites that must not be presented as an unremarkable comparison.

    A reference site is flagged when the health worker recorded something there, a finding sits
    there, or the pipeline gave the site a MODERATE or HIGH level. The device never checks that
    a reference site is unaffected, so such a site is shown with its level and finding.
    """
    lesion_sites = {OralSite(les.site) for les in result.lesions}
    return {
        OralSite(s.site)
        for s in result.sites
        if is_flagged_reference(s, OralSite(s.site) in lesion_sites)
    }


def is_flagged_reference(site: SiteResult, has_finding: bool) -> bool:
    if site.role != SiteRole.REFERENCE_ONLY:
        return False
    raised = RiskLevel(site.site_risk_level) in (RiskLevel.MODERATE, RiskLevel.HIGH)
    return has_finding or raised or visual_seen(site.visual_flag)
