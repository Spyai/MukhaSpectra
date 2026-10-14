"""Rule-based fusion and triage: per-site evidence -> lesions -> patient risk level and action.

Status: baseline heuristic, not trained, not clinically validated. The rules live in
``config/triage_rules.v1.yaml``; this module only applies them, transparently. Learned fusion
can later replace ``run_triage`` behind the same ``TriageOutput``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from mukha_pipeline.config import LoadedConfig
from mukha_pipeline.fusion.factors import POINT_DECIMALS, total_points
from mukha_pipeline.fusion.lesions import build_lesion_drafts, site_readable
from mukha_pipeline.fusion.narrative import (
    action_from_template,
    build_limitations,
    dynamic_steps,
    explanation,
    lesion_summary,
)
from mukha_pipeline.fusion.overrides import (
    ClinicalContext,
    matching_overrides,
    override_factor,
    site_appearances,
)
from mukha_pipeline.fusion.patient import (
    decide_level,
    patient_factors,
    risk_factor_factors,
    risk_items,
)
from mukha_pipeline.fusion.rules import TriageRules, parse_triage_rules
from mukha_pipeline.fusion.scoring import ScoredLesion, score_lesion
from mukha_pipeline.schema.models import (
    RISK_ORDER,
    LesionAssessment,
    Limitation,
    PatientInfo,
    RiskFactors,
    RiskLevel,
    ScoreThresholds,
    SiteResult,
    SiteRole,
    SpectralReadingResult,
    TriageResult,
)
from mukha_pipeline.sites import OralSite

# Display / aggregation order. INCONCLUSIVE ranks above LOW (it is never reassuring) and below
# MODERATE (a positive finding is more specific than "could not tell").
SEVERITY: dict[RiskLevel, int] = {
    RiskLevel.LOW: 0,
    RiskLevel.INCONCLUSIVE: 1,
    RiskLevel.MODERATE: 2,
    RiskLevel.HIGH: 3,
}


@dataclass
class TriageOutput:
    lesions: list[LesionAssessment]
    triage: TriageResult
    site_levels: dict[OralSite, RiskLevel]
    limitations: list[Limitation]


def _assessment(s: ScoredLesion, rules: TriageRules) -> LesionAssessment:
    d = s.draft
    return LesionAssessment(
        lesion_id=s.lesion_id,
        site=d.site,
        origins=list(d.origins),
        region_ids=[d.region.region_id] if d.region is not None else [],
        spectral_reading_ids=[r.reading_id for r in d.spectral],
        location=d.region.centroid if d.region is not None else None,
        risk_level=s.level,
        score=s.score,
        contributing_factors=list(s.factors),
        confounders=list(d.region.confounders) if d.region is not None else [],
        recommended_action=action_from_template(s.level, rules),
        summary=lesion_summary(s, rules),
    )


def site_levels(
    sites: Sequence[SiteResult], lesions: Sequence[ScoredLesion]
) -> dict[OralSite, RiskLevel]:
    """Per site: the most severe lesion there; else LOW if readable, INCONCLUSIVE if not."""
    out: dict[OralSite, RiskLevel] = {}

    def raise_to(site: OralSite, level: RiskLevel) -> None:
        current = out.get(site)
        if current is None or SEVERITY[level] > SEVERITY[current]:
            out[site] = level

    by_site: dict[OralSite, list[RiskLevel]] = {}
    for s in lesions:
        by_site.setdefault(s.draft.site, []).append(s.level)
    for site in sites:
        if site.site in by_site:
            continue
        raise_to(site.site, RiskLevel.LOW if site_readable(site) else RiskLevel.INCONCLUSIVE)
    for site, levels in by_site.items():
        for level in levels:
            raise_to(site, level)
    return out


def lesion_rank(s: ScoredLesion) -> tuple[int, float]:
    """Ranking for the highest-risk lesion: level, then score.

    An INCONCLUSIVE lesion is one whose own evidence scored LOW where the device could not look,
    so it ranks with LOW lesions by score. This keeps the highest-risk lesion, the patient score
    and the explanation pointing at the same finding.
    """
    level = RiskLevel.LOW if s.level == RiskLevel.INCONCLUSIVE else s.level
    return RISK_ORDER[level], s.score


def highest_risk(lesions: Sequence[ScoredLesion]) -> ScoredLesion | None:
    """Most severe lesion; ties broken by score, then by order."""
    return max(lesions, key=lesion_rank, default=None)


def run_triage(
    sites: list[SiteResult],
    spectral: list[SpectralReadingResult],
    patient: PatientInfo,
    risk_factors: RiskFactors,
    rules: LoadedConfig,
) -> TriageOutput:
    """Apply the triage rules to fully analysed sites and spectral readings."""
    tr = parse_triage_rules(rules.data)
    drafts = build_lesion_drafts(sites, spectral, tr.wl)
    scored = [score_lesion(d, f"L{i}", tr, risk_factors) for i, d in enumerate(drafts, start=1)]

    items = risk_items(risk_factors, patient, tr)
    rf_factors = risk_factor_factors(items, tr, lesion_present=bool(scored))
    factors = patient_factors(scored, rf_factors)
    score = round(total_points(factors), POINT_DECIMALS)

    ctx = ClinicalContext(risk_factors, site_appearances(sites), any_lesion=bool(scored))
    floors = matching_overrides(tr.overrides, ctx)
    # Lesion-only floors (e.g. a red area with FVL) must also hold for the patient: the overall
    # level is never lower than any lesion's level.
    seen_codes = {o.code for o in floors}
    for s in scored:
        for o in s.floors:
            if o.code not in seen_codes:
                seen_codes.add(o.code)
                floors.append(o)
    examined = [s for s in sites if s.role == SiteRole.EXAMINED]
    readable = sum(1 for s in examined if site_readable(s))
    decision = decide_level(
        score,
        floors,
        examined_readable=readable,
        examined_unreadable=len(examined) - readable,
        lesions=scored,
        rules=tr,
    )
    top = highest_risk(scored)
    explain = explanation(
        decision,
        lesions=scored,
        top=top,
        floors=floors,
        sites=sites,
        rf=risk_factors,
        rf_points=total_points(rf_factors),
        rf_names=[it.short for it in items],
        rules=tr,
    )
    action = action_from_template(
        decision.level,
        tr,
        dynamic_steps(decision.level, sites=sites, lesions=scored, rf=risk_factors),
    )
    triage = TriageResult(
        overall_risk_level=decision.level,
        score=score,
        thresholds=ScoreThresholds(moderate=tr.moderate_min_score, high=tr.high_min_score),
        contributing_factors=factors,
        overrides_applied=[override_factor(o, "overall") for o in floors],
        recommended_action=action,
        highest_risk_lesion_id=top.lesion_id if top is not None else None,
        rule_config=rules.ref,
        explanation=explain,
        requires_clinician_review=True,
        review_status="pending_review",
    )
    limitations = build_limitations(
        sites=sites, lesions=scored, spectral=spectral, decision=decision, rules=tr
    )
    return TriageOutput(
        lesions=[_assessment(s, tr) for s in scored],
        triage=triage,
        site_levels=site_levels(sites, scored),
        limitations=limitations,
    )
