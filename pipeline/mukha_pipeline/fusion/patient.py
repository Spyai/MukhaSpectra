"""Patient-level scoring, red-flag floors and the INCONCLUSIVE safety rule.

Patient score = highest lesion score + capped risk-factor points, and only when a lesion exists:
habits change counselling, not triage. The patient's factors are the top-scoring lesion's factors
followed by the risk-factor factors, so the patient score is also the exact sum of its factors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from mukha_pipeline.fusion.factors import Ledger, make_factor
from mukha_pipeline.fusion.overrides import highest_floor
from mukha_pipeline.fusion.rules import Override, TriageRules
from mukha_pipeline.fusion.scoring import ScoredLesion
from mukha_pipeline.fusion.text import SMOKELESS_NOUN, SMOKING_NOUN
from mukha_pipeline.schema.models import (
    RISK_ORDER,
    ContributingFactor,
    FactorSource,
    PatientInfo,
    RiskFactors,
    RiskLevel,
)
from mukha_pipeline.sites import site_label

InconclusiveReason = Literal[
    "no_sites_analysed", "too_few_sites", "sites_unreadable", "lesion_unassessed"
]

HISTORY = FactorSource.CLINICAL_HISTORY
NO_LESION_DETAIL = (
    "Counselling point. Habits and history raise the level only when a lesion is found."
)


@dataclass(frozen=True)
class RiskItem:
    code: str
    label: str  # factor label, e.g. "Smokeless tobacco (gutka)"
    short: str  # for sentences, e.g. "gutka"
    points: float  # full points before the cap


@dataclass(frozen=True)
class PatientDecision:
    level: RiskLevel
    score_level: RiskLevel
    inconclusive_reason: InconclusiveReason | None
    device_analysis_unavailable: bool


def _use_item(
    key: str, uses: bool, former: bool, label: str, short: str, rules: TriageRules
) -> RiskItem | None:
    """Current use scores fully; former use at ``former_use_fraction``."""
    points = rules.patient.risk_factor_points[key]
    if uses:
        return RiskItem(f"history.{key}", label, short, points)
    if former:
        return RiskItem(
            f"history.{key}_former",
            f"Former use: {label[0].lower()}{label[1:]}",
            f"former {short}",
            points * rules.patient.former_use_fraction,
        )
    return None


def risk_items(rf: RiskFactors, patient: PatientInfo, rules: TriageRules) -> list[RiskItem]:
    """Risk factors present, with their full (uncapped) points."""
    st, sm, an = rf.smokeless_tobacco, rf.smoking, rf.areca_nut
    chew = ", ".join(SMOKELESS_NOUN[t] for t in st.types) or "chewing tobacco"
    smoke = ", ".join(SMOKING_NOUN[t] for t in sm.types) or "smoking"
    candidates = [
        _use_item(
            "smokeless_tobacco", st.uses, st.former, f"Smokeless tobacco ({chew})", chew, rules
        ),
        _use_item("smoking", sm.uses, sm.former, f"Smoking ({smoke})", smoke, rules),
        _use_item(
            "areca_nut", an.uses, an.former, "Areca nut (supari) chewing", "areca nut", rules
        ),
    ]
    items = [c for c in candidates if c is not None]
    pts = rules.patient.risk_factor_points
    if rf.alcohol.frequency == "daily":
        items.append(
            RiskItem(
                "history.alcohol_daily",
                "Drinks alcohol daily",
                "daily alcohol",
                pts["alcohol_daily"],
            )
        )
    for key, text in (
        ("prior_oral_lesion", "previous oral lesion"),
        ("prior_oral_cancer", "previous oral cancer"),
    ):
        if getattr(rf, key):
            items.append(RiskItem(f"history.{key}", text.capitalize(), text, pts[key]))
    if patient.age_years is not None and patient.age_years >= 40:
        items.append(
            RiskItem(
                "history.age_40_plus",
                f"Age {patient.age_years} (40 or over)",
                f"age {patient.age_years}",
                pts["age_40_plus"],
            )
        )
    return items


def risk_factor_factors(
    items: list[RiskItem], rules: TriageRules, lesion_present: bool
) -> list[ContributingFactor]:
    """Risk factors as factors: scored and capped with a lesion, zero-point notes without."""
    led = Ledger()
    if not lesion_present:
        for it in items:
            led.note(it.code, it.label, HISTORY, NO_LESION_DETAIL)
        return led.factors
    for it in items:
        led.add(it.code, it.label, HISTORY, it.points)
    raw = led.total
    cap = rules.patient.risk_factor_cap
    if raw > cap:
        led.add(
            "history.cap",
            f"Habits and history count for at most {cap:g} points in total",
            HISTORY,
            cap - raw,
        )
    return led.factors


def top_scoring(lesions: list[ScoredLesion]) -> ScoredLesion | None:
    return max(lesions, key=lambda s: s.score, default=None)


def patient_factors(
    lesions: list[ScoredLesion], rf_factors: list[ContributingFactor]
) -> list[ContributingFactor]:
    """Top-scoring lesion's factors (overrides excluded, they are listed separately) + history."""
    top = top_scoring(lesions)
    out: list[ContributingFactor] = []
    if top is not None:
        where = f"{top.lesion_id} ({site_label(top.draft.site)})"
        for f in top.factors:
            if f.source == FactorSource.OVERRIDE:
                continue
            detail = f"{where}: {f.detail}" if f.detail else where
            out.append(make_factor(f.code, f.label, f.source, f.points, detail))
    return out + rf_factors


def decide_level(
    score: float,
    floors: list[Override],
    *,
    examined_readable: int,
    examined_unreadable: int,
    lesions: list[ScoredLesion],
    rules: TriageRules,
) -> PatientDecision:
    """Combine score, floors and the INCONCLUSIVE safety rule into the overall level.

    * Floors only ever raise the level.
    * Too few readable examined sites: INCONCLUSIVE, unless a clinical floor gives MODERATE/HIGH
      or the score alone reaches HIGH (an urgent referral is never downgraded to a retake).
    * LOW needs every examined site readable and every lesion assessed; otherwise INCONCLUSIVE.
    """
    score_level = rules.level_for_score(score)
    level = score_level
    floor = highest_floor(floors)
    if floor is not None and RISK_ORDER[floor] > RISK_ORDER[level]:
        level = floor
    if examined_readable < rules.inconclusive.min_sites_analysed:
        none = examined_readable == 0
        if floor is not None or score_level == RiskLevel.HIGH:
            return PatientDecision(level, score_level, None, none)
        reason: InconclusiveReason = "no_sites_analysed" if none else "too_few_sites"
        return PatientDecision(RiskLevel.INCONCLUSIVE, score_level, reason, none)
    if level == RiskLevel.LOW and rules.inconclusive.low_requires_all_examined_sites_analysed:
        if examined_unreadable:
            return PatientDecision(RiskLevel.INCONCLUSIVE, score_level, "sites_unreadable", False)
        if any(s.level == RiskLevel.INCONCLUSIVE for s in lesions):
            return PatientDecision(RiskLevel.INCONCLUSIVE, score_level, "lesion_unassessed", False)
    return PatientDecision(level, score_level, None, False)
