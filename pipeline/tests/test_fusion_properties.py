"""Randomised scenarios: safety and bookkeeping invariants hold for any combination of inputs."""

from __future__ import annotations

import numpy as np
from fusion_helpers import (
    all_texts,
    assert_invariants,
    default_rules,
    patient,
    reading,
    region,
    risk,
    site,
    wl_ok,
    wl_unavailable,
)

from mukha_pipeline.fusion import run_triage
from mukha_pipeline.fusion.lesions import site_readable, visual_seen
from mukha_pipeline.schema.models import (
    RISK_ORDER,
    ActionCode,
    ConfounderType,
    Diascopy,
    ReferenceType,
    RiskLevel,
    SiteRole,
    VisualAppearance,
    WLLabel,
)
from mukha_pipeline.sites import OralSite


def lesion_rank_of(lesion) -> tuple[int, float]:
    level = RiskLevel.LOW if lesion.risk_level == RiskLevel.INCONCLUSIVE else lesion.risk_level
    return RISK_ORDER[level], lesion.score


def _random_site(rng: np.random.Generator, oral_site: OralSite):
    analysed = rng.random() > 0.2
    n_regions = int(rng.integers(0, 3)) if analysed else 0
    regions = [
        region(
            f"r{i + 1}",
            mean_rel=float(rng.uniform(0.2, 0.74)),
            area_mm2=float(rng.uniform(1, 120)) if rng.random() < 0.5 else None,
            fraction=float(rng.uniform(0.0, 0.3)),
            border=float(rng.uniform(0, 1)),
            contrast=float(rng.uniform(1.0, 2.5)),
            confounders=[
                (ConfounderType(c), float(rng.uniform(0, 1)))
                for c in rng.choice(list(ConfounderType), size=int(rng.integers(0, 3)))
            ],
        )
        for i in range(n_regions)
    ]
    appearance = [
        VisualAppearance(a)
        for a in rng.choice(list(VisualAppearance), size=int(rng.integers(0, 2)), replace=False)
    ]
    wl = wl_ok(WLLabel(rng.choice(list(WLLabel))), float(rng.uniform(0.3, 1.0)))
    return site(
        oral_site,
        regions,
        role=SiteRole.REFERENCE_ONLY if rng.random() < 0.15 else SiteRole.EXAMINED,
        analysed=analysed,
        af_status="ok" if rng.random() > 0.05 else "insufficient_mucosa",
        ref_type=ReferenceType(rng.choice(list(ReferenceType))) if rng.random() < 0.3 else None,
        appearance=appearance,
        lesion_seen=bool(appearance) or rng.random() < 0.05,
        diascopy=Diascopy(rng.choice(list(Diascopy))),
        wl=wl if rng.random() < 0.3 else wl_unavailable(),
        seg_status=[None, "ok", "ok", "low_confidence", "failed"][int(rng.integers(0, 5))],
    )


def _random_rf(rng: np.random.Generator):
    def flag(p: float = 0.2) -> bool:
        return bool(rng.random() < p)

    return risk(
        smokeless_tobacco={"uses": flag(0.4), "former": flag(), "types": ["gutka"]},
        smoking={"uses": flag(), "types": ["bidi"]},
        # details without the "uses" box count as current use
        areca_nut={"uses": flag(0.3), "former": flag(), "years": 5.0 if flag(0.1) else None},
        alcohol={"frequency": "daily" if flag() else "never"},
        prior_oral_lesion=flag(0.1),
        prior_oral_cancer=flag(0.05),
        symptoms={
            k: flag(0.07)
            for k in [
                "non_healing_ulcer_over_2_weeks",
                "white_patch",
                "red_patch",
                "lump",
                "difficulty_opening_mouth",
                "bleeding",
                "neck_lump",
                "pain",
                "burning_sensation",
            ]
        },
        irritants={"sharp_tooth": flag(), "ill_fitting_denture": flag(0.1)},
    )


def test_random_scenarios_keep_invariants(rng):
    rules = default_rules()
    all_sites = list(OralSite)
    for _ in range(300):
        chosen = rng.choice(all_sites, size=int(rng.integers(0, 6)), replace=False)
        sites = [_random_site(rng, OralSite(s)) for s in chosen]
        spectral = [
            reading(
                f"s{i}",
                OralSite(rng.choice(all_sites)),
                hb_delta=float(rng.uniform(-0.02, 0.1)) if rng.random() < 0.7 else None,
                ratio_delta=float(rng.uniform(-0.15, 0.02)),
                label=rng.choice(["suspicious", "indeterminate", "not_suspicious"]),
                qc_passed=bool(rng.random() > 0.1),
            )
            for i in range(int(rng.integers(0, 3)))
        ]
        age = int(rng.integers(18, 85))
        out = run_triage(sites, spectral, patient(age), _random_rf(rng), rules)
        assert_invariants(out)

        t = out.triage
        examined = [s for s in sites if s.role == SiteRole.EXAMINED]
        if t.overall_risk_level == RiskLevel.LOW:
            assert examined and all(site_readable(s) for s in examined)
        if not out.lesions:
            assert t.score == 0.0
        for s in sites:
            assert s.site in out.site_levels
            if not site_readable(s):
                assert out.site_levels[s.site] != RiskLevel.LOW
        if t.highest_risk_lesion_id is not None:
            top = next(x for x in out.lesions if x.lesion_id == t.highest_risk_lesion_id)
            assert all(lesion_rank_of(top) >= lesion_rank_of(x) for x in out.lesions)
            assert top.score == max(x.score for x in out.lesions) or top.risk_level in (
                RiskLevel.MODERATE,
                RiskLevel.HIGH,
            )
        assert [x.lesion_id for x in out.lesions] == [f"L{i + 1}" for i in range(len(out.lesions))]
        _assert_safety_floors(out, sites, spectral)


def _assert_safety_floors(out, sites, spectral) -> None:
    """Review findings clinical.0/4/5/7 and honesty.0/11, for any combination of inputs."""
    t = out.triage
    covered = {rid for x in out.lesions for rid in x.spectral_reading_ids}
    assert {r.reading_id for r in spectral if r.role == "lesion"} <= covered
    imaged = {s.site for s in sites if site_readable(s)}
    for lesion in out.lesions:
        if lesion.site not in imaged:  # no photos, or the device could not look properly
            assert lesion.risk_level != RiskLevel.LOW
        if "visual" in lesion.origins:
            assert lesion.risk_level in (RiskLevel.MODERATE, RiskLevel.HIGH)
        regions = [
            r for s in sites if s.af is not None for r in s.af.regions if s.site == lesion.site
        ]
        deep = [
            r for r in regions if r.region_id in lesion.region_ids and r.mean_relative_af <= 0.55
        ]
        if deep:
            assert lesion.risk_level in (RiskLevel.MODERATE, RiskLevel.HIGH)
        if t.overall_risk_level in (RiskLevel.MODERATE, RiskLevel.HIGH):
            assert lesion.recommended_action.code != ActionCode.ROUTINE_RESCREEN
    if any(visual_seen(s.visual_flag) for s in sites):
        assert t.overall_risk_level in (RiskLevel.MODERATE, RiskLevel.HIGH)
    texts = [x.lower() for x in all_texts(out)]
    assert not [x for x in texts if "healthy" in x or "not suspicious" in x or "no referral" in x]
