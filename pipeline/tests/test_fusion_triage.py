"""Patient-level triage: levels, the INCONCLUSIVE safety rule, floors, actions, explanations."""

from __future__ import annotations

import pytest
from fusion_helpers import (
    S,
    assert_invariants,
    codes,
    default_rules,
    factor,
    limitation_codes,
    parse_rules_default,
    patient,
    reading,
    region,
    risk,
    site,
    wl_ok,
    write_rules,
)

from mukha_pipeline.fusion import TriageOutput, run_triage
from mukha_pipeline.fusion.rules import Override
from mukha_pipeline.schema.models import (
    RISK_ORDER,
    ActionCode,
    Diascopy,
    FactorSource,
    ReferenceType,
    ReferralUrgency,
    RiskFactors,
    RiskLevel,
    SiteResult,
    SiteRole,
    SpectralReadingResult,
    VisualAppearance,
    WLLabel,
)

V = VisualAppearance


def triage(
    sites: list[SiteResult],
    spectral: list[SpectralReadingResult] | None = None,
    *,
    age: int | None = 35,
    rf: RiskFactors | None = None,
    rules=None,
) -> TriageOutput:
    out = run_triage(sites, spectral or [], patient(age), rf or risk(), rules or default_rules())
    assert_invariants(out)
    return out


def deep_region(**kw):
    params = dict(mean_rel=0.40, area_mm2=60.0, border=0.5, contrast=1.8)
    params.update(kw)
    return region(**params)


# --------------------------------------------------------------------------------------------
# Levels
# --------------------------------------------------------------------------------------------


def test_no_lesions_is_low_and_never_reassures():
    out = triage([site(S.BUCCAL_MUCOSA_LEFT), site(S.BUCCAL_MUCOSA_RIGHT)])
    t = out.triage
    assert t.overall_risk_level == RiskLevel.LOW
    assert t.score == 0.0
    assert out.lesions == []
    assert t.highest_risk_lesion_id is None
    assert "does not rule out disease" in t.explanation
    assert "re-screening in 12 months" in t.explanation
    assert t.recommended_action.code == ActionCode.ROUTINE_RESCREEN
    assert t.recommended_action.referral_urgency == ReferralUrgency.NONE
    assert out.site_levels == {
        S.BUCCAL_MUCOSA_LEFT: RiskLevel.LOW,
        S.BUCCAL_MUCOSA_RIGHT: RiskLevel.LOW,
    }
    assert t.rule_config == default_rules().ref
    assert (t.thresholds.moderate, t.thresholds.high) == (3.0, 6.0)


def test_examined_site_failed_qc_makes_low_inconclusive():
    out = triage([site(S.BUCCAL_MUCOSA_LEFT, analysed=False), site(S.BUCCAL_MUCOSA_RIGHT)])
    t = out.triage
    assert t.overall_risk_level == RiskLevel.INCONCLUSIVE
    assert "1 of 2 examined sites could not be analysed (Left buccal mucosa)" in t.explanation
    assert "A low result needs every site to be readable" in t.explanation
    assert t.recommended_action.code == ActionCode.RETAKE_OR_EXAMINE
    assert "Sites to retake: Left buccal mucosa." in t.recommended_action.steps
    assert out.site_levels[S.BUCCAL_MUCOSA_LEFT] == RiskLevel.INCONCLUSIVE
    assert out.site_levels[S.BUCCAL_MUCOSA_RIGHT] == RiskLevel.LOW
    assert "sites_failed_qc" in limitation_codes(out)
    assert "device_analysis_unavailable" not in limitation_codes(out)


def test_af_insufficient_mucosa_counts_as_unreadable():
    """'analysed' with no usable AF is silence, not a negative finding."""
    out = triage([site(S.BUCCAL_MUCOSA_LEFT, af_status="insufficient_mucosa")])
    assert out.triage.overall_risk_level == RiskLevel.INCONCLUSIVE
    assert out.site_levels[S.BUCCAL_MUCOSA_LEFT] == RiskLevel.INCONCLUSIVE


def test_no_sites_at_all_is_inconclusive():
    out = triage([])
    assert out.triage.overall_risk_level == RiskLevel.INCONCLUSIVE
    assert "no examined site had usable images" in out.triage.explanation
    assert "device_analysis_unavailable" in limitation_codes(out)


def test_all_sites_failed_is_inconclusive_with_device_unavailable():
    out = triage(
        [site(S.BUCCAL_MUCOSA_LEFT, analysed=False), site(S.TONGUE_DORSUM, analysed=False)]
    )
    t = out.triage
    assert t.overall_risk_level == RiskLevel.INCONCLUSIVE
    assert "none of the examined sites could be analysed" in t.explanation
    assert {"sites_failed_qc", "device_analysis_unavailable"} <= set(limitation_codes(out))


def test_failed_qc_plus_red_patch_seen_is_high():
    out = triage([site(S.BUCCAL_MUCOSA_LEFT, analysed=False, appearance=[V.RED_PATCH])])
    t = out.triage
    assert t.overall_risk_level == RiskLevel.HIGH
    assert "visual.red_patch" in codes(t.overrides_applied)
    assert "device_analysis_unavailable" in limitation_codes(out)
    assert "rests on the clinical findings" in t.explanation
    (lesion,) = out.lesions
    assert lesion.origins == ["visual"]
    assert lesion.risk_level == RiskLevel.HIGH
    assert "qc.device_analysis_unavailable" in codes(lesion.contributing_factors)
    assert any("do not delay the referral" in s for s in t.recommended_action.steps)
    assert out.site_levels[S.BUCCAL_MUCOSA_LEFT] == RiskLevel.HIGH


def test_visual_other_at_failed_site_is_moderate_with_retake_step():
    """Any lesion the worker saw is at least MODERATE, even where the device could not look."""
    out = triage([site(S.BUCCAL_MUCOSA_LEFT, analysed=False, lesion_seen=True)])
    t = out.triage
    assert t.overall_risk_level == RiskLevel.MODERATE
    (lesion,) = out.lesions
    assert lesion.risk_level == RiskLevel.MODERATE
    assert "visual.lesion_seen" in codes(t.overrides_applied)
    assert "rests on the clinical findings" in t.explanation
    assert (
        "Retake the captures that could not be analysed: Left buccal mucosa."
        in t.recommended_action.steps
    )


def test_visual_other_at_failed_site_without_floor_stays_inconclusive(tmp_path):
    def drop(d: dict) -> None:
        d["overrides"] = [o for o in d["overrides"] if o["code"] != "visual.lesion_seen"]

    rules = write_rules(tmp_path, drop)
    out = triage([site(S.BUCCAL_MUCOSA_LEFT, analysed=False, lesion_seen=True)], rules=rules)
    t = out.triage
    assert t.overall_risk_level == RiskLevel.INCONCLUSIVE
    (lesion,) = out.lesions
    assert lesion.risk_level == RiskLevel.INCONCLUSIVE
    assert lesion.summary.startswith("Finding at the left buccal mucosa that the device could not")
    assert "refer for clinical examination" in t.explanation
    assert any(
        s.startswith("A lesion was seen at Left buccal mucosa") for s in t.recommended_action.steps
    )


def test_mild_fvl_is_low_and_more_features_make_it_moderate():
    # Upper lip mucosa: not a high-risk site (where any FVL is at least MODERATE), so the level
    # follows the score; the lower lip is its analogous reference (x0.9).
    mild = triage([site(S.UPPER_LIP_MUCOSA, [region(mean_rel=0.70)]), site(S.LOWER_LIP_MUCOSA)])
    (lesion,) = mild.lesions
    expected = (1.5 + 0.75 * 0.5) * 0.9
    assert lesion.score == pytest.approx(expected, abs=1e-3)
    assert lesion.risk_level == RiskLevel.LOW
    assert mild.triage.overall_risk_level == RiskLevel.LOW
    assert "The only finding scored low on this screen" in mild.triage.explanation

    more_region = region(mean_rel=0.70, fraction=0.06, border=0.2, contrast=1.5)
    more = triage([site(S.UPPER_LIP_MUCOSA, [more_region])])
    assert more.lesions[0].score == pytest.approx((1.5 + 0.375 + 0.5 + 0.5 + 0.5) * 0.9, abs=1e-3)
    assert more.lesions[0].risk_level == RiskLevel.MODERATE
    assert more.triage.overall_risk_level == RiskLevel.MODERATE
    assert more.triage.recommended_action.code == ActionCode.REEXAMINE_2_WEEKS


def test_deep_large_irregular_fvl_at_high_risk_site_with_gutka_is_high():
    rf = risk(smokeless_tobacco={"uses": True, "types": ["gutka"], "years": 20})
    out = triage(
        [site(S.LATERAL_TONGUE_RIGHT, [deep_region()]), site(S.LATERAL_TONGUE_LEFT)], age=52, rf=rf
    )
    (lesion,) = out.lesions
    assert lesion.score == pytest.approx(5.5 * 1.15, abs=1e-3)
    assert lesion.risk_level == RiskLevel.HIGH
    t = out.triage
    assert t.overall_risk_level == RiskLevel.HIGH
    assert t.score == pytest.approx(5.5 * 1.15 + 1.0 + 0.5, abs=1e-3)
    assert factor(t.contributing_factors, "history.smokeless_tobacco").points == 1.0
    assert factor(t.contributing_factors, "history.age_40_plus").points == 0.5
    assert t.highest_risk_lesion_id == lesion.lesion_id == "L1"
    assert t.explanation.startswith(
        "HIGH risk: strong loss of fluorescence at the right lateral tongue (relative AF 0.40 vs "
        "the left lateral tongue)"
    )
    assert "Habits and history (gutka and age 52) added 1.5 points." in t.explanation
    assert t.recommended_action.code == ActionCode.URGENT_REFERRAL
    assert t.recommended_action.referral_urgency == ReferralUrgency.URGENT
    assert lesion.location is not None and lesion.region_ids == ["r1"]
    assert out.site_levels[S.LATERAL_TONGUE_RIGHT] == RiskLevel.HIGH
    assert out.site_levels[S.LATERAL_TONGUE_LEFT] == RiskLevel.LOW


def test_spec_example_explanation_names_top_reasons_with_numbers():
    rf = risk(symptoms={"non_healing_ulcer_over_2_weeks": True})
    out = triage(
        [
            site(S.LATERAL_TONGUE_RIGHT, [deep_region(mean_rel=0.42)], appearance=[V.RED_PATCH]),
            site(S.LATERAL_TONGUE_LEFT),
        ],
        rf=rf,
    )
    assert out.triage.explanation.startswith(
        "HIGH risk: strong loss of fluorescence at the right lateral tongue (relative AF 0.42 vs "
        "the left lateral tongue), a red patch seen by the health worker and a non-healing ulcer "
        "reported for more than 2 weeks."
    )


def test_habits_alone_never_raise_above_low():
    rf = risk(
        smokeless_tobacco={"uses": True, "types": ["gutka", "khaini"]},
        smoking={"uses": True, "types": ["bidi"]},
        areca_nut={"uses": True},
        alcohol={"uses": True, "frequency": "daily"},
        prior_oral_lesion=True,
    )
    out = triage([site(S.BUCCAL_MUCOSA_LEFT), site(S.BUCCAL_MUCOSA_RIGHT)], age=70, rf=rf)
    t = out.triage
    assert t.overall_risk_level == RiskLevel.LOW
    assert t.score == 0.0
    history = [f for f in t.contributing_factors if f.source == FactorSource.CLINICAL_HISTORY]
    assert len(history) == 6 and all(f.points == 0 for f in history)
    assert all("Counselling point" in (f.detail or "") for f in history)
    assert "stopping gutka, khaini, bidi, areca nut and daily alcohol" in t.explanation
    assert "The patient reports a previous oral lesion" in t.explanation
    assert any(s.startswith("Habits reported (gutka") for s in t.recommended_action.steps)


def test_risk_factor_cap_former_use_and_alcohol_rules():
    lesion_site = site(S.BUCCAL_MUCOSA_LEFT, [region(mean_rel=0.70)])
    rf = risk(
        smokeless_tobacco={"former": True, "types": ["gutka"]},
        smoking={"uses": True},
        areca_nut={"uses": True},
        alcohol={"uses": True, "frequency": "weekly"},
        prior_oral_cancer=True,
    )
    out = triage([lesion_site], age=39, rf=rf)
    f = out.triage.contributing_factors
    assert factor(f, "history.smokeless_tobacco_former").points == 0.5
    assert factor(f, "history.smoking").points == 0.5
    assert "history.alcohol_daily" not in codes(f)
    assert "history.age_40_plus" not in codes(f)
    raw = 0.5 + 0.5 + 0.75 + 1.5
    assert factor(f, "history.cap").points == pytest.approx(2.5 - raw)
    assert out.triage.score == pytest.approx(out.lesions[0].score + 2.5, abs=1e-6)


def test_former_only_habits_get_encouragement_step():
    out = triage([site(S.BUCCAL_MUCOSA_LEFT)], rf=risk(smoking={"former": True, "types": ["bidi"]}))
    assert "Encourage the patient to stay off bidi." in out.triage.recommended_action.steps
    assert factor(out.triage.contributing_factors, "history.smoking_former").points == 0.0


# --------------------------------------------------------------------------------------------
# Overrides
# --------------------------------------------------------------------------------------------


def _scenario_for(o: Override) -> tuple[list[SiteResult], RiskFactors]:
    appearance = sorted(o.visual_appearance)[:1] if o.visual_appearance else []
    regions = [region(mean_rel=0.72)] if o.any_lesion else []
    seen = True if o.visual_lesion_seen else None
    sites = [
        site(S.BUCCAL_MUCOSA_LEFT, regions, appearance=appearance, lesion_seen=seen),
        site(S.BUCCAL_MUCOSA_RIGHT),
    ]
    rf: dict = {}
    if o.symptom:
        rf["symptoms"] = {o.symptom: True}
    if o.history:
        rf[o.history] = True
    return sites, risk(**rf)


@pytest.mark.parametrize("override", parse_rules_default().overrides, ids=lambda o: o.code)
def test_every_override_floor_applies(override: Override):
    sites, rf = _scenario_for(override)
    out = triage(sites, rf=rf)
    t = out.triage
    assert override.code in codes(t.overrides_applied)
    applied = factor(t.overrides_applied, override.code)
    assert applied.source == FactorSource.OVERRIDE and applied.label == override.label
    assert RISK_ORDER[t.overall_risk_level] >= RISK_ORDER[override.floor]
    if override.lesion_scoped:
        (lesion,) = out.lesions
        assert RISK_ORDER[lesion.risk_level] >= RISK_ORDER[override.floor]
        assert override.code in codes(lesion.contributing_factors)


def test_lesion_conditional_floors_need_a_lesion():
    """Without a lesion the HIGH floors do not hold, but the lump and previous oral cancer still
    set their own MODERATE floors."""
    rf = risk(symptoms={"lump": True}, prior_oral_cancer=True)
    out = triage([site(S.BUCCAL_MUCOSA_LEFT)], rf=rf)
    applied = codes(out.triage.overrides_applied)
    assert "symptom.lump_with_lesion" not in applied
    assert "history.prior_cancer_with_lesion" not in applied
    assert {"symptom.lump", "history.prior_oral_cancer"} <= set(applied)
    assert out.triage.overall_risk_level == RiskLevel.MODERATE


def test_floor_cannot_be_lowered_by_low_device_score():
    rf = risk(symptoms={"neck_lump": True})
    out = triage([site(S.BUCCAL_MUCOSA_LEFT), site(S.BUCCAL_MUCOSA_RIGHT)], rf=rf)
    assert out.triage.overall_risk_level == RiskLevel.HIGH
    assert out.triage.explanation.startswith("HIGH risk: a lump in the neck.")


def test_floor_leads_explanation_when_it_decides_the_level():
    rf = risk(symptoms={"non_healing_ulcer_over_2_weeks": True})
    out = triage([site(S.BUCCAL_MUCOSA_LEFT, [region(mean_rel=0.70)])], rf=rf)
    assert out.triage.overall_risk_level == RiskLevel.HIGH
    assert out.triage.explanation.startswith(
        "HIGH risk: a non-healing ulcer reported for more than 2 weeks and mild loss of "
        "fluorescence"
    )


def test_visual_floor_at_other_site_is_named_in_explanation():
    sites = [
        site(S.LATERAL_TONGUE_RIGHT, [deep_region()]),
        site(S.LATERAL_TONGUE_LEFT),
        site(S.BUCCAL_MUCOSA_LEFT, appearance=[V.WHITE_PATCH]),
    ]
    out = triage(sites)
    t = out.triage
    assert t.overall_risk_level == RiskLevel.HIGH
    assert "a white patch seen by the health worker at the left buccal mucosa" in t.explanation
    assert len(out.lesions) == 2
    assert t.highest_risk_lesion_id == "L1"
    assert t.explanation.endswith("1 other suspicious finding is listed with the sites.")


def test_reference_only_visual_flag_still_counts():
    sites = [
        site(S.BUCCAL_MUCOSA_LEFT),
        site(S.BUCCAL_MUCOSA_RIGHT, role=SiteRole.REFERENCE_ONLY, appearance=[V.RED_PATCH]),
    ]
    out = triage(sites)
    assert out.triage.overall_risk_level == RiskLevel.HIGH
    assert out.lesions[0].site == S.BUCCAL_MUCOSA_RIGHT


def test_score_high_is_kept_when_no_examined_site_readable():
    """An urgent level is never downgraded to a retake, even without device analysis."""
    wl = wl_ok(WLLabel.RED_PATCH, 0.9)
    sites = [site(S.TONGUE_DORSUM, af_status="insufficient_mucosa", lesion_seen=True, wl=wl)]
    spectral = [reading("s1", S.TONGUE_DORSUM, hb_delta=0.08, ratio_delta=-0.2)]
    out = triage(sites, spectral, rf=risk(smokeless_tobacco={"uses": True}), age=60)
    assert out.lesions[0].score == pytest.approx(3.0 + 2.0 + 1.0)
    assert out.triage.overall_risk_level == RiskLevel.HIGH
    assert "device_analysis_unavailable" in limitation_codes(out)


def test_moderate_score_without_floor_and_no_readable_site_is_inconclusive():
    wl = wl_ok(WLLabel.RED_PATCH, 0.9)
    sites = [site(S.BUCCAL_MUCOSA_LEFT, af_status="insufficient_mucosa", wl=wl)]
    out = triage(sites, rf=risk(smokeless_tobacco={"uses": True}), age=60)
    assert out.lesions[0].risk_level == RiskLevel.MODERATE
    assert out.triage.overrides_applied == []
    assert out.triage.overall_risk_level == RiskLevel.INCONCLUSIVE
    assert any("refer for clinical examination" in s for s in out.triage.recommended_action.steps)


# --------------------------------------------------------------------------------------------
# Lesions not assessable by the device
# --------------------------------------------------------------------------------------------


def test_spectral_only_lesion_at_unimaged_site_blocks_low():
    spectral = [reading("s1", S.LOWER_LIP_MUCOSA, hb_delta=0.04, label="indeterminate")]
    out = triage([site(S.BUCCAL_MUCOSA_LEFT)], spectral)
    (lesion,) = out.lesions
    assert lesion.origins == ["spectral"]
    assert lesion.risk_level == RiskLevel.INCONCLUSIVE
    assert "qc.no_photos_at_site" in codes(lesion.contributing_factors)
    assert lesion.summary.startswith("Finding at the lower lip mucosa with no analysed photos")
    t = out.triage
    assert t.overall_risk_level == RiskLevel.INCONCLUSIVE
    assert "the finding at the lower lip mucosa could not be checked" in t.explanation
    assert any(
        s.startswith("Take white-light and violet-light photos of Lower lip mucosa")
        for s in t.recommended_action.steps
    )
    assert out.site_levels[S.LOWER_LIP_MUCOSA] == RiskLevel.INCONCLUSIVE


def test_low_requirement_can_be_relaxed_in_config(tmp_path):
    rules = write_rules(
        tmp_path, lambda d: d["inconclusive"].update(low_requires_all_examined_sites_analysed=False)
    )
    out = triage(
        [site(S.BUCCAL_MUCOSA_LEFT, analysed=False), site(S.BUCCAL_MUCOSA_RIGHT)], rules=rules
    )
    assert out.triage.overall_risk_level == RiskLevel.LOW


# --------------------------------------------------------------------------------------------
# Reference sites, limitations, site levels
# --------------------------------------------------------------------------------------------


def test_reference_only_fvl_is_scored_and_flagged():
    sites = [
        site(S.BUCCAL_MUCOSA_LEFT),
        site(S.BUCCAL_MUCOSA_RIGHT, [region(mean_rel=0.55)], role=SiteRole.REFERENCE_ONLY),
    ]
    out = triage(sites)
    (lesion,) = out.lesions
    assert lesion.site == S.BUCCAL_MUCOSA_RIGHT and lesion.score > 0
    assert "reference_site_fvl" in limitation_codes(out)
    msg = next(x.message for x in out.limitations if x.code == "reference_site_fvl")
    assert msg == (
        "The reference site also showed loss of fluorescence; comparisons may be less reliable."
    )
    assert out.site_levels[S.BUCCAL_MUCOSA_RIGHT] == lesion.risk_level


def test_bilateral_fvl_flags_reference_but_unilateral_does_not():
    bilateral = triage(
        [
            site(S.BUCCAL_MUCOSA_LEFT, [region(mean_rel=0.6)]),
            site(S.BUCCAL_MUCOSA_RIGHT, [region(mean_rel=0.6)]),
        ]
    )
    assert "reference_site_fvl" in limitation_codes(bilateral)
    unilateral = triage([site(S.BUCCAL_MUCOSA_LEFT, [region()]), site(S.BUCCAL_MUCOSA_RIGHT)])
    assert "reference_site_fvl" not in limitation_codes(unilateral)


def test_unreadable_reference_only_site_does_not_block_low():
    sites = [
        site(S.BUCCAL_MUCOSA_LEFT),
        site(S.BUCCAL_MUCOSA_RIGHT, role=SiteRole.REFERENCE_ONLY, analysed=False),
    ]
    out = triage(sites)
    assert out.triage.overall_risk_level == RiskLevel.LOW
    assert out.site_levels[S.BUCCAL_MUCOSA_RIGHT] == RiskLevel.INCONCLUSIVE
    assert "Retake the captures that could not be analysed: Right buccal mucosa." in (
        out.triage.recommended_action.steps
    )


def test_limitations_cover_wl_reference_and_spectral():
    sites = [site(S.TONGUE_DORSUM, [region()], ref_type=ReferenceType.IN_IMAGE_AUTO)]
    spectral = [
        reading("s1", S.TONGUE_DORSUM),
        reading("s2", S.TONGUE_DORSUM, hb_delta=0.1, qc_passed=False),
    ]
    out = triage(sites, spectral)
    lims = limitation_codes(out)
    for code in [
        "reference_in_image",
        "wl_model_unavailable",
        "spectral_no_reference",
        "spectral_qc_failed",
        "triage_rules_unvalidated",
    ]:
        assert code in lims
    msg = next(x.message for x in out.limitations if x.code == "reference_in_image")
    assert msg.startswith("At Tongue dorsum, no opposite-side or similar site was available")


def test_site_levels_take_most_severe_lesion():
    # Upper lip mucosa is not a high-risk site, so the mild region r1 keeps its LOW level.
    sites = [
        site(S.UPPER_LIP_MUCOSA, [region("r1", mean_rel=0.70), deep_region(region_id="r2")]),
        site(S.LOWER_LIP_MUCOSA),
    ]
    out = triage(sites)
    assert [lesion.risk_level for lesion in out.lesions] == [RiskLevel.LOW, RiskLevel.MODERATE]
    assert out.site_levels[S.UPPER_LIP_MUCOSA] == RiskLevel.MODERATE
    assert out.triage.highest_risk_lesion_id == "L2"


# --------------------------------------------------------------------------------------------
# Actions
# --------------------------------------------------------------------------------------------


def test_irritant_steps_for_moderate_and_high():
    rf = risk(irritants={"sharp_tooth": True, "ill_fitting_denture": True})
    moderate = triage([site(S.BUCCAL_MUCOSA_LEFT, appearance=[V.WHITE_PATCH])], rf=rf)
    steps = moderate.triage.recommended_action.steps
    assert moderate.triage.overall_risk_level == RiskLevel.MODERATE
    assert steps[:3] == list(default_rules().data["actions"]["MODERATE"]["steps"])
    assert (
        "Smooth or treat the sharp tooth (dental referral) so it stops injuring the mucosa."
        in steps
    )
    assert any(s.startswith("Get the ill-fitting denture adjusted") for s in steps)

    high = triage([site(S.BUCCAL_MUCOSA_LEFT, appearance=[V.GROWTH])], rf=rf)
    steps = high.triage.recommended_action.steps
    assert high.triage.overall_risk_level == RiskLevel.HIGH
    assert any(
        s.startswith("Alongside the referral, smooth or treat the sharp tooth") for s in steps
    )


def test_trismus_step_and_moderate_floor():
    rf = risk(symptoms={"difficulty_opening_mouth": True}, areca_nut={"uses": True})
    out = triage([site(S.BUCCAL_MUCOSA_LEFT)], rf=rf)
    assert out.triage.overall_risk_level == RiskLevel.MODERATE
    assert any("oral submucous fibrosis" in s for s in out.triage.recommended_action.steps)


def test_lesion_actions_follow_lesion_level():
    out = triage([site(S.LATERAL_TONGUE_RIGHT, [deep_region()]), site(S.LATERAL_TONGUE_LEFT)])
    lesion = out.lesions[0]
    assert lesion.recommended_action.code == ActionCode.URGENT_REFERRAL
    assert lesion.recommended_action.timeline_days == 14
    assert lesion.summary.startswith("HIGH risk at the right lateral tongue: strong loss of")
    assert "about 60 mm²" in lesion.summary


# --------------------------------------------------------------------------------------------
# Configurable thresholds
# --------------------------------------------------------------------------------------------


def test_thresholds_come_from_yaml(tmp_path):
    rules = write_rules(
        tmp_path,
        lambda d: d["levels"].update(moderate_min_score=1.0, high_min_score=2.0),
    )
    out = triage([site(S.BUCCAL_MUCOSA_LEFT, [region(mean_rel=0.70)])], rules=rules)
    assert out.lesions[0].risk_level == RiskLevel.HIGH
    assert out.triage.overall_risk_level == RiskLevel.HIGH
    assert (out.triage.thresholds.moderate, out.triage.thresholds.high) == (1.0, 2.0)
    assert out.triage.rule_config.sha256 != default_rules().ref.sha256


def test_weights_come_from_yaml(tmp_path):
    def mutate(d: dict) -> None:
        d["lesion"]["af"]["fvl_present"] = 4.0
        d["lesion"]["site"]["high_risk_multiplier"] = 1.0

    out = triage(
        [site(S.BUCCAL_MUCOSA_LEFT, [region(mean_rel=0.75)])], rules=write_rules(tmp_path, mutate)
    )
    assert out.lesions[0].score == pytest.approx(4.0)
    assert "site.high_risk" not in codes(out.lesions[0].contributing_factors)


def test_low_without_timeline_still_asks_for_rescreening(tmp_path):
    rules = write_rules(tmp_path, lambda d: d["actions"]["LOW"].update(timeline_days=None))
    out = triage([site(S.BUCCAL_MUCOSA_LEFT)], rules=rules)
    assert "regular re-screening matters" in out.triage.explanation
    assert out.triage.recommended_action.timeline_days is None


def test_explanation_and_top_lesion_follow_the_scoring_lesion():
    """An unassessed low-evidence finding must not hijack the explanation of a MODERATE level."""
    sites = [
        site(
            S.LOWER_GINGIVA,
            [region(mean_rel=0.5, fraction=0.2)],
            diascopy=Diascopy.FVL_RESOLVES,
            wl=wl_ok(WLLabel.WHITE_PATCH, 0.7),
        ),
        site(S.UPPER_GINGIVA),
    ]
    spectral = [reading("s1", S.LOWER_LIP_MUCOSA, hb_delta=0.04, label="indeterminate")]
    rf = risk(alcohol={"uses": True, "frequency": "daily"})
    out = triage(sites, spectral, age=45, rf=rf)
    gum, lip = out.lesions
    # Strong FVL (relative AF 0.50) stays MODERATE even though diascopy discounted it.
    assert gum.risk_level == RiskLevel.MODERATE and lip.risk_level == RiskLevel.INCONCLUSIVE
    t = out.triage
    assert t.overall_risk_level == RiskLevel.MODERATE
    assert t.highest_risk_lesion_id == gum.lesion_id
    assert "white patch at the lower gingiva" in t.explanation
    assert "returned under glass-slide pressure" in gum.summary
    assert out.site_levels[S.LOWER_LIP_MUCOSA] == RiskLevel.INCONCLUSIVE


def test_minimum_readable_sites_is_configurable(tmp_path):
    rules = write_rules(tmp_path, lambda d: d["inconclusive"].update(min_sites_analysed=2))
    out = triage([site(S.BUCCAL_MUCOSA_LEFT)], rules=rules)
    t = out.triage
    assert t.overall_risk_level == RiskLevel.INCONCLUSIVE
    assert "only 1 examined site could be analysed and at least 2 are needed" in t.explanation
    assert "device_analysis_unavailable" not in limitation_codes(out)
    floored = triage([site(S.BUCCAL_MUCOSA_LEFT, appearance=[V.ULCER])], rules=rules)
    assert floored.triage.overall_risk_level == RiskLevel.MODERATE
    assert "device_analysis_unavailable" not in limitation_codes(floored)
