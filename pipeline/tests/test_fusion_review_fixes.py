"""Regression tests for the triage-rule review findings (rules v1.1.0).

Each test names the finding it pins down. All inputs are hand-built test fixtures (not patient
data, not synthetic images). The common thread is the product's worst failure, false
reassurance: a finding the device or the health worker could see must never be reported LOW.
"""

from __future__ import annotations

import pytest
from fusion_helpers import (
    S,
    all_texts,
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
    write_rules,
)

from mukha_pipeline.fusion import TriageOutput, TriageRulesError, parse_triage_rules, run_triage
from mukha_pipeline.positioning import find_positioning_violations
from mukha_pipeline.schema.models import (
    RISK_ORDER,
    ActionCode,
    ConfounderType,
    Diascopy,
    FactorSource,
    ReferenceType,
    RiskFactors,
    RiskLevel,
    SiteResult,
    SpectralReadingResult,
    VisualAppearance,
)
from mukha_pipeline.sites import OralSite, reference_site_for

C = ConfounderType
V = VisualAppearance
CLEAN_STEP = "lean the area (gauze/rinse) and retake the photos"


def triage(
    sites: list[SiteResult],
    spectral: list[SpectralReadingResult] | None = None,
    *,
    age: int | None = 30,
    rf: RiskFactors | None = None,
    rules=None,
) -> TriageOutput:
    out = run_triage(sites, spectral or [], patient(age), rf or risk(), rules or default_rules())
    assert_invariants(out)
    return out


def at_least(level: RiskLevel, floor: RiskLevel) -> bool:
    return level != RiskLevel.INCONCLUSIVE and RISK_ORDER[level] >= RISK_ORDER[floor]


def strong_region(**kw):
    """The reviewers' max-strength buccal region: HIGH (6.325) with no discounts."""
    params = dict(mean_rel=0.30, fraction=0.20, border=0.5, contrast=2.0)
    params.update(kw)
    return region(**params)


def steps(out: TriageOutput) -> list[str]:
    return out.triage.recommended_action.steps


# --------------------------------------------------------------------------------------------
# clinical.0: discounts must never turn strong fluorescence loss into LOW
# --------------------------------------------------------------------------------------------


def test_strong_region_without_discounts_is_high():
    out = triage([site(S.BUCCAL_MUCOSA_LEFT, [strong_region()]), site(S.BUCCAL_MUCOSA_RIGHT)])
    assert out.lesions[0].score == pytest.approx(5.5 * 1.15, abs=1e-3)
    assert out.lesions[0].risk_level == RiskLevel.HIGH


@pytest.mark.parametrize(
    ("confounders", "diascopy"),
    [
        ([(C.PORPHYRIN_RED, 1.0)], Diascopy.NOT_DONE),
        ([(C.PIGMENTATION, 0.96)], Diascopy.NOT_DONE),
        ([(C.GLARE, 1.0)], Diascopy.NOT_DONE),
        ([(C.PORPHYRIN_RED, 1.0), (C.PIGMENTATION, 1.0), (C.GLARE, 1.0)], Diascopy.NOT_DONE),
        ([], Diascopy.FVL_RESOLVES),
        ([(C.PORPHYRIN_RED, 1.0)], Diascopy.FVL_RESOLVES),
    ],
)
def test_no_discount_turns_strong_fvl_into_low(confounders, diascopy):
    sites = [
        site(S.BUCCAL_MUCOSA_LEFT, [strong_region(confounders=confounders)], diascopy=diascopy),
        site(S.BUCCAL_MUCOSA_RIGHT),
    ]
    out = triage(sites)
    (lesion,) = out.lesions
    assert lesion.score < 5.5 * 1.15 - 1.0  # the discounts still apply and still order lesions
    assert lesion.risk_level == RiskLevel.MODERATE
    assert "af.strong_fvl_floor" in codes(lesion.contributing_factors)
    assert out.triage.overall_risk_level == RiskLevel.MODERATE
    assert "af.strong_fvl_floor" in codes(out.triage.overrides_applied)
    assert lesion.recommended_action.code == ActionCode.REEXAMINE_2_WEEKS


def test_strong_floor_from_raw_points_and_from_depth_alone():
    # Mean 0.60 (not deep enough on its own) but raw points 1.5 + 1.125 + 1.0 + 0.5 + 0.5 =
    # 4.625 >= 3.0; plaque fluorescence and the in-image reference cut it to about 1.5.
    big = region(
        mean_rel=0.60,
        fraction=0.2,
        border=0.5,
        contrast=2.0,
        confounders=[(C.PORPHYRIN_RED, 1.0)],
    )
    raw = triage([site(S.SOFT_PALATE, [big])])
    assert raw.lesions[0].score < 3.0
    assert raw.lesions[0].risk_level == RiskLevel.MODERATE
    assert "af.strong_fvl_floor" in codes(raw.lesions[0].contributing_factors)
    # Mean relative AF 0.55 with a 0.60 cut-off: only 1.875 raw points, floored by depth.
    deep = triage(
        [
            site(
                S.TONGUE_DORSUM,
                [region(mean_rel=0.55, confounders=[(C.PIGMENTATION, 1.0)])],
                threshold=0.60,
            )
        ]
    )
    (lesion,) = deep.lesions
    assert lesion.risk_level == RiskLevel.MODERATE
    floor = factor(lesion.contributing_factors, "af.strong_fvl_floor")
    assert floor.points == 0.0 and floor.source == FactorSource.OVERRIDE


def test_mild_region_is_not_floored():
    out = triage([site(S.BUCCAL_MUCOSA_LEFT, [region(mean_rel=0.70)]), site(S.BUCCAL_MUCOSA_RIGHT)])
    assert out.lesions[0].risk_level == RiskLevel.LOW
    assert "af.strong_fvl_floor" not in codes(out.lesions[0].contributing_factors)


def test_strong_floor_is_configurable(tmp_path):
    def mutate(d: dict) -> None:
        d["lesion"]["af"]["strong_fvl_floor"].update(raw_points_min=99, mean_relative_af_max=0.0)

    rules = write_rules(tmp_path, mutate)
    sites = [
        site(S.BUCCAL_MUCOSA_LEFT, [strong_region(confounders=[(C.PORPHYRIN_RED, 1.0)])]),
        site(S.BUCCAL_MUCOSA_RIGHT),
    ]
    assert triage(sites, rules=rules).lesions[0].risk_level == RiskLevel.LOW


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("floor", "LOW"),
        ("raw_points_min", 0),
        ("raw_points_min", "three"),
        ("mean_relative_af_max", 1.5),
        ("label", "Malignant change likely"),
    ],
)
def test_strong_floor_validation(key, value):
    data = default_rules().data
    data = {**data, "lesion": {**data["lesion"], "af": dict(data["lesion"]["af"])}}
    data["lesion"]["af"]["strong_fvl_floor"] = {
        **data["lesion"]["af"]["strong_fvl_floor"],
        key: value,
    }
    with pytest.raises(TriageRulesError):
        parse_triage_rules(data)


@pytest.mark.parametrize("confounder", [C.PORPHYRIN_RED, C.GLARE])
def test_cleanable_confounder_adds_clean_and_retake_step(confounder):
    sites = [
        site(S.BUCCAL_MUCOSA_LEFT, [region(mean_rel=0.70, confounders=[(confounder, 0.6)])]),
        site(S.BUCCAL_MUCOSA_RIGHT),
    ]
    out = triage(sites)
    assert out.triage.overall_risk_level == RiskLevel.LOW
    assert any(CLEAN_STEP in s and "Left buccal mucosa" in s for s in steps(out))
    assert any(CLEAN_STEP in s for s in out.lesions[0].recommended_action.steps)


def test_pigmentation_or_blood_do_not_ask_for_cleaning():
    sites = [
        site(
            S.BUCCAL_MUCOSA_LEFT,
            [region(mean_rel=0.70, confounders=[(C.PIGMENTATION, 0.6), (C.BLOOD, 0.05)])],
        ),
        site(S.BUCCAL_MUCOSA_RIGHT),
    ]
    assert not any(CLEAN_STEP in s for s in steps(triage(sites)))


def test_clean_step_never_delays_a_high_referral():
    sites = [
        site(
            S.BUCCAL_MUCOSA_LEFT,
            [strong_region(confounders=[(C.GLARE, 0.2)])],
            appearance=[V.GROWTH],
        ),
        site(S.BUCCAL_MUCOSA_RIGHT),
    ]
    out = triage(sites)
    assert out.triage.overall_risk_level == RiskLevel.HIGH
    (step,) = [s for s in steps(out) if CLEAN_STEP in s]
    assert "do not delay the referral" in step


def test_low_explanation_never_says_below_re_examination_level():
    out = triage([site(S.BUCCAL_MUCOSA_LEFT, [region(mean_rel=0.70)]), site(S.BUCCAL_MUCOSA_RIGHT)])
    assert out.triage.overall_risk_level == RiskLevel.LOW
    assert "below the level that needs re-examination" not in out.triage.explanation


# --------------------------------------------------------------------------------------------
# clinical.1: no site may be unable to reach MODERATE (or HIGH at a high-risk site) from AF
# --------------------------------------------------------------------------------------------


def test_lower_gingiva_is_high_risk_and_not_low_specificity():
    rules = parse_rules_default()
    assert OralSite.LOWER_GINGIVA in rules.site.high_risk_sites
    assert OralSite.LOWER_GINGIVA not in rules.site.af_low_specificity_sites


@pytest.mark.parametrize("oral_site", list(OralSite), ids=lambda s: s.value)
def test_max_attainable_af_only_level(oral_site: OralSite):
    """Strongest AF evidence, the best reference the site can have, diascopy confirming it,
    a 30-year-old with no habits and no visual flag: never LOW anywhere, and HIGH at every
    high-risk site. Checked on the score itself, not only through the strong-FVL floor."""
    rules = parse_rules_default()
    strongest = region(mean_rel=0.10, fraction=0.40, border=0.8, contrast=8.0)
    sites = [site(oral_site, [strongest], diascopy=Diascopy.FVL_PERSISTS)]
    paired, _ = reference_site_for(oral_site)
    if paired is not None:
        sites.append(site(paired))
    (lesion,) = triage(sites).lesions
    assert lesion.score >= rules.moderate_min_score, lesion.contributing_factors
    assert at_least(lesion.risk_level, RiskLevel.MODERATE)
    if oral_site in rules.site.high_risk_sites:
        assert lesion.score >= rules.high_min_score, lesion.contributing_factors
        assert lesion.risk_level == RiskLevel.HIGH


@pytest.mark.parametrize(
    ("ref", "applied", "skipped"),
    [
        (ReferenceType.IN_IMAGE_AUTO, "af.low_specificity_site", "af.reference.in_image_auto"),
        (ReferenceType.NONE, "af.reference.none", "af.low_specificity_site"),
    ],
)
def test_unpaired_site_takes_only_the_larger_discount(ref, applied, skipped):
    rules = parse_rules_default()
    out = triage([site(S.TONGUE_DORSUM, [region(mean_rel=0.70)], ref_type=ref)])
    f = out.lesions[0].contributing_factors
    assert applied in codes(f)
    assert skipped not in codes(f)
    note = factor(f, "af.discount_not_compounded")
    assert note.points == 0.0
    mult = min(rules.af.reference_multipliers[ref], rules.site.af_low_specificity_multiplier)
    assert out.lesions[0].score == pytest.approx((1.5 + 0.375) * mult, abs=1e-3)


def test_paired_low_specificity_site_still_compounds():
    out = triage(
        [
            site(S.UPPER_GINGIVA, [region(mean_rel=0.70)], ref_type=ReferenceType.ANALOGOUS_SITE),
            site(S.LOWER_GINGIVA),
        ]
    )
    f = out.lesions[0].contributing_factors
    assert {"af.reference.analogous_site", "af.low_specificity_site"} <= set(codes(f))


# --------------------------------------------------------------------------------------------
# clinical.4: any lesion the health worker saw is at least MODERATE
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("oral_site", "appearance", "seen"),
    [
        (S.BUCCAL_MUCOSA_LEFT, [], True),
        (S.SOFT_PALATE, [V.OTHER], True),
        (S.SOFT_PALATE, [V.OTHER], False),  # appearance recorded, checkbox missed
    ],
)
def test_any_seen_lesion_is_at_least_moderate(oral_site, appearance, seen):
    rf = risk(smokeless_tobacco={"uses": True, "types": ["gutka"]})
    out = triage([site(oral_site, appearance=appearance, lesion_seen=seen)], age=55, rf=rf)
    (lesion,) = out.lesions
    assert lesion.risk_level == RiskLevel.MODERATE
    assert "visual.lesion_seen" in codes(lesion.contributing_factors)
    t = out.triage
    assert t.overall_risk_level == RiskLevel.MODERATE
    applied = factor(t.overrides_applied, "visual.lesion_seen")
    assert applied.label == (
        "Abnormality seen by the health worker: examine again in 2 weeks; refer if it persists"
    )
    assert "below the level" not in t.explanation


def test_seen_lesion_floor_applies_at_reference_only_and_unreadable_sites():
    from mukha_pipeline.schema.models import SiteRole

    ref = triage(
        [
            site(S.BUCCAL_MUCOSA_LEFT),
            site(S.BUCCAL_MUCOSA_RIGHT, role=SiteRole.REFERENCE_ONLY, lesion_seen=True),
        ]
    )
    assert ref.triage.overall_risk_level == RiskLevel.MODERATE
    failed = triage([site(S.BUCCAL_MUCOSA_LEFT, analysed=False, lesion_seen=True)])
    assert failed.triage.overall_risk_level == RiskLevel.MODERATE
    assert failed.lesions[0].risk_level == RiskLevel.MODERATE
    assert "rests on the clinical findings" in failed.triage.explanation


def test_visual_lesion_seen_must_be_a_bool():
    data = default_rules().data
    bad = [*data["overrides"], {"code": "x", "label": "x", "floor": "MODERATE"}]
    bad[-1]["when"] = {"visual_lesion_seen": "yes"}
    with pytest.raises(TriageRulesError):
        parse_triage_rules({**data, "overrides": bad})


# --------------------------------------------------------------------------------------------
# clinical.5: every lesion-role probe reading is a lesion; unimaged lesions are never LOW
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("label", ["not_suspicious", None])
def test_lesion_reading_at_unimaged_site_is_never_dropped(label):
    sites = [site(S.BUCCAL_MUCOSA_RIGHT), site(S.LATERAL_TONGUE_RIGHT)]
    spectral = [reading("s1", S.LATERAL_TONGUE_LEFT, hb_delta=0.0, label=label)]
    out = triage(sites, spectral)
    (lesion,) = out.lesions
    assert lesion.origins == ["spectral"] and lesion.site == S.LATERAL_TONGUE_LEFT
    assert lesion.risk_level == RiskLevel.INCONCLUSIVE
    assert out.triage.overall_risk_level == RiskLevel.INCONCLUSIVE
    assert out.site_levels[S.LATERAL_TONGUE_LEFT] == RiskLevel.INCONCLUSIVE
    assert any(
        s.startswith("Take white-light and violet-light photos of Left lateral tongue")
        for s in steps(out)
    )


def test_qc_failed_lesion_reading_at_unimaged_site_is_inconclusive():
    spectral = [reading("s1", S.LATERAL_TONGUE_LEFT, label=None, qc_passed=False)]
    out = triage([site(S.BUCCAL_MUCOSA_RIGHT)], spectral)
    assert out.triage.overall_risk_level == RiskLevel.INCONCLUSIVE
    assert any("etake the probe reading" in s for s in steps(out))


def test_qc_failed_lesion_reading_alone_is_not_low():
    spectral = [reading("s1", S.BUCCAL_MUCOSA_LEFT, label=None, qc_passed=False)]
    out = triage([site(S.BUCCAL_MUCOSA_LEFT), site(S.BUCCAL_MUCOSA_RIGHT)], spectral)
    (lesion,) = out.lesions
    assert lesion.risk_level == RiskLevel.INCONCLUSIVE
    assert out.triage.overall_risk_level == RiskLevel.INCONCLUSIVE
    assert any("etake the probe reading" in s and "Left buccal mucosa" in s for s in steps(out))


def test_lesion_reading_at_unreadable_site_is_inconclusive():
    spectral = [reading("s1", S.BUCCAL_MUCOSA_LEFT, hb_delta=0.0, label="not_suspicious")]
    out = triage(
        [site(S.BUCCAL_MUCOSA_LEFT, af_status="insufficient_mucosa"), site(S.BUCCAL_MUCOSA_RIGHT)],
        spectral,
    )
    assert out.lesions[0].risk_level == RiskLevel.INCONCLUSIVE
    assert out.triage.overall_risk_level == RiskLevel.INCONCLUSIVE


def test_not_suspicious_label_is_never_described_as_reassurance():
    spectral = [reading("s1", S.BUCCAL_MUCOSA_LEFT, hb_delta=0.0, label="not_suspicious")]
    out = triage([site(S.BUCCAL_MUCOSA_LEFT), site(S.BUCCAL_MUCOSA_RIGHT)], spectral)
    (lesion,) = out.lesions
    assert lesion.origins == ["spectral"]
    texts = " ".join(all_texts(out)).lower()
    assert "not suspicious" not in texts and "not_suspicious" not in texts


# --------------------------------------------------------------------------------------------
# clinical.7: say plainly when the overall level comes from outside the device finding
# --------------------------------------------------------------------------------------------


def test_level_raised_by_habits_is_explained_and_lesion_keeps_its_level():
    rf = risk(
        smokeless_tobacco={"uses": True, "types": ["gutka"]},
        areca_nut={"uses": True},
        alcohol={"uses": True, "frequency": "daily"},
    )
    sites = [site(S.BUCCAL_MUCOSA_LEFT, [region(mean_rel=0.68)]), site(S.BUCCAL_MUCOSA_RIGHT)]
    out = triage(sites, age=52, rf=rf)
    (lesion,) = out.lesions
    assert lesion.risk_level == RiskLevel.LOW
    assert out.site_levels[S.BUCCAL_MUCOSA_LEFT] == RiskLevel.LOW
    t = out.triage
    assert t.overall_risk_level == RiskLevel.MODERATE
    assert "Overall MODERATE because habits and history" in t.explanation
    assert "the finding at the left buccal mucosa scored LOW on its own" in t.explanation
    action = lesion.recommended_action
    assert "No high-risk features" not in action.summary
    assert action.code == ActionCode.REEXAMINE_2_WEEKS
    assert "overall result is MODERATE" in action.summary


def test_level_raised_by_floor_is_explained():
    rf = risk(symptoms={"non_healing_ulcer_over_2_weeks": True})
    out = triage(
        [site(S.BUCCAL_MUCOSA_LEFT, [region(mean_rel=0.70)]), site(S.BUCCAL_MUCOSA_RIGHT)], rf=rf
    )
    t = out.triage
    assert t.overall_risk_level == RiskLevel.HIGH
    assert out.lesions[0].risk_level == RiskLevel.LOW
    assert (
        "Overall HIGH because of a non-healing ulcer reported for more than 2 weeks, even though "
        "the finding at the left buccal mucosa scored LOW on its own."
    ) in t.explanation
    assert out.lesions[0].recommended_action.code == ActionCode.URGENT_REFERRAL


def test_moderate_lesion_under_high_overall_points_to_overall_action():
    rf = risk(symptoms={"neck_lump": True})
    out = triage([site(S.BUCCAL_MUCOSA_LEFT, appearance=[V.WHITE_PATCH])], rf=rf)
    (lesion,) = out.lesions
    assert lesion.risk_level == RiskLevel.MODERATE
    assert out.triage.overall_risk_level == RiskLevel.HIGH
    assert "Overall HIGH because of a lump in the neck" in out.triage.explanation
    assert any("overall result is HIGH" in s for s in lesion.recommended_action.steps)


# --------------------------------------------------------------------------------------------
# clinical.8: diascopy is one test on one spot
# --------------------------------------------------------------------------------------------


def test_diascopy_resolves_not_applied_when_site_has_several_regions():
    sites = [
        site(
            S.BUCCAL_MUCOSA_LEFT,
            [region("r1", mean_rel=0.70, fraction=0.03), strong_region(region_id="r2")],
            diascopy=Diascopy.FVL_RESOLVES,
        ),
        site(S.BUCCAL_MUCOSA_RIGHT),
    ]
    out = triage(sites)
    r1, r2 = out.lesions
    for lesion in (r1, r2):
        assert "af.diascopy.fvl_resolves" not in codes(lesion.contributing_factors)
        assert factor(lesion.contributing_factors, "af.diascopy.not_applied").points == 0.0
        assert "glass-slide pressure" not in lesion.summary
    assert r2.score == pytest.approx(5.5 * 1.15, abs=1e-3)
    assert r2.risk_level == RiskLevel.HIGH


def test_diascopy_resolves_applies_to_a_single_region():
    out = triage(
        [
            site(S.BUCCAL_MUCOSA_LEFT, [region(mean_rel=0.70)], diascopy=Diascopy.FVL_RESOLVES),
            site(S.BUCCAL_MUCOSA_RIGHT),
        ]
    )
    (lesion,) = out.lesions
    assert "af.diascopy.fvl_resolves" in codes(lesion.contributing_factors)
    assert "returned under glass-slide pressure" in lesion.summary


def test_diascopy_persists_applies_to_the_darkest_region_only():
    sites = [
        site(
            S.BUCCAL_MUCOSA_LEFT,
            [region("r1", mean_rel=0.70), region("r2", mean_rel=0.60)],
            diascopy=Diascopy.FVL_PERSISTS,
        ),
        site(S.BUCCAL_MUCOSA_RIGHT),
    ]
    r1, r2 = triage(sites).lesions
    assert "af.diascopy.fvl_persists" not in codes(r1.contributing_factors)
    assert "af.diascopy.not_applied" in codes(r1.contributing_factors)
    assert "af.diascopy.fvl_persists" in codes(r2.contributing_factors)


# --------------------------------------------------------------------------------------------
# clinical.9 + honesty.0: symptoms and history are never silently dropped
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("rf", "code", "phrase"),
    [
        ({"symptoms": {"lump": True}}, "symptom.lump", "lump in the mouth"),
        (
            {"symptoms": {"burning_sensation": True}},
            "symptom.burning_sensation",
            "possible early submucous fibrosis",
        ),
        ({"symptoms": {"pain": True}}, "symptom.pain", "persistent pain"),
        (
            {"prior_oral_cancer": True},
            "history.prior_oral_cancer",
            "regular specialist follow-up",
        ),
    ],
)
def test_symptom_and_history_floors_without_a_lesion(rf, code, phrase):
    out = triage([site(S.BUCCAL_MUCOSA_LEFT), site(S.BUCCAL_MUCOSA_RIGHT)], age=55, rf=risk(**rf))
    t = out.triage
    assert out.lesions == []
    assert t.overall_risk_level == RiskLevel.MODERATE
    assert code in codes(t.overrides_applied)
    assert phrase in t.explanation.lower()
    assert any(phrase in s.lower() for s in steps(out))


def test_reported_symptoms_appear_as_zero_point_factors():
    rf = risk(symptoms={"lump": True, "pain": True})
    out = triage([site(S.BUCCAL_MUCOSA_LEFT)], rf=rf)
    f = out.triage.contributing_factors
    lump = factor(f, "symptom.reported.lump")
    assert lump.points == 0.0 and lump.source == FactorSource.SYMPTOM
    assert factor(f, "symptom.reported.pain").points == 0.0


def test_low_with_history_names_it_and_never_says_no_referral():
    out = triage(
        [site(S.BUCCAL_MUCOSA_LEFT), site(S.BUCCAL_MUCOSA_RIGHT)],
        age=55,
        rf=risk(prior_oral_lesion=True),
    )
    t = out.triage
    assert t.overall_risk_level == RiskLevel.LOW
    assert "previous oral lesion" in t.explanation
    assert "sooner if any symptom appears" not in t.explanation
    assert any("examine the whole mouth" in s for s in steps(out))
    assert all("no referral" not in x.lower() for x in all_texts(out))


def test_low_with_a_symptom_under_edited_rules_names_the_symptom(tmp_path):
    """If a clinician removes a symptom floor, LOW must still name the symptom."""

    def drop_pain(d: dict) -> None:
        d["overrides"] = [o for o in d["overrides"] if o["code"] != "symptom.pain"]

    rules = write_rules(tmp_path, drop_pain)
    rf = risk(symptoms={"pain": True}, smokeless_tobacco={"uses": True, "types": ["gutka"]})
    out = triage([site(S.BUCCAL_MUCOSA_LEFT), site(S.BUCCAL_MUCOSA_RIGHT)], rf=rf, rules=rules)
    t = out.triage
    assert t.overall_risk_level == RiskLevel.LOW
    assert "pain in the mouth" in t.explanation
    assert "sooner if any symptom appears" not in t.explanation
    assert "stopping gutka" in t.explanation.lower()
    assert any("pain in the mouth" in s and "examine" in s for s in steps(out))
    assert all("no referral" not in x.lower() for x in all_texts(out))


def test_low_without_symptoms_keeps_the_symptom_advice():
    out = triage([site(S.BUCCAL_MUCOSA_LEFT), site(S.BUCCAL_MUCOSA_RIGHT)])
    assert "sooner if any symptom appears" in out.triage.explanation


# --------------------------------------------------------------------------------------------
# clinical.10: recorded habit details count as use
# --------------------------------------------------------------------------------------------


def test_habit_details_without_uses_count_as_current_use():
    sites = [site(S.BUCCAL_MUCOSA_LEFT, [region(mean_rel=0.70)]), site(S.BUCCAL_MUCOSA_RIGHT)]
    rf = risk(
        smokeless_tobacco={"types": ["gutka"], "years": 20},
        smoking={"per_day": 10},
        areca_nut={"years": 5},
    )
    out = triage(sites, age=30, rf=rf)
    f = out.triage.contributing_factors
    assert factor(f, "history.smokeless_tobacco").points == 1.0
    assert factor(f, "history.smoking").points == 0.5
    assert factor(f, "history.areca_nut").points == 0.75
    assert any(s.startswith("Habits reported (gutka") for s in steps(out))


def test_former_use_with_details_stays_former():
    sites = [site(S.BUCCAL_MUCOSA_LEFT, [region(mean_rel=0.70)])]
    rf = risk(smokeless_tobacco={"former": True, "types": ["gutka"], "years": 20})
    f = triage(sites, rf=rf).triage.contributing_factors
    assert "history.smokeless_tobacco_former" in codes(f)
    assert "history.smokeless_tobacco" not in codes(f)


def test_zero_years_alone_is_not_use():
    rf = risk(smokeless_tobacco={"years": 0, "per_day": 0})
    out = triage([site(S.BUCCAL_MUCOSA_LEFT, [region(mean_rel=0.70)])], rf=rf)
    assert not [c for c in codes(out.triage.contributing_factors) if "smokeless" in c]


# --------------------------------------------------------------------------------------------
# honesty.11: the reference tissue is never called healthy
# --------------------------------------------------------------------------------------------


def test_reference_tissue_is_never_called_healthy():
    sites = [
        site(S.TONGUE_DORSUM, [region()], ref_type=ReferenceType.IN_IMAGE_AUTO),
        site(S.BUCCAL_MUCOSA_LEFT, [region(mean_rel=0.6)]),
        site(S.BUCCAL_MUCOSA_RIGHT),
    ]
    spectral = [
        reading("s1", S.TONGUE_DORSUM),  # no reference reading
        reading("s2", S.BUCCAL_MUCOSA_LEFT, hb_delta=0.08, ratio_delta=-0.12),
    ]
    out = triage(sites, spectral)
    assert "spectral_no_reference" in limitation_codes(out)
    assert "reference_in_image" in limitation_codes(out)
    texts = all_texts(out)
    assert any("blood-absorption dip" in x for x in texts)
    assert [x for x in texts if "healthy" in x.lower()] == []


# --------------------------------------------------------------------------------------------
# correctness.1: unreliable segmentation never allows LOW, but its findings still count
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("status", ["low_confidence", "failed"])
def test_unreliable_segmentation_blocks_low(status):
    out = triage([site(S.BUCCAL_MUCOSA_LEFT, seg_status=status), site(S.BUCCAL_MUCOSA_RIGHT)])
    t = out.triage
    assert t.overall_risk_level == RiskLevel.INCONCLUSIVE
    assert out.site_levels[S.BUCCAL_MUCOSA_LEFT] == RiskLevel.INCONCLUSIVE
    assert out.site_levels[S.BUCCAL_MUCOSA_RIGHT] == RiskLevel.LOW
    assert "segmentation_unreliable" in limitation_codes(out)
    assert any("Left buccal mucosa" in s for s in steps(out))


def test_unreliable_segmentation_alone_is_inconclusive_not_low():
    out = triage([site(S.BUCCAL_MUCOSA_LEFT, seg_status="failed")])
    assert out.triage.overall_risk_level == RiskLevel.INCONCLUSIVE


def test_findings_at_unreliable_segmentation_site_still_count():
    out = triage(
        [
            site(S.BUCCAL_MUCOSA_LEFT, [strong_region()], seg_status="low_confidence"),
            site(S.BUCCAL_MUCOSA_RIGHT, seg_status="ok"),
        ]
    )
    (lesion,) = out.lesions
    assert lesion.risk_level == RiskLevel.HIGH
    assert out.triage.overall_risk_level == RiskLevel.HIGH
    assert "qc.segmentation_unreliable" in codes(lesion.contributing_factors)
    mild = triage([site(S.BUCCAL_MUCOSA_LEFT, [region(mean_rel=0.70)], seg_status="failed")])
    assert mild.lesions[0].risk_level == RiskLevel.INCONCLUSIVE
    assert mild.triage.overall_risk_level == RiskLevel.INCONCLUSIVE


def test_moderate_finding_at_only_site_with_unreliable_segmentation_stays_moderate():
    out = triage(
        [site(S.BUCCAL_MUCOSA_LEFT, [region(mean_rel=0.70, fraction=0.06, contrast=1.5)],
              seg_status="low_confidence")]
    )  # fmt: skip
    assert out.lesions[0].risk_level == RiskLevel.MODERATE
    assert out.triage.overall_risk_level == RiskLevel.MODERATE
    assert "device_analysis_unavailable" not in limitation_codes(out)


# --------------------------------------------------------------------------------------------
# Rules file
# --------------------------------------------------------------------------------------------


def test_rules_version_bumped():
    rules = parse_rules_default()
    assert rules.version == "1.1.0"
    assert all(find_positioning_violations(o.label) == [] for o in rules.overrides)
    assert rules.af.strong_fvl_floor is not None
    assert rules.af.strong_fvl_floor.floor == RiskLevel.MODERATE


def test_strong_floor_label_is_not_repeated_in_the_explanation():
    sites = [
        site(S.BUCCAL_MUCOSA_LEFT, [strong_region(confounders=[(C.PORPHYRIN_RED, 1.0)])]),
        site(S.BUCCAL_MUCOSA_RIGHT),
    ]
    t = triage(sites).triage
    assert t.explanation == (
        "MODERATE risk: strong loss of fluorescence at the left buccal mucosa (relative AF 0.30 "
        "vs the right buccal mucosa)."
    )


def test_unchecked_finding_asks_for_missing_photos_or_probe_readings():
    spectral = [reading("s1", S.BUCCAL_MUCOSA_LEFT, label=None, qc_passed=False)]
    out = triage([site(S.BUCCAL_MUCOSA_LEFT), site(S.BUCCAL_MUCOSA_RIGHT)], spectral)
    assert "take the missing photos or probe readings now" in out.triage.explanation
    assert "retake the images" not in out.triage.explanation
    assert out.lesions[0].summary.startswith(
        "Finding at the left buccal mucosa that the probe could not measure"
    )


def test_strong_floor_can_be_removed_and_must_be_well_formed(tmp_path):
    def drop(d: dict) -> None:
        del d["lesion"]["af"]["strong_fvl_floor"]

    rules = write_rules(tmp_path, drop)
    assert parse_triage_rules(rules.data).af.strong_fvl_floor is None
    sites = [
        site(S.BUCCAL_MUCOSA_LEFT, [strong_region(confounders=[(C.PORPHYRIN_RED, 1.0)])]),
        site(S.BUCCAL_MUCOSA_RIGHT),
    ]
    assert triage(sites, rules=rules).lesions[0].risk_level == RiskLevel.LOW
    for bad in ("yes", {**default_rules().data["lesion"]["af"]["strong_fvl_floor"], "extra": 1}):
        data = default_rules().data
        data = {**data, "lesion": {**data["lesion"], "af": {**data["lesion"]["af"]}}}
        data["lesion"]["af"]["strong_fvl_floor"] = bad
        with pytest.raises(TriageRulesError):
            parse_triage_rules(data)


def test_low_with_several_findings_names_the_strongest():
    sites = [
        site(S.BUCCAL_MUCOSA_LEFT, [region("r1", mean_rel=0.72), region("r2", mean_rel=0.70)]),
        site(S.BUCCAL_MUCOSA_RIGHT),
    ]
    t = triage(sites).triage
    assert t.overall_risk_level == RiskLevel.LOW
    assert t.explanation.startswith(
        "All findings scored low on this screen; the strongest is mild loss of fluorescence at "
        "the left buccal mucosa (relative AF 0.70"
    )


def test_low_with_habits_only_keeps_symptom_advice_and_cessation_sentence():
    rf = risk(smokeless_tobacco={"uses": True, "types": ["gutka"]})
    t = triage([site(S.BUCCAL_MUCOSA_LEFT), site(S.BUCCAL_MUCOSA_RIGHT)], rf=rf).triage
    assert t.explanation.endswith(
        "sooner if any symptom appears. Stopping gutka is the most important step to lower "
        "future risk."
    )
