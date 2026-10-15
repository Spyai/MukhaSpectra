"""Per-lesion scoring: AF points and multipliers, WL model, spectral deltas, visual flags."""

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
    wl_unavailable,
)

from mukha_pipeline.fusion import TriageOutput, run_triage
from mukha_pipeline.schema.models import (
    ConfounderType,
    Diascopy,
    FactorSource,
    ReferenceType,
    RiskLevel,
    SiteResult,
    SpectralReadingResult,
    VisualAppearance,
    WLAnalysis,
    WLLabel,
    WLPrediction,
)

V = VisualAppearance
C = ConfounderType


def triage(
    sites: list[SiteResult], spectral: list[SpectralReadingResult] | None = None
) -> TriageOutput:
    out = run_triage(sites, spectral or [], patient(), risk(), default_rules())
    assert_invariants(out)
    return out


def only_lesion(sites: list[SiteResult], spectral=None):
    out = triage(sites, spectral)
    assert len(out.lesions) == 1, [lesion.summary for lesion in out.lesions]
    return out.lesions[0]


def base_region(**kw):
    params = dict(mean_rel=0.45, area_mm2=60.0, border=0.5, contrast=1.8)
    params.update(kw)
    return region(**params)


# Lateral tongue: contralateral reference by default, high-risk site, not low-specificity.
TONGUE = S.LATERAL_TONGUE_LEFT


# --------------------------------------------------------------------------------------------
# AF points
# --------------------------------------------------------------------------------------------


def test_af_points_breakdown():
    lesion = only_lesion([site(TONGUE, [base_region()])])
    f = lesion.contributing_factors
    assert factor(f, "af.fvl_present").points == 1.5
    assert factor(f, "af.depth").points == 2.0  # 0.75 * 3.0 capped at 2.0
    assert factor(f, "af.area").points == 1.0
    assert factor(f, "af.border_irregular").points == 0.5
    assert factor(f, "af.contrast").points == 0.5
    assert factor(f, "site.high_risk").points == pytest.approx(5.5 * 0.15, abs=1e-4)
    assert lesion.score == pytest.approx(5.5 * 1.15, abs=1e-4)
    assert lesion.origins == ["af_fvl"]
    assert (
        "Fluorescence 0.45 of the reference, below the 0.75 cut-off" in factor(f, "af.depth").label
    )


@pytest.mark.parametrize(
    ("area_mm2", "fraction", "expected"),
    [
        (5.0, 0.5, 0.0),  # mm² known: fraction ignored even if large
        (10.0, 0.0, 0.5),
        (49.9, 0.0, 0.5),
        (50.0, 0.0, 1.0),
        (None, 0.04, 0.0),
        (None, 0.05, 0.5),
        (None, 0.15, 1.0),
    ],
)
def test_area_tiers_use_mm2_when_known_else_fraction(area_mm2, fraction, expected):
    lesion = only_lesion([site(TONGUE, [region(area_mm2=area_mm2, fraction=fraction)])])
    area = [f.points for f in lesion.contributing_factors if f.code == "af.area"]
    assert sum(area) == expected


def test_depth_uses_site_threshold_and_ignores_regions_above_it():
    lesion = only_lesion([site(TONGUE, [region(mean_rel=0.74)])])
    assert factor(lesion.contributing_factors, "af.depth").points == pytest.approx(0.075)
    lesion = only_lesion([site(TONGUE, [region(mean_rel=0.80)])])
    assert "af.depth" not in codes(lesion.contributing_factors)


def test_porphyrin_confounder_lowers_score():
    clean = only_lesion([site(TONGUE, [base_region()])])
    plaque = only_lesion([site(TONGUE, [base_region(confounders=[(C.PORPHYRIN_RED, 0.5)])])])
    assert plaque.score < clean.score
    f = factor(plaque.contributing_factors, "af.confounders")
    assert f.points == pytest.approx(5.5 * (0.7 - 1.0), abs=1e-4)
    assert "red plaque fluorescence (porphyrin) over 50% of it" in f.label
    assert plaque.confounders[0].type == C.PORPHYRIN_RED


def test_confounder_multiplier_is_floored():
    confs = [(C.PORPHYRIN_RED, 1.0), (C.BLOOD, 1.0), (C.GLARE, 1.0)]
    lesion = only_lesion([site(TONGUE, [base_region(confounders=confs)])])
    f = factor(lesion.contributing_factors, "af.confounders")
    assert f.points == pytest.approx(5.5 * (0.2 - 1.0), abs=1e-4)
    assert f.detail.startswith("x0.20")


def test_diascopy_resolves_lowers_and_persists_raises():
    base = only_lesion([site(TONGUE, [base_region()])])
    resolves = only_lesion([site(TONGUE, [base_region()], diascopy=Diascopy.FVL_RESOLVES)])
    persists = only_lesion([site(TONGUE, [base_region()], diascopy=Diascopy.FVL_PERSISTS)])
    assert resolves.score < base.score < persists.score
    assert factor(resolves.contributing_factors, "af.diascopy.fvl_resolves").points == (
        pytest.approx(5.5 * -0.7, abs=1e-4)
    )
    # The discount still orders lesions, but strong FVL is never taken below MODERATE.
    assert resolves.score < 3.0
    assert resolves.risk_level == RiskLevel.MODERATE
    assert "af.strong_fvl_floor" in codes(resolves.contributing_factors)
    assert "af.diascopy.not_done" not in codes(base.contributing_factors)


def test_in_image_reference_lowers_vs_contralateral():
    contra = only_lesion([site(TONGUE, [base_region()])])
    in_image = triage([site(TONGUE, [base_region()], ref_type=ReferenceType.IN_IMAGE_AUTO)])
    lesion = in_image.lesions[0]
    assert lesion.score < contra.score
    mult = parse_rules_default().af.reference_multipliers[ReferenceType.IN_IMAGE_AUTO]
    assert mult == 0.8
    assert factor(lesion.contributing_factors, "af.reference.in_image_auto").points == (
        pytest.approx(5.5 * (mult - 1.0), abs=1e-4)
    )
    assert "reference_in_image" in limitation_codes(in_image)


@pytest.mark.parametrize(
    ("ref", "mult"), [(ReferenceType.ANALOGOUS_SITE, 0.9), (ReferenceType.NONE, 0.5)]
)
def test_other_reference_types(ref, mult):
    lesion = only_lesion([site(TONGUE, [base_region()], ref_type=ref)])
    f = factor(lesion.contributing_factors, f"af.reference.{ref.value}")
    assert f.points == pytest.approx(5.5 * (mult - 1.0), abs=1e-4)


def test_low_specificity_site_lowers():
    lip = only_lesion([site(S.UPPER_LIP_MUCOSA, [base_region()])])
    gum = only_lesion([site(S.UPPER_GINGIVA, [base_region()])])
    assert gum.score == pytest.approx(lip.score * 0.6, abs=1e-3)
    assert (
        "inflammation, coating or keratin"
        in factor(gum.contributing_factors, "af.low_specificity_site").label
    )
    lower = only_lesion([site(S.LOWER_GINGIVA, [base_region()])])
    lower_codes = set(codes(lower.contributing_factors))
    assert "site.high_risk" in lower_codes and "af.low_specificity_site" not in lower_codes


def test_multiplier_order_is_reference_diascopy_confounders_then_site():
    lesion = only_lesion(
        [
            site(
                S.UPPER_GINGIVA,
                [base_region(confounders=[(C.PIGMENTATION, 0.5)])],
                ref_type=ReferenceType.IN_IMAGE_AUTO,
                diascopy=Diascopy.FVL_PERSISTS,
            )
        ]
    )
    order = [c for c in codes(lesion.contributing_factors) if "." in c]
    assert order[-5:] == [
        "af.reference.in_image_auto",
        "af.diascopy.fvl_persists",
        "af.confounders",
        "af.low_specificity_site",
        "af.strong_fvl_floor",  # zero-point floor record, after the points
    ]
    assert lesion.score == pytest.approx(5.5 * 0.8 * 1.2 * 0.7 * 0.6, abs=1e-3)


def test_high_risk_multiplier_comes_last():
    lesion = only_lesion(
        [
            site(
                TONGUE,
                [base_region(confounders=[(C.PIGMENTATION, 0.5)])],
                ref_type=ReferenceType.IN_IMAGE_AUTO,
                diascopy=Diascopy.FVL_PERSISTS,
            )
        ]
    )
    order = [c for c in codes(lesion.contributing_factors) if "." in c]
    assert order[-5:] == [
        "af.diascopy.fvl_persists",
        "af.confounders",
        "site.high_risk",
        "af.strong_fvl_floor",  # zero-point floor records, after the points
        "af.high_risk_site_fvl_floor",
    ]
    assert lesion.score == pytest.approx(5.5 * 0.8 * 1.2 * 0.7 * 1.15, abs=1e-3)
    # Any FVL at a high-risk site is at least MODERATE (rules 1.2.0).
    assert lesion.risk_level == RiskLevel.MODERATE
    assert factor(lesion.contributing_factors, "af.high_risk_site_fvl_floor").points == 0.0


# --------------------------------------------------------------------------------------------
# WL model
# --------------------------------------------------------------------------------------------


def test_wl_unavailable_contributes_nothing():
    with_wl = triage([site(TONGUE, [base_region()], wl=wl_unavailable())])
    assert "wl_model" not in with_wl.lesions[0].origins
    assert not [f for f in with_wl.lesions[0].contributing_factors if f.source == "wl_model"]
    assert "wl_model_unavailable" in limitation_codes(with_wl)
    no_lesion = triage([site(TONGUE, wl=wl_unavailable())])
    assert no_lesion.lesions == []


def test_wl_ok_red_patch_adds_points_to_primary_lesion():
    sites = [
        site(
            TONGUE,
            [region("r1", mean_rel=0.70), region("r2", mean_rel=0.50)],
            wl=wl_ok(WLLabel.RED_PATCH, 0.8),
        )
    ]
    out = triage(sites)
    r1, r2 = out.lesions
    assert "wl_model" not in r1.origins
    assert r2.origins == ["af_fvl", "wl_model"]
    f = factor(r2.contributing_factors, "wl.red_patch")
    assert f.points == 3.0 and f.source == FactorSource.WL_MODEL
    assert "unvalidated" in f.label and "0.80" in f.label
    assert "wl_model_unavailable" not in limitation_codes(out)


def test_wl_creates_its_own_lesion_when_nothing_else_at_site():
    lesion = only_lesion([site(TONGUE, wl=wl_ok(WLLabel.WHITE_PATCH, 0.6))])
    assert lesion.origins == ["wl_model"]
    assert lesion.score == pytest.approx(1.5 * 1.15, abs=1e-4)
    assert "photo model suggests a white patch" in lesion.summary


@pytest.mark.parametrize(
    "wl",
    [
        wl_ok(WLLabel.RED_PATCH, 0.4),  # below min_probability
        wl_ok(WLLabel.NORMAL, 0.9),
        WLAnalysis(**{**wl_ok(WLLabel.RED_PATCH, 0.9).model_dump(), "status": "error"}),
        WLAnalysis(
            **{
                **wl_ok(WLLabel.RED_PATCH, 0.9).model_dump(),
                "predictions": [WLPrediction(label=WLLabel.NORMAL, probability=1.0)],
            }
        ),  # top label without a probability cannot be trusted
    ],
)
def test_wl_ignored_when_not_confident_or_not_ok(wl):
    assert triage([site(TONGUE, wl=wl)]).lesions == []


def test_wl_top_label_missing_uses_highest_probability():
    lesion = only_lesion([site(TONGUE, wl=wl_ok(WLLabel.ULCER_MASS, 0.7, set_top=False))])
    assert factor(lesion.contributing_factors, "wl.ulcer_mass").points == 3.0


# --------------------------------------------------------------------------------------------
# Spectral
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("hb", "ratio", "hb_pts", "ratio_pts"),
    [
        (0.02, 0.0, 0.0, 0.0),
        (0.04, -0.06, 0.5, 0.5),
        (0.07, -0.12, 1.0, 1.0),
        (0.03, -0.05, 0.5, 0.5),
    ],
)
def test_spectral_delta_tiers(hb, ratio, hb_pts, ratio_pts):
    spectral = [reading("s1", TONGUE, hb_delta=hb, ratio_delta=ratio)]
    lesion = only_lesion([site(TONGUE, [region()])], spectral)
    pts = {f.code: f.points for f in lesion.contributing_factors}
    assert pts.get("spectral.hb_absorption", 0.0) == hb_pts
    assert pts.get("spectral.ratio_555_630", 0.0) == ratio_pts
    assert lesion.spectral_reading_ids == ["s1"]
    assert "spectral" in lesion.origins


def test_spectral_best_tier_over_readings_not_sum():
    spectral = [
        reading("s1", TONGUE, hb_delta=0.04, ratio_delta=-0.12),
        reading("s2", TONGUE, hb_delta=0.08, ratio_delta=0.0),
    ]
    lesion = only_lesion([site(TONGUE, [region()])], spectral)
    assert factor(lesion.contributing_factors, "spectral.hb_absorption").points == 1.0
    assert factor(lesion.contributing_factors, "spectral.hb_absorption").detail.endswith("s2")
    assert factor(lesion.contributing_factors, "spectral.ratio_555_630").points == 1.0


def test_spectral_without_reference_scores_zero_with_note():
    out = triage([site(TONGUE, [region()])], [reading("s1", TONGUE)])
    f = factor(out.lesions[0].contributing_factors, "spectral.no_reference")
    assert f.points == 0.0 and "not scored" in f.label
    assert "spectral_no_reference" in limitation_codes(out)


def test_spectral_qc_failed_scores_zero():
    out = triage([site(TONGUE, [region()])], [reading("s1", TONGUE, hb_delta=0.1, qc_passed=False)])
    assert factor(out.lesions[0].contributing_factors, "spectral.qc_failed").points == 0.0
    assert "spectral_qc_failed" in limitation_codes(out)


def test_spectral_lesion_ref_targets_matching_region():
    sites = [site(TONGUE, [region("r1", mean_rel=0.5), region("r2", mean_rel=0.7)])]
    spectral = [reading("s1", TONGUE, hb_delta=0.07, lesion_ref="r2")]
    out = triage(sites, spectral)
    assert out.lesions[0].spectral_reading_ids == []
    assert out.lesions[1].spectral_reading_ids == ["s1"]


def test_spectral_unmatched_ref_falls_back_to_primary():
    sites = [site(TONGUE, [region("r1", mean_rel=0.7), region("r2", mean_rel=0.5)])]
    out = triage(sites, [reading("s1", TONGUE, hb_delta=0.07, lesion_ref="r9")])
    assert out.lesions[1].spectral_reading_ids == ["s1"]


@pytest.mark.parametrize("label", ["suspicious", "indeterminate", "not_suspicious", None])
def test_every_lesion_reading_becomes_a_lesion_whatever_the_label(label):
    """The reading's role is the health worker's designation; the placeholder label never
    decides whether a finding exists."""
    out = triage(
        [site(TONGUE)], [reading("s1", TONGUE, hb_delta=0.07, ratio_delta=-0.12, label=label)]
    )
    (lesion,) = out.lesions
    assert lesion.origins == ["spectral"]
    assert lesion.score == pytest.approx(2.0 * 1.15, abs=1e-4)
    assert "deeper blood-absorption dip" in lesion.summary


def test_spectral_only_lesion_with_unscored_reading_is_described():
    out = triage([site(TONGUE)], [reading("s1", TONGUE, label="suspicious")])
    (lesion,) = out.lesions
    assert lesion.score == 0.0
    assert "probe reading marked suspicious" in lesion.summary


def test_reference_readings_are_not_lesions():
    out = triage([site(TONGUE)], [reading("ref", TONGUE, role="reference", label="suspicious")])
    assert out.lesions == []


# --------------------------------------------------------------------------------------------
# Visual flag
# --------------------------------------------------------------------------------------------


def test_visual_merges_into_darkest_region():
    # Upper lip mucosa is not a high-risk site (where any FVL is at least MODERATE), so r1's
    # level shows that the visual observation and its floor did not reach it.
    sites = [
        site(
            S.UPPER_LIP_MUCOSA,
            [region("r1", mean_rel=0.70), region("r2", mean_rel=0.45)],
            appearance=[V.WHITE_PATCH, V.RED_PATCH],
        )
    ]
    out = triage(sites)
    r1, r2 = out.lesions
    assert r1.origins == ["af_fvl"]
    assert r2.origins == ["af_fvl", "visual"]
    f = factor(r2.contributing_factors, "visual.seen.red_patch")
    assert f.points == 4.0 and f.detail == "Also recorded: a white patch"
    assert "visual.red_patch" in codes(r2.contributing_factors)
    assert r2.risk_level == RiskLevel.HIGH
    assert r1.risk_level == RiskLevel.LOW


def test_visual_without_fvl_notes_af_negative():
    lesion = only_lesion([site(S.TONGUE_DORSUM, appearance=[V.WHITE_PATCH])])
    assert lesion.origins == ["visual"]
    assert "af.no_fvl_at_site" in codes(lesion.contributing_factors)
    assert lesion.score == 2.0  # tongue dorsum is not a high-risk site
    assert lesion.risk_level == RiskLevel.MODERATE  # white-patch floor at lesion level
    assert lesion.location is None


def test_appearance_without_checkbox_still_counts():
    lesion = only_lesion([site(S.TONGUE_DORSUM, appearance=[V.ULCER], lesion_seen=False)])
    assert factor(lesion.contributing_factors, "visual.seen.ulcer").points == 3.0
