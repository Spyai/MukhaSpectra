"""Integration regression tests: places where two independently fixed modules had to agree.

* A site whose analysis crashed (``not_analysed``) is a software problem, not a capture-quality
  one: the triage never sends the worker to retake it, never calls it a QC failure, and never
  allows LOW.
* The spectral 555/630 delta is a relative change (quotient semantics), so it is shown as %.
* The AF reference notes never call the comparison tissue "healthy".
* Border irregularity is now 1 - hull perimeter / perimeter: smooth elongated outlines score 0
  and get no "irregular border" points; ragged ones still do.

All inputs are hand-built test fixtures (not patient data).
"""

from __future__ import annotations

import pytest
from fusion_helpers import (
    S,
    all_texts,
    assert_invariants,
    default_rules,
    factor,
    limitation_codes,
    patient,
    reading,
    region,
    risk,
    site,
)

from mukha_pipeline.af import resolve_reference
from mukha_pipeline.fusion import run_triage
from mukha_pipeline.positioning import find_positioning_violations
from mukha_pipeline.schema.models import RiskLevel, SiteResult, VisualAppearance


def _software_error(s: SiteResult) -> SiteResult:
    return s.model_copy(update={"analysis_status": "not_analysed", "af": None, "wl": None})


def _triage(sites, spectral=()):
    out = run_triage(list(sites), list(spectral), patient(40), risk(), default_rules())
    assert_invariants(out)
    return out


# --------------------------------------------------------------------------------------------
# Software errors (honesty.2 across orchestrator and fusion)
# --------------------------------------------------------------------------------------------


def test_software_error_site_is_inconclusive_and_not_called_a_qc_failure():
    out = _triage(
        [
            site(S.BUCCAL_MUCOSA_LEFT),
            site(S.BUCCAL_MUCOSA_RIGHT),
            _software_error(site(S.LATERAL_TONGUE_LEFT)),
        ]
    )
    assert out.triage.overall_risk_level == RiskLevel.INCONCLUSIVE
    codes = limitation_codes(out)
    assert "sites_failed_qc" not in codes
    assert "analysis_error" in codes
    steps = out.triage.recommended_action.steps
    assert not [s for s in steps if "retake" in s.lower() and "left lateral tongue" in s.lower()]
    assert any("software error" in s and "left lateral tongue" in s.lower() for s in steps)
    text = out.triage.explanation
    assert "software error" in text
    assert "retake the images now" not in text
    assert find_positioning_violations(" ".join(all_texts(out))) == []


def test_qc_failure_and_software_error_are_told_apart():
    out = _triage(
        [
            site(S.BUCCAL_MUCOSA_LEFT),
            site(S.BUCCAL_MUCOSA_RIGHT, analysed=False),
            _software_error(site(S.LATERAL_TONGUE_LEFT)),
        ]
    )
    assert out.triage.overall_risk_level == RiskLevel.INCONCLUSIVE
    lims = {lim.code: lim.message for lim in out.limitations}
    assert "Right buccal mucosa" in lims["sites_failed_qc"]
    assert "lateral tongue" not in lims["sites_failed_qc"].lower()
    assert "Left lateral tongue" in lims["analysis_error"]
    steps = out.triage.recommended_action.steps
    assert "Sites to retake: Right buccal mucosa." in steps
    assert "retake the images now" in out.triage.explanation


def test_software_error_under_a_high_result_does_not_ask_for_a_retake():
    strong = region(mean_rel=0.30, fraction=0.20, border=0.5, contrast=2.0)
    out = _triage(
        [
            site(S.BUCCAL_MUCOSA_LEFT, [strong]),
            site(S.BUCCAL_MUCOSA_RIGHT),
            _software_error(site(S.LATERAL_TONGUE_LEFT)),
        ]
    )
    assert out.triage.overall_risk_level == RiskLevel.HIGH
    text = out.triage.explanation
    assert "should be retaken" not in text
    assert "software error" in text
    steps = out.triage.recommended_action.steps
    assert not [s for s in steps if "retake" in s.lower() and "lateral tongue" in s.lower()]


# --------------------------------------------------------------------------------------------
# Spectral delta semantics (science.5 across spectral and fusion)
# --------------------------------------------------------------------------------------------


def test_spectral_ratio_factor_is_shown_as_a_relative_change():
    out = _triage(
        [site(S.BUCCAL_MUCOSA_LEFT), site(S.BUCCAL_MUCOSA_RIGHT)],
        [reading("s1", S.BUCCAL_MUCOSA_LEFT, hb_delta=0.03, ratio_delta=-0.064)],
    )
    (lesion,) = out.lesions
    f = factor(lesion.contributing_factors, "spectral.ratio_555_630")
    assert "(-6.4%)" in f.label
    hb = factor(lesion.contributing_factors, "spectral.hb_absorption")
    assert "(+0.030)" in hb.label


# --------------------------------------------------------------------------------------------
# Reference wording (honesty.11 in the AF notes)
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("own", "available"),
    [
        (0.40, {S.BUCCAL_MUCOSA_RIGHT: 0.30}),  # paired side dimmer: own level used
        (0.40, {S.BUCCAL_MUCOSA_RIGHT: 0.50}),  # paired side brighter
        (0.40, {}),  # in-image
        (None, {}),  # none
    ],
)
def test_reference_notes_never_call_the_comparison_tissue_healthy(own, available):
    ref = resolve_reference(S.BUCCAL_MUCOSA_LEFT, own, available)
    assert ref.note
    assert "healthy" not in ref.note.lower()
    assert find_positioning_violations(ref.note) == []


# --------------------------------------------------------------------------------------------
# Border irregularity scale (science.8 across af and the triage rules)
# --------------------------------------------------------------------------------------------


def test_border_rule_matches_the_new_convexity_deficit_scale():
    rules = default_rules().data["lesion"]["af"]["border_irregularity"]
    threshold = float(rules["min"])
    # Smooth ellipses score 0 at any elongation; a jagged blob scores ~0.31, a strong star 0.46.
    assert 0.0 < threshold <= 0.2

    def border_points(border: float) -> float:
        out = _triage(
            [
                site(
                    S.BUCCAL_MUCOSA_LEFT,
                    [region(mean_rel=0.6, fraction=0.05, border=border)],
                    appearance=[VisualAppearance.WHITE_PATCH],
                ),
                site(S.BUCCAL_MUCOSA_RIGHT),
            ]
        )
        (lesion,) = out.lesions
        f = [x for x in lesion.contributing_factors if x.code == "af.border_irregular"]
        return sum(x.points for x in f)

    assert border_points(0.0) == 0.0  # smooth (any elongation)
    assert border_points(0.31) > 0.0  # jagged blob
