"""Regression tests for spectral review findings science.5, honesty.3 and clinical.5.

science.5: the lesion-vs-reference comparison must not depend on the probe's LED spectrum, filter
responses or sensor gain, and calibrated and uncalibrated readings of the same tissue must give
the same comparison.

honesty.3 / clinical.5 (spectral labelling): the rule-based classifier is a placeholder heuristic,
says so in its output, and its "not_suspicious" label is worded as "no spectral difference
detected (this does not rule out disease)", never as normal.

All readings are SYNTHETIC (mukha_pipeline.synthetic.generate_spectral_reading).
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime

import numpy as np
import pytest
from wlspec_helpers import qc_pass, scale_device_gain

from mukha_pipeline.positioning import find_positioning_violations
from mukha_pipeline.schema.models import (
    SpectralClassifierOutput,
    SpectralFeatures,
    SpectralReadingResult,
)
from mukha_pipeline.spectral import (
    RuleBasedSpectralClassifier,
    analyze_spectral,
    compute_features,
    reference_delta,
)
from mukha_pipeline.synthetic import (
    generate_spectral_reading,
    generate_white_reference,
    make_spectral_calibration,
)
from mukha_pipeline.types import SpectralCalibration, SpectralFrame, SpectralInput

NOW = datetime(2026, 10, 14, 9, 30, tzinfo=UTC)
REF_HB = 0.25  # synthetic normal mucosa
LESION_SITE, REF_SITE = "buccal_mucosa_left", "buccal_mucosa_right"

# A second probe: same tissue, a different LED spectrum x filter transmission x sensor gain in
# every channel (warmer LED: less blue, more red).
DEVICE_B_GAIN: dict[str, float] = {
    "F1_415": 0.35,
    "F2_445": 0.55,
    "F3_480": 0.7,
    "F4_515": 0.9,
    "F5_555": 1.2,
    "F6_590": 1.45,
    "F7_630": 1.7,
    "F8_680": 1.5,
    "CLEAR": 1.1,
    "NIR": 1.3,
}

FIELDS = tuple(SpectralFeatures.model_fields)
NORMAL_WORD = re.compile(r"\bnormal\b", re.I)


def _pair(
    seed: int, lesion_hb: float, *, noise: float = 0.01
) -> tuple[SpectralInput, SpectralInput, SpectralCalibration]:
    rng = np.random.default_rng(seed)
    cal = make_spectral_calibration(*generate_white_reference(rng), created_at=NOW)
    lesion = generate_spectral_reading(
        rng,
        hb=lesion_hb,
        reading_id="L",
        site=LESION_SITE,
        role="lesion",
        reference_reading_id="R",
        noise=noise,
    )
    ref = generate_spectral_reading(
        rng, hb=REF_HB, reading_id="R", site=REF_SITE, role="reference", noise=noise
    )
    return lesion, ref, cal


def _lesion_result(
    readings: Sequence[SpectralInput],
    cal: SpectralCalibration | None,
    analysis_cfg: dict,
    qc_cfg: dict,
) -> SpectralReadingResult:
    cals = {cal.device_id: cal} if cal is not None else {}
    results = analyze_spectral(readings, cals, qc_cfg, analysis_cfg, NOW, qc_fn=qc_pass)
    lesion = next(r for r in results if r.role == "lesion")
    assert lesion.delta_vs_reference is not None
    return lesion


def _delta(
    lesion: SpectralInput,
    ref: SpectralInput,
    cal: SpectralCalibration | None,
    analysis_cfg: dict,
    qc_cfg: dict,
) -> SpectralFeatures:
    result = _lesion_result([lesion, ref], cal, analysis_cfg, qc_cfg)
    assert result.signal_type == ("reflectance" if cal is not None else "relative_signal")
    assert result.delta_vs_reference is not None  # narrows the type for the checker
    return result.delta_vs_reference


def _tier_points(tiers: Sequence[Mapping[str, float]], value: float) -> float:
    """Best tier reached by ``value`` in a raw triage_rules tier list (min: >=, max: <=)."""
    best = 0.0
    for t in tiers:
        hit = value >= t["min"] if "min" in t else value <= t["max"]
        if hit:
            best = max(best, float(t["points"]))
    return best


@pytest.fixture(scope="module")
def spectral_tiers(pipeline_config) -> Mapping[str, Sequence[Mapping[str, float]]]:
    return pipeline_config.triage.data["lesion"]["spectral"]


# --------------------------------------------------------------------------------------------
# science.5: device-invariant lesion-vs-reference comparison
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("lesion_hb", [0.25, 0.36, 0.40, 0.6])
def test_relative_signal_delta_does_not_depend_on_the_probe(lesion_hb, analysis_cfg, qc_cfg):
    """Same tissue, two probes with different per-channel gain: same uncalibrated comparison."""
    lesion, ref, _ = _pair(11, lesion_hb)
    a = _delta(lesion, ref, None, analysis_cfg, qc_cfg)
    b = _delta(
        scale_device_gain(lesion, DEVICE_B_GAIN),
        scale_device_gain(ref, DEVICE_B_GAIN),
        None,
        analysis_cfg,
        qc_cfg,
    )
    for name in FIELDS:
        assert getattr(b, name) == pytest.approx(getattr(a, name), abs=2e-3), name


@pytest.mark.parametrize("lesion_hb", [0.25, 0.30, 0.36, 0.40, 0.6])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_calibrated_and_uncalibrated_deltas_agree(lesion_hb, seed, analysis_cfg, qc_cfg):
    lesion, ref, cal = _pair(seed, lesion_hb)
    calibrated = _delta(lesion, ref, cal, analysis_cfg, qc_cfg)
    relative = _delta(lesion, ref, None, analysis_cfg, qc_cfg)
    for name in FIELDS:
        assert getattr(relative, name) == pytest.approx(getattr(calibrated, name), abs=5e-3), name


def test_calibrated_and_uncalibrated_agree_across_probes(analysis_cfg, qc_cfg):
    """Probe B uncalibrated gives the comparison probe A gives with its white calibration."""
    lesion, ref, cal = _pair(5, 0.40)
    calibrated_a = _delta(lesion, ref, cal, analysis_cfg, qc_cfg)
    relative_b = _delta(
        scale_device_gain(lesion, DEVICE_B_GAIN),
        scale_device_gain(ref, DEVICE_B_GAIN),
        None,
        analysis_cfg,
        qc_cfg,
    )
    for name in FIELDS:
        assert getattr(relative_b, name) == pytest.approx(getattr(calibrated_a, name), abs=5e-3)


def test_delta_is_defined_on_the_lesion_over_reference_quotient():
    """delta = features(lesion / reference) - features(all-ones spectrum)."""
    lesion = {
        "F1_415": 0.08,
        "F2_445": 0.15,
        "F3_480": 0.22,
        "F4_515": 0.30,
        "F5_555": 0.20,
        "F6_590": 0.26,
        "F7_630": 0.45,
        "F8_680": 0.50,
    }
    ref = {
        "F1_415": 0.12,
        "F2_445": 0.18,
        "F3_480": 0.25,
        "F4_515": 0.32,
        "F5_555": 0.26,
        "F6_590": 0.31,
        "F7_630": 0.46,
        "F8_680": 0.52,
    }
    d = reference_delta(lesion, ref)
    assert d is not None
    fl, fr = compute_features(lesion), compute_features(ref)
    # Ratios: relative change of the lesion's ratio against the reference's ratio.
    assert d.ratio_555_630 == pytest.approx(fl.ratio_555_630 / fr.ratio_555_630 - 1.0)
    assert d.ratio_590_680 == pytest.approx(fl.ratio_590_680 / fr.ratio_590_680 - 1.0)
    assert d.ratio_515_630 == pytest.approx(fl.ratio_515_630 / fr.ratio_515_630 - 1.0)
    assert d.ratio_415_555 == pytest.approx(fl.ratio_415_555 / fr.ratio_415_555 - 1.0)
    # Log-slopes of the quotient are differences of log-slopes.
    assert d.slope_480_555 == pytest.approx(fl.slope_480_555 - fr.slope_480_555)
    assert d.slope_590_680 == pytest.approx(fl.slope_590_680 - fr.slope_590_680)
    q = {c: lesion[c] / ref[c] for c in lesion}
    assert d.hb_absorption_proxy == pytest.approx(compute_features(q).hb_absorption_proxy)
    assert d.hb_absorption_proxy > 0  # deeper 555/590 dip in the lesion

    # A per-channel gain (another probe's LED x filters) and an overall scale both cancel.
    gain = {c: 0.5 + 0.2 * i for i, c in enumerate(lesion)}
    scaled = reference_delta(
        {c: 3.0 * gain[c] * v for c, v in lesion.items()},
        {c: 0.7 * gain[c] * v for c, v in ref.items()},
    )
    assert scaled is not None
    for name in FIELDS:
        assert getattr(scaled, name) == pytest.approx(getattr(d, name), abs=1e-12)

    same = reference_delta(ref, ref)
    assert same is not None and all(getattr(same, n) == pytest.approx(0.0) for n in FIELDS)


def test_delta_needs_usable_reference_channels():
    ref = {c: 0.3 for c in ("F4_515", "F5_555", "F6_590", "F7_630")}
    assert reference_delta(dict(ref), dict(ref)) is not None
    assert reference_delta(dict(ref), {**ref, "F5_555": 0.0}) is None  # no reference signal
    missing_630 = {k: v for k, v in ref.items() if k != "F7_630"}
    assert reference_delta(dict(ref), missing_630) is None
    assert reference_delta({}, ref) is None
    assert reference_delta(dict(ref), {**ref, "F6_590": float("nan")}) is None
    # Optional channels missing on one side give 0 for their features, never a spurious delta.
    partial = reference_delta({**ref, "F1_415": 0.1}, ref)
    assert partial is not None and partial.ratio_415_555 == 0.0


def test_unusable_reference_is_reported_not_compared(analysis_cfg, qc_cfg):
    """A reference with no 630 nm signal cannot be compared: indeterminate, never reassuring."""
    lesion, ref, _ = _pair(3, 0.6)

    def dead_630(frames: Sequence[SpectralFrame]) -> list[SpectralFrame]:
        return [
            SpectralFrame({**f.channels, "F7_630": 0.0}, f.gain, f.integration_ms) for f in frames
        ]

    dead = SpectralInput(
        reading_id=ref.reading_id,
        site=ref.site,
        role=ref.role,
        lit=dead_630(ref.lit),
        dark=dead_630(ref.dark),
        meta=ref.meta,
    )
    results = analyze_spectral([lesion, dead], {}, qc_cfg, analysis_cfg, NOW, qc_fn=qc_pass)
    les = results[0]
    assert les.delta_vs_reference is None
    clf = les.classifier
    assert clf is not None and clf.label == "indeterminate" and clf.suspicion_score is None
    assert "no usable signal" in clf.reasons[0]
    for reason in clf.reasons:
        assert not find_positioning_violations(reason), reason


@pytest.mark.parametrize("calibrated", [True, False], ids=["calibrated", "relative_signal"])
@pytest.mark.parametrize("seed", range(6))
def test_equal_blood_volume_gives_no_points(calibrated, seed, analysis_cfg, qc_cfg, spectral_tiers):
    lesion, ref, cal = _pair(seed, REF_HB)
    result = _lesion_result([lesion, ref], cal if calibrated else None, analysis_cfg, qc_cfg)
    d = result.delta_vs_reference
    assert d is not None
    assert _tier_points(spectral_tiers["hb_absorption_proxy_delta"], d.hb_absorption_proxy) == 0
    assert _tier_points(spectral_tiers["ratio_555_630_delta"], d.ratio_555_630) == 0
    assert result.classifier is not None and result.classifier.label == "not_suspicious"


@pytest.mark.parametrize("calibrated", [True, False], ids=["calibrated", "relative_signal"])
@pytest.mark.parametrize("seed", range(6))
def test_moderate_blood_volume_reaches_first_tier(
    calibrated, seed, analysis_cfg, qc_cfg, spectral_tiers
):
    """hb 0.36 vs 0.25 (the moderate demo) is at least 'indeterminate' / first tier."""
    lesion, ref, cal = _pair(seed, 0.36)
    result = _lesion_result([lesion, ref], cal if calibrated else None, analysis_cfg, qc_cfg)
    d = result.delta_vs_reference
    assert d is not None
    hb_pts = _tier_points(spectral_tiers["hb_absorption_proxy_delta"], d.hb_absorption_proxy)
    ratio_pts = _tier_points(spectral_tiers["ratio_555_630_delta"], d.ratio_555_630)
    assert hb_pts >= 0.5 and ratio_pts >= 0.5
    assert result.classifier is not None
    assert result.classifier.label in ("indeterminate", "suspicious")


@pytest.mark.parametrize("calibrated", [True, False], ids=["calibrated", "relative_signal"])
@pytest.mark.parametrize("seed", range(6))
def test_high_blood_volume_reaches_top_tier(calibrated, seed, analysis_cfg, qc_cfg, spectral_tiers):
    lesion, ref, cal = _pair(seed, 0.6)
    result = _lesion_result([lesion, ref], cal if calibrated else None, analysis_cfg, qc_cfg)
    d = result.delta_vs_reference
    assert d is not None
    top_hb = max(float(t["points"]) for t in spectral_tiers["hb_absorption_proxy_delta"])
    top_ratio = max(float(t["points"]) for t in spectral_tiers["ratio_555_630_delta"])
    hb_tiers = spectral_tiers["hb_absorption_proxy_delta"]
    assert _tier_points(hb_tiers, d.hb_absorption_proxy) == top_hb
    assert _tier_points(spectral_tiers["ratio_555_630_delta"], d.ratio_555_630) == top_ratio
    assert result.classifier is not None and result.classifier.label == "suspicious"


def test_classifier_thresholds_match_triage_tiers(analysis_cfg, spectral_tiers):
    """The classifier label and the triage points use the same cut-offs, so they cannot disagree."""
    clf = analysis_cfg["spectral"]["classifier"]
    hb = sorted(float(t["min"]) for t in spectral_tiers["hb_absorption_proxy_delta"])
    ratio = sorted((float(t["max"]) for t in spectral_tiers["ratio_555_630_delta"]), reverse=True)
    assert hb == [clf["hb_proxy_delta_indeterminate"], clf["hb_proxy_delta_suspicious"]]
    assert ratio == [
        clf["ratio_555_630_delta_indeterminate"],
        clf["ratio_555_630_delta_suspicious"],
    ]


# --------------------------------------------------------------------------------------------
# honesty.3 / clinical.5: placeholder status and wording
# --------------------------------------------------------------------------------------------

ZERO = SpectralFeatures(**{name: 0.0 for name in FIELDS})


def _out(clf: RuleBasedSpectralClassifier, hb: float, ratio: float) -> SpectralClassifierOutput:
    return clf.classify(
        ZERO, ZERO.model_copy(update={"hb_absorption_proxy": hb, "ratio_555_630": ratio})
    )


@pytest.fixture
def clf(analysis_cfg) -> RuleBasedSpectralClassifier:
    return RuleBasedSpectralClassifier(analysis_cfg["spectral"])


def test_model_info_says_placeholder_not_trained_not_validated(clf):
    assert clf.info.notes == (
        "Heuristic rules on lesion-vs-reference differences; placeholder, not a trained model, "
        "not validated."
    )
    assert clf.info.validated is False
    assert clf.info.version == "2.0.0"


@pytest.mark.parametrize(
    ("hb", "ratio", "label"),
    [(0.2, -0.3, "suspicious"), (0.03, 0.0, "indeterminate"), (0.0, 0.0, "not_suspicious")],
)
def test_classified_output_is_marked_placeholder(clf, hb, ratio, label):
    out = _out(clf, hb, ratio)
    assert out.label == label
    assert out.status == "placeholder"  # never presented as a model output
    assert out.reasons[0].startswith("Placeholder heuristic")
    assert "not a trained model" in out.reasons[0] and "not validated" in out.reasons[0]
    for reason in out.reasons:
        assert not NORMAL_WORD.search(reason), reason
        assert not find_positioning_violations(reason), reason


def test_not_suspicious_is_worded_as_no_difference_detected(clf):
    out = _out(clf, 0.0, 0.0)
    assert out.label == "not_suspicious"
    text = " ".join(out.reasons)
    assert "No spectral difference detected" in text
    assert "(this does not rule out disease)" in text
    assert not NORMAL_WORD.search(text)


def test_skipped_and_no_reference_outputs_carry_no_reassurance(clf):
    for out in (clf.skipped_qc_failed(), clf.classify(ZERO, None)):
        assert out.label != "not_suspicious"
        for reason in out.reasons:
            assert not NORMAL_WORD.search(reason)
            assert not find_positioning_violations(reason)
