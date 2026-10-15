"""Tests for mukha_pipeline.spectral (summary, features, rules classifier, analyze_spectral).

All spectra are SYNTHETIC: a scattering background attenuated by oxy-haemoglobin-like absorption
(Soret 415 nm, Q bands 542/577 nm) scaled by blood volume, under a phosphor white LED.
"""

from __future__ import annotations

import math
import sys
import types
from datetime import timedelta

import numpy as np
import pytest
from wlspec_helpers import (
    make_calibration,
    make_reading,
    qc_fail,
    qc_pass,
    true_band_reflectance,
)

from mukha_pipeline.positioning import find_positioning_violations
from mukha_pipeline.schema.models import (
    AS7341_CHANNELS,
    CalibrationStatus,
    ModelComponent,
    ModelStatus,
    SpectralFeatures,
    SpectralReadingResult,
)
from mukha_pipeline.sites import OralSite
from mukha_pipeline.spectral import (
    RuleBasedSpectralClassifier,
    SpectralClassifier,
    analyze_spectral,
    calibration_info,
    compute_features,
    delta_features,
    reference_delta,
    summarise_reading,
)
from mukha_pipeline.spectral import analyze as analyze_mod
from mukha_pipeline.spectral.features import hb_absorption_proxy, log_slope
from mukha_pipeline.types import SpectralFrame

L, R = OralSite.BUCCAL_MUCOSA_LEFT, OralSite.BUCCAL_MUCOSA_RIGHT
ZERO = SpectralFeatures(
    ratio_555_630=0.0,
    ratio_590_680=0.0,
    ratio_515_630=0.0,
    ratio_415_555=0.0,
    hb_absorption_proxy=0.0,
    slope_480_555=0.0,
    slope_590_680=0.0,
)


def _delta(hb: float = 0.0, ratio: float = 0.0) -> SpectralFeatures:
    return ZERO.model_copy(update={"hb_absorption_proxy": hb, "ratio_555_630": ratio})


def _texts(result: SpectralReadingResult) -> list[str]:
    out = [result.calibration.note or ""]
    if result.classifier:
        out += result.classifier.reasons + [result.classifier.model.notes or ""]
    out += [i.message for i in result.qc.issues]
    return out


# --------------------------------------------------------------------------------------------
# summarise_reading
# --------------------------------------------------------------------------------------------


def test_calibrated_reflectance_matches_ground_truth(rng, fixed_now):
    reading = make_reading(rng, "s1", L, "lesion", 1.0, noise=0.0)
    values, std, signal, info = summarise_reading(reading, make_calibration(fixed_now), fixed_now)
    assert signal == "reflectance"
    assert info.status == CalibrationStatus.CALIBRATED and info.calibration_id == "spcal-test"
    truth = true_band_reflectance(1.0)
    for c in AS7341_CHANNELS:
        assert values[c] == pytest.approx(truth[c], rel=1e-6)
        assert std[c] == pytest.approx(0.0, abs=1e-12)


def test_std_reflects_repeat_noise(rng, fixed_now):
    reading = make_reading(rng, "s1", L, "lesion", 1.0, noise=0.02, n=8)
    values, std, _, _ = summarise_reading(reading, make_calibration(fixed_now), fixed_now)
    assert all(std[c] > 0 for c in AS7341_CHANNELS)
    assert std["F7_630"] / values["F7_630"] == pytest.approx(0.02, rel=0.6)


def test_gain_and_integration_are_scaled(rng, fixed_now):
    """A white standard read at 16x / 25 ms still calibrates a reading taken at 64x / 50 ms."""
    reading = make_reading(rng, "s1", L, "lesion", 1.0, noise=0.0, gain=64, integration_ms=50)
    cal = make_calibration(fixed_now, gain=16, integration_ms=25)
    values, _, signal, info = summarise_reading(reading, cal, fixed_now)
    assert signal == "reflectance"
    truth = true_band_reflectance(1.0)
    assert values["F5_555"] == pytest.approx(truth["F5_555"], rel=1e-6)
    assert "scaled by 8" in (info.note or "")


def test_expired_calibration_still_gives_reflectance(rng, fixed_now):
    reading = make_reading(rng, "s1", L, "lesion", 1.0)
    cal = make_calibration(fixed_now - timedelta(hours=30))
    _, _, signal, info = summarise_reading(reading, cal, fixed_now)
    assert signal == "reflectance" and info.status == CalibrationStatus.EXPIRED
    assert "expired" in (info.note or "")


def test_naive_datetimes_are_utc(rng, fixed_now):
    cal = make_calibration(fixed_now.replace(tzinfo=None))
    info = calibration_info(cal, (fixed_now + timedelta(hours=23)).replace(tzinfo=None))
    assert info.status == CalibrationStatus.CALIBRATED
    assert info.expires_at is not None and info.expires_at.tzinfo is not None
    assert calibration_info(cal, fixed_now + timedelta(hours=25)).status == (
        CalibrationStatus.EXPIRED
    )


def test_uncalibrated_relative_signal(rng, fixed_now):
    reading = make_reading(rng, "s1", L, "lesion", 1.0, noise=0.0)
    values, _, signal, info = summarise_reading(reading, None, fixed_now)
    assert signal == "relative_signal" and info.status == CalibrationStatus.UNCALIBRATED
    assert sum(values[c] for c in AS7341_CHANNELS[:8]) == pytest.approx(1.0)
    assert "CLEAR" in values and "NIR" in values


def test_relative_signal_ratios_match_reflectance_ratios_after_reference_comparison(rng, fixed_now):
    """The lesion / reference quotient gives the same delta calibrated and uncalibrated."""
    cal = make_calibration(fixed_now)
    lo = make_reading(rng, "a", R, "reference", 1.0, noise=0.0)
    hi = make_reading(rng, "b", L, "lesion", 2.0, noise=0.0)
    deltas = []
    for c in (cal, None):
        d = reference_delta(
            summarise_reading(hi, c, fixed_now)[0], summarise_reading(lo, c, fixed_now)[0]
        )
        assert d is not None
        assert d.hb_absorption_proxy > 0.05
        assert d.ratio_555_630 < -0.1
        assert d.ratio_415_555 < 0  # Soret band absorption
        deltas.append(d)
    for name in SpectralFeatures.model_fields:
        assert getattr(deltas[1], name) == pytest.approx(getattr(deltas[0], name), abs=1e-9)


def test_unusable_white_falls_back_to_relative(rng, fixed_now):
    reading = make_reading(rng, "s1", L, "lesion", 1.0)
    cal = make_calibration(fixed_now)
    cal.white["F5_555"] = cal.dark["F5_555"]  # white not above dark
    values, _, signal, info = summarise_reading(reading, cal, fixed_now)
    assert signal == "relative_signal"
    assert info.status == CalibrationStatus.UNCALIBRATED and info.calibration_id == "spcal-test"
    assert "F5_555" in (info.note or "")


def test_optional_channel_without_white_is_dropped(rng, fixed_now):
    reading = make_reading(rng, "s1", L, "lesion", 1.0)
    cal = make_calibration(fixed_now)
    del cal.white["NIR"]
    values, _, signal, _ = summarise_reading(reading, cal, fixed_now)
    assert signal == "reflectance" and "NIR" not in values and "CLEAR" in values


def test_degenerate_readings_never_produce_nan(fixed_now):
    from wlspec_helpers import make_meta

    from mukha_pipeline.types import SpectralInput

    def reading(lit, dark):
        return SpectralInput("x", L, "lesion", lit, dark, make_meta(L))

    empty = reading([], [])
    assert summarise_reading(empty, None, fixed_now)[:3] == ({}, {}, "relative_signal")

    zeros = {c: 100.0 for c in AS7341_CHANNELS}
    flat = reading([SpectralFrame(dict(zeros))], [SpectralFrame(dict(zeros))])
    values, std, _, _ = summarise_reading(flat, None, fixed_now)
    assert all(v == 0.0 for v in values.values()) and all(s == 0.0 for s in std.values())

    weird = dict(zeros, F2_445=float("nan"))
    one = reading([SpectralFrame(weird), SpectralFrame(dict(zeros, F1_415=300.0))], [])
    values, std, _, _ = summarise_reading(one, None, fixed_now)
    assert all(math.isfinite(v) for v in values.values())
    assert all(math.isfinite(v) for v in std.values())
    assert summarise_reading(one, make_calibration(fixed_now), fixed_now)[2] == "reflectance"

    clear_only = reading([SpectralFrame({"CLEAR": 900.0})], [SpectralFrame({"CLEAR": 100.0})])
    values, _, signal, _ = summarise_reading(clear_only, None, fixed_now)
    assert signal == "relative_signal" and values == {"CLEAR": 0.0}


# --------------------------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------------------------


def test_more_blood_deepens_the_haemoglobin_dip(fixed_now, rng):
    cal = make_calibration(fixed_now)
    feats = []
    for blood in (0.5, 1.0, 1.5, 2.0, 3.0):
        values = summarise_reading(make_reading(rng, "s", L, "lesion", blood), cal, fixed_now)[0]
        feats.append(compute_features(values))
    hb = [f.hb_absorption_proxy for f in feats]
    r555 = [f.ratio_555_630 for f in feats]
    assert all(b > a for a, b in zip(hb, hb[1:], strict=False))
    assert all(b < a for a, b in zip(r555, r555[1:], strict=False))
    assert all(
        b < a
        for a, b in zip(
            [f.slope_480_555 for f in feats], [f.slope_480_555 for f in feats[1:]], strict=False
        )
    )


def test_feature_formulas_on_known_values():
    values = {
        "F1_415": 0.1,
        "F3_480": 0.3,
        "F4_515": 0.4,
        "F5_555": 0.3,
        "F6_590": 0.35,
        "F7_630": 0.5,
        "F8_680": 0.6,
    }
    f = compute_features(values)
    assert f.ratio_555_630 == pytest.approx(0.6)
    assert f.ratio_590_680 == pytest.approx(0.35 / 0.6)
    assert f.ratio_515_630 == pytest.approx(0.8)
    assert f.ratio_415_555 == pytest.approx(1 / 3)
    b555 = 0.4 + (0.1 / 115) * 40
    b590 = 0.4 + (0.1 / 115) * 75
    assert f.hb_absorption_proxy == pytest.approx(1 - 0.65 / (b555 + b590))
    x = np.array([4.8, 5.15, 5.55])
    expected = np.polyfit(x, np.log([0.3, 0.4, 0.3]), 1)[0]
    assert f.slope_480_555 == pytest.approx(expected)
    flat = {c: 0.5 for c in AS7341_CHANNELS}
    assert compute_features(flat).hb_absorption_proxy == pytest.approx(0.0)
    assert compute_features(flat).slope_590_680 == pytest.approx(0.0)


def test_features_are_nan_safe():
    assert compute_features({}) == ZERO
    bad = {"F5_555": float("nan"), "F7_630": 0.0, "F4_515": "x", "F6_590": None}
    assert compute_features(bad) == ZERO  # type: ignore[arg-type]
    assert hb_absorption_proxy({"F4_515": 0, "F5_555": 0, "F6_590": 0, "F7_630": 0}) == 0.0
    assert (
        log_slope(
            {"F6_590": 0.5, "F7_630": -1.0, "F8_680": float("inf")}, ("F6_590", "F7_630", "F8_680")
        )
        == 0.0
    )
    two_points = log_slope({"F6_590": 0.5, "F8_680": 1.0}, ("F6_590", "F7_630", "F8_680"))
    assert two_points == pytest.approx(np.log(2.0) / 0.9)  # missing 630 nm is skipped


def test_delta_features():
    a = _delta(hb=0.3, ratio=0.5)
    b = _delta(hb=0.1, ratio=0.7)
    d = delta_features(a, b)
    assert d.hb_absorption_proxy == pytest.approx(0.2)
    assert d.ratio_555_630 == pytest.approx(-0.2)
    assert d.slope_480_555 == 0.0


# --------------------------------------------------------------------------------------------
# Rules classifier
# --------------------------------------------------------------------------------------------


@pytest.fixture
def clf(analysis_cfg) -> RuleBasedSpectralClassifier:
    return RuleBasedSpectralClassifier(analysis_cfg["spectral"])


def test_classifier_model_info(clf):
    assert isinstance(clf, SpectralClassifier)
    info = clf.info
    assert info.component == ModelComponent.SPECTRAL_CLASSIFIER
    assert (info.name, info.version, info.status) == (
        "spectral-rules",
        "2.0.0",
        ModelStatus.PLACEHOLDER,
    )
    assert info.validated is False
    assert info.notes == (
        "Heuristic rules on lesion-vs-reference differences; placeholder, not a trained model, "
        "not validated."
    )
    assert clf.method == "rules_v2"


@pytest.mark.parametrize(
    ("hb", "ratio", "label"),
    [
        (0.10, 0.0, "suspicious"),
        (0.06, 0.0, "suspicious"),
        (0.0, -0.12, "suspicious"),
        (0.04, 0.0, "indeterminate"),
        (0.0, -0.06, "indeterminate"),
        (0.01, -0.01, "not_suspicious"),
        (-0.2, 0.3, "not_suspicious"),
    ],
)
def test_rules(clf, hb: float, ratio: float, label: str):
    out = clf.classify(ZERO, _delta(hb, ratio))
    assert out.status == "placeholder" and out.method == "rules_v2" and out.label == label
    assert out.suspicion_score is not None and 0.0 <= out.suspicion_score <= 1.0
    assert "not a calibrated probability" in out.reasons[-1]
    if label == "suspicious":
        assert out.suspicion_score >= 0.5
    if label == "not_suspicious":
        assert out.suspicion_score < 0.5
        assert any("does not rule out disease" in r for r in out.reasons)
    for reason in out.reasons:
        assert not find_positioning_violations(reason)


def test_score_is_monotone_and_capped(clf):
    scores = [clf.score(hb, 0.0) for hb in (0.0, 0.03, 0.06, 0.09, 0.12, 0.5)]
    assert scores == sorted(scores) and scores[2] == pytest.approx(0.5) and scores[-1] == 1.0
    assert clf.score(0.0, -0.2) == 1.0


def test_no_reference_is_placeholder(clf):
    out = clf.classify(ZERO, None)
    assert out.status == "placeholder" and out.label == "indeterminate"
    assert out.suspicion_score is None
    assert out.reasons == [
        "No reference reading from the opposite side; spectral comparison not possible."
    ]
    nan = clf.classify(
        ZERO, ZERO.model_construct(**{**ZERO.model_dump(), "hb_absorption_proxy": float("nan")})
    )
    assert nan.status == "placeholder"


def test_classifier_accepts_any_config_level(analysis_cfg):
    full = RuleBasedSpectralClassifier(analysis_cfg)
    section = RuleBasedSpectralClassifier(analysis_cfg["spectral"]["classifier"])
    default = RuleBasedSpectralClassifier(None)
    assert full.hb_suspicious == section.hb_suspicious == default.hb_suspicious
    weird = RuleBasedSpectralClassifier(
        {"classifier": {"hb_proxy_delta_suspicious": 0, "ratio_555_630_delta_suspicious": 0}}
    )
    assert weird.score(1.0, -1.0) == 0.0


# --------------------------------------------------------------------------------------------
# analyze_spectral
# --------------------------------------------------------------------------------------------


def test_analyze_pairs_lesion_with_reference(rng, fixed_now, qc_cfg, analysis_cfg):
    lesion = make_reading(rng, "s1", L, "lesion", 2.0, reference_reading_id="s2")
    ref = make_reading(rng, "s2", R, "reference", 1.0)
    cal = make_calibration(fixed_now)
    results = analyze_spectral(
        [lesion, ref], {"confirm-01": cal}, qc_cfg, analysis_cfg, fixed_now, qc_fn=qc_pass
    )
    assert [r.reading_id for r in results] == ["s1", "s2"]
    les, rf = results
    assert les.signal_type == "reflectance"
    assert les.calibration.status == CalibrationStatus.CALIBRATED
    assert les.capture.calibration_id == "spcal-test"
    assert les.n_readings == 5 and les.qc.passed
    assert les.reference_reading_id == "s2"
    assert les.delta_vs_reference is not None
    assert les.delta_vs_reference.hb_absorption_proxy > 0.06
    assert les.classifier is not None and les.classifier.label == "suspicious"
    assert rf.classifier is None and rf.delta_vs_reference is None
    for r in results:
        assert set(r.values) == set(AS7341_CHANNELS)
        for text in _texts(r):
            assert not find_positioning_violations(text), text
        SpectralReadingResult.model_validate_json(r.model_dump_json())  # JSON-safe


def test_analyze_similar_tissue_is_not_suspicious(rng, fixed_now, qc_cfg, analysis_cfg):
    lesion = make_reading(rng, "s1", L, "lesion", 1.0)
    ref = make_reading(rng, "s2", R, "reference", 1.0)
    results = analyze_spectral([lesion, ref], {}, qc_cfg, analysis_cfg, fixed_now, qc_fn=qc_pass)
    les = results[0]
    assert les.reference_reading_id == "s2"  # auto-paired contralateral reference
    assert les.signal_type == "relative_signal"
    assert les.calibration.status == CalibrationStatus.UNCALIBRATED
    assert les.classifier is not None and les.classifier.label == "not_suspicious"


def test_analyze_calibration_lookup_by_id_then_device(rng, fixed_now, qc_cfg, analysis_cfg):
    cal = make_calibration(fixed_now, device_id="other-device")
    by_id = make_reading(rng, "s1", L, "lesion", 1.0, device_id=None, calibration_id="spcal-test")
    by_device = make_reading(rng, "s2", R, "reference", 1.0, device_id="other-device")
    none = make_reading(rng, "s3", OralSite.HARD_PALATE, "lesion", 1.0, device_id=None)
    results = analyze_spectral(
        [by_id, by_device, none],
        {"other-device": cal},
        qc_cfg,
        analysis_cfg,
        fixed_now,
        qc_fn=qc_pass,
    )
    assert [r.signal_type for r in results] == ["reflectance", "reflectance", "relative_signal"]
    assert results[1].capture.calibration_id == "spcal-test"
    assert results[2].classifier is not None and results[2].classifier.status == "placeholder"


def test_analyze_skips_classifier_when_qc_fails(rng, fixed_now, qc_cfg, analysis_cfg):
    lesion = make_reading(rng, "s1", L, "lesion", 3.0, reference_reading_id="s2")
    ref = make_reading(rng, "s2", R, "reference", 1.0)
    results = analyze_spectral([lesion, ref], {}, qc_cfg, analysis_cfg, fixed_now, qc_fn=qc_fail)
    les = results[0]
    assert not les.qc.passed
    assert les.classifier is not None and les.classifier.status == "skipped_qc_failed"
    assert les.classifier.label is None and les.classifier.suspicion_score is None
    assert les.features is not None  # still reported for transparency


def test_reference_failing_qc_is_not_used(rng, fixed_now, qc_cfg, analysis_cfg):
    lesion = make_reading(rng, "s1", L, "lesion", 3.0, reference_reading_id="s2")
    ref = make_reading(rng, "s2", R, "reference", 1.0)

    def qc_ref_fails(lit, dark, cfg):
        return qc_fail(lit, dark, cfg) if lit[0] is ref.lit[0] else qc_pass(lit, dark, cfg)

    results = analyze_spectral(
        [lesion, ref], {}, qc_cfg, analysis_cfg, fixed_now, qc_fn=qc_ref_fails
    )
    les = results[0]
    assert les.qc.passed and not results[1].qc.passed
    assert les.delta_vs_reference is None
    assert les.classifier is not None and les.classifier.status == "placeholder"
    assert "did not pass quality checks" in les.classifier.reasons[0]
    assert les.classifier.reasons[-1].startswith("No reference reading")


def test_missing_reference_and_signal_mismatch(rng, fixed_now, qc_cfg, analysis_cfg):
    lesion = make_reading(rng, "s1", L, "lesion", 2.0, reference_reading_id="ghost")
    results = analyze_spectral([lesion], {}, qc_cfg, analysis_cfg, fixed_now, qc_fn=qc_pass)
    assert "was not found" in results[0].classifier.reasons[0]  # type: ignore[union-attr]

    calibrated = make_reading(
        rng,
        "s1",
        L,
        "lesion",
        2.0,
        reference_reading_id="s2",
        calibration_id="spcal-test",
        device_id=None,
    )
    uncal = make_reading(rng, "s2", R, "reference", 1.0, device_id=None)
    results = analyze_spectral(
        [calibrated, uncal],
        {"x": make_calibration(fixed_now)},
        qc_cfg,
        analysis_cfg,
        fixed_now,
        qc_fn=qc_pass,
    )
    assert results[0].delta_vs_reference is None
    assert "cannot be compared" in results[0].classifier.reasons[0]  # type: ignore[union-attr]


def test_qc_function_errors_and_processing_errors(
    rng, fixed_now, qc_cfg, analysis_cfg, monkeypatch
):
    lesion = make_reading(rng, "s1", L, "lesion", 2.0)

    def crash(lit, dark, cfg):
        raise RuntimeError("qc bug")

    out = analyze_spectral([lesion], {}, qc_cfg, analysis_cfg, fixed_now, qc_fn=crash)[0]
    assert not out.qc.passed and out.qc.issues[0].code == "spectral_qc_error"

    def broken_summary(*args, **kwargs):
        raise ValueError("bad")

    monkeypatch.setattr(analyze_mod, "summarise_reading", broken_summary)
    out = analyze_spectral([lesion], {}, qc_cfg, analysis_cfg, fixed_now, qc_fn=qc_pass)[0]
    assert out.qc.issues[0].code == "spectral_processing_error"
    assert out.features is None and out.values == {}
    assert out.classifier is not None and out.classifier.status == "skipped_qc_failed"


def test_custom_classifier_without_helpers(rng, fixed_now, qc_cfg, analysis_cfg):
    inner = RuleBasedSpectralClassifier(analysis_cfg)

    class Minimal:
        info = inner.info

        def classify(self, features, delta):
            return inner.classify(features, delta)

    lesion = make_reading(rng, "s1", L, "lesion", 2.0)
    out = analyze_spectral(
        [lesion], {}, qc_cfg, analysis_cfg, fixed_now, classifier=Minimal(), qc_fn=qc_fail
    )[0]
    assert out.classifier is not None and out.classifier.status == "skipped_qc_failed"
    assert out.classifier.method == "spectral-rules"
    ok = analyze_spectral(
        [lesion], {}, qc_cfg, {"spectral": {}}, fixed_now, classifier=Minimal(), qc_fn=qc_pass
    )[0]
    assert ok.classifier is not None and ok.classifier.status == "placeholder"


def test_default_qc_is_lazy(rng, fixed_now, qc_cfg, analysis_cfg, monkeypatch):
    lesion = make_reading(rng, "s1", L, "lesion", 2.0)
    # No qc module importable: readings are not interpreted rather than trusted blindly.
    monkeypatch.setitem(sys.modules, "mukha_pipeline.qc", None)
    monkeypatch.setitem(sys.modules, "mukha_pipeline.qc.spectral", None)
    out = analyze_spectral([lesion], {}, qc_cfg, analysis_cfg, fixed_now)[0]
    assert out.qc.issues[0].code == "spectral_qc_unavailable"
    assert out.classifier is not None and out.classifier.status == "skipped_qc_failed"

    fake = types.ModuleType("mukha_pipeline.qc")
    fake.qc_spectral = qc_pass  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mukha_pipeline.qc", fake)
    out = analyze_spectral([lesion], {}, qc_cfg, analysis_cfg, fixed_now)[0]
    assert out.qc.passed


def test_empty_input(fixed_now, qc_cfg, analysis_cfg):
    assert analyze_spectral([], {}, qc_cfg, analysis_cfg, fixed_now, qc_fn=qc_pass) == []


def test_parsed_file_end_to_end(rng, fixed_now, qc_cfg, analysis_cfg):
    """CSV with a white calibration -> parse -> analyse: reflectance, auto-pairing, suspicion."""
    from wlspec_helpers import spectral_csv

    from mukha_pipeline.io.spectral import parse_spectral_file

    lesion = make_reading(rng, "s1", L, "lesion", 2.0)
    ref = make_reading(rng, "s2", R, "reference", 1.0)
    data = spectral_csv([lesion, ref], make_calibration(fixed_now), explicit_refs=False)
    parsed = parse_spectral_file(data, "probe.csv", now=fixed_now, synthetic=True)
    assert parsed.calibration is not None
    cals = {parsed.calibration.device_id: parsed.calibration}
    results = analyze_spectral(
        parsed.readings, cals, qc_cfg, analysis_cfg, fixed_now, qc_fn=qc_pass
    )
    les = results[0]
    assert les.signal_type == "reflectance" and les.capture.synthetic
    assert les.reference_reading_id == "s2"
    assert les.classifier is not None and les.classifier.label == "suspicious"
    truth = true_band_reflectance(2.0)
    assert les.values["F5_555"] == pytest.approx(truth["F5_555"], rel=0.02)
