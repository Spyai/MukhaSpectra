"""Spectral probe QC (AS7341 lit + dark frames, SYNTHETIC counts)."""

import math

import numpy as np
import pytest
import qc_helpers as qh

from mukha_pipeline.qc import qc_spectral
from mukha_pipeline.qc.spectral_qc import frame_matrix
from mukha_pipeline.schema.models import AS7341_CHANNELS, QCSeverity
from mukha_pipeline.types import SpectralFrame


def codes(result):
    return [i.code for i in result.issues]


def issue(result, code):
    return next(i for i in result.issues if i.code == code)


@pytest.fixture
def good(rng):
    return qh.make_frames(rng, 5, qh.LIT_COUNTS), qh.make_frames(rng, 2, qh.DARK_COUNTS)


def test_good_reading_passes_with_metrics(qc_cfg, good):
    r = qc_spectral(*good, qc_cfg)
    assert r.passed and r.score == 1.0, codes(r)
    for key in (
        "spectral.n_lit",
        "spectral.n_dark",
        "spectral.max_raw_fraction",
        "spectral.ambient_ratio",
        "spectral.clear_net",
        "spectral.dark_nir",
        "spectral.min_net_signal",
        "spectral.max_cv",
    ):
        assert key in r.metrics, key
    assert r.metrics["spectral.max_cv"] < qc_cfg["spectral"]["repeatability_cv_max"]


@pytest.mark.parametrize("which", ["lit", "dark", "both"])
def test_no_readings(qc_cfg, good, which):
    lit, dark = good
    lit = [] if which in ("lit", "both") else lit
    dark = [] if which in ("dark", "both") else dark
    r = qc_spectral(lit, dark, qc_cfg)
    assert codes(r) == ["spectral_no_readings"] and not r.passed


def test_none_inputs_are_no_readings(qc_cfg):
    assert codes(qc_spectral(None, None, qc_cfg)) == ["spectral_no_readings"]


def test_few_readings_is_a_warning(qc_cfg, rng):
    lit = qh.make_frames(rng, 2, qh.LIT_COUNTS)
    r = qc_spectral(lit, qh.make_frames(rng, 1, qh.DARK_COUNTS), qc_cfg)
    assert codes(r) == ["spectral_few_readings"] and r.passed
    assert issue(r, "spectral_few_readings").value == 2.0


def test_single_reading_skips_repeatability(qc_cfg, rng):
    r = qc_spectral(
        qh.make_frames(rng, 1, qh.LIT_COUNTS), qh.make_frames(rng, 1, qh.DARK_COUNTS), qc_cfg
    )
    assert "spectral.max_cv" not in r.metrics
    assert codes(r) == ["spectral_few_readings"]


def test_saturation(qc_cfg, rng, good):
    lit = qh.make_frames(rng, 5, qh.LIT_COUNTS, override={"F7_630": 64000.0})
    r = qc_spectral(lit, good[1], qc_cfg)
    assert "spectral_saturated" in codes(r) and not r.passed
    assert issue(r, "spectral_saturated").value == pytest.approx(64000 / 65535)


def test_ambient_leak(qc_cfg, rng, good):
    dark = qh.make_frames(rng, 2, qh.DARK_COUNTS, override={"CLEAR": 6000.0})
    r = qc_spectral(good[0], dark, qc_cfg)
    assert codes(r) == ["spectral_ambient_leak"] and not r.passed
    assert r.metrics["spectral.ambient_ratio"] == pytest.approx(0.2, abs=0.01)


def test_no_contact_from_clear(qc_cfg, rng, good):
    weak = {k: 60.0 for k in qh.LIT_COUNTS} | {"CLEAR": 300.0, "NIR": 150.0}
    r = qc_spectral(qh.make_frames(rng, 5, weak), good[1], qc_cfg)
    assert "probe_no_contact" in codes(r) and not r.passed
    assert issue(r, "probe_no_contact").metric == "spectral.clear_net"
    assert "spectral_signal_low" not in codes(r)  # one cause, one prompt


def test_no_contact_from_dark_nir(qc_cfg, rng, good):
    dark = qh.make_frames(rng, 2, qh.DARK_COUNTS, override={"NIR": 2600.0})
    r = qc_spectral(good[0], dark, qc_cfg)
    assert codes(r) == ["probe_no_contact"]
    assert issue(r, "probe_no_contact").metric == "spectral.dark_nir"


def test_poor_repeatability(qc_cfg, rng, good):
    lit = qh.make_frames(rng, 5, qh.LIT_COUNTS, cv=0.25)
    r = qc_spectral(lit, good[1], qc_cfg)
    assert "spectral_poor_repeatability" in codes(r)
    assert issue(r, "spectral_poor_repeatability").severity == QCSeverity.WARNING


def test_one_weak_band_is_a_warning(qc_cfg, rng, good):
    lit = qh.make_frames(rng, 5, qh.LIT_COUNTS | {"F1_415": 70.0}, cv=0.0)
    r = qc_spectral(lit, good[1], qc_cfg)
    assert codes(r) == ["spectral_signal_low"] and r.passed
    low = issue(r, "spectral_signal_low")
    assert low.severity == QCSeverity.WARNING and low.value == pytest.approx(30.0, abs=1.0)
    assert r.metrics["spectral.weak_bands"] == 1.0
    # the weak band is left out of the repeatability check (its CV is noise-dominated)
    assert r.metrics["spectral.max_cv"] < 0.05


def test_mostly_weak_spectrum_is_an_error(qc_cfg, rng, good):
    weak = {k: 70.0 for k in qh.LIT_COUNTS} | {"CLEAR": 30000.0, "NIR": 3000.0}
    r = qc_spectral(qh.make_frames(rng, 5, weak, cv=0.0), good[1], qc_cfg)
    assert codes(r) == ["spectral_signal_low"] and not r.passed
    assert issue(r, "spectral_signal_low").severity == QCSeverity.ERROR


def test_missing_spectral_band_is_an_error(qc_cfg, rng, good):
    lit = qh.make_frames(rng, 5, qh.LIT_COUNTS, drop=("F5_555",))
    r = qc_spectral(lit, good[1], qc_cfg)
    assert codes(r) == ["spectral_channels_missing"] and not r.passed


@pytest.mark.parametrize("drop", [("CLEAR",), ("NIR",), ("CLEAR", "NIR")])
def test_missing_clear_or_nir_skips_checks_with_warning(qc_cfg, rng, drop):
    lit = qh.make_frames(rng, 5, qh.LIT_COUNTS, drop=drop)
    dark = qh.make_frames(rng, 2, qh.DARK_COUNTS, drop=drop)
    r = qc_spectral(lit, dark, qc_cfg)
    assert codes(r) == ["spectral_checks_skipped"] and r.passed
    assert issue(r, "spectral_checks_skipped").value == float(len(drop))


def test_nan_and_junk_values_are_treated_as_missing(qc_cfg, rng, good):
    lit = qh.make_frames(rng, 5, qh.LIT_COUNTS)
    for f in lit:
        f.channels["F3_480"] = math.nan
        f.channels["CLEAR"] = "n/a"
    r = qc_spectral(lit, good[1], qc_cfg)
    assert codes(r) == ["spectral_channels_missing", "spectral_checks_skipped"]
    assert not r.passed


def test_partly_missing_frames_use_the_rest(qc_cfg, rng, good):
    lit = qh.make_frames(rng, 5, qh.LIT_COUNTS)
    del lit[0].channels["F2_445"]
    assert qc_spectral(lit, good[1], qc_cfg).passed


def test_frames_without_channels_dict(qc_cfg, good):
    broken = [SpectralFrame(channels=None), SpectralFrame(channels={})]  # type: ignore[arg-type]
    r = qc_spectral(broken, good[1], qc_cfg)
    assert "spectral_channels_missing" in codes(r) and not r.passed


def test_zero_lit_signal_with_dark_light(qc_cfg, rng):
    lit = qh.make_frames(rng, 3, {k: 0.0 for k in AS7341_CHANNELS}, cv=0.0)
    dark = qh.make_frames(rng, 1, qh.DARK_COUNTS, cv=0.0)
    r = qc_spectral(lit, dark, qc_cfg)
    assert {"spectral_ambient_leak", "probe_no_contact"} <= set(codes(r))
    assert "spectral.ambient_ratio" not in r.metrics  # infinite ratio is not stored
    assert issue(r, "spectral_ambient_leak").value is None


def test_all_zero_reading(qc_cfg, rng):
    zeros = {k: 0.0 for k in AS7341_CHANNELS}
    r = qc_spectral(
        qh.make_frames(rng, 3, zeros, cv=0), qh.make_frames(rng, 1, zeros, cv=0), qc_cfg
    )
    assert "probe_no_contact" in codes(r) and "spectral_ambient_leak" not in codes(r)
    assert r.metrics["spectral.ambient_ratio"] == 0.0


def test_gain_or_integration_change_is_flagged(qc_cfg, rng, good):
    dark = qh.make_frames(rng, 2, qh.DARK_COUNTS, gain=16.0)
    r = qc_spectral(good[0], dark, qc_cfg)
    assert codes(r) == ["spectral_settings_mismatch"] and r.passed
    dark = qh.make_frames(rng, 2, qh.DARK_COUNTS, integration_ms=100.0)
    assert codes(qc_spectral(good[0], dark, qc_cfg)) == ["spectral_settings_mismatch"]
    unknown = qh.make_frames(rng, 2, qh.DARK_COUNTS, gain=None, integration_ms=None)
    assert qc_spectral(good[0], unknown, qc_cfg).passed


def test_frame_matrix_shape_and_missing(rng):
    frames = qh.make_frames(rng, 3, qh.LIT_COUNTS, drop=("NIR",))
    m = frame_matrix(frames, AS7341_CHANNELS)
    assert m.shape == (3, len(AS7341_CHANNELS))
    assert np.isnan(m[:, -1]).all() and np.isfinite(m[:, :-1]).all()
    assert frame_matrix([], AS7341_CHANNELS).shape == (0, len(AS7341_CHANNELS))


def test_score_reflects_severity(qc_cfg, rng, good):
    lit = qh.make_frames(rng, 2, qh.LIT_COUNTS, override={"F7_630": 64000.0})
    r = qc_spectral(lit, good[1], qc_cfg)
    w = qc_cfg["score_weights"]
    assert r.score == pytest.approx(max(0.0, 1 - w["error"] - w["warning"]))


def test_clear_no_contact_is_not_repeated_by_nir(qc_cfg, rng):
    lit = qh.make_frames(rng, 5, qh.LIT_COUNTS | {"CLEAR": 300.0})
    dark = qh.make_frames(rng, 2, qh.DARK_COUNTS, override={"NIR": 5000.0})
    r = qc_spectral(lit, dark, qc_cfg)
    assert codes(r).count("probe_no_contact") == 1
    assert issue(r, "probe_no_contact").metric == "spectral.clear_net"
    assert r.metrics["spectral.dark_nir"] == pytest.approx(5000.0)


def test_junk_settings_do_not_crash(qc_cfg, rng, good):
    dark = qh.make_frames(rng, 2, qh.DARK_COUNTS, gain="high")  # type: ignore[arg-type]
    assert qc_spectral(good[0], dark, qc_cfg).passed


def test_results_are_json_safe(qc_cfg, rng):
    zeros = {k: 0.0 for k in AS7341_CHANNELS}
    r = qc_spectral(qh.make_frames(rng, 3, zeros), qh.make_frames(rng, 1, qh.DARK_COUNTS), qc_cfg)
    text = r.model_dump_json()
    assert "NaN" not in text and "Infinity" not in text
    assert all(k.startswith("spectral.") for k in r.metrics)
