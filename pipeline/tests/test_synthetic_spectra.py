"""Synthetic AS7341 readings, white reference, calibration and the spectral CSV v1 writer."""

from __future__ import annotations

import csv
import io
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from mukha_pipeline.schema.models import (
    AS7341_CHANNELS,
    CaptureKind,
    SourceAdapter,
)
from mukha_pipeline.sites import OralSite
from mukha_pipeline.synthetic import (
    generate_spectral_reading,
    generate_white_reference,
    make_spectral_calibration,
    spectral_inputs_to_csv,
)
from mukha_pipeline.synthetic.spectra import (
    ADC_FULL_SCALE,
    CSV_COLUMNS,
    expected_signal,
    tissue_reflectance,
)
from mukha_pipeline.types import SpectralCalibration, SpectralInput

NOW = datetime(2026, 10, 14, 9, 0, tzinfo=UTC)
CONTRACT_HEADER = (
    "reading_id,site,role,kind,repeat,F1_415,F2_445,F3_480,F4_515,F5_555,F6_590,F7_630,"
    "F8_680,CLEAR,NIR,gain,integration_ms,reference_reading_id,lesion_ref,device_id"
)


def _reading(rng, hb=0.25, **kw) -> SpectralInput:
    args = {"reading_id": "r1", "site": "buccal_mucosa_left", "role": "reference"}
    args.update(kw)
    return generate_spectral_reading(rng, hb=hb, **args)


def _mean(frames, ch):
    return float(np.mean([f.channels[ch] for f in frames]))


def test_reading_structure_and_metadata() -> None:
    r = _reading(
        np.random.default_rng(0),
        n=4,
        role="lesion",
        reference_reading_id="r2",
        lesion_ref="r1-region",
        calibration_id="cal-1",
    )
    assert isinstance(r, SpectralInput)
    assert len(r.lit) == 4 and len(r.dark) >= 1
    assert r.site is OralSite.BUCCAL_MUCOSA_LEFT and r.role == "lesion"
    assert r.reference_reading_id == "r2" and r.lesion_ref == "r1-region"
    for f in r.lit + r.dark:
        assert set(f.channels) == set(AS7341_CHANNELS)
        assert f.gain == 64.0 and f.integration_ms == 50.0
    m = r.meta
    assert m.synthetic is True and m.source_adapter is SourceAdapter.SYNTHETIC
    assert m.kind is CaptureKind.SPECTRAL and m.device_id == "SYN-CONFIRM-01"
    assert m.calibration_id == "cal-1" and m.firmware_version == "synthetic"


def test_counts_realistic_and_never_saturated() -> None:
    r = _reading(np.random.default_rng(1), hb=0.6)
    for f in r.lit:
        assert max(f.channels.values()) < 0.95 * ADC_FULL_SCALE
        assert f.channels["CLEAR"] - _mean(r.dark, "CLEAR") > 5000
        for ch in AS7341_CHANNELS[:8]:
            assert f.channels[ch] - _mean(r.dark, ch) > 300, ch
    for f in r.dark:
        assert all(150 <= v <= 450 for v in f.channels.values())
    assert _mean(r.lit, "NIR") - _mean(r.dark, "NIR") < 1000, "white LED barely reaches NIR"
    assert max(_mean(r.lit, c) for c in AS7341_CHANNELS[:8]) > 5000, "thousands of counts"


def test_white_reference_bright_but_unsaturated() -> None:
    white, white_dark = generate_white_reference(np.random.default_rng(2))
    assert len(white) == 5 and len(white_dark) == 3
    for f in white:
        assert max(f.channels.values()) < 0.95 * ADC_FULL_SCALE
    tissue = _reading(np.random.default_rng(2))
    for ch in AS7341_CHANNELS[:8]:
        assert _mean(white, ch) > _mean(tissue.lit, ch)


def test_tissue_reflectance_shape() -> None:
    wl = np.array([415.0, 680.0])
    base = tissue_reflectance(0.0, wl)
    assert base[0] == pytest.approx(0.25, abs=0.01) and base[1] == pytest.approx(0.55, abs=0.01)
    assert tissue_reflectance(0.6, wl)[0] < tissue_reflectance(0.25, wl)[0]


def test_more_blood_deepens_the_haemoglobin_dip() -> None:
    def dip(hb: float) -> tuple[float, float]:
        s = expected_signal(hb)
        shoulders = (s["F4_515"] + s["F7_630"]) / 2
        return s["F5_555"] / shoulders, s["F6_590"] / s["F7_630"]

    normal, moderate, high = dip(0.25), dip(0.36), dip(0.6)
    assert normal[0] > moderate[0] > high[0]
    assert normal[1] > moderate[1] > high[1]


def test_dip_ordering_survives_noise_in_raw_counts() -> None:
    rng = np.random.default_rng(3)

    def ratio(hb: float) -> float:
        r = _reading(rng, hb=hb, n=5)
        sig = {ch: _mean(r.lit, ch) - _mean(r.dark, ch) for ch in AS7341_CHANNELS}
        return sig["F5_555"] / (sig["F4_515"] + sig["F7_630"])

    assert ratio(0.25) > ratio(0.36) > ratio(0.6)


def test_repeats_are_repeatable() -> None:
    r = _reading(np.random.default_rng(4), n=8)
    for ch in AS7341_CHANNELS[:8]:
        values = np.array([f.channels[ch] for f in r.lit])
        assert values.std() / values.mean() < 0.05


def test_gain_and_integration_scale_the_signal() -> None:
    a = _reading(np.random.default_rng(5))
    b = _reading(np.random.default_rng(5), gain=32.0, integration_ms=50.0)
    assert _mean(b.lit, "F5_555") < 0.6 * _mean(a.lit, "F5_555")
    assert b.lit[0].gain == 32.0


def test_hb_dip_alias_and_validation() -> None:
    a = generate_spectral_reading(
        np.random.default_rng(6),
        hb_dip=0.3,
        reading_id="x",
        site="lateral_tongue_left",
        role="reference",
    )
    b = _reading(np.random.default_rng(6), hb=0.3, reading_id="x", site="lateral_tongue_left")
    assert a.lit[0].channels == b.lit[0].channels
    with pytest.raises(ValueError, match="hb"):
        generate_spectral_reading(
            np.random.default_rng(0), reading_id="x", site="lateral_tongue_left", role="reference"
        )
    with pytest.raises(ValueError, match="role"):
        _reading(np.random.default_rng(0), role="normal")
    with pytest.raises(ValueError, match="at least one"):
        _reading(np.random.default_rng(0), n=0)


def test_determinism() -> None:
    a = _reading(np.random.default_rng(7))
    b = _reading(np.random.default_rng(7))
    assert [f.channels for f in a.lit] == [f.channels for f in b.lit]


def test_spectral_calibration_is_synthetic_and_expires() -> None:
    white, white_dark = generate_white_reference(np.random.default_rng(8))
    cal = make_spectral_calibration(white, white_dark, created_at=NOW)
    assert isinstance(cal, SpectralCalibration) and cal.synthetic is True
    assert cal.expires_at - cal.created_at == timedelta(hours=24)
    assert cal.device_id == "SYN-CONFIRM-01" and cal.calibration_id.startswith("SYN-")
    assert set(cal.white) == set(AS7341_CHANNELS) == set(cal.dark)
    assert cal.white["F5_555"] > 10 * cal.dark["F5_555"]
    assert cal.gain == 64.0 and cal.integration_ms == 50.0
    with pytest.raises(ValueError):
        make_spectral_calibration([], white_dark, created_at=NOW)


def _parse(text: str) -> tuple[list[str], list[dict[str, str]]]:
    reader = csv.DictReader(io.StringIO(text))
    return list(reader.fieldnames or []), list(reader)


def test_csv_header_matches_the_contract_exactly() -> None:
    assert ",".join(CSV_COLUMNS) == CONTRACT_HEADER
    text = spectral_inputs_to_csv([])
    assert text.splitlines() == [CONTRACT_HEADER]


def test_csv_round_trip_structure_with_white_frames() -> None:
    rng = np.random.default_rng(9)
    lesion = _reading(
        rng,
        hb=0.6,
        reading_id="L",
        site="lateral_tongue_right",
        role="lesion",
        reference_reading_id="R",
        n=5,
    )
    ref = _reading(rng, hb=0.25, reading_id="R", site="lateral_tongue_left", n=5)
    white, white_dark = generate_white_reference(rng, n=4, n_dark=2)
    header, rows = _parse(spectral_inputs_to_csv([lesion, ref], (white, white_dark)))
    assert header == list(CSV_COLUMNS)
    assert len(rows) == 4 + 2 + 2 * (5 + 3)
    kinds = [r["kind"] for r in rows]
    assert kinds.count("white") == 4 and kinds.count("white_dark") == 2
    assert kinds.count("lit") == 10 and kinds.count("dark") == 6
    for r in rows:
        if r["kind"].startswith("white"):
            assert r["reading_id"] == "calibration" and r["site"] == "" and r["role"] == ""
        else:
            assert r["site"] in {s.value for s in OralSite} and r["role"] in {"lesion", "reference"}
        for ch in AS7341_CHANNELS:
            assert r[ch].isdigit(), "raw counts are integers"
        assert r["gain"] == "64" and r["integration_ms"] == "50"
        assert r["device_id"] == "SYN-CONFIRM-01"
    lesion_rows = [r for r in rows if r["reading_id"] == "L"]
    assert {r["reference_reading_id"] for r in lesion_rows} == {"R"}
    lit_repeats = sorted(int(r["repeat"]) for r in lesion_rows if r["kind"] == "lit")
    assert lit_repeats == [0, 1, 2, 3, 4]
    first_lit = next(r for r in lesion_rows if r["kind"] == "lit")
    assert int(first_lit["F5_555"]) == int(lesion.lit[0].channels["F5_555"])


def test_csv_with_calibration_object_writes_mean_rows() -> None:
    rng = np.random.default_rng(10)
    cal = make_spectral_calibration(*generate_white_reference(rng), created_at=NOW)
    reading = _reading(rng)
    _, rows = _parse(spectral_inputs_to_csv([reading], cal))
    white_rows = [r for r in rows if r["reading_id"] == "calibration"]
    assert [r["kind"] for r in white_rows] == ["white", "white_dark"]
    assert float(white_rows[0]["F5_555"]) == pytest.approx(cal.white["F5_555"], abs=1.0)


def test_csv_formats_non_integral_values() -> None:
    rng = np.random.default_rng(11)
    r = _reading(rng, gain=0.5, integration_ms=27.8)
    _, rows = _parse(spectral_inputs_to_csv([r]))
    assert rows[0]["gain"] == "0.5" and rows[0]["integration_ms"] == "27.8"
