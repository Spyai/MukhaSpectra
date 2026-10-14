"""Tests for mukha_pipeline.calibration (status, image calibration, reflectance, storage)."""

from __future__ import annotations

import io
import math
from datetime import UTC, datetime, timedelta, timezone

import numpy as np
import pytest

from mukha_pipeline.calibration import (
    UNCALIBRATED_NOTE,
    CalibrationError,
    as_utc,
    compute_af_scale,
    compute_color_matrix,
    compute_flat_field,
    create_image_calibration,
    create_spectral_calibration,
    image_calibration_from_bytes,
    image_calibration_status,
    image_calibration_to_bytes,
    reflectance,
    spectral_calibration_status,
)
from mukha_pipeline.calibration.common import central_region, finite_median, luminance
from mukha_pipeline.positioning import find_positioning_violations
from mukha_pipeline.schema.models import CalibrationStatus
from mukha_pipeline.types import ImageCalibration, SpectralCalibration

# --------------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------------


def vignetted_white(h: int = 120, w: int = 160, tint=(0.80, 0.90, 0.70)) -> np.ndarray:
    """White target lit by a centred beam: radial fall-off plus a colour tint."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r2 = ((xx - w / 2) / w) ** 2 + ((yy - h / 2) / h) ** 2
    profile = 1.0 - 0.8 * r2
    return (profile[..., None] * np.array(tint, np.float32)).astype(np.float32)


def fluorescence_card(green: float = 0.25, h: int = 80, w: int = 100) -> np.ndarray:
    card = np.zeros((h, w, 3), np.float32)
    card[..., 1] = green
    card[..., 0] = 0.03
    card[:5, :] = 0.95  # bright edge / glare outside the central region must not matter
    return card


def image_cal(now: datetime, hours: float = 24) -> ImageCalibration:
    return ImageCalibration(
        calibration_id="cal-test",
        device_id="scout-01",
        created_at=now,
        expires_at=now + timedelta(hours=hours),
        flat_field=np.ones((4, 4), np.float32),
        color_matrix=np.eye(3),
    )


def spectral_cal(now: datetime, hours: float = 24) -> SpectralCalibration:
    return SpectralCalibration(
        calibration_id="cal-spec",
        device_id="confirm-01",
        created_at=now,
        expires_at=now + timedelta(hours=hours),
        dark={"F1_415": 10.0},
        white={"F1_415": 1000.0},
    )


# --------------------------------------------------------------------------------------------
# Status
# --------------------------------------------------------------------------------------------


def test_status_none_is_uncalibrated(fixed_now):
    for fn in (image_calibration_status, spectral_calibration_status):
        info = fn(None, fixed_now)
        assert info.status is CalibrationStatus.UNCALIBRATED
        assert info.note == UNCALIBRATED_NOTE
        assert info.calibration_id is None and info.expires_at is None


def test_status_calibrated_and_expired(fixed_now):
    cal = image_cal(fixed_now - timedelta(hours=2))
    info = image_calibration_status(cal, fixed_now)
    assert info.status is CalibrationStatus.CALIBRATED
    assert info.calibration_id == "cal-test"
    assert info.expires_at == cal.expires_at
    assert info.note is None

    old = image_cal(fixed_now - timedelta(hours=30))
    info = image_calibration_status(old, fixed_now)
    assert info.status is CalibrationStatus.EXPIRED
    assert info.calibration_id == "cal-test"
    assert "expired" in info.note and "white target" in info.note

    spec = spectral_calibration_status(spectral_cal(fixed_now - timedelta(hours=25)), fixed_now)
    assert spec.status is CalibrationStatus.EXPIRED
    assert "white reflectance standard" in spec.note


def test_status_expiry_boundary_is_strict(fixed_now):
    cal = image_cal(fixed_now - timedelta(hours=24))  # expires exactly now
    assert image_calibration_status(cal, fixed_now).status is CalibrationStatus.CALIBRATED
    later = fixed_now + timedelta(microseconds=1)
    assert image_calibration_status(cal, later).status is CalibrationStatus.EXPIRED


def test_status_mixes_naive_and_aware_datetimes(fixed_now):
    naive_created = datetime(2026, 10, 14, 8, 0)  # naive = UTC
    cal = image_cal(naive_created)
    info = image_calibration_status(cal, fixed_now)  # aware now
    assert info.status is CalibrationStatus.CALIBRATED
    assert info.expires_at.tzinfo is not None
    # Naive now against an aware record.
    aware = spectral_cal(fixed_now)
    naive_now = datetime(2026, 10, 16, 0, 0)
    assert spectral_calibration_status(aware, naive_now).status is CalibrationStatus.EXPIRED
    # Other time zones are converted, not compared by wall clock.
    ist = timezone(timedelta(hours=5, minutes=30))
    now_ist = fixed_now.astimezone(ist)
    assert image_calibration_status(cal, now_ist).status is CalibrationStatus.CALIBRATED


def test_status_notes_future_record(fixed_now):
    cal = image_cal(fixed_now + timedelta(hours=3))
    info = image_calibration_status(cal, fixed_now)
    assert info.status is CalibrationStatus.CALIBRATED
    assert "device clock" in info.note


def test_status_without_clock_uses_current_time():
    cal = image_cal(datetime.now(UTC) - timedelta(hours=1))
    assert image_calibration_status(cal, None).status is CalibrationStatus.CALIBRATED


def test_as_utc():
    assert as_utc(datetime(2026, 1, 1)).tzinfo is UTC
    ist = timezone(timedelta(hours=5, minutes=30))
    assert as_utc(datetime(2026, 1, 1, 5, 30, tzinfo=ist)) == datetime(2026, 1, 1, tzinfo=UTC)
    assert as_utc(None).tzinfo is UTC


# --------------------------------------------------------------------------------------------
# Flat field
# --------------------------------------------------------------------------------------------


def test_flat_field_mean_one_and_corrects_vignetting():
    white = vignetted_white()
    flat = compute_flat_field(white, blur_sigma_px=5)
    assert flat.dtype == np.float32 and flat.shape == white.shape[:2]
    assert abs(float(flat.mean()) - 1.0) < 1e-3
    assert flat[60, 80] > flat[0, 0]  # centre brighter than the corner
    corrected = luminance(white) / flat
    raw_cv = luminance(white).std() / luminance(white).mean()
    assert corrected.std() / corrected.mean() < 0.25 * raw_cv


def test_flat_field_is_clipped_and_handles_bad_input():
    white = np.full((50, 60, 3), 0.8, np.float32)
    white[:, :10] = 0.0  # target does not fill the field
    flat = compute_flat_field(white, blur_sigma_px=0)
    assert flat.min() >= 0.2 and flat.max() <= 5.0
    assert np.all(compute_flat_field(np.zeros((20, 30, 3), np.float32)) == 1.0)
    nan_img = np.full((20, 30, 3), np.nan, np.float32)
    assert np.all(compute_flat_field(nan_img) == 1.0)
    partial = np.full((20, 30, 3), 0.5, np.float32)
    partial[3:6, 4:9] = np.nan
    assert np.all(np.isfinite(compute_flat_field(partial)))
    assert compute_flat_field(np.zeros((0, 0, 3), np.float32)).shape == (1, 1)
    gray_u8 = np.full((10, 10), 200, np.uint8)
    np.testing.assert_allclose(compute_flat_field(gray_u8), 1.0, atol=1e-5)


# --------------------------------------------------------------------------------------------
# Colour matrix
# --------------------------------------------------------------------------------------------


def test_color_matrix_white_balance_makes_target_neutral():
    white = vignetted_white(tint=(0.6, 0.8, 0.5))
    m = compute_color_matrix(white)
    assert m.shape == (3, 3) and m.dtype == np.float64
    assert m[1, 1] == pytest.approx(1.0)
    assert np.count_nonzero(m - np.diag(np.diag(m))) == 0
    balanced = central_region(white).reshape(-1, 3) @ m.T
    med = np.median(balanced, axis=0)
    assert med[0] == pytest.approx(med[1], rel=1e-3)
    assert med[2] == pytest.approx(med[1], rel=1e-3)


def test_color_matrix_ignores_clipped_pixels_and_clamps_gains():
    white = np.full((40, 40, 3), (0.5, 0.6, 0.4), np.float32)
    white[15:25, 15:25] = 1.0  # a clipped highlight in the centre
    m = compute_color_matrix(white)
    assert m[0, 0] == pytest.approx(0.6 / 0.5, rel=1e-3)
    tinted = np.full((20, 20, 3), (0.01, 0.6, 0.6), np.float32)
    assert compute_color_matrix(tinted)[0, 0] == pytest.approx(5.0)
    dark_blue = np.full((20, 20, 3), (0.5, 0.5, 0.0), np.float32)
    assert compute_color_matrix(dark_blue)[2, 2] == 1.0


def test_color_matrix_chart_least_squares(rng):
    true_m = np.array([[1.2, -0.1, 0.0], [0.05, 0.95, 0.02], [-0.02, 0.1, 1.1]])
    reference = rng.uniform(0.05, 0.9, size=(24, 3))
    measured = reference @ np.linalg.inv(true_m).T  # reference = measured @ M.T
    m = compute_color_matrix(None, measured, reference)
    np.testing.assert_allclose(m, true_m, atol=1e-9)
    # NaN patches are dropped, the rest still fit.
    measured_nan = measured.copy()
    measured_nan[0, 1] = np.nan
    np.testing.assert_allclose(
        compute_color_matrix(None, measured_nan, reference), true_m, atol=1e-9
    )


def test_color_matrix_degenerate_inputs_fall_back():
    eye = np.eye(3)
    np.testing.assert_array_equal(compute_color_matrix(None), eye)
    # Too few patches, rank-deficient chart, mismatched shapes, unreshapeable arrays.
    white = np.full((10, 10, 3), (0.4, 0.8, 0.8), np.float32)
    fallback = compute_color_matrix(white)
    two = np.ones((2, 3))
    assert np.array_equal(compute_color_matrix(white, two, two), fallback)
    rank1 = np.ones((5, 3))
    assert np.array_equal(compute_color_matrix(white, rank1, rank1), fallback)
    assert np.array_equal(compute_color_matrix(white, np.ones((6, 3)), np.ones((3, 3))), fallback)
    assert np.array_equal(compute_color_matrix(white, np.ones(4), np.ones(4)), fallback)
    np.testing.assert_array_equal(compute_color_matrix(np.ones((5, 5), np.float32)), eye)
    np.testing.assert_array_equal(compute_color_matrix(np.full((5, 5, 3), np.nan)), eye)
    inf_chart = np.array([[1e308, 0, 0], [0, 1e-308, 0], [0, 0, 1e-308]])
    assert np.all(np.isfinite(compute_color_matrix(white, inf_chart, np.eye(3) * 1e308)))


# --------------------------------------------------------------------------------------------
# AF scale
# --------------------------------------------------------------------------------------------


def test_af_scale_uses_central_median():
    card = fluorescence_card(green=0.25)
    assert compute_af_scale(card, 0.5) == pytest.approx(2.0)
    gray = np.full((10, 10), 0.4, np.float32)
    assert compute_af_scale(gray, 0.2) == pytest.approx(0.5)


@pytest.mark.parametrize(
    "card, expected",
    [
        (np.zeros((10, 10, 3), np.float32), 0.5),
        (np.full((10, 10, 3), np.nan, np.float32), 0.5),
        (np.zeros((0, 0, 3), np.float32), 0.5),
        (np.full((10, 10, 3), 0.3, np.float32), 0.0),
        (np.full((10, 10, 3), 0.3, np.float32), float("nan")),
        (np.full((10, 10, 3), 0.3, np.float32), "junk"),
    ],
)
def test_af_scale_guards(card, expected):
    assert compute_af_scale(card, expected) == 1.0


# --------------------------------------------------------------------------------------------
# Record creation
# --------------------------------------------------------------------------------------------


def test_create_image_calibration(fixed_now):
    cal = create_image_calibration(
        "scout-01", vignetted_white(), fluorescence_card(0.25), now=fixed_now, synthetic=True
    )
    assert cal.calibration_id.startswith("cal-") and len(cal.calibration_id) > 10
    assert cal.device_id == "scout-01"
    assert cal.created_at == fixed_now
    assert cal.expires_at == fixed_now + timedelta(hours=24)
    assert cal.af_scale == pytest.approx(2.0)
    assert cal.synthetic is True
    assert cal.flat_field.shape == (120, 160)
    assert cal.color_matrix.shape == (3, 3)
    other = create_image_calibration(
        "scout-01", vignetted_white(), None, now=fixed_now, valid_hours=8
    )
    assert other.calibration_id != cal.calibration_id
    assert other.af_scale == 1.0
    assert other.expires_at - other.created_at == timedelta(hours=8)
    assert image_calibration_status(cal, fixed_now).status is CalibrationStatus.CALIBRATED


def test_create_image_calibration_naive_now_is_utc():
    cal = create_image_calibration("d", vignetted_white(), None, now=datetime(2026, 10, 14, 9))
    assert cal.created_at == datetime(2026, 10, 14, 9, tzinfo=UTC)


@pytest.mark.parametrize(
    "white, card, fragment",
    [
        (None, None, "missing"),
        (np.zeros((20, 20, 3), np.float32), None, "too dark"),
        (np.ones((20, 20, 3), np.float32), None, "overexposed"),
        (vignetted_white(), np.zeros((0, 0, 3), np.float32), "empty"),
        (vignetted_white(), fluorescence_card(0.005), "violet LED"),
        (vignetted_white(), np.ones((20, 20, 3), np.float32), "overexposed"),
    ],
)
def test_create_image_calibration_rejects_unusable_captures(fixed_now, white, card, fragment):
    with pytest.raises(CalibrationError) as err:
        create_image_calibration("d", white, card, now=fixed_now)
    assert fragment in str(err.value)
    assert find_positioning_violations(str(err.value)) == []


@pytest.mark.parametrize("hours", [0, -1, float("nan"), float("inf"), "abc"])
def test_validity_must_be_positive(fixed_now, hours):
    with pytest.raises(CalibrationError):
        create_image_calibration("d", vignetted_white(), None, now=fixed_now, valid_hours=hours)


def test_create_spectral_calibration(fixed_now):
    cal = create_spectral_calibration(
        "confirm-01",
        {"F1_415": 100, "F2_445": 120, "NIR": "bad"},
        {"F1_415": 5000, "F2_445": 90, "NIR": 300},
        now=fixed_now,
        valid_hours=12,
        gain=np.float64(64),
        integration_ms="junk",
        synthetic=True,
    )
    assert cal.calibration_id.startswith("cal-")
    assert cal.dark["F1_415"] == 100.0 and math.isnan(cal.dark["NIR"])
    assert cal.white["F2_445"] == 90.0
    assert cal.gain == 64.0 and cal.integration_ms is None
    assert cal.expires_at == fixed_now + timedelta(hours=12)
    assert cal.synthetic
    assert spectral_calibration_status(cal, fixed_now).status is CalibrationStatus.CALIBRATED


def test_create_spectral_calibration_rejects_no_white_signal(fixed_now):
    with pytest.raises(CalibrationError) as err:
        create_spectral_calibration("c", {"F1_415": 100}, {"F1_415": 100}, now=fixed_now)
    assert "white standard" in str(err.value)
    with pytest.raises(CalibrationError):
        create_spectral_calibration("c", {}, {}, now=fixed_now)
    with pytest.raises(CalibrationError):
        create_spectral_calibration("c", {"F1_415": 1}, {"F2_445": 900}, now=fixed_now)


# --------------------------------------------------------------------------------------------
# Reflectance
# --------------------------------------------------------------------------------------------


def test_reflectance_basic_and_white_dark():
    sample = {"F1_415": 1100.0, "F5_555": 600.0}
    dark = {"F1_415": 100.0, "F5_555": 100.0}
    white = {"F1_415": 2100.0, "F5_555": 1100.0}
    r = reflectance(sample, dark, white)
    assert r == {"F1_415": pytest.approx(0.5), "F5_555": pytest.approx(0.5)}
    r2 = reflectance(sample, dark, white, white_dark={"F1_415": 600.0, "F5_555": 100.0})
    assert r2["F1_415"] == pytest.approx(1000.0 / 1500.0)


def test_reflectance_is_nan_safe_and_clipped():
    sample = {"a": 50.0, "b": 9000.0, "c": 500.0, "d": float("nan"), "e": 400.0, "f": 300.0}
    dark = {"a": 100.0, "b": 0.0, "c": 0.0, "d": 0.0, "e": 0.0}
    white = {"a": 1000.0, "b": 1000.0, "c": 0.0, "d": 1000.0, "e": "x", "f": 1000.0}
    r = reflectance(sample, dark, white)
    assert r["a"] == 0.0  # below dark -> clipped at 0
    assert r["b"] == 1.5  # far above white -> clipped
    assert math.isnan(r["c"])  # no white signal
    assert math.isnan(r["d"])  # NaN sample
    assert math.isnan(r["e"])  # junk white value
    assert math.isnan(r["f"])  # missing dark
    assert reflectance({}, {}, {}) == {}
    assert math.isnan(reflectance({"a": 1.0}, {"a": 0.0}, {"a": 0.5}, {"a": 0.5})["a"])


# --------------------------------------------------------------------------------------------
# Serialisation
# --------------------------------------------------------------------------------------------


def test_image_calibration_round_trip(fixed_now):
    cal = create_image_calibration(
        "scout-01", vignetted_white(), fluorescence_card(), now=fixed_now, synthetic=True
    )
    blob = image_calibration_to_bytes(cal)
    assert isinstance(blob, bytes)
    back = image_calibration_from_bytes(blob)
    assert back.calibration_id == cal.calibration_id
    assert back.device_id == cal.device_id
    assert back.created_at == cal.created_at and back.expires_at == cal.expires_at
    assert back.af_scale == pytest.approx(cal.af_scale)
    assert back.synthetic is True
    np.testing.assert_array_equal(back.flat_field, cal.flat_field)
    np.testing.assert_array_equal(back.color_matrix, cal.color_matrix)
    # No pickled objects inside: every member loads with allow_pickle=False.
    with np.load(io.BytesIO(blob), allow_pickle=False) as npz:
        assert {"metadata", "color_matrix", "flat_field"} <= set(npz.files)


def test_image_calibration_round_trip_without_flat_and_naive_dates():
    cal = ImageCalibration(
        calibration_id="cal-x",
        device_id="d",
        created_at=datetime(2026, 1, 1),
        expires_at=datetime(2026, 1, 2),
        flat_field=None,
        color_matrix=np.eye(3),
        af_scale=1.5,
    )
    back = image_calibration_from_bytes(image_calibration_to_bytes(cal))
    assert back.flat_field is None
    assert back.created_at == datetime(2026, 1, 1, tzinfo=UTC)
    assert back.af_scale == 1.5 and back.synthetic is False


def _npz(**arrays) -> bytes:
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    return buf.getvalue()


def _meta(text: str) -> np.ndarray:
    return np.frombuffer(text.encode(), dtype=np.uint8)


@pytest.mark.parametrize(
    "blob",
    [
        b"not a calibration",
        b"",
        _npz(color_matrix=np.eye(3)),
        _npz(metadata=_meta('{"format": "other"}'), color_matrix=np.eye(3)),
        _npz(metadata=_meta("[1, 2]"), color_matrix=np.eye(3)),
        _npz(metadata=_meta("{not json"), color_matrix=np.eye(3)),
        _npz(
            metadata=_meta('{"format": "mukhaspectra.image_calibration", "format_version": 9}'),
            color_matrix=np.eye(3),
        ),
        _npz(
            metadata=_meta('{"format": "mukhaspectra.image_calibration", "format_version": "x"}'),
            color_matrix=np.eye(3),
        ),
        _npz(
            metadata=_meta('{"format": "mukhaspectra.image_calibration"}'), color_matrix=np.eye(3)
        ),
        _npz(metadata=_meta('{"format": "x"}'), color_matrix=np.eye(2)),
        _npz(
            metadata=_meta(
                '{"format": "mukhaspectra.image_calibration", "calibration_id": "c",'
                ' "device_id": "d", "created_at": "yesterday",'
                ' "expires_at": "2026-01-01T00:00:00"}'
            ),
            color_matrix=np.eye(3),
        ),
        _npz(metadata=np.array([object()], dtype=object), color_matrix=np.eye(3)),
    ],
)
def test_image_calibration_from_bad_bytes(blob):
    with pytest.raises(CalibrationError) as err:
        image_calibration_from_bytes(blob)
    assert find_positioning_violations(str(err.value)) == []


# --------------------------------------------------------------------------------------------
# Small helpers and positioning
# --------------------------------------------------------------------------------------------


def test_common_helpers():
    assert central_region(np.zeros((1, 1))).shape == (1, 1)
    assert central_region(np.zeros((8, 12))).shape == (4, 6)
    assert finite_median(np.array([np.nan, np.inf])) is None
    assert finite_median(np.array([1.0, np.nan, 3.0])) == 2.0
    assert luminance(np.ones((3, 3, 1), np.float32)).shape == (3, 3)
    assert luminance(np.ones((3, 3, 3), bool)).dtype == np.float32
    with pytest.raises(CalibrationError):
        luminance(np.ones(5))


def test_user_facing_calibration_text_is_positioning_clean(fixed_now):
    texts = [UNCALIBRATED_NOTE]
    texts.append(image_calibration_status(image_cal(fixed_now - timedelta(days=3)), fixed_now).note)
    texts.append(
        spectral_calibration_status(spectral_cal(fixed_now - timedelta(days=3)), fixed_now).note
    )
    texts.append(image_calibration_status(image_cal(fixed_now + timedelta(days=1)), fixed_now).note)
    for text in texts:
        assert text and find_positioning_violations(text) == []
