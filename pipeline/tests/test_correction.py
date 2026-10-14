"""Tests for mukha_pipeline.preprocess.correction."""

from __future__ import annotations

import numpy as np
import pytest

from mukha_pipeline.preprocess.correction import (
    apply_color_matrix,
    apply_flat_field,
    exposure_factor,
    subtract_dark,
)
from mukha_pipeline.schema.models import CaptureKind, CaptureMetadata, SourceAdapter
from mukha_pipeline.sites import OralSite


def meta(exposure: float | None = None, iso: float | None = None) -> CaptureMetadata:
    return CaptureMetadata(
        capture_id="c1",
        kind=CaptureKind.AUTOFLUORESCENCE,
        oral_site=OralSite.BUCCAL_MUCOSA_LEFT,
        source_adapter=SourceAdapter.UPLOAD,
        exposure_time_ms=exposure,
        iso=iso,
    )


# --------------------------------------------------------------------------------------------
# Dark subtraction
# --------------------------------------------------------------------------------------------


def test_subtract_dark_same_size_clips_at_zero():
    img = np.full((4, 5, 3), 0.30, np.float32)
    dark = np.full((4, 5, 3), 0.05, np.float32)
    dark[0, 0] = 0.9
    out = subtract_dark(img, dark)
    assert out.dtype == np.float32 and out.shape == img.shape
    np.testing.assert_allclose(out[1, 1], 0.25, atol=1e-6)
    assert np.all(out[0, 0] == 0.0)
    assert img[1, 1, 0] == pytest.approx(0.30)  # input untouched


def test_subtract_dark_resizes_and_broadcasts():
    img = np.full((40, 60, 3), 0.5, np.float32)
    dark_small = np.full((10, 15, 3), 0.1, np.float32)
    np.testing.assert_allclose(subtract_dark(img, dark_small), 0.4, atol=1e-6)
    dark_big = np.full((80, 120, 3), 0.1, np.float32)
    np.testing.assert_allclose(subtract_dark(img, dark_big), 0.4, atol=1e-6)
    dark_gray = np.full((40, 60), 0.2, np.float32)
    np.testing.assert_allclose(subtract_dark(img, dark_gray), 0.3, atol=1e-6)
    gray_img = np.full((40, 60), 0.5, np.float32)
    np.testing.assert_allclose(subtract_dark(gray_img, dark_small), 0.4, atol=1e-6)
    dark_single = np.full((40, 60, 1), 0.1, np.float32)
    np.testing.assert_allclose(subtract_dark(img, dark_single), 0.4, atol=1e-6)
    dark_rgba = np.full((40, 60, 4), 0.1, np.float32)
    np.testing.assert_allclose(subtract_dark(img, dark_rgba), 0.4, atol=1e-6)
    dark_single_small = np.full((20, 30, 1), 0.1, np.float32)
    np.testing.assert_allclose(subtract_dark(img, dark_single_small), 0.4, atol=1e-6)


def test_subtract_dark_bad_inputs():
    img = np.full((6, 6, 3), 0.5, np.float32)
    img[0, 0, 0] = np.nan
    dark = np.full((6, 6, 3), np.nan, np.float32)
    out = subtract_dark(img, dark)
    assert np.all(np.isfinite(out))
    assert out[0, 0, 0] == 0.0 and out[1, 1, 1] == pytest.approx(0.5)
    np.testing.assert_allclose(subtract_dark(img, None)[1:, 1:], 0.5)
    np.testing.assert_allclose(subtract_dark(img, np.zeros((0, 0, 3)))[1:, 1:], 0.5)
    assert subtract_dark(np.zeros((0, 0, 3)), dark).shape == (0, 0, 3)
    u8 = np.full((3, 3, 3), 255, np.uint8)
    np.testing.assert_allclose(subtract_dark(u8, np.full((3, 3, 3), 51, np.uint8)), 0.8)


# --------------------------------------------------------------------------------------------
# Flat field
# --------------------------------------------------------------------------------------------


def test_apply_flat_field_divides_and_resizes():
    img = np.full((20, 40, 3), 0.4, np.float32)
    flat = np.ones((10, 20), np.float32)
    flat[:, :10] = 0.5
    out = apply_flat_field(img, flat)
    assert out.shape == img.shape and out.dtype == np.float32
    assert out[10, 2, 0] == pytest.approx(0.8, abs=1e-5)
    assert out[10, 37, 1] == pytest.approx(0.4, abs=1e-5)
    assert out.max() <= 1.0
    bright = apply_flat_field(np.full((4, 4, 3), 0.9, np.float32), np.full((4, 4), 0.5))
    assert np.all(bright == 1.0)


def test_apply_flat_field_bad_flats_are_neutral():
    img = np.full((6, 8, 3), 0.4, np.float32)
    flat = np.array([[0.0, np.nan, -1.0, np.inf]] * 3, np.float32)
    out = apply_flat_field(img, flat)
    assert np.all(np.isfinite(out))
    np.testing.assert_allclose(out[:, :, 0][:, :4], 0.4, atol=1e-5)
    np.testing.assert_allclose(apply_flat_field(img, None), 0.4)
    np.testing.assert_allclose(apply_flat_field(img, np.zeros((0, 0))), 0.4)
    rgb_flat = np.full((6, 8, 3), 2.0, np.float32)
    np.testing.assert_allclose(apply_flat_field(img, rgb_flat), 0.2)
    gray_img = np.full((6, 8), 0.4, np.float32)
    np.testing.assert_allclose(apply_flat_field(gray_img, np.full((3, 4), 2.0)), 0.2)
    assert apply_flat_field(np.zeros((0, 0, 3)), flat).shape == (0, 0, 3)


# --------------------------------------------------------------------------------------------
# Colour matrix
# --------------------------------------------------------------------------------------------


def test_apply_color_matrix():
    img = np.zeros((2, 3, 3), np.float32)
    img[..., 0], img[..., 1], img[..., 2] = 0.2, 0.4, 0.6
    m = np.diag([2.0, 1.0, 0.5])
    out = apply_color_matrix(img, m)
    np.testing.assert_allclose(out[0, 0], [0.4, 0.4, 0.3], atol=1e-6)
    swap = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]], float)
    np.testing.assert_allclose(apply_color_matrix(img, swap)[1, 2], [0.4, 0.2, 0.6], atol=1e-6)
    assert apply_color_matrix(img, np.eye(3) * 10).max() == 1.0
    assert apply_color_matrix(img, -np.eye(3)).min() == 0.0


@pytest.mark.parametrize("matrix", [None, np.eye(2), np.full((3, 3), np.nan), np.ones(9)])
def test_apply_color_matrix_invalid_passes_through(matrix):
    img = np.full((2, 2, 3), 0.3, np.float32)
    np.testing.assert_allclose(apply_color_matrix(img, matrix), 0.3)


def test_apply_color_matrix_non_rgb_passes_through():
    gray = np.full((2, 2), 0.3, np.float32)
    np.testing.assert_allclose(apply_color_matrix(gray, np.eye(3) * 2), 0.3)


# --------------------------------------------------------------------------------------------
# Exposure factor
# --------------------------------------------------------------------------------------------


def test_exposure_factor_time_and_iso():
    ref = meta(100.0, 400.0)
    assert exposure_factor(meta(50.0, 400.0), ref) == pytest.approx(2.0)
    assert exposure_factor(meta(100.0, 800.0), ref) == pytest.approx(0.5)
    assert exposure_factor(meta(200.0, 200.0), ref) == pytest.approx(1.0)
    assert exposure_factor(meta(25.0, 100.0), ref) == pytest.approx(16.0)


def test_exposure_factor_missing_values():
    assert exposure_factor(meta(50.0, 400.0), None) == 1.0
    # Missing ISO on either side: exposure-time ratio only.
    assert exposure_factor(meta(50.0, None), meta(100.0, 400.0)) == pytest.approx(2.0)
    assert exposure_factor(meta(50.0, 400.0), meta(100.0, None)) == pytest.approx(2.0)
    # Missing exposure time on either side: unknown -> 1.0 even if ISO differs.
    assert exposure_factor(meta(None, 100.0), meta(100.0, 400.0)) == 1.0
    assert exposure_factor(meta(50.0, 100.0), meta(None, 400.0)) == 1.0
    # Zero / negative / NaN values are unknown, never divided by.
    assert exposure_factor(meta(0.0, 100.0), meta(100.0, 100.0)) == 1.0
    assert exposure_factor(meta(-5.0, 100.0), meta(100.0, 100.0)) == 1.0
    assert exposure_factor(meta(float("nan"), 100.0), meta(100.0, 100.0)) == 1.0
    assert exposure_factor(meta(50.0, 0.0), meta(100.0, 100.0)) == pytest.approx(2.0)
    assert exposure_factor(meta(1e-300, None), meta(1e300, None)) == 1.0  # overflow -> unknown


def test_exposure_factor_unvalidated_metadata():
    """Metadata built without validation (e.g. model_construct) may carry junk; never raise."""
    junk = CaptureMetadata.model_construct(exposure_time_ms="fast", iso=None)
    assert exposure_factor(junk, meta(100.0, 100.0)) == 1.0
