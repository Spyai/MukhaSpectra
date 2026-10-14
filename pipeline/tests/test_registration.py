"""Tests for mukha_pipeline.preprocess.registration (AF -> WL alignment). Data is SYNTHETIC."""

from __future__ import annotations

import time

import cv2
import numpy as np
import pytest
from registration_helpers import (
    add_fixed_overlay,
    corner_errors,
    make_pair,
    overlay_mask,
    resize_matrix,
)

from mukha_pipeline.positioning import find_positioning_violations
from mukha_pipeline.preprocess import registration as reg
from mukha_pipeline.preprocess.registration import register_af_to_wl

CORNER_TOL_PX = 1.5


@pytest.fixture(scope="module")
def cfg(pipeline_config) -> dict:
    return dict(pipeline_config.analysis.data["registration"])


@pytest.fixture(scope="module")
def pair():
    """1024x768 WL; AF moved by 2 deg rotation, scale 1.02, shift (12, -8) px."""
    return make_pair(np.random.default_rng(11))


@pytest.fixture(scope="module")
def pair_small_af():
    """AF captured at 640x480 while WL is 1024x768."""
    return make_pair(np.random.default_rng(12), af_size=(480, 640))


def _residual(result, truth) -> float:
    return float(np.abs(result.registered - truth)[result.valid].mean())


def _assert_positioning_clean(result) -> None:
    if result.note:
        assert find_positioning_violations(result.note) == []


# --------------------------------------------------------------------------------------------
# Successful alignment
# --------------------------------------------------------------------------------------------


def test_ecc_recovers_known_affine(pair, cfg):
    wl, af, true_warp, truth = pair
    t0 = time.perf_counter()
    result = register_af_to_wl(wl, af, cfg)
    elapsed = time.perf_counter() - t0
    assert result.method == "ecc_affine"
    assert result.passed is True
    assert result.confidence > 0.55
    assert result.note is None
    assert result.warp.shape == (3, 3) and result.warp.dtype == np.float64
    assert corner_errors(result.warp, true_warp, 1024, 768).max() < CORNER_TOL_PX
    assert elapsed < 1.5, f"registration took {elapsed:.2f} s"


def test_registered_image_aligns_with_ground_truth(pair, cfg):
    wl, af, _, truth = pair
    result = register_af_to_wl(wl, af, cfg)
    assert result.registered.shape == wl.shape and result.registered.dtype == np.float32
    assert result.valid.shape == wl.shape[:2] and result.valid.dtype == bool
    assert 0.9 < result.valid.mean() < 1.0  # the moved AF leaves an uncovered rim
    unaligned = float(np.abs(af - truth).mean())
    aligned = _residual(result, truth)
    assert aligned < 0.008  # at the level of the added sensor noise (sigma 0.01)
    assert aligned < 0.5 * unaligned
    # Where the anatomy has edges (teeth row, vessels) misalignment matters most.
    edges = cv2.Laplacian(cv2.GaussianBlur(truth[..., 1], (0, 0), 1.0), cv2.CV_32F)
    strong = (np.abs(edges) > np.percentile(np.abs(edges), 95)) & result.valid
    aligned_edges = float(np.abs(result.registered - truth)[strong].mean())
    unaligned_edges = float(np.abs(af - truth)[strong].mean())
    assert aligned_edges < 0.3 * unaligned_edges


def test_warp_convention_matches_opencv_forward_warp(pair, cfg):
    """warp maps AF pixel -> WL pixel, so warpPerspective(af, warp) gives `registered`."""
    wl, af, _, _ = pair
    result = register_af_to_wl(wl, af, cfg)
    expected = cv2.warpPerspective(af, result.warp, (wl.shape[1], wl.shape[0]))
    diff = np.abs(expected - result.registered)[result.valid]
    assert float(diff.mean()) < 1e-3


def test_valid_mask_is_eroded_at_the_warped_border(pair, cfg):
    wl, af, _, _ = pair
    result = register_af_to_wl(wl, af, cfg)
    covered = cv2.warpPerspective(
        np.ones(af.shape[:2], np.uint8),
        result.warp,
        (wl.shape[1], wl.shape[0]),
        flags=cv2.INTER_NEAREST,
    ).astype(bool)
    assert not np.any(result.valid & ~covered)
    near_uncovered = cv2.dilate((~covered).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    assert not np.any(result.valid & near_uncovered)  # 2 px band next to the zero fill
    near_covered = cv2.dilate(covered.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    assert np.all(result.registered[~near_covered] == 0)  # zero fill outside the AF footprint


def test_orb_fallback_path_recovers_known_affine(pair, cfg):
    wl, af, true_warp, truth = pair
    result = register_af_to_wl(wl, af, dict(cfg, method_order=["orb_homography"]))
    assert result.method == "orb_homography"
    assert result.passed and result.confidence > 0.3
    assert corner_errors(result.warp, true_warp, 1024, 768).max() < CORNER_TOL_PX
    assert _residual(result, truth) < 0.012


def test_ecc_homography_path(pair, cfg):
    wl, af, true_warp, _ = pair
    result = register_af_to_wl(wl, af, dict(cfg, method_order=["ecc_homography"]))
    assert result.method == "ecc_homography" and result.passed
    assert corner_errors(result.warp, true_warp, 1024, 768).max() < CORNER_TOL_PX


def test_smaller_af_image_scale_is_folded_into_warp(pair_small_af, cfg):
    wl, af, true_warp, truth = pair_small_af
    assert af.shape[:2] == (480, 640)
    for order in (["ecc_affine"], ["orb_homography"]):
        result = register_af_to_wl(wl, af, dict(cfg, method_order=order))
        assert result.passed and result.method == order[0]
        assert result.registered.shape == wl.shape
        assert corner_errors(result.warp, true_warp, 640, 480).max() < CORNER_TOL_PX
        assert _residual(result, truth) < 0.015


def test_larger_af_image(cfg):
    wl, af, true_warp, _ = make_pair(np.random.default_rng(13), af_size=(1152, 1536))
    result = register_af_to_wl(wl, af, cfg)
    assert result.passed and result.method == "ecc_affine"
    assert corner_errors(result.warp, true_warp, 1536, 1152).max() < CORNER_TOL_PX


def test_large_shift_within_limits_uses_phase_correlation_start(cfg):
    wl, af, true_warp, _ = make_pair(
        np.random.default_rng(14), shift=(90.0, -60.0), angle_deg=1.0, scale=1.0
    )
    result = register_af_to_wl(wl, af, dict(cfg, method_order=["ecc_affine"]))
    assert result.passed
    assert corner_errors(result.warp, true_warp, 1024, 768).max() < CORNER_TOL_PX


def test_ecc_failure_falls_back_to_orb_with_note(pair, cfg, monkeypatch):
    wl, af, true_warp, _ = pair

    def failing_ecc(*_args, **_kwargs):
        return reg._Attempt("ecc_affine", None, 0.0, "did not converge")

    monkeypatch.setattr(reg, "_ecc_attempt", failing_ecc)
    result = register_af_to_wl(wl, af, cfg)
    assert result.method == "orb_homography" and result.passed
    assert "ecc_affine: did not converge" in result.note
    assert corner_errors(result.warp, true_warp, 1024, 768).max() < CORNER_TOL_PX
    _assert_positioning_clean(result)


# --------------------------------------------------------------------------------------------
# Failure paths: identity, passed=False, never a crash
# --------------------------------------------------------------------------------------------


def _assert_identity(result, wl_shape) -> None:
    assert result.method == "identity"
    assert result.passed is False
    assert 0.0 <= result.confidence <= reg.IDENTITY_CONFIDENCE_CAP
    assert result.registered.shape == (wl_shape[0], wl_shape[1], 3)
    assert result.valid.shape == wl_shape[:2]
    assert result.note
    _assert_positioning_clean(result)


def test_random_noise_af_returns_identity(pair, cfg):
    wl, _, _, _ = pair
    noise = np.random.default_rng(5).random(wl.shape).astype(np.float32)
    result = register_af_to_wl(wl, noise, cfg)
    _assert_identity(result, wl.shape)
    np.testing.assert_allclose(result.warp, np.eye(3))
    np.testing.assert_array_equal(result.registered, noise)
    assert result.valid.all()
    assert "could not be aligned" in result.note and "Retake" in result.note


def test_identity_warp_includes_resize_for_smaller_af(pair, cfg):
    wl, _, _, _ = pair
    noise = np.random.default_rng(6).random((300, 400, 3)).astype(np.float32)
    result = register_af_to_wl(wl, noise, cfg)
    _assert_identity(result, wl.shape)
    np.testing.assert_allclose(result.warp, resize_matrix(400, 300, 1024, 768))


def test_out_of_range_zoom_is_rejected(cfg):
    wl, af, _, _ = make_pair(np.random.default_rng(15), angle_deg=0.0, scale=1.3, shift=(0, 0))
    result = register_af_to_wl(wl, af, cfg)
    _assert_identity(result, wl.shape)
    assert "scale change" in result.note


def test_out_of_range_shift_cannot_produce_a_false_lock(cfg):
    """A 25 % shift exceeds the limit; ECC may lock onto the repeating teeth row instead.

    ORB measures the true (out-of-range) shift, the cross-check vetoes ECC, and the site
    fails registration instead of being analysed with a wrong alignment.
    """
    wl, af, _, _ = make_pair(
        np.random.default_rng(16), shift=(260.0, 0.0), angle_deg=1.0, scale=1.0
    )
    result = register_af_to_wl(wl, af, cfg)
    _assert_identity(result, wl.shape)


def test_dark_af_returns_identity(pair, cfg):
    wl, _, _, _ = pair
    result = register_af_to_wl(wl, np.zeros_like(wl), cfg)
    _assert_identity(result, wl.shape)
    assert "did not converge" in result.note
    assert "OpenCV" not in result.note and "/" not in result.note.split("(")[1][:40]


def test_tiny_images_return_identity(cfg):
    wl = np.random.default_rng(1).random((10, 12, 3)).astype(np.float32)
    result = register_af_to_wl(wl, wl.copy(), cfg)
    _assert_identity(result, wl.shape)
    assert "too small" in result.note


def test_empty_images_return_identity(cfg):
    wl = np.zeros((0, 0, 3), np.float32)
    result = register_af_to_wl(wl, np.ones((10, 10, 3), np.float32), cfg)
    assert result.method == "identity" and not result.passed
    assert result.registered.shape == (0, 0, 3) and result.valid.shape == (0, 0)
    result = register_af_to_wl(np.ones((8, 8, 3), np.float32), np.zeros((0, 4)), cfg)
    assert result.registered.shape == (8, 8, 3) and not result.valid.any()
    bad_ndim = register_af_to_wl(np.ones((8, 8, 3), np.float32), np.zeros(5), cfg)
    assert bad_ndim.method == "identity"


def test_rotated_orientation_af_is_rejected_with_aspect_note(pair, cfg):
    wl, af, _, _ = pair
    portrait = np.ascontiguousarray(np.transpose(af, (1, 0, 2)))
    result = register_af_to_wl(wl, portrait, cfg)
    _assert_identity(result, wl.shape)
    assert "aspect ratio" in result.note


def test_featureless_af_fails_orb(pair, cfg):
    wl, _, _, _ = pair
    flat = np.full(wl.shape, 0.4, np.float32)
    result = register_af_to_wl(wl, flat, dict(cfg, method_order=["orb_homography"]))
    _assert_identity(result, wl.shape)
    assert "too few distinctive features" in result.note


def test_too_few_matches_fails_orb(pair, cfg):
    wl, af, _, _ = pair
    result = register_af_to_wl(
        wl, af, dict(cfg, method_order=["orb_homography"], orb_min_matches=100000)
    )
    _assert_identity(result, wl.shape)
    assert "reliable feature matches" in result.note


def test_low_confidence_is_not_accepted(pair, cfg):
    wl, af, _, _ = pair
    result = register_af_to_wl(wl, af, dict(cfg, accept_confidence_min=0.999))
    _assert_identity(result, wl.shape)
    assert "at or below" in result.note


# --------------------------------------------------------------------------------------------
# Robust input handling and configuration
# --------------------------------------------------------------------------------------------


def test_input_formats_do_not_crash(pair, cfg):
    wl, af, true_warp, _ = pair
    wl_u8 = (wl * 255).astype(np.uint8)
    af_gray = np.ascontiguousarray(af[..., 1])  # single-channel AF = green
    result = register_af_to_wl(wl_u8, af_gray, cfg)
    assert result.passed
    assert corner_errors(result.warp, true_warp, 1024, 768).max() < CORNER_TOL_PX
    rgba = np.concatenate([af, np.ones(af.shape[:2] + (1,), np.float32)], axis=2)
    assert register_af_to_wl(wl[..., :1], rgba, cfg).registered.shape == (768, 1024, 3)
    af_nan = af.copy()
    af_nan[300:320, 400:430] = np.nan
    result = register_af_to_wl(wl, af_nan, cfg)
    assert np.all(np.isfinite(result.registered))
    assert result.passed


@pytest.mark.parametrize("config", [None, {}, {"method_order": "ecc_affine"}])
def test_missing_config_uses_defaults(pair, config):
    wl, af, true_warp, _ = pair
    result = register_af_to_wl(wl, af, config)
    assert result.passed and result.method == "ecc_affine"
    assert corner_errors(result.warp, true_warp, 1024, 768).max() < CORNER_TOL_PX


def test_unknown_or_empty_method_order(pair, cfg):
    wl, af, _, _ = pair
    result = register_af_to_wl(wl, af, dict(cfg, method_order=["magic", "magic"]))
    _assert_identity(result, wl.shape)
    assert "no method configured" in result.note
    order = reg._method_order({"method_order": ["orb_homography", "x", "orb_homography"]})
    assert order == ["orb_homography"]


def test_cross_check_can_be_disabled(pair, cfg):
    wl, af, _, _ = pair
    result = register_af_to_wl(wl, af, dict(cfg, cross_check=False, ecc_gauss_filt_size=4))
    assert result.passed and result.method == "ecc_affine"


# --------------------------------------------------------------------------------------------
# Internal helpers
# --------------------------------------------------------------------------------------------


def test_resize_matrix_matches_opencv_pixel_centres():
    img = np.zeros((60, 80), np.float32)
    img[30, 40] = 1.0
    img = cv2.GaussianBlur(img, (0, 0), 3)
    small = cv2.resize(img, (40, 30), interpolation=cv2.INTER_AREA)
    yy, xx = np.mgrid[0:30, 0:40]
    cx = float((small * xx).sum() / small.sum())
    cy = float((small * yy).sum() / small.sum())
    u, v, _ = reg._apply(reg._resize_matrix(80, 60, 40, 30), 40.0, 30.0)
    assert abs(u - cx) < 0.05 and abs(v - cy) < 0.05


def _shift(dx: float, dy: float) -> np.ndarray:
    return np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]], dtype=float)


@pytest.mark.parametrize(
    "warp, fragment",
    [
        (np.full((3, 3), np.nan), "non-finite"),
        (np.array([[1, 0, 0], [0, 1, 0], [-0.01, 0, 1]], float), "perspective fold"),
        (_shift(200, 0), "shift"),
        (np.diag([1.3, 1.3, 1.0]), "scale change"),
        (_shift(999, 0) @ np.diag([-1.0, 1.0, 1.0]), "mirrored"),
    ],
)
def test_sanity_rejections(cfg, warp, fragment):
    params = reg._params(cfg)
    assert fragment in reg._sanity_problem(warp, 1000, 800, params)


def test_sanity_rotation_limit(cfg):
    a = np.radians(20)
    rot = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    centred = _shift(500, 400) @ rot @ _shift(-500, -400)
    params = reg._params(cfg)
    assert "rotation" in reg._sanity_problem(centred, 1000, 800, params)
    assert reg._sanity_problem(np.eye(3), 1000, 800, params) is None


def test_cross_check_conflict_helper(cfg):
    params = reg._params(cfg)
    winner = reg._Attempt("ecc_affine", np.eye(3), 0.9, None, residual=np.eye(3))
    close = reg._Attempt("orb_homography", np.eye(3), 0.8, None, residual=_shift(3, 0))
    far = reg._Attempt("orb_homography", np.eye(3), 0.8, "shift", residual=_shift(200, 0))
    weak = reg._Attempt("orb_homography", np.eye(3), 0.1, None, residual=_shift(200, 0))
    assert reg._cross_check_conflict(winner, [winner, close], 1000, 800, params) is None
    assert reg._cross_check_conflict(winner, [winner, weak], 1000, 800, params) is None
    assert "disagrees" in reg._cross_check_conflict(winner, [winner, far], 1000, 800, params)
    off = dict(params, cross_check=False)
    assert reg._cross_check_conflict(winner, [winner, far], 1000, 800, off) is None


def test_refine_prefers_homography_when_perspective_is_real(rng):
    true_h = np.array([[1.01, 0.02, 5.0], [-0.01, 0.99, -3.0], [2e-4, -1e-4, 1.0]])
    src = rng.uniform(0, 500, size=(200, 2))
    pts = np.hstack([src, np.ones((200, 1))]) @ true_h.T
    dst = pts[:, :2] / pts[:, 2:3] + rng.normal(0, 0.2, size=(200, 2))
    refined = reg._refine_on_inliers(src, dst, true_h * 2.0)
    assert abs(refined[2, 0]) > 1e-4  # perspective kept
    errs = reg._reprojection_errors(refined, src, dst)
    assert np.median(errs) < 0.5


def test_refine_falls_back_with_few_points():
    src = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], float)
    h = np.eye(3) * 2
    out = reg._refine_on_inliers(src, src, h)
    np.testing.assert_allclose(out, np.eye(3))
    collinear = np.array([[i, i] for i in range(10)], float)
    np.testing.assert_allclose(reg._refine_on_inliers(collinear, collinear, np.eye(3)), np.eye(3))
    assert reg._fit_affine(src[:2], src[:2]) is None


def test_correlation_helper_edge_cases():
    flat = np.zeros((20, 20), np.float32)
    assert reg._correlation(flat, flat) == 0.0
    tiny = np.zeros((3, 3), np.float32)
    assert reg._correlation(tiny, tiny, np.eye(3)) == 0.0
    assert np.array_equal(reg._initial_translation(tiny, tiny), np.eye(3))


def test_short_error_text():
    err = RuntimeError(
        "OpenCV(5.0.0) /src/ecc.cpp:582: error: (-7:Iterations do not converge) NaN "
        "encountered. in function 'findTransformECC'\n"
    )
    assert reg._short(err) == "Iterations do not converge: NaN encountered."
    assert reg._short(RuntimeError("plain")) == "plain"
    assert reg._short(RuntimeError("")) == "RuntimeError"


# --------------------------------------------------------------------------------------------
# Camera-fixed overlays (watermark, timestamp, housing)
# --------------------------------------------------------------------------------------------


def test_exclude_mask_removes_camera_fixed_overlay_bias(pair, cfg):
    """A label at the same pixel position in both photos votes for identity; excluding it
    restores accurate alignment for both methods."""
    wl, af, true_warp, _ = pair
    wl_s, af_s = add_fixed_overlay(wl), add_fixed_overlay(af)
    mask = overlay_mask(wl.shape)
    explicit_only = dict(cfg, auto_exclude_identical=False)
    for order in (["ecc_affine", "orb_homography"], ["orb_homography"]):
        conf = dict(explicit_only, method_order=order)
        result = register_af_to_wl(wl_s, af_s, conf, exclude=mask)
        assert result.passed and result.method == order[0]
        assert corner_errors(result.warp, true_warp, 1024, 768).max() < CORNER_TOL_PX


def test_identical_overlay_is_detected_automatically(pair, cfg):
    wl, af, true_warp, _ = pair
    wl_s, af_s = add_fixed_overlay(wl), add_fixed_overlay(af)
    for order in (["ecc_affine", "orb_homography"], ["orb_homography"]):
        result = register_af_to_wl(wl_s, af_s, dict(cfg, method_order=order))
        assert result.passed and result.method == order[0]
        assert corner_errors(result.warp, true_warp, 1024, 768).max() < CORNER_TOL_PX
        assert "identical" in result.note and "Ignored" in result.note
        _assert_positioning_clean(result)
    # Detection helper: covers the label, nothing on clean tissue.
    params = reg._params(cfg)
    found = reg._identical_overlay_mask(wl_s, af_s, params)
    assert found is not None and found[overlay_mask(wl.shape)].mean() > 0.95
    assert reg._identical_overlay_mask(wl, af, params) is None
    specks = af.copy()
    specks[100, 100] = wl[100, 100]  # a single coincidental pixel is not a label
    assert reg._identical_overlay_mask(wl, specks, params) is None


def test_duplicate_upload_fails_with_hint(pair, cfg):
    """The WL photo uploaded again as the AF photo must never pass as aligned."""
    wl, _, _, _ = pair
    result = register_af_to_wl(wl, wl.copy(), cfg)
    _assert_identity(result, wl.shape)
    assert "identical over most of the frame" in result.note


def test_exclude_mask_variants(pair, cfg):
    wl, af, true_warp, _ = pair
    small_rgb_mask = np.repeat(overlay_mask((384, 512))[..., None], 3, axis=2)
    result = register_af_to_wl(wl, af, cfg, exclude=small_rgb_mask)
    assert result.passed
    assert corner_errors(result.warp, true_warp, 1024, 768).max() < CORNER_TOL_PX
    for useless in (np.zeros(wl.shape[:2], bool), np.zeros((0,)), np.zeros(5)):
        assert register_af_to_wl(wl, af, cfg, exclude=useless).passed
    everything = np.ones(wl.shape[:2], bool)
    result = register_af_to_wl(wl, af, cfg, exclude=everything)
    _assert_identity(result, wl.shape)
    assert "too little of the image" in result.note
