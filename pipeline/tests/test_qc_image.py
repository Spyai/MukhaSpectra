"""Image QC: white light, autofluorescence, dark frame, registration, segmentation (SYNTHETIC)."""

import copy

import cv2
import numpy as np
import pytest
import qc_helpers as qh

from mukha_pipeline.qc import (
    combine_qc,
    missing_dark_frame_issue,
    qc_autofluorescence,
    qc_dark_frame,
    qc_registration,
    qc_segmentation,
    qc_white_light,
)
from mukha_pipeline.qc.imageops import prepare_image, resize_longest_side
from mukha_pipeline.schema.models import QCSeverity


def codes(result):
    return [i.code for i in result.issues]


@pytest.fixture(scope="module")
def wl():
    return qh.make_wl_image(np.random.default_rng(11))


@pytest.fixture(scope="module")
def af():
    return qh.make_af_image(np.random.default_rng(12))


# --------------------------------------------------------------------------------------------
# White light
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("texture", [0.03, 0.06, 0.10])
def test_plausible_wl_passes_cleanly(qc_cfg, texture):
    img = qh.make_wl_image(np.random.default_rng(1), texture=texture)
    r = qc_white_light(img, qc_cfg)
    assert r.passed and r.score == 1.0, codes(r)
    for key in (
        "wl.blur_laplacian_var",
        "wl.mean_brightness",
        "wl.clip_low_fraction",
        "wl.clip_high_fraction",
        "wl.glare_fraction",
    ):
        assert key in r.metrics
    assert r.metrics["wl.blur_laplacian_var"] > 3 * qc_cfg["white_light"]["blur_laplacian_var_min"]


def test_very_smooth_wl_without_teeth_or_noise_still_passes(qc_cfg):
    img = qh.make_wl_image(
        np.random.default_rng(2), texture=0.03, teeth=False, speckles=0, sensor_noise=0.0
    )
    assert qc_white_light(img, qc_cfg).passed


def test_mild_softness_passes_but_strong_blur_fails(qc_cfg, wl):
    assert qc_white_light(qh.blur(wl, 1.0), qc_cfg).passed
    for sigma in (4.0, 6.0):
        r = qc_white_light(qh.blur(wl, sigma), qc_cfg)
        assert codes(r) == ["blur"], sigma
        assert r.issues[0].severity == QCSeverity.ERROR
        assert r.issues[0].metric == "wl.blur_laplacian_var"
        assert r.issues[0].value < r.issues[0].threshold


def test_blur_metric_is_resolution_independent(qc_cfg, wl):
    big = cv2.resize(wl, (4096, 3072), interpolation=cv2.INTER_CUBIC).clip(0, 1)
    small = qc_white_light(wl, qc_cfg).metrics["wl.blur_laplacian_var"]
    large = qc_white_light(big, qc_cfg).metrics["wl.blur_laplacian_var"]
    assert large == pytest.approx(small, rel=0.35)
    # a 12 MP photo blurred by 16 px is the same defocus as 4 px at the 1024 working size
    assert codes(qc_white_light(qh.blur(big, 16.0), qc_cfg)) == ["blur"]


def test_too_dark_reports_light_not_blur(qc_cfg, wl):
    r = qc_white_light(qh.scale(wl, 0.15), qc_cfg)
    assert codes(r) == ["too_dark"]
    assert not r.passed


def test_too_bright_reports_root_cause_only(qc_cfg, wl):
    r = qc_white_light(qh.scale(wl, 1.7), qc_cfg)
    assert codes(r) == ["too_bright"]
    assert r.metrics["wl.clip_high_fraction"] > 0.5  # recorded even though not prompted


def test_black_areas_are_a_warning(qc_cfg, wl):
    img = wl.copy()
    img[: int(0.35 * img.shape[0])] = 0.0  # dark oropharynx / shadow across a third of the view
    r = qc_white_light(img, qc_cfg)
    assert "clipping_dark" in codes(r)
    assert r.passed
    assert next(i for i in r.issues if i.code == "clipping_dark").severity == QCSeverity.WARNING


def test_saturated_red_channel_is_clipping_bright(qc_cfg, wl):
    img = wl.copy()
    img[200:500, 200:700, 0] = 1.0  # red channel blown out over mucosa, not white
    r = qc_white_light(img, qc_cfg)
    assert "clipping_bright" in codes(r) and "glare" not in codes(r)
    assert not r.passed


def test_glare_levels(qc_cfg, wl):
    rng = np.random.default_rng(3)
    minor = qc_white_light(qh.add_glare(wl, 0.025, rng), qc_cfg)
    assert "glare_minor" in codes(minor) and minor.passed
    heavy = qc_white_light(qh.add_glare(wl, 0.08, rng), qc_cfg)
    assert "glare" in codes(heavy) and "glare_minor" not in codes(heavy)
    assert not heavy.passed


def test_glare_does_not_make_a_blurred_photo_look_sharp(qc_cfg, wl):
    blurred = qh.add_glare(qh.blur(wl, 5.0), 0.01, np.random.default_rng(4))
    assert "blur" in codes(qc_white_light(blurred, qc_cfg))


def test_bright_teeth_are_not_glare(qc_cfg):
    img = qh.make_wl_image(np.random.default_rng(5), exposure=1.05)
    assert qc_white_light(img, qc_cfg).metrics["wl.glare_fraction"] < 0.01


# --------------------------------------------------------------------------------------------
# Input robustness (shared by all image checks)
# --------------------------------------------------------------------------------------------


def test_uint8_and_uint16_inputs_match_float(qc_cfg, wl):
    ref = qc_white_light(wl, qc_cfg).metrics["wl.mean_brightness"]
    as8 = qc_white_light((wl * 255).round().astype(np.uint8), qc_cfg)
    as16 = qc_white_light((wl * 65535).round().astype(np.uint16), qc_cfg)
    assert as8.passed and as16.passed
    assert as8.metrics["wl.mean_brightness"] == pytest.approx(ref, abs=0.6)
    assert as16.metrics["wl.mean_brightness"] == pytest.approx(ref, abs=0.1)


def test_rgba_is_accepted(qc_cfg, wl):
    rgba = np.concatenate([wl, np.ones((*wl.shape[:2], 1), np.float32)], axis=2)
    assert qc_white_light(rgba, qc_cfg).passed


@pytest.mark.parametrize("shape", ["2d", "1ch"])
def test_grayscale_input_is_converted_and_flagged(qc_cfg, wl, shape):
    gray = wl.mean(axis=2)
    img = gray if shape == "2d" else gray[..., None]
    r = qc_white_light(img, qc_cfg)
    assert "image_grayscale" in codes(r) and not r.passed
    assert "wl.blur_laplacian_var" in r.metrics  # still measured after conversion
    assert "image_grayscale" in codes(qc_autofluorescence(img, qc_cfg))


@pytest.mark.parametrize(
    "bad",
    [
        None,
        np.zeros((0, 0, 3), np.float32),
        np.zeros(10, np.float32),
        np.zeros((8, 8, 5), np.float32),
        np.array([["a", "b"], ["c", "d"]]),
        [[1, 2], [3]],
        np.zeros((4, 4, 3), np.complex64),
    ],
    ids=["none", "empty", "1d", "5ch", "strings", "ragged", "complex"],
)
def test_unusable_input_is_an_issue_not_a_crash(qc_cfg, bad):
    for fn in (qc_white_light, qc_autofluorescence, qc_dark_frame):
        r = fn(bad, qc_cfg)
        assert codes(r) == ["image_invalid"] and not r.passed


def test_mostly_nan_image_is_invalid(qc_cfg, wl):
    img = wl.copy()
    img[:100] = np.nan
    r = qc_white_light(img, qc_cfg)
    assert codes(r) == ["image_invalid"]
    assert r.issues[0].metric == "wl.nonfinite_fraction"


def test_few_nan_pixels_are_tolerated(qc_cfg, wl):
    img = wl.copy()
    img[5, 5] = np.nan
    img[6, 6] = np.inf
    r = qc_white_light(img, qc_cfg)
    assert r.passed
    assert 0 < r.metrics["wl.nonfinite_fraction"] < 0.01


def test_tiny_and_thin_images_do_not_crash(qc_cfg):
    tiny = qc_white_light(np.full((1, 1, 3), 0.5, np.float32), qc_cfg)
    assert "image_too_small" in codes(tiny) and not tiny.passed
    thin = qc_autofluorescence(np.full((3, 5000, 3), 0.3, np.float32), qc_cfg)
    assert "image_too_small" in codes(thin)
    assert qc_dark_frame(np.zeros((2, 2), np.float32), qc_cfg).metrics["dark.mean"] == 0.0


def test_odd_sizes_and_out_of_range_values(qc_cfg, wl):
    odd = wl[:517, :731]
    assert qc_white_light(odd, qc_cfg).passed
    loud = wl * 3.0 - 1.0  # values outside [0, 1] are clipped, then judged
    assert not qc_white_light(loud, qc_cfg).passed


def test_prepare_image_resizes_to_metric_scale(qc_cfg, wl):
    prep = prepare_image(wl, qc_cfg, "wl")
    assert max(prep.rgb.shape[:2]) == qc_cfg["metric_max_side_px"]
    assert prep.rgb.dtype == np.float32 and prep.issues == []
    same, scale = resize_longest_side(prep.rgb, max(prep.rgb.shape[:2]))
    assert scale == 1.0 and same is not prep.rgb
    up, s = resize_longest_side(np.full((10, 20, 1), 0.5, np.float32), 40)
    assert up.shape == (20, 40, 1) and s == 2.0


def test_bool_and_signed_inputs(qc_cfg):
    assert "image_invalid" not in codes(qc_dark_frame(np.zeros((80, 80), bool), qc_cfg))
    signed = np.full((80, 80, 3), -5, np.int16)
    assert qc_dark_frame(signed, qc_cfg).metrics["dark.mean"] == 0.0


def test_defaults_used_when_optional_keys_missing(qc_cfg, wl, af):
    cfg = copy.deepcopy(qc_cfg)
    for key in ("metric_max_side_px", "min_image_side_px"):
        cfg.pop(key)
    for key in (
        "signal_sum_min",
        "green_fraction_min",
        "white_light_blue_fraction_min",
        "white_light_red_excess_min",
    ):
        cfg["autofluorescence"].pop(key)
    assert qc_white_light(wl, cfg).passed
    assert qc_autofluorescence(af, cfg).passed
    assert codes(qc_autofluorescence(wl, cfg)) == ["af_white_light"]


# --------------------------------------------------------------------------------------------
# Autofluorescence
# --------------------------------------------------------------------------------------------


def test_plausible_af_passes_cleanly(qc_cfg, af):
    r = qc_autofluorescence(af, qc_cfg)
    assert r.passed and r.score == 1.0, codes(r)
    m = r.metrics
    assert m["af.green_fraction"] > 0.5 and m["af.blue_fraction"] < 0.2
    for key in ("af.green_p90", "af.blur_laplacian_var", "af.clip_high_fraction"):
        assert key in m


def test_smooth_af_without_teeth_passes(qc_cfg):
    img = qh.make_af_image(np.random.default_rng(6), teeth=False, sensor_noise=0.0, texture=0.05)
    assert qc_autofluorescence(img, qc_cfg).passed


def test_af_strong_blur_fails(qc_cfg, af):
    assert qc_autofluorescence(qh.blur(af, 1.0), qc_cfg).passed
    r = qc_autofluorescence(qh.blur(af, 4.0), qc_cfg)
    assert codes(r) == ["af_blur"] and not r.passed


def test_violet_light_off(qc_cfg):
    img = qh.make_af_image(np.random.default_rng(7), green=0.03, red=0.01, blue=0.005)
    r = qc_autofluorescence(img, qc_cfg)
    assert codes(r) == ["af_signal_low"]  # no blur prompt on top of the real cause
    assert r.issues[0].metric == "af.green_p90"


def test_af_completely_black(qc_cfg):
    r = qc_autofluorescence(np.zeros((300, 400, 3), np.float32), qc_cfg)
    assert codes(r) == ["af_signal_low"]
    assert "af.blue_fraction" not in r.metrics  # undefined without signal, not NaN


def test_missing_emission_filter(qc_cfg, af):
    r = qc_autofluorescence(qh.remove_emission_filter(af), qc_cfg)
    assert codes(r) == ["af_filter_leak"]
    assert r.issues[0].value > qc_cfg["autofluorescence"]["blue_fraction_max"]


def test_af_overexposed_by_clipping_and_by_mean(qc_cfg, af):
    r = qc_autofluorescence(qh.scale(af, 3.0), qc_cfg)
    assert codes(r) == ["af_overexposed"]
    assert r.issues[0].metric == "af.clip_high_fraction"
    flat = np.full((300, 400, 3), [0.85, 0.97, 0.85], np.float32)  # bright, nothing clipped
    flat += np.random.default_rng(8).normal(0, 0.01, flat.shape).astype(np.float32)
    r2 = qc_autofluorescence(np.clip(flat, 0, 0.985), qc_cfg)
    assert "af_overexposed" in codes(r2)
    assert next(i for i in r2.issues if i.code == "af_overexposed").metric == "af.mean_brightness"


def test_white_light_photo_in_af_slot_is_blocked(qc_cfg, wl):
    r = qc_autofluorescence(wl, qc_cfg)
    assert codes(r) == ["af_white_light"] and not r.passed


def test_auto_white_balance_af_is_a_warning(qc_cfg):
    af = qh.make_af_image(np.random.default_rng(9), teeth=False)
    r = qc_autofluorescence(qh.grey_world(af), qc_cfg)
    assert codes(r) == ["af_not_green"] and r.passed
    assert r.issues[0].metric == "af.green_fraction"


def test_white_light_through_filter_is_a_warning(qc_cfg, wl):
    r = qc_autofluorescence(qh.through_emission_filter(wl), qc_cfg)
    assert codes(r) == ["af_not_green"] and r.passed
    assert r.issues[0].metric == "af.red_minus_green_fraction"


def test_heavy_porphyrin_is_not_mistaken_for_white_light(qc_cfg, af):
    r = qc_autofluorescence(qh.add_porphyrin(af, 0.5), qc_cfg)
    assert codes(r) == ["af_not_green"] and r.passed


def test_some_porphyrin_spots_are_fine(qc_cfg, af):
    img = af.copy()
    img[300:400, 300:500, 0] = 0.6  # a plaque patch fluorescing red at ~635 nm
    assert qc_autofluorescence(img, qc_cfg).passed


# --------------------------------------------------------------------------------------------
# Dark frame
# --------------------------------------------------------------------------------------------


def test_dark_frame_levels(qc_cfg, rng):
    assert qc_dark_frame(qh.make_dark_frame(rng), qc_cfg).passed
    minor = qc_dark_frame(qh.make_dark_frame(rng, level=0.04), qc_cfg)
    assert codes(minor) == ["ambient_light_minor"] and minor.passed
    leak = qc_dark_frame(qh.make_dark_frame(rng, level=0.12), qc_cfg)
    assert codes(leak) == ["ambient_light"] and not leak.passed
    assert leak.metrics["dark.mean"] == pytest.approx(0.12, abs=0.005)


def test_grayscale_dark_frame_is_fine(qc_cfg, rng):
    gray = qh.make_dark_frame(rng)[..., 0]
    assert qc_dark_frame(gray, qc_cfg).passed


# --------------------------------------------------------------------------------------------
# Registration and segmentation
# --------------------------------------------------------------------------------------------


def test_registration_thresholds(qc_cfg):
    c = qc_cfg["registration"]
    good = qc_registration(qh.make_registration(c["confidence_warning"] + 0.1), qc_cfg)
    assert good.passed and good.issues == []
    assert good.metrics["registration.confidence"] == pytest.approx(c["confidence_warning"] + 0.1)
    marginal = qc_registration(qh.make_registration(c["confidence_min"] + 0.05), qc_cfg)
    assert codes(marginal) == ["alignment_marginal"] and marginal.passed
    bad = qc_registration(qh.make_registration(c["confidence_min"] - 0.05), qc_cfg)
    assert codes(bad) == ["misaligned"] and not bad.passed


def test_registration_rejected_by_aligner_fails_whatever_confidence(qc_cfg):
    r = qc_registration(qh.make_registration(0.95, passed=False), qc_cfg)
    assert codes(r) == ["misaligned"]
    assert r.issues[0].metric == "registration.passed"


def test_registration_junk_confidence_fails(qc_cfg):
    reg = qh.make_registration(0.9)
    reg.confidence = None  # type: ignore[assignment]
    assert codes(qc_registration(reg, qc_cfg)) == ["misaligned"]


def test_registration_nan_confidence_fails(qc_cfg):
    r = qc_registration(qh.make_registration(float("nan")), qc_cfg)
    assert codes(r) == ["misaligned"]
    assert r.metrics["registration.confidence"] == 0.0


def test_segmentation_mucosa_fraction(qc_cfg):
    floor = qc_cfg["segmentation"]["mucosa_fraction_min"]
    ok = qc_segmentation(qh.make_masks(floor + 0.2), qc_cfg)
    assert ok.passed and ok.metrics["segmentation.mucosa_fraction"] == pytest.approx(floor + 0.2)
    low = qc_segmentation(qh.make_masks(floor / 2), qc_cfg)
    assert codes(low) == ["insufficient_mucosa"] and not low.passed


def test_segmentation_handles_empty_and_odd_masks(qc_cfg):
    masks = qh.make_masks(0.5)
    masks.mucosa = np.zeros((0, 0), bool)
    masks.confidence = float("nan")
    r = qc_segmentation(masks, qc_cfg)
    assert codes(r) == ["insufficient_mucosa"]
    assert "segmentation.confidence" not in r.metrics
    masks.mucosa = np.ones((10, 10), np.uint8)  # non-bool mask
    masks.specular = None
    r2 = qc_segmentation(masks, qc_cfg)
    assert r2.passed and r2.metrics["segmentation.specular_fraction"] == 0.0
    masks.mucosa = [[1, 1], [1]]  # ragged: not a mask at all
    assert codes(qc_segmentation(masks, qc_cfg)) == ["insufficient_mucosa"]


# --------------------------------------------------------------------------------------------
# A whole site
# --------------------------------------------------------------------------------------------


def test_good_site_combines_to_pass(qc_cfg, wl, af, rng):
    site = combine_qc(
        [
            qc_white_light(wl, qc_cfg),
            qc_autofluorescence(af, qc_cfg),
            qc_dark_frame(qh.make_dark_frame(rng), qc_cfg),
            qc_registration(qh.make_registration(0.9), qc_cfg),
            qc_segmentation(qh.make_masks(0.6), qc_cfg),
        ]
    )
    assert site.passed and site.score == 1.0
    assert {"wl.blur_laplacian_var", "af.green_p90", "dark.mean"} <= set(site.metrics)


def test_site_without_dark_frame_runs_with_warning(qc_cfg, wl, af):
    from mukha_pipeline.qc import build_result

    site = combine_qc(
        [
            qc_white_light(wl, qc_cfg),
            qc_autofluorescence(af, qc_cfg),
            build_result([missing_dark_frame_issue()], {}, qc_cfg),
        ]
    )
    assert site.passed and codes(site) == ["no_dark_frame"]
    assert site.score == pytest.approx(1 - qc_cfg["score_weights"]["warning"])


def test_metrics_are_namespaced_and_json_safe(qc_cfg, wl, af, rng):
    results = {
        "wl.": qc_white_light(np.zeros((100, 100, 3), np.float32), qc_cfg),
        "af.": qc_autofluorescence(np.zeros((100, 100, 3), np.float32), qc_cfg),
        "dark.": qc_dark_frame(qh.make_dark_frame(rng), qc_cfg),
        "registration.": qc_registration(qh.make_registration(float("nan")), qc_cfg),
        "segmentation.": qc_segmentation(qh.make_masks(0.0), qc_cfg),
    }
    for prefix, r in results.items():
        assert r.metrics and all(k.startswith(prefix) for k in r.metrics), prefix
        assert all(i.metric is None or i.metric.startswith(prefix) for i in r.issues)
        text = r.model_dump_json()
        assert "NaN" not in text and "Infinity" not in text
