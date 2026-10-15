"""Tests for the classical mucosa segmenter (SYNTHETIC scenes from af_helpers)."""

from __future__ import annotations

import copy

import numpy as np
import pytest
from af_helpers import BACKGROUND_RGB, PATCH_RGB, TEETH_RGB, disc, ellipse, make_wl_scene, paint

from mukha_pipeline.preprocess.segmentation import (
    ClassicalSegmenter,
    Segmenter,
    classify_pixels,
    confidence_heuristic,
    fill_enclosed_holes,
    fill_partially_enclosed,
    refine_masks_with_af,
    remove_small_components,
    sanitize_rgb,
    segmentation_summary,
)
from mukha_pipeline.types import Masks


def iou(a: np.ndarray, b: np.ndarray) -> float:
    return float((a & b).sum()) / max(int((a | b).sum()), 1)


@pytest.fixture
def seg_cfg(analysis_cfg: dict) -> dict:
    return copy.deepcopy(analysis_cfg["segmentation"])


@pytest.fixture
def scene(rng: np.random.Generator):
    return make_wl_scene(rng)


def test_segmenter_protocol_and_identity(seg_cfg: dict) -> None:
    seg = ClassicalSegmenter(seg_cfg)
    assert isinstance(seg, Segmenter)
    assert seg.name == "classical_hsv_lab"
    assert seg.version == seg_cfg["version"]


def test_mucosa_teeth_background(scene, seg_cfg: dict) -> None:
    masks = ClassicalSegmenter(seg_cfg).segment(scene.wl)
    assert masks.mucosa.dtype == bool and masks.mucosa.shape == scene.wl.shape[:2]
    assert iou(masks.mucosa, scene.tissue) > 0.97
    assert masks.mucosa[scene.teeth].mean() == 0.0  # teeth on the border stay excluded
    assert masks.teeth[scene.teeth].mean() > 0.95
    assert masks.background[scene.background & ~scene.teeth].mean() > 0.95
    assert not (masks.mucosa & masks.background).any()
    assert 0.8 < masks.confidence <= 1.0


def test_enclosed_white_patch_stays_in_mucosa(scene, seg_cfg: dict) -> None:
    """Leukoplakia looks like teeth to colour rules; dropping it would be false reassurance."""
    seg = ClassicalSegmenter(seg_cfg)
    classes = classify_pixels(sanitize_rgb(scene.wl), seg_cfg)
    patch = scene.patch & ~scene.specular
    assert classes.teeth[patch].mean() > 0.9  # the naive rule calls it teeth...
    masks, diag = seg.segment_detailed(scene.wl)
    assert masks.mucosa[patch].mean() > 0.97  # ...hole filling keeps it as tissue
    assert diag.filled[patch].mean() > 0.9
    assert masks.teeth[patch].mean() < 0.02


def test_without_hole_filling_the_patch_is_lost(scene, seg_cfg: dict) -> None:
    seg_cfg.update(fill_holes_max_fraction=0.0, partial_enclosure_min=2.0)
    masks = ClassicalSegmenter(seg_cfg).segment(scene.wl)
    assert masks.mucosa[scene.patch & ~scene.specular].mean() < 0.1


def test_hole_larger_than_limit_is_not_filled(scene, seg_cfg: dict) -> None:
    patch_frac = scene.patch.mean()
    seg_cfg.update(fill_holes_max_fraction=patch_frac / 2)
    masks = ClassicalSegmenter(seg_cfg).segment(scene.wl)
    assert masks.mucosa[scene.patch & ~scene.specular].mean() < 0.1


def test_specular_dots_excluded_but_reported(scene, seg_cfg: dict) -> None:
    masks = ClassicalSegmenter(seg_cfg).segment(scene.wl)
    assert scene.specular.any()
    assert not (masks.mucosa & scene.specular).any()
    assert masks.specular[scene.specular].all()


def test_partially_enclosed_patch_touching_shadow_is_kept(rng, seg_cfg: dict) -> None:
    """A white patch at the cheek edge that touches a sliver of shadow is still tissue."""
    sc = make_wl_scene(rng, white_patch=False, specular_dots=False)
    h, w = sc.wl.shape[:2]
    cy, cx = int(h * 0.55), int(w * 0.5)
    patch = disc((h, w), cy, cx, 30)
    # A 12 px dark fold runs from the patch to the shadow outside the cheek, so the patch is not
    # strictly enclosed by mucosa any more.
    fold = np.zeros((h, w), bool)
    fold[cy - 6 : cy + 6, cx:] = True
    paint(sc.wl, fold & ~patch, BACKGROUND_RGB)
    paint(sc.wl, patch, PATCH_RGB)
    masks = ClassicalSegmenter(seg_cfg).segment(sc.wl)
    assert masks.mucosa[patch].mean() > 0.95
    seg_cfg.update(partial_enclosure_min=2.0)  # disabled -> strict enclosure only
    strict = ClassicalSegmenter(seg_cfg).segment(sc.wl)
    assert strict.mucosa[patch].mean() < 0.1


def test_summary_ok(scene, seg_cfg: dict) -> None:
    seg = ClassicalSegmenter(seg_cfg)
    masks = seg.segment(scene.wl)
    s = segmentation_summary(masks, seg, seg_cfg)
    assert s.status == "ok"
    assert s.method == "classical_hsv_lab" and s.version == seg_cfg["version"]
    assert s.mucosa_fraction == pytest.approx(masks.mucosa.mean())
    assert s.teeth_fraction > 0.05
    assert 0 < s.specular_fraction < 0.01


def test_summary_low_confidence_with_ambiguous_colours(rng, seg_cfg: dict) -> None:
    sc = make_wl_scene(rng, white_patch=False, specular_dots=False)
    h, w = sc.wl.shape[:2]
    # Yellow-green cast over most of the tissue: no rule claims it.
    cast = np.zeros((h, w), bool)
    cast[:, : int(w * 0.5)] = True
    paint(sc.wl, cast & sc.tissue, (0.6, 0.7, 0.2))
    seg = ClassicalSegmenter(seg_cfg)
    s = segmentation_summary(seg.segment(sc.wl), seg, seg_cfg)
    assert s.status == "low_confidence"
    assert s.confidence < 0.5


def test_summary_failed_on_all_dark_image(seg_cfg: dict) -> None:
    seg = ClassicalSegmenter(seg_cfg)
    masks = seg.segment(np.zeros((64, 80, 3), np.float32))
    assert not masks.mucosa.any()
    assert masks.confidence == 0.0
    assert segmentation_summary(masks, seg, seg_cfg).status == "failed"


def test_summary_handles_nan_confidence(seg_cfg: dict) -> None:
    z = np.zeros((10, 10), bool)
    m = Masks(mucosa=~z, teeth=z, background=z, specular=z, confidence=float("nan"))
    s = segmentation_summary(m, ClassicalSegmenter(seg_cfg), seg_cfg)
    assert s.status == "failed" and s.confidence == 0.0


@pytest.mark.parametrize(
    "img",
    [
        np.full((40, 50, 3), np.nan, np.float32),
        np.zeros((0, 0, 3), np.float32),
        np.zeros((3, 4, 3), np.float32),
        (np.full((30, 40, 3), [220, 130, 140], np.uint8)),
        np.full((30, 40, 3), [56000, 33000, 36000], np.uint16),
        np.full((30, 40), 0.7, np.float32),
        np.full((30, 40, 1), 0.7, np.float32),
        np.full((30, 40, 4), 0.7, np.float32),
    ],
)
def test_robust_to_odd_inputs(img: np.ndarray, seg_cfg: dict) -> None:
    seg = ClassicalSegmenter(seg_cfg)
    masks = seg.segment(img)
    assert masks.mucosa.shape == img.shape[:2]
    assert 0.0 <= masks.confidence <= 1.0
    segmentation_summary(masks, seg, seg_cfg)


def test_uint8_pink_image_is_mucosa(seg_cfg: dict) -> None:
    img = np.full((60, 80, 3), [220, 130, 140], np.uint8)
    assert ClassicalSegmenter(seg_cfg).segment(img).mucosa.mean() > 0.95


def test_nan_pixels_do_not_crash_and_are_not_mucosa(scene, seg_cfg: dict) -> None:
    wl = scene.wl.copy()
    wl[100:110, 100:300] = np.nan  # a NaN stripe touching the tissue
    masks = ClassicalSegmenter(seg_cfg).segment(wl)
    assert iou(masks.mucosa, scene.tissue) > 0.9


def test_invalid_shape_raises() -> None:
    with pytest.raises(ValueError):
        sanitize_rgb(np.zeros((4, 4, 2), np.float32))


def test_works_without_config(scene) -> None:
    seg = ClassicalSegmenter(None)
    assert seg.version == "1.1.0"
    assert iou(seg.segment(scene.wl).mucosa, scene.tissue) > 0.95


def test_lab_redness_rule_catches_mucosa_outside_hue_range(seg_cfg: dict) -> None:
    img = np.full((40, 40, 3), [0.85, 0.55, 0.40], np.float32)  # orange-ish pink, hue ~ 23
    seg_cfg.update(mucosa_hue_ranges=[[0, 5]])
    assert ClassicalSegmenter(seg_cfg).segment(img).mucosa.mean() > 0.9
    seg_cfg.update(mucosa_lab_a_min=200, teeth_lab_a_max=200)
    assert ClassicalSegmenter(seg_cfg).segment(img).mucosa.mean() < 0.1


def test_small_helpers() -> None:
    m = np.zeros((20, 20), bool)
    m[2:4, 2:4] = True
    m[10:18, 10:18] = True
    out = remove_small_components(m, 10)
    assert not out[2:4, 2:4].any() and out[10:18, 10:18].all()
    assert remove_small_components(m, 1).sum() == m.sum()

    ring = np.zeros((20, 20), bool)
    ring[5:15, 5:15] = True
    ring[8:12, 8:12] = False
    assert fill_enclosed_holes(ring, 100).sum() == 16
    assert fill_enclosed_holes(ring, 10).sum() == 0
    assert fill_enclosed_holes(np.ones((5, 5), bool), 10).sum() == 0

    assert not fill_partially_enclosed(ring, np.zeros_like(ring), 100, 0.8).any()
    border_blob = np.zeros((20, 20), bool)
    border_blob[0:3, 0:3] = True
    assert not fill_partially_enclosed(ring, border_blob, 100, 0.0).any()
    inner = ~ring & ~np.pad(np.ones((18, 18), bool), 1, constant_values=False)
    assert not fill_partially_enclosed(ring, inner, 100, 0.5).any()


def test_confidence_heuristic_monotone() -> None:
    hi = confidence_heuristic(0.5, 0.0, 0.0, None)
    assert hi == pytest.approx(1.0)
    assert confidence_heuristic(0.05, 0.0, 0.0, None) < hi
    assert confidence_heuristic(0.5, 0.4, 0.0, None) < hi
    assert confidence_heuristic(0.5, 0.0, 0.5, None) == pytest.approx(0.5)
    assert confidence_heuristic(0.0, 0.0, 0.0, None) == 0.0


def test_pale_blanched_mucosa_is_not_called_teeth(rng, seg_cfg: dict) -> None:
    """Blanched mucosa (anaemia, submucous fibrosis) is bright and unsaturated like teeth but
    keeps a pink tinge (Lab a*); calling it teeth would hide it from the analysis."""
    sc = make_wl_scene(rng, teeth_band=False, white_patch=False, specular_dots=False)
    h, w = sc.wl.shape[:2]
    pale = np.zeros((h, w), bool)
    pale[:, int(w * 0.6) :] = True  # reaches the image border: hole filling cannot rescue it
    pale &= sc.tissue
    paint(sc.wl, pale, (0.90, 0.80, 0.80))
    masks = ClassicalSegmenter(seg_cfg).segment(sc.wl)
    assert masks.mucosa[pale].mean() > 0.95
    assert masks.teeth[pale].mean() < 0.02
    seg_cfg.update(teeth_lab_a_max=100.0)  # guard disabled -> the naive rule loses it
    assert ClassicalSegmenter(seg_cfg).segment(sc.wl).mucosa[pale].mean() < 0.1


# ------------------------------------------------------------------------------------------
# Regression: large white patches (review clinical.6 / science.1)
# ------------------------------------------------------------------------------------------

GREY_WHITE_RGB = (0.84, 0.83, 0.80)  # keratin plaque with no pink tinge (Lab a* ~ 0)


def test_bundled_hole_limit_allows_large_white_patches(seg_cfg: dict) -> None:
    assert seg_cfg["fill_holes_max_fraction"] >= 0.35


def test_large_enclosed_grey_white_patch_stays_in_mucosa(rng, seg_cfg: dict) -> None:
    """A homogeneous leukoplakia filling ~20% of a close-up used to be labelled teeth and
    silently dropped from the AF analysis, with confidence still high."""
    sc = make_wl_scene(rng, white_patch=False, specular_dots=False)
    h, w = sc.wl.shape[:2]
    patch = disc((h, w), h * 0.55, w * 0.62, 105) & sc.tissue
    assert 0.18 < patch.mean() < 0.22
    paint(sc.wl, patch, GREY_WHITE_RGB)
    classes = classify_pixels(sanitize_rgb(sc.wl), seg_cfg)
    assert classes.teeth[patch].mean() > 0.9  # the colour rule alone calls it teeth
    masks = ClassicalSegmenter(seg_cfg).segment(sc.wl)
    assert masks.mucosa[patch].mean() > 0.97
    assert masks.teeth[patch].mean() < 0.02


def test_enclosed_cavity_keeps_the_old_size_limit(rng, seg_cfg: dict) -> None:
    """Frontal view: the lips enclose teeth and the dark oral cavity. That opening is not a white
    patch and must not be filled into the mucosa just because the white-patch limit was raised."""
    sc = make_wl_scene(rng, teeth_band=False, white_patch=False, specular_dots=False)
    h, w = sc.wl.shape[:2]
    opening = ellipse((h, w), h * 0.56, w * 0.5, h * 0.2, w * 0.3)
    assert 0.12 < opening.mean() < 0.3
    upper = opening & (np.arange(h)[:, None] < h * 0.53)
    paint(sc.wl, opening, BACKGROUND_RGB)
    paint(sc.wl, upper, TEETH_RGB)
    masks = ClassicalSegmenter(seg_cfg).segment(sc.wl)
    assert masks.mucosa[opening].mean() < 0.05
    assert masks.teeth[upper].mean() > 0.9


# ------------------------------------------------------------------------------------------
# Regression: specular = small highlights only (review science.6)
# ------------------------------------------------------------------------------------------


@pytest.mark.parametrize("rgb", [(0.99, 0.96, 0.95), (1.0, 0.90, 0.90)], ids=["white", "pinkish"])
def test_bright_white_patch_is_not_cut_out_as_glare(rng, seg_cfg: dict, rgb) -> None:
    """A slightly over-exposed white patch is near-white (V >= 0.96, S low) over a large area.
    It is tissue, not a specular highlight, so it must stay in the analysable mucosa."""
    sc = make_wl_scene(rng, white_patch=False, specular_dots=False)
    h, w = sc.wl.shape[:2]
    patch = disc((h, w), h * 0.55, w * 0.55, 45) & sc.tissue
    tex = 1 + 0.02 * rng.standard_normal(int(patch.sum()))
    sc.wl[patch] = np.clip(np.asarray(rgb, np.float32)[None, :] * tex[:, None], 0, 1)
    masks = ClassicalSegmenter(seg_cfg).segment(sc.wl)
    assert masks.specular[patch].mean() < 0.02
    assert masks.mucosa[patch].mean() > 0.97
    s = segmentation_summary(masks, ClassicalSegmenter(seg_cfg), seg_cfg)
    assert s.specular_fraction < 0.005


def test_large_glint_field_is_not_specular_but_small_glints_are(rng, seg_cfg: dict) -> None:
    sc = make_wl_scene(rng, white_patch=False, specular_dots=True)
    h, w = sc.wl.shape[:2]
    big = disc((h, w), h * 0.62, w * 0.7, 30) & sc.tissue
    paint(sc.wl, big, (1.0, 1.0, 1.0))
    masks = ClassicalSegmenter(seg_cfg).segment(sc.wl)
    assert masks.specular[sc.specular].all()  # the small dots are still glare
    assert masks.specular[big].mean() < 0.02


# ------------------------------------------------------------------------------------------
# refine_masks_with_af: whitish areas that do not fluoresce like teeth are tissue
# ------------------------------------------------------------------------------------------


def _border_patch_scene(rng):
    """Mucosa ellipse, teeth band on top, a grey-white patch running out of the right border."""
    sc = make_wl_scene(rng, white_patch=False, specular_dots=False)
    h, w = sc.wl.shape[:2]
    patch = np.zeros((h, w), bool)
    patch[int(h * 0.42) : int(h * 0.7), int(w * 0.83) :] = True
    paint(sc.wl, patch, GREY_WHITE_RGB)
    return sc, patch


def _af_green(sc, patch: np.ndarray, patch_level: float) -> np.ndarray:
    green = np.full(sc.tissue.shape, 0.02, np.float32)  # dark cavity
    green[sc.tissue] = 0.40
    green[sc.teeth] = 0.90  # enamel and dentin fluoresce brightly under 405 nm
    green[patch] = patch_level
    return green


def test_dim_white_patch_at_the_border_is_moved_back_into_mucosa(rng, seg_cfg: dict) -> None:
    sc, patch = _border_patch_scene(rng)
    seg = ClassicalSegmenter(seg_cfg)
    masks = seg.segment(sc.wl)
    assert masks.teeth[patch].mean() > 0.9  # touches the border: colour rules call it teeth
    valid = np.ones(patch.shape, bool)
    refined, excluded = refine_masks_with_af(masks, _af_green(sc, patch, 0.20), valid, seg_cfg)
    assert refined.mucosa[patch].mean() > 0.95
    assert refined.teeth[patch].mean() < 0.02
    assert refined.teeth[sc.teeth].mean() > 0.95  # real teeth stay excluded
    assert not (refined.mucosa & refined.teeth).any()
    assert not (refined.mucosa & refined.specular).any()
    assert excluded < 0.005
    # The input is not modified.
    assert masks.teeth[patch].mean() > 0.9
    assert refined.confidence == masks.confidence


def test_bright_white_patch_stays_excluded_and_is_reported(rng, seg_cfg: dict) -> None:
    """Hyperkeratotic leukoplakia can fluoresce brightly: it stays excluded (documented
    limitation), and the returned fraction lets the caller warn about it."""
    sc, patch = _border_patch_scene(rng)
    masks = ClassicalSegmenter(seg_cfg).segment(sc.wl)
    valid = np.ones(patch.shape, bool)
    refined, excluded = refine_masks_with_af(masks, _af_green(sc, patch, 0.60), valid, seg_cfg)
    assert refined.mucosa[patch].mean() < 0.05
    touching = patch & masks.teeth
    assert excluded == pytest.approx(touching.mean(), rel=0.1)
    assert excluded > 0.02


def test_refine_ignores_misregistration_slivers_and_invalid_af(seg_cfg: dict) -> None:
    h, w = 200, 240
    z = np.zeros((h, w), bool)
    mucosa, teeth = z.copy(), z.copy()
    mucosa[:100] = True
    teeth[100:] = True  # teeth fill the lower half and touch the mucosa
    masks = Masks(mucosa=mucosa, teeth=teeth, background=z.copy(), specular=z.copy(), confidence=1)
    green = np.where(mucosa, 0.4, 0.9).astype(np.float32)
    green[100:103] = 0.05  # 3 px of shifted dark AF along the tooth edge
    refined, excluded = refine_masks_with_af(masks, green, np.ones((h, w), bool), seg_cfg)
    assert (refined.mucosa == mucosa).all() and (refined.teeth == teeth).all()
    assert excluded == pytest.approx(0.5)
    # Dim AF where the registered image has no data cannot be judged: nothing moves.
    green[100:] = 0.1
    valid = np.ones((h, w), bool)
    valid[100:] = False
    refined, _ = refine_masks_with_af(masks, green, valid, seg_cfg)
    assert (refined.teeth == teeth).all()
    # Dim and valid: the blob does not fluoresce like teeth, so it is tissue.
    refined, excluded = refine_masks_with_af(masks, green, np.ones((h, w), bool), seg_cfg)
    assert refined.mucosa[100:].all() and not refined.teeth.any() and excluded == 0.0


def test_refine_edge_cases(seg_cfg: dict) -> None:
    z = np.zeros((40, 50), bool)
    empty = Masks(mucosa=z, teeth=z.copy(), background=~z, specular=z.copy(), confidence=0.0)
    out, excluded = refine_masks_with_af(empty, np.zeros((40, 50), np.float32), ~z, seg_cfg)
    assert not out.mucosa.any() and excluded == 0.0
    with pytest.raises(ValueError):
        refine_masks_with_af(empty, np.zeros((10, 10), np.float32), ~z, seg_cfg)
    nan_green = np.full((40, 50), np.nan, np.float32)
    teeth = z.copy()
    teeth[20:] = True
    m = Masks(mucosa=~teeth, teeth=teeth, background=z.copy(), specular=z.copy(), confidence=1)
    out, excluded = refine_masks_with_af(m, nan_green, ~z, None)
    assert (out.teeth == teeth).all() and excluded == pytest.approx(0.5)


def test_summary_failed_when_mucosa_is_absent(seg_cfg: dict) -> None:
    z = np.zeros((20, 20), bool)
    m = Masks(mucosa=z, teeth=~z, background=z.copy(), specular=z.copy(), confidence=0.9)
    seg_cfg.update(failed_mucosa_fraction=0.0)
    assert segmentation_summary(m, ClassicalSegmenter(seg_cfg), seg_cfg).status == "failed"
