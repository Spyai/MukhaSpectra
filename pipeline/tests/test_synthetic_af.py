"""Synthetic autofluorescence images and dark frames."""

from __future__ import annotations

import cv2
import numpy as np
import pytest
from synthetic_helpers import MEDIUM, SMALL, assert_float_image, relative_af_proxy

from mukha_pipeline.preprocess.correction import linear_to_srgb, srgb_to_linear
from mukha_pipeline.synthetic import (
    LesionSpec,
    generate_af_from_wl,
    generate_dark_frame,
    generate_wl_scene,
    random_misalignment,
    stamp_mask,
)
from mukha_pipeline.synthetic._fields import dilate, erode
from mukha_pipeline.synthetic.af import (
    DEFAULT_DARK_LEVEL,
    MAX_ROTATION_DEG,
    MAX_SCALE_CHANGE,
    MAX_SHIFT_PX,
    MUCOSA_GREEN,
)

# The AF generator writes what a phone file holds: sRGB-ENCODED values. Its physics (green
# emission, FVL depth, porphyrin red) is modelled in linear light, so tests of that physics
# linearise first, exactly as the analysis does.
DARK_LINEAR = float(srgb_to_linear(np.float32(DEFAULT_DARK_LEVEL)))


@pytest.fixture(scope="module")
def clean_scene():
    return generate_wl_scene(np.random.default_rng(21), MEDIUM, site="buccal_mucosa_left")


@pytest.fixture(scope="module")
def lesion_scene():
    spec = LesionSpec(kind="white_patch", center=(0.5, 0.42), radii=(0.14, 0.095), fvl_depth=0.4)
    return generate_wl_scene(
        np.random.default_rng(22), MEDIUM, site="buccal_mucosa_left", lesion=spec
    )


def _af(scene, seed: int = 0, **kwargs):
    defaults = {
        "tissue_mask": scene.tissue_mask,
        "teeth_mask": scene.teeth_mask,
        "specular_mask": scene.specular_mask,
        "misalign": False,
    }
    defaults.update(kwargs)
    return generate_af_from_wl(scene.wl, np.random.default_rng(seed), **defaults)


def test_af_is_green_dominant_with_little_blue(clean_scene) -> None:
    af = _af(clean_scene)
    assert_float_image(af, MEDIUM)
    lin = srgb_to_linear(af)
    tissue = lin[clean_scene.tissue_mask & ~clean_scene.specular_mask]
    r, g, b = np.median(tissue, axis=0)
    assert g > 2 * r and g > 3 * b, "healthy mucosa glows green"
    blue_fraction = np.median(af[..., 2] / (af.sum(axis=-1) + 1e-6))
    assert blue_fraction < 0.2, "long-pass filter keeps blue low (also in the encoded file)"
    assert np.percentile(af[..., 1], 90) > 0.12, "violet torch clearly on"


def test_af_output_is_srgb_encoded_and_fvl_depth_is_a_linear_fraction(lesion_scene) -> None:
    """science.0: the file is sRGB-encoded like a phone JPEG; the FVL depth is a fraction of
    LIGHT, so it is only recovered after linearising (encoded values hide much of the loss)."""
    depth = lesion_scene.lesion.fvl_depth
    af = _af(lesion_scene, bloom=0.0, fvl_mask=lesion_scene.lesion_mask, fvl_depth=depth)
    inner = erode(lesion_scene.lesion_mask, 6)
    ring = dilate(lesion_scene.lesion_mask, 30) & ~dilate(lesion_scene.lesion_mask, 10)
    ring &= lesion_scene.tissue_mask & ~lesion_scene.specular_mask
    lin_green = srgb_to_linear(af)[..., 1] - DARK_LINEAR
    lin_ratio = np.median(lin_green[inner]) / np.median(lin_green[ring])
    enc_ratio = np.median(af[..., 1][inner]) / np.median(af[..., 1][ring])
    assert lin_ratio == pytest.approx(1.0 - depth, abs=0.06)
    assert enc_ratio > lin_ratio + 0.12, "encoding compresses the loss"
    # Healthy tissue: encoded value of the modelled linear level (torch hot spot ~MUCOSA_GREEN).
    healthy_lin = float(np.percentile(lin_green[ring], 90))
    assert 0.6 * MUCOSA_GREEN < healthy_lin < 1.3 * MUCOSA_GREEN
    assert float(np.percentile(af[..., 1][ring], 90)) == pytest.approx(
        float(linear_to_srgb(np.float32(healthy_lin + DARK_LINEAR))), abs=0.02
    )


def test_teeth_fluoresce_brighter_than_mucosa(clean_scene) -> None:
    af = srgb_to_linear(_af(clean_scene))
    teeth = erode(clean_scene.teeth_mask, 3)
    assert af[teeth][:, 1].mean() > af[clean_scene.tissue_mask][:, 1].mean() * 1.3


def test_fvl_region_is_darker_by_about_the_depth(lesion_scene) -> None:
    depth = lesion_scene.lesion.fvl_depth
    af = _af(lesion_scene, bloom=0.0, fvl_mask=lesion_scene.lesion_mask, fvl_depth=depth)
    green = srgb_to_linear(af)[..., 1] - DARK_LINEAR
    inner = erode(lesion_scene.lesion_mask, 6)
    ring = dilate(lesion_scene.lesion_mask, 30) & ~dilate(lesion_scene.lesion_mask, 10)
    ring &= lesion_scene.tissue_mask & ~lesion_scene.specular_mask
    ratio = np.median(green[inner]) / np.median(green[ring])
    assert ratio == pytest.approx(1.0 - depth, abs=0.06)


def test_red_to_green_ratio_rises_in_fvl(lesion_scene) -> None:
    af = srgb_to_linear(
        _af(lesion_scene, bloom=0.0, fvl_mask=lesion_scene.lesion_mask, fvl_depth=0.4)
    )
    inner = erode(lesion_scene.lesion_mask, 6)
    ring = dilate(lesion_scene.lesion_mask, 30) & ~dilate(lesion_scene.lesion_mask, 10)
    ring &= lesion_scene.tissue_mask

    def rg(m):
        return np.median(af[..., 0][m]) / np.median(af[..., 1][m])

    assert rg(inner) > rg(ring)


def test_lesion_whiteness_does_not_brighten_af(lesion_scene) -> None:
    no_loss = _af(lesion_scene, fvl_mask=lesion_scene.lesion_mask, fvl_depth=0.0, bloom=0.0)
    inner = erode(lesion_scene.lesion_mask, 6)
    ring = dilate(lesion_scene.lesion_mask, 30) & ~dilate(lesion_scene.lesion_mask, 10)
    ring &= lesion_scene.tissue_mask & ~lesion_scene.specular_mask
    ratio = np.median(no_loss[..., 1][inner]) / np.median(no_loss[..., 1][ring])
    assert ratio == pytest.approx(1.0, abs=0.06)


def test_clean_scene_has_no_false_fluorescence_loss(clean_scene) -> None:
    af = srgb_to_linear(_af(clean_scene))
    tissue = clean_scene.tissue_mask & ~clean_scene.specular_mask
    rel = relative_af_proxy(af, tissue, DARK_LINEAR)
    low = (rel < 0.75) & tissue
    opened = cv2.morphologyEx(low.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    assert opened.sum() < 50
    assert np.median(rel[tissue]) == pytest.approx(1.0, abs=0.08)


def test_fvl_mask_below_threshold_in_relative_af(lesion_scene) -> None:
    af = srgb_to_linear(_af(lesion_scene, fvl_mask=lesion_scene.lesion_mask, fvl_depth=0.4))
    tissue = lesion_scene.tissue_mask & ~lesion_scene.specular_mask
    healthy = tissue & ~dilate(lesion_scene.lesion_mask, 15)
    rel = relative_af_proxy(af, healthy, DARK_LINEAR)
    inner = erode(lesion_scene.lesion_mask, 6)
    assert rel[inner].mean() < 0.7
    assert (rel[healthy] >= 0.75).mean() > 0.995


def test_porphyrin_spots_raise_red_only_where_placed(clean_scene) -> None:
    base = _af(clean_scene, seed=5, porphyrin_spots=0)
    spots = _af(clean_scene, seed=5, porphyrin_spots=3)
    diff = spots[..., 0] - base[..., 0]
    assert diff.max() > 0.15, "orange-red porphyrin fluorescence"
    changed = np.abs(spots - base).max(axis=-1) > 1e-3
    assert 0 < changed.mean() < 0.02
    assert np.array_equal(spots[..., 2], base[..., 2])
    hot = diff > 0.1
    lin = srgb_to_linear(spots)
    assert (lin[..., 0][hot] / lin[..., 1][hot]).mean() > 0.85, "flaggable as porphyrin"
    near_teeth = dilate(clean_scene.teeth_mask, 30)
    assert near_teeth[hot].mean() > 0.8, "plaque sits along the gingival margin"


def test_porphyrin_without_teeth_uses_tissue_interior() -> None:
    scene = generate_wl_scene(np.random.default_rng(23), SMALL, site="tongue_dorsum")
    base = _af(scene, seed=1)
    spots = _af(scene, seed=1, porphyrin_spots=2)
    assert (spots[..., 0] - base[..., 0]).max() > 0.1


def test_misalignment_transform_is_returned_and_bounded(clean_scene) -> None:
    af, transform = generate_af_from_wl(
        clean_scene.wl,
        np.random.default_rng(31),
        tissue_mask=clean_scene.tissue_mask,
        teeth_mask=clean_scene.teeth_mask,
        return_transform=True,
    )
    assert_float_image(af, MEDIUM)
    assert transform.shape == (3, 3)
    a = transform[:2, :2]
    scale = np.sqrt(abs(np.linalg.det(a)))
    angle = np.degrees(np.arctan2(a[1, 0], a[0, 0]))
    assert abs(scale - 1) <= MAX_SCALE_CHANGE + 1e-9
    assert abs(angle) <= MAX_ROTATION_DEG + 1e-9
    h, w = MEDIUM
    centre = transform @ np.array([w / 2, h / 2, 1.0])
    shift = np.hypot(centre[0] - w / 2, centre[1] - h / 2)
    assert shift <= MAX_SHIFT_PX * min(h, w) / 768 + 1e-6
    assert not np.allclose(transform, np.eye(3))


def test_transform_maps_wl_coordinates_to_af(lesion_scene) -> None:
    af, transform = generate_af_from_wl(
        lesion_scene.wl,
        np.random.default_rng(32),
        tissue_mask=lesion_scene.tissue_mask,
        teeth_mask=lesion_scene.teeth_mask,
        fvl_mask=lesion_scene.lesion_mask,
        fvl_depth=0.6,
        return_transform=True,
    )
    h, w = MEDIUM
    back = cv2.warpAffine(srgb_to_linear(af), np.linalg.inv(transform)[:2], (w, h))
    inner = erode(lesion_scene.lesion_mask, 6)
    ring = dilate(lesion_scene.lesion_mask, 30) & ~dilate(lesion_scene.lesion_mask, 10)
    ring &= lesion_scene.tissue_mask
    assert np.median(back[..., 1][inner]) < 0.6 * np.median(back[..., 1][ring])


def test_no_misalignment_gives_identity() -> None:
    scene = generate_wl_scene(np.random.default_rng(24), SMALL, site="hard_palate")
    _, transform = generate_af_from_wl(
        scene.wl, np.random.default_rng(0), misalign=False, return_transform=True
    )
    assert np.array_equal(transform, np.eye(3))


def test_random_misalignment_respects_custom_limits() -> None:
    rng = np.random.default_rng(0)
    for _ in range(20):
        m = random_misalignment(
            rng, (100, 200), max_rotation_deg=0.0, max_shift_px=3.0, max_scale_change=0.0
        )
        assert np.allclose(m[:2, :2], np.eye(2))
        assert np.hypot(m[0, 2], m[1, 2]) <= 3.0 + 1e-9


def test_masks_are_estimated_when_not_given(clean_scene) -> None:
    af = srgb_to_linear(
        generate_af_from_wl(clean_scene.wl, np.random.default_rng(3), misalign=False)
    )
    tissue = af[clean_scene.tissue_mask & ~clean_scene.specular_mask]
    assert np.median(tissue[:, 1]) > 2 * np.median(tissue[:, 0])
    assert af[erode(clean_scene.teeth_mask, 3)][:, 1].mean() > np.median(tissue[:, 1])


def test_af_is_deterministic(clean_scene) -> None:
    a = generate_af_from_wl(clean_scene.wl, np.random.default_rng(9), porphyrin_spots=2)
    b = generate_af_from_wl(clean_scene.wl, np.random.default_rng(9), porphyrin_spots=2)
    assert np.array_equal(a, b)


def test_af_carries_camera_fixed_stamp_text(clean_scene) -> None:
    stamped = _af(clean_scene, seed=4)
    plain = _af(clean_scene, seed=4, watermark=False)
    stamp = stamp_mask(MEDIUM)
    assert np.array_equal(stamped[~stamp], plain[~stamp])
    text = np.abs(stamped - plain).max(axis=-1) > 0.2
    assert text.sum() > 100
    # Identical text pixels in WL and AF: registration recognises a burned-in label.
    same = np.all(np.abs(clean_scene.wl - stamped) <= 0.05, axis=-1) & text
    assert same.sum() > 50


def test_af_input_validation(clean_scene) -> None:
    with pytest.raises(ValueError, match=r"\(H, W, 3\)"):
        generate_af_from_wl(clean_scene.wl[..., 0], np.random.default_rng(0))
    with pytest.raises(ValueError, match="fvl_depth"):
        generate_af_from_wl(clean_scene.wl, np.random.default_rng(0), fvl_depth=1.5)


def test_af_without_any_tissue_still_renders() -> None:
    wl = np.zeros((*SMALL, 3), np.float32)
    af = generate_af_from_wl(wl, np.random.default_rng(0), tissue_mask=np.zeros(SMALL, bool))
    assert_float_image(af, SMALL)


def test_dark_frame_level_shape_and_optional_stamp() -> None:
    """The dark frame is what the file holds (encoded, like the AF capture it goes with)."""
    dark = generate_dark_frame(SMALL, np.random.default_rng(0), level=0.01)
    assert_float_image(dark, SMALL)
    assert np.median(dark) == pytest.approx(0.01, abs=0.003)
    assert dark.mean() < 0.03, "passes the dark-frame QC with room to spare"
    stamped = generate_dark_frame((*SMALL, 3), np.random.default_rng(0), watermark=True)
    stamp = stamp_mask(SMALL)
    assert (stamped[stamp] - dark[stamp]).max() > 0.03, "faint SYNTHETIC text on request"
    assert np.array_equal(stamped[~stamp], dark[~stamp])
    hot = generate_dark_frame((1000, 1000), np.random.default_rng(1), watermark=False)
    assert hot.max() > 0.05, "a few hot pixels"


def test_dark_frame_is_deterministic() -> None:
    a = generate_dark_frame(SMALL, np.random.default_rng(5))
    b = generate_dark_frame(SMALL, np.random.default_rng(5))
    assert np.array_equal(a, b)
