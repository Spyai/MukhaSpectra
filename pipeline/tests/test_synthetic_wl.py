"""Synthetic white-light scenes and lesions."""

from __future__ import annotations

import numpy as np
import pytest
from synthetic_helpers import MEDIUM, SMALL, assert_float_image, hsv_value

from mukha_pipeline.sites import OralSite
from mukha_pipeline.synthetic import (
    LESION_KINDS,
    LesionSpec,
    SyntheticScene,
    generate_wl_image,
    generate_wl_scene,
    stamp_mask,
)
from mukha_pipeline.synthetic._fields import dilate, erode
from mukha_pipeline.synthetic.lesions import make_lesion_patch, paint_lesion


@pytest.mark.parametrize("site", list(OralSite))
def test_every_site_renders_with_consistent_masks(site: OralSite) -> None:
    scene = generate_wl_scene(np.random.default_rng(3), SMALL, site=site)
    assert isinstance(scene, SyntheticScene)
    assert_float_image(scene.wl, SMALL)
    for mask in (scene.tissue_mask, scene.teeth_mask, scene.lesion_mask, scene.background_mask):
        assert mask.dtype == bool and mask.shape == SMALL
    # Tissue, teeth and background partition the frame.
    assert not (scene.tissue_mask & scene.teeth_mask).any()
    assert not (scene.tissue_mask & scene.background_mask).any()
    assert not (scene.teeth_mask & scene.background_mask).any()
    assert (scene.tissue_mask | scene.teeth_mask | scene.background_mask).all()
    assert scene.tissue_mask.mean() > 0.5, "an intraoral close-up is mostly mucosa"
    assert not scene.lesion_mask.any() and scene.lesion is None
    assert scene.site == site


def test_site_accepts_string_value() -> None:
    scene = generate_wl_scene(np.random.default_rng(0), SMALL, site="hard_palate")
    assert scene.site is OralSite.HARD_PALATE


@pytest.mark.parametrize(
    ("site", "has_teeth"),
    [
        (OralSite.BUCCAL_MUCOSA_LEFT, True),
        (OralSite.LOWER_GINGIVA, True),
        (OralSite.UPPER_LIP_MUCOSA, True),
        (OralSite.RETROMOLAR_RIGHT, True),
        (OralSite.LATERAL_TONGUE_LEFT, False),
        (OralSite.SOFT_PALATE, False),
    ],
)
def test_teeth_appear_only_where_anatomy_puts_them(site: OralSite, has_teeth: bool) -> None:
    scene = generate_wl_scene(np.random.default_rng(1), SMALL, site=site)
    assert scene.teeth_mask.any() == has_teeth


def test_teeth_edge_follows_the_jaw() -> None:
    lower = generate_wl_scene(np.random.default_rng(2), SMALL, site=OralSite.LOWER_GINGIVA)
    upper = generate_wl_scene(np.random.default_rng(2), SMALL, site=OralSite.UPPER_GINGIVA)
    h = SMALL[0]
    assert lower.teeth_mask[: h // 4].mean() > lower.teeth_mask[-h // 4 :].mean()
    assert upper.teeth_mask[-h // 4 :].mean() > upper.teeth_mask[: h // 4].mean()


def test_left_and_right_sites_are_mirrored_layouts() -> None:
    left = generate_wl_scene(np.random.default_rng(4), SMALL, site=OralSite.BUCCAL_MUCOSA_LEFT)
    right = generate_wl_scene(np.random.default_rng(4), SMALL, site=OralSite.BUCCAL_MUCOSA_RIGHT)
    w = SMALL[1]
    # Molars sit on the posterior side, which flips with the side of the mouth.
    assert left.teeth_mask[:, w // 2 :].sum() > left.teeth_mask[:, : w // 2].sum()
    assert right.teeth_mask[:, : w // 2].sum() > right.teeth_mask[:, w // 2 :].sum()


def test_mucosa_is_pink_red_and_teeth_off_white() -> None:
    scene = generate_wl_scene(np.random.default_rng(5), MEDIUM, site=OralSite.BUCCAL_MUCOSA_LEFT)
    tissue = scene.wl[scene.tissue_mask & ~scene.specular_mask]
    r, g, b = np.median(tissue, axis=0)
    assert r > g + 0.15 and r > b + 0.15, "mucosa is red-dominant"
    teeth = scene.wl[scene.teeth_mask & ~scene.specular_mask]
    tr, tg, tb = np.median(teeth, axis=0)
    assert min(tr, tg, tb) > 0.55 and tr - tb < 0.2, "teeth are bright and near-neutral"
    cavity = scene.wl[scene.background_mask]
    assert np.median(hsv_value(cavity)) <= 0.12, "oral cavity shadow reads as background"


def test_wet_highlights_are_small_and_present() -> None:
    scene = generate_wl_scene(np.random.default_rng(6), MEDIUM, site=OralSite.LOWER_LIP_MUCOSA)
    frac = scene.specular_mask.mean()
    assert 0.0 < frac < 0.015, "a few speckles, below the glare warning level"
    assert hsv_value(scene.wl)[scene.specular_mask].mean() > 0.8


def test_same_seed_is_identical_and_different_seed_differs() -> None:
    a = generate_wl_scene(np.random.default_rng(7), SMALL, site=OralSite.TONGUE_DORSUM)
    b = generate_wl_scene(np.random.default_rng(7), SMALL, site=OralSite.TONGUE_DORSUM)
    c = generate_wl_scene(np.random.default_rng(8), SMALL, site=OralSite.TONGUE_DORSUM)
    assert np.array_equal(a.wl, b.wl)
    assert np.array_equal(a.tissue_mask, b.tissue_mask)
    assert not np.array_equal(a.wl, c.wl)


def test_every_image_carries_the_stamp() -> None:
    scene = generate_wl_scene(np.random.default_rng(9), SMALL, site=OralSite.SOFT_PALATE)
    raw = generate_wl_scene(
        np.random.default_rng(9), SMALL, site=OralSite.SOFT_PALATE, watermark=False
    )
    stamp = stamp_mask(SMALL)
    assert not np.array_equal(scene.wl[stamp], raw.wl[stamp])
    assert np.array_equal(scene.wl[~stamp], raw.wl[~stamp])
    assert scene.background_mask[stamp].all() and not scene.tissue_mask[stamp].any()


@pytest.mark.parametrize("kind", LESION_KINDS)
def test_lesion_is_enclosed_by_mucosa_and_reported(kind: str) -> None:
    spec = LesionSpec(kind=kind, center=(0.5, 0.45), radii=(0.12, 0.08), irregularity=0.6)
    scene = generate_wl_scene(
        np.random.default_rng(10), MEDIUM, site="buccal_mucosa_left", lesion=spec
    )
    mask = scene.lesion_mask
    assert mask.sum() > 1000
    ring = dilate(mask, 4) & ~mask
    assert scene.tissue_mask[ring].all(), "lesion fully enclosed by mucosa"
    assert not (mask & stamp_mask(MEDIUM)).any(), "never under the SYNTHETIC stamp"
    ys, xs = np.nonzero(mask)
    assert ys.min() > 0 and xs.min() > 0 and ys.max() < MEDIUM[0] - 1
    assert scene.lesion is not None and scene.lesion.kind == kind


def test_lesion_kinds_change_colour_as_expected() -> None:
    def lesion_vs_ring(kind: str, core_px: int = 0) -> tuple[np.ndarray, np.ndarray]:
        spec = LesionSpec(kind=kind, center=(0.5, 0.45), radii=(0.12, 0.08), intensity=0.9)
        scene = generate_wl_scene(
            np.random.default_rng(11), MEDIUM, site="lateral_tongue_left", lesion=spec
        )
        inner = erode(scene.lesion_mask, core_px) & ~scene.specular_mask
        ring = dilate(scene.lesion_mask, 25) & ~dilate(scene.lesion_mask, 8)
        ring &= scene.tissue_mask & ~scene.specular_mask
        return np.median(scene.wl[inner], axis=0), np.median(scene.wl[ring], axis=0)

    white, around = lesion_vs_ring("white_patch")
    assert white.mean() > around.mean() + 0.1, "leukoplakia is whiter"
    assert (white.max() - white.min()) < (around.max() - around.min()), "and less saturated"
    red, around = lesion_vs_ring("red_patch")
    assert red[1] < around[1] - 0.05 and red[0] / red[1] > around[0] / around[1], "erythroplakia"
    ulcer, around = lesion_vs_ring("ulcer", core_px=14)  # the floor, inside the margin
    assert ulcer[1] > around[1] + 0.05, "paler, yellow-grey fibrinous floor"
    assert ulcer[1] - ulcer[2] > around[1] - around[2] + 0.03, "yellowish"


def test_lesion_requested_off_mucosa_is_moved_onto_mucosa() -> None:
    # The top-right corner of a left buccal photo is the dark oral cavity.
    spec = LesionSpec(kind="red_patch", center=(0.98, 0.03), radii=(0.08, 0.06))
    scene = generate_wl_scene(
        np.random.default_rng(12), SMALL, site="buccal_mucosa_left", lesion=spec
    )
    assert scene.lesion is not None
    assert scene.lesion.center != spec.center
    assert scene.tissue_mask[scene.lesion_mask].all()


def test_lesion_requested_on_mirrored_site_lands_where_asked() -> None:
    spec = LesionSpec(kind="white_patch", center=(0.35, 0.40), radii=(0.08, 0.06))
    scene = generate_wl_scene(
        np.random.default_rng(13), MEDIUM, site="buccal_mucosa_right", lesion=spec
    )
    ys, xs = np.nonzero(scene.lesion_mask)
    assert abs(xs.mean() / MEDIUM[1] - 0.35) < 0.05
    assert abs(ys.mean() / MEDIUM[0] - 0.40) < 0.05
    assert scene.lesion is not None
    assert scene.lesion.center == pytest.approx((0.35, 0.40), abs=0.01)


def test_oversized_lesion_is_shrunk_to_fit() -> None:
    spec = LesionSpec(kind="white_patch", radii=(0.45, 0.45))
    scene = generate_wl_scene(np.random.default_rng(14), SMALL, site="tongue_dorsum", lesion=spec)
    assert scene.lesion is not None and scene.lesion.radii[0] < 0.45
    assert scene.tissue_mask[scene.lesion_mask].all()


def test_generate_wl_image_is_a_thin_wrapper() -> None:
    spec = LesionSpec(kind="ulcer", center=(0.5, 0.5), radii=(0.07, 0.06))
    wl, mask = generate_wl_image(
        np.random.default_rng(15), SMALL, site="lower_lip_mucosa", lesion=spec
    )
    scene = generate_wl_scene(
        np.random.default_rng(15), SMALL, site="lower_lip_mucosa", lesion=spec
    )
    assert np.array_equal(wl, scene.wl) and np.array_equal(mask, scene.lesion_mask)


def test_teeth_stain_browns_the_teeth() -> None:
    clean = generate_wl_scene(np.random.default_rng(16), MEDIUM, site="lower_gingiva")
    stained = generate_wl_scene(
        np.random.default_rng(16), MEDIUM, site="lower_gingiva", teeth_stain=0.8
    )
    t = clean.teeth_mask & stained.teeth_mask
    blue_drop = clean.wl[t][:, 2].mean() - stained.wl[t][:, 2].mean()
    assert blue_drop > 0.02


def test_rejects_tiny_frames() -> None:
    with pytest.raises(ValueError, match="at least"):
        generate_wl_scene(np.random.default_rng(0), (32, 32), site="hard_palate")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"kind": "growth"},
        {"kind": "ulcer", "center": (1.2, 0.5)},
        {"kind": "ulcer", "radii": (0.0, 0.1)},
        {"kind": "ulcer", "radii": (0.5, 0.1)},
        {"kind": "ulcer", "irregularity": 1.5},
        {"kind": "ulcer", "fvl_depth": -0.1},
    ],
)
def test_lesion_spec_validation(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        LesionSpec(**kwargs)


def test_lesion_patch_shape_and_painter_guard() -> None:
    rng = np.random.default_rng(17)
    spec = LesionSpec(kind="ulcer", radii=(0.1, 0.08), irregularity=0.0)
    patch = make_lesion_patch(spec, MEDIUM, rng)
    size = 2 * patch.radius_px + 1
    assert patch.rho.shape == (size, size) and patch.mask.any()
    assert patch.reach_px > patch.mask.sum() ** 0.5 / 2, "ulcer halo reaches past the lesion"
    assert (patch.paint >= patch.alpha - 1e-6).all()
    with pytest.raises(ValueError, match="same shape"):
        paint_lesion(np.zeros((5, 5, 3), np.float32), patch, spec, rng)
