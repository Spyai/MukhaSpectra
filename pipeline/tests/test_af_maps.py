"""Tests for AF maps, the in-image reference level and reference resolution."""

from __future__ import annotations

import copy

import numpy as np
import pytest
from af_helpers import make_af_scene, uniform_maps

from mukha_pipeline.af import compute_af_maps, reference_green, resolve_reference
from mukha_pipeline.positioning import find_positioning_violations
from mukha_pipeline.schema.models import ReferenceType
from mukha_pipeline.sites import OralSite


@pytest.fixture
def af_cfg(analysis_cfg: dict) -> dict:
    return copy.deepcopy(analysis_cfg["af"])


# ------------------------------------------------------------------------------------------
# compute_af_maps
# ------------------------------------------------------------------------------------------


def test_maps_values_scale_and_valid_mask() -> None:
    img = np.zeros((4, 5, 3), np.float32)
    img[..., 0], img[..., 1], img[..., 2] = 0.1, 0.4, 0.05
    valid = np.ones((4, 5), bool)
    valid[0, 0] = False
    maps = compute_af_maps(img, valid, intensity_scale=2.0)
    assert maps.green[1, 1] == pytest.approx(0.8)
    assert maps.red[1, 1] == pytest.approx(0.2)
    assert maps.blue[1, 1] == pytest.approx(0.1)
    assert maps.gr_ratio[1, 1] == pytest.approx(0.8 / 0.201, rel=1e-4)
    assert maps.af_index[1, 1] == pytest.approx(0.8 / 1.101, rel=1e-4)
    for arr in (maps.green, maps.red, maps.blue, maps.gr_ratio, maps.af_index):
        assert arr.dtype == np.float32 and np.isnan(arr[0, 0])
        assert np.isfinite(arr[1:, 1:]).all()


def test_maps_handle_nan_negative_and_bad_scale() -> None:
    img = np.full((3, 3, 3), 0.2, np.float32)
    img[0, 0, 1] = np.nan
    img[1, 1, :] = -0.05  # over-subtracted dark frame
    for scale in (float("nan"), 0.0, -3.0):
        maps = compute_af_maps(img, None, intensity_scale=scale)
        assert np.isnan(maps.green[0, 0]) and np.isnan(maps.red[0, 0])
        assert maps.green[1, 1] == 0.0
        assert maps.green[2, 2] == pytest.approx(0.2)  # scale fell back to 1.0
        assert maps.gr_ratio[1, 1] == 0.0


def test_maps_accept_gray_and_rgba_and_reject_bad_shapes() -> None:
    assert compute_af_maps(np.full((3, 3), 0.3, np.float32), None).green[0, 0] == pytest.approx(0.3)
    rgba = np.full((3, 3, 4), 0.3, np.float32)
    assert compute_af_maps(rgba, None).green.shape == (3, 3)
    with pytest.raises(ValueError):
        compute_af_maps(np.zeros((3, 3, 2), np.float32), None)
    with pytest.raises(ValueError):
        compute_af_maps(np.zeros((3, 3, 3), np.float32), np.ones((2, 2), bool))


# ------------------------------------------------------------------------------------------
# reference_green
# ------------------------------------------------------------------------------------------


def test_reference_green_is_robust_to_fvl_and_glare(rng, af_cfg: dict) -> None:
    clean = make_af_scene(rng, fvl_discs=())
    lesion = make_af_scene(rng, fvl_discs=((0.45, 0.4, 0.2, 0.6),))  # ~14% of mucosa
    ref_clean = reference_green(compute_af_maps(clean.af, None), clean.mucosa, af_cfg)
    assert ref_clean == pytest.approx(clean.base_green, rel=0.03)

    maps = compute_af_maps(lesion.af, None)
    plain_mean = float(np.nanmean(maps.green[lesion.mucosa]))
    ref_fvl = reference_green(maps, lesion.mucosa, af_cfg)
    assert plain_mean < 0.96 * ref_clean  # a naive mean is dragged down by the dark disc
    assert ref_fvl == pytest.approx(ref_clean, rel=0.03)

    stripes = lesion.mucosa & (np.arange(maps.green.shape[1]) % 37 == 0)[None, :]
    maps.green[stripes] = 1.0  # glare-like saturated lines (~3% of mucosa)
    assert reference_green(maps, lesion.mucosa, af_cfg) == pytest.approx(ref_clean, rel=0.04)


def test_reference_green_none_for_tiny_or_dark_mucosa(af_cfg: dict) -> None:
    maps = uniform_maps((40, 40))
    tiny = np.zeros((40, 40), bool)
    tiny[:10, :10] = True  # 100 px < 200
    assert reference_green(maps, tiny, af_cfg) is None
    dark = uniform_maps((40, 40), green=0.0)
    assert reference_green(dark, np.ones((40, 40), bool), af_cfg) is None
    nan_maps = uniform_maps((40, 40))
    nan_maps.green[:] = np.nan
    assert reference_green(nan_maps, np.ones((40, 40), bool), af_cfg) is None
    assert reference_green(maps, np.ones((40, 40), bool), af_cfg) == pytest.approx(0.4)


def test_reference_green_swapped_percentiles_and_defaults() -> None:
    maps = uniform_maps((30, 30), green=0.3)
    m = np.ones((30, 30), bool)
    cfg = {"reference_percentile_low": 90, "reference_percentile_high": 25}
    assert reference_green(maps, m, cfg) == pytest.approx(0.3)
    assert reference_green(maps, m, None) == pytest.approx(0.3)


# ------------------------------------------------------------------------------------------
# resolve_reference
# ------------------------------------------------------------------------------------------


def test_contralateral_wins() -> None:
    ref = resolve_reference(
        OralSite.BUCCAL_MUCOSA_LEFT,
        0.40,
        {OralSite.BUCCAL_MUCOSA_RIGHT: 0.42, OralSite.TONGUE_DORSUM: 0.5},
    )
    assert ref.type == ReferenceType.CONTRALATERAL
    assert ref.reference_site == OralSite.BUCCAL_MUCOSA_RIGHT
    assert ref.reference_green == pytest.approx(0.42)
    assert ref.note == "Compared with the patient's own right buccal mucosa."


def test_analogous_site() -> None:
    ref = resolve_reference("upper_lip_mucosa", 0.3, {OralSite.LOWER_LIP_MUCOSA: 0.35})
    assert ref.type == ReferenceType.ANALOGOUS_SITE
    assert ref.reference_site == OralSite.LOWER_LIP_MUCOSA
    assert "lower lip mucosa" in (ref.note or "")


def test_unpaired_site_uses_in_image() -> None:
    ref = resolve_reference(OralSite.TONGUE_DORSUM, 0.33, {OralSite.BUCCAL_MUCOSA_LEFT: 0.4})
    assert ref.type == ReferenceType.IN_IMAGE_AUTO
    assert ref.reference_site is None
    assert ref.reference_green == pytest.approx(0.33)
    assert ref.note == (
        "No paired site captured; compared with the brightest healthy-looking tissue in the same "
        "image — less reliable."
    )


def test_paired_site_missing_or_unusable_falls_back_to_in_image() -> None:
    for available in (
        {},
        {OralSite.BUCCAL_MUCOSA_RIGHT: float("nan")},
        {OralSite.BUCCAL_MUCOSA_RIGHT: 0.0},
        {OralSite.BUCCAL_MUCOSA_RIGHT: None},
    ):
        ref = resolve_reference(OralSite.BUCCAL_MUCOSA_LEFT, 0.4, available)  # type: ignore[arg-type]
        assert ref.type == ReferenceType.IN_IMAGE_AUTO
        assert "right buccal mucosa" in (ref.note or "")


def test_much_dimmer_paired_site_is_not_used() -> None:
    """Dividing by a too-dim reference inflates relative AF and could hide a real loss."""
    ref = resolve_reference(
        OralSite.LATERAL_TONGUE_RIGHT, 0.40, {OralSite.LATERAL_TONGUE_LEFT: 0.20}
    )
    assert ref.type == ReferenceType.IN_IMAGE_AUTO
    assert "dimmer" in (ref.note or "")
    brighter = resolve_reference(
        OralSite.LATERAL_TONGUE_RIGHT, 0.20, {OralSite.LATERAL_TONGUE_LEFT: 0.40}
    )
    assert brighter.type == ReferenceType.CONTRALATERAL
    lenient = resolve_reference(
        OralSite.LATERAL_TONGUE_RIGHT,
        0.40,
        {OralSite.LATERAL_TONGUE_LEFT: 0.20},
        min_paired_ratio=0.4,
    )
    assert lenient.type == ReferenceType.CONTRALATERAL


def test_paired_site_used_when_own_reference_missing() -> None:
    ref = resolve_reference(OralSite.RETROMOLAR_LEFT, None, {OralSite.RETROMOLAR_RIGHT: 0.3})
    assert ref.type == ReferenceType.CONTRALATERAL


def test_no_reference() -> None:
    ref = resolve_reference(OralSite.HARD_PALATE, None, {})
    assert ref.type == ReferenceType.NONE
    assert ref.reference_green is None
    assert resolve_reference(OralSite.HARD_PALATE, float("inf"), {}).type == ReferenceType.NONE
    assert resolve_reference(OralSite.HARD_PALATE, "x", {}).type == ReferenceType.NONE  # type: ignore[arg-type]


def test_reference_notes_respect_positioning() -> None:
    notes = [
        resolve_reference(OralSite.BUCCAL_MUCOSA_LEFT, 0.4, {OralSite.BUCCAL_MUCOSA_RIGHT: 0.4}),
        resolve_reference(OralSite.UPPER_LIP_MUCOSA, 0.4, {OralSite.LOWER_LIP_MUCOSA: 0.4}),
        resolve_reference(OralSite.TONGUE_DORSUM, 0.4, {}),
        resolve_reference(OralSite.BUCCAL_MUCOSA_LEFT, 0.4, {}),
        resolve_reference(OralSite.BUCCAL_MUCOSA_LEFT, 0.4, {OralSite.BUCCAL_MUCOSA_RIGHT: 0.1}),
        resolve_reference(OralSite.TONGUE_DORSUM, None, {}),
    ]
    for ref in notes:
        assert ref.note and not find_positioning_violations(ref.note)
