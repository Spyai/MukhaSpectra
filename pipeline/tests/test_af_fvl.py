"""Tests for FVL detection and per-region measurements (SYNTHETIC AF scenes)."""

from __future__ import annotations

import copy

import numpy as np
import pytest
from af_helpers import disc, make_af_scene, uniform_maps

from mukha_pipeline.af import (
    compute_af_maps,
    detect_fvl,
    effective_min_area,
    empty_fvl_output,
    reference_green,
    resolve_reference,
)
from mukha_pipeline.schema.models import AFReference, ReferenceType
from mukha_pipeline.sites import OralSite


@pytest.fixture
def af_cfg(analysis_cfg: dict) -> dict:
    return copy.deepcopy(analysis_cfg["af"])


def run(scene, cfg: dict, *, site=OralSite.TONGUE_DORSUM, px_per_mm=None, mucosa=None):
    maps = compute_af_maps(scene.af, None)
    muc = scene.mucosa if mucosa is None else mucosa
    ref = resolve_reference(site, reference_green(maps, muc, cfg), {})
    return detect_fvl(maps, muc, ref, cfg, px_per_mm=px_per_mm), maps


def test_single_fvl_disc_found_at_the_right_place(rng, af_cfg: dict) -> None:
    sc = make_af_scene(rng, fvl_discs=((0.45, 0.4, 0.11, 0.5),))
    out, _ = run(sc, af_cfg, px_per_mm=10.0)
    a = out.analysis
    assert a.status == "ok"
    assert len(a.regions) == 1
    r = a.regions[0]
    assert r.region_id == "r1"
    ty, tx = np.argwhere(sc.fvl).mean(axis=0) / np.array(sc.fvl.shape)
    assert abs(r.centroid.x - tx) < 0.03 and abs(r.centroid.y - ty) < 0.03
    assert r.area_px == pytest.approx(int(sc.fvl.sum()), rel=0.25)
    assert r.area_mm2 == pytest.approx(r.area_px / 100.0)
    assert r.mean_relative_af == pytest.approx(1 - sc.depth, abs=0.06)
    assert r.min_relative_af <= r.mean_relative_af
    assert r.contrast_vs_surround > 1.6
    assert r.border_irregularity < 0.2  # a disc is round
    assert r.solidity > 0.95
    assert r.bbox.x <= r.centroid.x <= r.bbox.x + r.bbox.w
    assert r.bbox.y <= r.centroid.y <= r.bbox.y + r.bbox.h
    # labels and relative AF are consistent with the regions
    assert out.labels.dtype == np.int32 and set(np.unique(out.labels)) == {0, 1}
    assert int((out.labels == 1).sum()) == r.area_px
    assert out.relative_af.dtype == np.float32
    assert np.isnan(out.relative_af[~sc.mucosa]).all()
    assert np.isfinite(out.relative_af[sc.mucosa]).mean() > 0.95  # all but a thin edge rim
    assert a.fvl_area_fraction == pytest.approx(r.area_fraction_of_mucosa)
    assert a.site_medians is not None
    assert a.site_medians.relative_af == pytest.approx(1.0, abs=0.05)
    assert a.site_medians.green == pytest.approx(sc.base_green, rel=0.05)
    assert a.fvl_threshold == af_cfg["fvl_relative_threshold"]
    analysed = int(np.isfinite(out.relative_af).sum())
    assert a.min_region_area_px == effective_min_area(af_cfg, analysed)
    assert a.reference.type == ReferenceType.IN_IMAGE_AUTO


def test_uniform_scene_has_no_fvl(rng, af_cfg: dict) -> None:
    out, _ = run(make_af_scene(rng, fvl_discs=()), af_cfg)
    assert out.analysis.status == "ok"
    assert out.analysis.regions == []
    assert out.analysis.fvl_area_fraction == 0.0
    assert not out.labels.any()


def test_regions_ordered_by_severity(rng, af_cfg: dict) -> None:
    sc = make_af_scene(
        rng,
        fvl_discs=((0.35, 0.3, 0.09, 0.35), (0.6, 0.7, 0.07, 0.6), (0.65, 0.3, 0.1, 0.35)),
    )
    out, _ = run(sc, af_cfg)
    regs = out.analysis.regions
    assert [r.region_id for r in regs] == ["r1", "r2", "r3"]
    assert regs[0].centroid.x > 0.6  # the deepest disc comes first
    means = [r.mean_relative_af for r in regs]
    assert means == sorted(means)
    for k, r in enumerate(regs, start=1):
        assert int((out.labels == k).sum()) == r.area_px


def test_equal_depth_ties_break_by_area(af_cfg: dict) -> None:
    shape = (200, 300)
    maps = uniform_maps(shape, 0.4)
    small, big = disc(shape, 100, 70, 15), disc(shape, 100, 200, 30)
    maps.green[small | big] = 0.2
    ref = AFReference(type=ReferenceType.IN_IMAGE_AUTO, reference_green=0.4)
    regs = detect_fvl(maps, np.ones(shape, bool), ref, af_cfg).analysis.regions
    assert len(regs) == 2
    assert regs[0].area_px > regs[1].area_px


def test_small_regions_below_min_area_are_dropped(rng, af_cfg: dict) -> None:
    sc = make_af_scene(rng, fvl_discs=((0.45, 0.4, 0.02, 0.6),))  # r ~ 7 px, ~150 px area
    af_cfg["min_region_area_px"] = 400
    out, _ = run(sc, af_cfg)
    assert out.analysis.regions == []
    assert out.analysis.min_region_area_px == 400


def test_irregular_region_has_higher_border_irregularity(af_cfg: dict) -> None:
    shape = (240, 240)
    maps = uniform_maps(shape, 0.4)
    yy, xx = np.mgrid[:240, :240]
    ang = np.arctan2(yy - 120, xx - 120)
    star = np.hypot(yy - 120, xx - 120) <= 50 + 28 * np.cos(7 * ang)
    maps.green[star] = 0.15
    ref = AFReference(type=ReferenceType.IN_IMAGE_AUTO, reference_green=0.4)
    r = detect_fvl(maps, np.ones(shape, bool), ref, af_cfg).analysis.regions[0]
    assert r.border_irregularity > 0.35
    assert r.solidity < 0.9


def test_insufficient_mucosa(rng, af_cfg: dict) -> None:
    sc = make_af_scene(rng)
    tiny = disc(sc.mucosa.shape, 100, 100, 8)  # << 2% of the image
    maps = compute_af_maps(sc.af, None)
    ref = AFReference(type=ReferenceType.IN_IMAGE_AUTO, reference_green=0.45)
    out = detect_fvl(maps, tiny, ref, af_cfg)
    assert out.analysis.status == "insufficient_mucosa"
    assert out.analysis.regions == []
    assert np.isnan(out.relative_af).all() and not out.labels.any()


@pytest.mark.parametrize(
    "ref",
    [
        AFReference(type=ReferenceType.NONE),
        AFReference(type=ReferenceType.IN_IMAGE_AUTO, reference_green=None),
        AFReference(type=ReferenceType.IN_IMAGE_AUTO, reference_green=0.0),
        AFReference(type=ReferenceType.CONTRALATERAL, reference_green=float("nan")),
    ],
)
def test_no_reference(rng, af_cfg: dict, ref: AFReference) -> None:
    sc = make_af_scene(rng)
    out = detect_fvl(compute_af_maps(sc.af, None), sc.mucosa, ref, af_cfg)
    assert out.analysis.status == "no_reference"
    assert out.analysis.regions == [] and out.analysis.site_medians is None
    assert np.isnan(out.relative_af).all()


def test_contralateral_reference_reveals_diffuse_loss(rng, af_cfg: dict) -> None:
    """A loss covering the whole field is invisible in-image but visible against the pair."""
    sc = make_af_scene(rng, fvl_discs=(), base_green=0.22)
    maps = compute_af_maps(sc.af, None)
    own = reference_green(maps, sc.mucosa, af_cfg)
    in_image = detect_fvl(
        maps, sc.mucosa, resolve_reference(OralSite.BUCCAL_MUCOSA_LEFT, own, {}), af_cfg
    )
    assert in_image.analysis.regions == []
    paired = resolve_reference(
        OralSite.BUCCAL_MUCOSA_LEFT, own, {OralSite.BUCCAL_MUCOSA_RIGHT: 0.45}
    )
    out = detect_fvl(maps, sc.mucosa, paired, af_cfg)
    assert out.analysis.reference.type == ReferenceType.CONTRALATERAL
    assert len(out.analysis.regions) == 1
    r = out.analysis.regions[0]
    assert r.area_fraction_of_mucosa > 0.9
    assert r.mean_relative_af == pytest.approx(0.22 / 0.45, abs=0.05)
    # No healthy surround: contrast falls back to 1 / mean relative AF.
    assert r.contrast_vs_surround == pytest.approx(1 / r.mean_relative_af, rel=0.01)


def test_robust_to_nan_pixels(rng, af_cfg: dict) -> None:
    sc = make_af_scene(rng)
    af = sc.af.copy()
    holes = rng.random(sc.mucosa.shape) < 0.02
    af[holes] = np.nan
    valid = np.ones(sc.mucosa.shape, bool)
    valid[:, :20] = False  # registration did not cover a strip
    maps = compute_af_maps(af, valid)
    ref = resolve_reference(OralSite.TONGUE_DORSUM, reference_green(maps, sc.mucosa, af_cfg), {})
    out = detect_fvl(maps, sc.mucosa, ref, af_cfg)
    assert out.analysis.status == "ok"
    assert len(out.analysis.regions) == 1
    assert np.isnan(out.relative_af[holes]).all()
    assert out.analysis.regions[0].mean_relative_af == pytest.approx(0.5, abs=0.07)


def test_zero_signal_region_caps_contrast(af_cfg: dict) -> None:
    shape = (200, 200)
    maps = uniform_maps(shape, 0.4)
    maps.green[disc(shape, 100, 100, 25)] = 0.0
    ref = AFReference(type=ReferenceType.IN_IMAGE_AUTO, reference_green=0.4)
    r = detect_fvl(maps, np.ones(shape, bool), ref, af_cfg).analysis.regions[0]
    assert r.mean_relative_af >= 0.0
    assert 1.0 < r.contrast_vs_surround <= 50.0


def test_mask_shape_mismatch_raises(af_cfg: dict) -> None:
    maps = uniform_maps((20, 20))
    ref = AFReference(type=ReferenceType.IN_IMAGE_AUTO, reference_green=0.4)
    with pytest.raises(ValueError):
        detect_fvl(maps, np.ones((10, 10), bool), ref, af_cfg)


def test_empty_fvl_output_and_defaults() -> None:
    ref = AFReference(type=ReferenceType.NONE)
    out = empty_fvl_output((5, 6), None, status="skipped_qc_failed", reference=ref)
    assert out.analysis.status == "skipped_qc_failed"
    assert out.analysis.min_region_area_px == 150
    assert out.relative_af.shape == (5, 6) and np.isnan(out.relative_af).all()
    assert out.labels.dtype == np.int32
    assert (
        effective_min_area({"min_region_area_px": 10, "min_region_area_fraction": 0.01}, 5000) == 50
    )


def test_works_with_default_config(rng) -> None:
    out, _ = run(make_af_scene(rng), {})
    assert len(out.analysis.regions) == 1


def test_result_serialises_to_json(rng, af_cfg: dict) -> None:
    out, _ = run(make_af_scene(rng), af_cfg)
    text = out.analysis.model_dump_json()
    assert "NaN" not in text and '"r1"' in text


def test_misregistered_edge_does_not_create_fvl_band(rng, af_cfg: dict) -> None:
    """AF shifted by a few pixels puts dark background inside the WL mucosa border."""
    sc = make_af_scene(rng, fvl_discs=())
    shifted = np.roll(sc.af, (3, 3), axis=(0, 1))
    maps = compute_af_maps(shifted, None)
    ref = AFReference(type=ReferenceType.IN_IMAGE_AUTO, reference_green=sc.base_green)
    out = detect_fvl(maps, sc.mucosa, ref, af_cfg)
    assert out.analysis.regions == []
    rim = sc.mucosa & ~np.isfinite(out.relative_af)
    assert 0 < rim.sum() < 0.05 * sc.mucosa.sum()  # only a thin rim is set aside
    af_cfg["edge_margin_px"] = 0
    assert len(detect_fvl(maps, sc.mucosa, ref, af_cfg).analysis.regions) >= 1
