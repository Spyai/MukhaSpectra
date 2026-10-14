"""Tests for confounder flagging (porphyrin red, pigmentation, blood, glare)."""

from __future__ import annotations

import copy

import numpy as np
import pytest
from af_helpers import BLOOD_RGB, BROWN_RGB, disc, make_af_scene, paint, pink_wl, plain_masks

from mukha_pipeline.af import (
    compute_af_maps,
    confounder_masks,
    detect_fvl,
    flag_confounders,
    reference_green,
    region_overlaps,
    resolve_reference,
)
from mukha_pipeline.af.confounders import NOTES
from mukha_pipeline.positioning import find_positioning_violations
from mukha_pipeline.schema.models import AFReference, ConfounderType, ReferenceType
from mukha_pipeline.sites import OralSite


@pytest.fixture
def cfg(analysis_cfg: dict) -> dict:
    return copy.deepcopy(analysis_cfg)


def analyse(scene, cfg: dict):
    maps = compute_af_maps(scene.af, None)
    ref = resolve_reference(
        OralSite.TONGUE_DORSUM, reference_green(maps, scene.mucosa, cfg["af"]), {}
    )
    return detect_fvl(maps, scene.mucosa, ref, cfg["af"]), maps


def types_of(out) -> set[ConfounderType]:
    return {c.type for r in out.analysis.regions for c in r.confounders}


def test_porphyrin_red_over_fvl_is_flagged(rng, cfg: dict) -> None:
    sc = make_af_scene(rng, porphyrin_disc=(0.45, 0.4, 0.07))
    fvl, maps = analyse(sc, cfg)
    out = flag_confounders(fvl, pink_wl(sc.mucosa.shape), maps, plain_masks(sc.mucosa), cfg)
    flags = out.analysis.regions[0].confounders
    assert [f.type for f in flags] == [ConfounderType.PORPHYRIN_RED]
    assert 0.25 < flags[0].overlap_fraction < 0.6
    assert "plaque or bacteria" in flags[0].note
    assert 0.0 < out.analysis.porphyrin_red_fraction < 0.05
    # Input is not modified and arrays are carried over.
    assert fvl.analysis.regions[0].confounders == []
    assert fvl.analysis.porphyrin_red_fraction == 0.0
    assert out.labels is fvl.labels


def test_clean_scene_has_no_confounders(rng, cfg: dict) -> None:
    sc = make_af_scene(rng)
    fvl, maps = analyse(sc, cfg)
    out = flag_confounders(fvl, pink_wl(sc.mucosa.shape), maps, plain_masks(sc.mucosa), cfg)
    assert len(out.analysis.regions) == 1
    assert types_of(out) == set()
    assert out.analysis.porphyrin_red_fraction == 0.0


@pytest.mark.parametrize(
    ("rgb", "expected"),
    [(BROWN_RGB, ConfounderType.PIGMENTATION), (BLOOD_RGB, ConfounderType.BLOOD)],
)
def test_wl_pigment_and_blood_are_flagged(rng, cfg: dict, rgb, expected) -> None:
    sc = make_af_scene(rng)
    fvl, maps = analyse(sc, cfg)
    wl = pink_wl(sc.mucosa.shape)
    h, w = sc.mucosa.shape
    paint(wl, disc((h, w), 0.45 * h, 0.4 * w, 0.11 * h), rgb)
    out = flag_confounders(fvl, wl, maps, plain_masks(sc.mucosa), cfg)
    assert types_of(out) == {expected}
    flag = out.analysis.regions[0].confounders[0]
    assert flag.overlap_fraction > 0.8


def test_blood_is_not_double_counted_as_pigment(cfg: dict) -> None:
    wl = np.empty((10, 10, 3), np.float32)
    wl[:] = BLOOD_RGB
    maps = compute_af_maps(np.full((10, 10, 3), 0.3, np.float32), None)
    m = confounder_masks(wl, maps, plain_masks(np.ones((10, 10), bool)), cfg["confounders"])
    assert m.blood.all() and not m.pigmentation.any()


def test_glare_holes_inside_region_are_flagged(rng, cfg: dict) -> None:
    sc = make_af_scene(rng)
    h, w = sc.mucosa.shape
    glare = np.zeros((h, w), bool)
    for dy in (-20, 0, 20):
        for dx in (-20, 0, 20):
            glare |= disc((h, w), 0.45 * h + dy, 0.4 * w + dx, 6)
    mucosa = sc.mucosa & ~glare  # the segmenter cuts highlights out of the mucosa
    maps = compute_af_maps(sc.af, None)
    ref = AFReference(type=ReferenceType.IN_IMAGE_AUTO, reference_green=sc.base_green)
    fvl = detect_fvl(maps, mucosa, ref, cfg["af"])
    out = flag_confounders(fvl, pink_wl((h, w)), maps, plain_masks(mucosa, glare), cfg)
    assert ConfounderType.GLARE in types_of(out)


def test_saturated_af_counts_as_glare(cfg: dict) -> None:
    af = np.full((20, 20, 3), 0.3, np.float32)
    af[5:8, 5:8, 1] = 1.0
    maps = compute_af_maps(af, None)
    m = confounder_masks(pink_wl((20, 20)), maps, plain_masks(np.ones((20, 20), bool)), {})
    assert m.glare[5:8, 5:8].all() and m.glare.sum() == 9


def test_region_overlaps_empty_region() -> None:
    z = np.zeros((5, 5), bool)
    maps = compute_af_maps(np.full((5, 5, 3), 0.3, np.float32), None)
    conf = confounder_masks(pink_wl((5, 5)), maps, plain_masks(~z), {})
    assert set(region_overlaps(z, conf, 2).values()) == {0.0}
    assert region_overlaps(~z, conf, 0)[ConfounderType.GLARE] == 0.0


def test_accepts_section_only_config_and_fallback_region_ids(rng, cfg: dict) -> None:
    sc = make_af_scene(rng, porphyrin_disc=(0.45, 0.4, 0.07))
    fvl, maps = analyse(sc, cfg)
    odd = fvl.analysis.model_copy(
        update={"regions": [fvl.analysis.regions[0].model_copy(update={"region_id": "x"})]}
    )
    fvl = type(fvl)(analysis=odd, relative_af=fvl.relative_af, labels=fvl.labels)
    out = flag_confounders(
        fvl, pink_wl(sc.mucosa.shape), maps, plain_masks(sc.mucosa), cfg["confounders"]
    )
    assert types_of(out) == {ConfounderType.PORPHYRIN_RED}
    out_none = flag_confounders(fvl, pink_wl(sc.mucosa.shape), maps, plain_masks(sc.mucosa), None)
    assert types_of(out_none) == {ConfounderType.PORPHYRIN_RED}


def test_porphyrin_fraction_reported_without_regions(rng, cfg: dict) -> None:
    # Thick plaque on normal tissue: red emission above the green glow.
    sc = make_af_scene(rng, fvl_discs=(), porphyrin_disc=(0.5, 0.5, 0.1), porphyrin_red=0.5)
    fvl, maps = analyse(sc, cfg)
    assert fvl.analysis.regions == []
    out = flag_confounders(fvl, pink_wl(sc.mucosa.shape), maps, plain_masks(sc.mucosa), cfg)
    assert out.analysis.porphyrin_red_fraction > 0.01


def test_empty_mucosa_and_shape_errors(cfg: dict) -> None:
    from mukha_pipeline.af import empty_fvl_output

    shape = (12, 12)
    maps = compute_af_maps(np.full((*shape, 3), 0.3, np.float32), None)
    fvl = empty_fvl_output(
        shape, None, status="insufficient_mucosa", reference=AFReference(type=ReferenceType.NONE)
    )
    out = flag_confounders(fvl, pink_wl(shape), maps, plain_masks(np.zeros(shape, bool)), cfg)
    assert out.analysis.porphyrin_red_fraction == 0.0
    with pytest.raises(ValueError):
        flag_confounders(fvl, pink_wl((5, 5)), maps, plain_masks(np.zeros(shape, bool)), cfg)
    bad = type(fvl)(analysis=fvl.analysis, relative_af=fvl.relative_af, labels=np.zeros((3, 3)))
    with pytest.raises(ValueError):
        flag_confounders(bad, pink_wl(shape), maps, plain_masks(np.zeros(shape, bool)), cfg)


def test_notes_respect_positioning() -> None:
    for note in NOTES.values():
        assert not find_positioning_violations(note)
        assert "may" in note or "could not" in note  # never states a cause as fact
