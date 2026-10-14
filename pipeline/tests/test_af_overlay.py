"""Tests for the AF-loss heatmap overlay and the relative-AF grid."""

from __future__ import annotations

import copy
import json

import numpy as np
import pytest
from af_helpers import disc, make_af_scene, make_wl_scene, pink_wl

from mukha_pipeline.af import (
    af_loss,
    block_nanmean,
    compute_af_maps,
    detect_fvl,
    heatmap_legend,
    loss_colormap,
    reference_green,
    relative_af_grid,
    render_heatmap_overlay,
    resolve_reference,
)
from mukha_pipeline.af.overlay import LOSS_COLORMAP_STOPS
from mukha_pipeline.sites import OralSite


@pytest.fixture
def af_cfg(analysis_cfg: dict) -> dict:
    return copy.deepcopy(analysis_cfg["af"])


@pytest.fixture
def fvl_scene(rng, af_cfg: dict):
    sc = make_af_scene(rng)
    maps = compute_af_maps(sc.af, None)
    ref = resolve_reference(OralSite.TONGUE_DORSUM, reference_green(maps, sc.mucosa, af_cfg), {})
    return sc, detect_fvl(maps, sc.mucosa, ref, af_cfg)


def luminance(rgb: np.ndarray) -> np.ndarray:
    return rgb @ np.array([0.299, 0.587, 0.114])


def test_overlay_shape_dtype_and_tint(fvl_scene, af_cfg: dict) -> None:
    sc, out = fvl_scene
    wl = pink_wl(sc.mucosa.shape)
    img = render_heatmap_overlay(wl, out.relative_af, out.labels, 0.75, af_cfg)
    assert img.dtype == np.uint8 and img.shape == wl.shape
    wl8 = np.round(wl * 255).astype(np.int16)
    core = disc(sc.mucosa.shape, 0.45 * 360, 0.4 * 480, 25)  # well inside the FVL disc
    diff = np.abs(img.astype(np.int16) - wl8).sum(axis=2)
    assert diff[core].mean() > 40  # FVL pixels are clearly tinted
    healthy = sc.mucosa & ~disc(sc.mucosa.shape, 0.45 * 360, 0.4 * 480, 0.2 * 360)
    assert np.median(diff[healthy]) <= 3  # normal tissue is left untouched
    # Region outline: white core pixels appear around the disc.
    assert (img >= 240).all(axis=2).sum() > 100


def test_overlay_greys_out_unanalysed_area(fvl_scene, af_cfg: dict) -> None:
    sc, out = fvl_scene
    wl = pink_wl(sc.mucosa.shape)
    img = render_heatmap_overlay(wl, out.relative_af, out.labels, 0.75, af_cfg).astype(float)
    outside = ~sc.mucosa
    sat_in = np.ptp(img[sc.mucosa & ~(out.labels > 0)], axis=1).mean()
    sat_out = np.ptp(img[outside], axis=1).mean()
    assert sat_out < 0.8 * sat_in
    af_cfg["heatmap_unanalysed_desaturate"] = 0.0
    plain = render_heatmap_overlay(wl, out.relative_af, out.labels, 0.75, af_cfg)
    assert np.abs(plain[outside].astype(int) - np.round(wl[outside] * 255)).max() <= 1


def test_overlay_handles_uint8_wl_nan_and_no_labels(af_cfg: dict) -> None:
    sc = make_wl_scene(np.random.default_rng(0))
    wl8 = (sc.wl * 255).astype(np.uint8)
    rel = np.full(sc.wl.shape[:2], np.nan, np.float32)
    img = render_heatmap_overlay(wl8, rel, None, 0.75, af_cfg)
    assert img.shape == wl8.shape
    assert np.abs(img.astype(int) - wl8).max() <= 1  # nothing analysed -> image unchanged
    img2 = render_heatmap_overlay(sc.wl, rel, None, float("nan"), {"af": af_cfg})
    assert img2.dtype == np.uint8


def test_overlay_small_image_skips_labels(af_cfg: dict) -> None:
    rel = np.ones((30, 30), np.float32)
    rel[10:20, 10:20] = 0.3
    labels = np.zeros((30, 30), np.int32)
    labels[10:20, 10:20] = 1
    img = render_heatmap_overlay(pink_wl((30, 30)), rel, labels, 0.75, None)
    assert img.shape == (30, 30, 3)


def test_overlay_label_near_top_edge_goes_below(af_cfg: dict) -> None:
    rel = np.ones((200, 200), np.float32)
    rel[0:30, 80:120] = 0.3
    labels = np.zeros((200, 200), np.int32)
    labels[0:30, 80:120] = 1
    labels[150:160, 0:5] = 3  # id gap: label 2 is absent
    img = render_heatmap_overlay(pink_wl((200, 200)), rel, labels, 0.75, af_cfg)
    dark_box = (img.astype(int).sum(axis=2) < 150)[30:60, 70:140]
    assert dark_box.any()


def test_overlay_shape_errors() -> None:
    with pytest.raises(ValueError):
        render_heatmap_overlay(pink_wl((10, 10)), np.ones((5, 5)), None, 0.75, None)
    with pytest.raises(ValueError):
        render_heatmap_overlay(pink_wl((10, 10)), np.ones((10, 10)), np.zeros((5, 5)), 0.75, None)
    with pytest.raises(ValueError):
        render_heatmap_overlay(np.zeros((10, 10, 2)), np.ones((10, 10)), None, 0.75, None)


def test_loss_and_colormap_are_ordered() -> None:
    rel = np.array([1.0, 0.85, 0.75, 0.5, 0.2, 0.0, np.nan], np.float32)
    loss = af_loss(rel, 0.75)
    assert loss[0] == 0 and loss[1] == 0 and loss[-1] == 0
    assert loss[2] == pytest.approx(0.1 / 0.65, rel=1e-4)
    assert loss[4] == loss[5] == 1.0
    assert np.all(np.diff(loss[:6]) >= 0)
    cols = loss_colormap(np.linspace(0, 1, 50))
    assert cols.shape == (50, 3)
    lum = luminance(cols)
    assert np.all(np.diff(lum) < 0)  # lightness falls monotonically with loss
    assert np.allclose(loss_colormap(np.array([0.0]))[0], LOSS_COLORMAP_STOPS[0][1])


def test_heatmap_legend() -> None:
    legend = heatmap_legend(0.75, n=5)
    assert len(legend) == 5
    assert legend[0]["relative_af"] == pytest.approx(0.85)
    assert legend[-1]["relative_af"] == pytest.approx(0.2)
    assert all(str(e["color"]).startswith("#") and len(str(e["color"])) == 7 for e in legend)


# ------------------------------------------------------------------------------------------
# relative_af_grid
# ------------------------------------------------------------------------------------------


def test_grid_size_and_values(fvl_scene) -> None:
    _, out = fvl_scene
    g = relative_af_grid(out.relative_af, 96, threshold=0.75)
    assert max(g["width"], g["height"]) <= 96
    assert g["width"] == 96 and g["height"] == 72
    assert len(g["values"]) == g["height"] and all(len(r) == g["width"] for r in g["values"])
    assert g["values"][0][0] is None  # corner is outside the mucosa
    assert 0.4 < g["min"] < 0.6 and 0.95 < g["max"] < 1.2
    assert g["threshold"] == 0.75 and g["block_px"] == 5
    text = json.dumps(g, allow_nan=False)
    assert "NaN" not in text
    vals = [v for row in g["values"] for v in row if v is not None]
    assert all(round(v, 3) == v for v in vals)


def test_grid_never_upsamples_and_handles_all_nan() -> None:
    small = np.ones((10, 20), np.float32)
    g = relative_af_grid(small, 96)
    assert (g["width"], g["height"], g["block_px"]) == (20, 10, 1)
    assert g["threshold"] is None
    empty = relative_af_grid(np.full((50, 50), np.nan), 10)
    assert empty["min"] is None and empty["max"] is None
    assert all(v is None for row in empty["values"] for v in row)
    assert relative_af_grid(np.zeros((0, 0)), 10)["values"] == []
    assert relative_af_grid(np.zeros((3, 3, 3)), 10, threshold=0.7)["threshold"] == 0.7
    inf = np.ones((4, 4))
    inf[0, 0] = np.inf
    assert relative_af_grid(inf, 2, threshold=0.75)["values"][0][0] == 1.0
    assert relative_af_grid(np.ones((7, 7)), 0)["width"] == 1


def test_block_nanmean_partial_blocks() -> None:
    arr = np.array([[1.0, 3.0, 5.0], [np.nan, np.nan, 7.0]])
    out = block_nanmean(arr, 2)
    assert out.shape == (1, 2)
    assert out[0, 0] == pytest.approx(2.0) and out[0, 1] == pytest.approx(6.0)
