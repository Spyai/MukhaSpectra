"""Fluorescence maps and the robust in-image reference level.

Physics: under 405 nm excitation, healthy oral mucosa fluoresces green (collagen cross-links in
the stroma, NADH and FAD in the epithelium). Dysplasia thickens the epithelium, breaks down
stromal collagen and shifts metabolism, so the green glow fades: fluorescence visualisation loss
(FVL). Bacterial porphyrins (plaque, coating) add red emission near 635 nm. The green channel is
therefore the primary signal; green/red and green/(R+G+B) are kept as secondary maps because
they are less sensitive to illumination fall-off.
"""

from __future__ import annotations

import numpy as np

from mukha_pipeline.af._common import AF_DEFAULTS, EPS, as_bool_mask, get, masked_smooth
from mukha_pipeline.types import AFMaps


def compute_af_maps(
    af_registered: np.ndarray, valid: np.ndarray | None, *, intensity_scale: float = 1.0
) -> AFMaps:
    """Per-pixel AF maps in the WL frame, NaN outside ``valid`` (or where the input is NaN).

    ``intensity_scale`` = exposure factor x calibration af_scale, so values from different
    captures and devices are comparable. A non-finite or non-positive scale falls back to 1.0.
    Negative values (over-subtracted dark frame) are clipped to 0.
    """
    img = np.asarray(af_registered, dtype=np.float32)
    if img.ndim == 2:
        img = np.repeat(img[:, :, None], 3, axis=2)
    if img.ndim != 3 or img.shape[2] < 3:
        raise ValueError(f"expected an RGB AF image (H, W, 3), got shape {img.shape}")
    img = img[:, :, :3]
    shape = img.shape[:2]
    valid_m = as_bool_mask(valid, shape, "valid")
    scale = float(intensity_scale) if np.isfinite(intensity_scale) else 1.0
    if scale <= 0:
        scale = 1.0
    finite = np.isfinite(img).all(axis=2) & valid_m
    chans = np.clip(np.where(np.isfinite(img), img, 0.0), 0.0, None) * scale
    r, g, b = (np.where(finite, chans[:, :, i], np.nan).astype(np.float32) for i in range(3))
    with np.errstate(invalid="ignore", divide="ignore"):
        gr_ratio = (g / (r + EPS)).astype(np.float32)
        af_index = (g / (r + g + b + EPS)).astype(np.float32)
    return AFMaps(green=g, red=r, blue=b, gr_ratio=gr_ratio, af_index=af_index)


def reference_green(maps: AFMaps, mucosa: np.ndarray, cfg: dict | None) -> float | None:
    """Robust "normal tissue" green level of this image (cfg = analysis.af).

    Mean of the smoothed mucosa green between two percentiles (default 50th-95th). The estimate
    comes from the upper half of the distribution because a health worker centres the lesion in
    a close-up: fluorescence loss can cover up to about half of the mucosa, and a lower cut
    below that share would let the loss drag the reference down (depth under-reported, moderate
    losses missed). Glare or teeth fluorescence sits above the upper cut. Erring high only makes
    the analysis more sensitive. Smoothing matches what :func:`detect_fvl` divides by, so normal
    tissue lands at a relative AF close to 1. Returns None when there is too little usable mucosa.

    Because the estimate comes from the bright half, isolated AF highlights (glare in the
    AF frame that the WL specular mask does not cover) are set aside first: pixels brighter than
    ``reference_outlier_ratio`` x the 90th percentile of mucosa green. The cut is relative to a
    high percentile, so even a loss covering most of the field never removes healthy tissue.
    """
    shape = maps.green.shape
    m = as_bool_mask(mucosa, shape, "mucosa") & np.isfinite(maps.green)
    min_px = int(get(cfg, "reference_min_pixels", AF_DEFAULTS))
    if int(m.sum()) < min_px:
        return None
    outlier = float(get(cfg, "reference_outlier_ratio", AF_DEFAULTS))
    if outlier > 0:
        p90 = float(np.percentile(maps.green[m], 90))
        if np.isfinite(p90) and p90 > EPS:
            m &= maps.green <= outlier * p90
    sigma = float(get(cfg, "smoothing_sigma_px", AF_DEFAULTS))
    vals = masked_smooth(maps.green, m, sigma)[m]
    vals = vals[np.isfinite(vals)]
    if vals.size < min_px:
        return None
    lo = float(get(cfg, "reference_percentile_low", AF_DEFAULTS))
    hi = float(get(cfg, "reference_percentile_high", AF_DEFAULTS))
    lo, hi = sorted((float(np.clip(lo, 0, 100)), float(np.clip(hi, 0, 100))))
    p_lo, p_hi = np.percentile(vals, [lo, hi])
    sel = vals[(vals >= p_lo) & (vals <= p_hi)]
    ref = float(sel.mean()) if sel.size else float(np.median(vals))
    return ref if np.isfinite(ref) and ref > EPS else None
