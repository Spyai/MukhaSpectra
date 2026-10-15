"""Small numerical helpers shared by the AF modules (not part of the public API)."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

# Added to ratio denominators. About a quarter of an 8-bit quantisation step: large enough that
# a dark red channel does not explode G/R, small enough not to bias normal tissue ratios.
EPS = 1e-3

AF_DEFAULTS: dict[str, Any] = {
    "smoothing_sigma_px": 3.0,
    "fvl_relative_threshold": 0.75,
    "min_region_area_fraction": 0.002,
    "min_region_area_px": 150,
    "morphology_open_px": 5,
    "morphology_close_px": 9,
    "surround_ring_px": 15,
    "edge_margin_px": 3,
    "reference_percentile_low": 50,
    "reference_percentile_high": 95,
    "reference_outlier_ratio": 1.5,
    "reference_min_pixels": 200,
    "insufficient_mucosa_fraction": 0.02,
    "relative_af_grid_size": 96,
    "heatmap_alpha_max": 0.75,
    "heatmap_unanalysed_desaturate": 0.35,
}

CONFOUNDER_DEFAULTS: dict[str, Any] = {
    "porphyrin_red_over_green_min": 0.85,
    "porphyrin_red_rel_min": 1.4,
    "porphyrin_red_ref_min_of_green": 0.10,
    "porphyrin_smoothing_sigma_px": 2.0,
    "pigmentation_lab_l_max": 45.0,
    "pigmentation_lab_a_max": 20.0,
    "pigmentation_hue_range": [5, 30],
    "pigmentation_saturation_min": 0.15,
    "blood_lab_a_min": 25.0,
    "blood_lab_l_max": 45.0,
    "af_saturation_min": 0.98,
    "glare_dilate_px": 2,
    "flag_overlap_min": 0.15,
}


def get(cfg: dict | None, key: str, defaults: dict[str, Any]) -> Any:
    """Config value with a documented default, so a partial config never crashes analysis."""
    if cfg and key in cfg and cfg[key] is not None:
        return cfg[key]
    return defaults[key]


def as_bool_mask(mask: np.ndarray | None, shape: tuple[int, int], name: str) -> np.ndarray:
    """Validate a mask against the expected (H, W). ``None`` means "everything"."""
    if mask is None:
        return np.ones(shape, dtype=bool)
    m = np.asarray(mask)
    if m.shape != shape:
        raise ValueError(f"{name} has shape {m.shape}, expected {shape}")
    return m.astype(bool, copy=False)


def to_float_rgb(img: np.ndarray) -> np.ndarray:
    """Any RGB-like array (uint8/uint16/float, gray or RGBA) -> float32 RGB in [0, 1], NaN -> 0."""
    arr = np.asarray(img)
    if arr.dtype == np.uint8:
        arr = arr.astype(np.float32) / 255.0
    elif arr.dtype == np.uint16:
        arr = arr.astype(np.float32) / 65535.0
    else:
        arr = arr.astype(np.float32, copy=True)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    if arr.ndim != 3 or arr.shape[2] not in (1, 3, 4):
        raise ValueError(f"expected an RGB image (H, W, 3), got shape {np.shape(img)}")
    if arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    arr = arr[:, :, :3]
    return np.clip(np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)


def disk(radius_px: float) -> np.ndarray:
    """Elliptical (disc) structuring element of radius ``radius_px`` (diameter 2r+1)."""
    r = max(0, int(round(radius_px)))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def kernel(diameter_px: float) -> np.ndarray:
    """Disc structuring element with the given (odd) diameter."""
    d = max(1, int(round(diameter_px)))
    d += 1 - d % 2
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d))


def morph(mask: np.ndarray, op: int, diameter_px: float) -> np.ndarray:
    if diameter_px <= 1 or not mask.any():
        return mask.copy()
    return cv2.morphologyEx(mask.astype(np.uint8), op, kernel(diameter_px)).astype(bool)


def masked_smooth(values: np.ndarray, mask: np.ndarray, sigma_px: float) -> np.ndarray:
    """Normalised convolution: blur(v * m) / blur(m), NaN outside ``mask``.

    Unlike a plain blur, bright teeth or the dark background next to the mucosa do not bleed
    into tissue values at the edges, so the mucosa border is not mistaken for fluorescence loss.
    """
    m = mask & np.isfinite(values)
    out = np.full(values.shape, np.nan, dtype=np.float32)
    if not m.any():
        return out
    mf = m.astype(np.float32)
    v = np.where(m, values, 0.0).astype(np.float32)
    if sigma_px and sigma_px > 0:
        num = cv2.GaussianBlur(v, (0, 0), float(sigma_px), borderType=cv2.BORDER_REFLECT)
        den = cv2.GaussianBlur(mf, (0, 0), float(sigma_px), borderType=cv2.BORDER_REFLECT)
    else:
        num, den = v, mf
    ok = m & (den > 1e-6)
    out[ok] = num[ok] / den[ok]
    return out


def finite_median(values: np.ndarray) -> float | None:
    v = values[np.isfinite(values)]
    return float(np.median(v)) if v.size else None
