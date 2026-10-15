"""Image normalisation and metric primitives shared by the image QC checks.

QC must never crash on what a phone, an upload form or an upstream bug hands it, so
``prepare_image`` accepts almost anything array-like and turns problems into QC issues:
uint8/uint16/bool/float input, grayscale (H, W) or (H, W, 1), RGBA, NaN/inf pixels, empty or
tiny arrays. Metrics are then computed on a copy resized so the longest side is
``metric_max_side_px``, which makes the thresholds independent of camera resolution.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import cv2
import numpy as np

from mukha_pipeline.qc.messages import make_issue
from mukha_pipeline.schema.models import QCIssue

LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)  # ITU-R BT.601, as cv2 RGB2GRAY
CLIP_LOW_255 = 3.0  # a pixel whose brightest channel is <= 3/255 carries no information
CLIP_HIGH_255 = 252.0  # a pixel with any channel >= 252/255 has lost that channel
HIGHLIGHT_LEVEL = 0.98  # near-saturated pixels excluded from the sharpness metric
HIGHLIGHT_DILATE_PX = 3
MIN_SHARPNESS_SUPPORT = 0.25  # keep the mask only if >= 25 % of pixels remain
MAX_NONFINITE_FRACTION = 0.01
DEFAULT_METRIC_MAX_SIDE_PX = 512
DEFAULT_MIN_IMAGE_SIDE_PX = 64
# A colour photo stored as grey (R = G = B in every pixel): decoders expand single-channel files
# to three equal channels, so the channel count alone cannot catch it. Judged on lit pixels
# only (a black frame has equal channels too, and has its own exposure prompts).
DEFAULT_GRAYSCALE_CHANNEL_DIFF_MAX = 0.01  # 99th pct of max(|R-G|, |G-B|), 0-1 scale
GRAYSCALE_LIT_LEVEL = 0.05  # brightest channel at least this for a pixel to be judged
GRAYSCALE_MIN_LIT_FRACTION = 0.01


@dataclass
class PreparedImage:
    """``rgb`` is float32 (h, w, 3) in [0, 1] at metric scale, or None if unusable."""

    rgb: np.ndarray | None
    issues: list[QCIssue] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)


def _to_float(arr: np.ndarray) -> np.ndarray | None:
    """Map any numeric dtype onto float32 [0, 1] (integers by their full scale)."""
    if arr.dtype == np.bool_:
        return arr.astype(np.float32)
    if np.issubdtype(arr.dtype, np.integer):
        scaled = np.multiply(arr, 1.0 / np.iinfo(arr.dtype).max, dtype=np.float32)
        return np.maximum(scaled, 0.0, out=scaled)  # signed: negatives carry no light
    if np.issubdtype(arr.dtype, np.floating):
        return arr.astype(np.float32, copy=False)
    return None


def _to_channels(arr: np.ndarray) -> tuple[np.ndarray | None, bool]:
    """Return (H, W, 3) and whether the input was single-channel."""
    if arr.ndim == 2:
        return np.repeat(arr[..., None], 3, axis=2), True
    if arr.ndim != 3:
        return None, False
    channels = arr.shape[2]
    if channels == 1:
        return np.repeat(arr, 3, axis=2), True
    if channels in (3, 4):
        return arr[..., :3], False  # RGBA: alpha carries no optical information
    return None, False


def channels_equal(rgb: np.ndarray, tolerance: float) -> tuple[bool, float | None]:
    """(looks greyscale, 99th percentile channel difference over lit pixels).

    ``None`` when too few pixels are lit to judge (the photo is then left to the exposure
    checks). Measured on the metric-scale copy, whose area averaging suppresses JPEG noise.
    """
    lit = rgb.max(axis=2) >= GRAYSCALE_LIT_LEVEL
    if float(lit.mean()) < GRAYSCALE_MIN_LIT_FRACTION:
        return False, None
    px = rgb[lit]
    diff = np.maximum(np.abs(px[:, 0] - px[:, 1]), np.abs(px[:, 1] - px[:, 2]))
    p99 = float(np.percentile(diff, 99))
    return p99 <= tolerance, p99


def resize_longest_side(img: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    """Resize so the longest side equals ``max_side`` (area-average down, cubic up)."""
    h, w = img.shape[:2]
    scale = float(max_side) / float(max(h, w))
    if scale == 1.0:
        return img.copy(), 1.0
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    out = cv2.resize(np.ascontiguousarray(img), size, interpolation=interp)
    if out.ndim == 2:  # cv2 drops a singleton channel axis
        out = out[..., None]
    return np.clip(out, 0.0, 1.0).astype(np.float32, copy=False), scale


def prepare_image(
    img: object, cfg: Mapping, prefix: str, *, require_colour: bool = True
) -> PreparedImage:
    """Validate and normalise ``img`` for QC; problems become issues, never exceptions."""
    out = PreparedImage(rgb=None)
    try:
        arr = np.asarray(img)
    except Exception:  # anything that cannot become an array is simply "not an image"
        arr = np.empty(0)
    rgb, was_gray = _to_channels(arr) if arr.size else (None, False)
    rgb = _to_float(rgb) if rgb is not None else None
    if rgb is None or min(rgb.shape[:2]) < 1:
        out.issues.append(
            make_issue("image_invalid", metric=f"{prefix}.valid", value=0.0, threshold=1.0)
        )
        return out

    floating = np.issubdtype(arr.dtype, np.floating)
    nonfinite = 1.0 - float(np.isfinite(rgb).mean()) if floating else 0.0
    if nonfinite > 0:
        out.metrics[f"{prefix}.nonfinite_fraction"] = nonfinite
        if nonfinite > MAX_NONFINITE_FRACTION:
            out.issues.append(
                make_issue(
                    "image_invalid",
                    metric=f"{prefix}.nonfinite_fraction",
                    value=nonfinite,
                    threshold=MAX_NONFINITE_FRACTION,
                )
            )
            return out
        rgb = np.nan_to_num(rgb, nan=0.0, posinf=1.0, neginf=0.0)
    if floating:  # integer and bool inputs are already within [0, 1]
        rgb = np.clip(rgb, 0.0, 1.0)

    h, w = rgb.shape[:2]
    out.metrics[f"{prefix}.width_px"] = float(w)
    out.metrics[f"{prefix}.height_px"] = float(h)
    min_side = int(cfg.get("min_image_side_px", DEFAULT_MIN_IMAGE_SIDE_PX))
    if min(h, w) < min_side:
        out.issues.append(
            make_issue(
                "image_too_small",
                metric=f"{prefix}.min_side_px",
                value=float(min(h, w)),
                threshold=float(min_side),
            )
        )
    if was_gray and require_colour:
        out.issues.append(
            make_issue("image_grayscale", metric=f"{prefix}.channels", value=1.0, threshold=3.0)
        )
    max_side = int(cfg.get("metric_max_side_px", DEFAULT_METRIC_MAX_SIDE_PX))
    out.rgb, scale = resize_longest_side(rgb, max_side)
    out.metrics[f"{prefix}.metric_scale"] = scale
    if require_colour and not was_gray:
        tol = float(cfg.get("grayscale_channel_diff_max", DEFAULT_GRAYSCALE_CHANNEL_DIFF_MAX))
        grey, p99 = channels_equal(out.rgb, tol)
        if p99 is not None:
            out.metrics[f"{prefix}.channel_diff_p99"] = p99
        if grey:
            out.issues.append(
                make_issue(
                    "image_grayscale",
                    metric=f"{prefix}.channel_diff_p99",
                    value=p99,
                    threshold=tol,
                )
            )
    return out


def gray255(rgb: np.ndarray) -> np.ndarray:
    """Luma on the 0-255 scale, float64 (what the Laplacian thresholds are defined on)."""
    return (rgb.astype(np.float32) @ LUMA).astype(np.float64) * 255.0


def highlight_mask(rgb: np.ndarray) -> np.ndarray:
    """Near-saturated pixels plus a small margin.

    A clipped highlight has a hard plateau edge even when the optics are out of focus, so it
    would make a blurred photo look sharp. Excluding it keeps the sharpness metric honest.
    """
    hl = (rgb.max(axis=2) >= HIGHLIGHT_LEVEL).astype(np.uint8)
    if not hl.any():
        return hl.astype(bool)
    k = 2 * HIGHLIGHT_DILATE_PX + 1
    return cv2.dilate(hl, np.ones((k, k), np.uint8)) > 0


def laplacian_variance(rgb: np.ndarray) -> float:
    """Variance of the 3x3 Laplacian of luma (0-255), ignoring clipped highlights."""
    lap = cv2.Laplacian(gray255(rgb), cv2.CV_64F)
    keep = ~highlight_mask(rgb)
    if keep.mean() >= MIN_SHARPNESS_SUPPORT:
        return float(lap[keep].var())
    return float(lap.var())


def clip_fractions(rgb: np.ndarray) -> tuple[float, float]:
    """(fraction crushed to black, fraction with any channel saturated).

    The high side uses the brightest channel because the red channel of pink mucosa saturates
    first; once it clips, the red/white contrast that marks erythroplakia is gone even though
    the pixel does not look white.
    """
    vmax = rgb.max(axis=2) * 255.0
    return float((vmax <= CLIP_LOW_255).mean()), float((vmax >= CLIP_HIGH_255).mean())


def glare_fraction(rgb: np.ndarray, value_min: float, saturation_max: float) -> float:
    """Fraction of specular glare: bright (HSV V) and colourless (HSV S) pixels."""
    v = rgb.max(axis=2)
    s = np.where(v > 0, (v - rgb.min(axis=2)) / np.maximum(v, 1e-6), 0.0)
    return float(((v >= value_min) & (s <= saturation_max)).mean())
