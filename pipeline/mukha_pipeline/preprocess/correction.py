"""Radiometric corrections applied before registration and AF analysis.

Order: dark subtraction (removes sensor offset and ambient light leaking past the clip; for the
weak AF signal this leak can be a large share of the green channel) -> flat field (removes the
LED / vignetting profile so the image rim is not mistaken for fluorescence loss) -> colour
matrix (WL only; never white-balance an AF image, it would distort the green/red ratio used to
separate tissue autofluorescence from porphyrin red) -> exposure factor (AF intensities brought
onto a common exposure before comparing sites).

All functions return float32, never raise on odd sizes or non-finite pixels, and never modify
their inputs. Non-finite input pixels become 0.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from mukha_pipeline.schema.models import CaptureMetadata


def _as_float(img: np.ndarray) -> np.ndarray:
    """Float32 copy with integer inputs scaled to [0, 1] and non-finite values set to 0."""
    arr = np.asarray(img)
    if np.issubdtype(arr.dtype, np.integer):
        out = arr.astype(np.float32) / float(np.iinfo(arr.dtype).max)
    else:
        out = arr.astype(np.float32, copy=True)
    return np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0)


def _resize_to(arr: np.ndarray, height: int, width: int) -> np.ndarray:
    """Resize a non-empty 2-D or (H, W, C) map to (height, width) > 0.

    Area averaging when shrinking (a dark frame's noise must not alias), linear otherwise.
    """
    if arr.shape[0] == height and arr.shape[1] == width:
        return arr
    shrinking = arr.shape[0] > height and arr.shape[1] > width
    interp = cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR
    out = cv2.resize(arr, (width, height), interpolation=interp)
    if arr.ndim == 3 and out.ndim == 2:  # OpenCV drops a singleton channel axis
        out = out[..., None]
    return out


def _match_channels(ref_img: np.ndarray, arr: np.ndarray) -> np.ndarray:
    """Make ``arr`` broadcastable against ``ref_img`` (gray map vs RGB image and vice versa)."""
    if ref_img.ndim == 3 and arr.ndim == 2:
        return arr[..., None]
    if ref_img.ndim == 2 and arr.ndim == 3:
        return arr.mean(axis=2)
    if ref_img.ndim == 3 and arr.ndim == 3 and arr.shape[2] != ref_img.shape[2]:
        return arr.mean(axis=2, keepdims=True)
    return arr


def subtract_dark(img: np.ndarray, dark: np.ndarray | None) -> np.ndarray:
    """``img - dark``, clipped at 0. The dark frame is resized to the image if needed.

    The dark frame is captured with all LEDs off at the AF exposure, so it holds the sensor
    offset plus ambient light. A missing dark frame returns the image unchanged (QC reports it).
    """
    out = _as_float(img)
    if dark is None or out.size == 0:
        return out
    d = _as_float(dark)
    if d.size == 0:
        return out
    d = _match_channels(out, _resize_to(d, out.shape[0], out.shape[1]))
    return np.clip(out - d, 0.0, None).astype(np.float32)


def apply_flat_field(img: np.ndarray, flat: np.ndarray | None) -> np.ndarray:
    """Divide by the flat field (resized with linear interpolation) and clip to [0, 1].

    Non-positive or non-finite flat values are treated as 1 (no correction there).
    """
    out = _as_float(img)
    if flat is None or out.size == 0:
        return np.clip(out, 0.0, 1.0)
    f = np.asarray(flat, dtype=np.float32)
    if f.size == 0:
        return np.clip(out, 0.0, 1.0)
    if f.ndim == 3:
        f = f.mean(axis=2)
    f = np.where(np.isfinite(f) & (f > 0), f, np.float32(1.0)).astype(np.float32)
    if f.shape != out.shape[:2]:
        f = cv2.resize(f, (out.shape[1], out.shape[0]), interpolation=cv2.INTER_LINEAR)
        f = np.where(f > 0, f, np.float32(1.0))
    f = _match_channels(out, f)
    return np.clip(out / f, 0.0, 1.0).astype(np.float32)


def apply_color_matrix(img: np.ndarray, matrix: np.ndarray | None) -> np.ndarray:
    """``img @ matrix.T`` clipped to [0, 1]. Invalid matrices or non-RGB images pass through."""
    out = _as_float(img)
    if matrix is None:
        return np.clip(out, 0.0, 1.0)
    m = np.asarray(matrix, dtype=np.float64)
    if m.shape != (3, 3) or not np.all(np.isfinite(m)) or out.ndim != 3 or out.shape[2] != 3:
        return np.clip(out, 0.0, 1.0)
    corrected = out.reshape(-1, 3) @ m.T.astype(np.float32)
    return np.clip(corrected.reshape(out.shape), 0.0, 1.0).astype(np.float32)


def _positive(value: float | None) -> float | None:
    """The value when it is a finite positive number, else None (unknown)."""
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) and v > 0 else None


def exposure_factor(meta: CaptureMetadata, reference: CaptureMetadata | None) -> float:
    """Multiplier that brings ``meta``'s intensities onto ``reference``'s exposure.

    Sensor signal scales with exposure time x ISO gain, so the factor is
    ``(t_ref * iso_ref) / (t * iso)`` when all four are finite positive numbers. When either ISO
    is missing, the exposure-time ratio alone is used (both times must be known); without both
    exposure times the factor is 1.0, because an ISO ratio alone cannot be trusted on phones that
    hide their exposure program. Exact for linear (raw) data, approximate for tone-mapped JPEGs.
    """
    if reference is None:
        return 1.0
    t, t_ref = _positive(meta.exposure_time_ms), _positive(reference.exposure_time_ms)
    if t is None or t_ref is None:
        return 1.0
    factor = t_ref / t
    iso, iso_ref = _positive(meta.iso), _positive(reference.iso)
    if iso is not None and iso_ref is not None:
        factor *= iso_ref / iso
    return float(factor) if math.isfinite(factor) and factor > 0 else 1.0
