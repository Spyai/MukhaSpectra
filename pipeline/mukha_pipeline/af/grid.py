"""Downsampled relative-AF grid for the interactive 3D "terrain" view (JSON-safe)."""

from __future__ import annotations

import math

import numpy as np


def _round_or_none(v: float) -> float | None:
    return round(float(v), 3) if math.isfinite(v) else None


def block_nanmean(values: np.ndarray, factor: int) -> np.ndarray:
    """NaN-aware mean over ``factor`` x ``factor`` blocks (edge blocks may be partial)."""
    arr = np.asarray(values, dtype=np.float64)
    h, w = arr.shape
    f = max(int(factor), 1)
    gh, gw = math.ceil(h / f), math.ceil(w / f)
    padded = np.full((gh * f, gw * f), np.nan)
    padded[:h, :w] = arr
    blocks = padded.reshape(gh, f, gw, f)
    finite = np.isfinite(blocks)
    total = np.where(finite, blocks, 0.0).sum(axis=(1, 3))
    count = finite.sum(axis=(1, 3))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, total / np.maximum(count, 1), np.nan)


def relative_af_grid(
    relative_af: np.ndarray, size: int, *, threshold: float | None = None
) -> dict[str, object]:
    """Block-average relative AF to at most ``size`` cells on the longest side.

    Returns {"width", "height", "values" (row-major lists, 3 dp, null = not analysed), "min",
    "max", "threshold", "block_px"}; serialisable with ``json.dumps(..., allow_nan=False)``.
    Never upsamples a small map.
    """
    arr = np.asarray(relative_af, dtype=np.float64)
    if arr.ndim != 2 or arr.size == 0:
        return {
            "width": 0,
            "height": 0,
            "values": [],
            "min": None,
            "max": None,
            "threshold": _round_or_none(threshold) if threshold is not None else None,
            "block_px": 1,
        }
    arr = np.where(np.isfinite(arr), arr, np.nan)
    size = max(int(size), 1)
    factor = max(1, math.ceil(max(arr.shape) / size))
    grid = block_nanmean(arr, factor)
    finite = grid[np.isfinite(grid)]
    return {
        "width": int(grid.shape[1]),
        "height": int(grid.shape[0]),
        "values": [[_round_or_none(v) for v in row] for row in grid.tolist()],
        "min": _round_or_none(finite.min()) if finite.size else None,
        "max": _round_or_none(finite.max()) if finite.size else None,
        "threshold": _round_or_none(threshold) if threshold is not None else None,
        "block_px": int(factor),
    }
