"""Shared helpers for the synthetic-data tests (test_synthetic_*.py)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from mukha_pipeline.positioning import find_positioning_violations

SMALL = (192, 256)  # fast, still large enough for every texture and the stamp
MEDIUM = (384, 512)
REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_DATA = REPO_ROOT / "sample_data"


def assert_float_image(img: np.ndarray, shape: tuple[int, int]) -> None:
    assert img.dtype == np.float32
    assert img.shape == (*shape, 3)
    assert np.isfinite(img).all()
    assert float(img.min()) >= 0.0 and float(img.max()) <= 1.0


def relative_af_proxy(af: np.ndarray, tissue: np.ndarray, dark_level: float) -> np.ndarray:
    """Smoothed green / robust tissue reference, smoothing only within tissue.

    A stand-in for the analysis's relative-AF map, so the generator can be checked without
    importing the analysis modules.
    """
    green = af[..., 1].astype(np.float32) - dark_level
    weight = tissue.astype(np.float32)
    num = cv2.GaussianBlur(green * weight, (0, 0), 3)
    den = cv2.GaussianBlur(weight, (0, 0), 3)
    smooth = num / np.maximum(den, 1e-3)
    values = green[tissue]
    lo, hi = np.percentile(values, [25, 90])
    reference = float(values[(values >= lo) & (values <= hi)].mean())
    return smooth / reference


def hsv_value(img: np.ndarray) -> np.ndarray:
    return img.max(axis=-1)


def assert_positioning_clean(text: str, where: str = "") -> None:
    violations = find_positioning_violations(text)
    assert violations == [], f"{where}: {violations}"
