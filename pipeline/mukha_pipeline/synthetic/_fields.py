"""Small numeric helpers for the synthetic generators: smooth noise, soft masks, blurs.

Everything here is deterministic given the ``np.random.Generator`` passed in, so the same seed
always yields byte-identical images (tests and the committed ``sample_data`` rely on that).
"""

from __future__ import annotations

import cv2
import numpy as np

RGB = tuple[float, float, float]


def value_noise(rng: np.random.Generator, shape: tuple[int, int], cell_px: float) -> np.ndarray:
    """Smooth random field (zero mean, roughly unit std) with features about ``cell_px`` wide.

    A coarse Gaussian grid upsampled with bicubic interpolation: cheap, smooth and isotropic
    enough for tissue texture.
    """
    h, w = shape
    gh = max(2, int(np.ceil(h / max(cell_px, 1.0))) + 1)
    gw = max(2, int(np.ceil(w / max(cell_px, 1.0))) + 1)
    grid = rng.standard_normal((gh, gw)).astype(np.float32)
    return cv2.resize(grid, (w, h), interpolation=cv2.INTER_CUBIC)


def fbm(
    rng: np.random.Generator,
    shape: tuple[int, int],
    cell_px: float,
    octaves: int = 4,
    persistence: float = 0.5,
) -> np.ndarray:
    """Fractal (multi-octave) noise normalised to zero mean and unit std."""
    out = np.zeros(shape, np.float32)
    amp, cell = 1.0, float(cell_px)
    for _ in range(octaves):
        out += amp * value_noise(rng, shape, max(cell, 1.5))
        amp *= persistence
        cell /= 2.0
    std = float(out.std())
    return (out - float(out.mean())) / (std if std > 1e-8 else 1.0)


def smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    """Hermite step from 0 at ``edge0`` to 1 at ``edge1`` (edges may be descending)."""
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32)


def blur(img: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur with reflected borders; a no-op for tiny sigmas."""
    if sigma < 0.3:
        return img.astype(np.float32, copy=True)
    return cv2.GaussianBlur(
        img.astype(np.float32), (0, 0), sigmaX=float(sigma), borderType=cv2.BORDER_REFLECT_101
    )


def normalized_blur(values: np.ndarray, weights: np.ndarray, sigma: float) -> np.ndarray:
    """Weighted Gaussian average: smooths ``values`` using only pixels where ``weights`` > 0.

    Used to estimate illumination from tissue only, so that teeth, shadow or a lesion do not
    leak into the estimate.
    """
    w = weights.astype(np.float32)
    num = blur(values.astype(np.float32) * w, sigma)
    den = blur(w, sigma)
    fallback = float(np.sum(values * w) / max(float(np.sum(w)), 1e-6))
    return np.where(den > 1e-4, num / np.maximum(den, 1e-4), fallback).astype(np.float32)


def lerp_rgb(img: np.ndarray, color: RGB | np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Blend ``img`` (H, W, 3) towards ``color`` with a per-pixel weight in [0, 1]."""
    c = np.asarray(color, np.float32)
    a = np.clip(weight, 0.0, 1.0)[..., None].astype(np.float32)
    return (img * (1.0 - a) + c * a).astype(np.float32)


def norm_grids(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Pixel-centre coordinates normalised to [0, 1]: (X along width, Y along height)."""
    h, w = shape
    x = (np.arange(w, dtype=np.float32) + 0.5) / w
    y = (np.arange(h, dtype=np.float32) + 0.5) / h
    return np.meshgrid(x, y)


def soft_ellipse(
    X: np.ndarray,
    Y: np.ndarray,
    cx: float,
    cy: float,
    rx: float,
    ry: float,
    soft: float,
    wobble: np.ndarray | None = None,
) -> np.ndarray:
    """Soft-edged ellipse alpha in normalised coordinates, optionally with a wobbly border."""
    d = np.sqrt(((X - cx) / rx) ** 2 + ((Y - cy) / ry) ** 2)
    if wobble is not None:
        d = d + wobble
    return 1.0 - smoothstep(1.0 - soft, 1.0 + soft, d)


def gaussian_dots(
    shape: tuple[int, int], ys: np.ndarray, xs: np.ndarray, amps: np.ndarray, sigma: float
) -> np.ndarray:
    """Sum of small Gaussian blobs at sub-pixel positions (papillae, speckles, glands)."""
    canvas = np.zeros(shape, np.float32)
    if len(xs) == 0:
        return canvas
    h, w = shape
    yi = np.clip(np.round(ys).astype(np.int64), 0, h - 1)
    xi = np.clip(np.round(xs).astype(np.int64), 0, w - 1)
    np.add.at(canvas, (yi, xi), amps.astype(np.float32))
    return blur(canvas, sigma) / _impulse_peak(sigma)


def _impulse_peak(sigma: float) -> float:
    """Peak of a blurred unit impulse, so a single isolated dot peaks at its amplitude."""
    r = int(np.ceil(4 * max(sigma, 0.3))) + 1
    probe = np.zeros((2 * r + 1, 2 * r + 1), np.float32)
    probe[r, r] = 1.0
    return float(blur(probe, sigma)[r, r])


def luminance(img: np.ndarray) -> np.ndarray:
    """Rec. 601 luma of a float RGB image."""
    return (0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]).astype(np.float32)


def hsv_sv(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """HSV saturation and value of a float RGB image, both in [0, 1]."""
    mx = img.max(axis=-1)
    mn = img.min(axis=-1)
    sat = np.where(mx > 1e-6, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
    return sat.astype(np.float32), mx.astype(np.float32)


def disk_kernel(radius_px: float) -> np.ndarray:
    r = max(1, int(round(radius_px)))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def dilate(mask: np.ndarray, radius_px: float) -> np.ndarray:
    if radius_px < 0.5:
        return mask.astype(bool, copy=True)
    return cv2.dilate(mask.astype(np.uint8), disk_kernel(radius_px)).astype(bool)


def erode(mask: np.ndarray, radius_px: float) -> np.ndarray:
    if radius_px < 0.5:
        return mask.astype(bool, copy=True)
    return cv2.erode(mask.astype(np.uint8), disk_kernel(radius_px)).astype(bool)


def child_rngs(rng: np.random.Generator, n: int) -> list[np.random.Generator]:
    """Independent child generators, so optional features do not shift other random draws."""
    return list(rng.spawn(n))
