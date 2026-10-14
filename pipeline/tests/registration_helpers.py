"""Synthetic WL / AF pairs with a known AF -> WL warp, for registration tests (SYNTHETIC)."""

from __future__ import annotations

import math

import cv2
import numpy as np

MARGIN = 128  # scene is rendered larger than the WL view so warped AF never samples outside it


def _smooth_noise(rng: np.random.Generator, h: int, w: int, sigma: float) -> np.ndarray:
    field = cv2.GaussianBlur(rng.standard_normal((h, w)).astype(np.float32), (0, 0), sigma)
    field -= field.mean()
    return field / (field.std() + 1e-8)


def make_scene(rng: np.random.Generator, h: int, w: int) -> np.ndarray:
    """WL-like oral scene: pink mucosa with smooth blobs, vessel-like edges, a teeth band."""
    blobs = 0.6 * _smooth_noise(rng, h, w, 30.0) + 0.4 * _smooth_noise(rng, h, w, 9.0)
    base = np.stack([0.78 + 0.06 * blobs, 0.42 + 0.07 * blobs, 0.46 + 0.05 * blobs], axis=2)
    img = np.clip(base, 0, 1).astype(np.float32)

    # Vessel-like dark red curves and fold edges.
    lines = np.zeros((h, w), dtype=np.float32)
    for _ in range(40):
        x0, y0 = int(rng.integers(0, w)), int(rng.integers(0, h))
        length, angle = rng.uniform(40, 220), rng.uniform(0, math.pi)
        x1, y1 = int(x0 + length * math.cos(angle)), int(y0 + length * math.sin(angle))
        cv2.line(lines, (x0, y0), (x1, y1), 1.0, thickness=int(rng.integers(2, 6)))
    for _ in range(12):
        centre = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        axes = (int(rng.integers(15, 80)), int(rng.integers(10, 50)))
        cv2.ellipse(lines, centre, axes, float(rng.uniform(0, 180)), 0, 360, 1.0, thickness=3)
    lines = cv2.GaussianBlur(lines, (0, 0), 1.5)[..., None]
    img = img * (1 - 0.35 * lines) + lines * 0.35 * np.array([0.55, 0.12, 0.15], np.float32)

    # Teeth band: bright, nearly white blocks with dark gaps, along the top of the view.
    band_top, band_bottom = int(0.06 * h), int(0.24 * h)
    tooth_w = max(8, w // 14)
    teeth = np.zeros((h, w), dtype=np.float32)
    for x in range(0, w, tooth_w):
        cv2.rectangle(teeth, (x + 3, band_top), (x + tooth_w - 4, band_bottom), 1.0, -1)
    teeth = cv2.GaussianBlur(teeth, (0, 0), 2.0)[..., None]
    img = img * (1 - teeth) + teeth * np.array([0.93, 0.91, 0.84], np.float32)
    return np.clip(img, 0, 1).astype(np.float32)


def af_from_wl(wl: np.ndarray) -> np.ndarray:
    """AF-like rendering of a WL scene: green ~ nonlinear luminance, low red / blue."""
    lum = 0.299 * wl[..., 0] + 0.587 * wl[..., 1] + 0.114 * wl[..., 2]
    green = 0.06 + 0.7 * np.power(np.clip(lum, 0, 1), 1.6)
    red = 0.03 + 0.08 * lum
    blue = 0.02 + 0.05 * lum
    return np.clip(np.stack([red, green, blue], axis=2), 0, 1).astype(np.float32)


def similarity_about_centre(
    w: int, h: int, *, angle_deg: float, scale: float, shift: tuple[float, float]
) -> np.ndarray:
    """3x3 AF -> WL warp: rotate/scale about the image centre, then shift (x, y) pixels."""
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    a = math.radians(angle_deg)
    rot = np.array(
        [[scale * math.cos(a), -scale * math.sin(a), 0], [scale * math.sin(a),
                                                          scale * math.cos(a), 0], [0, 0, 1]]
    )
    to_c = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], dtype=np.float64)
    back = np.array([[1, 0, cx + shift[0]], [0, 1, cy + shift[1]], [0, 0, 1]], dtype=np.float64)
    return back @ rot @ to_c


def resize_matrix(src_w: int, src_h: int, dst_w: int, dst_h: int) -> np.ndarray:
    """Pixel-centre map of ``cv2.resize`` from source to destination coordinates."""
    sx, sy = dst_w / src_w, dst_h / src_h
    return np.array([[sx, 0, 0.5 * (sx - 1)], [0, sy, 0.5 * (sy - 1)], [0, 0, 1]], dtype=float)


def make_pair(
    rng: np.random.Generator,
    *,
    h: int = 768,
    w: int = 1024,
    af_size: tuple[int, int] | None = None,
    angle_deg: float = 2.0,
    scale: float = 1.02,
    shift: tuple[float, float] = (12.0, -8.0),
    noise: float = 0.01,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(wl, af, true AF->WL warp, noise-free AF rendered in the WL frame).

    ``af_size`` = (height, width) of the AF image; defaults to the WL size. The true warp then
    includes the AF -> WL resolution change.
    """
    scene = make_scene(rng, h + 2 * MARGIN, w + 2 * MARGIN)
    wl = scene[MARGIN : MARGIN + h, MARGIN : MARGIN + w].copy()
    af_scene = af_from_wl(scene)
    af_truth = af_scene[MARGIN : MARGIN + h, MARGIN : MARGIN + w].copy()

    af_h, af_w = af_size if af_size is not None else (h, w)
    motion = similarity_about_centre(w, h, angle_deg=angle_deg, scale=scale, shift=shift)
    # AF pixel -> full-size AF pixel -> WL pixel.
    true_warp = motion @ np.linalg.inv(resize_matrix(w, h, af_w, af_h))
    to_scene = np.array([[1, 0, MARGIN], [0, 1, MARGIN], [0, 0, 1]], dtype=np.float64)
    sample = to_scene @ true_warp  # AF pixel -> scene pixel
    af = cv2.warpAffine(
        af_scene,
        sample[:2],
        (af_w, af_h),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    af = af + rng.normal(0.0, noise, af.shape).astype(np.float32)
    return wl, np.clip(af, 0, 1).astype(np.float32), true_warp, af_truth


def corner_errors(warp: np.ndarray, true_warp: np.ndarray, af_w: int, af_h: int) -> np.ndarray:
    """Distance (WL px) between where each AF corner lands under the two warps."""
    corners = np.array(
        [[0, 0, 1], [af_w - 1, 0, 1], [0, af_h - 1, 1], [af_w - 1, af_h - 1, 1]], dtype=float
    ).T
    a = warp @ corners
    b = true_warp @ corners
    a = a[:2] / a[2]
    b = b[:2] / b[2]
    return np.linalg.norm(a - b, axis=0)
