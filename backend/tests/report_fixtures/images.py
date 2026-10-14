"""SYNTHETIC artifact images for report tests and previews (numpy + OpenCV drawings).

Pink mucosa under white light, green autofluorescence with a dark (AF-loss) patch, and the AF-loss
heatmap overlay. Every image is stamped "SYNTHETIC" in the pixels when ``synthetic`` is true.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

H, W = 360, 480


@dataclass(frozen=True)
class LesionDraw:
    """Normalised bbox of the drawn lesion (matches FVLRegion.bbox in the fixture result)."""

    x: float
    y: float
    w: float
    h: float
    kind: str = "white"  # "white" patch or "red" patch under white light
    af_depth: float = 0.5  # fraction of green fluorescence lost inside the lesion
    porphyrin: bool = False  # red porphyrin spots over part of the lesion (confounder)


def _noise(rng: np.random.Generator, sigma: float) -> np.ndarray:
    n = rng.standard_normal((H, W)).astype(np.float32)
    n = cv2.GaussianBlur(n, (0, 0), sigma)
    return n / (np.abs(n).max() + 1e-6)


def _vignette() -> np.ndarray:
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    r = ((xx - W / 2) / (W * 0.6)) ** 2 + ((yy - H / 2) / (H * 0.62)) ** 2
    return np.clip(1.0 - 0.6 * r**1.4, 0.18, 1.0)


def lesion_mask(rng: np.random.Generator, les: LesionDraw) -> np.ndarray:
    """Soft (0..1) irregular blob filling the lesion bbox."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    cx, cy = (les.x + les.w / 2) * W, (les.y + les.h / 2) * H
    rx, ry = les.w / 2 * W, les.h / 2 * H
    d = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2
    hard = (d + 0.38 * _noise(rng, 7)) < 0.92
    soft = cv2.GaussianBlur(hard.astype(np.float32), (0, 0), 2.2)
    return np.clip(soft, 0, 1)


def white_light(
    rng: np.random.Generator, lesions: list[LesionDraw], *, blur: float = 0
) -> np.ndarray:
    base = np.array([0.82, 0.49, 0.50], np.float32)
    img = np.ones((H, W, 3), np.float32) * base
    img += _noise(rng, 45)[..., None] * np.array([0.05, 0.05, 0.05], np.float32)
    img += _noise(rng, 5)[..., None] * np.array([0.025, 0.018, 0.018], np.float32)

    # Fine vessels.
    vessels = np.zeros((H, W), np.float32)
    for _ in range(9):
        pts = [(int(rng.uniform(0, W)), int(rng.uniform(0, H)))]
        for _ in range(5):
            x, y = pts[-1]
            pts.append((int(x + rng.normal(0, 40)), int(y + rng.normal(0, 30))))
        cv2.polylines(vessels, [np.array(pts, np.int32)], False, 1.0, 1, cv2.LINE_AA)
    vessels = cv2.GaussianBlur(vessels, (0, 0), 0.9)
    img -= vessels[..., None] * np.array([0.05, 0.16, 0.14], np.float32)

    for les in lesions:
        m = lesion_mask(rng, les)[..., None]
        if les.kind == "red":
            target = np.array([0.72, 0.22, 0.24], np.float32)
        else:
            target = np.array([0.93, 0.89, 0.83], np.float32)
        texture = 1 + 0.06 * _noise(rng, 2.5)[..., None]
        img = img * (1 - 0.85 * m) + target * texture * 0.85 * m

    img *= _vignette()[..., None]
    # Specular highlights.
    spec = np.zeros((H, W), np.float32)
    for _ in range(4):
        c = (int(rng.uniform(60, W - 60)), int(rng.uniform(50, H - 50)))
        cv2.ellipse(
            spec,
            c,
            (int(rng.uniform(3, 8)), int(rng.uniform(2, 5))),
            float(rng.uniform(0, 180)),
            0,
            360,
            1.0,
            -1,
        )
    spec = cv2.GaussianBlur(spec, (0, 0), 1.6)
    img = img * (1 - spec[..., None]) + spec[..., None]
    if blur:
        img = cv2.GaussianBlur(img, (0, 0), blur)
    return np.clip(img, 0, 1)


def autofluorescence(
    rng: np.random.Generator, lesions: list[LesionDraw], *, signal: float = 1.0, blur: float = 0
) -> np.ndarray:
    vign = _vignette()
    g = (0.58 + 0.07 * _noise(rng, 30) + 0.02 * _noise(rng, 4)) * vign
    r = (0.10 + 0.02 * _noise(rng, 20)) * vign
    b = (0.13 + 0.015 * _noise(rng, 20)) * vign
    for les in lesions:
        m = lesion_mask(rng, les)
        g = g * (1 - les.af_depth * m)
        if les.porphyrin:
            spots = np.zeros((H, W), np.float32)
            for _ in range(14):
                cx = (les.x + rng.uniform(0.45, 1.0) * les.w) * W
                cy = (les.y + rng.uniform(0.0, 0.6) * les.h) * H
                cv2.circle(spots, (int(cx), int(cy)), int(rng.uniform(2, 5)), 1.0, -1)
            spots = cv2.GaussianBlur(spots, (0, 0), 1.5)
            r = r + 0.55 * spots
            g = g + 0.05 * spots
    img = np.stack([r, g, b], axis=-1) * signal
    if blur:
        img = cv2.GaussianBlur(img, (0, 0), blur)
    return np.clip(img, 0, 1)


def heatmap_overlay(wl: np.ndarray, af: np.ndarray, threshold: float = 0.75) -> np.ndarray:
    """AF-loss overlay drawn with the pipeline's own renderer (same colour ramp, opacity and
    "not analysed" desaturation as real results). Illumination is flat-field corrected and only
    the well-lit tissue counts as analysed, so the dark vignetted rim is not painted as loss."""
    from mukha_pipeline.af.overlay import render_heatmap_overlay

    vignette = _vignette()
    # Flat-field correction (the pipeline normalises illumination the same way), then keep only
    # the well-lit tissue: the dark rim is "not analysed" (desaturated), never "loss".
    g = cv2.GaussianBlur(af[..., 1], (0, 0), 2.0) / vignette
    lit = (vignette > 0.5).astype(np.uint8)
    analysed = cv2.erode(lit, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))) > 0
    ref = float(np.median(g[analysed]))
    rel = (g / max(ref, 1e-6)).astype(np.float32)
    rel[~analysed] = np.nan
    below = ((rel < threshold) & analysed).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(below, connectivity=8)
    keep = np.zeros_like(labels)
    k = 0
    for i in np.argsort(-stats[:, cv2.CC_STAT_AREA]):
        if i == 0 or stats[i, cv2.CC_STAT_AREA] < 150:
            continue
        k += 1
        keep[labels == i] = k
    out = render_heatmap_overlay(wl, rel, keep, threshold, None)
    return out.astype(np.float32) / 255.0


def to_png(img: np.ndarray, *, synthetic: bool) -> bytes:
    u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8).copy()
    if synthetic:
        for color, thick in (((20, 20, 20), 3), ((255, 255, 255), 1)):
            cv2.putText(
                u8,
                "SYNTHETIC",
                (10, H - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                thick,
                cv2.LINE_AA,
            )
    buf = io.BytesIO()
    Image.fromarray(u8).save(buf, format="PNG", compress_level=1)
    return buf.getvalue()


def site_artifacts(
    site: str,
    lesions: list[LesionDraw],
    *,
    seed: int,
    synthetic: bool = True,
    qc_failed: bool = False,
    af_signal: float = 1.0,
) -> dict[str, bytes]:
    """Artifacts for one site, keyed as the pipeline would key them."""
    rng = np.random.default_rng(seed)
    blur = 3.5 if qc_failed and af_signal >= 1 else 0
    wl = white_light(rng, lesions, blur=blur)
    af = autofluorescence(rng, lesions, signal=af_signal, blur=blur)
    out = {
        f"sites/{site}/wl.png": to_png(wl, synthetic=synthetic),
        f"sites/{site}/af.png": to_png(af, synthetic=synthetic),
    }
    if not qc_failed:
        out[f"sites/{site}/af_registered.png"] = to_png(af, synthetic=synthetic)
        out[f"sites/{site}/heatmap_overlay.png"] = to_png(
            heatmap_overlay(wl, af), synthetic=synthetic
        )
    return out
