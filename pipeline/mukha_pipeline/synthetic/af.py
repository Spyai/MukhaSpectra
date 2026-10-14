"""Synthetic 405 nm autofluorescence (AF) images derived from a synthetic WL scene.

Optics modelled (simplified, for software testing only):

* Healthy mucosa fluoresces green (collagen cross-links in the stroma, FAD/NADH in the
  epithelium). Seen through the long-pass emission filter the image is green-dominant with very
  little blue. Intensity follows the tissue: vessels and red (blood-rich) areas look darker
  because haemoglobin absorbs both the 405 nm excitation and the green emission.
* Fluorescence visualisation loss (FVL): dysplastic epithelium thickens, stromal collagen breaks
  down and blood volume rises, so green drops. Inside ``fvl_mask`` green is multiplied by
  ``1 - fvl_depth`` with a soft edge. The lesion's WL whiteness is deliberately NOT allowed to
  brighten the AF, so the FVL ground truth is exactly the mask and the depth.
* Teeth fluoresce bright pale green. Bacterial porphyrins (plaque, tongue coating) add red
  fluorescence near 635 nm: a known look-alike that the analysis must flag as a confounder.
* Camera: torch fall-off, bloom (veiling glare), a dark pedestal matching
  :func:`generate_dark_frame`, shot and read noise, and a small random misalignment against the
  WL frame (the phone moves between the two captures) to exercise registration.

Illumination is gentler than in the WL image so that only the FVL mask, not scene geometry,
produces relative-AF values below the analysis threshold in a clean image.
"""

from __future__ import annotations

from typing import Literal, overload

import cv2
import numpy as np

from mukha_pipeline.synthetic._fields import (
    blur,
    child_rngs,
    dilate,
    erode,
    fbm,
    gaussian_dots,
    hsv_sv,
    norm_grids,
    normalized_blur,
)
from mukha_pipeline.synthetic.watermark import watermark_synthetic

DEFAULT_DARK_LEVEL = 0.01
MAX_ROTATION_DEG = 1.5
MAX_SHIFT_PX = 12.0  # at the 768 px reference frame height; scales with image size
MAX_SCALE_CHANGE = 0.02

MUCOSA_GREEN = 0.42  # healthy mucosa green at the torch hot spot (0-1, before the pedestal)
_RED_PER_GREEN = 0.33  # red tail of the tissue emission plus sensor cross-talk
_BLUE_PER_GREEN = 0.10  # long-pass filter: very little blue reaches the sensor
_FVL_RED_SHARE = 0.5  # red drops half as much as green, so red/green rises in FVL
_TEETH_AF = np.array([0.50, 0.84, 0.32], np.float32)  # bright pale green enamel/dentin
_BACKGROUND_AF = np.array([0.012, 0.022, 0.009], np.float32)  # scattered light in the cavity
_PORPHYRIN_RED = 0.32
_PEDESTAL_TINT = np.array([1.05, 1.0, 0.95], np.float32)


def _estimate_tissue(wl: np.ndarray) -> np.ndarray:
    """Fallback mucosa mask for callers without a scene: reddish, saturated, not too dark."""
    sat, val = hsv_sv(wl)
    red_dominant = (wl[..., 0] >= wl[..., 1]) & (wl[..., 0] >= wl[..., 2])
    return red_dominant & (sat >= 0.15) & (val >= 0.15)


def _estimate_teeth(wl: np.ndarray, tissue: np.ndarray) -> np.ndarray:
    sat, val = hsv_sv(wl)
    return (sat <= 0.25) & (val >= 0.55) & ~tissue


def random_misalignment(
    rng: np.random.Generator,
    shape: tuple[int, int],
    *,
    max_rotation_deg: float = MAX_ROTATION_DEG,
    max_shift_px: float | None = None,
    max_scale_change: float = MAX_SCALE_CHANGE,
) -> np.ndarray:
    """Random similarity transform (3x3) mapping WL-frame pixel coords to AF-frame coords.

    Rotation about the image centre, isotropic scale and a shift whose length is at most
    ``max_shift_px`` (default 12 px at a 768 px frame, scaled with the image). The analysis's
    registration must recover the inverse (AF -> WL).
    """
    h, w = shape[:2]
    if max_shift_px is None:
        max_shift_px = MAX_SHIFT_PX * min(h, w) / 768.0
    angle = float(rng.uniform(-max_rotation_deg, max_rotation_deg))
    scale = float(1.0 + rng.uniform(-max_scale_change, max_scale_change))
    r = float(max_shift_px * np.sqrt(rng.uniform(0.0, 1.0)))
    phi = float(rng.uniform(0.0, 2.0 * np.pi))
    m = np.eye(3)
    m[:2] = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, scale)
    m[0, 2] += r * np.cos(phi)
    m[1, 2] += r * np.sin(phi)
    return m


def _porphyrin_map(
    rng: np.random.Generator,
    tissue: np.ndarray,
    teeth: np.ndarray,
    fvl: np.ndarray,
    n_spots: int,
    s: float,
) -> np.ndarray:
    """Clusters of red-fluorescing plaque: along the gingival margin, else on the tissue."""
    shape = tissue.shape
    if n_spots <= 0:
        return np.zeros(shape, np.float32)
    if teeth.any():
        candidates = dilate(teeth, 22 * s) & ~dilate(teeth, 3 * s) & tissue
    else:
        candidates = erode(tissue, 12 * s)
    candidates &= ~dilate(fvl, 25 * s)
    if not candidates.any():
        candidates = tissue & ~fvl
    ys, xs = np.nonzero(candidates)
    if len(xs) == 0:
        return np.zeros(shape, np.float32)
    centres = rng.choice(len(xs), size=n_spots, replace=len(xs) < n_spots)
    all_y: list[float] = []
    all_x: list[float] = []
    for c in centres:
        k = int(rng.integers(1, 5))
        all_y.extend(ys[c] + rng.normal(0.0, 4.0 * s, k))
        all_x.extend(xs[c] + rng.normal(0.0, 4.0 * s, k))
    amps = rng.uniform(0.6, 1.0, len(all_x))
    spots = gaussian_dots(shape, np.asarray(all_y), np.asarray(all_x), amps, 3.5 * s)
    return np.clip(spots, 0.0, 1.0) * tissue


@overload
def generate_af_from_wl(
    wl: np.ndarray,
    rng: np.random.Generator,
    *,
    tissue_mask: np.ndarray | None = ...,
    teeth_mask: np.ndarray | None = ...,
    fvl_mask: np.ndarray | None = ...,
    fvl_depth: float = ...,
    porphyrin_spots: int = ...,
    bloom: float = ...,
    misalign: bool = ...,
    return_transform: Literal[False] = ...,
    watermark: bool = ...,
    dark_level: float = ...,
    specular_mask: np.ndarray | None = ...,
) -> np.ndarray: ...


@overload
def generate_af_from_wl(
    wl: np.ndarray,
    rng: np.random.Generator,
    *,
    tissue_mask: np.ndarray | None = ...,
    teeth_mask: np.ndarray | None = ...,
    fvl_mask: np.ndarray | None = ...,
    fvl_depth: float = ...,
    porphyrin_spots: int = ...,
    bloom: float = ...,
    misalign: bool = ...,
    return_transform: Literal[True],
    watermark: bool = ...,
    dark_level: float = ...,
    specular_mask: np.ndarray | None = ...,
) -> tuple[np.ndarray, np.ndarray]: ...


def generate_af_from_wl(
    wl: np.ndarray,
    rng: np.random.Generator,
    *,
    tissue_mask: np.ndarray | None = None,
    teeth_mask: np.ndarray | None = None,
    fvl_mask: np.ndarray | None = None,
    fvl_depth: float = 0.5,
    porphyrin_spots: int = 0,
    bloom: float = 0.6,
    misalign: bool = True,
    return_transform: bool = False,
    watermark: bool = True,
    dark_level: float = DEFAULT_DARK_LEVEL,
    specular_mask: np.ndarray | None = None,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Synthesise the AF capture that matches a WL capture (float32 RGB in [0, 1]).

    ``tissue_mask``/``teeth_mask``/``specular_mask`` come from :class:`SyntheticScene`; when
    omitted they are estimated from the WL colours. With ``return_transform=True`` returns
    ``(af, transform)`` where ``transform`` (3x3) maps WL pixel coords to AF pixel coords
    (identity when ``misalign=False``).
    """
    if wl.ndim != 3 or wl.shape[2] != 3:
        raise ValueError(f"wl must be (H, W, 3), got {wl.shape}")
    if not 0.0 <= fvl_depth <= 1.0:
        raise ValueError(f"fvl_depth must be in 0..1, got {fvl_depth}")
    h, w = wl.shape[:2]
    s = min(h, w) / 768.0
    geo_rng, spot_rng, warp_rng, noise_rng = child_rngs(rng, 4)
    img = np.clip(wl.astype(np.float32), 0.0, 1.0)
    tissue = tissue_mask.astype(bool) if tissue_mask is not None else _estimate_tissue(img)
    teeth = teeth_mask.astype(bool) if teeth_mask is not None else _estimate_teeth(img, tissue)
    teeth &= ~tissue
    fvl = fvl_mask.astype(bool) if fvl_mask is not None else np.zeros((h, w), bool)
    if specular_mask is not None:
        spec = dilate(specular_mask.astype(bool), 2 * s)
    else:
        sat, val = hsv_sv(img)
        spec = dilate((val >= 0.92) & (sat <= 0.25) & (tissue | teeth), 2 * s)

    # Green weights blood absorption: vessels and red areas fluoresce less.
    lum = 0.25 * img[..., 0] + 0.65 * img[..., 1] + 0.10 * img[..., 2]
    fvl_zone = dilate(fvl, 8 * s)
    illum_w = tissue & ~fvl_zone & ~spec
    if not illum_w.any():
        illum_w = tissue if tissue.any() else np.ones((h, w), bool)
    s_raw = normalized_blur(lum, illum_w, 0.06 * min(h, w))
    ref = float(np.percentile(s_raw[illum_w], 75))
    shading = np.clip(s_raw / max(ref, 1e-4), 0.55, 1.2) ** 0.25
    local = normalized_blur(lum, tissue & ~spec, 5 * s)
    lum_filled = np.where(spec, local, lum)
    texture = np.clip(lum_filled / np.maximum(local, 1e-3), 0.7, 1.2) ** 1.2
    # Fade texture out at the tissue edge: the steep WL shading there (cavity, gingival
    # margin) is geometry, not fluorescence, and must not read as a rim of FVL.
    edge_dist = cv2.distanceTransform(np.pad(tissue.astype(np.uint8), 1), cv2.DIST_L2, 5)
    interior = np.clip(edge_dist[1:-1, 1:-1] / (12.0 * s), 0.0, 1.0)
    texture = 1.0 + (texture - 1.0) * interior
    # Inside the lesion the WL texture (plaque plates, fissures, speckles) is damped, not
    # copied: the FVL depth alone sets how dark the region is.
    texture = np.where(fvl_zone, 1.0 + 0.3 * (texture - 1.0), texture)

    X, Y = norm_grids((h, w))
    cx, cy = 0.5 + geo_rng.uniform(-0.08, 0.08), 0.45 + geo_rng.uniform(-0.08, 0.08)
    r2 = ((X - cx) * (w / h)) ** 2 + (Y - cy) ** 2
    torch = np.clip(1.0 - 0.06 * r2 / 0.55, 0.85, 1.0)
    variation = 1.0 + 0.035 * fbm(geo_rng, (h, w), 150 * s, 3)
    heterogeneity = 1.0 + 0.12 * fbm(geo_rng, (h, w), 12 * s, 2)
    loss = np.clip(fvl_depth * blur(fvl.astype(np.float32), 4 * s) * heterogeneity, 0.0, 0.95)
    green = MUCOSA_GREEN * shading * texture * torch * variation * (1.0 - loss)

    red = _RED_PER_GREEN * green / (1.0 - loss) * (1.0 - _FVL_RED_SHARE * loss)
    tissue_rgb = np.stack([red + 0.01, green, _BLUE_PER_GREEN * green + 0.005], axis=-1)
    teeth_lum = lum / max(float(np.percentile(lum[teeth], 90)), 1e-3) if teeth.any() else lum
    teeth_rgb = _TEETH_AF * (np.clip(teeth_lum, 0.4, 1.05) * torch)[..., None]
    # Dilate before blurring so tissue keeps full signal up to (and a little past) its edge:
    # a segmenter may count the last dim pixels before the cavity as mucosa.
    tw = blur(dilate(tissue, 1.5 * s).astype(np.float32), 1.0 * s)
    thw = blur(dilate(teeth, 1.5 * s).astype(np.float32), 1.0 * s)
    total = np.maximum(tw + thw, 1.0)
    tw, thw = tw / total, thw / total
    bg_w = np.clip(1.0 - tw - thw, 0.0, 1.0)
    af = tissue_rgb * tw[..., None] + teeth_rgb * thw[..., None] + _BACKGROUND_AF * bg_w[..., None]

    spots = _porphyrin_map(spot_rng, tissue, teeth, fvl, porphyrin_spots, s)
    af[..., 0] += _PORPHYRIN_RED * spots
    af[..., 1] *= 1.0 - 0.12 * spots

    if bloom > 0:
        glow = blur(af, 14 * s)
        highlights = blur(np.clip(af - 0.45, 0.0, None), 5 * s)
        af = af + bloom * (0.08 * glow + 0.25 * highlights)
    af = np.clip(af, 0.0, 1.0).astype(np.float32)

    transform = random_misalignment(warp_rng, (h, w)) if misalign else np.eye(3)
    if misalign:
        af = cv2.warpAffine(
            af,
            transform[:2].astype(np.float64),
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
    pedestal = dark_level * _PEDESTAL_TINT
    shot = np.sqrt(0.00025 * np.clip(af, 0.0, None)) * noise_rng.standard_normal(af.shape)
    read = 0.003 * noise_rng.standard_normal(af.shape)
    out = np.clip(af + pedestal + shot + read, 0.0, 1.0).astype(np.float32)
    if watermark:
        # Camera-fixed text, identical to the WL stamp's text pixels; see watermark.py.
        out = watermark_synthetic(out, band=False)
    if return_transform:
        return out, transform
    return out


def generate_dark_frame(
    shape: tuple[int, ...],
    rng: np.random.Generator,
    level: float = DEFAULT_DARK_LEVEL,
    *,
    watermark: bool = False,
) -> np.ndarray:
    """Dark frame (all LEDs off, AF exposure): pedestal, faint amp glow, hot pixels, noise.

    ``shape`` is (H, W) or (H, W, 3). Not stamped by default: dark subtraction would carry any
    label into the AF image (and break the camera-fixed label that registration ignores), so
    dark frames are labelled SYNTHETIC by file name and metadata instead. ``watermark=True``
    adds faint text just above the pedestal for standalone use.
    """
    h, w = int(shape[0]), int(shape[1])
    X, Y = norm_grids((h, w))
    corner = float(rng.uniform(0.0, 1.0)) < 0.5
    glow = np.exp(-(((X - (1.0 if corner else 0.0)) ** 2 + (Y - 1.0) ** 2) / 0.08))
    base = level * (1.0 + 0.10 * glow)
    frame = base[..., None] * _PEDESTAL_TINT
    frame = frame + 0.0006 * rng.standard_normal((h, w, 3))
    n_hot = max(1, int(2e-5 * h * w))
    hy = rng.integers(0, h, n_hot)
    hx = rng.integers(0, w, n_hot)
    frame[hy, hx] += rng.uniform(0.05, 0.2, (n_hot, 1))
    frame = np.clip(frame, 0.0, 1.0).astype(np.float32)
    if watermark:
        frame = watermark_synthetic(frame, dim=True)
    return frame
