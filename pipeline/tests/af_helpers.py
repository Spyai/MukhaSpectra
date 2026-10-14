"""Scene builders for the segmentation and AF tests (SYNTHETIC test images only).

The scenes are deliberately simple and fully known, so the tests can check geometry exactly:
a pink, textured mucosa on a dark oral-cavity background, a band of teeth along the top border,
an enclosed white patch (leukoplakia stand-in), specular dots, and an AF image whose green glow
is proportional to tissue with a dark FVL disc of known depth.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from mukha_pipeline.types import AFMaps, Masks

MUCOSA_RGB = (0.86, 0.52, 0.56)  # pink lining mucosa
TEETH_RGB = (0.93, 0.90, 0.80)  # bright, slightly yellow, low saturation
PATCH_RGB = (0.90, 0.87, 0.85)  # whitish leukoplakia stand-in (teeth-like colour)
BACKGROUND_RGB = (0.05, 0.03, 0.03)  # oral cavity shadow
BROWN_RGB = (0.36, 0.22, 0.12)  # melanin pigmentation, L* ~ 27
BLOOD_RGB = (0.50, 0.04, 0.08)  # deep red, a* > 35, L* < 45


def disc(shape: tuple[int, int], cy: float, cx: float, r: float) -> np.ndarray:
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= r**2


def ellipse(shape: tuple[int, int], cy: float, cx: float, ry: float, rx: float) -> np.ndarray:
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    return ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1.0


def smooth_texture(rng: np.random.Generator, shape: tuple[int, int], amp: float) -> np.ndarray:
    """Multiplicative tissue texture: 1 + smoothed noise of roughly +-amp."""
    noise = rng.standard_normal(shape).astype(np.float32)
    noise = cv2.GaussianBlur(noise, (0, 0), 4.0)
    noise /= max(float(noise.std()), 1e-6)
    return 1.0 + amp * noise


def paint(img: np.ndarray, mask: np.ndarray, rgb: tuple[float, float, float]) -> None:
    img[mask] = np.asarray(rgb, dtype=np.float32)


@dataclass
class WLScene:
    wl: np.ndarray
    tissue: np.ndarray  # mucosa incl. the white patch, excluding specular dots
    teeth: np.ndarray
    patch: np.ndarray
    specular: np.ndarray
    background: np.ndarray
    extra: dict[str, np.ndarray] = field(default_factory=dict)


def make_wl_scene(
    rng: np.random.Generator,
    shape: tuple[int, int] = (360, 480),
    *,
    teeth_band: bool = True,
    white_patch: bool = True,
    specular_dots: bool = True,
    patch_radius: float = 34.0,
) -> WLScene:
    """Pink textured mucosa ellipse on a dark background, teeth along the top border."""
    h, w = shape
    img = np.empty((h, w, 3), np.float32)
    paint(img, np.ones(shape, bool), BACKGROUND_RGB)
    tissue = ellipse(shape, h * 0.56, w * 0.5, h * 0.42, w * 0.46)
    tex = smooth_texture(rng, shape, 0.05)
    img[tissue] = (np.asarray(MUCOSA_RGB, np.float32)[None, :] * tex[tissue][:, None]).clip(0, 1)

    teeth = np.zeros(shape, bool)
    if teeth_band:
        teeth[: int(h * 0.12), int(w * 0.15) : int(w * 0.85)] = True
        paint(img, teeth, TEETH_RGB)
        tissue &= ~teeth

    patch = np.zeros(shape, bool)
    if white_patch:
        patch = disc(shape, h * 0.55, w * 0.62, patch_radius)
        paint(img, patch, PATCH_RGB)

    specular = np.zeros(shape, bool)
    if specular_dots:
        for cy, cx in ((h * 0.45, w * 0.3), (h * 0.7, w * 0.42), (h * 0.55, w * 0.62)):
            specular |= disc(shape, cy, cx, 3)
        specular &= tissue
        paint(img, specular, (1.0, 1.0, 1.0))
    background = ~(tissue | teeth)
    img += rng.normal(0, 0.004, img.shape).astype(np.float32)
    return WLScene(
        wl=img.clip(0, 1),
        tissue=tissue & ~specular,
        teeth=teeth,
        patch=patch,
        specular=specular,
        background=background,
    )


@dataclass
class AFScene:
    af: np.ndarray
    mucosa: np.ndarray
    fvl: np.ndarray
    base_green: float
    depth: float


def make_af_scene(
    rng: np.random.Generator,
    shape: tuple[int, int] = (360, 480),
    *,
    fvl_discs: tuple[tuple[float, float, float, float], ...] = ((0.45, 0.4, 0.11, 0.5),),
    base_green: float = 0.45,
    noise: float = 0.01,
    porphyrin_disc: tuple[float, float, float] | None = None,
    porphyrin_red: float = 0.32,
    mucosa: np.ndarray | None = None,
) -> AFScene:
    """AF image: green proportional to tissue, dark FVL discs (cy, cx, r as fractions of H/W/H,
    depth = fractional green loss), optional red porphyrin disc (cy, cx, r)."""
    h, w = shape
    muc = mucosa if mucosa is not None else ellipse(shape, h * 0.5, w * 0.5, h * 0.45, w * 0.47)
    tex = smooth_texture(rng, shape, 0.03)
    green = np.where(muc, base_green * tex, 0.02).astype(np.float32)
    fvl = np.zeros(shape, bool)
    depth = 0.0
    for cy, cx, r, d in fvl_discs:
        m = disc(shape, cy * h, cx * w, r * h) & muc
        green[m] *= 1.0 - d
        fvl |= m
        depth = d
    red = np.where(muc, 0.06, 0.01).astype(np.float32)
    blue = np.where(muc, 0.05, 0.01).astype(np.float32)
    if porphyrin_disc is not None:
        cy, cx, r = porphyrin_disc
        red[disc(shape, cy * h, cx * w, r * h) & muc] = porphyrin_red
    af = np.stack([red, green, blue], axis=-1)
    af += rng.normal(0, noise, af.shape).astype(np.float32)
    return AFScene(af=af.clip(0, 1), mucosa=muc, fvl=fvl, base_green=base_green, depth=depth)


def plain_masks(mucosa: np.ndarray, specular: np.ndarray | None = None) -> Masks:
    z = np.zeros_like(mucosa)
    return Masks(
        mucosa=mucosa.copy(),
        teeth=z.copy(),
        background=~mucosa,
        specular=specular if specular is not None else z.copy(),
        confidence=0.9,
    )


def pink_wl(shape: tuple[int, int]) -> np.ndarray:
    img = np.empty((*shape, 3), np.float32)
    img[:] = np.asarray(MUCOSA_RGB, np.float32)
    return img


def uniform_maps(shape: tuple[int, int], green: float = 0.4) -> AFMaps:
    g = np.full(shape, green, np.float32)
    r = np.full(shape, 0.05, np.float32)
    b = np.full(shape, 0.04, np.float32)
    return AFMaps(green=g, red=r, blue=b, gr_ratio=g / (r + 1e-3), af_index=g / (r + g + b + 1e-3))
