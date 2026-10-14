"""Synthetic lesion shapes and their white-light appearance.

The four kinds mirror what a health worker in an Indian screening camp ticks on the form:

* ``white_patch``: homogeneous leukoplakia. Thick, slightly raised, off-white plaque with a
  fissured ("cracked mud") surface and a fairly well-defined border.
* ``red_patch``: erythroplakia. Velvety deep red, finely granular, sharply demarcated, with a few
  punctate vessels.
* ``mixed_red_white``: speckled leukoplakia / erythroleukoplakia. Red base with white nodules.
* ``ulcer``: yellow-grey fibrinous floor, raised rolled margin and an erythematous halo.

Borders are made irregular by perturbing the radius with low harmonics (lobes) plus fine noise
(a ragged edge). Shapes are rendered on a small local patch, so the scene generator can move a
lesion to a valid spot (fully enclosed by mucosa) without changing its shape.

Ground truth convention: the lesion mask is ``rho < 1`` (the plaque, red area or ulcer including
its rolled margin). An ulcer's erythematous halo is painted outside the mask.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, get_args

import numpy as np

from mukha_pipeline.synthetic._fields import (
    RGB,
    blur,
    fbm,
    gaussian_dots,
    lerp_rgb,
    smoothstep,
)

LesionKind = Literal["white_patch", "red_patch", "mixed_red_white", "ulcer"]
LESION_KINDS: tuple[str, ...] = get_args(LesionKind)

_EDGE_SOFTNESS: dict[str, float] = {
    "white_patch": 0.05,
    "red_patch": 0.06,
    "mixed_red_white": 0.07,
    "ulcer": 0.05,
}
_HALO_REACH = 1.55  # ulcer halo extends to this normalised radius
_LOBE_AMPLITUDE = 0.35  # radius modulation at irregularity 1
_RAGGED_AMPLITUDE = 0.07  # fine border noise at irregularity 1


@dataclass(frozen=True)
class LesionSpec:
    """What lesion to paint and how it should look under violet light.

    ``center`` is (x, y) normalised to the image width and height. ``radii`` are (rx, ry) as
    fractions of the image's SHORTER side, so shapes do not stretch with the aspect ratio.
    ``irregularity`` (0 = smooth ellipse, 1 = very ragged), ``intensity`` (how pronounced the
    colour change is) and ``fvl_depth`` (fractional loss of green autofluorescence) are in 0..1.
    """

    kind: LesionKind
    center: tuple[float, float] = (0.5, 0.5)
    radii: tuple[float, float] = (0.10, 0.07)
    irregularity: float = 0.3
    intensity: float = 0.8
    fvl_depth: float = 0.4
    angle_deg: float = 0.0

    def __post_init__(self) -> None:
        if self.kind not in LESION_KINDS:
            raise ValueError(f"unknown lesion kind {self.kind!r}; expected one of {LESION_KINDS}")
        if not all(0.0 <= c <= 1.0 for c in self.center):
            raise ValueError(f"center must be normalised to 0..1, got {self.center}")
        if not all(0.0 < r <= 0.45 for r in self.radii):
            raise ValueError(f"radii must be in (0, 0.45] of the shorter side, got {self.radii}")
        for name in ("irregularity", "intensity", "fvl_depth"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in 0..1, got {value}")


@dataclass
class LesionPatch:
    """A lesion rendered on a local square patch centred on the lesion centre."""

    rho: np.ndarray  # float32: normalised radial coordinate; the border is rho == 1
    alpha: np.ndarray  # float32: soft lesion coverage (edge anti-aliasing)
    mask: np.ndarray  # bool: ground-truth lesion (rho < 1)
    paint: np.ndarray  # float32: where the renderer may change pixels (adds the ulcer halo)
    radius_px: int  # patch half-size; the patch is (2r+1) x (2r+1)
    scale_px: float  # min(rx, ry) in pixels, the lesion's texture scale

    @property
    def reach_px(self) -> float:
        """Largest distance from the centre to any painted pixel."""
        ys, xs = np.nonzero(self.paint > 0.02)
        if len(xs) == 0:
            return 0.0
        r = self.radius_px
        return float(np.sqrt(((ys - r) ** 2 + (xs - r) ** 2).max()))


def _border_harmonics(rng: np.random.Generator, theta: np.ndarray) -> np.ndarray:
    """Random low-harmonic radius modulation in [-1, 1] (lobes of an irregular border)."""
    ks = np.arange(2, 8)
    amps = rng.standard_normal(len(ks)) / ks**0.8
    phases = rng.uniform(0.0, 2.0 * np.pi, len(ks))
    probe = np.linspace(0.0, 2.0 * np.pi, 720, endpoint=False)

    def series(t: np.ndarray) -> np.ndarray:
        return sum(a * np.cos(k * t + p) for a, k, p in zip(amps, ks, phases, strict=True))

    peak = float(np.abs(series(probe)).max())
    return (series(theta) / (peak if peak > 1e-9 else 1.0)).astype(np.float32)


def make_lesion_patch(
    spec: LesionSpec, image_shape: tuple[int, int], rng: np.random.Generator
) -> LesionPatch:
    """Build the lesion's irregular shape on a local patch (independent of its position)."""
    short = float(min(image_shape[:2]))
    rx = max(spec.radii[0] * short, 2.0)
    ry = max(spec.radii[1] * short, 2.0)
    irr = spec.irregularity
    reach = 1.0 + _LOBE_AMPLITUDE * irr + 3 * _RAGGED_AMPLITUDE * irr + 0.1
    if spec.kind == "ulcer":
        reach *= _HALO_REACH
    r = int(np.ceil(max(rx, ry) * reach)) + 4
    yy, xx = np.mgrid[-r : r + 1, -r : r + 1].astype(np.float32)
    a = np.deg2rad(spec.angle_deg)
    u = (xx * np.cos(a) + yy * np.sin(a)) / rx
    v = (-xx * np.sin(a) + yy * np.cos(a)) / ry
    base = np.sqrt(u * u + v * v)
    theta = np.arctan2(v, u)
    radius_factor = 1.0 + _LOBE_AMPLITUDE * irr * _border_harmonics(rng, theta)
    ragged = fbm(rng, base.shape, max(3.0, 0.2 * min(rx, ry)), octaves=3)
    rho = (base / radius_factor + _RAGGED_AMPLITUDE * irr * ragged).astype(np.float32)
    soft = _EDGE_SOFTNESS[spec.kind]
    alpha = 1.0 - smoothstep(1.0 - soft, 1.0 + soft, rho)
    paint = 1.0 - smoothstep(1.0, _HALO_REACH, rho) if spec.kind == "ulcer" else alpha
    return LesionPatch(
        rho=rho,
        alpha=alpha,
        mask=rho < 1.0,
        paint=paint.astype(np.float32),
        radius_px=r,
        scale_px=float(min(rx, ry)),
    )


# --------------------------------------------------------------------------------------------
# Appearance
# --------------------------------------------------------------------------------------------


def _emboss(field: np.ndarray, scale_px: float, strength: float) -> np.ndarray:
    """Shading of a raised surface lit from the top-left: highlight on the near edge."""
    smooth = blur(field, max(1.0, 0.12 * scale_px))
    gy, gx = np.gradient(smooth)
    return np.clip((gx + gy) * scale_px * strength, -0.15, 0.15).astype(np.float32)


def _paint_white_patch(
    tissue: np.ndarray, p: LesionPatch, spec: LesionSpec, rng: np.random.Generator
) -> np.ndarray:
    shape = p.rho.shape
    s = p.scale_px
    thick = 0.5 + 0.5 * np.tanh(1.3 * fbm(rng, shape, 0.45 * s, 3))
    ridged = 1.0 - np.abs(fbm(rng, shape, 0.28 * s, 3))
    fissure = smoothstep(0.82, 0.96, ridged) * (0.6 + 0.4 * spec.irregularity)
    grain = fbm(rng, shape, 2.5, 2)
    plaque = np.empty((*shape, 3), np.float32)
    plaque[:] = (0.90, 0.86, 0.83)
    plaque *= ((0.95 + 0.07 * thick) * (1.0 + 0.04 * grain))[..., None]
    col = lerp_rgb(plaque, (0.70, 0.50, 0.50), 0.85 * fissure)
    col *= (1.0 + _emboss(p.alpha * (0.7 + 0.3 * thick), s, 0.15))[..., None]
    feather = 0.55 + 0.45 * smoothstep(1.0, 0.72, p.rho)  # plaque thins towards its border
    opacity = p.alpha * (0.45 + 0.45 * spec.intensity) * (0.55 + 0.45 * thick) * feather
    return lerp_rgb(tissue, col, opacity)


def _red_base(
    shape: tuple[int, int], p: LesionPatch, spec: LesionSpec, rng: np.random.Generator
) -> np.ndarray:
    """Velvety, finely granular erythroplakia colour with a few punctate vessels."""
    s = p.scale_px
    low = 0.5 + 0.5 * np.tanh(fbm(rng, shape, 0.5 * s, 3))
    gran = fbm(rng, shape, 1.8, 2)
    red = np.empty((*shape, 3), np.float32)
    red[:] = (0.74, 0.21, 0.24)
    col = lerp_rgb(red, (0.80, 0.30, 0.31), 0.5 * low)
    col *= ((1.0 + 0.12 * gran) * (0.93 + 0.07 * np.clip(p.rho, 0.0, 1.0)))[..., None]
    n_dots = int(np.count_nonzero(p.mask) / 260)
    if n_dots:
        ys, xs = np.nonzero(p.mask)
        pick = rng.choice(len(xs), size=n_dots, replace=len(xs) < n_dots)
        dots = gaussian_dots(shape, ys[pick], xs[pick], rng.uniform(0.5, 1.0, n_dots), 1.0)
        col = lerp_rgb(col, (0.50, 0.08, 0.12), 0.6 * np.clip(dots, 0.0, 1.0))
    return col


def _paint_red_patch(
    tissue: np.ndarray, p: LesionPatch, spec: LesionSpec, rng: np.random.Generator
) -> np.ndarray:
    col = _red_base(p.rho.shape, p, spec, rng)
    return lerp_rgb(tissue, col, p.alpha * (0.45 + 0.42 * spec.intensity))


def _paint_mixed(
    tissue: np.ndarray, p: LesionPatch, spec: LesionSpec, rng: np.random.Generator
) -> np.ndarray:
    shape = p.rho.shape
    col = _red_base(shape, p, spec, rng)
    specks = smoothstep(0.9, 1.7, fbm(rng, shape, 4.5, 2))
    zones = 0.75 * smoothstep(0.9, 1.7, fbm(rng, shape, 0.5 * p.scale_px, 2))
    white_amt = np.maximum(specks, zones) * (0.45 + 0.45 * spec.intensity)
    white = np.empty((*shape, 3), np.float32)
    white[:] = (0.90, 0.85, 0.80)
    white *= (1.0 + 0.04 * fbm(rng, shape, 2.0, 2))[..., None]
    col = lerp_rgb(col, white, 0.85 * white_amt)
    col *= (1.0 + _emboss(white_amt, 6.0, 0.5))[..., None]
    return lerp_rgb(tissue, col, p.alpha * (0.5 + 0.4 * spec.intensity))


def _paint_ulcer(
    tissue: np.ndarray, p: LesionPatch, spec: LesionSpec, rng: np.random.Generator
) -> np.ndarray:
    shape = p.rho.shape
    rho = p.rho
    k = spec.intensity
    halo = smoothstep(_HALO_REACH, 0.95, rho) * smoothstep(0.6, 0.85, rho)
    out = lerp_rgb(tissue, (0.78, 0.30, 0.32), halo * (0.3 + 0.25 * k))
    ring = np.exp(-(((rho - 0.84) / 0.11) ** 2)).astype(np.float32)
    margin: RGB = (0.89, 0.62, 0.60)
    out = lerp_rgb(out, margin, ring * 0.55)
    out *= (1.0 + _emboss(ring, p.scale_px, 0.10))[..., None]
    mottle = 0.5 + 0.5 * np.tanh(fbm(rng, shape, 0.3 * p.scale_px, 3))
    fibrin = np.empty((*shape, 3), np.float32)
    fibrin[:] = (0.83, 0.76, 0.58)
    fibrin = lerp_rgb(fibrin, (0.72, 0.70, 0.63), 0.6 * mottle)
    fibrin *= (1.0 + 0.04 * fbm(rng, shape, 2.0, 2))[..., None]
    granulation = smoothstep(1.0, 1.8, fbm(rng, shape, 2.5, 2))
    fibrin = lerp_rgb(fibrin, (0.75, 0.30, 0.30), 0.6 * granulation)
    floor = 1.0 - smoothstep(0.55, 0.72, rho)
    out = lerp_rgb(out, fibrin, floor * (0.75 + 0.25 * k))
    crater_edge = np.exp(-(((rho - 0.68) / 0.05) ** 2)).astype(np.float32)
    return lerp_rgb(out, (0.55, 0.30, 0.28), 0.3 * crater_edge)


_PAINTERS = {
    "white_patch": _paint_white_patch,
    "red_patch": _paint_red_patch,
    "mixed_red_white": _paint_mixed,
    "ulcer": _paint_ulcer,
}


def paint_lesion(
    tissue: np.ndarray, patch: LesionPatch, spec: LesionSpec, rng: np.random.Generator
) -> np.ndarray:
    """Paint the lesion over a tissue crop of the patch's size; returns the new crop."""
    if tissue.shape[:2] != patch.rho.shape:
        raise ValueError("tissue crop and lesion patch must have the same shape")
    out = _PAINTERS[spec.kind](tissue.astype(np.float32), patch, spec, rng)
    return np.clip(out, 0.0, 1.0).astype(np.float32)
