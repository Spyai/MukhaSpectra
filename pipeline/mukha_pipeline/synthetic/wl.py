"""Synthetic white-light (WL) intraoral close-ups. Every image is labelled SYNTHETIC.

The goal is a plausible phone-torch photo of one oral site, with exact ground truth, so the
segmenter, registration, AF analysis and report can be exercised end to end:

* moist pink-red mucosa: low-frequency colour variation, multi-octave texture, fine branching
  submucosal vessels and small specular speckles from the wet surface;
* site-specific anatomy: tongue papillae and coating, stippled coral gingiva, linea alba on the
  buccal mucosa, palatal rugae and raphe, smooth labial mucosa with minor salivary glands,
  prominent ventral veins;
* a band of off-white teeth with interdental embrasures at the anatomically sensible edge, the
  dark oral cavity, ambient occlusion next to it and torch vignetting;
* an optional lesion (see :mod:`mukha_pipeline.synthetic.lesions`) placed so that it is fully
  enclosed by mucosa and never under the SYNTHETIC stamp.

Layouts are drawn for the patient's LEFT side (or the lower jaw) and mirrored for the other.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Literal

import cv2
import numpy as np

from mukha_pipeline.sites import OralSite
from mukha_pipeline.synthetic._fields import (
    RGB,
    blur,
    child_rngs,
    dilate,
    fbm,
    gaussian_dots,
    lerp_rgb,
    norm_grids,
    smoothstep,
    soft_ellipse,
)
from mukha_pipeline.synthetic.lesions import (
    LesionPatch,
    LesionSpec,
    make_lesion_patch,
    paint_lesion,
)
from mukha_pipeline.synthetic.watermark import stamp_mask, watermark_synthetic

DEFAULT_SIZE: tuple[int, int] = (768, 1024)  # (H, W), landscape phone frame

Edge = Literal["top", "bottom", "left", "right"]


@dataclass
class SyntheticScene:
    """One synthetic WL capture and its ground truth (all masks bool (H, W))."""

    wl: np.ndarray  # float32 RGB in [0, 1], SYNTHETIC stamp included
    tissue_mask: np.ndarray  # visible oral mucosa, lesion and wet highlights included
    teeth_mask: np.ndarray
    lesion_mask: np.ndarray  # ground-truth lesion (all False without a lesion)
    background_mask: np.ndarray  # dark oral cavity, embrasures and the SYNTHETIC stamp
    specular_mask: np.ndarray  # wet-surface highlights (on tissue or teeth)
    site: OralSite
    lesion: LesionSpec | None = None  # as placed; its centre may have moved onto mucosa


# --------------------------------------------------------------------------------------------
# Canvas and layers
# --------------------------------------------------------------------------------------------


@dataclass
class _Ctx:
    rng: np.random.Generator
    h: int
    w: int

    def __post_init__(self) -> None:
        self.s = min(self.h, self.w) / 768.0  # texture scale relative to the default frame
        self.X, self.Y = norm_grids((self.h, self.w))

    @property
    def shape(self) -> tuple[int, int]:
        return self.h, self.w

    def noise(self, cell: float, octaves: int = 4) -> np.ndarray:
        """fBm noise with features ``cell`` pixels wide at the default 768 px frame height."""
        return fbm(self.rng, self.shape, max(cell * self.s, 1.5), octaves)

    def uniform(self, color: RGB) -> np.ndarray:
        out = np.empty((self.h, self.w, 3), np.float32)
        out[:] = color
        return out


@dataclass(frozen=True)
class _TeethSpec:
    edge: Edge
    start: float  # extent along the edge, 0..1
    end: float
    depth: float  # crown depth from the edge, fraction of the shorter side
    tooth_width: float  # fraction of the edge length
    curve: float = 0.0  # extra depth at the middle of the extent (arch), fraction of short side
    papilla: float = 0.45  # interdental papilla tip depth relative to crown depth


@dataclass
class _Layers:
    base: np.ndarray  # (H, W, 3) tissue colour before lighting
    deep: RGB  # low-frequency colour variation: towards this redder tone...
    pale: RGB  # ...and towards this paler tone
    shadow: np.ndarray  # dark oral cavity alpha
    relief: np.ndarray  # height field (pixels) for shading: folds, ridges, papillae
    vessel_weight: np.ndarray  # where submucosal vessels show through
    vessel_trees: int = 9
    vessel_color: RGB = (0.60, 0.16, 0.22)
    vessel_strength: float = 0.34
    teeth: _TeethSpec | None = None
    gloss: float = 1.0  # specular abundance
    overlays: list[tuple[RGB, np.ndarray]] = field(default_factory=list)


def _new_layers(ctx: _Ctx, base: RGB, deep: RGB, pale: RGB) -> _Layers:
    zeros = np.zeros(ctx.shape, np.float32)
    return _Layers(
        base=ctx.uniform(base),
        deep=deep,
        pale=pale,
        shadow=zeros.copy(),
        relief=zeros.copy(),
        vessel_weight=np.ones(ctx.shape, np.float32),
    )


def _add_shadow(ctx: _Ctx, lay: _Layers, cx: float, cy: float, rx: float, ry: float) -> None:
    """Dark oral cavity (vestibule, oropharynx) as a soft ellipse with a wobbly edge."""
    wobble = 0.05 * ctx.noise(170, 2) + 0.012 * ctx.noise(45, 2)
    ell = soft_ellipse(ctx.X, ctx.Y, cx, cy, rx, ry, 0.07, wobble)
    lay.shadow = np.maximum(lay.shadow, ell)


# --------------------------------------------------------------------------------------------
# Reusable anatomy features
# --------------------------------------------------------------------------------------------


def _line_field(
    ctx: _Ctx,
    *,
    horizontal: bool,
    pos: float,
    width_px: float,
    amp: float = 0.012,
    freq: float = 2.0,
    start: float = 0.0,
    end: float = 1.0,
) -> np.ndarray:
    """Gaussian profile across a gently wavy line (linea alba, raphe, sulcus, frenum)."""
    along, across = (ctx.X, ctx.Y) if horizontal else (ctx.Y, ctx.X)
    size = ctx.h if horizontal else ctx.w
    phase = float(ctx.rng.uniform(0, 2 * np.pi))
    centre = pos + amp * np.sin(2 * np.pi * freq * along + phase) + 0.004 * ctx.noise(60, 2)
    dist_px = (across - centre) * size
    w = max(width_px * ctx.s, 1.0)
    prof = np.exp(-0.5 * (dist_px / w) ** 2).astype(np.float32)
    fade = smoothstep(start - 0.05, start + 0.05, along) * (1.0 - smoothstep(end - 0.05, end + 0.05, along))
    return prof * fade


def _papillae(
    ctx: _Ctx, region: np.ndarray, density: float
) -> tuple[np.ndarray, np.ndarray]:
    """Filiform (dense, tiny, whitish tips) and fungiform (sparse, red dots) papillae."""
    area = ctx.h * ctx.w
    n_fil = int(density * area / (55.0 * ctx.s**2) * float(region.mean()))
    n_fun = max(1, n_fil // 45)
    out = []
    for n, sigma, amp in ((n_fil, 1.0, (0.5, 1.0)), (n_fun, 2.0, (0.6, 1.0))):
        flat = region.ravel().astype(np.float64)
        total = float(flat.sum())
        if n <= 0 or total <= 0:
            out.append(np.zeros(ctx.shape, np.float32))
            continue
        idx = ctx.rng.choice(area, size=n, p=flat / total)
        ys, xs = np.divmod(idx, ctx.w)
        jitter = ctx.rng.uniform(-0.5, 0.5, (2, n))
        dots = gaussian_dots(
            ctx.shape, ys + jitter[0], xs + jitter[1], ctx.rng.uniform(*amp, n), sigma * ctx.s
        )
        out.append(np.clip(dots, 0.0, 1.0))
    return out[0], out[1]


def _bumps(ctx: _Ctx, centres: list[tuple[float, float]], radius_px: float) -> np.ndarray:
    """Soft dome-shaped bumps (papilla, caruncles, glands) at normalised centres."""
    out = np.zeros(ctx.shape, np.float32)
    r = max(radius_px * ctx.s, 1.5)
    for cx, cy in centres:
        d = np.sqrt(((ctx.X - cx) * ctx.w) ** 2 + ((ctx.Y - cy) * ctx.h) ** 2) / r
        out = np.maximum(out, np.clip(1.0 - d * d, 0.0, 1.0))
    return out


def _smooth_curve(
    ctx: _Ctx, start: tuple[float, float], end: tuple[float, float], bend: float
) -> np.ndarray:
    """Pixel points of a gently bent curve between two normalised points."""
    t = np.linspace(0.0, 1.0, 64)
    x0, y0 = start
    x1, y1 = end
    nx, ny = -(y1 - y0), (x1 - x0)
    wiggle = bend * np.sin(np.pi * t) + 0.018 * np.sin(2 * np.pi * (3.0 * t + ctx.rng.uniform()))
    xs = (x0 + (x1 - x0) * t + nx * wiggle) * ctx.w
    ys = (y0 + (y1 - y0) * t + ny * wiggle) * ctx.h
    return np.stack([xs, ys], axis=1)


def _draw_bands(ctx: _Ctx, curves: list[np.ndarray], width_px: float) -> np.ndarray:
    """Thick, soft bands along curves (ventral veins, palatal rugae), drawn anti-aliased."""
    canvas = np.zeros(ctx.shape, np.uint8)
    for pts in curves:
        w = max(1, int(round(width_px * ctx.s * ctx.rng.uniform(0.7, 1.2))))
        arr = np.round(pts * 16).astype(np.int32)
        cv2.polylines(canvas, [arr], False, 255, thickness=w, lineType=cv2.LINE_AA, shift=4)
    soft = blur(canvas.astype(np.float32) / 255.0, 0.3 * width_px * ctx.s)
    return np.clip(soft * (0.7 + 0.3 * smoothstep(-1.0, 1.0, ctx.noise(50, 2))), 0.0, 1.0)


def _vessel_map(ctx: _Ctx, n_trees: int, weight: np.ndarray) -> np.ndarray:
    """Fine branching submucosal vessels: random walks that fork and thin out."""
    canvas = np.zeros(ctx.shape, np.uint8)
    flat = np.clip(weight, 0.0, None).ravel().astype(np.float64)
    if n_trees <= 0 or flat.sum() <= 0:
        return np.zeros(ctx.shape, np.float32)
    rng = ctx.rng
    starts = rng.choice(flat.size, size=n_trees, p=flat / flat.sum())
    step = 5.0 * ctx.s
    for start in starts:
        y, x = divmod(int(start), ctx.w)
        theta, width = rng.uniform(0, 2 * np.pi), rng.uniform(1.6, 3.2)
        stack = [(float(x), float(y), theta, width, int(rng.integers(40, 110)), 0)]
        while stack:
            px, py, th, width, n, depth = stack.pop()
            pts = [(px, py)]
            for _ in range(n):
                th += rng.normal(0.0, 0.16)
                px += np.cos(th) * step
                py += np.sin(th) * step
                pts.append((px, py))
                if not (-20 < px < ctx.w + 20 and -20 < py < ctx.h + 20):
                    break
                if depth < 3 and rng.random() < 0.05:
                    turn = rng.choice([-1.0, 1.0]) * rng.uniform(0.35, 0.9)
                    stack.append((px, py, th + turn, width * 0.68, int(n * 0.55), depth + 1))
            value = int(255 * min(1.0, 0.3 + 0.22 * width))
            thick = max(1, int(round(width * ctx.s)))
            arr = np.round(np.asarray(pts) * 16).astype(np.int32)
            cv2.polylines(canvas, [arr], False, value, thickness=thick, lineType=cv2.LINE_AA,
                          shift=4)
    vessels = blur(canvas.astype(np.float32) / 255.0, 0.6 * ctx.s)
    fade = smoothstep(-0.7, 0.6, ctx.noise(90, 3))  # vessels dive in and out of view
    return np.clip(vessels, 0.0, 1.0) * (0.3 + 0.7 * fade) * np.clip(weight, 0.0, 1.0)


# --------------------------------------------------------------------------------------------
# Teeth
# --------------------------------------------------------------------------------------------


@dataclass
class _TeethLayer:
    alpha: np.ndarray  # crown coverage
    gap: np.ndarray  # dark interdental embrasures
    rgb: np.ndarray  # crown colour before lighting
    highlight: np.ndarray  # enamel highlight streaks


def _edge_coords(ctx: _Ctx, edge: Edge) -> tuple[np.ndarray, np.ndarray, int]:
    """(u along the edge 0..1, v distance from the edge in px, edge length in px)."""
    ys = (ctx.Y * ctx.h - 0.5).astype(np.float32)
    xs = (ctx.X * ctx.w - 0.5).astype(np.float32)
    if edge == "bottom":
        return ctx.X, (ctx.h - 1) - ys, ctx.w
    if edge == "top":
        return ctx.X, ys, ctx.w
    if edge == "left":
        return ctx.Y, xs, ctx.h
    return ctx.Y, (ctx.w - 1) - xs, ctx.h


def _teeth(ctx: _Ctx, spec: _TeethSpec, stain: float) -> _TeethLayer:
    rng = ctx.rng
    u, v, length = _edge_coords(ctx, spec.edge)
    short = float(min(ctx.h, ctx.w))
    bounds = [spec.start]
    while bounds[-1] < spec.end:
        bounds.append(bounds[-1] + spec.tooth_width * rng.uniform(0.88, 1.12))
    b = np.asarray(bounds, np.float32)
    n = len(b) - 1
    idx = np.clip(np.searchsorted(b, u, side="right") - 1, 0, n - 1)
    lo, hi = b[idx], b[idx + 1]
    half = (hi - lo) / 2.0
    t = (u - (lo + half)) / half
    inside = (u >= b[0]) & (u <= b[-1])
    depth_i = rng.uniform(0.92, 1.08, n).astype(np.float32)[idx] * spec.depth * short
    mid = (spec.start + spec.end) / 2.0
    span = max((spec.end - spec.start) / 2.0, 1e-3)
    depth_i = depth_i + spec.curve * short * np.clip(1.0 - ((u - mid) / span) ** 2, 0.0, 1.0)
    papilla = spec.papilla * depth_i
    t2 = np.clip(1.0 - t * t, 0.0, 1.0)
    margin = papilla + (depth_i - papilla) * np.sqrt(t2)
    alpha = (1.0 - smoothstep(-1.2, 1.2, v - margin)) * inside
    # Embrasures: dark gaps between neighbouring crowns, widening towards the gum.
    dist_px = (1.0 - np.abs(t)) * half * length
    gap_w = max(1.0, 2.2 * ctx.s) * (0.5 + 1.2 * np.clip(v / np.maximum(margin, 1.0), 0, 1))
    gap = (1.0 - smoothstep(0.0, gap_w, dist_px)) * alpha
    # Crown colour: off-white, yellower at the neck, translucent at the incisal edge.
    vn = v / np.maximum(margin, 1.0)  # 0 at the image edge, 1 at the gum line
    shade = rng.uniform(0.94, 1.04, n).astype(np.float32)[idx]
    rgb = ctx.uniform((0.97, 0.93, 0.84)) * shade[..., None]
    cervical = smoothstep(0.5, 1.0, vn)
    rgb = lerp_rgb(rgb, (0.88, 0.79, 0.62), 0.55 * cervical)
    if stain > 0:
        blotch = 0.6 + 0.4 * smoothstep(-0.5, 0.8, ctx.noise(18, 3))
        rgb = lerp_rgb(rgb, (0.58, 0.42, 0.26), stain * cervical * blotch)
    incisal = 1.0 - smoothstep(0.0, 0.3, vn)
    rgb = lerp_rgb(rgb, (0.80, 0.82, 0.85), 0.3 * incisal)
    curvature = (0.6 + 0.4 * np.sqrt(t2)) * (1.0 - 0.1 * smoothstep(0.7, 1.0, vn))
    rgb *= (curvature * (1.0 + 0.008 * ctx.noise(5, 2)))[..., None]
    hx = rng.uniform(-0.45, -0.2, n).astype(np.float32)[idx]
    hy = rng.uniform(0.35, 0.55, n).astype(np.float32)[idx]
    blob = np.exp(-(((t - hx) / 0.13) ** 2) - ((vn - hy) / 0.2) ** 2)
    highlight = (0.75 * blob * alpha * (1.0 - gap)).astype(np.float32)
    return _TeethLayer(alpha.astype(np.float32), gap.astype(np.float32), rgb, highlight)


# --------------------------------------------------------------------------------------------
# Site painters (canonical orientation: patient's left side, or the lower jaw)
# --------------------------------------------------------------------------------------------


def _paint_buccal(ctx: _Ctx) -> _Layers:
    lay = _new_layers(ctx, (0.80, 0.47, 0.47), (0.72, 0.30, 0.33), (0.86, 0.62, 0.60))
    lay.teeth = _TeethSpec("bottom", 0.30, 1.02, 0.17, 0.16, papilla=0.72)
    _add_shadow(ctx, lay, 1.04, 0.02, 0.30, 0.30)
    _add_shadow(ctx, lay, 0.06, -0.06, 0.16, 0.12)
    alba = _line_field(ctx, horizontal=True, pos=0.66, width_px=4.0, amp=0.01, start=0.12)
    lay.overlays.append(((0.88, 0.70, 0.68), 0.32 * alba))
    lay.relief += 1.5 * alba
    papilla = _bumps(ctx, [(0.80, 0.30)], 10.0)
    lay.overlays.append(((0.86, 0.58, 0.58), 0.4 * papilla))
    lay.relief += 3.0 * papilla
    # Fordyce granules: ectopic sebaceous glands, tiny pale-yellow dots, common on the cheek.
    n = 45
    xs = ctx.rng.normal(0.28, 0.07, n) * ctx.w
    ys = ctx.rng.normal(0.45, 0.07, n) * ctx.h
    fordyce = gaussian_dots(ctx.shape, ys, xs, ctx.rng.uniform(0.4, 1.0, n), 1.3 * ctx.s)
    lay.overlays.append(((0.90, 0.80, 0.58), 0.35 * np.clip(fordyce, 0, 1)))
    lay.relief += 2.0 * ctx.noise(120, 2)
    lay.vessel_trees = 9
    return lay


def _paint_lateral_tongue(ctx: _Ctx) -> _Layers:
    """Tongue pulled aside: dorsal silhouette at the top, lateral border, ventral surface below."""
    lay = _new_layers(ctx, (0.82, 0.45, 0.46), (0.74, 0.28, 0.31), (0.86, 0.58, 0.57))
    phase = float(ctx.rng.uniform(0, 2 * np.pi))
    top = 0.10 + 0.05 * np.sin(np.pi * ctx.X + phase * 0.2) + 0.01 * ctx.noise(120, 2)
    lay.shadow = np.maximum(lay.shadow, 1.0 - smoothstep(-0.02, 0.02, ctx.Y - top))
    _add_shadow(ctx, lay, 0.5, 1.14, 0.80, 0.19)
    _add_shadow(ctx, lay, 1.10, 0.50, 0.12, 0.40)
    rel = ctx.Y - top
    dorsum = 1.0 - smoothstep(0.17, 0.26, rel)
    ventral = smoothstep(0.50, 0.60, rel)
    fil, fun = _papillae(ctx, dorsum, 1.3)
    coat = dorsum * (0.45 + 0.55 * smoothstep(-0.6, 0.8, ctx.noise(110, 3)))
    lay.overlays.append(((0.88, 0.80, 0.74), 0.42 * coat * fil + 0.12 * coat))
    lay.overlays.append(((0.83, 0.30, 0.33), 0.5 * fun))
    lay.relief += 1.6 * fil + 1.0 * fun
    # Foliate papillae: a few irregular vertical folds at the posterior lateral border.
    folds = 0.5 + 0.5 * np.sin(2 * np.pi * ctx.X / 0.035 + 1.2 * ctx.noise(60, 2))
    band = smoothstep(0.70, 0.82, ctx.X) * np.exp(-(((rel - 0.30) / 0.11) ** 2))
    lay.relief += 4.0 * folds * band
    lay.overlays.append(((0.68, 0.30, 0.33), 0.22 * (1.0 - folds) ** 2 * band))
    lay.overlays.append(((0.78, 0.44, 0.52), 0.45 * ventral))
    veins = [
        _smooth_curve(ctx, (-0.02, 0.70 + 0.09 * k), (1.02, 0.74 + 0.08 * k), 0.04 * (k - 0.5))
        for k in range(2)
    ]
    vein_map = _draw_bands(ctx, veins, 15.0) * ventral
    lay.overlays.append(((0.56, 0.32, 0.48), 0.38 * vein_map))
    lay.vessel_weight = 0.3 + 0.7 * (1.0 - dorsum)
    lay.vessel_trees = 7
    lay.relief += 1.2 * ctx.noise(100, 2)
    return lay


def _paint_tongue_dorsum(ctx: _Ctx) -> _Layers:
    lay = _new_layers(ctx, (0.80, 0.46, 0.46), (0.74, 0.30, 0.33), (0.86, 0.60, 0.58))
    half_width = 0.47 - 0.13 * np.clip((ctx.Y - 0.5) / 0.5, 0.0, 1.0) ** 2
    d = np.abs(ctx.X - 0.5) / half_width + 0.015 * ctx.noise(150, 2)
    tongue = 1.0 - smoothstep(0.95, 1.02, d)
    lay.shadow = np.maximum(lay.shadow, 1.0 - tongue)
    _add_shadow(ctx, lay, 0.5, -0.12, 0.36, 0.20)
    region = tongue.copy()
    fil, fun = _papillae(ctx, region, 1.6)
    coat_field = smoothstep(-0.4, 0.9, ctx.noise(120, 3)) * (1.0 - smoothstep(0.35, 0.8, ctx.Y))
    coat = region * (0.25 + 0.75 * coat_field)
    lay.overlays.append(((0.89, 0.82, 0.74), 0.45 * coat * fil + 0.16 * coat))
    lay.overlays.append(((0.83, 0.30, 0.33), 0.5 * fun * smoothstep(0.35, 0.8, ctx.Y)))
    sulcus = _line_field(ctx, horizontal=False, pos=0.5, width_px=5.0, amp=0.008, start=0.05,
                         end=0.85)
    lay.overlays.append(((0.70, 0.34, 0.37), 0.35 * sulcus))
    lay.relief += 1.6 * fil + 1.0 * fun - 3.0 * sulcus + 1.5 * ctx.noise(100, 2)
    lay.vessel_weight = np.full(ctx.shape, 0.25, np.float32)
    lay.vessel_trees = 4
    lay.gloss = 0.8
    return lay


def _paint_gingiva(ctx: _Ctx) -> _Layers:
    """Lower gingiva: teeth at the top, coral attached gingiva, redder alveolar mucosa."""
    lay = _new_layers(ctx, (0.90, 0.62, 0.57), (0.84, 0.48, 0.47), (0.92, 0.70, 0.65))
    lay.teeth = _TeethSpec("top", -0.02, 1.02, 0.28, 0.12, papilla=0.68)
    _add_shadow(ctx, lay, 0.5, 1.16, 0.75, 0.18)
    mgj = 0.66 + 0.03 * np.sin(2 * np.pi * 1.5 * ctx.X) + 0.015 * ctx.noise(70, 2)
    alveolar = smoothstep(-0.04, 0.05, ctx.Y - mgj)
    lay.overlays.append(((0.80, 0.40, 0.42), 0.85 * alveolar))
    stipple = ctx.noise(4.5, 2)
    lay.relief += 0.35 * stipple * (1.0 - alveolar)
    lay.overlays.append(((0.80, 0.52, 0.50), 0.06 * np.clip(-stipple, 0, 1) * (1 - alveolar)))
    lay.vessel_weight = 0.15 + 0.85 * alveolar
    lay.vessel_trees = 10
    lay.gloss = 0.8
    return lay


def _paint_lip(ctx: _Ctx) -> _Layers:
    """Everted lower lip: lower incisors at the top, smooth bluish-pink labial mucosa."""
    lay = _new_layers(ctx, (0.80, 0.42, 0.48), (0.72, 0.28, 0.35), (0.86, 0.58, 0.62))
    lay.teeth = _TeethSpec("top", 0.18, 0.82, 0.13, 0.105, papilla=0.62)
    _add_shadow(ctx, lay, 0.02, -0.02, 0.20, 0.16)
    _add_shadow(ctx, lay, 0.98, -0.02, 0.20, 0.16)
    n = int(70 * ctx.rng.uniform(0.8, 1.2))
    xs = ctx.rng.uniform(0.05, 0.95, n) * ctx.w
    ys = ctx.rng.uniform(0.3, 0.95, n) * ctx.h
    glands = gaussian_dots(ctx.shape, ys, xs, ctx.rng.uniform(0.5, 1.0, n), 3.0 * ctx.s)
    glands = np.clip(glands, 0, 1)
    lay.overlays.append(((0.86, 0.60, 0.64), 0.30 * glands))
    lay.relief += 2.5 * glands + 0.8 * ctx.noise(140, 2)
    lay.vessel_trees = 12
    lay.vessel_strength = 0.5
    lay.gloss = 1.2
    return lay


def _paint_hard_palate(ctx: _Ctx) -> _Layers:
    lay = _new_layers(ctx, (0.88, 0.63, 0.62), (0.82, 0.50, 0.50), (0.92, 0.72, 0.70))
    lay.teeth = _TeethSpec("top", -0.05, 1.05, 0.17, 0.10, curve=-0.09, papilla=0.62)
    _add_shadow(ctx, lay, 0.5, 1.2, 0.55, 0.22)
    anterior = 1.0 - smoothstep(0.30, 0.55, ctx.Y)
    raphe = _line_field(ctx, horizontal=False, pos=0.5, width_px=3.0, amp=0.004, start=0.18)
    curves = []
    for side in (-1.0, 1.0):
        for k in range(4):
            y0 = 0.24 + 0.075 * k + ctx.rng.uniform(-0.015, 0.015)
            length = ctx.rng.uniform(0.14, 0.26)
            end = (0.5 + side * (0.03 + length), y0 + ctx.rng.uniform(0.0, 0.06))
            curves.append(_smooth_curve(ctx, (0.5 + side * 0.03, y0), end, side * 0.05))
    rugae = _draw_bands(ctx, curves, 9.0) * anterior
    lay.overlays.append(((0.93, 0.74, 0.72), 0.45 * rugae))
    lay.overlays.append(((0.93, 0.76, 0.74), 0.4 * raphe))
    papilla = _bumps(ctx, [(0.5, 0.2)], 14.0)
    lay.overlays.append(((0.92, 0.70, 0.68), 0.35 * papilla))
    posterior = smoothstep(0.7, 0.95, ctx.Y)
    lay.overlays.append(((0.80, 0.40, 0.42), 0.6 * posterior))
    lay.relief += 4.0 * rugae + 1.0 * raphe + 4.0 * papilla + 0.25 * ctx.noise(3.0, 2)
    lay.vessel_weight = 0.1 + 0.9 * posterior
    lay.vessel_trees = 5
    lay.gloss = 0.7
    return lay


def _paint_soft_palate(ctx: _Ctx) -> _Layers:
    lay = _new_layers(ctx, (0.80, 0.38, 0.40), (0.72, 0.25, 0.29), (0.86, 0.52, 0.53))
    _add_shadow(ctx, lay, 0.29, 1.02, 0.19, 0.27)
    _add_shadow(ctx, lay, 0.71, 1.02, 0.19, 0.27)
    hard = 1.0 - smoothstep(0.08, 0.2, ctx.Y)
    lay.overlays.append(((0.88, 0.62, 0.61), 0.7 * hard))
    lay.relief += 1.2 * ctx.noise(120, 2)
    lay.vessel_trees = 14
    lay.vessel_strength = 0.5
    return lay


def _paint_retromolar(ctx: _Ctx) -> _Layers:
    lay = _new_layers(ctx, (0.82, 0.48, 0.48), (0.74, 0.32, 0.34), (0.88, 0.62, 0.60))
    lay.teeth = _TeethSpec("bottom", -0.02, 0.40, 0.24, 0.19, papilla=0.72)
    _add_shadow(ctx, lay, 1.02, -0.02, 0.36, 0.36)
    fold = _line_field(ctx, horizontal=False, pos=0.62, width_px=14.0, amp=0.02, start=0.0)
    lay.overlays.append(((0.86, 0.58, 0.57), 0.2 * fold))
    lay.relief += 6.0 * fold + 1.5 * ctx.noise(110, 2)
    lay.vessel_trees = 8
    return lay


def _paint_ventral_fom(ctx: _Ctx) -> _Layers:
    lay = _new_layers(ctx, (0.82, 0.42, 0.45), (0.74, 0.28, 0.32), (0.86, 0.56, 0.58))
    lay.teeth = _TeethSpec("bottom", 0.26, 0.74, 0.10, 0.10, papilla=0.62)
    _add_shadow(ctx, lay, 0.04, 1.06, 0.24, 0.20)
    _add_shadow(ctx, lay, 0.96, 1.06, 0.24, 0.20)
    ventral = 1.0 - smoothstep(0.50, 0.58, ctx.Y)
    lay.overlays.append(((0.78, 0.42, 0.52), 0.5 * ventral))
    veins = [
        _smooth_curve(ctx, (0.38, -0.02), (0.44, 0.50), 0.06),
        _smooth_curve(ctx, (0.62, -0.02), (0.56, 0.50), -0.06),
    ]
    lay.overlays.append(((0.56, 0.32, 0.48), 0.4 * _draw_bands(ctx, veins, 16.0)))
    frenum = _line_field(ctx, horizontal=False, pos=0.5, width_px=7.0, amp=0.004, start=0.32,
                         end=0.86)
    crease = _line_field(ctx, horizontal=True, pos=0.55, width_px=8.0, amp=0.025, freq=1.0)
    caruncles = _bumps(ctx, [(0.46, 0.8), (0.54, 0.8)], 9.0)
    lay.overlays.append(((0.88, 0.62, 0.64), 0.35 * frenum + 0.3 * caruncles))
    lay.overlays.append(((0.64, 0.28, 0.31), 0.18 * crease))
    lay.relief += 4.0 * frenum + 3.0 * caruncles - 2.0 * crease + 1.0 * ctx.noise(110, 2)
    lay.vessel_trees = 12
    lay.vessel_strength = 0.5
    lay.gloss = 1.2
    return lay


Painter = Callable[[_Ctx], _Layers]

# site -> (painter, flip horizontally, flip vertically) from the canonical layout
_SITE_PLAN: dict[OralSite, tuple[Painter, bool, bool]] = {
    OralSite.BUCCAL_MUCOSA_LEFT: (_paint_buccal, False, False),
    OralSite.BUCCAL_MUCOSA_RIGHT: (_paint_buccal, True, False),
    OralSite.LATERAL_TONGUE_LEFT: (_paint_lateral_tongue, False, False),
    OralSite.LATERAL_TONGUE_RIGHT: (_paint_lateral_tongue, True, False),
    OralSite.TONGUE_DORSUM: (_paint_tongue_dorsum, False, False),
    OralSite.VENTRAL_TONGUE_FLOOR_OF_MOUTH: (_paint_ventral_fom, False, False),
    OralSite.LOWER_GINGIVA: (_paint_gingiva, False, False),
    OralSite.UPPER_GINGIVA: (_paint_gingiva, False, True),
    OralSite.LOWER_LIP_MUCOSA: (_paint_lip, False, False),
    OralSite.UPPER_LIP_MUCOSA: (_paint_lip, False, True),
    OralSite.HARD_PALATE: (_paint_hard_palate, False, False),
    OralSite.SOFT_PALATE: (_paint_soft_palate, False, False),
    OralSite.RETROMOLAR_LEFT: (_paint_retromolar, False, False),
    OralSite.RETROMOLAR_RIGHT: (_paint_retromolar, True, False),
}
assert set(_SITE_PLAN) == set(OralSite), "every OralSite needs a synthetic layout"


# --------------------------------------------------------------------------------------------
# Lesion placement
# --------------------------------------------------------------------------------------------


def _place_lesion(
    spec: LesionSpec, allowed: np.ndarray, rng: np.random.Generator
) -> tuple[LesionSpec, LesionPatch, int, int]:
    """Find the nearest spot to the requested centre where the lesion sits fully on mucosa.

    Returns (spec as placed, patch, centre row, centre column). Shrinks the lesion if even the
    roomiest spot is too small.
    """
    h, w = allowed.shape
    margin = max(4.0, 0.02 * min(h, w))
    padded = np.pad(allowed.astype(np.uint8), 1)  # the image border counts as "not mucosa"
    dist = cv2.distanceTransform(padded, cv2.DIST_L2, 5)[1:-1, 1:-1]
    for _ in range(6):
        patch = make_lesion_patch(spec, (h, w), rng)
        need = patch.reach_px + margin
        valid = dist >= need
        if valid.any():
            cy = int(np.clip(round(spec.center[1] * h - 0.5), 0, h - 1))
            cx = int(np.clip(round(spec.center[0] * w - 0.5), 0, w - 1))
            if not valid[cy, cx]:
                ys, xs = np.nonzero(valid)
                k = int(np.argmin((ys - cy) ** 2 + (xs - cx) ** 2))
                cy, cx = int(ys[k]), int(xs[k])
            placed = replace(spec, center=((cx + 0.5) / w, (cy + 0.5) / h))
            return placed, patch, cy, cx
        factor = max(0.3, 0.9 * (float(dist.max()) - margin) / max(patch.reach_px, 1.0))
        spec = replace(spec, radii=(spec.radii[0] * factor, spec.radii[1] * factor))
    raise ValueError("no room on the mucosa for the requested lesion")


def _paint_placed_lesion(
    base: np.ndarray,
    spec: LesionSpec,
    patch: LesionPatch,
    cy: int,
    cx: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Composite the lesion patch into the base colour; returns (base, mask, soft coverage)."""
    r = patch.radius_px
    h, w = base.shape[:2]
    padded = np.pad(base, ((r, r), (r, r), (0, 0)), mode="edge")
    crop = padded[cy : cy + 2 * r + 1, cx : cx + 2 * r + 1]
    padded[cy : cy + 2 * r + 1, cx : cx + 2 * r + 1] = paint_lesion(crop, patch, spec, rng)
    mask = np.zeros((h + 2 * r, w + 2 * r), bool)
    mask[cy : cy + 2 * r + 1, cx : cx + 2 * r + 1] = patch.mask
    cover = np.zeros((h + 2 * r, w + 2 * r), np.float32)
    cover[cy : cy + 2 * r + 1, cx : cx + 2 * r + 1] = patch.alpha
    crop_back = (slice(r, r + h), slice(r, r + w))
    return padded[crop_back].copy(), mask[crop_back].copy(), cover[crop_back].copy()


# --------------------------------------------------------------------------------------------
# Lighting, highlights and camera
# --------------------------------------------------------------------------------------------


def _relief_shading(ctx: _Ctx, relief: np.ndarray) -> np.ndarray:
    """Lambert-like shading of a height field lit from the top-left (torch beside the lens)."""
    gy, gx = np.gradient(blur(relief, 1.0 * ctx.s))
    return np.clip(1.0 + 0.18 * (gx + gy), 0.75, 1.25).astype(np.float32)


def _torch(ctx: _Ctx) -> tuple[np.ndarray, np.ndarray]:
    """(vignette multiplier, hot-spot field) of the phone torch."""
    cx = 0.5 + ctx.rng.uniform(-0.06, 0.06)
    cy = 0.45 + ctx.rng.uniform(-0.06, 0.06)
    aspect = ctx.w / ctx.h
    r2 = ((ctx.X - cx) * aspect) ** 2 + (ctx.Y - cy) ** 2
    vignette = np.clip(1.0 - 0.40 * r2 / 0.55, 0.5, 1.0).astype(np.float32)
    hot = np.exp(-r2 / 0.10).astype(np.float32)
    return vignette, hot


def _speculars(ctx: _Ctx, region: np.ndarray, hot: np.ndarray, gloss: float) -> np.ndarray:
    """Wet-surface highlights: short streaks of micro-glints and a few soft glossy patches,
    clustered towards the torch hot spot, each with a faint halo."""
    rng = ctx.rng
    weights = (region * (0.04 + hot**4)).ravel().astype(np.float64)
    n = int(round(30 * gloss * ctx.h * ctx.w / (768 * 1024)))
    canvas = np.zeros(ctx.shape, np.uint8)
    if n <= 0 or weights.sum() <= 0:
        return np.zeros(ctx.shape, np.float32)
    picks = rng.choice(weights.size, size=n, p=weights / weights.sum())
    for p in picks:
        y, x = divmod(int(p), ctx.w)
        patch = rng.random() < 0.25
        n_dots = 1 if patch else int(rng.integers(1, 4))
        ang = rng.uniform(0, np.pi)
        for k in range(n_dots):
            ox = rng.normal(0, 3.0 * ctx.s) if k else 0.0
            oy = rng.normal(0, 3.0 * ctx.s) if k else 0.0
            a = (rng.uniform(4.0, 10.0) if patch else rng.uniform(0.9, 2.8)) * ctx.s
            b = a * rng.uniform(0.35, 0.8)
            centre = (int(round((x + ox) * 16)), int(round((y + oy) * 16)))
            axes = (max(1, int(round(a * 16))), max(1, int(round(b * 16))))
            value = int(255 * (rng.uniform(0.35, 0.6) if patch else rng.uniform(0.75, 1.0)))
            cv2.ellipse(canvas, centre, axes, float(np.rad2deg(ang)) + rng.normal(0, 15), 0, 360,
                        value, -1, cv2.LINE_AA, 4)
    core = blur(canvas.astype(np.float32) / 255.0, 0.7 * ctx.s)
    halo = blur(core, 3.5 * ctx.s)
    return np.clip(core + 0.6 * halo, 0.0, 1.0) * region


def _apply_camera(img: np.ndarray, rng: np.random.Generator, s: float) -> np.ndarray:
    """Slight lens blur plus luma/chroma sensor noise."""
    out = blur(img, 0.55 * s)
    h, w = out.shape[:2]
    out += rng.normal(0.0, 0.006, (h, w, 1)).astype(np.float32)
    out += rng.normal(0.0, 0.003, (h, w, 3)).astype(np.float32)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


# --------------------------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------------------------


def _flip(a: np.ndarray, flip_h: bool, flip_v: bool) -> np.ndarray:
    if flip_h:
        a = a[:, ::-1]
    if flip_v:
        a = a[::-1]
    return np.ascontiguousarray(a)


def generate_wl_scene(
    rng: np.random.Generator,
    size: tuple[int, int] = DEFAULT_SIZE,
    *,
    site: OralSite | str,
    lesion: LesionSpec | None = None,
    teeth_stain: float = 0.0,
    watermark: bool = True,
) -> SyntheticScene:
    """Generate a synthetic WL close-up of ``site`` with exact masks.

    ``teeth_stain`` (0..1) adds the brown cervical staining typical of gutka/khaini users.
    ``watermark=False`` exists only for tests of the stamp itself; demo data is always stamped.
    """
    site = OralSite(site)
    h, w = int(size[0]), int(size[1])
    if h < 48 or w < 48:
        raise ValueError(f"size must be at least 48 x 48, got {size}")
    geo_rng, tex_rng, lesion_rng, light_rng, cam_rng = child_rngs(rng, 5)
    painter, flip_h, flip_v = _SITE_PLAN[site]
    ctx = _Ctx(geo_rng, h, w)
    lay = painter(ctx)
    ctx.rng = tex_rng

    teeth = _teeth(ctx, lay.teeth, teeth_stain) if lay.teeth else None
    teeth_alpha = teeth.alpha if teeth else np.zeros(ctx.shape, np.float32)

    # Colour: low-frequency variation, vessels, site overlays.
    base = lerp_rgb(lay.base, lay.deep, 0.45 * smoothstep(-1.0, 1.5, ctx.noise(200, 3)))
    base = lerp_rgb(base, lay.pale, 0.35 * smoothstep(-1.0, 1.5, ctx.noise(160, 3)))
    vessels = _vessel_map(ctx, lay.vessel_trees, lay.vessel_weight)
    base = lerp_rgb(base, lay.vessel_color, lay.vessel_strength * vessels)
    for color, weight in lay.overlays:
        base = lerp_rgb(base, color, weight)

    # Lesion, placed in canonical coordinates so it lands where requested after flipping.
    stamp = _flip(stamp_mask((h, w)), flip_h, flip_v)
    tissue_geom = (lay.shadow < 0.5) & (teeth_alpha < 0.5) & ~stamp
    lesion_mask = np.zeros(ctx.shape, bool)
    lesion_cover = np.zeros(ctx.shape, np.float32)
    placed: LesionSpec | None = None
    if lesion is not None:
        cx, cy = lesion.center
        canon = replace(lesion, center=(1 - cx if flip_h else cx, 1 - cy if flip_v else cy))
        allowed = tissue_geom & ~dilate(lay.shadow > 0.15, 6 * ctx.s)
        canon, patch, py, px = _place_lesion(canon, allowed, lesion_rng)
        base, lesion_mask, lesion_cover = _paint_placed_lesion(
            base, canon, patch, py, px, lesion_rng
        )
        ccx, ccy = canon.center
        placed = replace(canon, center=(1 - ccx if flip_h else ccx, 1 - ccy if flip_v else ccy))

    # Lighting.
    ctx.rng = light_rng
    lum = 1.0 + 0.02 * ctx.noise(60, 3) + 0.012 * ctx.noise(14, 2) + 0.008 * ctx.noise(3, 1)
    shade = _relief_shading(ctx, lay.relief + 3.0 * ctx.noise(220, 2))
    ao = 1.0 - 0.6 * blur(lay.shadow, 40 * ctx.s)
    vignette, hot = _torch(ctx)
    light = (vignette * ao)[..., None]
    img = base * (lum * shade)[..., None] * light

    if teeth is not None:
        sulcus = np.clip(blur(teeth_alpha, 2.0 * ctx.s) - teeth_alpha, 0.0, 1.0)
        free_gingiva = np.clip(blur(teeth_alpha, 6.0 * ctx.s) - teeth_alpha, 0.0, 1.0)
        img = lerp_rgb(img, (0.62, 0.22, 0.25), 1.2 * free_gingiva * vignette)
        img *= (1.0 - 0.8 * sulcus)[..., None]
        crowns = teeth.rgb * np.sqrt(light)
        img = lerp_rgb(img, crowns, teeth_alpha)
        img = lerp_rgb(img, (0.12, 0.06, 0.05), teeth.gap * vignette)

    deep = smoothstep(0.5, 1.0, blur(lay.shadow, 25 * ctx.s))
    cavity = ctx.uniform((0.065, 0.028, 0.028)) * (1.0 - 0.6 * deep)[..., None]
    img = lerp_rgb(img, cavity, lay.shadow)

    tissue_soft = np.clip(1.0 - lay.shadow - teeth_alpha, 0.0, 1.0)
    img = img + (0.06 * hot * tissue_soft)[..., None] * (1.0 - img)  # broad moist sheen
    spec_region = np.clip(tissue_soft + 0.3 * teeth_alpha, 0.0, 1.0)
    # Keratinised plaque and velvety erythroplakia are duller than wet healthy mucosa.
    spec = _speculars(ctx, spec_region, hot, lay.gloss) * (1.0 - 0.7 * lesion_cover)
    if teeth is not None:
        spec = np.maximum(spec, 0.9 * teeth.highlight * vignette)
    img = img + spec[..., None] * (0.985 - img)
    img = _apply_camera(img, cam_rng, ctx.s)

    background = (lay.shadow >= 0.5) | ((teeth.gap >= 0.5) if teeth else False)
    teeth_mask = (teeth_alpha >= 0.5) & ~background
    tissue_mask = ~background & ~teeth_mask
    specular_mask = (spec >= 0.35) & ~background
    arrays = [img, tissue_mask, teeth_mask, lesion_mask & tissue_mask, background, specular_mask]
    img, tissue_mask, teeth_mask, lesion_mask, background, specular_mask = (
        _flip(a, flip_h, flip_v) for a in arrays
    )
    final_stamp = stamp_mask((h, w))
    if watermark:
        img = watermark_synthetic(img)
        background = background | final_stamp
        tissue_mask = tissue_mask & ~final_stamp
        teeth_mask = teeth_mask & ~final_stamp
        specular_mask = specular_mask & ~final_stamp
    return SyntheticScene(
        wl=img.astype(np.float32),
        tissue_mask=tissue_mask,
        teeth_mask=teeth_mask,
        lesion_mask=lesion_mask,
        background_mask=background,
        specular_mask=specular_mask,
        site=site,
        lesion=placed,
    )


def generate_wl_image(
    rng: np.random.Generator,
    size: tuple[int, int] = DEFAULT_SIZE,
    *,
    site: OralSite | str,
    lesion: LesionSpec | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """(WL float32 RGB image, lesion ground-truth mask). Thin wrapper of generate_wl_scene."""
    scene = generate_wl_scene(rng, size, site=site, lesion=lesion)
    return scene.wl, scene.lesion_mask
