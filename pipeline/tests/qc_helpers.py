"""Procedural test images and readings for the QC tests (all SYNTHETIC).

The images mimic what the Scout clip sees: a close-up of pink oral mucosa with multi-octave
texture, a few superficial vessels, a band of teeth and small specular speckles under white light;
under 405 nm excitation the same tissue glows green (collagen / NADH / FAD), vessels absorb
(haemoglobin), teeth fluoresce brightly and the long-pass emission filter leaves little blue.
Degradations (blur, under/over-exposure, missing emission filter, room light) are applied on top so
tests can check that QC passes plausible captures and fails clearly degraded ones.
"""

from __future__ import annotations

import cv2
import numpy as np

from mukha_pipeline.schema.models import AS7341_CHANNELS
from mukha_pipeline.types import Masks, RegistrationResult, SpectralFrame

Size = tuple[int, int]

MUCOSA_RGB = np.array([0.80, 0.50, 0.52], dtype=np.float32)
VESSEL_RGB = np.array([0.58, 0.22, 0.28], dtype=np.float32)
TEETH_RGB = np.array([0.88, 0.86, 0.79], dtype=np.float32)


def fractal_noise(
    rng: np.random.Generator, size: Size, *, octaves: tuple[int, ...] = (4, 8, 16, 32, 64, 128)
) -> np.ndarray:
    """Zero-mean, unit-std multi-octave value noise (float32, H x W)."""
    h, w = size
    out = np.zeros((h, w), dtype=np.float32)
    for i, cell in enumerate(octaves):
        gh, gw = max(2, h // cell + 2), max(2, w // cell + 2)
        grid = rng.standard_normal((gh, gw)).astype(np.float32)
        layer = cv2.resize(grid, (w, h), interpolation=cv2.INTER_CUBIC)
        out += layer * (0.75**i if cell > 8 else 0.55)
    out -= out.mean()
    std = float(out.std())
    return out / std if std > 0 else out


def _vessel_mask(rng: np.random.Generator, size: Size, n: int = 6) -> np.ndarray:
    h, w = size
    canvas = np.zeros((h, w), dtype=np.uint8)
    for _ in range(n):
        pts = [(int(rng.uniform(0, w)), int(rng.uniform(0.25 * h, h)))]
        for _ in range(6):
            x, y = pts[-1]
            pts.append((int(x + rng.normal(0, w * 0.08)), int(y + rng.normal(0, h * 0.06))))
        thickness = int(rng.integers(1, 4))
        cv2.polylines(canvas, [np.array(pts, dtype=np.int32)], False, 255, thickness)
    soft = cv2.GaussianBlur(canvas.astype(np.float32) / 255.0, (0, 0), 1.2)
    return np.clip(soft, 0.0, 1.0)


def make_wl_image(
    rng: np.random.Generator,
    size: Size = (768, 1024),
    *,
    texture: float = 0.06,
    teeth: bool = True,
    speckles: int = 40,
    sensor_noise: float = 0.006,
    exposure: float = 1.0,
) -> np.ndarray:
    """A plausible white-light mucosa close-up, float32 RGB in [0, 1]."""
    h, w = size
    n = fractal_noise(rng, size)
    img = MUCOSA_RGB[None, None, :] * (1.0 + texture * n[..., None])
    # slow illumination fall-off from the clip's LED ring
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r2 = ((xx - w / 2) / w) ** 2 + ((yy - h / 2) / h) ** 2
    img *= (1.0 - 0.35 * r2)[..., None]
    v = _vessel_mask(rng, size)[..., None]
    img = img * (1 - 0.8 * v) + VESSEL_RGB[None, None, :] * 0.8 * v
    if teeth:
        band = int(0.18 * h)
        edge = (band + 6 * np.sin(np.linspace(0, 9 * np.pi, w))).astype(int)
        teeth_mask = yy < edge[None, :]
        tex = 1.0 + 0.03 * fractal_noise(rng, size, octaves=(16, 32, 64))
        img[teeth_mask] = (TEETH_RGB[None, None, :] * tex[..., None])[teeth_mask]
    if speckles:
        spec = np.zeros((h, w), dtype=np.uint8)
        for _ in range(speckles):
            c = (int(rng.uniform(0, w)), int(rng.uniform(0.2 * h, h)))
            cv2.circle(spec, c, int(rng.integers(1, 4)), 255, -1)
        img[spec > 0] = 1.0
    img = img * exposure + rng.normal(0.0, sensor_noise, img.shape).astype(np.float32)
    return np.clip(img, 0.0, 1.0).astype(np.float32)


def make_af_image(
    rng: np.random.Generator,
    size: Size = (768, 1024),
    *,
    green: float = 0.42,
    red: float = 0.11,
    blue: float = 0.04,
    texture: float = 0.08,
    teeth: bool = True,
    sensor_noise: float = 0.006,
) -> np.ndarray:
    """A plausible 405 nm autofluorescence capture through a long-pass filter (green glow)."""
    h, w = size
    n = cv2.GaussianBlur(fractal_noise(rng, size), (0, 0), 1.5)
    n /= max(float(n.std()), 1e-6)
    base = 1.0 + texture * n
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r2 = ((xx - w / 2) / w) ** 2 + ((yy - h / 2) / h) ** 2
    base *= 1.0 - 0.4 * r2
    v = cv2.GaussianBlur(_vessel_mask(rng, size), (0, 0), 2.0)  # absorption is diffuse in AF
    base *= 1.0 - 0.45 * v
    img = np.stack([red * base, green * base, blue * base], axis=-1)
    if teeth:
        band = int(0.18 * h)
        # enamel/dentin fluoresce brighter than mucosa under the same excitation
        img[:band] = np.array([0.6, 1.8, 0.7], dtype=np.float32) * green
    img += rng.normal(0.0, sensor_noise, img.shape).astype(np.float32)
    return np.clip(img, 0.0, 1.0).astype(np.float32)


def remove_emission_filter(af: np.ndarray, leak: float = 0.55) -> np.ndarray:
    """Simulate a missing long-pass filter: reflected 405 nm light floods the blue channel."""
    out = af.copy()
    out[..., 2] = np.clip(out[..., 2] + leak, 0.0, 1.0)
    out[..., 0] = np.clip(out[..., 0] + 0.15 * leak, 0.0, 1.0)  # Bayer red leaks in the violet
    return out


def grey_world(img: np.ndarray) -> np.ndarray:
    """Auto white balance (grey world): equal channel means, as an unlocked phone camera does."""
    means = img.reshape(-1, 3).mean(axis=0)
    return np.clip(img * (means.mean() / np.maximum(means, 1e-6)), 0.0, 1.0).astype(np.float32)


def through_emission_filter(img: np.ndarray) -> np.ndarray:
    """White light seen through the long-pass emission filter: blue blocked, image orange-pink."""
    out = img.copy()
    out[..., 2] *= 0.1
    return out


def add_porphyrin(af: np.ndarray, level: float = 0.5) -> np.ndarray:
    """Heavy red (~635 nm) porphyrin fluorescence from bacterial coating over the whole field."""
    out = af.copy()
    out[..., 0] = np.clip(out[..., 0] + level, 0.0, 1.0)
    return out


def make_dark_frame(rng: np.random.Generator, size: Size = (768, 1024), level: float = 0.01):
    """All LEDs off: sensor offset plus read noise; `level` adds uniform room light."""
    noise = rng.normal(0.0, 0.004, (*size, 3)).astype(np.float32)
    return np.clip(level + noise, 0.0, 1.0).astype(np.float32)


def blur(img: np.ndarray, sigma: float) -> np.ndarray:
    return cv2.GaussianBlur(img, (0, 0), sigma).astype(np.float32)


def scale(img: np.ndarray, factor: float) -> np.ndarray:
    return np.clip(img * factor, 0.0, 1.0).astype(np.float32)


def add_glare(img: np.ndarray, fraction: float, rng: np.random.Generator) -> np.ndarray:
    """Blow out roughly `fraction` of the image as white specular blobs."""
    out = img.copy()
    h, w = out.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    target = fraction * h * w
    while (mask > 0).sum() < target:
        c = (int(rng.uniform(0, w)), int(rng.uniform(0, h)))
        cv2.circle(mask, c, int(rng.integers(6, 20)), 255, -1)
    out[mask > 0] = 1.0
    return out


# ---- registration / segmentation ---------------------------------------------------------


def make_registration(confidence: float, passed: bool = True, size: Size = (32, 48)):
    return RegistrationResult(
        warp=np.eye(3),
        method="ecc_affine",
        confidence=confidence,
        passed=passed,
        registered=np.zeros((*size, 3), dtype=np.float32),
        valid=np.ones(size, dtype=bool),
    )


def make_masks(mucosa_fraction: float, size: Size = (100, 100), confidence: float = 0.9):
    h, w = size
    mucosa = np.zeros(size, dtype=bool)
    n = int(round(mucosa_fraction * h * w))
    mucosa.reshape(-1)[:n] = True
    empty = np.zeros(size, dtype=bool)
    return Masks(
        mucosa=mucosa, teeth=empty, background=~mucosa, specular=empty, confidence=confidence
    )


# ---- spectral -----------------------------------------------------------------------------

# Typical lit counts for buccal mucosa with the Confirm probe's white LED (diffuse reflectance:
# haemoglobin dip around 540-580 nm, weak 415 nm because of the Soret band and LED output).
LIT_COUNTS: dict[str, float] = {
    "F1_415": 900.0,
    "F2_445": 6000.0,
    "F3_480": 4200.0,
    "F4_515": 5200.0,
    "F5_555": 4300.0,
    "F6_590": 6800.0,
    "F7_630": 9000.0,
    "F8_680": 8200.0,
    "CLEAR": 30000.0,
    "NIR": 3000.0,
}
DARK_COUNTS: dict[str, float] = {k: 40.0 for k in AS7341_CHANNELS} | {"NIR": 120.0}


def make_frames(
    rng: np.random.Generator,
    n: int,
    base: dict[str, float],
    *,
    cv: float = 0.01,
    drop: tuple[str, ...] = (),
    override: dict[str, float] | None = None,
    gain: float | None = 64.0,
    integration_ms: float | None = 50.0,
) -> list[SpectralFrame]:
    """N frames around `base` counts with multiplicative noise `cv`."""
    frames = []
    for _ in range(n):
        ch = {k: float(v * (1.0 + rng.normal(0.0, cv))) for k, v in base.items() if k not in drop}
        if override:
            ch.update(override)
        frames.append(SpectralFrame(channels=ch, gain=gain, integration_ms=integration_ms))
    return frames
