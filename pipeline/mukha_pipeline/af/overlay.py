"""AF-loss heatmap drawn over the white-light image (goes into the clinician PDF).

Design choices for a clinical figure:
* Colour encodes *how much* green fluorescence is lost, using a perceptually ordered ramp (the
  upper part of matplotlib's "inferno", reversed): lightness falls monotonically from pale yellow
  (slight loss) to deep violet (strong loss), so it reads correctly in greyscale print and for
  colour-blind readers, and stays distinct from pink mucosa.
* Opacity grows with loss, so normal tissue is left untouched and the WL anatomy stays visible.
* Areas that were not analysed (teeth, shadow, glare) are gently desaturated, so the reader sees
  what the device actually looked at.
* Region outlines use a white core with a dark halo, legible on any background; each region
  carries its id ("r1" = most severe), matching the measurements table.
"""

from __future__ import annotations

import cv2
import numpy as np

from mukha_pipeline.af._common import AF_DEFAULTS, get, to_float_rgb

# (loss position 0..1, RGB 0..1): inferno sampled from 0.92 down to 0.20.
LOSS_COLORMAP_STOPS: tuple[tuple[float, tuple[float, float, float]], ...] = (
    (0.000, (0.955, 0.882, 0.337)),
    (0.111, (0.985, 0.728, 0.121)),
    (0.222, (0.981, 0.572, 0.029)),
    (0.333, (0.942, 0.436, 0.125)),
    (0.444, (0.865, 0.317, 0.226)),
    (0.556, (0.764, 0.233, 0.311)),
    (0.667, (0.640, 0.171, 0.381)),
    (0.778, (0.516, 0.126, 0.421)),
    (0.889, (0.385, 0.079, 0.433)),
    (1.000, (0.258, 0.039, 0.406)),
)
LOSS_FLOOR = 0.2  # relative AF at which the colour ramp saturates
_HALO_RGB = (18, 18, 28)
_CORE_RGB = (255, 255, 255)


def loss_colormap(loss: np.ndarray) -> np.ndarray:
    """Map loss in [0, 1] to float32 RGB with the perceptually ordered AF-loss ramp."""
    x = np.clip(np.nan_to_num(np.asarray(loss, dtype=np.float32), nan=0.0), 0.0, 1.0)
    pos = np.array([p for p, _ in LOSS_COLORMAP_STOPS], dtype=np.float32)
    cols = np.array([c for _, c in LOSS_COLORMAP_STOPS], dtype=np.float32)
    return np.stack([np.interp(x, pos, cols[:, i]) for i in range(3)], axis=-1).astype(np.float32)


def heatmap_legend(threshold: float, n: int = 5) -> list[dict[str, object]]:
    """Legend entries (relative AF value -> hex colour) for reports and the web UI."""
    upper = threshold + 0.1
    out: list[dict[str, object]] = []
    for loss in np.linspace(0.0, 1.0, max(n, 2)):
        rel = upper - float(loss) * (upper - LOSS_FLOOR)
        r, g, b = (int(round(v * 255)) for v in loss_colormap(np.array([loss]))[0])
        out.append({"relative_af": round(rel, 3), "color": f"#{r:02x}{g:02x}{b:02x}"})
    return out


def af_loss(relative_af: np.ndarray, threshold: float) -> np.ndarray:
    """0..1 loss: 0 at ``threshold + 0.1`` (just above the FVL cut), 1 at relative AF 0.2."""
    upper = float(threshold) + 0.1
    span = max(upper - LOSS_FLOOR, 1e-6)
    rel = np.asarray(relative_af, dtype=np.float32)
    with np.errstate(invalid="ignore"):
        loss = np.clip((upper - rel) / span, 0.0, 1.0)
        loss = np.where(np.isfinite(rel) & (rel < upper), loss, 0.0)
    return loss.astype(np.float32)


def _section(cfg: dict | None) -> dict:
    if cfg and isinstance(cfg.get("af"), dict):
        return cfg["af"]
    return cfg or {}


def _desaturate(img: np.ndarray, where: np.ndarray, amount: float) -> np.ndarray:
    if amount <= 0 or not where.any():
        return img
    gray = img @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    mixed = img * (1.0 - amount) + gray[:, :, None] * amount
    return np.where(where[:, :, None], mixed, img).astype(np.float32)


def _draw_outlines(out: np.ndarray, labels: np.ndarray, thickness: int) -> None:
    for k in range(1, int(labels.max()) + 1):
        region = (labels == k).astype(np.uint8)
        if not region.any():
            continue
        contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # Anti-aliasing turns a 1 px line grey, so keep thin outlines crisp.
        line = cv2.LINE_AA if thickness >= 2 else cv2.LINE_8
        cv2.drawContours(out, contours, -1, _HALO_RGB, thickness + 2, line)
        cv2.drawContours(out, contours, -1, _CORE_RGB, thickness, line)


def _draw_labels(out: np.ndarray, labels: np.ndarray) -> None:
    h, w = labels.shape
    short = min(h, w)
    if short < 48:
        return
    scale = float(np.clip(short / 700.0, 0.35, 1.2))
    thick = 1 if scale < 0.8 else 2
    pad = max(2, int(round(3 * scale)))
    for k in range(1, int(labels.max()) + 1):
        ys, xs = np.nonzero(labels == k)
        if ys.size == 0:
            continue
        text = f"r{k}"
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        bw, bh = tw + 2 * pad, th + base + 2 * pad
        x0 = int(np.clip(xs.min(), 0, max(w - bw, 0)))
        y0 = int(ys.min()) - bh - 3
        if y0 < 0:
            y0 = int(ys.max()) + 3
        y0 = int(np.clip(y0, 0, max(h - bh, 0)))
        box = out[y0 : y0 + bh, x0 : x0 + bw].astype(np.float32)
        box = box * 0.2 + np.array(_HALO_RGB, dtype=np.float32) * 0.8
        out[y0 : y0 + bh, x0 : x0 + bw] = box.astype(np.uint8)
        org = (x0 + pad, y0 + pad + th)
        cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, _CORE_RGB, thick, cv2.LINE_AA)


def render_heatmap_overlay(
    wl: np.ndarray,
    relative_af: np.ndarray,
    labels: np.ndarray | None,
    threshold: float,
    cfg: dict | None,
) -> np.ndarray:
    """Blend the AF-loss heatmap over the WL image; returns uint8 RGB, same size as ``wl``.

    ``cfg`` = analysis.af (the full analysis config is accepted too). Loss is
    clip((threshold + 0.1 - rel) / (threshold + 0.1 - 0.2), 0, 1), drawn only where relative AF
    is below threshold + 0.1, with opacity loss * heatmap_alpha_max.
    """
    base = to_float_rgb(wl)
    shape = base.shape[:2]
    rel = np.asarray(relative_af, dtype=np.float32)
    if rel.shape != shape:
        raise ValueError(f"relative_af has shape {rel.shape}, expected {shape}")
    lab = np.zeros(shape, np.int32) if labels is None else np.asarray(labels).astype(np.int32)
    if lab.shape != shape:
        raise ValueError(f"labels has shape {lab.shape}, expected {shape}")
    c = _section(cfg)
    thr = (
        float(threshold) if np.isfinite(threshold) else float(AF_DEFAULTS["fvl_relative_threshold"])
    )

    analysed = np.isfinite(rel)
    if analysed.any():
        base = _desaturate(
            base, ~analysed, float(get(c, "heatmap_unanalysed_desaturate", AF_DEFAULTS))
        )
    loss = af_loss(rel, thr)
    alpha = (loss * float(np.clip(get(c, "heatmap_alpha_max", AF_DEFAULTS), 0.0, 1.0)))[..., None]
    blended = base * (1.0 - alpha) + loss_colormap(loss) * alpha
    out = np.ascontiguousarray(np.clip(blended * 255.0 + 0.5, 0, 255).astype(np.uint8))
    if lab.size and lab.max() > 0:
        thickness = max(1, int(round(min(shape) / 450.0)))
        _draw_outlines(out, lab, thickness)
        _draw_labels(out, lab)
    return out
