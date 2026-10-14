"""Visible "SYNTHETIC" stamp burned into every generated image.

The stamp is a near-opaque dark label flush with the bottom-left corner. Design choices:
* Flush with the image border, so a hole-filling segmenter never treats it as an enclosed
  patch of mucosa (an enclosed dark label would look like a large fluorescence-loss region).
* Dark enough (HSV value <= 0.12) to be classed as background, and the text stays below the
  glare threshold (value < 0.96) so it never counts as specular highlight.
* Small (about 2% of the frame) and in a corner, so it never covers the lesion: the scene
  generator keeps lesions out of :func:`stamp_mask`.
* Camera-fixed: the text sits at the same pixels in the WL and the AF capture, like a label
  burned in by the capture app. Registration recognises pixel-identical overlays and ignores
  them; a label that moved with the tissue would instead give feature matching a degenerate
  cluster of keypoints. AF captures get the text without the dark band (``band=False``), so
  after registration no dark label can land on mucosa and read as fluorescence loss.
"""

from __future__ import annotations

import cv2
import numpy as np

STAMP_TEXT = "SYNTHETIC"
STAMP_SUBTEXT = "generated image - not a real patient"

_BAND_RGB = np.array([0.035, 0.035, 0.045], np.float32)
_TEXT_RGB = np.array([0.93, 0.93, 0.93], np.float32)
_BAND_OPACITY = 0.93  # keeps HSV value <= 0.12 even over white teeth
_DIM_TEXT_DELTA = 0.06  # dark frames: text only this much above the frame's own level


def _layout(shape: tuple[int, ...]) -> dict[str, float | int]:
    """Font scales and band size for an image of this shape (all derived from its height)."""
    h, w = int(shape[0]), int(shape[1])
    unit = max(min(h, w), 64) / 768.0
    main_scale = 0.95 * unit
    sub_scale = 0.42 * unit
    main_thick = max(1, int(round(2.2 * unit)))
    sub_thick = max(1, int(round(1.0 * unit)))
    (mw, mh), _ = cv2.getTextSize(STAMP_TEXT, cv2.FONT_HERSHEY_DUPLEX, main_scale, main_thick)
    (sw, sh), _ = cv2.getTextSize(STAMP_SUBTEXT, cv2.FONT_HERSHEY_SIMPLEX, sub_scale, sub_thick)
    pad = int(round(10 * unit))
    gap = int(round(7 * unit))
    band_w = min(w, max(mw, sw) + 2 * pad)
    band_h = min(h, mh + sh + gap + 2 * pad + int(round(4 * unit)))
    return {
        "h": h,
        "w": w,
        "unit": unit,
        "main_scale": main_scale,
        "sub_scale": sub_scale,
        "main_thick": main_thick,
        "sub_thick": sub_thick,
        "main_h": mh,
        "sub_h": sh,
        "pad": pad,
        "gap": gap,
        "band_w": band_w,
        "band_h": band_h,
    }


def stamp_box(shape: tuple[int, ...]) -> tuple[int, int, int, int]:
    """(x0, y0, x1, y1) pixel bounds (exclusive end) of the stamp band."""
    lay = _layout(shape)
    h = int(lay["h"])
    return 0, h - int(lay["band_h"]), int(lay["band_w"]), h


def _band_alpha(shape: tuple[int, ...]) -> np.ndarray:
    """Band coverage in [0, 1] with an anti-aliased rounded top-right corner."""
    lay = _layout(shape)
    h, w = int(lay["h"]), int(lay["w"])
    x0, y0, x1, y1 = stamp_box(shape)
    canvas = np.zeros((h, w), np.uint8)
    r = max(2, int(round(9 * float(lay["unit"]))))
    cv2.rectangle(canvas, (x0, y0 + r), (x1 - 1, y1 - 1), 255, -1)
    cv2.rectangle(canvas, (x0, y0), (x1 - 1 - r, y1 - 1), 255, -1)
    cv2.circle(canvas, (x1 - 1 - r, y0 + r), r, 255, -1, lineType=cv2.LINE_AA)
    return canvas.astype(np.float32) / 255.0


def _text_alpha(shape: tuple[int, ...]) -> np.ndarray:
    lay = _layout(shape)
    h, w = int(lay["h"]), int(lay["w"])
    canvas = np.zeros((h, w), np.uint8)
    _, y0, _, _ = stamp_box(shape)
    pad, gap = int(lay["pad"]), int(lay["gap"])
    base1 = y0 + pad + int(lay["main_h"])
    cv2.putText(
        canvas,
        STAMP_TEXT,
        (pad, base1),
        cv2.FONT_HERSHEY_DUPLEX,
        float(lay["main_scale"]),
        255,
        int(lay["main_thick"]),
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        STAMP_SUBTEXT,
        (pad, base1 + gap + int(lay["sub_h"])),
        cv2.FONT_HERSHEY_SIMPLEX,
        float(lay["sub_scale"]),
        255,
        int(lay["sub_thick"]),
        cv2.LINE_AA,
    )
    return canvas.astype(np.float32) / 255.0


def stamp_mask(shape: tuple[int, ...]) -> np.ndarray:
    """Bool (H, W) mask of the pixels the stamp may change."""
    return _band_alpha(shape) > 0.0


def watermark_synthetic(img: np.ndarray, *, dim: bool = False, band: bool = True) -> np.ndarray:
    """Return a copy of ``img`` with a visible "SYNTHETIC" stamp in the bottom-left corner.

    Accepts float images in [0, 1] or uint8, RGB (H, W, 3) or grey (H, W); returns the same
    dtype and shape. ``dim=True`` draws only faint text a little above the image's own median
    level, for dark frames, where a bright label would distort dark subtraction.
    ``band=False`` draws the text without its dark backing band (used for AF captures).
    """
    if img.ndim not in (2, 3):
        raise ValueError(f"expected an (H, W) or (H, W, C) image, got shape {img.shape}")
    is_uint8 = img.dtype == np.uint8
    f = img.astype(np.float32) / 255.0 if is_uint8 else img.astype(np.float32, copy=True)
    grey = f.ndim == 2
    if grey:
        f = f[..., None]
    channels = f.shape[-1]
    band_col = np.resize(_BAND_RGB, channels) if channels != 3 else _BAND_RGB
    text_col = np.resize(_TEXT_RGB, channels) if channels != 3 else _TEXT_RGB
    text = _text_alpha(f.shape)[..., None]
    if dim:
        level = float(np.median(f)) + _DIM_TEXT_DELTA
        f = f * (1.0 - text) + text * level
    else:
        if band:
            cover = (_band_alpha(f.shape) * _BAND_OPACITY)[..., None]
            f = f * (1.0 - cover) + cover * band_col
        f = f * (1.0 - text) + text * text_col
    f = np.clip(f, 0.0, 1.0)
    if grey:
        f = f[..., 0]
    if is_uint8:
        return np.round(f * 255.0).astype(np.uint8)
    return f.astype(np.float32)
