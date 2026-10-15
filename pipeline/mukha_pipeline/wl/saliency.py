"""Model-agnostic occlusion saliency for the white-light lesion classifier.

Grad-CAM needs gradients, which an ONNX Runtime session does not expose. Occlusion works with any
black box: hide one patch at a time and measure how much the target class probability drops. Areas
whose removal lowers the score most are the ones the model relied on. The map is shown to
clinicians only, as a check that the model looked at the lesion rather than at teeth or glare.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import cv2
import numpy as np

from mukha_pipeline.io.images import resize_max_side, to_float_rgb

DEFAULT_PATCH_FRACTION = 0.125
DEFAULT_STRIDE_FRACTION = 0.0625
DEFAULT_MAX_SIDE_PX = 448  # the model sees ~224 px; finer occlusion adds cost, not detail


def wl_section(cfg: Mapping[str, Any] | None) -> Mapping[str, Any]:
    """Accept the full analysis config or its 'wl_classifier' section."""
    cfg = cfg or {}
    section = cfg.get("wl_classifier")
    return section if isinstance(section, Mapping) else cfg


def _positions(n: int, patch: int, stride: int) -> list[int]:
    if patch >= n:
        return [0]
    pos = list(range(0, n - patch + 1, stride))
    if pos[-1] + patch < n:  # make sure the last rows/columns are covered too
        pos.append(n - patch)
    return pos


def occlusion_saliency(
    predict: Callable[[np.ndarray], np.ndarray],
    img: np.ndarray,
    target_index: int,
    cfg: Mapping[str, Any] | None,
) -> np.ndarray:
    """Float32 (H, W) map in [0, 1]: drop in the target probability when each area is hidden.

    Patches are filled with the image's mean colour (a neutral occluder), averaged where they
    overlap, clipped at 0 (only evidence *for* the target is shown) and normalised to max 1.
    """
    section = wl_section(cfg)
    arr = to_float_rgb(img)
    H, W = arr.shape[:2]
    if H == 0 or W == 0:
        return np.zeros((H, W), dtype=np.float32)

    max_side = int(section.get("saliency_max_side_px", DEFAULT_MAX_SIDE_PX))
    work, _ = resize_max_side(arr, max(8, max_side))
    work = np.array(work, dtype=np.float32, copy=True)  # occluded in place, never the caller's
    h, w = work.shape[:2]
    short = min(h, w)
    patch = max(
        1, round(float(section.get("saliency_patch_fraction", DEFAULT_PATCH_FRACTION)) * short)
    )
    stride = max(
        1, round(float(section.get("saliency_stride_fraction", DEFAULT_STRIDE_FRACTION)) * short)
    )

    base_probs = np.asarray(predict(work), dtype=np.float64).reshape(-1)
    if not 0 <= target_index < base_probs.size:
        raise ValueError(f"target_index {target_index} is outside 0..{base_probs.size - 1}")
    base = float(base_probs[target_index])
    fill = work.reshape(-1, 3).mean(axis=0)

    heat = np.zeros((h, w), dtype=np.float64)
    count = np.zeros((h, w), dtype=np.float64)
    for y in _positions(h, patch, stride):
        for x in _positions(w, patch, stride):
            saved = work[y : y + patch, x : x + patch].copy()
            work[y : y + patch, x : x + patch] = fill
            try:
                prob = float(np.asarray(predict(work), dtype=np.float64).reshape(-1)[target_index])
            finally:
                work[y : y + patch, x : x + patch] = saved
            if np.isfinite(prob):
                heat[y : y + patch, x : x + patch] += base - prob
                count[y : y + patch, x : x + patch] += 1.0

    avg = np.clip(np.divide(heat, count, out=np.zeros_like(heat), where=count > 0), 0.0, None)
    top = float(avg.max())
    sal = (avg / top if top > 1e-12 else np.zeros_like(avg)).astype(np.float32)
    if (h, w) != (H, W):
        sal = cv2.resize(sal, (W, H), interpolation=cv2.INTER_LINEAR)
    return np.clip(sal, 0.0, 1.0).astype(np.float32)
