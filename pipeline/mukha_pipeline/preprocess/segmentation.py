"""Mucosa segmentation of the white-light (WL) image.

Phase 1 uses transparent HSV/Lab colour rules; a U-Net can replace it later behind the same
:class:`Segmenter` interface.

Clinical point that drives the design: leukoplakia (a white patch, the commonest oral lesion
with a risk of progression) is whitish and low-saturation, exactly like teeth. Naive colour
rules therefore label it "teeth" and silently drop the lesion from the AF analysis — the worst
kind of false reassurance. We keep any non-mucosa blob that is enclosed (or almost enclosed) by
mucosa *inside* the mucosa mask, while teeth, which reach the image border or the dark oral
cavity, stay excluded. Specular highlights are removed from the analysable mucosa afterwards,
because they carry no tissue signal, but they are reported in ``Masks.specular``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import cv2
import numpy as np
from scipy import ndimage as ndi

from mukha_pipeline.schema.models import SegmentationSummary
from mukha_pipeline.types import Masks

METHOD_NAME = "classical_hsv_lab"

_DEFAULTS: dict[str, Any] = {
    "version": "1.0.0",
    "mucosa_hue_ranges": [[0, 22], [160, 180]],
    "mucosa_saturation_min": 0.18,
    "mucosa_value_min": 0.18,
    "mucosa_lab_a_min": 14.0,
    "teeth_saturation_max": 0.22,
    "teeth_value_min": 0.62,
    "teeth_lab_a_max": 8.0,
    "background_value_max": 0.12,
    "specular_value_min": 0.96,
    "specular_saturation_max": 0.20,
    "specular_dilate_px": 1,
    "fill_holes_max_fraction": 0.10,
    "partial_enclosure_min": 0.85,
    "min_component_fraction": 0.01,
    "morphology_kernel_px": 7,
    "confidence_full_mucosa_fraction": 0.25,
    "confidence_ambiguous_max": 0.6,
    "failed_mucosa_fraction": 0.02,
    "low_confidence_below": 0.5,
}


def _cfg(cfg: dict | None, key: str) -> Any:
    """Config value with a safe default, so a partial config never crashes the pipeline."""
    if cfg and key in cfg and cfg[key] is not None:
        return cfg[key]
    return _DEFAULTS[key]


@runtime_checkable
class Segmenter(Protocol):
    """Anything that turns a WL image into mucosa / teeth / background / specular masks."""

    name: str
    version: str

    def segment(self, wl: np.ndarray) -> Masks: ...


@dataclass
class PixelClasses:
    """Raw per-pixel colour classes before any morphology (all bool (H, W))."""

    mucosa: np.ndarray
    teeth: np.ndarray
    background: np.ndarray
    specular: np.ndarray

    @property
    def ambiguous(self) -> np.ndarray:
        """Pixels no rule claimed (e.g. yellow coating, odd colour cast)."""
        return ~(self.mucosa | self.teeth | self.background | self.specular)


@dataclass
class SegmentationDiagnostics:
    """Intermediate results, exposed for tests, debugging and the confidence heuristic."""

    classes: PixelClasses
    mucosa_cleaned: np.ndarray  # after morphology, before hole filling
    filled: np.ndarray  # pixels added back by enclosed / partially enclosed hole filling
    ambiguous_fraction: float  # of non-background pixels
    filled_fraction_of_mucosa: float


# --------------------------------------------------------------------------------------------
# Small pure helpers
# --------------------------------------------------------------------------------------------


def sanitize_rgb(img: np.ndarray) -> np.ndarray:
    """Coerce any image-like array to float32 RGB in [0, 1]; NaN/inf become 0 (dark)."""
    arr = np.asarray(img)
    if arr.dtype == np.uint8:
        arr = arr.astype(np.float32) / 255.0
    elif arr.dtype == np.uint16:
        arr = arr.astype(np.float32) / 65535.0
    else:
        arr = arr.astype(np.float32, copy=True)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    elif arr.ndim == 3 and arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    elif arr.ndim == 3 and arr.shape[2] >= 4:
        arr = arr[:, :, :3]
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"expected an RGB image (H, W, 3), got shape {np.shape(img)}")
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(arr, 0.0, 1.0)


def ellipse_kernel(size_px: float) -> np.ndarray:
    """Elliptical structuring element with an odd diameter >= 1."""
    d = max(1, int(round(size_px)))
    d += 1 - d % 2
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d))


def remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    """Drop 8-connected components smaller than ``min_area`` pixels."""
    if min_area <= 1 or not mask.any():
        return mask.copy()
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    return keep[labels]


def _morph(mask: np.ndarray, op: int, size_px: float) -> np.ndarray:
    if size_px <= 1 or not mask.any():
        return mask.copy()
    out = cv2.morphologyEx(mask.astype(np.uint8), op, ellipse_kernel(size_px))
    return out.astype(bool)


def _hue_in_ranges(hue180: np.ndarray, ranges: list[list[float]]) -> np.ndarray:
    out = np.zeros(hue180.shape, dtype=bool)
    for lo, hi in ranges:
        out |= (hue180 >= float(lo)) & (hue180 <= float(hi))
    return out


def classify_pixels(wl: np.ndarray, cfg: dict | None) -> PixelClasses:
    """Per-pixel colour rules. ``wl`` must already be sanitised float32 RGB in [0, 1].

    Mucosa is pink/red: hue in the configured OpenCV ranges (0-180) or Lab a* (redness) high,
    with enough saturation and brightness. Teeth are bright, unsaturated and not pinkish
    (a* < ``teeth_lab_a_max``); bright, unsaturated but pinkish pixels are pale mucosa (anaemia,
    blanched submucous fibrosis), not teeth. The oral cavity shadow is very dark. Specular
    highlights are near-white (V high, S low).
    """
    hsv = cv2.cvtColor(wl, cv2.COLOR_RGB2HSV)  # float32: H 0-360, S 0-1, V 0-1
    hue = hsv[:, :, 0] / 2.0  # OpenCV 8-bit hue convention, matches the YAML ranges
    sat, val = hsv[:, :, 1], hsv[:, :, 2]
    lab_a = cv2.cvtColor(wl, cv2.COLOR_RGB2Lab)[:, :, 1]

    background = val <= float(_cfg(cfg, "background_value_max"))
    specular = (val >= float(_cfg(cfg, "specular_value_min"))) & (
        sat <= float(_cfg(cfg, "specular_saturation_max"))
    )
    bright = val >= float(_cfg(cfg, "teeth_value_min"))
    pinkish = lab_a >= float(_cfg(cfg, "teeth_lab_a_max"))
    teeth = (sat <= float(_cfg(cfg, "teeth_saturation_max"))) & bright & ~pinkish & ~specular
    reddish = _hue_in_ranges(hue, _cfg(cfg, "mucosa_hue_ranges")) | (
        lab_a >= float(_cfg(cfg, "mucosa_lab_a_min"))
    )
    coloured = reddish & (sat >= float(_cfg(cfg, "mucosa_saturation_min")))
    mucosa = (
        (coloured | (pinkish & bright))
        & (val >= float(_cfg(cfg, "mucosa_value_min")))
        & ~teeth
        & ~background
        & ~specular
    )
    return PixelClasses(mucosa=mucosa, teeth=teeth, background=background, specular=specular)


def fill_enclosed_holes(mucosa: np.ndarray, max_area: int) -> np.ndarray:
    """Return the pixels of holes fully enclosed by mucosa with area <= ``max_area``.

    Holes touching the image border are not holes (binary_fill_holes semantics), so a band of
    teeth along the border stays excluded.
    """
    holes = ndi.binary_fill_holes(mucosa) & ~mucosa
    if not holes.any():
        return holes
    labels, n = ndi.label(holes, structure=np.ones((3, 3), dtype=bool))
    areas = np.bincount(labels.ravel(), minlength=n + 1)
    keep = areas <= max_area
    keep[0] = False
    return keep[labels]


def fill_partially_enclosed(
    mucosa: np.ndarray, candidates: np.ndarray, max_area: int, min_rim_fraction: float
) -> np.ndarray:
    """Return candidate blobs whose outer rim lies mostly on mucosa.

    A white patch at the edge of the cheek can touch a sliver of dark shadow, so it is not
    strictly enclosed; it is still almost certainly tissue. Blobs touching the image border are
    never added (teeth usually do), and neither are blobs larger than ``max_area``.
    """
    out = np.zeros_like(mucosa)
    if min_rim_fraction > 1.0 or not candidates.any() or not mucosa.any():
        return out
    labels, n = ndi.label(candidates, structure=np.ones((3, 3), dtype=bool))
    if n == 0:
        return out
    h, w = mucosa.shape
    for idx, sl in enumerate(ndi.find_objects(labels), start=1):
        if sl is None:
            continue
        r0, r1, c0, c1 = sl[0].start, sl[0].stop, sl[1].start, sl[1].stop
        if r0 == 0 or c0 == 0 or r1 == h or c1 == w:
            continue
        rr0, rr1, cc0, cc1 = max(r0 - 2, 0), min(r1 + 2, h), max(c0 - 2, 0), min(c1 + 2, w)
        blob = labels[rr0:rr1, cc0:cc1] == idx
        area = int(blob.sum())
        if area > max_area:
            continue
        rim = ndi.binary_dilation(blob, structure=np.ones((3, 3), dtype=bool)) & ~blob
        rim_n = int(rim.sum())
        if rim_n and float(mucosa[rr0:rr1, cc0:cc1][rim].mean()) >= min_rim_fraction:
            out[rr0:rr1, cc0:cc1] |= blob
    return out


def confidence_heuristic(
    mucosa_fraction: float,
    ambiguous_fraction: float,
    filled_fraction_of_mucosa: float,
    cfg: dict | None,
) -> float:
    """Heuristic 0..1 trust in the masks (not a calibrated probability).

    Low when little tissue is in view, when many pixels matched no colour rule (colour cast,
    coating, instruments) or when much of the mucosa was inferred by hole filling.
    """
    full = max(float(_cfg(cfg, "confidence_full_mucosa_fraction")), 1e-6)
    amb_max = max(float(_cfg(cfg, "confidence_ambiguous_max")), 1e-6)
    frac_term = float(np.clip(mucosa_fraction / full, 0.0, 1.0)) ** 0.5
    amb_term = 1.0 - float(np.clip(ambiguous_fraction / amb_max, 0.0, 1.0))
    fill_term = 1.0 - 0.5 * float(np.clip(filled_fraction_of_mucosa / 0.5, 0.0, 1.0))
    conf = frac_term * amb_term * fill_term
    return float(np.clip(conf, 0.0, 1.0)) if np.isfinite(conf) else 0.0


def _empty_masks(shape: tuple[int, int]) -> Masks:
    z = np.zeros(shape, dtype=bool)
    return Masks(mucosa=z, teeth=z.copy(), background=z.copy(), specular=z.copy(), confidence=0.0)


# --------------------------------------------------------------------------------------------
# Segmenter
# --------------------------------------------------------------------------------------------


class ClassicalSegmenter:
    """HSV/Lab colour rules + morphology + enclosed-hole filling (cfg = analysis.segmentation)."""

    name: str = METHOD_NAME

    def __init__(self, cfg: dict | None = None) -> None:
        self.cfg: dict = dict(cfg or {})
        self.version: str = str(_cfg(self.cfg, "version"))

    def segment(self, wl: np.ndarray) -> Masks:
        return self.segment_detailed(wl)[0]

    def segment_detailed(self, wl: np.ndarray) -> tuple[Masks, SegmentationDiagnostics]:
        """Segment and also return the intermediate masks used for the decision."""
        img = sanitize_rgb(wl)
        h, w = img.shape[:2]
        if h == 0 or w == 0:
            return _empty_masks((h, w)), _empty_diagnostics((h, w))
        n_px = h * w
        cfg = self.cfg
        classes = classify_pixels(img, cfg)
        k = float(_cfg(cfg, "morphology_kernel_px"))
        min_comp = int(np.ceil(float(_cfg(cfg, "min_component_fraction")) * n_px))

        mucosa = _morph(classes.mucosa, cv2.MORPH_OPEN, k)
        mucosa = _morph(mucosa, cv2.MORPH_CLOSE, k)
        mucosa = remove_small_components(mucosa, min_comp)
        cleaned = mucosa.copy()

        max_hole = int(float(_cfg(cfg, "fill_holes_max_fraction")) * n_px)
        filled = fill_enclosed_holes(mucosa, max_hole)
        mucosa = mucosa | filled
        # Whitish / unclassified blobs (not dark shadow) that are almost surrounded by mucosa.
        candidates = ~mucosa & ~classes.background
        partial = fill_partially_enclosed(
            mucosa, candidates, max_hole, float(_cfg(cfg, "partial_enclosure_min"))
        )
        filled |= partial
        mucosa = mucosa | partial

        specular = classes.specular
        dil = int(_cfg(cfg, "specular_dilate_px"))
        if dil > 0 and specular.any():
            specular = cv2.dilate(specular.astype(np.uint8), ellipse_kernel(2 * dil + 1)) > 0
        analysable = mucosa & ~specular
        teeth = _morph(classes.teeth & ~mucosa, cv2.MORPH_OPEN, 3)
        background = classes.background & ~mucosa

        non_bg = max(int((~classes.background).sum()), 1)
        ambiguous_fraction = float((classes.ambiguous & ~mucosa).sum()) / non_bg
        filled_frac = float(filled.sum()) / max(int(mucosa.sum()), 1)
        mucosa_fraction = float(analysable.sum()) / n_px
        conf = confidence_heuristic(mucosa_fraction, ambiguous_fraction, filled_frac, cfg)
        masks = Masks(
            mucosa=analysable,
            teeth=teeth,
            background=background,
            specular=specular,
            confidence=conf,
        )
        diag = SegmentationDiagnostics(
            classes=classes,
            mucosa_cleaned=cleaned,
            filled=filled,
            ambiguous_fraction=ambiguous_fraction,
            filled_fraction_of_mucosa=filled_frac,
        )
        return masks, diag


def _empty_diagnostics(shape: tuple[int, int]) -> SegmentationDiagnostics:
    z = np.zeros(shape, dtype=bool)
    classes = PixelClasses(mucosa=z, teeth=z.copy(), background=z.copy(), specular=z.copy())
    return SegmentationDiagnostics(
        classes=classes,
        mucosa_cleaned=z.copy(),
        filled=z.copy(),
        ambiguous_fraction=1.0,
        filled_fraction_of_mucosa=0.0,
    )


def _fraction(mask: np.ndarray) -> float:
    return float(np.clip(mask.mean(), 0.0, 1.0)) if mask.size else 0.0


def segmentation_summary(masks: Masks, segmenter: Segmenter, cfg: dict) -> SegmentationSummary:
    """Summarise masks for the result. ``failed`` when almost no mucosa is visible,
    ``low_confidence`` when the heuristic confidence is below the configured level."""
    mucosa_fraction = _fraction(masks.mucosa)
    conf = float(masks.confidence) if np.isfinite(masks.confidence) else 0.0
    conf = float(np.clip(conf, 0.0, 1.0))
    if mucosa_fraction < float(_cfg(cfg, "failed_mucosa_fraction")) or conf <= 0.0:
        status = "failed"
    elif conf < float(_cfg(cfg, "low_confidence_below")):
        status = "low_confidence"
    else:
        status = "ok"
    return SegmentationSummary(
        method=str(getattr(segmenter, "name", METHOD_NAME)),
        version=str(getattr(segmenter, "version", _DEFAULTS["version"])),
        status=status,
        mucosa_fraction=mucosa_fraction,
        teeth_fraction=_fraction(masks.teeth),
        specular_fraction=_fraction(masks.specular),
        confidence=conf,
    )
