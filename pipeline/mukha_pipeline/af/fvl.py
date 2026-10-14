"""Fluorescence visualisation loss (FVL) detection and per-region measurements.

relative AF = smoothed green / reference green. Normal mucosa sits near 1; a region whose green
glow drops below ``fvl_relative_threshold`` (default 0.75) of the reference is flagged. Darkness
alone is not specific: blood (haemoglobin absorbs at 415/542/577 nm), melanin, plaque and glare
also darken the image, which is why :mod:`mukha_pipeline.af.confounders` annotates each region.
Regions are reported as suspicious findings for clinical examination, never as a verdict.
"""

from __future__ import annotations

import math
from typing import Literal

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.measure import label as sk_label
from skimage.measure import regionprops

from mukha_pipeline.af._common import (
    AF_DEFAULTS,
    EPS,
    as_bool_mask,
    disk,
    finite_median,
    get,
    masked_smooth,
    morph,
)
from mukha_pipeline.schema.models import (
    AFAnalysis,
    AFReference,
    AFSiteMedians,
    FVLRegion,
    NormBox,
    NormPoint,
    ReferenceType,
)
from mukha_pipeline.types import AFMaps, FVLOutput

FVLStatus = Literal["ok", "skipped_qc_failed", "insufficient_mucosa", "no_reference"]

MAX_CONTRAST = 50.0  # cap for regions with ~zero signal, keeps the value JSON-friendly
_MIN_RING_PX = 20


def _unit(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0)) if math.isfinite(x) else 0.0


def effective_min_area(cfg: dict | None, mucosa_area: int) -> int:
    """Minimum FVL region size: the larger of an absolute and a mucosa-relative floor."""
    abs_px = int(get(cfg, "min_region_area_px", AF_DEFAULTS))
    frac = float(get(cfg, "min_region_area_fraction", AF_DEFAULTS))
    return max(abs_px, int(math.ceil(frac * max(mucosa_area, 0))), 1)


def empty_fvl_output(
    shape: tuple[int, int],
    cfg: dict | None,
    *,
    status: FVLStatus,
    reference: AFReference,
    min_region_area_px: int | None = None,
) -> FVLOutput:
    """FVL output with no regions (skipped, insufficient mucosa, no reference)."""
    analysis = AFAnalysis(
        status=status,
        reference=reference,
        fvl_threshold=float(get(cfg, "fvl_relative_threshold", AF_DEFAULTS)),
        min_region_area_px=int(
            min_region_area_px
            if min_region_area_px is not None
            else get(cfg, "min_region_area_px", AF_DEFAULTS)
        ),
    )
    return FVLOutput(
        analysis=analysis,
        relative_af=np.full(shape, np.nan, dtype=np.float32),
        labels=np.zeros(shape, dtype=np.int32),
    )


def fvl_mask(relative_af: np.ndarray, mucosa: np.ndarray, cfg: dict | None) -> np.ndarray:
    """Threshold relative AF inside mucosa, open (remove speckle), close (bridge texture gaps)."""
    thr = float(get(cfg, "fvl_relative_threshold", AF_DEFAULTS))
    with np.errstate(invalid="ignore"):
        raw = mucosa & np.isfinite(relative_af) & (relative_af < thr)
    m = morph(raw, cv2.MORPH_OPEN, float(get(cfg, "morphology_open_px", AF_DEFAULTS)))
    m = morph(m, cv2.MORPH_CLOSE, float(get(cfg, "morphology_close_px", AF_DEFAULTS)))
    return m & mucosa


def _surround_contrast(
    labels: np.ndarray,
    label_id: int,
    bbox: tuple[int, int, int, int],
    relative_af: np.ndarray,
    mucosa: np.ndarray,
    ring_px: int,
    region_mean: float,
) -> float:
    """Mean relative AF in a ring around the region (healthy mucosa only) / region mean.

    The ring excludes every FVL region, so two neighbouring patches do not dilute each other's
    contrast. Falls back to 1 / region mean (the reference is the "normal" level) when there is
    no usable surround, e.g. the region fills the field of view.
    """
    h, w = labels.shape
    r0, c0, r1, c1 = bbox
    rr0, rr1 = max(r0 - ring_px - 1, 0), min(r1 + ring_px + 1, h)
    cc0, cc1 = max(c0 - ring_px - 1, 0), min(c1 + ring_px + 1, w)
    lab = labels[rr0:rr1, cc0:cc1]
    reg = lab == label_id
    grown = cv2.dilate(reg.astype(np.uint8), disk(ring_px)).astype(bool)
    ring = grown & (lab == 0) & mucosa[rr0:rr1, cc0:cc1]
    vals = relative_af[rr0:rr1, cc0:cc1][ring]
    vals = vals[np.isfinite(vals)]
    surround = float(vals.mean()) if vals.size >= _MIN_RING_PX else 1.0
    if region_mean <= EPS:
        return MAX_CONTRAST
    return float(np.clip(surround / region_mean, 0.0, MAX_CONTRAST))


def _shape_metrics(region_filled: np.ndarray) -> tuple[float, float]:
    """(border_irregularity, solidity) of a hole-filled region.

    Holes (e.g. specular dots punched out of the mucosa) are filled first so they do not count
    as border. Irregularity = 1 - 4*pi*A/P^2 (0 for a disc, higher for ragged outlines).
    """
    props = regionprops(region_filled.astype(np.uint8))
    if not props:
        return 0.0, 1.0
    p = props[0]
    perim = float(p.perimeter)
    irregularity = 1.0 - 4.0 * math.pi * float(p.area) / (perim**2) if perim > 0 else 0.0
    return _unit(irregularity), _unit(float(p.solidity))


def _measure_regions(
    labels: np.ndarray,
    relative_af: np.ndarray,
    mucosa: np.ndarray,
    cfg: dict | None,
    px_per_mm: float | None,
) -> list[tuple[FVLRegion, int]]:
    """Measure every labelled region; returns (region with placeholder id, original label)."""
    h, w = labels.shape
    mucosa_area = max(int(mucosa.sum()), 1)
    ring_px = int(get(cfg, "surround_ring_px", AF_DEFAULTS))
    mm2_per_px = (
        1.0 / float(px_per_mm) ** 2
        if px_per_mm is not None and math.isfinite(px_per_mm) and px_per_mm > 0
        else None
    )
    out: list[tuple[FVLRegion, int]] = []
    for p in regionprops(labels):
        r0, c0, r1, c1 = p.bbox
        sub = labels[r0:r1, c0:c1] == p.label
        vals = relative_af[r0:r1, c0:c1][sub]
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:  # pragma: no cover - regions are built from finite pixels
            continue
        mean_rel = float(vals.mean())
        irregularity, solidity = _shape_metrics(ndi.binary_fill_holes(sub))
        area = int(p.area)
        cy, cx = p.centroid
        region = FVLRegion(
            region_id="r0",
            area_px=area,
            area_mm2=area * mm2_per_px if mm2_per_px is not None else None,
            area_fraction_of_mucosa=_unit(area / mucosa_area),
            centroid=NormPoint(x=_unit((cx + 0.5) / w), y=_unit((cy + 0.5) / h)),
            bbox=NormBox(
                x=_unit(c0 / w), y=_unit(r0 / h), w=_unit((c1 - c0) / w), h=_unit((r1 - r0) / h)
            ),
            mean_relative_af=max(mean_rel, 0.0),
            min_relative_af=max(float(np.percentile(vals, 5)), 0.0),
            contrast_vs_surround=_surround_contrast(
                labels, int(p.label), p.bbox, relative_af, mucosa, ring_px, mean_rel
            ),
            border_irregularity=irregularity,
            solidity=solidity,
        )
        out.append((region, int(p.label)))
    return out


def _site_medians(maps: AFMaps, mucosa: np.ndarray, relative_af: np.ndarray) -> AFSiteMedians | None:
    meds = [
        finite_median(arr[mucosa])
        for arr in (maps.green, maps.gr_ratio, maps.af_index, relative_af)
    ]
    g, gr, idx, rel = meds
    if g is None or gr is None or idx is None or rel is None:
        return None
    return AFSiteMedians(green=g, gr_ratio=gr, af_index=idx, relative_af=rel)


def detect_fvl(
    maps: AFMaps,
    mucosa: np.ndarray,
    reference: AFReference,
    cfg: dict | None,
    *,
    px_per_mm: float | None = None,
) -> FVLOutput:
    """Find FVL regions in the analysable mucosa (cfg = analysis.af).

    Statuses: ``insufficient_mucosa`` when too little tissue is analysable, ``no_reference``
    when there is no usable reference level; both return no regions and an all-NaN relative AF
    (the triage then cannot call the site low risk on AF grounds). Regions are ordered by
    severity: lowest mean relative AF first, larger area first on ties; ids are "r1".."rN" and
    ``labels`` uses the same numbering.
    """
    shape = maps.green.shape
    muc = as_bool_mask(mucosa, shape, "mucosa") & np.isfinite(maps.green)
    n_px = max(int(np.prod(shape)), 1)
    mucosa_area = int(muc.sum())
    min_area = effective_min_area(cfg, mucosa_area)
    min_frac = float(get(cfg, "insufficient_mucosa_fraction", AF_DEFAULTS))
    min_px = max(int(get(cfg, "reference_min_pixels", AF_DEFAULTS)), min_area)
    if mucosa_area / n_px < min_frac or mucosa_area < min_px:
        return empty_fvl_output(
            shape, cfg, status="insufficient_mucosa", reference=reference, min_region_area_px=min_area
        )
    ref = reference.reference_green
    if reference.type == ReferenceType.NONE or ref is None or not math.isfinite(ref) or ref <= 0:
        return empty_fvl_output(
            shape, cfg, status="no_reference", reference=reference, min_region_area_px=min_area
        )

    smoothed = masked_smooth(maps.green, muc, float(get(cfg, "smoothing_sigma_px", AF_DEFAULTS)))
    relative = (smoothed / float(ref)).astype(np.float32)
    relative[~muc] = np.nan

    cleaned = fvl_mask(relative, muc, cfg)
    raw_labels = sk_label(cleaned, connectivity=2)
    if raw_labels.max() > 0:
        areas = np.bincount(raw_labels.ravel())
        keep = areas >= min_area
        keep[0] = False
        raw_labels = np.where(keep[raw_labels], raw_labels, 0)

    measured = _measure_regions(raw_labels, relative, muc, cfg, px_per_mm)
    measured.sort(key=lambda t: (t[0].mean_relative_af, -t[0].area_px))
    lut = np.zeros(int(raw_labels.max()) + 1, dtype=np.int32)
    regions: list[FVLRegion] = []
    for k, (region, old) in enumerate(measured, start=1):
        lut[old] = k
        regions.append(region.model_copy(update={"region_id": f"r{k}"}))
    labels = lut[raw_labels].astype(np.int32)

    analysis = AFAnalysis(
        status="ok",
        reference=reference,
        fvl_threshold=float(get(cfg, "fvl_relative_threshold", AF_DEFAULTS)),
        min_region_area_px=min_area,
        site_medians=_site_medians(maps, muc, relative),
        fvl_area_fraction=_unit(float((labels > 0).sum()) / max(mucosa_area, 1)),
        regions=regions,
    )
    return FVLOutput(analysis=analysis, relative_af=relative, labels=labels)
