"""Things that darken the AF image without a tissue change, flagged per FVL region.

* Porphyrin red: bacteria in plaque or tongue coating make protoporphyrin/coproporphyrin, which
  fluoresce red (~635 nm) and cover the tissue's green glow. Only an EXCESS of red counts: the
  red channel must be well above this image's typical mucosa red. A high red/green ratio alone is
  not enough, because in a deep fluorescence loss green falls more than red, so red/green rises
  with no red emission at all (and a raised red/green is itself reported in dysplasia).
* Pigmentation: melanin absorbs both the 405 nm excitation and the green emission. Melanin brown
  has little redness (Lab a* about 8-17), so reddish pixels are never treated as pigment.
* Blood / red area: haemoglobin absorbs strongly at 415 nm (Soret band) and 542/577 nm, so
  haemorrhage or inflammation looks dark. A dark reddish area may equally be a red patch
  (erythroplakia), so the triage rules never discount this class; they keep the lesion at least
  MODERATE instead. Diascopy (pressing a glass slide) helps tell blood from tissue change.
* Glare: specular highlights carry no tissue signal at all.

A flag lowers how much an FVL region counts in the triage rules but never removes it: a red patch
that also looks like blood still needs a clinician. Notes say "may", never "is".

AF thresholds are relative to the same image's mucosa (the AF image is linear light, scaled by
exposure and calibration), except the saturation test, which is about the sensor clipping and is
therefore judged on the unscaled values (pass ``intensity_scale``).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy import ndimage as ndi

from mukha_pipeline.af._common import (
    CONFOUNDER_DEFAULTS,
    as_bool_mask,
    disk,
    finite_median,
    get,
    masked_smooth,
    to_float_rgb,
)
from mukha_pipeline.schema.models import ConfounderFlag, ConfounderType
from mukha_pipeline.types import AFMaps, FVLOutput, Masks

NOTES: dict[ConfounderType, str] = {
    ConfounderType.PORPHYRIN_RED: (
        "Red fluorescence from plaque or bacteria overlaps this area; the darkness may not "
        "reflect a tissue change. Clean the area and capture again if possible."
    ),
    ConfounderType.PIGMENTATION: (
        "Dark brown pigment overlaps this area; pigment absorbs the violet light and the green "
        "glow, so the darkness may not reflect a tissue change."
    ),
    ConfounderType.BLOOD: (
        "Deep red colour (blood or inflammation) overlaps this area; blood absorbs violet light, "
        "so the darkness may partly come from blood. Pressing a glass slide on the area "
        "(diascopy) helps tell them apart. A red patch still needs clinical examination."
    ),
    ConfounderType.GLARE: (
        "Glare (shiny reflection) overlaps this area, so part of it could not be measured."
    ),
}


@dataclass
class ConfounderMasks:
    """Bool (H, W) masks of each confounder in the WL frame."""

    porphyrin_red: np.ndarray
    pigmentation: np.ndarray
    blood: np.ndarray
    glare: np.ndarray

    def get(self, kind: ConfounderType) -> np.ndarray:
        return getattr(self, kind.value)


def _section(cfg: dict | None, key: str) -> dict:
    """Accept the full analysis config (preferred) or the bare section."""
    if not cfg:
        return {}
    sub = cfg.get(key)
    return sub if isinstance(sub, dict) else cfg


def _valid_scale(intensity_scale: float) -> float:
    """Same fallback as :func:`compute_af_maps`: non-finite or non-positive -> 1.0."""
    try:
        scale = float(intensity_scale)
    except (TypeError, ValueError):
        return 1.0
    return scale if np.isfinite(scale) and scale > 0 else 1.0


def porphyrin_mask(maps: AFMaps, mucosa: np.ndarray, cfg: dict | None) -> np.ndarray:
    """Red porphyrin fluorescence: a red EXCESS over this image's typical mucosa red.

    A pixel is flagged when its (lightly smoothed) red is at least ``porphyrin_red_rel_min`` x
    the median mucosa red AND red/green >= ``porphyrin_red_over_green_min``. The red reference
    is floored at ``porphyrin_red_ref_min_of_green`` x the median mucosa green, so a near-zero red
    channel (noise only) cannot make every dark pixel look like porphyrin. Both tests are ratios,
    so the result does not depend on exposure or calibration scaling. Only mucosa is evaluated.
    """
    c = cfg or {}
    shape = maps.green.shape
    muc = as_bool_mask(mucosa, shape, "mucosa") & np.isfinite(maps.red) & np.isfinite(maps.green)
    out = np.zeros(shape, dtype=bool)
    med_red, med_green = finite_median(maps.red[muc]), finite_median(maps.green[muc])
    if med_red is None or med_green is None:
        return out
    floor = float(get(c, "porphyrin_red_ref_min_of_green", CONFOUNDER_DEFAULTS)) * med_green
    red_ref = max(med_red, floor)
    if not np.isfinite(red_ref) or red_ref <= 0:
        return out
    sigma = float(get(c, "porphyrin_smoothing_sigma_px", CONFOUNDER_DEFAULTS))
    red = masked_smooth(maps.red, muc, sigma)
    green = masked_smooth(maps.green, muc, sigma)
    rel_min = float(get(c, "porphyrin_red_rel_min", CONFOUNDER_DEFAULTS))
    ratio_min = float(get(c, "porphyrin_red_over_green_min", CONFOUNDER_DEFAULTS))
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = red / np.maximum(green, 1e-6)
        out = muc & np.isfinite(red) & (red >= rel_min * red_ref) & (ratio >= ratio_min)
    return out


def confounder_masks(
    wl: np.ndarray,
    maps: AFMaps,
    masks: Masks,
    cfg: dict | None,
    *,
    intensity_scale: float = 1.0,
) -> ConfounderMasks:
    """Compute confounder masks (``cfg`` = the ``confounders`` section).

    ``intensity_scale`` is the factor :func:`compute_af_maps` multiplied the AF image by
    (exposure x calibration). It is only used to judge AF saturation on the sensor's own scale.
    """
    shape = maps.green.shape
    img = to_float_rgb(wl)
    if img.shape[:2] != shape:
        raise ValueError(f"wl has shape {img.shape[:2]}, expected {shape}")
    c = cfg or {}
    mucosa = as_bool_mask(masks.mucosa, shape, "masks.mucosa")

    porphyrin = porphyrin_mask(maps, mucosa, c)

    lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    lab_l, lab_a = lab[:, :, 0], lab[:, :, 1]
    hue, sat = hsv[:, :, 0] / 2.0, hsv[:, :, 1]
    # Dark and reddish = blood / red area (possible red patch): never discounted by the rules.
    blood = (lab_a >= float(get(c, "blood_lab_a_min", CONFOUNDER_DEFAULTS))) & (
        lab_l <= float(get(c, "blood_lab_l_max", CONFOUNDER_DEFAULTS))
    )
    h_lo, h_hi = (float(v) for v in get(c, "pigmentation_hue_range", CONFOUNDER_DEFAULTS))
    # Pigment is dark brown with little redness. Reddish pixels (a* >= pigmentation_lab_a_max)
    # are never pigment, and blood takes precedence so a pixel is not discounted twice.
    pigmentation = (
        (lab_l <= float(get(c, "pigmentation_lab_l_max", CONFOUNDER_DEFAULTS)))
        & (lab_a < float(get(c, "pigmentation_lab_a_max", CONFOUNDER_DEFAULTS)))
        & (hue >= h_lo)
        & (hue <= h_hi)
        & (sat >= float(get(c, "pigmentation_saturation_min", CONFOUNDER_DEFAULTS)))
        & ~blood
    )

    sat_level = float(get(c, "af_saturation_min", CONFOUNDER_DEFAULTS))
    scale = _valid_scale(intensity_scale)
    with np.errstate(invalid="ignore"):
        af_max = np.fmax(np.fmax(maps.red, maps.green), maps.blue) / scale
        af_saturated = np.isfinite(af_max) & (af_max >= sat_level)
    glare = as_bool_mask(masks.specular, shape, "masks.specular") | af_saturated
    return ConfounderMasks(
        porphyrin_red=porphyrin, pigmentation=pigmentation, blood=blood, glare=glare
    )


def region_overlaps(
    region: np.ndarray, conf: ConfounderMasks, glare_dilate_px: int
) -> dict[ConfounderType, float]:
    """Fraction of the (hole-filled) region covered by each confounder.

    Glare pixels are cut out of the analysable mucosa, so they appear as holes in or next to a
    region; the glare footprint is therefore the filled region grown by a few pixels.
    """
    footprint = ndi.binary_fill_holes(region)
    n = int(footprint.sum())
    out: dict[ConfounderType, float] = {}
    if n == 0:
        return {t: 0.0 for t in ConfounderType}
    for kind in (ConfounderType.PORPHYRIN_RED, ConfounderType.PIGMENTATION, ConfounderType.BLOOD):
        out[kind] = float((conf.get(kind) & footprint).sum()) / n
    grown = footprint
    if glare_dilate_px > 0:
        grown = cv2.dilate(footprint.astype(np.uint8), disk(glare_dilate_px)).astype(bool)
    out[ConfounderType.GLARE] = float((conf.glare & grown).sum()) / max(int(grown.sum()), 1)
    return out


def flag_confounders(
    fvl: FVLOutput,
    wl: np.ndarray,
    maps: AFMaps,
    masks: Masks,
    cfg: dict | None,
    *,
    intensity_scale: float = 1.0,
) -> FVLOutput:
    """Annotate each FVL region with confounders (``cfg`` = full analysis config).

    Returns a new :class:`FVLOutput`; the input is not modified. Also sets
    ``analysis.porphyrin_red_fraction`` (share of analysable mucosa with red fluorescence), which
    is reported even when no region was found. ``intensity_scale`` must be the value passed to
    :func:`compute_af_maps` so AF saturation (glare) is judged on the sensor's own scale.
    """
    c = _section(cfg, "confounders")
    conf = confounder_masks(wl, maps, masks, c, intensity_scale=intensity_scale)
    shape = maps.green.shape
    mucosa = as_bool_mask(masks.mucosa, shape, "masks.mucosa") & np.isfinite(maps.green)
    n_mucosa = int(mucosa.sum())
    porph_frac = float((conf.porphyrin_red & mucosa).sum()) / n_mucosa if n_mucosa else 0.0

    labels = np.asarray(fvl.labels)
    if labels.shape != shape:
        raise ValueError(f"fvl.labels has shape {labels.shape}, expected {shape}")
    min_overlap = float(get(c, "flag_overlap_min", CONFOUNDER_DEFAULTS))
    glare_px = int(get(c, "glare_dilate_px", CONFOUNDER_DEFAULTS))
    regions = []
    for idx, region in enumerate(fvl.analysis.regions, start=1):
        label_id = _label_of(region.region_id, idx)
        overlaps = region_overlaps(labels == label_id, conf, glare_px)
        flags = [
            ConfounderFlag(type=kind, overlap_fraction=float(np.clip(frac, 0, 1)), note=NOTES[kind])
            for kind, frac in overlaps.items()
            if frac >= min_overlap and frac > 0
        ]
        regions.append(region.model_copy(update={"confounders": flags}))
    analysis = fvl.analysis.model_copy(
        update={"regions": regions, "porphyrin_red_fraction": float(np.clip(porph_frac, 0, 1))}
    )
    return FVLOutput(analysis=analysis, relative_af=fvl.relative_af, labels=fvl.labels)


def _label_of(region_id: str, fallback: int) -> int:
    """Label value for region "rK" (falls back to list position for unexpected ids)."""
    if region_id.startswith("r") and region_id[1:].isdigit():
        return int(region_id[1:])
    return fallback
