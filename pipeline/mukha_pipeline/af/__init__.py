"""Autofluorescence (AF) analysis: maps, reference normalisation, FVL, confounders, overlay.

Under 405 nm light healthy mucosa glows green; a local loss of that glow (fluorescence
visualisation loss, FVL) is a suspicious finding that needs clinical examination. It is not
specific on its own, so every region is annotated with confounders (plaque porphyrins, pigment,
blood, glare) and measured against the patient's own reference tissue.
"""

from mukha_pipeline.af.confounders import (
    ConfounderMasks,
    confounder_masks,
    flag_confounders,
    region_overlaps,
)
from mukha_pipeline.af.fvl import (
    FVLStatus,
    analysable_mucosa,
    detect_fvl,
    effective_min_area,
    empty_fvl_output,
)
from mukha_pipeline.af.grid import block_nanmean, relative_af_grid
from mukha_pipeline.af.maps import compute_af_maps, reference_green
from mukha_pipeline.af.overlay import (
    af_loss,
    heatmap_legend,
    loss_colormap,
    render_heatmap_overlay,
)
from mukha_pipeline.af.reference import resolve_reference

__all__ = [
    "ConfounderMasks",
    "FVLStatus",
    "af_loss",
    "analysable_mucosa",
    "block_nanmean",
    "compute_af_maps",
    "confounder_masks",
    "detect_fvl",
    "effective_min_area",
    "empty_fvl_output",
    "flag_confounders",
    "heatmap_legend",
    "loss_colormap",
    "reference_green",
    "region_overlaps",
    "relative_af_grid",
    "render_heatmap_overlay",
    "resolve_reference",
]
