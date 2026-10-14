"""Lesion building: gather every piece of evidence about one spot into one lesion draft.

Evidence sources, in order: AF fluorescence-loss regions, the health worker's own visual flag,
the white-light model, and lesion-role probe readings. Nothing is dropped: reference-only sites
and sites whose images failed QC still contribute whatever evidence they have, because the worst
failure for a screening aid is ignoring a finding.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from mukha_pipeline.fusion.rules import WLRules
from mukha_pipeline.schema.models import (
    FVLRegion,
    LesionOrigin,
    SiteResult,
    SpectralReadingResult,
    VisualFlag,
    WLAnalysis,
    WLLabel,
    WLPrediction,
)
from mukha_pipeline.sites import OralSite

SPECTRAL_LESION_LABELS: frozenset[str] = frozenset({"suspicious", "indeterminate"})


@dataclass
class LesionDraft:
    """Evidence for one lesion before scoring."""

    site: OralSite
    site_result: SiteResult | None
    origins: list[LesionOrigin] = field(default_factory=list)
    region: FVLRegion | None = None
    visual: VisualFlag | None = None
    wl: WLPrediction | None = None
    spectral: list[SpectralReadingResult] = field(default_factory=list)

    def add_origin(self, origin: LesionOrigin) -> None:
        if origin not in self.origins:
            self.origins.append(origin)


def site_readable(site: SiteResult) -> bool:
    """True when the device could actually look for fluorescence loss at this site.

    Stricter than ``analysis_status == "analysed"``: a site whose AF step reported
    insufficient mucosa or no reference produced no FVL regions, and treating that silence as
    "nothing found" would be false reassurance.
    """
    return site.analysis_status == "analysed" and site.af is not None and site.af.status == "ok"


def visual_seen(flag: VisualFlag) -> bool:
    """A recorded appearance counts as a lesion seen even if the checkbox was missed."""
    return flag.lesion_seen or bool(flag.appearance)


def top_wl_prediction(wl: WLAnalysis) -> WLPrediction | None:
    """The model's top prediction with its probability (None if it cannot be determined)."""
    if wl.top_label is not None:
        matches = [p for p in wl.predictions if p.label == wl.top_label]
        return max(matches, key=lambda p: p.probability, default=None)
    return max(wl.predictions, key=lambda p: p.probability, default=None)


def wl_finding(site: SiteResult, rules: WLRules) -> WLPrediction | None:
    """A confident non-normal WL prediction, or None. A missing model is never a finding."""
    if site.wl is None or site.wl.status != "ok":
        return None
    top = top_wl_prediction(site.wl)
    if top is None or top.label == WLLabel.NORMAL or top.probability < rules.min_probability:
        return None
    return top


def primary_draft(drafts: list[LesionDraft]) -> LesionDraft | None:
    """The darkest FVL region at a site, else the first non-region lesion there."""
    regions = [d for d in drafts if d.region is not None]
    if regions:
        return min(regions, key=lambda d: d.region.mean_relative_af)  # type: ignore[union-attr]
    return drafts[0] if drafts else None


def _site_drafts(site: SiteResult, wl_rules: WLRules) -> list[LesionDraft]:
    drafts: list[LesionDraft] = []
    if site_readable(site) and site.af is not None:
        for region in site.af.regions:
            drafts.append(LesionDraft(site.site, site, [LesionOrigin.AF_FVL], region=region))
    if visual_seen(site.visual_flag):
        target = primary_draft(drafts)
        if target is None:
            target = LesionDraft(site.site, site)
            drafts.append(target)
        target.visual = site.visual_flag
        target.add_origin(LesionOrigin.VISUAL)
    finding = wl_finding(site, wl_rules)
    if finding is not None:
        target = primary_draft(drafts)
        if target is None:
            target = LesionDraft(site.site, site)
            drafts.append(target)
        target.wl = finding
        target.add_origin(LesionOrigin.WL_MODEL)
    return drafts


def _attach_spectral(
    reading: SpectralReadingResult,
    by_site: dict[OralSite, list[LesionDraft]],
    site_results: dict[OralSite, SiteResult],
) -> None:
    drafts = by_site.get(reading.site, [])
    target = None
    if reading.lesion_ref:
        target = next(
            (d for d in drafts if d.region and d.region.region_id == reading.lesion_ref), None
        )
    if target is None:
        target = primary_draft(drafts)
    if target is None:
        label = reading.classifier.label if reading.classifier else None
        if label not in SPECTRAL_LESION_LABELS:
            return
        target = LesionDraft(reading.site, site_results.get(reading.site))
        by_site.setdefault(reading.site, []).append(target)
    target.spectral.append(reading)
    target.add_origin(LesionOrigin.SPECTRAL)


def build_lesion_drafts(
    sites: list[SiteResult], spectral: list[SpectralReadingResult], wl_rules: WLRules
) -> list[LesionDraft]:
    """Collect lesion drafts, grouped by site in input order."""
    by_site: dict[OralSite, list[LesionDraft]] = {}
    site_results: dict[OralSite, SiteResult] = {}
    for site in sites:
        site_results.setdefault(site.site, site)
        by_site.setdefault(site.site, []).extend(_site_drafts(site, wl_rules))
    for reading in spectral:
        if reading.role == "lesion":
            _attach_spectral(reading, by_site, site_results)
    return [d for drafts in by_site.values() for d in drafts]
