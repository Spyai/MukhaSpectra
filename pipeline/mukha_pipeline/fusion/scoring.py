"""Per-lesion rule scoring.

Order of operations (each step is a factor, so ``score == sum(factor.points)``):

1. AF points for the FVL region (presence, depth, area, border, contrast), then the AF-only
   multipliers as deltas: reference type, diascopy, confounders (combined, floored), and the
   low-specificity site discount.
2. WL model points (only when the model actually ran).
3. Spectral delta points (only lesion-minus-reference deltas are scored).
4. The health worker's visual observation.
5. The site high-risk multiplier on the lesion total, as one delta.

Then the level: score thresholds, raised by any visual-appearance floor that applies to this
lesion, and INCONCLUSIVE when a LOW lesion sits where the device could not look.
"""

from __future__ import annotations

from dataclasses import dataclass

from mukha_pipeline.fusion.factors import POINT_DECIMALS, Ledger, total_points
from mukha_pipeline.fusion.lesions import LesionDraft, site_readable
from mukha_pipeline.fusion.overrides import (
    ClinicalContext,
    highest_floor,
    matching_overrides,
    override_factor,
)
from mukha_pipeline.fusion.rules import Override, TriageRules
from mukha_pipeline.fusion.text import (
    APPEARANCE_NOUN,
    CONFOUNDER_NOUN,
    WL_NOUN,
    capitalize,
    join_words,
    site_phrase,
)
from mukha_pipeline.schema.models import (
    RISK_ORDER,
    ContributingFactor,
    Diascopy,
    FactorSource,
    FVLRegion,
    ReferenceType,
    RiskFactors,
    RiskLevel,
    SiteResult,
    SpectralReadingResult,
    VisualAppearance,
    VisualFlag,
    WLPrediction,
)
from mukha_pipeline.sites import site_label

AF = FactorSource.AUTOFLUORESCENCE

_REFERENCE_LABEL: dict[ReferenceType, str] = {
    ReferenceType.CONTRALATERAL: "Compared with the same site on the opposite side",
    ReferenceType.ANALOGOUS_SITE: (
        "Compared with a similar site rather than the opposite side, so it counts a little less"
    ),
    ReferenceType.IN_IMAGE_AUTO: (
        "No matching healthy site, so compared with surrounding tissue in the same photo "
        "(less reliable)"
    ),
    ReferenceType.NONE: "No reference tissue for comparison (much less reliable)",
}

_DIASCOPY_LABEL: dict[Diascopy, str] = {
    Diascopy.NOT_DONE: "Glass-slide pressure test (diascopy) not done",
    Diascopy.FVL_PERSISTS: "Dark area stayed dark under glass-slide pressure (diascopy)",
    Diascopy.FVL_RESOLVES: (
        "Fluorescence came back under glass-slide pressure (diascopy): more likely blood "
        "vessels or inflammation"
    ),
}


@dataclass
class ScoredLesion:
    draft: LesionDraft
    lesion_id: str
    factors: list[ContributingFactor]
    score: float
    score_level: RiskLevel
    level: RiskLevel
    floors: list[Override]
    readable: bool


# --------------------------------------------------------------------------------------------
# Evidence groups
# --------------------------------------------------------------------------------------------


def _confounder_multiplier(region: FVLRegion, rules: TriageRules) -> tuple[float, list[str]]:
    mult = 1.0
    names: list[str] = []
    for c in region.confounders:
        mult *= 1.0 - rules.af.confounder_discount[c.type] * c.overlap_fraction
        names.append(f"{CONFOUNDER_NOUN[c.type]} over {c.overlap_fraction:.0%} of it")
    return max(mult, rules.af.min_confounder_multiplier), names


def af_factors(region: FVLRegion, site: SiteResult, rules: TriageRules) -> list[ContributingFactor]:
    """Points for one fluorescence-visualisation-loss region and its AF-only adjustments."""
    assert site.af is not None
    ar = rules.af
    threshold = site.af.fvl_threshold
    led = Ledger()
    led.add(
        "af.fvl_present",
        "Dark area under violet light (loss of normal fluorescence)",
        AF,
        ar.fvl_present,
        f"Region {region.region_id}",
    )
    depth = min(ar.depth_max, ar.depth_points_per_0_1 * (threshold - region.mean_relative_af) / 0.1)
    if depth > 0:
        led.add(
            "af.depth",
            f"Fluorescence {region.mean_relative_af:.2f} of the reference, below the "
            f"{threshold:.2f} cut-off",
            AF,
            depth,
            f"{ar.depth_points_per_0_1} points per 0.1 below the cut-off, at most {ar.depth_max}",
        )
    if region.area_mm2 is not None:
        tier = ar.area_mm2.best(region.area_mm2)
        area_label = f"Dark area about {region.area_mm2:.0f} mm²"
    else:
        tier = ar.area_fraction.best(region.area_fraction_of_mucosa)
        area_label = f"Dark area covers {region.area_fraction_of_mucosa:.0%} of the visible mucosa"
    if tier is not None and tier.points:
        led.add("af.area", area_label, AF, tier.points, f"Tier from {tier.bound:g}")
    if region.border_irregularity >= ar.border_irregularity.minimum:
        led.add(
            "af.border_irregular",
            f"Irregular border (irregularity {region.border_irregularity:.2f})",
            AF,
            ar.border_irregularity.points,
        )
    if region.contrast_vs_surround >= ar.contrast_vs_surround.minimum:
        led.add(
            "af.contrast",
            f"Much darker than the tissue around it (contrast {region.contrast_vs_surround:.2f})",
            AF,
            ar.contrast_vs_surround.points,
        )
    ref_type = site.af.reference.type
    led.scale(
        ar.reference_multipliers[ref_type],
        f"af.reference.{ref_type.value}",
        _REFERENCE_LABEL[ref_type],
        AF,
    )
    led.scale(
        ar.diascopy_multipliers[site.diascopy],
        f"af.diascopy.{site.diascopy.value}",
        _DIASCOPY_LABEL[site.diascopy],
        AF,
    )
    conf_mult, conf_names = _confounder_multiplier(region, rules)
    led.scale(
        conf_mult,
        "af.confounders",
        f"Possible look-alikes overlap the dark area ({join_words(conf_names)}), so it counts less",
        AF,
        detail=f"combined multiplier floored at {ar.min_confounder_multiplier:.2f}",
    )
    if site.site in rules.site.af_low_specificity_sites:
        led.scale(
            rules.site.af_low_specificity_multiplier,
            "af.low_specificity_site",
            f"Dark areas at {site_phrase(site.site)} are often due to inflammation, coating or "
            "keratin, so they count less",
            AF,
        )
    return led.factors


def wl_factors(pred: WLPrediction, rules: TriageRules) -> list[ContributingFactor]:
    led = Ledger()
    led.add(
        f"wl.{pred.label.value}",
        f"Photo model (unvalidated) suggests {WL_NOUN[pred.label]} "
        f"(model score {pred.probability:.2f})",
        FactorSource.WL_MODEL,
        rules.wl.points[pred.label],
    )
    return led.factors


def spectral_factors(
    readings: list[SpectralReadingResult], rules: TriageRules
) -> list[ContributingFactor]:
    """Score lesion-minus-reference deltas; per feature the best tier over all readings wins."""
    sr = rules.spectral
    led = Ledger()
    best: dict[str, tuple[float, float, str]] = {}  # feature -> (points, delta, reading_id)
    for r in readings:
        if not r.qc.passed:
            led.note(
                "spectral.qc_failed",
                f"Probe reading {r.reading_id} failed quality checks and was not scored",
                FactorSource.SPECTRAL,
            )
            continue
        if r.delta_vs_reference is None:
            # v1 scores deltas only; absolute features are uninterpretable without a reference,
            # so the reading contributes 0 points x no_reference_multiplier = 0.
            led.note(
                "spectral.no_reference",
                f"Probe reading {r.reading_id} had no reference reading from the healthy side, "
                "so it was not scored",
                FactorSource.SPECTRAL,
                f"no_reference_multiplier {sr.no_reference_multiplier} applied to 0 points",
            )
            continue
        delta = r.delta_vs_reference
        for name, rule, value in (
            ("hb", sr.hb_absorption_proxy_delta, delta.hb_absorption_proxy),
            ("ratio", sr.ratio_555_630_delta, delta.ratio_555_630),
        ):
            tier = rule.best(value)
            if tier is not None and tier.points > best.get(name, (0.0, 0.0, ""))[0]:
                best[name] = (tier.points, value, r.reading_id)
    if "hb" in best:
        pts, value, rid = best["hb"]
        led.add(
            "spectral.hb_absorption",
            f"Deeper blood-absorption dip (540–580 nm) than the reference reading ({value:+.3f})",
            FactorSource.SPECTRAL,
            pts,
            f"Reading {rid}",
        )
    if "ratio" in best:
        pts, value, rid = best["ratio"]
        led.add(
            "spectral.ratio_555_630",
            f"Lower 555/630 nm reflectance ratio than the reference reading ({value:+.3f})",
            FactorSource.SPECTRAL,
            pts,
            f"Reading {rid}",
        )
    return led.factors


def best_appearance(flag: VisualFlag, rules: TriageRules) -> VisualAppearance:
    """Highest-scoring recorded appearance; 'other' when a lesion was ticked without one."""
    if not flag.appearance:
        return VisualAppearance.OTHER
    return max(flag.appearance, key=lambda a: rules.visual_points[a])


def visual_factors(flag: VisualFlag, rules: TriageRules) -> list[ContributingFactor]:
    app = best_appearance(flag, rules)
    detail = None
    if len(flag.appearance) > 1:
        detail = "Also recorded: " + join_words(
            [APPEARANCE_NOUN[a] for a in flag.appearance if a != app]
        )
    led = Ledger()
    led.add(
        f"visual.seen.{app.value}",
        f"{capitalize(APPEARANCE_NOUN[app])} seen by the health worker",
        FactorSource.CLINICAL_VISUAL,
        rules.visual_points[app],
        detail,
    )
    return led.factors


# --------------------------------------------------------------------------------------------
# Lesion
# --------------------------------------------------------------------------------------------


def _availability_notes(draft: LesionDraft, readable: bool, led: Ledger) -> None:
    if draft.site_result is None:
        led.note(
            "qc.no_photos_at_site",
            "No photos were analysed at this site; only the probe reading is available",
            FactorSource.QC,
        )
    elif not readable:
        led.note(
            "qc.device_analysis_unavailable",
            "Device analysis not available at this site (images could not be analysed); "
            "the health worker's observation still counts",
            FactorSource.QC,
        )
    elif draft.region is None:
        led.note(
            "af.no_fvl_at_site",
            "No dark area under violet light at this site; this does not lower the concern "
            "from the other findings",
            AF,
        )


def score_lesion(
    draft: LesionDraft, lesion_id: str, rules: TriageRules, risk_factors: RiskFactors
) -> ScoredLesion:
    """Score one lesion and decide its level."""
    readable = draft.site_result is not None and site_readable(draft.site_result)
    led = Ledger()
    if draft.region is not None and draft.site_result is not None:
        led.extend(af_factors(draft.region, draft.site_result, rules))
    if draft.wl is not None:
        led.extend(wl_factors(draft.wl, rules))
    if draft.spectral:
        led.extend(spectral_factors(draft.spectral, rules))
    if draft.visual is not None:
        led.extend(visual_factors(draft.visual, rules))
    _availability_notes(draft, readable, led)
    if draft.site in rules.site.high_risk_sites:
        led.scale(
            rules.site.high_risk_multiplier,
            "site.high_risk",
            f"{site_label(draft.site)} is a common site for oral cancer and pre-cancer in "
            "tobacco and areca-nut users",
            FactorSource.SITE,
        )
    score = round(total_points(led.factors), POINT_DECIMALS)
    score_level = rules.level_for_score(score)

    floors: list[Override] = []
    if draft.visual is not None:
        ctx = ClinicalContext(risk_factors, frozenset(draft.visual.appearance), any_lesion=True)
        floors = matching_overrides(rules.overrides, ctx, visual_only=True)
    level = score_level
    floor = highest_floor(floors)
    if floor is not None and RISK_ORDER[floor] > RISK_ORDER[level]:
        level = floor
    for o in floors:
        led.factors.append(override_factor(o, "this lesion's"))
    if (
        level == RiskLevel.LOW
        and not readable
        and rules.inconclusive.low_requires_all_examined_sites_analysed
    ):
        level = RiskLevel.INCONCLUSIVE
    return ScoredLesion(draft, lesion_id, led.factors, score, score_level, level, floors, readable)
