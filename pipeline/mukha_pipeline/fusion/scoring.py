"""Per-lesion rule scoring.

Order of operations (each step is a factor, so ``score == sum(factor.points)``):

1. AF points for the FVL region (presence, depth, area, border, contrast), then the AF-only
   multipliers as deltas: reference type, diascopy, confounders (combined, floored), and the
   low-specificity site discount.
   * Diascopy is one test recorded once per site. "Fluorescence came back" is applied only when
     the site has exactly one FVL region; "stayed dark" only to the darkest region. Otherwise
     a zero-point note says why it was not applied.
   * At a site with no paired site the in-image reference is the only one possible, so the
     reference and low-specificity discounts are not compounded there: only the larger of the
     two (smaller multiplier) applies, with a zero-point note.
2. WL model points (only when the model actually ran).
3. Spectral delta points (only lesion-minus-reference deltas are scored).
4. The health worker's visual observation.
5. The site high-risk multiplier on the lesion total, as one delta.

Then the level: score thresholds, raised by floors that apply to this lesion (the health
worker's observation, a red area with FVL, strong FVL whatever the discounts), and
INCONCLUSIVE when a LOW lesion sits where the device could not look properly. Floors are
zero-point factors, so the score stays the exact sum of the factors.
"""

from __future__ import annotations

from dataclasses import dataclass

from mukha_pipeline.fusion.factors import POINT_DECIMALS, Ledger, total_points
from mukha_pipeline.fusion.lesions import (
    LesionDraft,
    probe_only_unchecked,
    segmentation_reliable,
    site_analysed,
    site_readable,
    visual_seen,
)
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
    ConfounderType,
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

# Look-alikes the health worker can remove before retaking: plaque (porphyrin red fluorescence)
# with gauze or a rinse, and glare by drying and re-angling. Pigment cannot be cleaned and blood
# is never discounted (see red_area_floor).
CLEANABLE_CONFOUNDERS: frozenset[ConfounderType] = frozenset(
    {ConfounderType.PORPHYRIN_RED, ConfounderType.GLARE}
)

_REFERENCE_LABEL: dict[ReferenceType, str] = {
    ReferenceType.CONTRALATERAL: "Compared with the same site on the opposite side",
    ReferenceType.ANALOGOUS_SITE: (
        "Compared with a similar site rather than the opposite side, so it counts a little less"
    ),
    ReferenceType.IN_IMAGE_AUTO: (
        "No opposite-side or similar site to compare with, so compared with surrounding tissue "
        "in the same photo (less reliable)"
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
    readable: bool  # the site was readable (analysed, reliable segmentation)
    assessed: bool = True  # readable, and the device had a usable measurement of this lesion
    af_raw_points: float = 0.0  # AF points before any multiplier
    cleanable: tuple[ConfounderType, ...] = ()  # look-alikes a clean and retake could remove
    diascopy_applied: Diascopy = Diascopy.NOT_DONE


@dataclass(frozen=True)
class AFPoints:
    factors: list[ContributingFactor]
    raw_points: float  # before any multiplier
    diascopy_applied: Diascopy


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


def diascopy_for_region(region: FVLRegion, site: SiteResult) -> tuple[Diascopy, str | None]:
    """The diascopy result that applies to this region, and why it does not when it does not.

    Diascopy is a manual test on one dark spot but is recorded once per site, and the record
    does not say which region was pressed. "Fluorescence came back" lowers the score a lot, so
    it is applied only when the site has a single region; "stayed dark" raises it, so it is
    applied to the darkest (primary) region only.
    """
    assert site.af is not None
    recorded = site.diascopy
    if recorded == Diascopy.NOT_DONE:
        return recorded, None
    regions = site.af.regions
    n = len(regions)
    if recorded == Diascopy.FVL_RESOLVES:
        if n <= 1:
            return recorded, None
        return Diascopy.NOT_DONE, (
            f"Glass-slide pressure test recorded once for this site, which has {n} dark areas; "
            "it is not known which one was tested, so the result was not applied"
        )
    darkest = min(regions, key=lambda r: r.mean_relative_af) if regions else region
    if darkest.region_id == region.region_id:
        return recorded, None
    return Diascopy.NOT_DONE, (
        "Glass-slide pressure test recorded once for this site; its result was applied only to "
        f"the darkest dark area there ({darkest.region_id})"
    )


@dataclass(frozen=True)
class _Discounts:
    """Which of the reference-type and low-specificity multipliers apply at a site."""

    reference: float | None  # multiplier to apply, None when skipped
    site: float | None
    not_compounded: str | None  # zero-point note when one of them was skipped


def _reference_and_site_discounts(site: SiteResult, rules: TriageRules) -> _Discounts:
    """Reference-type and low-specificity multipliers.

    At a site with no paired site the in-image reference is the only reference possible, and
    compounding it with the low-specificity discount left the strongest fluorescence loss at the
    tongue dorsum or hard palate unable to reach MODERATE. There only the larger discount (the
    smaller multiplier) applies; at paired sites both apply.
    """
    assert site.af is not None
    ref_mult = rules.af.reference_multipliers[site.af.reference.type]
    if site.site not in rules.site.af_low_specificity_sites:
        return _Discounts(ref_mult, None, None)
    site_mult = rules.site.af_low_specificity_multiplier
    if site.paired_site is not None:
        return _Discounts(ref_mult, site_mult, None)
    kept = min(ref_mult, site_mult)
    note = (
        f"x{kept:.2f} applied instead of x{ref_mult:.2f} times x{site_mult:.2f}"
        if ref_mult != 1.0 and site_mult != 1.0
        else None
    )
    if ref_mult <= site_mult:
        return _Discounts(ref_mult, None, note)
    return _Discounts(None, site_mult, note)


def af_factors(region: FVLRegion, site: SiteResult, rules: TriageRules) -> AFPoints:
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
    raw = led.total
    discounts = _reference_and_site_discounts(site, rules)
    ref_type = site.af.reference.type
    if discounts.reference is not None:
        led.scale(
            discounts.reference, f"af.reference.{ref_type.value}", _REFERENCE_LABEL[ref_type], AF
        )
    diascopy, not_applied = diascopy_for_region(region, site)
    if not_applied is not None:
        led.note("af.diascopy.not_applied", not_applied, AF)
    led.scale(
        ar.diascopy_multipliers[diascopy],
        f"af.diascopy.{diascopy.value}",
        _DIASCOPY_LABEL[diascopy],
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
    if discounts.site is not None:
        led.scale(
            discounts.site,
            "af.low_specificity_site",
            f"Dark areas at {site_phrase(site.site)} are often due to inflammation, coating or "
            "keratin, so they count less",
            AF,
        )
    if discounts.not_compounded is not None:
        led.note(
            "af.discount_not_compounded",
            f"{site_label(site.site)} has no opposite-side partner, so only the larger of the "
            "reference and site discounts was applied",
            AF,
            discounts.not_compounded,
        )
    return AFPoints(led.factors, raw, diascopy)


def cleanable_confounders(region: FVLRegion, rules: TriageRules) -> tuple[ConfounderType, ...]:
    """Plaque or glare that discounted this region: cleaning and retaking may remove them."""
    return tuple(
        dict.fromkeys(
            c.type
            for c in region.confounders
            if c.type in CLEANABLE_CONFOUNDERS
            and c.overlap_fraction > 0
            and rules.af.confounder_discount[c.type] > 0
        )
    )


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
                f"Probe reading {r.reading_id} had no reference reading from the opposite side "
                "or a comparison site, so it was not scored",
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
            # The delta is a relative change (ratio_lesion / ratio_reference - 1).
            f"Lower 555/630 nm reflectance ratio than the reference reading ({value:+.1%})",
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
    elif not site_analysed(draft.site_result):
        led.note(
            "qc.device_analysis_unavailable",
            "Device analysis not available at this site (images could not be analysed); "
            "the health worker's observation still counts",
            FactorSource.QC,
        )
    elif not segmentation_reliable(draft.site_result):
        led.note(
            "qc.segmentation_unreliable",
            "The device could not reliably outline the mucosa at this site, so part of it may "
            "not have been checked; what was measured still counts",
            FactorSource.QC,
        )
    elif draft.region is None:
        led.note(
            "af.no_fvl_at_site",
            "No dark area under violet light at this site; this does not lower the concern "
            "from the other findings",
            AF,
        )


def red_area_override(draft: LesionDraft, rules: TriageRules) -> Override | None:
    """Floor for FVL overlapping a deep-red area (possible red patch); never a discount."""
    raf = rules.af.red_area_floor
    if raf is None or draft.region is None:
        return None
    for c in draft.region.confounders:
        if c.type == raf.confounder and c.overlap_fraction >= raf.min_overlap:
            return Override(code="af.red_area_floor", label=raf.label, floor=raf.floor)
    return None


def strong_fvl_override(
    draft: LesionDraft, raw_points: float, rules: TriageRules
) -> Override | None:
    """Floor for fluorescence loss strong enough before discounts (see ``StrongFVLFloor``)."""
    sf = rules.af.strong_fvl_floor
    if sf is None or draft.region is None:
        return None
    strong_points = raw_points >= sf.raw_points_min - 1e-9
    deep = draft.region.mean_relative_af <= sf.mean_relative_af_max + 1e-9
    if strong_points or deep:
        return Override(code="af.strong_fvl_floor", label=sf.label, floor=sf.floor)
    return None


def score_lesion(
    draft: LesionDraft, lesion_id: str, rules: TriageRules, risk_factors: RiskFactors
) -> ScoredLesion:
    """Score one lesion and decide its level."""
    readable = draft.site_result is not None and site_readable(draft.site_result)
    assessed = readable and not probe_only_unchecked(draft)
    led = Ledger()
    raw_af = 0.0
    diascopy = Diascopy.NOT_DONE
    cleanable: tuple[ConfounderType, ...] = ()
    if draft.region is not None and draft.site_result is not None:
        af_points = af_factors(draft.region, draft.site_result, rules)
        led.extend(af_points.factors)
        raw_af, diascopy = af_points.raw_points, af_points.diascopy_applied
        cleanable = cleanable_confounders(draft.region, rules)
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
        ctx = ClinicalContext(
            risk_factors,
            frozenset(draft.visual.appearance),
            any_lesion=True,
            lesion_seen=visual_seen(draft.visual),
        )
        floors = matching_overrides(rules.overrides, ctx, visual_only=True)
    for extra in (red_area_override(draft, rules), strong_fvl_override(draft, raw_af, rules)):
        if extra is not None:
            floors.append(extra)
    level = score_level
    floor = highest_floor(floors)
    if floor is not None and RISK_ORDER[floor] > RISK_ORDER[level]:
        level = floor
    for o in floors:
        led.factors.append(override_factor(o, "this lesion's"))
    if (
        level == RiskLevel.LOW
        and not assessed
        and rules.inconclusive.low_requires_all_examined_sites_analysed
    ):
        level = RiskLevel.INCONCLUSIVE
    return ScoredLesion(
        draft,
        lesion_id,
        led.factors,
        score,
        score_level,
        level,
        floors,
        readable,
        assessed=assessed,
        af_raw_points=raw_af,
        cleanable=cleanable,
        diascopy_applied=diascopy,
    )
