"""Explanations, lesion summaries, recommended actions and limitations.

The explanation names the top reasons with their numbers, in at most three sentences. A LOW
result always says it does not rule out disease and that re-screening matters, and names every
symptom or oral history the patient reported; an INCONCLUSIVE result always says why and what to
do now. When the overall level is above every lesion's own level (raised by a clinical red flag
or by habits and history), the explanation says so plainly. Dynamic action steps are appended
after the mandated YAML steps and never contradict them (for HIGH, a retake must not delay the
referral).

Wording rules: the comparison tissue is "the opposite side" or "the reference site", never
"healthy" (the device does not check that it is); a placeholder classifier's "not suspicious"
label is never repeated as reassurance.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from mukha_pipeline.fusion.lesions import (
    probe_only_unchecked,
    segmentation_reliable,
    site_analysed,
    site_readable,
    site_software_error,
    visual_seen,
)
from mukha_pipeline.fusion.overrides import highest_floor
from mukha_pipeline.fusion.patient import PatientDecision
from mukha_pipeline.fusion.rules import Override, TriageRules
from mukha_pipeline.fusion.scoring import ScoredLesion, best_appearance
from mukha_pipeline.fusion.text import (
    APPEARANCE_NOUN,
    WL_NOUN,
    capitalize,
    fvl_strength,
    habit_names,
    join_words,
    reference_phrase,
    reported_history,
    reported_symptoms,
    site_list,
    site_phrase,
)
from mukha_pipeline.schema.models import (
    RISK_ORDER,
    ConfounderType,
    Diascopy,
    FactorSource,
    Limitation,
    RecommendedAction,
    RiskFactors,
    RiskLevel,
    SiteResult,
    SiteRole,
    SpectralReadingResult,
)
from mukha_pipeline.sites import OralSite

OVERRIDE_PHRASE: dict[str, str] = {
    "symptom.non_healing_ulcer": "a non-healing ulcer reported for more than 2 weeks",
    "symptom.neck_lump": "a lump in the neck",
    "symptom.lump_with_lesion": "a lump reported by the patient",
    "history.prior_cancer_with_lesion": "previous oral cancer",
    "symptom.trismus": "difficulty opening the mouth (possible oral submucous fibrosis)",
    "symptom.red_patch_reported": "a red patch reported by the patient",
    "symptom.white_patch_reported": "a white patch reported by the patient",
    "symptom.bleeding": "unexplained bleeding in the mouth",
    "symptom.lump": "a lump in the mouth reported by the patient",
    "symptom.burning_sensation": (
        "a burning sensation in the mouth (possible early submucous fibrosis)"
    ),
    "symptom.pain": "persistent pain in the mouth reported by the patient",
    "history.prior_oral_cancer": "previous oral cancer (needs regular specialist follow-up)",
    "af.strong_fvl_floor": "strong or extensive loss of fluorescence at another finding",
    "af.high_risk_site_fvl_floor": "loss of fluorescence at a high-risk site",
}

# Lesion floors that the lesion's own evidence already describes ("strong loss of fluorescence
# at ..."): not repeated in the explanation for the lesion that carries them.
SELF_DESCRIBING_FLOORS: frozenset[str] = frozenset(
    {"af.strong_fvl_floor", "af.high_risk_site_fvl_floor"}
)

# Extra steps for symptoms and history that set a floor without a device finding: the
# template's "re-examine the same site" does not say what to look at.
SYMPTOM_STEP: dict[str, str] = {
    "lump": "The patient reports a lump in the mouth: look at and feel the whole mouth and the "
    "neck, including sites that were not photographed.",
    "pain": "The patient reports pain in the mouth: persistent pain needs clinical examination "
    "of the whole mouth, including sites that were not photographed.",
    "burning_sensation": "The patient reports a burning sensation (possible early submucous "
    "fibrosis): check how wide the mouth opens and look for pale, stiff mucosa or bands.",
}
HISTORY_STEP: dict[str, str] = {
    "prior_oral_cancer": "Previous oral cancer: the patient needs regular specialist follow-up; "
    "routine re-screening alone is not enough.",
}

CLEAN_NOUN: dict[ConfounderType, str] = {
    ConfounderType.PORPHYRIN_RED: "plaque fluorescence",
    ConfounderType.GLARE: "glare",
}

LIMITATION_TEXT: dict[str, str] = {
    "sites_failed_qc": "{sites} could not be analysed (images failed quality checks or showed "
    "too little usable tissue).",
    "analysis_error": "{sites}: software error while analysing — not a problem with the photos. "
    "Examine clinically and report the problem.",
    "device_analysis_unavailable": "No examined site could be analysed by the device; this "
    "result rests on the health worker's observation and the reported symptoms.",
    "segmentation_unreliable": "At {sites}, the device could not reliably outline the mucosa, "
    "so part of the tissue may not have been checked for loss of fluorescence; a low result "
    "cannot be given until the photos are retaken.",
    "reference_in_image": "At {sites}, no opposite-side or similar site was available for "
    "comparison, so fluorescence was compared with surrounding tissue in the same photo, which "
    "is less reliable.",
    "reference_site_fvl": "The reference site also showed loss of fluorescence; comparisons may "
    "be less reliable.",
    "wl_model_unavailable": "The white-light photo model is not available yet, so it did not "
    "contribute to this result.",
    "spectral_no_reference": "A probe reading had no reference reading from the opposite side "
    "or a comparison site, so it was not scored.",
    "spectral_qc_failed": "A probe reading failed quality checks and was not scored.",
    "triage_rules_unvalidated": "The triage rules are a baseline heuristic that has not been "
    "clinically validated; a clinician must review every result.",
}


@dataclass(frozen=True)
class Evidence:
    points: float
    short: str  # for lesion summaries (site named separately)
    long: str  # for the patient explanation, without the site
    long_at_site: str  # for the patient explanation, naming the site


# --------------------------------------------------------------------------------------------
# Evidence phrases
# --------------------------------------------------------------------------------------------


def _points_by_source(s: ScoredLesion, source: FactorSource) -> float:
    return sum(f.points for f in s.factors if f.source == source)


def lesion_evidence(s: ScoredLesion, rules: TriageRules) -> list[Evidence]:
    """Plain-language evidence for one lesion, strongest first."""
    d = s.draft
    where = site_phrase(d.site)
    out: list[Evidence] = []
    if d.region is not None and d.site_result is not None and d.site_result.af is not None:
        r = d.region
        strength = fvl_strength(r.mean_relative_af)
        ref = reference_phrase(d.site_result.af.reference)
        if r.area_mm2 is not None:
            size = f", about {r.area_mm2:.0f} mm²"
        else:
            size = f", {r.area_fraction_of_mucosa:.0%} of the visible mucosa"
        resolved = ""
        if s.diascopy_applied == Diascopy.FVL_RESOLVES:
            resolved = " that returned under glass-slide pressure"
        loss = f"{strength} loss of fluorescence"
        rel = f"relative AF {r.mean_relative_af:.2f}"
        out.append(
            Evidence(
                _points_by_source(s, FactorSource.AUTOFLUORESCENCE),
                f"{loss}{resolved} ({rel}{size})",
                f"{loss}{resolved} ({rel} {ref})",
                f"{loss} at {where}{resolved} ({rel} {ref})",
            )
        )
    if d.visual is not None:
        noun = APPEARANCE_NOUN[best_appearance(d.visual, rules)]
        text = f"{noun} seen by the health worker"
        out.append(
            Evidence(
                _points_by_source(s, FactorSource.CLINICAL_VISUAL),
                text,
                text,
                f"{noun} seen by the health worker at {where}",
            )
        )
    if d.wl is not None:
        noun = WL_NOUN[d.wl.label]
        out.append(
            Evidence(
                _points_by_source(s, FactorSource.WL_MODEL),
                f"photo model suggests {noun}",
                f"the photo model (unvalidated) suggesting {noun}",
                f"the photo model (unvalidated) suggesting {noun} at {where}",
            )
        )
    if d.spectral:
        pts = _points_by_source(s, FactorSource.SPECTRAL)
        if pts > 0:
            short = "a deeper blood-absorption dip on the probe than on the reference reading"
            long = (
                "a deeper blood-absorption dip on the probe reading than on the reference reading"
            )
            at = f"a deeper blood-absorption dip on the probe reading at {where}"
        else:
            rest = _unscored_probe_phrase(d.spectral)
            short = f"probe reading {rest}"
            long = f"a probe reading {rest}"
            at = f"a probe reading at {where} {rest}"
        out.append(Evidence(pts, short, long, at))
    return sorted(out, key=lambda e: -e.points)


def _unscored_probe_phrase(readings: Sequence[SpectralReadingResult]) -> str:
    """Neutral words for probe readings that added no points.

    The placeholder classifier is unvalidated, so its "not suspicious" label is never repeated:
    it would read as reassurance about a spot the health worker chose to measure.
    """
    passed = [r for r in readings if r.qc.passed]
    if not passed:
        return "that failed quality checks"
    labels = {r.classifier.label for r in passed if r.classifier is not None}
    for mark in ("suspicious", "indeterminate"):
        if mark in labels:
            return f"marked {mark} by the placeholder classifier"
    if all(r.delta_vs_reference is None for r in passed):
        return "that could not be scored (no reference reading)"
    return "that added no points"


def lesion_summary(s: ScoredLesion, rules: TriageRules) -> str:
    """One plain sentence describing the lesion and its level."""
    evidence = [e.short for e in lesion_evidence(s, rules)] or ["a finding with no scored features"]
    where = site_phrase(s.draft.site)
    if s.level == RiskLevel.INCONCLUSIVE:
        if s.draft.site_result is None:
            return (
                f"Finding at {where} with no analysed photos ({join_words(evidence)}); take "
                "photos or examine clinically."
            )
        if s.readable and probe_only_unchecked(s.draft):
            return (
                f"Finding at {where} that the probe could not measure ({join_words(evidence)}); "
                "retake the probe reading or examine clinically."
            )
        return (
            f"Finding at {where} that the device could not fully check "
            f"({join_words(evidence)}); retake or examine clinically."
        )
    return f"{s.level.value} risk at {where}: {join_words(evidence)}."


# --------------------------------------------------------------------------------------------
# Patient explanation
# --------------------------------------------------------------------------------------------


def _override_phrase(
    o: Override, top: ScoredLesion | None, sites: Sequence[SiteResult]
) -> str | None:
    """Phrase for a floor, or None when the top lesion's evidence already says it."""
    if o.code in SELF_DESCRIBING_FLOORS and top is not None and o in top.floors:
        return None
    if not o.lesion_scoped:
        return OVERRIDE_PHRASE.get(o.code, o.label[:1].lower() + o.label[1:])
    top_visual = top.draft.visual if top is not None else None
    if top_visual is not None and (
        o.visual_lesion_seen or (o.visual_appearance or set()) & set(top_visual.appearance)
    ):
        return None
    for site in sites:
        where = site_phrase(site.site)
        matched = [a for a in site.visual_flag.appearance if a in (o.visual_appearance or ())]
        if matched:
            return f"{APPEARANCE_NOUN[matched[0]]} seen by the health worker at {where}"
        if o.visual_lesion_seen and visual_seen(site.visual_flag):
            return f"an abnormality seen by the health worker at {where}"
    return o.label[:1].lower() + o.label[1:]  # pragma: no cover - floors only hold with a site


def _reasons(
    decision: PatientDecision,
    top: ScoredLesion | None,
    floors: list[Override],
    sites: Sequence[SiteResult],
    rules: TriageRules,
) -> list[str]:
    evidence = lesion_evidence(top, rules) if top is not None else []
    lesion_phrases = [
        e.long_at_site if i == 0 else e.long for i, e in enumerate(evidence) if e.points > 0
    ] or [e.long_at_site for e in evidence[:1]]
    level = decision.level
    decisive = [o for o in floors if o.floor == level]
    others = [o for o in floors if o.floor != level]
    floor_phrases = [_override_phrase(o, top, sites) for o in decisive]
    other_phrases = [_override_phrase(o, top, sites) for o in others]
    floor = highest_floor(floors)
    level_from_floor = floor is not None and RISK_ORDER[floor] > RISK_ORDER[decision.score_level]
    ordered = (
        floor_phrases + lesion_phrases + other_phrases
        if level_from_floor
        else lesion_phrases + floor_phrases + other_phrases
    )
    return list(dict.fromkeys(p for p in ordered if p))[:3]


def _months(days: int | None) -> str:
    if days is None or days <= 0:
        return "regular re-screening matters"
    months = max(1, round(days / 30.4))
    return f"re-screening in {months} month{'s' if months != 1 else ''} matters"


def _rank_level(s: ScoredLesion) -> RiskLevel:
    """A lesion's own level for comparisons: INCONCLUSIVE lesions scored LOW."""
    return RiskLevel.LOW if s.level == RiskLevel.INCONCLUSIVE else s.level


def outranks(overall: RiskLevel, lesion_level: RiskLevel) -> bool:
    """True when a MODERATE/HIGH overall level is above a lesion's own level."""
    if overall not in (RiskLevel.MODERATE, RiskLevel.HIGH):
        return False
    own = RiskLevel.LOW if lesion_level == RiskLevel.INCONCLUSIVE else lesion_level
    return RISK_ORDER[overall] > RISK_ORDER[own]


def _raised_sentence(
    decision: PatientDecision,
    top: ScoredLesion | None,
    floors: list[Override],
    sites: Sequence[SiteResult],
    rf_points: float,
    rf_names: Sequence[str],
) -> tuple[str | None, bool]:
    """Say plainly when the overall level is above the strongest finding's own level.

    Returns (sentence, raised_by_habits). Lesion levels are never changed for this: the finding
    keeps its own level and the explanation says where the overall level came from.
    """
    level = decision.level
    if top is None or not outranks(level, top.level):
        return None, False
    own = _rank_level(top)
    where = site_phrase(top.draft.site)
    tail = f", even though the finding at {where} scored {own.value} on its own."
    decisive = [o for o in floors if o.floor == level and RISK_ORDER[o.floor] > RISK_ORDER[own]]
    phrases = [p for p in (_override_phrase(o, None, sites) for o in decisive) if p]
    if phrases:
        return f"Overall {level.value} because of {join_words(phrases)}{tail}", False
    if rf_points > 0:
        return (
            f"Overall {level.value} because habits and history ({join_words(rf_names)}) added "
            f"{rf_points:.1f} points{tail}",
            True,
        )
    return f"Overall {level.value}{tail}", False  # pragma: no cover - level needs a cause


def _concerns(rf: RiskFactors) -> list[str]:
    """Reported symptoms and oral history, as phrases."""
    return [noun for _, noun in reported_symptoms(rf)] + reported_history(rf)


def _low_explanation(
    *,
    examined: Sequence[SiteResult],
    lesions: list[ScoredLesion],
    top: ScoredLesion | None,
    rf: RiskFactors,
    rules: TriageRules,
) -> str:
    sentences: list[str] = []
    if top is None:
        n = len(examined)
        sentences.append(
            f"No high-risk features were found by this screen at the {n} examined "
            f"site{'s' if n != 1 else ''}."
        )
    else:
        phrase = lesion_evidence(top, rules)[0].long_at_site  # every lesion has evidence
        if len(lesions) == 1:
            sentences.append(f"The only finding scored low on this screen: {phrase}.")
        else:
            sentences.append(f"All findings scored low on this screen; the strongest is {phrase}.")
    concerns = _concerns(rf)
    habits = habit_names(rf, current_only=True)
    rescreen = (
        f"This does not rule out disease; {_months(rules.actions[RiskLevel.LOW].timeline_days)}"
    )
    if concerns:
        # A reported symptom is never answered with "come back if symptoms appear".
        sentences.append(
            f"The patient reports {join_words(concerns)}: examine the whole mouth and neck "
            "clinically."
        )
        if habits:
            rescreen += (
                f", and stopping {join_words(habits)} is the most important step to lower "
                "future risk"
            )
        sentences.append(f"{rescreen}.")
        return " ".join(sentences)
    sentences.append(f"{rescreen}, sooner if any symptom appears.")
    if habits:
        sentences.append(
            f"Stopping {join_words(habits)} is the most important step to lower future risk."
        )
    return " ".join(sentences)


def _software_error_sentence(errored: Sequence[OralSite]) -> str:
    verb = "was" if len(errored) == 1 else "were"
    return (
        f"{site_list(errored)} {verb} not analysed because of a software error (not a problem "
        "with the photos): examine clinically and report the problem."
    )


def explanation(
    decision: PatientDecision,
    *,
    lesions: list[ScoredLesion],
    top: ScoredLesion | None,
    floors: list[Override],
    sites: Sequence[SiteResult],
    rf: RiskFactors,
    rf_points: float,
    rf_names: Sequence[str],
    rules: TriageRules,
) -> str:
    """1-3 plain-language sentences explaining the overall level."""
    examined = [s for s in sites if s.role == SiteRole.EXAMINED]
    unreadable = [s.site for s in examined if not site_readable(s)]
    errored = [s.site for s in examined if site_software_error(s)]
    retakeable = [x for x in unreadable if x not in errored]
    sentences: list[str] = []
    level = decision.level

    if level == RiskLevel.INCONCLUSIVE:
        if decision.inconclusive_reason == "no_sites_analysed":
            if examined:
                sentences.append(
                    "The result is inconclusive because none of the examined sites could be "
                    f"analysed ({site_list(unreadable)})."
                )
            else:
                sentences.append(
                    "The result is inconclusive because no examined site had usable images."
                )
        elif decision.inconclusive_reason == "too_few_sites":
            k, need = len(examined) - len(unreadable), rules.inconclusive.min_sites_analysed
            sentences.append(
                f"The result is inconclusive because only {k} examined site"
                f"{'s' if k != 1 else ''} could be analysed and at least {need} are needed."
            )
        elif decision.inconclusive_reason == "sites_unreadable":
            sentences.append(
                f"The result is inconclusive because {len(unreadable)} of {len(examined)} "
                f"examined sites could not be analysed ({site_list(unreadable)})."
            )
        else:
            unassessed = [s.draft.site for s in lesions if s.level == RiskLevel.INCONCLUSIVE]
            sentences.append(
                "The result is inconclusive because the finding at "
                f"{site_phrase(unassessed[0])} could not be checked by the device."
            )
        if decision.inconclusive_reason == "lesion_unassessed":
            sentences.append(
                "A low result needs every finding to be checked by the device, so take the "
                "missing photos or probe readings now."
            )
        elif retakeable:
            sentences.append(
                "A low result needs every site to be readable, so retake the images now."
            )
        if errored:
            sentences.append(_software_error_sentence(errored))
        seen = [s.draft.site for s in lesions if s.draft.visual is not None and not s.readable]
        if seen:
            sentences.append(
                f"The health worker saw a lesion at {site_phrase(seen[0])}: if a retake is not "
                "possible, refer for clinical examination."
            )
        return " ".join(sentences)

    if level == RiskLevel.LOW:
        return _low_explanation(examined=examined, lesions=lesions, top=top, rf=rf, rules=rules)

    reasons = _reasons(decision, top, floors, sites, rules)
    sentences.append(f"{level.value} risk: {join_words(reasons)}.")
    raised, by_habits = _raised_sentence(decision, top, floors, sites, rf_points, rf_names)
    if raised:
        sentences.append(raised)
    if decision.device_analysis_unavailable:
        sentences.append(
            "The device could not analyse the images, so this level rests on the clinical "
            "findings; retake if possible, without delaying the next step."
        )
    elif retakeable:
        sentences.append(f"{site_list(retakeable)} could not be analysed and should be retaken.")
    elif errored:
        sentences.append(_software_error_sentence(errored))
    if rf_points > 0 and not by_habits:
        sentences.append(
            f"Habits and history ({join_words(rf_names)}) added {rf_points:.1f} points."
        )
    others = [
        x for x in lesions if x is not top and x.level in (RiskLevel.MODERATE, RiskLevel.HIGH)
    ]
    if others:
        n = len(others)
        sentences.append(
            f"{n} other suspicious finding{'s are' if n > 1 else ' is'} listed with the sites."
        )
    return " ".join(sentences[:3])


# --------------------------------------------------------------------------------------------
# Actions
# --------------------------------------------------------------------------------------------


def action_from_template(
    level: RiskLevel, rules: TriageRules, extra_steps: Sequence[str] = ()
) -> RecommendedAction:
    t = rules.actions[level]
    steps = list(t.steps)
    for step in extra_steps:
        if step not in steps:
            steps.append(step)
    return RecommendedAction(
        code=t.code,
        summary=t.summary,
        steps=steps,
        timeline_days=t.timeline_days,
        referral_urgency=t.referral_urgency,
    )


def _retake_step(what: str, sites: Sequence[OralSite], urgent: bool, why: str) -> str:
    """'<What> at <sites>: <why>.', or, when a referral is urgent, a step that never delays it."""
    if urgent:
        return (
            f"If the patient is still here, {what[:1].lower()}{what[1:]} {site_list(sites)}; "
            "do not delay the referral for this."
        )
    return f"{what} {site_list(sites)}: {why}."


def _clean_step(lesions: Sequence[ScoredLesion], urgent: bool) -> str | None:
    """Plaque fluorescence or glare discounted a dark area: cleaning and retaking may show
    what is underneath (necrotic or ulcerated surfaces often carry plaque)."""
    sites = list(dict.fromkeys(s.draft.site for s in lesions if s.cleanable))
    if not sites:
        return None
    nouns = list(dict.fromkeys(CLEAN_NOUN[c] for s in lesions for c in s.cleanable))
    return _retake_step(
        "Clean the area (gauze/rinse) and retake the photos of",
        sites,
        urgent,
        f"{join_words(nouns)} overlapped a dark area and may hide what is underneath",
    )


def _probe_retake_step(lesions: Sequence[ScoredLesion], urgent: bool) -> str | None:
    """A lesion probe reading failed quality checks: the probe never measured that spot."""
    sites = list(
        dict.fromkeys(s.draft.site for s in lesions for r in s.draft.spectral if not r.qc.passed)
    )
    if not sites:
        return None
    return _retake_step(
        "Retake the probe reading at", sites, urgent, "the lesion reading failed quality checks"
    )


def lesion_action(s: ScoredLesion, overall: RiskLevel, rules: TriageRules) -> RecommendedAction:
    """The lesion's action, consistent with the overall result.

    The lesion keeps its own level. When the overall level is above it, a LOW lesion does not
    carry the routine re-screening action (it would contradict the overall result): it carries
    the overall action with a summary saying it scored low on its own. Other lesions below the
    overall level point to the overall action.
    """
    urgent = RiskLevel.HIGH in (s.level, overall)
    extra = [step for step in (_clean_step([s], urgent), _probe_retake_step([s], urgent)) if step]
    if outranks(overall, s.level):
        if s.level == RiskLevel.LOW:
            action = action_from_template(overall, rules, extra)
            summary = (
                f"This finding scored LOW on its own; the overall result is {overall.value}. "
                f"{action.summary}"
            )
            return action.model_copy(update={"summary": summary})
        extra.append(
            f"The patient's overall result is {overall.value}: follow the overall recommended "
            "action first."
        )
    return action_from_template(s.level, rules, extra)


def dynamic_steps(
    level: RiskLevel,
    *,
    sites: Sequence[SiteResult],
    lesions: list[ScoredLesion],
    rf: RiskFactors,
) -> list[str]:
    """Patient-specific steps added after the mandated ones."""
    steps: list[str] = []
    urgent = level == RiskLevel.HIGH
    # A software error is not fixed by retaking the photos: examine clinically, report it.
    errored = list(dict.fromkeys(s.site for s in sites if site_software_error(s)))
    unreadable = list(
        dict.fromkeys(s.site for s in sites if not site_readable(s) and s.site not in errored)
    )
    if errored:
        steps.append(
            f"{site_list(errored)}: not analysed because of a software error (not a problem with "
            "the photos) — examine clinically and report the problem."
        )
    if unreadable:
        if level == RiskLevel.HIGH:
            steps.append(
                f"If the patient is still here, retake {site_list(unreadable)}; do not delay "
                "the referral for this."
            )
        elif level == RiskLevel.INCONCLUSIVE:
            steps.append(f"Sites to retake: {site_list(unreadable)}.")
        else:
            steps.append(
                f"Retake the captures that could not be analysed: {site_list(unreadable)}."
            )
    for step in (_clean_step(lesions, urgent), _probe_retake_step(lesions, urgent)):
        if step:
            steps.append(step)
    unimaged: list[OralSite] = list(
        dict.fromkeys(s.draft.site for s in lesions if s.draft.site_result is None)
    )
    if unimaged:
        steps.append(
            f"Take white-light and violet-light photos of {site_list(unimaged)}, where a lesion "
            "probe reading was taken."
        )
    seen = list(
        dict.fromkeys(
            s.draft.site
            for s in lesions
            if s.draft.visual is not None and not s.readable and s.draft.site_result is not None
        )
    )
    if seen and level == RiskLevel.INCONCLUSIVE:
        steps.append(
            f"A lesion was seen at {site_list(seen)}: if a retake is not possible today, refer "
            "for clinical examination."
        )
    concerns = _concerns(rf)
    if level == RiskLevel.LOW and concerns:
        steps.append(
            f"The patient reports {join_words(concerns)}: examine the whole mouth and neck "
            "clinically, including sites that were not photographed."
        )
    elif level != RiskLevel.LOW:
        steps.extend(SYMPTOM_STEP[k] for k, _ in reported_symptoms(rf) if k in SYMPTOM_STEP)
        steps.extend(step for key, step in HISTORY_STEP.items() if getattr(rf, key))
    prefix = "Alongside the referral, " if level == RiskLevel.HIGH else ""
    if rf.irritants.sharp_tooth:
        steps.append(
            capitalize(
                f"{prefix}smooth or treat the sharp tooth (dental referral) so it stops "
                "injuring the mucosa."
            )
        )
    if rf.irritants.ill_fitting_denture:
        steps.append(
            capitalize(
                f"{prefix}get the ill-fitting denture adjusted or replaced, and leave it out "
                "until then."
            )
        )
    current = habit_names(rf, current_only=True)
    former = [h for h in habit_names(rf) if h not in current and h.startswith("former ")]
    if current:
        steps.append(
            f"Habits reported ({join_words(current)}): give personal cessation advice and offer "
            "referral to a tobacco cessation service."
        )
    elif former:
        quit_names = [h.removeprefix("former ") for h in former]
        steps.append(f"Encourage the patient to stay off {join_words(quit_names)}.")
    if rf.symptoms.difficulty_opening_mouth:
        steps.append(
            "Record how wide the mouth opens (in finger widths) so change can be tracked; "
            "reduced opening needs clinical examination for oral submucous fibrosis."
        )
    return steps


# --------------------------------------------------------------------------------------------
# Limitations
# --------------------------------------------------------------------------------------------


def _reference_sites_with_fvl(sites: Sequence[SiteResult]) -> list[OralSite]:
    """Reference tissue that itself showed FVL: a reference-only site with FVL regions, or a
    site with FVL whose own comparison site also has FVL (bilateral loss, common in long-term
    gutka and areca users). Either way relative AF may understate the loss."""
    with_fvl = {s.site for s in sites if site_analysed(s) and s.af is not None and s.af.regions}
    out = []
    for s in sites:
        if s.site not in with_fvl or s.af is None:
            continue
        if s.role == SiteRole.REFERENCE_ONLY or s.af.reference.reference_site in with_fvl:
            out.append(s.site)
    return list(dict.fromkeys(out))


def build_limitations(
    *,
    sites: Sequence[SiteResult],
    lesions: list[ScoredLesion],
    spectral: Sequence[SpectralReadingResult],
    decision: PatientDecision,
    rules: TriageRules,
) -> list[Limitation]:
    out: list[Limitation] = []

    def add(code: str, **fmt: str) -> None:
        out.append(Limitation(code=code, message=LIMITATION_TEXT[code].format(**fmt)))

    errored = list(dict.fromkeys(s.site for s in sites if site_software_error(s)))
    if errored:
        add("analysis_error", sites=site_list(errored))
    failed = list(
        dict.fromkeys(s.site for s in sites if not site_analysed(s) and s.site not in errored)
    )
    if failed:
        add("sites_failed_qc", sites=site_list(failed))
    unreliable = list(
        dict.fromkeys(s.site for s in sites if site_analysed(s) and not segmentation_reliable(s))
    )
    if unreliable:
        add("segmentation_unreliable", sites=site_list(unreliable))
    if decision.device_analysis_unavailable:
        add("device_analysis_unavailable")
    in_image = list(
        dict.fromkeys(
            s.draft.site
            for s in lesions
            if s.draft.region is not None
            and s.draft.site_result is not None
            and s.draft.site_result.af is not None
            and s.draft.site_result.af.reference.type.value == "in_image_auto"
        )
    )
    if in_image:
        add("reference_in_image", sites=site_list(in_image))
    if _reference_sites_with_fvl(sites):
        add("reference_site_fvl")
    if any(s.wl is not None and s.wl.status == "model_unavailable" for s in sites):
        add("wl_model_unavailable")
    lesion_readings = [r for r in spectral if r.role == "lesion"]
    if any(r.qc.passed and r.delta_vs_reference is None for r in lesion_readings):
        add("spectral_no_reference")
    if any(not r.qc.passed for r in lesion_readings):
        add("spectral_qc_failed")
    if "unvalidated" in rules.status.lower():
        add("triage_rules_unvalidated")
    return out
