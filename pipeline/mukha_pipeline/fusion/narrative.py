"""Explanations, lesion summaries, recommended actions and limitations.

The explanation names the top reasons with their numbers, in at most three sentences. A LOW
result always says it does not rule out disease and that re-screening matters; an INCONCLUSIVE
result always says why and what to do now. Dynamic action steps are appended after the mandated
YAML steps and never contradict them (for HIGH, a retake must not delay the referral).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from mukha_pipeline.fusion.lesions import site_readable
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
    site_list,
    site_phrase,
)
from mukha_pipeline.schema.models import (
    RISK_ORDER,
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
}

LIMITATION_TEXT: dict[str, str] = {
    "sites_failed_qc": "{sites} could not be analysed (images failed quality checks or showed "
    "too little usable tissue).",
    "device_analysis_unavailable": "No examined site could be analysed by the device; this "
    "result rests on the health worker's observation and the reported symptoms.",
    "reference_in_image": "At {sites}, no matching healthy site was available, so fluorescence "
    "was compared with surrounding tissue in the same photo, which is less reliable.",
    "reference_site_fvl": "The reference site also showed loss of fluorescence; comparisons may "
    "be less reliable.",
    "wl_model_unavailable": "The white-light photo model is not available yet, so it did not "
    "contribute to this result.",
    "spectral_no_reference": "A probe reading had no reference reading from the healthy side, "
    "so it was not scored.",
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
        if d.site_result.diascopy == Diascopy.FVL_RESOLVES:
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
            short = "a deeper blood-absorption dip on the probe than on the healthy side"
            long = "a deeper blood-absorption dip on the probe reading than on the healthy side"
            at = f"a deeper blood-absorption dip on the probe reading at {where}"
        else:
            labels = [r.classifier.label for r in d.spectral if r.classifier and r.classifier.label]
            mark = labels[0].replace("_", " ") if labels else "not scored"
            short = f"probe reading marked {mark}"
            long = f"a probe reading marked {mark}"
            at = f"a probe reading marked {mark} at {where}"
        out.append(Evidence(pts, short, long, at))
    return sorted(out, key=lambda e: -e.points)


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
        return (
            f"Finding at {where} that the device could not check ({join_words(evidence)}); "
            "retake or examine clinically."
        )
    return f"{s.level.value} risk at {where}: {join_words(evidence)}."


# --------------------------------------------------------------------------------------------
# Patient explanation
# --------------------------------------------------------------------------------------------


def _override_phrase(
    o: Override, top: ScoredLesion | None, sites: Sequence[SiteResult]
) -> str | None:
    """Phrase for a floor, or None when the top lesion's visual evidence already says it."""
    if o.visual_appearance is None:
        return OVERRIDE_PHRASE.get(o.code, o.label[:1].lower() + o.label[1:])
    top_visual = top.draft.visual if top is not None else None
    if top_visual is not None and o.visual_appearance & set(top_visual.appearance):
        return None
    for site in sites:
        matched = [a for a in site.visual_flag.appearance if a in o.visual_appearance]
        if matched:
            where = site_phrase(site.site)
            return f"{APPEARANCE_NOUN[matched[0]]} seen by the health worker at {where}"
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
    habits = habit_names(rf, current_only=True)
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
        sentences.append("A low result needs every site to be readable, so retake the images now.")
        seen = [s.draft.site for s in lesions if s.draft.visual is not None and not s.readable]
        if seen:
            sentences.append(
                f"The health worker saw a lesion at {site_phrase(seen[0])}: if a retake is not "
                "possible, refer for clinical examination."
            )
        return " ".join(sentences)

    if level == RiskLevel.LOW:
        if top is None:
            n = len(examined)
            sentences.append(
                f"No high-risk features were found by this screen at the {n} examined "
                f"site{'s' if n != 1 else ''}."
            )
        else:
            phrase = lesion_evidence(top, rules)[0].long_at_site  # every lesion has evidence
            sentences.append(
                f"Only a low-scoring finding was seen: {phrase}, below the level that needs "
                "re-examination."
            )
        sentences.append(
            "This does not rule out disease; "
            f"{_months(rules.actions[RiskLevel.LOW].timeline_days)}, sooner if any symptom "
            "appears."
        )
        if habits:
            sentences.append(
                f"Stopping {join_words(habits)} is the most important step to lower future risk."
            )
        return " ".join(sentences)

    reasons = _reasons(decision, top, floors, sites, rules)
    sentences.append(f"{level.value} risk: {join_words(reasons)}.")
    if decision.device_analysis_unavailable:
        sentences.append(
            "The device could not analyse the images, so this level rests on the clinical "
            "findings; retake if possible, without delaying the next step."
        )
    elif unreadable:
        sentences.append(f"{site_list(unreadable)} could not be analysed and should be retaken.")
    if rf_points > 0:
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


def dynamic_steps(
    level: RiskLevel,
    *,
    sites: Sequence[SiteResult],
    lesions: list[ScoredLesion],
    rf: RiskFactors,
) -> list[str]:
    """Patient-specific steps added after the mandated ones."""
    steps: list[str] = []
    unreadable = list(dict.fromkeys(s.site for s in sites if not site_readable(s)))
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
    unimaged: list[OralSite] = list(
        dict.fromkeys(s.draft.site for s in lesions if s.draft.site_result is None)
    )
    if unimaged:
        steps.append(
            f"Take white-light and violet-light photos of {site_list(unimaged)}, where the probe "
            "reading was flagged."
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
    with_fvl = {s.site for s in sites if site_readable(s) and s.af is not None and s.af.regions}
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

    unreadable = list(dict.fromkeys(s.site for s in sites if not site_readable(s)))
    if unreadable:
        add("sites_failed_qc", sites=site_list(unreadable))
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
