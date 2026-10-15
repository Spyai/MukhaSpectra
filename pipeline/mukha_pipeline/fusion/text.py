"""Plain-language building blocks for factor labels, summaries and explanations.

Written for a community health worker first and a clinician second: short words, numbers with
units, no jargon without a gloss. Nothing here may say "diagnosis", "malignant" or "cancer
detected"; tests run every produced string through the positioning guard.
"""

from __future__ import annotations

from collections.abc import Sequence

from mukha_pipeline.fusion.habits import current_use
from mukha_pipeline.schema.models import (
    AFReference,
    ConfounderType,
    ReferenceType,
    RiskFactors,
    VisualAppearance,
    WLLabel,
)
from mukha_pipeline.sites import OralSite, site_label

APPEARANCE_NOUN: dict[VisualAppearance, str] = {
    VisualAppearance.WHITE_PATCH: "a white patch",
    VisualAppearance.RED_PATCH: "a red patch",
    VisualAppearance.MIXED_RED_WHITE: "a mixed red-white patch",
    VisualAppearance.ULCER: "an ulcer",
    VisualAppearance.GROWTH: "a growth",
    VisualAppearance.OTHER: "a lesion",
}

WL_NOUN: dict[WLLabel, str] = {
    WLLabel.NORMAL: "normal-looking mucosa",
    WLLabel.WHITE_PATCH: "a white patch",
    WLLabel.RED_PATCH: "a red patch",
    WLLabel.ULCER_MASS: "an ulcer or mass",
    WLLabel.OTHER: "another visible change",
}

CONFOUNDER_NOUN: dict[ConfounderType, str] = {
    ConfounderType.PORPHYRIN_RED: "red plaque fluorescence (porphyrin)",
    ConfounderType.PIGMENTATION: "pigmentation",
    ConfounderType.BLOOD: "blood",
    ConfounderType.GLARE: "glare",
}

# Every Symptoms field, phrased to follow "the patient reports ...".
SYMPTOM_NOUN: dict[str, str] = {
    "non_healing_ulcer_over_2_weeks": "a non-healing ulcer for more than 2 weeks",
    "white_patch": "a white patch",
    "red_patch": "a red patch",
    "lump": "a lump in the mouth",
    "pain": "pain in the mouth",
    "difficulty_opening_mouth": "difficulty opening the mouth",
    "bleeding": "unexplained bleeding in the mouth",
    "burning_sensation": "a burning sensation in the mouth",
    "neck_lump": "a lump in the neck",
}

HISTORY_NOUN: dict[str, str] = {
    "prior_oral_lesion": "a previous oral lesion",
    "prior_oral_cancer": "previous oral cancer",
}

SMOKELESS_NOUN: dict[str, str] = {
    "gutka": "gutka",
    "khaini": "khaini",
    "zarda": "zarda",
    "paan_with_tobacco": "paan with tobacco",
    "mawa": "mawa",
    "mishri": "mishri",
    "snuff": "snuff",
    "other": "chewing tobacco",
}

SMOKING_NOUN: dict[str, str] = {
    "bidi": "bidi",
    "cigarette": "cigarettes",
    "hookah": "hookah",
    "chutta_reverse": "reverse chutta smoking",
    "other": "smoking",
}


def capitalize(text: str) -> str:
    return text[:1].upper() + text[1:]


def site_phrase(site: OralSite | str) -> str:
    """'the right lateral tongue' for use inside a sentence."""
    label = site_label(site)
    return f"the {label[:1].lower()}{label[1:]}"


def join_words(items: Sequence[str]) -> str:
    """'a', 'a and b', 'a, b and c'."""
    items = [i for i in items if i]
    if len(items) <= 1:
        return "".join(items)
    return f"{', '.join(items[:-1])} and {items[-1]}"


def site_list(sites: Sequence[OralSite]) -> str:
    return ", ".join(site_label(s) for s in sites)


def reference_phrase(ref: AFReference) -> str:
    """What the relative-AF number was compared against."""
    if (
        ref.type in (ReferenceType.CONTRALATERAL, ReferenceType.ANALOGOUS_SITE)
        and ref.reference_site
    ):
        return f"vs {site_phrase(ref.reference_site)}"
    if ref.type == ReferenceType.IN_IMAGE_AUTO:
        return "vs surrounding tissue in the same photo"
    return "with no reference tissue"


def fvl_strength(mean_relative_af: float) -> str:
    """Words for how dark a region is. Bands are descriptive only, they carry no points."""
    if mean_relative_af <= 0.5:
        return "strong"
    if mean_relative_af <= 0.65:
        return "clear"
    return "mild"


def reported_symptoms(rf: RiskFactors) -> list[tuple[str, str]]:
    """(field, phrase) for every symptom the patient reports, in questionnaire order."""
    return [(k, SYMPTOM_NOUN.get(k, k.replace("_", " "))) for k, v in rf.symptoms if v is True]


def reported_history(rf: RiskFactors) -> list[str]:
    """Phrases for the oral history the patient reports."""
    return [noun for key, noun in HISTORY_NOUN.items() if getattr(rf, key)]


def habit_names(rf: RiskFactors, *, current_only: bool = False) -> list[str]:
    """Short names of tobacco / areca / alcohol habits, for counselling text.

    Recorded habit details count as current use unless marked former (``fusion.habits``).
    """
    names: list[str] = []
    st = rf.smokeless_tobacco
    st_now = current_use(st)
    if st_now or (st.former and not current_only):
        kinds = [SMOKELESS_NOUN[t] for t in st.types] or ["chewing tobacco"]
        names.extend(kinds if st_now else [f"former {k}" for k in kinds])
    sm = rf.smoking
    sm_now = current_use(sm)
    if sm_now or (sm.former and not current_only):
        kinds = [SMOKING_NOUN[t] for t in sm.types] or ["smoking"]
        names.extend(kinds if sm_now else [f"former {k}" for k in kinds])
    an = rf.areca_nut
    an_now = current_use(an)
    if an_now or (an.former and not current_only):
        names.append("areca nut" if an_now else "former areca nut")
    if rf.alcohol.frequency == "daily":
        names.append("daily alcohol")
    return list(dict.fromkeys(names))
