"""Plain-language building blocks for factor labels, summaries and explanations.

Written for a community health worker first and a clinician second: short words, numbers with
units, no jargon without a gloss. Nothing here may say "diagnosis", "malignant" or "cancer
detected"; tests run every produced string through the positioning guard.
"""

from __future__ import annotations

from collections.abc import Sequence

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


def habit_names(rf: RiskFactors, *, current_only: bool = False) -> list[str]:
    """Short names of tobacco / areca / alcohol habits, for counselling text."""
    names: list[str] = []
    st = rf.smokeless_tobacco
    if st.uses or (st.former and not current_only):
        kinds = [SMOKELESS_NOUN[t] for t in st.types] or ["chewing tobacco"]
        names.extend(kinds if st.uses else [f"former {k}" for k in kinds])
    sm = rf.smoking
    if sm.uses or (sm.former and not current_only):
        kinds = [SMOKING_NOUN[t] for t in sm.types] or ["smoking"]
        names.extend(kinds if sm.uses else [f"former {k}" for k in kinds])
    an = rf.areca_nut
    if an.uses or (an.former and not current_only):
        names.append("areca nut" if an.uses else "former areca nut")
    if rf.alcohol.frequency == "daily":
        names.append("daily alcohol")
    return list(dict.fromkeys(names))
