"""Plain-language wording and number formatting for reports.

All user-facing words live here so the positioning rules (screening aid, never a diagnosis) are
easy to audit in one place. Nothing here may claim accuracy, sensitivity or specificity.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta

from mukha_pipeline.schema.models import (
    CalibrationStatus,
    ConfounderType,
    Diascopy,
    FactorSource,
    LesionOrigin,
    ModelComponent,
    ModelInfo,
    ModelStatus,
    NormPoint,
    ReferenceType,
    ReferralUrgency,
    RiskFactors,
    RiskLevel,
    WLLabel,
)
from mukha_pipeline.sites import OralSite, site_info

MINUS = "−"
DASH = "—"

# Severity order used to rank lesions and sites (INCONCLUSIVE sits between LOW and MODERATE:
# it needs action, but less urgently than a suspicious finding).
SEVERITY_RANK: dict[RiskLevel, float] = {
    RiskLevel.HIGH: 3.0,
    RiskLevel.MODERATE: 2.0,
    RiskLevel.INCONCLUSIVE: 1.5,
    RiskLevel.LOW: 1.0,
}

RISK_TITLE: dict[RiskLevel, str] = {
    RiskLevel.LOW: "Low risk",
    RiskLevel.MODERATE: "Moderate risk",
    RiskLevel.HIGH: "High risk",
    RiskLevel.INCONCLUSIVE: "Inconclusive",
}

# Risk palette shared by the badge (CSS), site map (SVG) and gauges. Never colour alone:
# every use pairs the colour with an icon and the level text.
RISK_COLORS: dict[RiskLevel, dict[str, str]] = {
    RiskLevel.LOW: {"strong": "#0f766e", "fill": "#5ec2b4", "tint": "#e2f4f1", "text": "#0b3d39"},
    RiskLevel.MODERATE: {
        "strong": "#b45309",
        "fill": "#f2b13e",
        "tint": "#fdf0d8",
        "text": "#4a2a06",
    },
    RiskLevel.HIGH: {"strong": "#b42318", "fill": "#dd4b3e", "tint": "#fde7e4", "text": "#ffffff"},
    RiskLevel.INCONCLUSIVE: {
        "strong": "#475467",
        "fill": "#c5ccd6",
        "tint": "#eef1f5",
        "text": "#1d2939",
    },
}

CALIBRATION_TEXT: dict[CalibrationStatus, str] = {
    CalibrationStatus.CALIBRATED: "Calibrated",
    CalibrationStatus.UNCALIBRATED: "Uncalibrated",
    CalibrationStatus.EXPIRED: "Calibration expired",
    CalibrationStatus.PARTIAL: "Partially calibrated",
}

ANALYSIS_STATUS_TEXT: dict[str, str] = {
    "analysed": "Analysed",
    "inconclusive_qc_failed": "Not analysed — QC failed",
    # A software error, not a QC failure: the site still needs a clinical look.
    "not_analysed": "Not analysed (software error) — examine clinically",
}

REFERENCE_TEXT: dict[ReferenceType, str] = {
    ReferenceType.CONTRALATERAL: "Contralateral",
    ReferenceType.ANALOGOUS_SITE: "Analogous site",
    ReferenceType.IN_IMAGE_AUTO: "In-image (automatic)",
    ReferenceType.NONE: "None",
}

CONFOUNDER_TEXT: dict[ConfounderType, str] = {
    ConfounderType.PORPHYRIN_RED: "Porphyrin / plaque (red fluorescence)",
    ConfounderType.PIGMENTATION: "Pigmentation",
    ConfounderType.BLOOD: "Blood / inflammation",
    ConfounderType.GLARE: "Glare",
}

DIASCOPY_TEXT: dict[Diascopy, str] = {
    Diascopy.NOT_DONE: "Not done",
    Diascopy.FVL_PERSISTS: "AF loss persists under pressure",
    Diascopy.FVL_RESOLVES: "AF loss resolves under pressure (likely vascular)",
}

ORIGIN_TEXT: dict[LesionOrigin, str] = {
    LesionOrigin.AF_FVL: "AF loss",
    LesionOrigin.VISUAL: "Health-worker observation",
    LesionOrigin.WL_MODEL: "White-light model",
    LesionOrigin.SPECTRAL: "Spectral probe",
}

SOURCE_TEXT: dict[FactorSource, str] = {
    FactorSource.AUTOFLUORESCENCE: "Autofluorescence",
    FactorSource.WL_MODEL: "White-light model",
    FactorSource.SPECTRAL: "Spectral probe",
    FactorSource.CLINICAL_VISUAL: "Health-worker observation",
    FactorSource.CLINICAL_HISTORY: "History",
    FactorSource.SYMPTOM: "Symptom",
    FactorSource.SITE: "Site",
    FactorSource.QC: "Quality control",
    FactorSource.OVERRIDE: "Clinical red flag",
}

COMPONENT_TEXT: dict[ModelComponent, str] = {
    ModelComponent.SEGMENTER: "Mucosa segmentation",
    ModelComponent.WL_LESION_CLASSIFIER: "White-light classifier",
    ModelComponent.SPECTRAL_CLASSIFIER: "Spectral classifier",
    ModelComponent.FUSION: "Triage rules",
}

WL_LABEL_TEXT: dict[WLLabel, str] = {
    WLLabel.NORMAL: "No visible change",
    WLLabel.WHITE_PATCH: "White patch",
    WLLabel.RED_PATCH: "Red patch",
    WLLabel.ULCER_MASS: "Ulcer / mass",
    WLLabel.OTHER: "Other change",
}

# The Phase 1 spectral classifier is a rule-based placeholder: its "not suspicious" label must
# never read as reassurance.
SPECTRAL_LABEL_TEXT: dict[str, str] = {
    "not_suspicious": "No spectral difference detected (does not rule out disease)",
    "indeterminate": "Indeterminate",
    "suspicious": "Suspicious (heuristic)",
}

URGENCY_TEXT: dict[ReferralUrgency, str] = {
    ReferralUrgency.NONE: "Routine re-screening — clinician review required",
    ReferralUrgency.REEXAMINE: "Re-examine, refer if persistent",
    ReferralUrgency.URGENT: "Urgent referral",
    ReferralUrgency.RETAKE: "Retake, or refer for clinical examination",
}

CAPTURE_KIND_TEXT: dict[str, str] = {
    "white_light": "White light",
    "autofluorescence": "Autofluorescence",
    "dark_frame": "Dark frame",
    "spectral": "Spectral reading",
}

SEX_TEXT: dict[str, str] = {
    "female": "Female",
    "male": "Male",
    "other": "Other",
    "undisclosed": "Not disclosed",
}

PROVENANCE_TEXT: dict[str, str] = {
    "real": "Real capture",
    "synthetic": "SYNTHETIC",
    "phantom": "Phantom (bench test object)",
    "ex_vivo": "Ex vivo",
    "volunteer": "Volunteer",
    "mixed": "Mixed (includes synthetic)",
}

# The pipeline's lesion-vs-reference delta divides the lesion by the reference band by band, so
# the probe's own LED / filter / sensor gains cancel (spectral/features.py reference_delta):
# ratios are a relative change, the Hb dip is the dip of the quotient, slopes subtract.
SPECTRAL_DELTA_NOTE = (
    "Δ: lesion ÷ reference, band by band (the probe's own gains cancel): ratios as % change, "
    "Hb dip of the quotient, slopes as difference."
)

SPECTRAL_FEATURE_TEXT: dict[str, str] = {
    "hb_absorption_proxy": "Hb dip (540–580 nm)",
    "ratio_555_630": "555 / 630 nm ratio",
    "ratio_590_680": "590 / 680 nm ratio",
    "ratio_515_630": "515 / 630 nm ratio",
    "ratio_415_555": "415 / 555 nm ratio",
    "slope_480_555": "Slope 480→555 nm",
    "slope_590_680": "Slope 590→680 nm",
}

_SMOKELESS_TEXT = {
    "gutka": "Gutka",
    "khaini": "Khaini",
    "zarda": "Zarda",
    "paan_with_tobacco": "Paan with tobacco",
    "mawa": "Mawa",
    "mishri": "Mishri",
    "snuff": "Snuff",
    "other": "Smokeless tobacco",
}
_SMOKING_TEXT = {
    "bidi": "bidi",
    "cigarette": "cigarettes",
    "hookah": "hookah",
    "chutta_reverse": "chutta (reverse smoking)",
    "other": "smoking",
}
_SYMPTOM_TEXT = {
    "non_healing_ulcer_over_2_weeks": "non-healing ulcer > 2 weeks",
    "white_patch": "white patch noticed",
    "red_patch": "red patch noticed",
    "lump": "lump in the mouth",
    "pain": "pain",
    "difficulty_opening_mouth": "difficulty opening the mouth",
    "bleeding": "bleeding",
    "burning_sensation": "burning sensation",
    "neck_lump": "lump in the neck",
}


# --------------------------------------------------------------------------------------------
# Numbers and dates
# --------------------------------------------------------------------------------------------


def _finite(x: float | None) -> bool:
    return x is not None and not (isinstance(x, float) and (math.isnan(x) or math.isinf(x)))


def num(x: float | None, digits: int = 2) -> str:
    """Fixed-point number with a typographic minus; em dash when missing."""
    if not _finite(x):
        return DASH
    text = f"{x:.{digits}f}"
    return text.replace("-", MINUS) if text.startswith("-") else text


def signed(x: float | None, digits: int = 1) -> str:
    """Points with an explicit sign: +1.5, −0.4, 0.0."""
    if not _finite(x):
        return DASH
    assert x is not None
    if abs(x) < 0.5 * 10**-digits:
        return f"{0:.{digits}f}"
    return f"+{x:.{digits}f}" if x > 0 else f"{MINUS}{abs(x):.{digits}f}"


def signed_pct(fraction: float | None, digits: int = 1) -> str:
    """A relative change with an explicit sign: +4.2 %, −6.4 %, 0.0 %."""
    if not _finite(fraction):
        return DASH
    assert fraction is not None
    return f"{signed(fraction * 100, digits)} %"


def pct(fraction: float | None, digits: int = 1) -> str:
    if not _finite(fraction):
        return DASH
    assert fraction is not None
    return f"{fraction * 100:.{digits}f} %"


def years(value: float | None) -> str:
    if not _finite(value):
        return ""
    assert value is not None
    return f"{value:g} y"


def short_sha(sha: str | None, n: int = 8) -> str:
    return (sha or "")[:n] or DASH


def fmt_datetime(dt: datetime | None) -> str:
    if dt is None:
        return DASH
    text = dt.strftime("%d %b %Y, %H:%M")
    if dt.tzinfo is not None:
        name = dt.tzname() or ""
        offset = dt.strftime("%z")
        if name in ("UTC", "IST"):
            text += f" {name}"
        elif offset:
            text += f" (UTC{offset[:3]}:{offset[3:]})"
    return text


def fmt_date(dt: datetime | None) -> str:
    return dt.strftime("%d %b %Y") if dt is not None else DASH


SAME_DAY_TEXT = "Same day (at the screening visit)"


def timeline_text(days: int | None, start: datetime | None = None) -> str:
    """Plain-words timeline; with ``start`` the due date is added ("Within 2 weeks — by …")."""
    if days is None:
        return "No fixed timeline"
    if days == 0:
        return SAME_DAY_TEXT
    if days >= 330 and days % 365 < 31:
        n_years = max(1, round(days / 365))
        short = long = "In 12 months" if n_years == 1 else f"In {n_years * 12} months"
    elif days % 7 == 0 and days >= 14:
        short = f"Within {days // 7} weeks"
        long = f"{short} ({days} days)"
    else:
        short = long = f"Within {days} day{'s' if days != 1 else ''}"
    if start is not None:
        return f"{short} — by {fmt_date(start + timedelta(days=days))}"
    return long


def mask_id(value: str | None, keep: int = 4, mask: str = "X") -> str | None:
    """Mask an external ID (ABHA / Aadhaar-linked numbers), keeping its last ``keep`` letters
    or digits and every separator: "91-2345-6789-0123" -> "XX-XXXX-XXXX-0123"."""
    if not value:
        return value
    positions = [i for i, ch in enumerate(value) if ch.isalnum()]
    n = len(positions)
    show = keep if n > keep else max(0, n - 2)
    hidden = set(positions[: n - show])
    return "".join(mask if i in hidden else ch for i, ch in enumerate(value))


# --------------------------------------------------------------------------------------------
# Domain wording
# --------------------------------------------------------------------------------------------


def site_name(site: OralSite | str) -> str:
    return site_info(site).label


def position_text(point: NormPoint | None) -> str:
    """Where a region sits in the white-light frame, in words (frame-relative, not anatomy)."""
    if point is None:
        return DASH
    row = "upper" if point.y < 1 / 3 else ("lower" if point.y > 2 / 3 else "middle")
    col = "left" if point.x < 1 / 3 else ("right" if point.x > 2 / 3 else "centre")
    where = "centre" if (row, col) == ("middle", "centre") else f"{row} {col}"
    return f"In photo: {where}"


def model_status_text(model: ModelInfo) -> str:
    """Honest status wording for a model component."""
    if model.status == ModelStatus.MODEL_UNAVAILABLE:
        if model.component == ModelComponent.WL_LESION_CLASSIFIER:
            return "Not available — not used in triage"
        return "Not available — not used"
    if model.status == ModelStatus.PLACEHOLDER:
        return "Placeholder heuristic, not validated"
    if model.status == ModelStatus.BASELINE_RULES:
        return "Baseline heuristic, unvalidated"
    return "Available — not clinically validated" if not model.validated else "Available"


def risk_factor_groups(rf: RiskFactors) -> dict[str, list[str]]:
    """Risk factors grouped into plain-language phrases."""
    habits: list[str] = []

    st = rf.smokeless_tobacco
    if st.uses or st.former:
        kinds = [_SMOKELESS_TEXT.get(t, t) for t in st.types] or ["Smokeless tobacco"]
        phrase = " / ".join(kinds)
        detail = _habit_detail(st.years, st.per_day)
        if detail:
            phrase += f" {detail}"
        habits.append(f"former {phrase.lower()}" if (st.former and not st.uses) else phrase)

    sm = rf.smoking
    if sm.uses or sm.former:
        kinds = [_SMOKING_TEXT.get(t, t) for t in sm.types] or ["smoking"]
        phrase = " / ".join(kinds)
        detail = _habit_detail(sm.years, sm.per_day)
        if detail:
            phrase += f" {detail}"
        habits.append(f"former {phrase}" if (sm.former and not sm.uses) else phrase)

    an = rf.areca_nut
    if an.uses or an.former:
        phrase = "areca nut (supari)"
        detail = _habit_detail(an.years, an.per_day)
        if detail:
            phrase += f" {detail}"
        habits.append(f"former {phrase}" if (an.former and not an.uses) else phrase)

    if rf.alcohol.uses or rf.alcohol.frequency != "never":
        freq = {"occasional": "occasionally", "never": ""}.get(
            rf.alcohol.frequency, rf.alcohol.frequency
        )
        habits.append(f"alcohol {freq}".strip())

    symptoms = [text for key, text in _SYMPTOM_TEXT.items() if getattr(rf.symptoms, key)]

    history: list[str] = []
    if rf.prior_oral_cancer:
        history.append("previous oral cancer")
    if rf.prior_oral_lesion:
        history.append("previous oral lesion")

    irritants: list[str] = []
    if rf.irritants.sharp_tooth:
        irritants.append("sharp tooth")
    if rf.irritants.ill_fitting_denture:
        irritants.append("ill-fitting denture")

    return {"Habits": habits, "Symptoms": symptoms, "History": history, "Irritants": irritants}


def risk_factor_summary(rf: RiskFactors) -> str:
    """One line, e.g. "Gutka 20 y (6/day) · bidi 15 y · alcohol daily; non-healing ulcer > 2
    weeks" (items joined by " · ", groups by "; ")."""
    groups = risk_factor_groups(rf)
    parts = [" · ".join(items) for items in groups.values() if items]
    if not parts:
        return "No risk factors or symptoms reported"
    text = "; ".join(parts)
    return text[:1].upper() + text[1:]


def _per_day(value: float | None) -> str:
    if not _finite(value):
        return ""
    return f"{value:g}/day"


def _habit_detail(years_used: float | None, per_day: float | None) -> str:
    """ "20 y (6/day)", "20 y", "(6/day)" or ""."""
    y, d = years(years_used), _per_day(per_day)
    if y and d:
        return f"{y} ({d})"
    return y or (f"({d})" if d else "")
