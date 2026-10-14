"""Build the template view model from a ScreeningResult.

All decisions about what to show (and how to word it) are made here in plain Python, so the
Jinja template stays declarative and every wording rule is unit-testable.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta, timezone, tzinfo
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from markupsafe import Markup

from app.reports import labels as lb
from app.reports.charts import risk_icon_svg, score_gauge_svg, spectral_plot_svg
from app.reports.images import EmbeddedImage, embed_artifact, svg_data_uri, union_box
from app.reports.qr import qr_data_uri
from app.reports.ranking import lesion_numbers, ordered_lesions
from app.reports.sitemap import render_site_map_svg
from app.reports.types import ArtifactReader, Facility, PatientIdentity, ReportOptions
from mukha_pipeline.schema.models import (
    SYNTHETIC_NOTICE,
    CalibrationStatus,
    ContributingFactor,
    DataProvenance,
    FactorSource,
    FVLRegion,
    LesionAssessment,
    LesionOrigin,
    ModelComponent,
    ModelStatus,
    NormBox,
    QCIssue,
    QCSeverity,
    ReferralUrgency,
    RiskLevel,
    ScreeningResult,
    SiteResult,
    SiteRole,
    SpectralReadingResult,
    WLAnalysis,
)
from mukha_pipeline.sites import OralSite

log = logging.getLogger(__name__)

# Two findings get the full evidence block (images, measurements, spectra, factors); the rest
# are listed compactly. More detailed blocks break the 2–4 page limit.
MAX_DETAILED_LESIONS = 2
MAX_FACILITIES = 3
N_SITES = len(OralSite)
PANEL_ASPECT = (58.0, 30.0)  # printed image frame (mm): crops fill it instead of letterboxing
OVERRIDE_TAG = "Red flag (floor)"
OVERRIDE_FOOTNOTE = (
    "Red flag (floor): a clinical red flag sets a minimum risk level; the device cannot lower it."
)
SYMPTOM_SITE_NOTE = "patient-reported; site not recorded — examine the whole mouth and neck"
SCREENING_AID_TITLE = "Screening aid — not a diagnosis"
SIGNATURE_NOTE = "(recorded in MukhaSpectra — not a digital signature)"
FACILITY_MISSING = "Referral facility not configured — contact the screening programme"
IST_FALLBACK = timezone(timedelta(hours=5, minutes=30), "IST")

PROVENANCE_NOTICE: dict[DataProvenance, tuple[str, str]] = {
    DataProvenance.PHANTOM: (
        "PHANTOM DATA",
        "PHANTOM DATA — captured from a bench test object, not a patient. For engineering "
        "tests only.",
    ),
    DataProvenance.EX_VIVO: (
        "EX VIVO DATA",
        "EX VIVO DATA — captured from a tissue sample, not a live patient. For research only.",
    ),
    DataProvenance.VOLUNTEER: (
        "VOLUNTEER DATA",
        "VOLUNTEER DATA — research volunteer, not a screening patient. For research only.",
    ),
    DataProvenance.MIXED: (
        "MIXED DATA",
        "MIXED DATA — includes inputs that are not from this patient. For research only.",
    ),
}
SEX_SHORT = {"male": "M", "female": "F", "other": "other", "undisclosed": ""}
FACILITY_LEAD_IN = {
    ReferralUrgency.NONE: "If a symptom appears or a patch persists > 2 weeks:",
    ReferralUrgency.RETAKE: "If a retake is not possible or a lesion is visible:",
}
# Words that tie a spectral feature to a classifier reason or a contributing factor.
_FEATURE_CUES: dict[str, re.Pattern[str]] = {
    "hb_absorption_proxy": re.compile(r"haemoglobin|hemoglobin|\bhb\b", re.I),
    "ratio_555_630": re.compile(r"555\s*/\s*630|ratio_555_630", re.I),
    "ratio_590_680": re.compile(r"590\s*/\s*680|ratio_590_680", re.I),
    "ratio_515_630": re.compile(r"515\s*/\s*630|ratio_515_630", re.I),
    "ratio_415_555": re.compile(r"415\s*/\s*555|ratio_415_555", re.I),
    "slope_480_555": re.compile(r"480\s*(?:→|->|–|-|to)\s*555|slope_480_555", re.I),
    "slope_590_680": re.compile(r"590\s*(?:→|->|–|-|to)\s*680|slope_590_680", re.I),
}

# --------------------------------------------------------------------------------------------
# View-model types
# --------------------------------------------------------------------------------------------


@dataclass
class Badge:
    level: str
    title: str
    css: str
    icon: Markup
    icon_inverse: Markup


@dataclass
class FactorRow:
    label: str
    detail: str | None
    source: str
    points: str
    is_override: bool = False
    negative: bool = False


@dataclass
class HeatmapScale:
    """Colour key of the AF-loss heatmap (same ramp as ``mukha_pipeline.af.overlay``)."""

    gradient: str  # CSS linear-gradient
    upper: str  # relative AF where colour starts (threshold + 0.1)
    lower: str  # relative AF where the ramp saturates


@dataclass
class ImagePanel:
    title: str
    note: str
    image: EmbeddedImage
    scale: HeatmapScale | None = None


@dataclass
class RegionRow:
    region_id: str
    site: str
    position: str
    area: str
    area_note: str | None
    mean_rel: str
    min_rel: str
    contrast: str
    border: str
    reference: str
    confounders: list[str]


@dataclass
class FeatureRow:
    name: str
    lesion: str
    reference: str
    delta: str
    cited: bool = False


@dataclass
class SpectralView:
    reading_id: str
    plot_uri: str
    site: str
    reference_site: str | None
    signal_text: str
    calibration_text: str
    qc_text: str | None
    features: list[FeatureRow]
    classifier_label: str | None
    classifier_status: str
    classifier_score: str | None
    classifier_reasons: list[str]
    classifier_model: str | None
    notes: list[str] = field(default_factory=list)


@dataclass
class WLView:
    status_text: str
    available: bool
    predictions: list[tuple[str, str]]
    model_text: str
    note: str | None
    saliency: EmbeddedImage | None

    @property
    def short_text(self) -> str:
        """Compact status for the observations line ("WL classifier: …")."""
        return self.status_text.replace("White-light classifier", "WL classifier")


@dataclass
class LesionView:
    number: int
    lesion_id: str
    site: str
    badge: Badge
    origins: str
    summary: str
    action: str
    panels: list[ImagePanel]
    marks_drawn: bool
    regions: list[RegionRow]
    fvl_threshold: str | None
    region_note: str | None
    confounder_notes: list[str]
    visual: str
    diascopy: str
    spectral: list[SpectralView]
    wl: WLView
    factors: list[FactorRow]
    score: str


@dataclass
class CompactLesionRow:
    number: int
    site: str
    badge: Badge
    score: str
    origins: str
    summary: str


@dataclass
class IssueRow:
    site: str
    capture: str
    severity: str
    css: str
    message: str


@dataclass
class SiteRow:
    site: str
    role: str
    qc: str
    qc_ok: bool
    analysis: str
    calibration: str
    calibration_ok: bool
    badge: Badge
    reference_only: bool = False

    @property
    def show_level(self) -> bool:
        """A healthy reference site shows "Reference only" (as on the map), not a LOW pill."""
        return not (self.reference_only and self.badge.level == RiskLevel.LOW.value)


@dataclass
class Coverage:
    """How much of the mouth the device looked at (14 catalogue sites)."""

    imaged: int
    total: int
    reference_only: int
    not_examined: list[str]
    not_assessed: list[str]

    @property
    def summary(self) -> str:
        text = f"Device coverage: {self.imaged} of {self.total} sites"
        if self.reference_only:
            text += f" ({self.reference_only} used only as a healthy reference)"
        return text + "."


@dataclass
class ModelRow:
    component: str
    name: str
    version: str
    status: str
    validated: str
    css: str

    @property
    def available(self) -> bool:
        return not self.status.startswith("Not available")

    @property
    def status_short(self) -> str:
        """Lower-case status for the header line, e.g. "not available — not used in triage"."""
        return self.status[:1].lower() + self.status[1:]


# --------------------------------------------------------------------------------------------
# Small builders
# --------------------------------------------------------------------------------------------


def badge(level: RiskLevel | str) -> Badge:
    lv = RiskLevel(level)
    colors = lb.RISK_COLORS[lv]
    inverse_color = colors["strong"] if lv == RiskLevel.INCONCLUSIVE else "#ffffff"
    return Badge(
        level=lv.value,
        title=lb.RISK_TITLE[lv],
        css=lv.value.lower(),
        icon=Markup(risk_icon_svg(lv, colors["strong"])),
        icon_inverse=Markup(risk_icon_svg(lv, inverse_color)),
    )


def _is_symptom(f: ContributingFactor) -> bool:
    return f.source == FactorSource.SYMPTOM or f.code.startswith("symptom.")


def _override_detail(f: ContributingFactor) -> str | None:
    """Red-flag detail; a reported symptom has no site, so say where to look."""
    if not _is_symptom(f):
        return f.detail
    return f"{f.detail} · {SYMPTOM_SITE_NOTE}" if f.detail else SYMPTOM_SITE_NOTE


def _factor_rows(
    factors: list[ContributingFactor],
    overrides: list[ContributingFactor] | None = None,
    *,
    sources: dict[str, str] | None = None,
) -> list[FactorRow]:
    """Rows for a factor table; ``sources`` replaces the source text by factor code."""
    override_codes = {o.code for o in overrides or []}
    sources = sources or {}
    rows: list[FactorRow] = []
    for f in overrides or []:
        rows.append(
            FactorRow(
                label=f.label,
                detail=_override_detail(f),
                source=lb.SOURCE_TEXT.get(f.source, str(f.source)),
                points="floor",
                is_override=True,
            )
        )
    for f in factors:
        if f.code in override_codes:
            continue
        is_override = f.source == FactorSource.OVERRIDE
        rows.append(
            FactorRow(
                label=f.label,
                detail=_override_detail(f) if is_override else f.detail,
                source=sources.get(f.code) or lb.SOURCE_TEXT.get(f.source, str(f.source)),
                points="floor" if is_override else lb.signed(f.points),
                is_override=is_override,
                negative=f.points < 0,
            )
        )
    return rows


def _scored_level(score: float | None, moderate: float, high: float) -> RiskLevel | None:
    if score is None:
        return None
    if score >= high:
        return RiskLevel.HIGH
    if score >= moderate:
        return RiskLevel.MODERATE
    return RiskLevel.LOW


def _reference_text(site: SiteResult) -> str:
    if site.af is None:
        return lb.DASH
    ref = site.af.reference
    text = lb.REFERENCE_TEXT.get(ref.type, str(ref.type))
    if ref.reference_site is not None:
        text += f" — {lb.site_name(ref.reference_site)}"
    return text


def _region_row(site: SiteResult, region: FVLRegion) -> RegionRow:
    if region.area_mm2 is not None:
        area = f"{lb.num(region.area_mm2, 1)} mm²"
        area_note: str | None = f"{lb.pct(region.area_fraction_of_mucosa)} of mucosa"
    else:
        area = f"{lb.pct(region.area_fraction_of_mucosa)} of mucosa"
        area_note = "mm scale unknown"
    return RegionRow(
        region_id=region.region_id,
        site=lb.site_name(site.site),
        position=lb.position_text(region.centroid),
        area=area,
        area_note=area_note,
        mean_rel=lb.num(region.mean_relative_af),
        min_rel=lb.num(region.min_relative_af),
        contrast=f"{lb.num(region.contrast_vs_surround)}×",
        border=lb.num(region.border_irregularity),
        reference=_reference_text(site),
        confounders=[
            f"{lb.CONFOUNDER_TEXT.get(c.type, str(c.type))} {lb.pct(c.overlap_fraction, 0)}"
            for c in region.confounders
        ],
    )


def _wl_view(site: SiteResult | None, reader: ArtifactReader) -> WLView:
    wl: WLAnalysis | None = site.wl if site is not None else None
    if wl is None:
        return WLView(
            "White-light classifier: not run for this site — not used in triage",
            False, [], "", None, None,
        )  # fmt: skip
    model_text = f"{wl.model.name} {wl.model.version}"
    if wl.status == "ok" and wl.model.status not in (
        ModelStatus.MODEL_UNAVAILABLE,
        ModelStatus.PLACEHOLDER,
    ):
        preds = sorted(wl.predictions, key=lambda p: -p.probability)
        saliency = None
        if wl.saliency_artifact or (site is not None and site.artifacts.wl_saliency):
            key = wl.saliency_artifact or (site.artifacts.wl_saliency if site else None)
            saliency = embed_artifact(reader, key, max_side=360)
            if not saliency.available:
                saliency = None
        return WLView(
            status_text="White-light classifier output (model output, not clinically validated)",
            available=True,
            predictions=[
                (lb.WL_LABEL_TEXT.get(p.label, str(p.label)), lb.num(p.probability)) for p in preds
            ],
            model_text=model_text,
            note=wl.note,
            saliency=saliency,
        )
    if wl.status == "model_unavailable" or wl.model.status == ModelStatus.MODEL_UNAVAILABLE:
        # Name and note add nothing when there are no weights; keep the line short.
        return WLView(
            "White-light classifier: not available — not used in triage", False, [], "", None, None
        )
    if wl.status == "skipped_qc_failed":
        text = "White-light classifier: skipped because the capture failed QC — not used"
    elif wl.status == "error":
        text = "White-light classifier: error while running — not used in triage"
    else:
        text = "White-light classifier: placeholder — not used in triage"
    return WLView(text, False, [], model_text, wl.note, None)


def _classifier_status(status: str) -> str:
    return {
        "placeholder": "Placeholder heuristic, not validated",
        "ok": "Model output, not clinically validated",
        "model_unavailable": "Not available — not used",
        "skipped_qc_failed": "Skipped — reading failed QC",
    }.get(status, status)


def _cited_features(texts: list[str]) -> set[str]:
    """Spectral feature keys named in classifier reasons or contributing factors."""
    joined = " | ".join(texts)
    return {key for key, cue in _FEATURE_CUES.items() if cue.search(joined)}


def _spectral_view(
    reading: SpectralReadingResult,
    readings_by_id: dict[str, SpectralReadingResult],
    factors: list[ContributingFactor] | None = None,
) -> SpectralView:
    reference = (
        readings_by_id.get(reading.reference_reading_id) if reading.reference_reading_id else None
    )
    notes: list[str] = []
    if reference is None:
        notes.append("No reference reading — absolute values are hard to interpret.")
    elif reference.signal_type != reading.signal_type:
        notes.append("Lesion and reference were measured with different signal types.")
    if reference is not None and not reference.qc.passed:
        notes.append("The reference reading failed QC; the comparison is unreliable.")

    clf = reading.classifier
    cue_texts = list(clf.reasons) if clf else []
    cue_texts += [
        f"{f.code} {f.label} {f.detail or ''}"
        for f in factors or []
        if f.source == FactorSource.SPECTRAL
    ]
    cited = _cited_features(cue_texts)
    features: list[FeatureRow] = []
    if reading.features is not None:
        lf = reading.features.model_dump()
        rf = reference.features.model_dump() if reference and reference.features else {}
        df = reading.delta_vs_reference.model_dump() if reading.delta_vs_reference else {}
        for key, name in lb.SPECTRAL_FEATURE_TEXT.items():
            delta = df.get(key)
            if delta is None and key in rf:
                delta = lf[key] - rf[key]
            features.append(
                FeatureRow(
                    name,
                    lb.num(lf.get(key), 2),
                    lb.num(rf.get(key), 2),
                    lb.signed(delta, 2) if delta is not None else lb.DASH,
                    cited=key in cited,
                )
            )

    qc_text = None
    if not reading.qc.passed:
        msgs = "; ".join(i.message for i in reading.qc.issues if i.severity == QCSeverity.ERROR)
        qc_text = f"Reading failed QC{': ' + msgs if msgs else ''}"
    cal = reading.calibration.status
    if reading.signal_type == "reflectance":
        calibration_text = (
            "reflectance, calibrated on the white standard"
            if cal == CalibrationStatus.CALIBRATED
            else f"reflectance ({lb.CALIBRATION_TEXT.get(cal, str(cal)).lower()})"
        )
    else:
        calibration_text = (
            f"relative signal ({lb.CALIBRATION_TEXT.get(cal, str(cal)).lower()}; not reflectance)"
        )
    return SpectralView(
        reading_id=reading.reading_id,
        plot_uri=svg_data_uri(spectral_plot_svg(reading, reference)),
        site=lb.site_name(reading.site),
        reference_site=lb.site_name(reference.site) if reference else None,
        signal_text="Reflectance" if reading.signal_type == "reflectance" else "Relative signal",
        calibration_text=calibration_text,
        qc_text=qc_text,
        features=features,
        classifier_label=(
            lb.SPECTRAL_LABEL_TEXT.get(clf.label, clf.label) if clf and clf.label else None
        ),
        classifier_status=_classifier_status(clf.status) if clf else "Not run",
        # A placeholder heuristic's score invites over-reading: show the label and status only.
        classifier_score=(
            lb.num(clf.suspicion_score)
            if clf
            and clf.suspicion_score is not None
            and clf.status != "placeholder"
            and clf.model.status != ModelStatus.PLACEHOLDER
            else None
        ),
        classifier_reasons=list(clf.reasons) if clf else [],
        classifier_model=f"{clf.model.name} {clf.model.version}" if clf else None,
        notes=notes,
    )


def _lesion_view(
    number: int,
    lesion: LesionAssessment,
    result: ScreeningResult,
    sites: dict[OralSite, SiteResult],
    readings_by_id: dict[str, SpectralReadingResult],
    reader: ArtifactReader,
    heatmap_scale: HeatmapScale | None = None,
) -> LesionView:
    site = sites.get(OralSite(lesion.site))
    regions: list[FVLRegion] = []
    if site is not None and site.af is not None:
        wanted = set(lesion.region_ids)
        regions = [r for r in site.af.regions if r.region_id in wanted]
    boxes: list[NormBox] = [r.bbox for r in regions]

    arts = site.artifacts if site is not None else None
    af_key = None
    af_note = "405 nm"
    af_in_wl_frame = False
    if arts is not None:
        if arts.af_registered:
            af_key, af_in_wl_frame = arts.af_registered, True
            af_note = "405 nm, registered to WL"
        elif arts.af_image:
            af_key = arts.af_image
            af_note = "405 nm, unregistered"
    # One crop for every panel in the white-light frame, centred on the flagged region(s).
    focus = union_box(boxes)
    af_boxes, af_focus = (boxes, focus) if af_in_wl_frame else (None, None)
    heatmap = embed_artifact(
        reader,
        arts.heatmap_overlay if arts else None,
        boxes=boxes,
        aspect=PANEL_ASPECT,
        focus=focus,
    )
    panels = [
        ImagePanel("White light", "as captured", embed_artifact(
            reader, arts.wl_image if arts else None, boxes=boxes, aspect=PANEL_ASPECT,
            focus=focus)),
        ImagePanel("Autofluorescence", af_note, embed_artifact(
            reader, af_key, boxes=af_boxes, aspect=PANEL_ASPECT, focus=af_focus)),
        ImagePanel("AF-loss heatmap", "on white light", heatmap,
                   scale=heatmap_scale if heatmap.available else None),
    ]  # fmt: skip

    region_note = None
    if not regions:
        if LesionOrigin.AF_FVL in lesion.origins:
            region_note = "AF-loss region details are not available for this finding."
        else:
            region_note = (
                "No AF-loss region at this finding — it was raised by "
                f"{_origins_text(lesion).lower()}."
            )

    confounder_notes: list[str] = []
    seen: set[str] = set()
    for c in [c for r in regions for c in r.confounders] + list(lesion.confounders):
        key = f"{c.type}:{c.note}"
        if key in seen:
            continue
        seen.add(key)
        confounder_notes.append(
            f"{lb.CONFOUNDER_TEXT.get(c.type, str(c.type))} "
            f"({lb.pct(c.overlap_fraction, 0)} overlap): {c.note}"
        )

    visual = "no lesion recorded"
    if site is not None and site.visual_flag.lesion_seen:
        appearance = ", ".join(a.value.replace("_", " ") for a in site.visual_flag.appearance)
        visual = f"Seen by health worker: {appearance or 'lesion'}"
        if site.visual_flag.description:
            visual += f" — “{site.visual_flag.description}”"

    spectral_ids = list(lesion.spectral_reading_ids)
    primaries = [
        readings_by_id[i]
        for i in spectral_ids
        if i in readings_by_id and readings_by_id[i].role == "lesion"
    ]
    if not primaries:
        primaries = [readings_by_id[i] for i in spectral_ids if i in readings_by_id]

    return LesionView(
        number=number,
        lesion_id=lesion.lesion_id,
        site=lb.site_name(lesion.site),
        badge=badge(lesion.risk_level),
        origins=_origins_text(lesion),
        summary=lesion.summary,
        action=lesion.recommended_action.summary,
        panels=panels,
        marks_drawn=bool(boxes),
        regions=[_region_row(site, r) for r in regions] if site is not None else [],
        fvl_threshold=lb.num(site.af.fvl_threshold) if site and site.af else None,
        region_note=region_note,
        confounder_notes=confounder_notes,
        visual=visual,
        diascopy=lb.DIASCOPY_TEXT.get(site.diascopy, str(site.diascopy)) if site else lb.DASH,
        spectral=[
            _spectral_view(r, readings_by_id, lesion.contributing_factors) for r in primaries
        ],
        wl=_wl_view(site, reader),
        factors=_factor_rows(lesion.contributing_factors),
        score=lb.num(lesion.score, 1),
    )


def _origins_text(lesion: LesionAssessment) -> str:
    return ", ".join(lb.ORIGIN_TEXT.get(o, str(o)) for o in lesion.origins) or lb.DASH


def _issue_rows(result: ScreeningResult) -> list[IssueRow]:
    rows: list[IssueRow] = []
    seen: set[tuple[str, str, str]] = set()

    def add(site_label: str, capture: str, issue: QCIssue) -> None:
        key = (site_label, issue.code, issue.message)
        if key in seen or issue.severity == QCSeverity.INFO:
            return
        seen.add(key)
        error = issue.severity == QCSeverity.ERROR
        rows.append(
            IssueRow(
                site_label,
                capture,
                "Retake needed" if error else "Warning",
                "error" if error else "warning",
                issue.message,
            )
        )

    for site in result.sites:
        label = lb.site_name(site.site)
        for cap in site.captures:
            kind = lb.CAPTURE_KIND_TEXT.get(cap.metadata.kind.value, cap.metadata.kind.value)
            for issue in cap.qc.issues:
                add(label, kind, issue)
        for issue in site.qc.issues:
            add(label, "Site analysis", issue)
    for reading in result.spectral_readings:
        for issue in reading.qc.issues:
            add(lb.site_name(reading.site), "Spectral reading", issue)
    for issue in result.qc_summary.issues:
        if not any(r.message == issue.message for r in rows):
            add(lb.DASH, lb.DASH, issue)
    rows.sort(key=lambda r: (r.css != "error", r.site))
    return rows


def _site_rows(result: ScreeningResult) -> list[SiteRow]:
    order = {s: i for i, s in enumerate(OralSite)}
    rows = []
    for site in sorted(result.sites, key=lambda s: (s.role != SiteRole.EXAMINED, order[s.site])):
        cal = site.calibration.status
        rows.append(
            SiteRow(
                site=lb.site_name(site.site),
                role="Examined" if site.role == SiteRole.EXAMINED else "Reference only",
                qc="Passed" if site.qc.passed else "Failed",
                qc_ok=site.qc.passed,
                analysis=lb.ANALYSIS_STATUS_TEXT.get(site.analysis_status, site.analysis_status),
                calibration=lb.CALIBRATION_TEXT.get(cal, str(cal)),
                calibration_ok=cal == CalibrationStatus.CALIBRATED,
                badge=badge(site.site_risk_level),
                reference_only=site.role == SiteRole.REFERENCE_ONLY,
            )
        )
    return rows


def _model_rows(result: ScreeningResult) -> list[ModelRow]:
    order = list(ModelComponent)
    rows = []
    for m in sorted(result.versions.models, key=lambda m: order.index(m.component)):
        css = "muted" if m.status == ModelStatus.MODEL_UNAVAILABLE else ""
        rows.append(
            ModelRow(
                component=lb.COMPONENT_TEXT.get(m.component, str(m.component)),
                name=m.name,
                version=m.version,
                status=lb.model_status_text(m),
                validated="Yes (see model card)" if m.validated else "No",
                css=css,
            )
        )
    return rows


def _ensure_wl_model_row(result: ScreeningResult, rows: list[ModelRow]) -> list[ModelRow]:
    """Always state the white-light classifier status, even if the result omits it."""
    if any(m.component == ModelComponent.WL_LESION_CLASSIFIER for m in result.versions.models):
        return rows
    return [
        ModelRow("White-light classifier", lb.DASH, lb.DASH,
                 "Not available — not used in triage", "No", "muted"),
        *rows,
    ]  # fmt: skip


# --------------------------------------------------------------------------------------------
# Time, coverage, review, referral helpers
# --------------------------------------------------------------------------------------------


def display_timezone(name: str | None) -> tzinfo:
    """``ZoneInfo(name)``; IST without the tz database for Asia/Kolkata; UTC otherwise."""
    if not name:
        return UTC
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError):
        if name == "Asia/Kolkata":
            return IST_FALLBACK
        log.warning("unknown report display time zone %r; using UTC", name)
        return UTC


def _local(dt: datetime | None, tz: tzinfo) -> datetime | None:
    if dt is None:
        return None
    if dt.tzinfo is None:  # stored datetimes are UTC
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(tz)


def _lc(text: str) -> str:
    return text[:1].lower() + text[1:]


def _coverage(result: ScreeningResult) -> Coverage:
    present = {OralSite(s.site): s for s in result.sites}
    failed = {OralSite(s) for s in result.qc_summary.sites_failed}
    failed |= {
        OralSite(s.site)
        for s in result.sites
        if s.role == SiteRole.EXAMINED and s.analysis_status != "analysed"
    }
    return Coverage(
        imaged=len(present),
        total=N_SITES,
        reference_only=sum(1 for s in present.values() if s.role == SiteRole.REFERENCE_ONLY),
        not_examined=[_lc(lb.site_name(s)) for s in OralSite if s not in present],
        not_assessed=[lb.site_name(s) for s in OralSite if s in failed],
    )


def _heatmap_scale(threshold: float | None) -> HeatmapScale | None:
    """Colour key from the pipeline's own heatmap ramp (lazy import; omitted if unavailable)."""
    if threshold is None or not math.isfinite(threshold):
        return None
    try:
        from mukha_pipeline.af import heatmap_legend

        entries = heatmap_legend(float(threshold), n=9)
        colors = [str(e["color"]) for e in entries]
        upper, lower = float(entries[0]["relative_af"]), float(entries[-1]["relative_af"])
    except Exception:  # the AF module is optional for the report engine
        log.warning("AF heatmap legend unavailable; heatmap printed without a colour key")
        return None
    if len(colors) < 2 or not all(re.fullmatch(r"#[0-9a-fA-F]{6}", c) for c in colors):
        return None
    return HeatmapScale(
        gradient=f"linear-gradient(to right, {', '.join(colors)})",
        upper=lb.num(upper),
        lower=lb.num(lower),
    )


def _review_view(
    result: ScreeningResult, overall: RiskLevel, tz: tzinfo
) -> tuple[Badge | None, str | None, RiskLevel]:
    """(reviewed badge, reviewer line, effective level) from the clinician review, if any."""
    review = result.review
    if review is None:
        return None, None, overall
    who = review.reviewer_name
    when = lb.fmt_date(_local(review.signed_at, tz))
    modified = RiskLevel(review.modified_risk_level) if review.modified_risk_level else None
    if modified is not None and not review.agreed and modified != overall:
        line = (
            f"Clinician-reviewed level: {modified.value} (device: {overall.value}) — {who}, {when}"
        )
        return badge(modified), line, modified
    if review.agreed or modified == overall:
        return (
            None,
            f"Clinician-reviewed: agrees with the device level ({overall.value}) — {who}, {when}",
            overall,
        )
    return (
        None,
        f"Clinician-reviewed: see the reviewer's notes (device level {overall.value}) — "
        f"{who}, {when}",
        overall,
    )


def _select_facilities(
    result: ScreeningResult, options: ReportOptions, level: RiskLevel
) -> tuple[list[Facility], int]:
    """Facilities for this report and how many more matched than are printed."""
    facilities = list(options.facilities)
    if not result.synthetic:
        # Placeholder entries must never reach a report about a real (or bench) capture.
        facilities = [f for f in facilities if not f.example]
    facilities = [f for f in facilities if not f.levels or level.value in f.levels]
    if not facilities:
        log.warning(
            "report %s: no referral facility configured for level %s",
            options.report_id,
            level.value,
        )
    return facilities[:MAX_FACILITIES], max(0, len(facilities) - MAX_FACILITIES)


def _slip_reason(
    result: ScreeningResult,
    lesions: list[LesionAssessment],
    overall: RiskLevel,
    reviewed: RiskLevel,
    coverage: Coverage,
) -> str:
    """One line a referral desk can act on without the rest of the report."""
    if reviewed != overall:
        parts = [f"{reviewed.value} risk level (clinician-reviewed; device {overall.value})"]
    elif overall == RiskLevel.INCONCLUSIVE:
        parts = ["INCONCLUSIVE screen"]
    else:
        parts = [f"{overall.value} risk level"]
    if lesions:
        top = lesions[0]
        text = f"Finding 1, {_lc(lb.site_name(top.site))} ({RiskLevel(top.risk_level).value})"
        if len(lesions) > 1:
            more = len(lesions) - 1
            text += f" + {more} more finding{'s' if more > 1 else ''}"
        parts.append(text)
    elif coverage.not_assessed:
        parts.append(f"{', '.join(coverage.not_assessed)} could not be assessed")
    else:
        parts.append("no suspicious finding at the examined sites")
    flags = [_lc(f.label) for f in result.triage.overrides_applied]
    if flags:
        parts.append(f"red flag{'s' if len(flags) > 1 else ''}: {'; '.join(flags)}")
    return " · ".join(parts)


def _models_line(result: ScreeningResult, rows: list[ModelRow]) -> str:
    models = result.versions.models
    validated = sum(1 for m in models if m.validated)
    head = (
        "all unvalidated"
        if validated == 0
        else f"{len(models) - validated} of {len(models)} unvalidated"
    )
    wl = next((r for r in rows if r.component == "White-light classifier"), None)
    wl_text = f" · White-light classifier: {wl.status_short}" if wl else ""
    return f"{head}{wl_text} (see §5)"


# --------------------------------------------------------------------------------------------
# Main builder
# --------------------------------------------------------------------------------------------


def build_context(
    result: ScreeningResult,
    artifacts: ArtifactReader,
    *,
    identity: PatientIdentity | None,
    options: ReportOptions,
) -> dict[str, Any]:
    tri = result.triage
    v = result.versions
    tz = display_timezone(options.display_tz)
    sites = {OralSite(s.site): s for s in result.sites}
    readings_by_id = {r.reading_id: r for r in result.spectral_readings}

    lesions = ordered_lesions(result)
    detailed = [
        _lesion_view(i, les, result, sites, readings_by_id, artifacts,
                     _heatmap_scale(_site_threshold(sites, les)))
        for i, les in enumerate(lesions[:MAX_DETAILED_LESIONS], start=1)
    ]  # fmt: skip
    compact = [
        CompactLesionRow(i, lb.site_name(les.site), badge(les.risk_level), lb.num(les.score, 1),
                         _origins_text(les), les.summary)
        for i, les in enumerate(lesions, start=1)
        if i > MAX_DETAILED_LESIONS
    ]  # fmt: skip

    linked = {rid for les in result.lesions for rid in les.spectral_reading_ids}
    linked |= {
        r.reference_reading_id
        for r in result.spectral_readings
        if r.reading_id in linked and r.reference_reading_id
    }
    unlinked = [r for r in result.spectral_readings if r.reading_id not in linked]

    scored = _scored_level(tri.score, tri.thresholds.moderate, tri.thresholds.high)
    overall = RiskLevel(tri.overall_risk_level)
    level_note = None
    if (
        tri.overrides_applied
        and scored is not None
        and overall != RiskLevel.INCONCLUSIVE
        and lb.SEVERITY_RANK[overall] > lb.SEVERITY_RANK[scored]
    ):
        level_note = (
            f"The rule score alone gives {lb.RISK_TITLE[scored].lower()}; a clinical red "
            f"flag raises the level to {lb.RISK_TITLE[overall].lower()}."
        )

    action = tri.recommended_action
    urgency = ReferralUrgency(action.referral_urgency)
    identity = identity or PatientIdentity()
    captured_local = _local(result.context.captured_at or result.created_at, tz)
    timeline = lb.timeline_text(action.timeline_days, start=captured_local)
    urgency_text = lb.URGENCY_TEXT.get(urgency, str(urgency))
    coverage = _coverage(result)
    reviewed_badge, reviewer_line, effective_level = _review_view(result, overall, tz)

    rule = v.rule_config
    footer = (
        f"MukhaSpectra {v.software} · pipeline {v.pipeline} · schema {v.schema_version} · "
        f"rules {rule.version} ({lb.short_sha(rule.sha256)}) · result v{result.result_version}"
    )
    versions_rows = [
        ("Result", f"v{result.result_version} {result.result_id}"),
        ("Screening", result.screening_id),
        ("Software", f"MukhaSpectra {v.software}"),
        ("Pipeline", v.pipeline),
        ("Schema", v.schema_version),
        ("Rules", f"{rule.name} {rule.version} (sha {lb.short_sha(rule.sha256)})"),
        ("QC config", f"{v.qc_config.version} ({lb.short_sha(v.qc_config.sha256)})"),
        ("Analysis config",
         f"{v.analysis_config.version} ({lb.short_sha(v.analysis_config.sha256)})"),
    ]  # fmt: skip

    model_rows = _ensure_wl_model_row(result, _model_rows(result))
    review = result.review
    facilities, more_facilities = _select_facilities(result, options, effective_level)
    single_facility = urgency in FACILITY_LEAD_IN

    numbers = lesion_numbers(result)
    factor_sources: dict[str, str] = {}
    if tri.highest_risk_lesion_id in numbers:
        factor_sources["lesion.max_score"] = f"Finding {numbers[tri.highest_risk_lesion_id]}"

    age = f"{result.patient.age_years} y" if result.patient.age_years is not None else lb.DASH
    sex = lb.SEX_TEXT.get(result.patient.sex.value, result.patient.sex.value)
    pid = result.patient.pseudonymous_id
    running_patient = " ".join(
        p
        for p in (
            f"{result.patient.age_years} y" if result.patient.age_years is not None else "",
            SEX_SHORT.get(result.patient.sex.value, ""),
        )
        if p
    )
    running_header = " · ".join(
        p for p in (f"Report {options.report_id}", pid, running_patient) if p
    )

    provenance = DataProvenance(result.data_provenance)
    provenance_notice = None
    provenance_tag = ""
    if not result.synthetic and provenance in PROVENANCE_NOTICE:
        provenance_tag, provenance_notice = PROVENANCE_NOTICE[provenance]

    clinician_note = None
    if overall == RiskLevel.INCONCLUSIVE:
        where = (
            ", ".join(_lc(s) for s in coverage.not_assessed)
            if coverage.not_assessed
            else "every site"
        )
        clinician_note = (
            f"If this patient reaches you: the device could not assess {where}; "
            "examine them clinically."
        )

    review_levels = [lv for lv in ("LOW", "MODERATE", "HIGH") if lv != overall.value]

    return {
        "r": result,
        "options": options,
        "synthetic": result.synthetic,
        "synthetic_notice": result.notices.synthetic_notice or SYNTHETIC_NOTICE,
        "provenance_notice": provenance_notice,
        "provenance_tag": provenance_tag,
        "banner": result.notices.prototype_banner,
        "disclaimer_title": SCREENING_AID_TITLE,
        "disclaimer_body": _disclaimer_body(result.notices.disclaimer),
        "footer_text": footer,
        "running_header": running_header,
        "header": {
            "report_id": options.report_id,
            "report_version": options.report_version,
            "generated_at": lb.fmt_datetime(_local(options.generated_at, tz)),
            "analysed_at": lb.fmt_datetime(_local(result.created_at, tz)),
            "captured_at": lb.fmt_datetime(captured_local),
            "result_id": result.result_id,
            "result_version": result.result_version,
            "supersedes": result.supersedes_result_id,
            "screening_id": result.screening_id,
            "camp": result.context.screening_site_name or result.context.camp_id or lb.DASH,
            "camp_id": result.context.camp_id,
            "health_worker": _health_worker(result),
            "devices": ", ".join(result.context.device_ids) or lb.DASH,
            "firmware": ", ".join(result.context.firmware_versions) or lb.DASH,
            "calibration": lb.CALIBRATION_TEXT.get(
                result.calibration_status, str(result.calibration_status)
            ),
            "calibration_ok": result.calibration_status == CalibrationStatus.CALIBRATED,
            "provenance": lb.PROVENANCE_TEXT.get(provenance.value, provenance.value),
            "review_status": (
                "Clinician-reviewed"
                if review is not None
                else ("Reviewed" if tri.review_status == "reviewed" else "Pending clinician review")
            ),
        },
        "versions_rows": versions_rows,
        "model_rows": model_rows,
        "models_line": _models_line(result, model_rows),
        "patient": {
            "name": identity.name,
            "external_id": lb.mask_id(identity.external_id),
            "pseudonymous_id": pid,
            "age": age,
            "sex": sex,
            "consent": _consent_text(result),
        },
        "risk_summary": lb.risk_factor_summary(result.risk_factors),
        "risk_groups": lb.risk_factor_groups(result.risk_factors),
        "triage": {
            "badge": badge(overall),
            "reviewed_badge": reviewed_badge,
            "reviewer_line": reviewer_line,
            "summary": action.summary,
            "steps": action.steps,
            "steps_label": (
                "At the screening visit:" if urgency == ReferralUrgency.RETAKE else None
            ),
            "clinician_note": clinician_note,
            "timeline": timeline,
            "urgency": urgency_text,
            "explanation": tri.explanation,
            "level_note": level_note,
            "factors": _factor_rows(
                tri.contributing_factors, tri.overrides_applied, sources=factor_sources
            ),
            "has_overrides": bool(tri.overrides_applied)
            or any(f.source == FactorSource.OVERRIDE for f in tri.contributing_factors),
            "score": lb.num(tri.score, 1) if tri.score is not None else None,
            "moderate": lb.num(tri.thresholds.moderate, 1),
            "high": lb.num(tri.thresholds.high, 1),
            "gauge": Markup(
                score_gauge_svg(
                    tri.score, tri.thresholds.moderate, tri.thresholds.high, overall, width=290
                )
            )
            if tri.score is not None
            else None,
            "rule_status": "Baseline heuristic, unvalidated",
        },
        "coverage": coverage,
        "override_tag": OVERRIDE_TAG,
        "override_footnote": OVERRIDE_FOOTNOTE,
        "sitemap_uri": svg_data_uri(render_site_map_svg(result, width=400, show_legend=False)),
        "map_legend": [
            {
                "badge": badge(lv),
                "fill": lb.RISK_COLORS[lv]["fill"],
                "stroke": lb.RISK_COLORS[lv]["strong"],
            }
            for lv in (RiskLevel.HIGH, RiskLevel.MODERATE, RiskLevel.LOW, RiskLevel.INCONCLUSIVE)
        ],
        "lesions": detailed,
        "compact_lesions": compact,
        "n_lesions": len(lesions),
        "max_detailed": MAX_DETAILED_LESIONS,
        "unlinked_spectral": [
            _spectral_view(r, readings_by_id) for r in unlinked if r.role == "lesion"
        ],
        "qc": {
            "passed_all": result.qc_summary.passed_all,
            "captures_failed": result.qc_summary.captures_failed,
            "sites_failed": [lb.site_name(s) for s in result.qc_summary.sites_failed],
            "retakes": result.qc_summary.retakes_recommended,
            "issues": _issue_rows(result),
            "sites": _site_rows(result),
            "calibration": lb.CALIBRATION_TEXT.get(
                result.calibration_status, str(result.calibration_status)
            ),
            "calibration_ok": result.calibration_status == CalibrationStatus.CALIBRATED,
        },
        "limitations": [lim.message for lim in result.limitations],
        "review": {
            "prefilled": review is not None,
            "reviewer": review.reviewer_name if review else None,
            "reviewer_id": review.reviewer_id if review else None,
            "agreed": review.agreed if review else None,
            "modified": review.modified_risk_level.value
            if review and review.modified_risk_level
            else None,
            "notes": review.notes if review else None,
            "signed_at": lb.fmt_datetime(_local(review.signed_at, tz)) if review else None,
            "signature_note": SIGNATURE_NOTE,
            "levels": review_levels,
            "current": overall.value,
        },
        "referral": {
            "urgency": urgency_text,
            "urgency_code": urgency.value,
            "timeline": timeline,
            "reviewer_line": reviewer_line,
            "reviewed_badge": reviewed_badge,
            "single": single_facility,
            "lead_in": FACILITY_LEAD_IN.get(urgency),
            "facilities": facilities[:1] if single_facility else facilities,
            "more_facilities": 0 if single_facility else more_facilities,
            "missing_text": FACILITY_MISSING,
            "slip_patient": " · ".join(
                p for p in (identity.name, pid, age if age != lb.DASH else "", sex) if p
            ),
            "slip_reason": _slip_reason(result, lesions, overall, effective_level, coverage),
            "programme": options.programme_contact,
            "record_url": options.record_url,
            "qr_uri": qr_data_uri(options.record_url) if options.record_url else None,
            "health_worker": _health_worker(result),
            "camp": result.context.screening_site_name or lb.DASH,
        },
    }


def _site_threshold(sites: dict[OralSite, SiteResult], lesion: LesionAssessment) -> float | None:
    site = sites.get(OralSite(lesion.site))
    return site.af.fvl_threshold if site is not None and site.af is not None else None


def _disclaimer_body(disclaimer: str) -> str:
    body = disclaimer.strip()
    prefix = SCREENING_AID_TITLE + "."
    if body.startswith(prefix):
        body = body[len(prefix) :].strip()
    return body


def _health_worker(result: ScreeningResult) -> str:
    ctx = result.context
    if ctx.health_worker_name and ctx.health_worker_id:
        return f"{ctx.health_worker_name} ({ctx.health_worker_id})"
    return ctx.health_worker_name or ctx.health_worker_id or lb.DASH


def _consent_text(result: ScreeningResult) -> str:
    c = result.consent
    if not c.consent_recorded:
        return "Not recorded"
    research = "research: opted in" if c.research_use_opt_in else "research: no"
    return f"Recorded · {research}"
