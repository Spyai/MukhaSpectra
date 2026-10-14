"""Build the template view model from a ScreeningResult.

All decisions about what to show (and how to word it) are made here in plain Python, so the
Jinja template stays declarative and every wording rule is unit-testable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from markupsafe import Markup
from mukha_pipeline.schema.models import (
    SYNTHETIC_NOTICE,
    CalibrationStatus,
    ContributingFactor,
    FactorSource,
    FVLRegion,
    LesionAssessment,
    LesionOrigin,
    ModelComponent,
    ModelStatus,
    NormBox,
    QCIssue,
    QCSeverity,
    RiskLevel,
    ScreeningResult,
    SiteResult,
    SiteRole,
    SpectralReadingResult,
    WLAnalysis,
)
from mukha_pipeline.sites import OralSite

from app.reports import labels as L
from app.reports.charts import risk_icon_svg, score_gauge_svg, spectral_plot_svg
from app.reports.images import EmbeddedImage, embed_artifact, svg_data_uri
from app.reports.qr import qr_data_uri
from app.reports.ranking import ordered_lesions
from app.reports.sitemap import render_site_map_svg
from app.reports.types import ArtifactReader, PatientIdentity, ReportOptions

MAX_DETAILED_LESIONS = 3
MAX_FACILITIES = 3
OVERRIDE_TAG = "Clinical red flag — the device cannot lower this"
SCREENING_AID_TITLE = "Screening aid — not a diagnosis"

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
class ImagePanel:
    title: str
    note: str
    image: EmbeddedImage


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
    visual: str | None
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


@dataclass
class ModelRow:
    component: str
    name: str
    version: str
    status: str
    validated: str
    css: str


# --------------------------------------------------------------------------------------------
# Small builders
# --------------------------------------------------------------------------------------------


def badge(level: RiskLevel | str) -> Badge:
    lv = RiskLevel(level)
    colors = L.RISK_COLORS[lv]
    inverse_color = colors["strong"] if lv == RiskLevel.INCONCLUSIVE else "#ffffff"
    return Badge(
        level=lv.value,
        title=L.RISK_TITLE[lv],
        css=lv.value.lower(),
        icon=Markup(risk_icon_svg(lv, colors["strong"])),
        icon_inverse=Markup(risk_icon_svg(lv, inverse_color)),
    )


def _factor_rows(
    factors: list[ContributingFactor], overrides: list[ContributingFactor] | None = None
) -> list[FactorRow]:
    override_codes = {o.code for o in overrides or []}
    rows: list[FactorRow] = []
    for f in overrides or []:
        rows.append(
            FactorRow(
                label=f.label,
                detail=f.detail,
                source=L.SOURCE_TEXT.get(f.source, str(f.source)),
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
                detail=f.detail,
                source=L.SOURCE_TEXT.get(f.source, str(f.source)),
                points="floor" if is_override else L.signed(f.points),
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
        return L.DASH
    ref = site.af.reference
    text = L.REFERENCE_TEXT.get(ref.type, str(ref.type))
    if ref.reference_site is not None:
        text += f" — {L.site_name(ref.reference_site)}"
    return text


def _region_row(site: SiteResult, region: FVLRegion) -> RegionRow:
    if region.area_mm2 is not None:
        area = f"{L.num(region.area_mm2, 1)} mm²"
        area_note: str | None = f"{L.pct(region.area_fraction_of_mucosa)} of mucosa"
    else:
        area = f"{L.pct(region.area_fraction_of_mucosa)} of mucosa"
        area_note = "mm scale unknown"
    return RegionRow(
        region_id=region.region_id,
        site=L.site_name(site.site),
        position=L.position_text(region.centroid),
        area=area,
        area_note=area_note,
        mean_rel=L.num(region.mean_relative_af),
        min_rel=L.num(region.min_relative_af),
        contrast=f"{L.num(region.contrast_vs_surround)}×",
        border=L.num(region.border_irregularity),
        reference=_reference_text(site),
        confounders=[
            f"{L.CONFOUNDER_TEXT.get(c.type, str(c.type))} {L.pct(c.overlap_fraction, 0)}"
            for c in region.confounders
        ],
    )


def _wl_view(site: SiteResult | None, reader: ArtifactReader) -> WLView:
    wl: WLAnalysis | None = site.wl if site is not None else None
    if wl is None:
        return WLView(
            "White-light classifier: not run for this site — not used in triage",
            False, [], L.DASH, None, None,
        )  # fmt: skip
    model_text = f"{wl.model.name} {wl.model.version} · {L.model_status_text(wl.model)}"
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
            predictions=[(L.WL_LABEL_TEXT.get(p.label, str(p.label)), L.num(p.probability))
                         for p in preds],
            model_text=model_text,
            note=wl.note,
            saliency=saliency,
        )
    if wl.status == "model_unavailable" or wl.model.status == ModelStatus.MODEL_UNAVAILABLE:
        text = "White-light classifier: not available — not used in triage"
    elif wl.status == "skipped_qc_failed":
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


def _spectral_view(
    reading: SpectralReadingResult, readings_by_id: dict[str, SpectralReadingResult]
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

    features: list[FeatureRow] = []
    if reading.features is not None:
        lf = reading.features.model_dump()
        rf = reference.features.model_dump() if reference and reference.features else {}
        df = reading.delta_vs_reference.model_dump() if reading.delta_vs_reference else {}
        for key, name in L.SPECTRAL_FEATURE_TEXT.items():
            delta = df.get(key)
            if delta is None and key in rf:
                delta = lf[key] - rf[key]
            features.append(
                FeatureRow(name, L.num(lf.get(key), 3), L.num(rf.get(key), 3),
                           L.signed(delta, 3) if delta is not None else L.DASH)
            )

    clf = reading.classifier
    qc_text = None
    if not reading.qc.passed:
        msgs = "; ".join(i.message for i in reading.qc.issues if i.severity == QCSeverity.ERROR)
        qc_text = f"Reading failed QC{': ' + msgs if msgs else ''}"
    cal = reading.calibration.status
    calibration_text = (
        "Calibrated against the white reference standard"
        if cal == CalibrationStatus.CALIBRATED
        else f"{L.CALIBRATION_TEXT.get(cal, str(cal))} — plotted as relative signal"
        if reading.signal_type != "reflectance"
        else L.CALIBRATION_TEXT.get(cal, str(cal))
    )
    return SpectralView(
        reading_id=reading.reading_id,
        plot_uri=svg_data_uri(spectral_plot_svg(reading, reference)),
        site=L.site_name(reading.site),
        reference_site=L.site_name(reference.site) if reference else None,
        signal_text="Reflectance" if reading.signal_type == "reflectance" else "Relative signal",
        calibration_text=calibration_text,
        qc_text=qc_text,
        features=features,
        classifier_label=(
            L.SPECTRAL_LABEL_TEXT.get(clf.label, clf.label) if clf and clf.label else None
        ),
        classifier_status=_classifier_status(clf.status) if clf else "Not run",
        classifier_score=(
            L.num(clf.suspicion_score) if clf and clf.suspicion_score is not None else None
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
) -> LesionView:
    site = sites.get(OralSite(lesion.site))
    regions: list[FVLRegion] = []
    if site is not None and site.af is not None:
        wanted = set(lesion.region_ids)
        regions = [r for r in site.af.regions if r.region_id in wanted]
    boxes: list[NormBox] = [r.bbox for r in regions]

    arts = site.artifacts if site is not None else None
    af_key = None
    af_note = "405 nm excitation"
    af_in_wl_frame = False
    if arts is not None:
        if arts.af_registered:
            af_key, af_in_wl_frame = arts.af_registered, True
            af_note = "405 nm excitation, registered to white light"
        elif arts.af_image:
            af_key = arts.af_image
            af_note = "405 nm excitation (unregistered)"
    panels = [
        ImagePanel("White light", "As captured", embed_artifact(
            reader, arts.wl_image if arts else None, boxes=boxes)),
        ImagePanel("Autofluorescence", af_note, embed_artifact(
            reader, af_key, boxes=boxes if af_in_wl_frame else None)),
        ImagePanel("AF-loss heatmap", "Overlay on white light", embed_artifact(
            reader, arts.heatmap_overlay if arts else None, boxes=boxes)),
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
            f"{L.CONFOUNDER_TEXT.get(c.type, str(c.type))} "
            f"({L.pct(c.overlap_fraction, 0)} overlap): {c.note}"
        )

    visual = None
    if site is not None and site.visual_flag.lesion_seen:
        appearance = ", ".join(a.value.replace("_", " ") for a in site.visual_flag.appearance)
        visual = f"Seen by health worker: {appearance or 'lesion'}"
        if site.visual_flag.description:
            visual += f" — “{site.visual_flag.description}”"

    spectral_ids = list(lesion.spectral_reading_ids)
    primaries = [readings_by_id[i] for i in spectral_ids
                 if i in readings_by_id and readings_by_id[i].role == "lesion"]
    if not primaries:
        primaries = [readings_by_id[i] for i in spectral_ids if i in readings_by_id]

    return LesionView(
        number=number,
        lesion_id=lesion.lesion_id,
        site=L.site_name(lesion.site),
        badge=badge(lesion.risk_level),
        origins=_origins_text(lesion),
        summary=lesion.summary,
        action=lesion.recommended_action.summary,
        panels=panels,
        marks_drawn=bool(boxes),
        regions=[_region_row(site, r) for r in regions] if site is not None else [],
        fvl_threshold=L.num(site.af.fvl_threshold) if site and site.af else None,
        region_note=region_note,
        confounder_notes=confounder_notes,
        visual=visual,
        diascopy=L.DIASCOPY_TEXT.get(site.diascopy, str(site.diascopy)) if site else L.DASH,
        spectral=[_spectral_view(r, readings_by_id) for r in primaries],
        wl=_wl_view(site, reader),
        factors=_factor_rows(lesion.contributing_factors),
        score=L.num(lesion.score, 1),
    )


def _origins_text(lesion: LesionAssessment) -> str:
    return ", ".join(L.ORIGIN_TEXT.get(o, str(o)) for o in lesion.origins) or L.DASH


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
            IssueRow(site_label, capture, "Retake needed" if error else "Warning",
                     "error" if error else "warning", issue.message)
        )

    for site in result.sites:
        label = L.site_name(site.site)
        for cap in site.captures:
            kind = L.CAPTURE_KIND_TEXT.get(cap.metadata.kind.value, cap.metadata.kind.value)
            for issue in cap.qc.issues:
                add(label, kind, issue)
        for issue in site.qc.issues:
            add(label, "Site analysis", issue)
    for reading in result.spectral_readings:
        for issue in reading.qc.issues:
            add(L.site_name(reading.site), "Spectral reading", issue)
    for issue in result.qc_summary.issues:
        if not any(r.message == issue.message for r in rows):
            add(L.DASH, L.DASH, issue)
    rows.sort(key=lambda r: (r.css != "error", r.site))
    return rows


def _site_rows(result: ScreeningResult) -> list[SiteRow]:
    order = {s: i for i, s in enumerate(OralSite)}
    rows = []
    for site in sorted(result.sites, key=lambda s: (s.role != SiteRole.EXAMINED, order[s.site])):
        cal = site.calibration.status
        rows.append(
            SiteRow(
                site=L.site_name(site.site),
                role="Examined" if site.role == SiteRole.EXAMINED else "Reference only",
                qc=("Passed" if site.qc.passed else "Failed") + f" · {L.num(site.qc.score)}",
                qc_ok=site.qc.passed,
                analysis=L.ANALYSIS_STATUS_TEXT.get(site.analysis_status, site.analysis_status),
                calibration=L.CALIBRATION_TEXT.get(cal, str(cal)),
                calibration_ok=cal == CalibrationStatus.CALIBRATED,
                badge=badge(site.site_risk_level),
            )
        )
    return rows


def _model_rows(result: ScreeningResult) -> list[ModelRow]:
    order = list(ModelComponent)
    rows = []
    for m in sorted(result.versions.models, key=lambda m: order.index(m.component)):
        css = "muted" if m.status == ModelStatus.MODEL_UNAVAILABLE else "caution"
        rows.append(
            ModelRow(
                component=L.COMPONENT_TEXT.get(m.component, str(m.component)),
                name=m.name,
                version=m.version,
                status=L.model_status_text(m),
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
        ModelRow("White-light classifier", L.DASH, L.DASH,
                 "Not available — not used in triage", "No", "muted"),
        *rows,
    ]  # fmt: skip


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
    sites = {OralSite(s.site): s for s in result.sites}
    readings_by_id = {r.reading_id: r for r in result.spectral_readings}

    lesions = ordered_lesions(result)
    detailed = [
        _lesion_view(i, les, result, sites, readings_by_id, artifacts)
        for i, les in enumerate(lesions[:MAX_DETAILED_LESIONS], start=1)
    ]
    compact = [
        CompactLesionRow(i, L.site_name(les.site), badge(les.risk_level), L.num(les.score, 1),
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
    if tri.overrides_applied and scored is not None and overall != RiskLevel.INCONCLUSIVE:
        if L.SEVERITY_RANK[overall] > L.SEVERITY_RANK[scored]:
            level_note = (
                f"The rule score alone gives {L.RISK_TITLE[scored].lower()}; a clinical red "
                f"flag raises the level to {L.RISK_TITLE[overall].lower()}."
            )

    action = tri.recommended_action
    identity = identity or PatientIdentity()

    rule = v.rule_config
    footer = (
        f"MukhaSpectra {v.software} · pipeline {v.pipeline} · schema {v.schema_version} · "
        f"rules {rule.version} ({L.short_sha(rule.sha256)}) · result v{result.result_version}"
    )
    versions_rows = [
        ("Software", f"MukhaSpectra {v.software}"),
        ("Pipeline / schema", f"{v.pipeline} / {v.schema_version}"),
        ("Rule config", f"{rule.name} {rule.version} · sha {L.short_sha(rule.sha256)}"),
        ("QC config", f"{v.qc_config.version} · sha {L.short_sha(v.qc_config.sha256)}"),
        ("Analysis config",
         f"{v.analysis_config.version} · sha {L.short_sha(v.analysis_config.sha256)}"),
    ]  # fmt: skip

    model_rows = _ensure_wl_model_row(result, _model_rows(result))
    review = result.review
    facilities = options.facilities[:MAX_FACILITIES]

    return {
        "r": result,
        "options": options,
        "synthetic": result.synthetic,
        "synthetic_notice": result.notices.synthetic_notice or SYNTHETIC_NOTICE,
        "banner": result.notices.prototype_banner,
        "disclaimer_title": SCREENING_AID_TITLE,
        "disclaimer_body": _disclaimer_body(result.notices.disclaimer),
        "footer_text": footer,
        "header": {
            "report_id": options.report_id,
            "report_version": options.report_version,
            "generated_at": L.fmt_datetime(options.generated_at),
            "analysed_at": L.fmt_datetime(result.created_at),
            "captured_at": L.fmt_datetime(result.context.captured_at),
            "result_id": result.result_id,
            "result_version": result.result_version,
            "supersedes": result.supersedes_result_id,
            "screening_id": result.screening_id,
            "camp": result.context.screening_site_name or result.context.camp_id or L.DASH,
            "camp_id": result.context.camp_id,
            "health_worker": _health_worker(result),
            "devices": ", ".join(result.context.device_ids) or L.DASH,
            "firmware": ", ".join(result.context.firmware_versions) or L.DASH,
            "calibration": L.CALIBRATION_TEXT.get(
                result.calibration_status, str(result.calibration_status)
            ),
            "calibration_ok": result.calibration_status == CalibrationStatus.CALIBRATED,
            "provenance": L.PROVENANCE_TEXT.get(
                result.data_provenance.value, result.data_provenance.value
            ),
            "review_status": (
                "Reviewed" if tri.review_status == "reviewed" or review else
                "Pending clinician review"
            ),
        },
        "versions_rows": versions_rows,
        "model_rows": model_rows,
        "patient": {
            "name": identity.name,
            "external_id": identity.external_id,
            "pseudonymous_id": result.patient.pseudonymous_id,
            "age": f"{result.patient.age_years} y" if result.patient.age_years is not None
            else L.DASH,
            "sex": L.SEX_TEXT.get(result.patient.sex.value, result.patient.sex.value),
            "consent": _consent_text(result),
        },
        "risk_summary": L.risk_factor_summary(result.risk_factors),
        "risk_groups": L.risk_factor_groups(result.risk_factors),
        "triage": {
            "badge": badge(overall),
            "summary": action.summary,
            "steps": action.steps,
            "timeline": L.timeline_text(action.timeline_days),
            "urgency": L.URGENCY_TEXT.get(action.referral_urgency, str(action.referral_urgency)),
            "explanation": tri.explanation,
            "level_note": level_note,
            "factors": _factor_rows(tri.contributing_factors, tri.overrides_applied),
            "has_overrides": bool(tri.overrides_applied),
            "score": L.num(tri.score, 1) if tri.score is not None else None,
            "moderate": L.num(tri.thresholds.moderate, 1),
            "high": L.num(tri.thresholds.high, 1),
            "gauge": Markup(
                score_gauge_svg(tri.score, tri.thresholds.moderate, tri.thresholds.high,
                                overall, width=330)
            ) if tri.score is not None else None,
            "rule_status": "Baseline heuristic, unvalidated",
        },
        "override_tag": OVERRIDE_TAG,
        "sitemap_uri": svg_data_uri(render_site_map_svg(result, width=400, show_legend=True)),
        "lesions": detailed,
        "compact_lesions": compact,
        "n_lesions": len(lesions),
        "max_detailed": MAX_DETAILED_LESIONS,
        "unlinked_spectral": [_spectral_view(r, readings_by_id) for r in unlinked
                              if r.role == "lesion"],
        "qc": {
            "passed_all": result.qc_summary.passed_all,
            "captures_failed": result.qc_summary.captures_failed,
            "sites_failed": [L.site_name(s) for s in result.qc_summary.sites_failed],
            "retakes": result.qc_summary.retakes_recommended,
            "issues": _issue_rows(result),
            "sites": _site_rows(result),
            "calibration": L.CALIBRATION_TEXT.get(
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
            if review and review.modified_risk_level else None,
            "notes": review.notes if review else None,
            "signed_at": L.fmt_datetime(review.signed_at) if review else None,
            "levels": ["LOW", "MODERATE", "HIGH"],
            "current": overall.value,
        },
        "referral": {
            "urgency": L.URGENCY_TEXT.get(action.referral_urgency, str(action.referral_urgency)),
            "urgency_code": action.referral_urgency.value,
            "timeline": L.timeline_text(action.timeline_days),
            "facilities": facilities,
            "more_facilities": max(0, len(options.facilities) - MAX_FACILITIES),
            "record_url": options.record_url,
            "qr_uri": qr_data_uri(options.record_url) if options.record_url else None,
            "health_worker": _health_worker(result),
            "camp": result.context.screening_site_name or L.DASH,
        },
    }


def _disclaimer_body(disclaimer: str) -> str:
    body = disclaimer.strip()
    prefix = SCREENING_AID_TITLE + "."
    if body.startswith(prefix):
        body = body[len(prefix):].strip()
    return body


def _health_worker(result: ScreeningResult) -> str:
    ctx = result.context
    if ctx.health_worker_name and ctx.health_worker_id:
        return f"{ctx.health_worker_name} ({ctx.health_worker_id})"
    return ctx.health_worker_name or ctx.health_worker_id or L.DASH


def _consent_text(result: ScreeningResult) -> str:
    c = result.consent
    if not c.consent_recorded:
        return "Not recorded"
    text = f"Recorded ({c.method}, {c.language.value.upper()})"
    text += " · research use: opted in" if c.research_use_opt_in else " · research use: no"
    return text
