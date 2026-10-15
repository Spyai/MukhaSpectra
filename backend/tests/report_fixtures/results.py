"""Realistic ScreeningResult fixtures built through the pydantic models (SYNTHETIC by default).

Cases
-----
* ``high``: HIGH, two findings (left buccal white patch with AF loss + spectral pair; left
  lateral tongue AF loss with a porphyrin confounder), WL model unavailable, right lateral
  tongue failed QC (blur), non-healing ulcer red flag.
* ``low``: LOW, all sites analysed, no findings, khaini user (counselling only).
* ``inconclusive``: INCONCLUSIVE, uncalibrated device, left buccal mucosa failed QC (weak AF
  signal, glare), no findings, no rule score.
* ``stress``: the HIGH case pushed to the layout limits: five findings, white-light classifier
  available, a clinician review that changes the level to MODERATE, long camp / health-worker /
  patient names and an ABHA-style external ID (identity carried on the case).

``synthetic=False`` exists only so tests can check the watermark is absent for non-synthetic
results; such variants are labelled "phantom" provenance and are never written to disk.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta, timezone

from mukha_pipeline.config import PipelineConfig
from mukha_pipeline.schema.models import (
    SYNTHETIC_NOTICE,
    AFAnalysis,
    AFReference,
    AFSiteMedians,
    AlcoholUse,
    CalibrationInfo,
    CalibrationStatus,
    CaptureKind,
    CaptureMetadata,
    CaptureRecord,
    ClinicianReview,
    ConfounderFlag,
    ConfounderType,
    ConsentInfo,
    ContributingFactor,
    DataProvenance,
    Diascopy,
    FactorSource,
    FVLRegion,
    LedMode,
    LesionAssessment,
    LesionOrigin,
    Limitation,
    ModelComponent,
    ModelInfo,
    ModelStatus,
    NormBox,
    NormPoint,
    Notices,
    PatientInfo,
    ProcessingInfo,
    QCIssue,
    QCResult,
    QCSeverity,
    QCSummary,
    RecommendedAction,
    ReferenceType,
    ReferralUrgency,
    RegistrationSummary,
    RiskFactors,
    RiskLevel,
    ScoreThresholds,
    ScreeningContext,
    ScreeningResult,
    SegmentationSummary,
    Sex,
    SiteArtifacts,
    SiteResult,
    SiteRole,
    SmokelessTobaccoUse,
    SmokingUse,
    SourceAdapter,
    SpectralClassifierOutput,
    SpectralFeatures,
    SpectralReadingResult,
    StageTiming,
    Symptoms,
    TriageResult,
    VersionInfo,
    VisualAppearance,
    VisualFlag,
    WLAnalysis,
    WLLabel,
    WLPrediction,
    WorkingResolution,
)
from mukha_pipeline.sites import OralSite, reference_site_for, site_label
from report_fixtures.images import LesionDraw, site_artifacts

IST = timezone(timedelta(hours=5, minutes=30), "IST")
CAPTURED_AT = datetime(2026, 10, 14, 4, 5, tzinfo=UTC)
ANALYSED_AT = datetime(2026, 10, 14, 4, 9, tzinfo=UTC)
GENERATED_AT = datetime(2026, 10, 14, 9, 45, tzinfo=IST)

SCOUT = "scout-dev-01"
CONFIRM = "confirm-dev-01"


@dataclass
class ReportCase:
    name: str
    result: ScreeningResult
    artifacts: dict[str, bytes] = field(default_factory=dict)
    identity_name: str | None = None  # PII the API would join at render time (stress case)
    identity_external_id: str | None = None

    def reader(self, key: str) -> bytes | None:
        return self.artifacts.get(key)


# --------------------------------------------------------------------------------------------
# Shared pieces
# --------------------------------------------------------------------------------------------


def _models() -> list[ModelInfo]:
    triage = PipelineConfig.default().triage.ref  # the fusion entry mirrors the bundled rules
    return [
        ModelInfo(component=ModelComponent.SEGMENTER, name="classical_hsv_lab", version="1.0.0",
                  status=ModelStatus.BASELINE_RULES),
        ModelInfo(component=ModelComponent.WL_LESION_CLASSIFIER, name="wl_lesion_classifier",
                  version="none", status=ModelStatus.MODEL_UNAVAILABLE,
                  notes="No trained weights in this build."),
        ModelInfo(component=ModelComponent.SPECTRAL_CLASSIFIER, name="spectral-rules",
                  version="2.0.0", status=ModelStatus.PLACEHOLDER),
        ModelInfo(component=ModelComponent.FUSION, name=triage.name,
                  version=triage.version, status=ModelStatus.BASELINE_RULES),
    ]  # fmt: skip


def _versions() -> VersionInfo:
    cfg = PipelineConfig.default()
    return VersionInfo(
        software="0.1.0",
        pipeline="0.1.0",
        rule_config=cfg.triage.ref,
        qc_config=cfg.qc.ref,
        analysis_config=cfg.analysis.ref,
        models=_models(),
    )


def _wl_unavailable() -> WLAnalysis:
    model = _models()[1]
    return WLAnalysis(
        status="model_unavailable",
        model=model,
        note="No trained white-light weights; not used in triage.",
    )


def _qc(passed: bool = True, score: float = 0.93, issues: list[QCIssue] | None = None) -> QCResult:
    return QCResult(passed=passed, score=score, issues=issues or [])


def _cal(status: CalibrationStatus, device: str = SCOUT) -> CalibrationInfo:
    if status == CalibrationStatus.CALIBRATED:
        return CalibrationInfo(
            status=status,
            calibration_id=f"cal-{device}-20261014",
            expires_at=CAPTURED_AT + timedelta(hours=20),
        )
    return CalibrationInfo(status=status, note="No calibration record for this device today.")


def _capture(
    site: OralSite,
    kind: CaptureKind,
    *,
    synthetic: bool,
    qc: QCResult,
    artifact: str | None,
    calibrated: bool = True,
) -> CaptureRecord:
    led = {
        CaptureKind.WHITE_LIGHT: LedMode.WHITE,
        CaptureKind.AUTOFLUORESCENCE: LedMode.VIOLET,
        CaptureKind.DARK_FRAME: LedMode.OFF,
    }.get(kind)
    meta = CaptureMetadata(
        capture_id=f"cap-{site.value}-{kind.value}",
        kind=kind,
        oral_site=site,
        source_adapter=SourceAdapter.SYNTHETIC if synthetic else SourceAdapter.UPLOAD,
        device_id=SCOUT,
        firmware_version="scout 0.1.0",
        led_mode=led,
        exposure_time_ms=33.0 if kind == CaptureKind.WHITE_LIGHT else 120.0,
        iso=200.0 if kind == CaptureKind.WHITE_LIGHT else 800.0,
        white_balance="locked:5000K",
        captured_at=CAPTURED_AT,
        calibration_id=f"cal-{SCOUT}-20261014" if calibrated else None,
        width_px=1024,
        height_px=768,
        mime_type="image/png",
        synthetic=synthetic,
    )
    return CaptureRecord(metadata=meta, qc=qc, artifact=artifact)


def _af(
    reference: AFReference, regions: list[FVLRegion], *, status: str = "ok", green: float = 0.56
) -> AFAnalysis:
    fvl_frac = sum(r.area_fraction_of_mucosa for r in regions)
    return AFAnalysis(
        status=status,  # type: ignore[arg-type]
        reference=reference,
        fvl_threshold=0.75,
        min_region_area_px=150,
        site_medians=AFSiteMedians(
            green=green, gr_ratio=4.6, af_index=0.71, relative_af=0.97 if not regions else 0.88
        ),
        fvl_area_fraction=min(1.0, fvl_frac),
        porphyrin_red_fraction=0.04 if any(r.confounders for r in regions) else 0.0,
        regions=regions,
    )


def _site(
    site: OralSite,
    level: RiskLevel,
    *,
    synthetic: bool,
    role: SiteRole = SiteRole.EXAMINED,
    af: AFAnalysis | None = None,
    visual: VisualFlag | None = None,
    diascopy: Diascopy = Diascopy.NOT_DONE,
    qc_issues: list[QCIssue] | None = None,
    calibration: CalibrationStatus = CalibrationStatus.CALIBRATED,
    spectral_ids: list[str] | None = None,
    px_per_mm: float | None = None,
) -> SiteResult:
    failed = bool(qc_issues and any(i.severity == QCSeverity.ERROR for i in qc_issues))
    wl_issues = [i for i in qc_issues or [] if not i.code.startswith("af_")]
    af_issues = [i for i in qc_issues or [] if i.code.startswith("af_")]
    base = f"sites/{site.value}"
    captures = [
        _capture(site, CaptureKind.WHITE_LIGHT, synthetic=synthetic,
                 qc=_qc(not any(i.severity == QCSeverity.ERROR for i in wl_issues),
                        0.41 if wl_issues else 0.94, wl_issues),
                 artifact=f"{base}/wl.png", calibrated=calibration == CalibrationStatus.CALIBRATED),
        _capture(site, CaptureKind.AUTOFLUORESCENCE, synthetic=synthetic,
                 qc=_qc(not any(i.severity == QCSeverity.ERROR for i in af_issues),
                        0.38 if af_issues else 0.9, af_issues),
                 artifact=f"{base}/af.png", calibrated=calibration == CalibrationStatus.CALIBRATED),
    ]  # fmt: skip
    if failed:
        artifacts = SiteArtifacts(wl_image=f"{base}/wl.png", af_image=f"{base}/af.png")
    else:
        artifacts = SiteArtifacts(
            wl_image=f"{base}/wl.png",
            af_image=f"{base}/af.png",
            af_registered=f"{base}/af_registered.png",
            heatmap_overlay=f"{base}/heatmap_overlay.png",
            relative_af_map=f"{base}/relative_af.png",
            relative_af_grid=f"{base}/relative_af_grid.json",
            mucosa_mask=f"{base}/mucosa_mask.png",
        )
    paired, pairing = reference_site_for(site)
    return SiteResult(
        site=site,
        site_label=site_label(site),
        role=role,
        paired_site=paired,
        pairing=pairing,
        visual_flag=visual or VisualFlag(),
        diascopy=diascopy,
        captures=captures,
        qc=_qc(not failed, 0.4 if failed else 0.92, list(qc_issues or [])),
        analysis_status="inconclusive_qc_failed" if failed else "analysed",
        calibration=_cal(calibration),
        registration=None if failed else RegistrationSummary(
            method="ecc_affine", confidence=0.93, passed=True,
            warp_matrix=[[1.0, 0.004, -3.1], [-0.003, 1.0, 2.4], [0.0, 0.0, 1.0]]),
        segmentation=None if failed else SegmentationSummary(
            method="classical_hsv_lab", version="1.0.0", status="ok", mucosa_fraction=0.61,
            teeth_fraction=0.04, specular_fraction=0.006, confidence=0.82),
        af=af if not failed else AFAnalysis(
            status="skipped_qc_failed",
            reference=AFReference(type=ReferenceType.NONE, note="Capture failed QC."),
            fvl_threshold=0.75, min_region_area_px=150),
        wl=_wl_unavailable(),
        spectral_reading_ids=spectral_ids or [],
        artifacts=artifacts,
        working_resolution=WorkingResolution(width_px=1024, height_px=768,
                                             scale_from_original=0.25),
        px_per_mm=px_per_mm,
        site_risk_level=level,
    )  # fmt: skip


def _action(level: RiskLevel) -> RecommendedAction:
    rules = PipelineConfig.default().triage.data["actions"][level.value]
    return RecommendedAction(
        code=rules["code"],
        summary=rules["summary"],
        steps=list(rules["steps"]),
        timeline_days=rules["timeline_days"],
        referral_urgency=ReferralUrgency(rules["referral_urgency"]),
    )


def _thresholds() -> ScoreThresholds:
    levels = PipelineConfig.default().triage.data["levels"]
    return ScoreThresholds(moderate=levels["moderate_min_score"], high=levels["high_min_score"])


def _consent() -> ConsentInfo:
    return ConsentInfo(
        consent_recorded=True,
        method="checkbox",
        research_use_opt_in=False,
        consent_id="consent-0001",
        recorded_at=CAPTURED_AT - timedelta(minutes=6),
    )


def _context(camp: str, *, devices: list[str]) -> ScreeningContext:
    return ScreeningContext(
        screening_site_name=camp,
        camp_id="camp-hsk-2026-10",
        health_worker_id="hw-17",
        health_worker_name="CHW Lakshmi R.",
        device_ids=devices,
        firmware_versions=[f"{d.split('-')[0]} 0.1.0" for d in devices],
        captured_at=CAPTURED_AT,
    )


def _provenance(synthetic: bool) -> tuple[DataProvenance, Notices]:
    if synthetic:
        return DataProvenance.SYNTHETIC, Notices(synthetic_notice=SYNTHETIC_NOTICE)
    return DataProvenance.PHANTOM, Notices()


def _processing() -> ProcessingInfo:
    return ProcessingInfo(working_max_side_px=1024, duration_ms=1840.0, stages=[
        StageTiming(stage="qc", duration_ms=120.0), StageTiming(stage="registration",
        duration_ms=410.0), StageTiming(stage="af_analysis", duration_ms=380.0),
        StageTiming(stage="fusion", duration_ms=4.0)])  # fmt: skip


def _f(
    code: str, label: str, source: FactorSource, points: float, detail: str | None = None
) -> ContributingFactor:
    return ContributingFactor(code=code, label=label, source=source, points=points, detail=detail)


# --------------------------------------------------------------------------------------------
# Spectral pair for the HIGH case
# --------------------------------------------------------------------------------------------

_REF_VALUES = {"F1_415": 0.182, "F2_445": 0.241, "F3_480": 0.302, "F4_515": 0.338,
               "F5_555": 0.297, "F6_590": 0.418, "F7_630": 0.552, "F8_680": 0.603,
               "CLEAR": 0.401, "NIR": 0.622}  # fmt: skip
_LES_VALUES = {"F1_415": 0.151, "F2_445": 0.203, "F3_480": 0.268, "F4_515": 0.309,
               "F5_555": 0.238, "F6_590": 0.362, "F7_630": 0.529, "F8_680": 0.588,
               "CLEAR": 0.362, "NIR": 0.611}  # fmt: skip


def _features(v: dict[str, float]) -> SpectralFeatures:
    import math

    base_555 = v["F4_515"] + (v["F7_630"] - v["F4_515"]) * (555 - 515) / (630 - 515)
    return SpectralFeatures(
        ratio_555_630=v["F5_555"] / v["F7_630"],
        ratio_590_680=v["F6_590"] / v["F8_680"],
        ratio_515_630=v["F4_515"] / v["F7_630"],
        ratio_415_555=v["F1_415"] / v["F5_555"],
        hb_absorption_proxy=1 - v["F5_555"] / base_555,
        slope_480_555=(math.log(v["F5_555"]) - math.log(v["F3_480"])) / 0.75,
        slope_590_680=(math.log(v["F8_680"]) - math.log(v["F6_590"])) / 0.9,
    )


def _delta(lesion: dict[str, float], reference: dict[str, float]) -> SpectralFeatures:
    """The pipeline's own (quotient-based, probe-independent) lesion-vs-reference delta."""
    from mukha_pipeline.spectral.features import reference_delta

    delta = reference_delta(lesion, reference)
    assert delta is not None
    return delta


def _spectral_pair(synthetic: bool) -> list[SpectralReadingResult]:
    def meta(rid: str, site: OralSite) -> CaptureMetadata:
        return CaptureMetadata(
            capture_id=f"cap-{rid}", kind=CaptureKind.SPECTRAL, oral_site=site,
            source_adapter=SourceAdapter.SYNTHETIC if synthetic else SourceAdapter.UPLOAD,
            device_id=CONFIRM, firmware_version="confirm 0.1.0", captured_at=CAPTURED_AT,
            calibration_id="cal-confirm-dev-01-20261014", synthetic=synthetic)  # fmt: skip

    cal = _cal(CalibrationStatus.CALIBRATED, CONFIRM)
    std = {k: round(0.012 + 0.02 * v, 4) for k, v in _REF_VALUES.items()}
    ref_f, les_f = _features(_REF_VALUES), _features(_LES_VALUES)
    delta = _delta(_LES_VALUES, _REF_VALUES)
    placeholder_model = _models()[2]
    reasons = [
        f"Haemoglobin dip deeper than the reference (Δ {delta.hb_absorption_proxy:+.3f})",
        f"555/630 nm ratio lower than the reference ({delta.ratio_555_630:+.1%})",
    ]
    lesion = SpectralReadingResult(
        reading_id="s1", site=OralSite.BUCCAL_MUCOSA_LEFT, role="lesion", lesion_ref="r1",
        capture=meta("s1", OralSite.BUCCAL_MUCOSA_LEFT), n_readings=5,
        signal_type="reflectance", calibration=cal, values=_LES_VALUES,
        values_std={k: s * 1.3 for k, s in std.items()}, qc=_qc(True, 0.9),
        features=les_f, reference_reading_id="s2", delta_vs_reference=delta,
        classifier=SpectralClassifierOutput(
            status="placeholder", method="rules_v2", suspicion_score=0.68, label="suspicious",
            reasons=reasons,
            model=placeholder_model),
    )  # fmt: skip
    reference = SpectralReadingResult(
        reading_id="s2", site=OralSite.BUCCAL_MUCOSA_RIGHT, role="reference",
        capture=meta("s2", OralSite.BUCCAL_MUCOSA_RIGHT), n_readings=5,
        signal_type="reflectance", calibration=cal, values=_REF_VALUES, values_std=std,
        qc=_qc(True, 0.92), features=ref_f,
    )  # fmt: skip
    return [lesion, reference]


# --------------------------------------------------------------------------------------------
# Cases
# --------------------------------------------------------------------------------------------

L1_BOX = NormBox(x=0.36, y=0.42, w=0.22, h=0.24)
L2_BOX = NormBox(x=0.52, y=0.30, w=0.20, h=0.30)


def build_high(synthetic: bool = True) -> ReportCase:
    provenance, notices = _provenance(synthetic)
    S = OralSite
    blur = QCIssue(
        code="blur",
        severity=QCSeverity.ERROR,
        message="Image blurry — hold the device steady and retake.",
        message_key="qc.blur",
        metric="laplacian_var",
        value=12.4,
        threshold=40.0,
    )

    r1 = FVLRegion(
        region_id="r1",
        area_px=5120,
        area_mm2=38.6,
        area_fraction_of_mucosa=0.072,
        centroid=NormPoint(x=0.47, y=0.54),
        bbox=L1_BOX,
        mean_relative_af=0.52,
        min_relative_af=0.38,
        contrast_vs_surround=1.71,
        border_irregularity=0.46,
        solidity=0.81,
    )
    r2 = FVLRegion(
        region_id="r1", area_px=9800, area_fraction_of_mucosa=0.16,
        centroid=NormPoint(x=0.62, y=0.45), bbox=L2_BOX, mean_relative_af=0.47,
        min_relative_af=0.36, contrast_vs_surround=1.52, border_irregularity=0.38, solidity=0.77,
        confounders=[ConfounderFlag(type=ConfounderType.PORPHYRIN_RED, overlap_fraction=0.34,
                                    note="red porphyrin fluorescence (plaque or bacteria) can "
                                         "mimic AF loss; points discounted")])  # fmt: skip

    contra = AFReference(
        type=ReferenceType.CONTRALATERAL, reference_site=S.BUCCAL_MUCOSA_RIGHT, reference_green=0.57
    )
    in_image = AFReference(
        type=ReferenceType.IN_IMAGE_AUTO,
        reference_green=0.55,
        note="Contralateral site failed QC; in-image reference used.",
    )
    sites = [
        _site(S.BUCCAL_MUCOSA_LEFT, RiskLevel.HIGH, synthetic=synthetic, af=_af(contra, [r1]),
              visual=VisualFlag(lesion_seen=True, appearance=[VisualAppearance.WHITE_PATCH],
                                description="Thick white patch, does not scrape off"),
              diascopy=Diascopy.FVL_PERSISTS, spectral_ids=["s1"], px_per_mm=11.5),
        _site(S.BUCCAL_MUCOSA_RIGHT, RiskLevel.LOW, synthetic=synthetic,
              role=SiteRole.REFERENCE_ONLY,
              af=_af(AFReference(type=ReferenceType.IN_IMAGE_AUTO, reference_green=0.57), []),
              spectral_ids=["s2"]),
        _site(S.LATERAL_TONGUE_LEFT, RiskLevel.MODERATE, synthetic=synthetic,
              af=_af(in_image, [r2])),
        _site(S.LATERAL_TONGUE_RIGHT, RiskLevel.INCONCLUSIVE, synthetic=synthetic,
              qc_issues=[blur]),
        _site(S.TONGUE_DORSUM, RiskLevel.LOW, synthetic=synthetic,
              af=_af(AFReference(type=ReferenceType.IN_IMAGE_AUTO, reference_green=0.53), [])),
    ]  # fmt: skip

    les1_factors = [
        _f("af.fvl_present", "AF-loss region present", FactorSource.AUTOFLUORESCENCE, 1.5),
        _f("af.depth", "Mean relative AF 0.52 (0.23 below threshold 0.75)",
           FactorSource.AUTOFLUORESCENCE, 1.7),
        _f("af.area", "Area 38.6 mm² (≥ 10 mm²)", FactorSource.AUTOFLUORESCENCE, 0.5),
        _f("af.border", "Irregular border (0.46 ≥ 0.35)", FactorSource.AUTOFLUORESCENCE, 0.5),
        _f("af.contrast", "Darker than surroundings (1.71× ≥ 1.4×)",
           FactorSource.AUTOFLUORESCENCE, 0.5),
        _f("af.diascopy", "AF loss persists under diascopy", FactorSource.AUTOFLUORESCENCE, 0.9,
           "AF points × 1.2"),
        _f("spectral.hb_delta", "Deeper haemoglobin dip than the reference reading",
           FactorSource.SPECTRAL, 1.0, "Δ Hb proxy +0.10 (≥ 0.06)"),
        _f("spectral.ratio_555_630", "Lower 555/630 nm ratio than the reference reading",
           FactorSource.SPECTRAL, 0.5, "Δ −0.09 (≤ −0.05)"),
        _f("visual.white_patch", "White patch seen by the health worker",
           FactorSource.CLINICAL_VISUAL, 2.0),
        _f("site.high_risk", "Common site for suspicious oral lesions (buccal mucosa)",
           FactorSource.SITE, 1.2, "× 1.15"),
    ]  # fmt: skip
    les2_factors = [
        _f("af.fvl_present", "AF-loss region present", FactorSource.AUTOFLUORESCENCE, 1.5),
        _f("af.depth", "Mean relative AF 0.47 (0.28 below threshold 0.75)",
           FactorSource.AUTOFLUORESCENCE, 2.0, "capped at 2.0"),
        _f("af.area", "Area 16 % of visible mucosa (≥ 15 %)", FactorSource.AUTOFLUORESCENCE, 1.0),
        _f("af.border", "Irregular border (0.38 ≥ 0.35)", FactorSource.AUTOFLUORESCENCE, 0.5),
        _f("af.contrast", "Darker than surroundings (1.52× ≥ 1.4×)",
           FactorSource.AUTOFLUORESCENCE, 0.5),
        _f("af.reference", "In-image reference only (contralateral site failed QC)",
           FactorSource.QC, -1.4, "AF points × 0.75"),
        _f("af.confounder.porphyrin_red", "Porphyrin red fluorescence overlaps 34 % of region",
           FactorSource.AUTOFLUORESCENCE, -0.9, "AF points × 0.80"),
        _f("site.high_risk", "Common site for suspicious oral lesions (lateral tongue)",
           FactorSource.SITE, 0.3, "× 1.15"),
    ]  # fmt: skip
    score1 = round(sum(f.points for f in les1_factors), 1)
    score2 = round(sum(f.points for f in les2_factors), 1)
    lesions = [
        LesionAssessment(
            lesion_id="les-002", site=S.LATERAL_TONGUE_LEFT, origins=[LesionOrigin.AF_FVL],
            region_ids=["r1"], location=r2.centroid, risk_level=RiskLevel.MODERATE, score=score2,
            contributing_factors=les2_factors, confounders=list(r2.confounders),
            recommended_action=_action(RiskLevel.MODERATE),
            summary="Area of fluorescence loss on the left lateral tongue, partly overlapped by "
                    "porphyrin (plaque) signal; re-examine after cleaning."),
        LesionAssessment(
            lesion_id="les-001", site=S.BUCCAL_MUCOSA_LEFT,
            origins=[LesionOrigin.AF_FVL, LesionOrigin.VISUAL, LesionOrigin.SPECTRAL],
            region_ids=["r1"], spectral_reading_ids=["s1"], location=r1.centroid,
            risk_level=RiskLevel.HIGH, score=score1, contributing_factors=les1_factors,
            recommended_action=_action(RiskLevel.HIGH),
            summary="White patch with deep, irregular fluorescence loss and a stronger "
                    "haemoglobin signal than the right cheek (reference): suspicious finding."),
    ]  # fmt: skip

    overrides = [
        _f("symptom.non_healing_ulcer", "Non-healing ulcer for more than 2 weeks",
           FactorSource.OVERRIDE, 0.0, "Floor: HIGH"),
        _f("visual.white_patch", "White patch seen by the health worker",
           FactorSource.OVERRIDE, 0.0, "Floor: MODERATE"),
    ]  # fmt: skip
    patient_factors = [
        _f("lesion.max_score", "Highest finding score (left buccal mucosa)", FactorSource.SITE,
           score1),
        _f("risk.smokeless_tobacco", "Smokeless tobacco (gutka, 20 years)",
           FactorSource.CLINICAL_HISTORY, 1.0),
        _f("risk.smoking", "Smoking (bidi, 15 years)", FactorSource.CLINICAL_HISTORY, 0.5),
        _f("risk.alcohol_daily", "Daily alcohol", FactorSource.CLINICAL_HISTORY, 0.5),
        _f("risk.age_40_plus", "Age 40 or over", FactorSource.CLINICAL_HISTORY, 0.5,
           "risk-factor points capped at 2.5"),
    ]  # fmt: skip
    total = round(sum(f.points for f in patient_factors), 1)

    result = ScreeningResult(
        result_id="res-7f3k92qd-v1",
        screening_id="scr-2026-10-14-0042",
        result_version=1,
        created_at=ANALYSED_AT,
        data_provenance=provenance,
        synthetic=synthetic,
        notices=notices,
        versions=_versions(),
        context=_context("PHC Hoskote screening camp", devices=[SCOUT, CONFIRM]),
        patient=PatientInfo(pseudonymous_id="P-7F3K-92QD", age_years=52, sex=Sex.MALE),
        consent=_consent(),
        risk_factors=RiskFactors(
            smokeless_tobacco=SmokelessTobaccoUse(uses=True, types=["gutka"], years=20, per_day=6),
            smoking=SmokingUse(uses=True, types=["bidi"], years=15),
            alcohol=AlcoholUse(uses=True, frequency="daily"),
            symptoms=Symptoms(non_healing_ulcer_over_2_weeks=True, burning_sensation=True),
        ),  # fmt: skip
        sites=sites,
        spectral_readings=_spectral_pair(synthetic),
        lesions=lesions,
        triage=TriageResult(
            overall_risk_level=RiskLevel.HIGH,
            score=total,
            thresholds=_thresholds(),
            contributing_factors=patient_factors,
            overrides_applied=overrides,
            recommended_action=_action(RiskLevel.HIGH),
            highest_risk_lesion_id="les-001",
            rule_config=_versions().rule_config,
            explanation=(
                f"HIGH: the left buccal mucosa finding scored {score1:.1f} (HIGH from 6.0) and "
                "the patient reports a non-healing ulcer for more than 2 weeks, a clinical red "
                "flag that sets a HIGH floor. The right lateral tongue could not be analysed."
            ),
        ),
        qc_summary=QCSummary(
            passed_all=False,
            sites_failed=[S.LATERAL_TONGUE_RIGHT],
            captures_failed=1,
            retakes_recommended=["Right lateral tongue, white light: hold steady and retake"],
            issues=[blur],
        ),  # fmt: skip
        calibration_status=CalibrationStatus.CALIBRATED,
        limitations=[
            Limitation(
                code="qc.site_failed",
                message="Right lateral tongue failed quality "
                "control (blur) and was not analysed; the left lateral tongue was compared "
                "with an in-image reference instead of its contralateral site.",
            ),
            Limitation(
                code="model.wl_unavailable",
                message="White-light classifier not "
                "available: triage used autofluorescence, the spectral probe and the "
                "health worker's observations only.",
            ),
            Limitation(
                code="rules.unvalidated",
                message="Triage rules are a baseline "
                "heuristic and have not been clinically validated.",
            ),
            *(
                [
                    Limitation(
                        code="data.synthetic",
                        message="Synthetic data generated for "
                        "demonstration; not from a real patient.",
                    )
                ]
                if synthetic
                else []
            ),
        ],  # fmt: skip
        processing=_processing(),
    )
    artifacts: dict[str, bytes] = {}
    artifacts |= site_artifacts(S.BUCCAL_MUCOSA_LEFT.value, [
        LesionDraw(L1_BOX.x, L1_BOX.y, L1_BOX.w, L1_BOX.h, "white", 0.55)], seed=11,
        synthetic=synthetic)  # fmt: skip
    artifacts |= site_artifacts(S.BUCCAL_MUCOSA_RIGHT.value, [], seed=12, synthetic=synthetic)
    artifacts |= site_artifacts(S.LATERAL_TONGUE_LEFT.value, [
        LesionDraw(L2_BOX.x, L2_BOX.y, L2_BOX.w, L2_BOX.h, "red", 0.5, porphyrin=True)], seed=13,
        synthetic=synthetic)  # fmt: skip
    artifacts |= site_artifacts(
        S.LATERAL_TONGUE_RIGHT.value, [], seed=14, synthetic=synthetic, qc_failed=True
    )
    artifacts |= site_artifacts(S.TONGUE_DORSUM.value, [], seed=15, synthetic=synthetic)
    return ReportCase("high", result, artifacts)


def build_low(synthetic: bool = True) -> ReportCase:
    provenance, notices = _provenance(synthetic)
    S = OralSite
    pairs = [S.BUCCAL_MUCOSA_LEFT, S.BUCCAL_MUCOSA_RIGHT, S.LATERAL_TONGUE_LEFT,
             S.LATERAL_TONGUE_RIGHT]  # fmt: skip
    sites = []
    for site in pairs:
        paired = {S.BUCCAL_MUCOSA_LEFT: S.BUCCAL_MUCOSA_RIGHT,
                  S.BUCCAL_MUCOSA_RIGHT: S.BUCCAL_MUCOSA_LEFT,
                  S.LATERAL_TONGUE_LEFT: S.LATERAL_TONGUE_RIGHT,
                  S.LATERAL_TONGUE_RIGHT: S.LATERAL_TONGUE_LEFT}[site]  # fmt: skip
        ref = AFReference(
            type=ReferenceType.CONTRALATERAL, reference_site=paired, reference_green=0.56
        )
        sites.append(_site(site, RiskLevel.LOW, synthetic=synthetic, af=_af(ref, [])))
    for site in (S.TONGUE_DORSUM, S.LOWER_GINGIVA):
        ref = AFReference(type=ReferenceType.IN_IMAGE_AUTO, reference_green=0.54)
        sites.append(_site(site, RiskLevel.LOW, synthetic=synthetic, af=_af(ref, [])))

    result = ScreeningResult(
        result_id="res-2b8m41xa-v1",
        screening_id="scr-2026-10-14-0043",
        result_version=1,
        created_at=ANALYSED_AT,
        data_provenance=provenance,
        synthetic=synthetic,
        notices=notices,
        versions=_versions(),
        context=_context("PHC Hoskote screening camp", devices=[SCOUT]),
        patient=PatientInfo(pseudonymous_id="P-2B8M-41XA", age_years=38, sex=Sex.FEMALE),
        consent=_consent(),
        risk_factors=RiskFactors(
            smokeless_tobacco=SmokelessTobaccoUse(uses=True, types=["khaini"], years=5)
        ),
        sites=sites,
        lesions=[],
        triage=TriageResult(
            overall_risk_level=RiskLevel.LOW,
            score=0.0,
            thresholds=_thresholds(),
            contributing_factors=[
                _f(
                    "af.no_fvl",
                    "No AF-loss region at any examined site",
                    FactorSource.AUTOFLUORESCENCE,
                    0.0,
                ),
                _f(
                    "visual.none",
                    "No lesion seen by the health worker",
                    FactorSource.CLINICAL_VISUAL,
                    0.0,
                ),
                _f(
                    "risk.counselling_only",
                    "Khaini use (5 years): counselling, no lesion",
                    FactorSource.CLINICAL_HISTORY,
                    0.0,
                    "habits add points only when a lesion exists",
                ),
            ],  # fmt: skip
            recommended_action=_action(RiskLevel.LOW),
            rule_config=_versions().rule_config,
            explanation="LOW: every examined site was analysed and none showed fluorescence "
            "loss or a visible lesion. This does not rule out disease.",
        ),
        qc_summary=QCSummary(passed_all=True),
        calibration_status=CalibrationStatus.CALIBRATED,
        limitations=[
            Limitation(
                code="model.wl_unavailable",
                message="White-light classifier not "
                "available: triage used autofluorescence and the health worker's "
                "observations only.",
            ),
            *(
                [
                    Limitation(
                        code="data.synthetic",
                        message="Synthetic data generated for "
                        "demonstration; not from a real patient.",
                    )
                ]
                if synthetic
                else []
            ),
        ],  # fmt: skip
        processing=_processing(),
    )
    artifacts: dict[str, bytes] = {}
    for i, s in enumerate(sites):
        artifacts |= site_artifacts(s.site.value, [], seed=30 + i, synthetic=synthetic)
    return ReportCase("low", result, artifacts)


def build_inconclusive(synthetic: bool = True) -> ReportCase:
    provenance, notices = _provenance(synthetic)
    S = OralSite
    unc = CalibrationStatus.UNCALIBRATED
    issues = [
        QCIssue(code="af_signal_low", severity=QCSeverity.ERROR,
                message="Fluorescence too weak — check the violet light is on and retake.",
                message_key="qc.af_signal_low", metric="green_p90", value=0.05, threshold=0.12),
        QCIssue(code="glare", severity=QCSeverity.WARNING,
                message="Some glare — tilt the device slightly to avoid reflections.",
                message_key="qc.glare", metric="glare_fraction", value=0.022, threshold=0.015),
    ]  # fmt: skip
    ref = AFReference(
        type=ReferenceType.IN_IMAGE_AUTO,
        reference_green=0.31,
        note="Contralateral site failed QC; in-image reference used.",
    )
    sites = [
        _site(S.BUCCAL_MUCOSA_LEFT, RiskLevel.INCONCLUSIVE, synthetic=synthetic,
              qc_issues=issues, calibration=unc),
        _site(S.BUCCAL_MUCOSA_RIGHT, RiskLevel.LOW, synthetic=synthetic,
              af=_af(ref, [], green=0.31), calibration=unc),
    ]  # fmt: skip
    result = ScreeningResult(
        result_id="res-9q4c77lm-v1",
        screening_id="scr-2026-10-14-0044",
        result_version=1,
        created_at=ANALYSED_AT,
        data_provenance=provenance,
        synthetic=synthetic,
        notices=notices,
        versions=_versions(),
        context=_context("Anganwadi centre, Doddaballapur (outreach)", devices=[SCOUT]),
        patient=PatientInfo(pseudonymous_id="P-9Q4C-77LM", age_years=61, sex=Sex.FEMALE),
        consent=_consent(),
        risk_factors=RiskFactors(
            smokeless_tobacco=SmokelessTobaccoUse(
                uses=True, types=["paan_with_tobacco"], years=30, per_day=4
            ),
            symptoms=Symptoms(burning_sensation=True),
        ),  # fmt: skip
        sites=sites,
        lesions=[],
        triage=TriageResult(
            overall_risk_level=RiskLevel.INCONCLUSIVE,
            score=None,
            thresholds=_thresholds(),
            contributing_factors=[
                _f(
                    "qc.site_failed",
                    "Left buccal mucosa could not be analysed (QC failed)",
                    FactorSource.QC,
                    0.0,
                    "LOW requires every examined site to be analysed",
                ),
            ],  # fmt: skip
            recommended_action=_action(RiskLevel.INCONCLUSIVE),
            rule_config=_versions().rule_config,
            explanation="INCONCLUSIVE: the left buccal mucosa images failed quality control, "
            "so a LOW level cannot be given. Retake now, or refer for clinical examination.",
        ),
        qc_summary=QCSummary(
            passed_all=False,
            sites_failed=[S.BUCCAL_MUCOSA_LEFT],
            captures_failed=1,
            retakes_recommended=[
                "Left buccal mucosa, autofluorescence: check the violet light and retake"
            ],
            issues=issues,
        ),  # fmt: skip
        calibration_status=unc,
        limitations=[
            Limitation(
                code="calibration.missing",
                message="Device not calibrated today: "
                "fluorescence intensities are not corrected against the reference card.",
            ),
            Limitation(
                code="model.wl_unavailable",
                message="White-light classifier not available: not used in triage.",
            ),
            *(
                [
                    Limitation(
                        code="data.synthetic",
                        message="Synthetic data generated for "
                        "demonstration; not from a real patient.",
                    )
                ]
                if synthetic
                else []
            ),
        ],  # fmt: skip
        processing=_processing(),
    )
    artifacts: dict[str, bytes] = {}
    artifacts |= site_artifacts(
        S.BUCCAL_MUCOSA_LEFT.value, [], seed=51, synthetic=synthetic, qc_failed=True, af_signal=0.25
    )
    artifacts |= site_artifacts(S.BUCCAL_MUCOSA_RIGHT.value, [], seed=52, synthetic=synthetic)
    return ReportCase("inconclusive", result, artifacts)


def build_stress(synthetic: bool = True) -> ReportCase:
    """HIGH case with five findings, WL classifier output, a modifying review, long strings."""
    base = build_high(synthetic=synthetic)
    r = base.result
    S = OralSite
    # Three more MODERATE findings at the already-imaged sites (images exist for them).
    extra = [
        les.model_copy(
            update={
                "lesion_id": f"les-x{i}",
                "risk_level": RiskLevel.MODERATE,
                "score": round(les.score - 4.0 - i * 0.6, 1)
                if les.score - 4.0 - i * 0.6 >= 3.0
                else 3.0 + i / 10,
                "summary": f"Additional area of fluorescence loss ({i + 1}); re-examine.",
            }
        )
        for i, les in enumerate([r.lesions[0], r.lesions[1], r.lesions[0]])
    ]
    wl_model = ModelInfo(
        component=ModelComponent.WL_LESION_CLASSIFIER,
        name="mobilenetv3_wl",
        version="0.0.1",
        status=ModelStatus.AVAILABLE,
        notes="Research weights; not clinically validated.",
    )
    sites = []
    for site in r.sites:
        if site.site == S.BUCCAL_MUCOSA_LEFT:
            wl = WLAnalysis(
                status="ok",
                model=wl_model,
                predictions=[
                    WLPrediction(label=WLLabel.WHITE_PATCH, probability=0.71),
                    WLPrediction(label=WLLabel.NORMAL, probability=0.20),
                    WLPrediction(label=WLLabel.RED_PATCH, probability=0.09),
                ],
                top_label=WLLabel.WHITE_PATCH,
            )
            site = site.model_copy(update={"wl": wl})
        sites.append(site)
    models = [
        wl_model if m.component == ModelComponent.WL_LESION_CLASSIFIER else m
        for m in r.versions.models
    ]
    context = r.context.model_copy(
        update={
            "health_worker_name": "Smt. Venkatalakshmi Ramachandraiah Gowda",
            "screening_site_name": "Primary Health Centre Doddaballapur Taluk outreach camp, "
            "Ward 14, Anganwadi building",
        }
    )
    review = ClinicianReview(
        reviewer_name="Dr. A. Rao",
        reviewer_id="KSDC-12345",
        agreed=False,
        modified_risk_level=RiskLevel.MODERATE,
        notes="Homogeneous leukoplakia-like plaque; biopsy planned. " * 4,
        signed_at=datetime(2026, 10, 20, 10, 0, tzinfo=UTC),
    )
    limitations = [
        lim
        if lim.code != "model.wl_unavailable"
        else Limitation(
            code="model.wl_research_weights",
            message="White-light classifier uses research weights that are not clinically "
            "validated; its output is shown for the clinician and was not used in triage.",
        )
        for lim in r.limitations
    ]
    result = r.model_copy(
        update={
            "limitations": limitations,
            "lesions": [*r.lesions, *extra],
            "sites": sites,
            "context": context,
            "review": review,
            "supersedes_result_id": "res-7f3k92qd-v0",
            "versions": r.versions.model_copy(update={"models": models}),
        }
    )
    return ReportCase(
        "stress",
        result,
        base.artifacts,
        identity_name="Shivakumaraswamy Narasimhamurthy Venkataramanappa",
        identity_external_id="ABHA 91-2345-6789-0123",
    )


BUILDERS = {
    "high": build_high,
    "low": build_low,
    "inconclusive": build_inconclusive,
    "stress": build_stress,
}


def build_case(name: str, synthetic: bool = True) -> ReportCase:
    try:
        builder = BUILDERS[name]
    except KeyError:
        raise ValueError(f"unknown fixture {name!r}; choose from {sorted(BUILDERS)}") from None
    return builder(synthetic=synthetic)
