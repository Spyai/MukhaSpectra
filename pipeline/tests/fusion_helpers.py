"""Factories for fusion tests: minimal, valid SiteResult / FVLRegion / spectral objects.

All objects are hand-built test fixtures (not patient data, not synthetic images).
"""

from __future__ import annotations

import copy
import math
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import yaml

from mukha_pipeline.config import TRIAGE_CONFIG, LoadedConfig, load_config
from mukha_pipeline.fusion import TriageOutput, TriageRules, parse_triage_rules
from mukha_pipeline.positioning import find_positioning_violations
from mukha_pipeline.schema.models import (
    AFAnalysis,
    AFReference,
    CalibrationInfo,
    CalibrationStatus,
    CaptureKind,
    CaptureMetadata,
    ConfounderFlag,
    ConfounderType,
    Diascopy,
    FVLRegion,
    ModelComponent,
    ModelInfo,
    ModelStatus,
    NormBox,
    NormPoint,
    PatientInfo,
    QCIssue,
    QCResult,
    QCSeverity,
    ReferenceType,
    RiskFactors,
    RiskLevel,
    SegmentationSummary,
    SiteResult,
    SiteRole,
    SourceAdapter,
    SpectralClassifierOutput,
    SpectralFeatures,
    SpectralReadingResult,
    VisualAppearance,
    VisualFlag,
    WLAnalysis,
    WLLabel,
    WLPrediction,
)
from mukha_pipeline.sites import OralSite, reference_site_for, site_label

S = OralSite


def default_rules() -> LoadedConfig:
    return load_config(TRIAGE_CONFIG)


def parse_rules_default() -> TriageRules:
    return parse_triage_rules(default_rules().data)


def rules_data() -> dict[str, Any]:
    return copy.deepcopy(default_rules().data)


def write_rules(tmp_path: Path, mutate: Callable[[dict[str, Any]], None]) -> LoadedConfig:
    """Load a modified copy of the bundled rules from ``tmp_path``."""
    data = rules_data()
    mutate(data)
    path = tmp_path / "triage_rules.test.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return load_config(path)


# --------------------------------------------------------------------------------------------
# Building blocks
# --------------------------------------------------------------------------------------------


def qc(passed: bool = True) -> QCResult:
    issues = []
    if not passed:
        issues.append(
            QCIssue(
                code="blur",
                severity=QCSeverity.ERROR,
                message="Image is blurred. Hold the device steady and retake.",
                message_key="qc.blur",
            )
        )
    return QCResult(passed=passed, score=1.0 if passed else 0.2, issues=issues)


def calibration() -> CalibrationInfo:
    return CalibrationInfo(status=CalibrationStatus.CALIBRATED, calibration_id="cal-test")


def region(
    region_id: str = "r1",
    *,
    mean_rel: float = 0.6,
    area_mm2: float | None = None,
    fraction: float = 0.02,
    border: float = 0.1,
    contrast: float = 1.1,
    confounders: Iterable[tuple[ConfounderType, float]] = (),
    centroid: tuple[float, float] = (0.5, 0.5),
) -> FVLRegion:
    return FVLRegion(
        region_id=region_id,
        area_px=1500,
        area_mm2=area_mm2,
        area_fraction_of_mucosa=fraction,
        centroid=NormPoint(x=centroid[0], y=centroid[1]),
        bbox=NormBox(x=0.4, y=0.4, w=0.2, h=0.2),
        mean_relative_af=mean_rel,
        min_relative_af=max(0.0, mean_rel - 0.1),
        contrast_vs_surround=contrast,
        border_irregularity=border,
        solidity=0.9,
        confounders=[
            ConfounderFlag(type=t, overlap_fraction=f, note=f"{t.value} test overlap")
            for t, f in confounders
        ],
    )


def af(
    site: OralSite,
    regions: Iterable[FVLRegion] = (),
    *,
    status: str = "ok",
    ref_type: ReferenceType | None = None,
    threshold: float = 0.75,
) -> AFAnalysis:
    paired, pairing = reference_site_for(site)
    if ref_type is None:
        ref_type = {
            "contralateral": ReferenceType.CONTRALATERAL,
            "analogous": ReferenceType.ANALOGOUS_SITE,
        }.get(pairing or "", ReferenceType.IN_IMAGE_AUTO)
    ref_site = (
        paired if ref_type in (ReferenceType.CONTRALATERAL, ReferenceType.ANALOGOUS_SITE) else None
    )
    regions = list(regions)
    return AFAnalysis(
        status=status,  # type: ignore[arg-type]
        reference=AFReference(type=ref_type, reference_site=ref_site, reference_green=0.4),
        fvl_threshold=threshold,
        min_region_area_px=150,
        fvl_area_fraction=min(1.0, sum(r.area_fraction_of_mucosa for r in regions)),
        regions=regions,
    )


def wl_unavailable() -> WLAnalysis:
    return WLAnalysis(
        status="model_unavailable",
        model=ModelInfo(
            component=ModelComponent.WL_LESION_CLASSIFIER,
            name="none",
            version="0",
            status=ModelStatus.MODEL_UNAVAILABLE,
        ),
    )


def wl_ok(label: WLLabel, prob: float, *, set_top: bool = True) -> WLAnalysis:
    rest = (1.0 - prob) / 4
    preds = [WLPrediction(label=label, probability=prob)] + [
        WLPrediction(label=other, probability=rest) for other in WLLabel if other != label
    ]
    return WLAnalysis(
        status="ok",
        model=ModelInfo(
            component=ModelComponent.WL_LESION_CLASSIFIER,
            name="test-wl",
            version="0",
            status=ModelStatus.AVAILABLE,
        ),
        predictions=preds,
        top_label=label if set_top else None,
    )


def segmentation(status: str = "ok") -> SegmentationSummary:
    return SegmentationSummary(
        method="test",
        version="0",
        status=status,  # type: ignore[arg-type]
        mucosa_fraction=0.5,
        teeth_fraction=0.1,
        specular_fraction=0.01,
        confidence={"ok": 0.9, "low_confidence": 0.3}.get(status, 0.0),
    )


def site(
    oral_site: OralSite = S.BUCCAL_MUCOSA_LEFT,
    regions: Iterable[FVLRegion] = (),
    *,
    role: SiteRole = SiteRole.EXAMINED,
    analysed: bool = True,
    af_status: str = "ok",
    ref_type: ReferenceType | None = None,
    appearance: Iterable[VisualAppearance] | None = None,
    lesion_seen: bool | None = None,
    diascopy: Diascopy = Diascopy.NOT_DONE,
    wl: WLAnalysis | None = None,
    seg_status: str | None = None,
    threshold: float = 0.75,
) -> SiteResult:
    """A site result. ``analysed=False`` means the images failed QC (no AF analysis).

    ``seg_status`` records a segmentation summary with that status (``None``: no summary, as for
    results built outside the orchestrator).
    """
    appearance = list(appearance or [])
    seen = bool(appearance) if lesion_seen is None else lesion_seen
    paired, pairing = reference_site_for(oral_site)
    return SiteResult(
        site=oral_site,
        site_label=site_label(oral_site),
        role=role,
        paired_site=paired,
        pairing=pairing,
        visual_flag=VisualFlag(lesion_seen=seen, appearance=appearance),
        diascopy=diascopy,
        qc=qc(analysed),
        analysis_status="analysed" if analysed else "inconclusive_qc_failed",
        calibration=calibration(),
        segmentation=segmentation(seg_status) if seg_status is not None and analysed else None,
        af=(
            af(oral_site, regions, status=af_status, ref_type=ref_type, threshold=threshold)
            if analysed
            else None
        ),
        wl=(wl or wl_unavailable()) if analysed else None,
        site_risk_level=RiskLevel.INCONCLUSIVE,
    )


def features(hb: float = 0.0, ratio: float = 0.0) -> SpectralFeatures:
    return SpectralFeatures(
        ratio_555_630=ratio,
        ratio_590_680=0.0,
        ratio_515_630=0.0,
        ratio_415_555=0.0,
        hb_absorption_proxy=hb,
        slope_480_555=0.0,
        slope_590_680=0.0,
    )


def reading(
    reading_id: str,
    oral_site: OralSite,
    *,
    role: str = "lesion",
    hb_delta: float | None = None,
    ratio_delta: float = 0.0,
    label: str | None = None,
    qc_passed: bool = True,
    lesion_ref: str | None = None,
) -> SpectralReadingResult:
    delta = features(hb_delta, ratio_delta) if hb_delta is not None else None
    return SpectralReadingResult(
        reading_id=reading_id,
        site=oral_site,
        role=role,  # type: ignore[arg-type]
        lesion_ref=lesion_ref,
        capture=CaptureMetadata(
            capture_id=f"cap-{reading_id}",
            kind=CaptureKind.SPECTRAL,
            oral_site=oral_site,
            source_adapter=SourceAdapter.UPLOAD,
        ),
        n_readings=5,
        signal_type="reflectance",
        calibration=calibration(),
        values={"F5_555": 0.3, "F7_630": 0.5},
        values_std={"F5_555": 0.01, "F7_630": 0.01},
        qc=qc(qc_passed),
        features=features(0.1, 0.6),
        reference_reading_id="ref" if delta is not None else None,
        delta_vs_reference=delta,
        classifier=SpectralClassifierOutput(
            status="placeholder",
            method="rules_v1",
            label=label,  # type: ignore[arg-type]
            model=ModelInfo(
                component=ModelComponent.SPECTRAL_CLASSIFIER,
                name="rules_v1",
                version="0",
                status=ModelStatus.PLACEHOLDER,
            ),
        ),
    )


def patient(age: int | None = 35) -> PatientInfo:
    return PatientInfo(pseudonymous_id="P-TEST", age_years=age)


def risk(**fields: Any) -> RiskFactors:
    """RiskFactors from nested dicts, e.g. risk(smokeless_tobacco={"uses": True})."""
    return RiskFactors.model_validate(fields)


# --------------------------------------------------------------------------------------------
# Invariants
# --------------------------------------------------------------------------------------------


def all_texts(out: TriageOutput) -> list[str]:
    """Every user-facing string the fusion step produced."""
    t = out.triage
    texts = [t.explanation, t.recommended_action.summary, *t.recommended_action.steps]
    for f in [*t.contributing_factors, *t.overrides_applied]:
        texts.extend([f.label, f.detail or ""])
    for lesion in out.lesions:
        texts.extend([lesion.summary, lesion.recommended_action.summary])
        texts.extend(lesion.recommended_action.steps)
        for f in lesion.contributing_factors:
            texts.extend([f.label, f.detail or ""])
    texts.extend(lim.message for lim in out.limitations)
    return texts


def assert_invariants(out: TriageOutput) -> None:
    """Score == sum of factors everywhere, positioning-clean text, review always required."""
    for lesion in out.lesions:
        total = math.fsum(f.points for f in lesion.contributing_factors)
        assert abs(total - lesion.score) < 1e-6, lesion.lesion_id
    t = out.triage
    assert abs(math.fsum(f.points for f in t.contributing_factors) - (t.score or 0.0)) < 1e-6
    assert all(f.points == 0 for f in t.overrides_applied)
    assert t.requires_clinician_review is True
    assert t.review_status == "pending_review"
    violations = [v for text in all_texts(out) for v in find_positioning_violations(text)]
    assert violations == []
    sentences = [s for s in t.explanation.split(". ") if s.strip()]
    assert 1 <= len(sentences) <= 3, t.explanation
    order = {RiskLevel.LOW: 0, RiskLevel.MODERATE: 1, RiskLevel.HIGH: 2}
    if t.overall_risk_level in order:
        for lesion in out.lesions:
            if lesion.risk_level in order:
                assert order[t.overall_risk_level] >= order[lesion.risk_level]
    if t.overall_risk_level == RiskLevel.LOW:
        assert "does not rule out" in t.explanation
        assert "re-screening" in t.explanation
        assert all(lesion.risk_level == RiskLevel.LOW for lesion in out.lesions)
    for lesion in out.lesions:
        assert lesion.summary.count(". ") == 0, lesion.summary


def codes(factors: Iterable[Any]) -> list[str]:
    return [f.code for f in factors]


def factor(factors: Iterable[Any], code: str) -> Any:
    matches = [f for f in factors if f.code == code]
    assert len(matches) == 1, (code, codes(factors))
    return matches[0]


def limitation_codes(out: TriageOutput) -> list[str]:
    return [lim.code for lim in out.limitations]
