"""Orchestrator: one ``ScreeningInput`` in, one ``ScreeningResult`` (+ artifacts) out.

Per site: QC → calibration and corrections → AF→WL registration → mucosa segmentation → AF
maps. Then, once every site's own reference level is known, AF is normalised against the
patient's contralateral (or analogous, or in-image) reference, FVL regions and confounders are
found, and the heatmap is drawn. Spectral readings are analysed separately, and the rule engine
fuses everything into lesion and patient risk levels with reasons.

A site whose captures fail QC is never analysed as if it were good: it is marked
``inconclusive_qc_failed`` and the triage cannot return LOW while it stays that way.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime

import cv2
import numpy as np

from mukha_pipeline import __version__
from mukha_pipeline.af import (
    compute_af_maps,
    detect_fvl,
    empty_fvl_output,
    flag_confounders,
    reference_green,
    relative_af_grid,
    render_heatmap_overlay,
    resolve_reference,
)
from mukha_pipeline.calibration import as_utc, image_calibration_status
from mukha_pipeline.config import PipelineConfig
from mukha_pipeline.fusion import run_triage
from mukha_pipeline.io import encode_png, resize_max_side, to_uint8
from mukha_pipeline.preprocess.correction import (
    apply_color_matrix,
    apply_flat_field,
    exposure_factor,
    subtract_dark,
)
from mukha_pipeline.preprocess.registration import register_af_to_wl
from mukha_pipeline.preprocess.segmentation import (
    ClassicalSegmenter,
    Segmenter,
    segmentation_summary,
)
from mukha_pipeline.qc import (
    combine_qc,
    missing_dark_frame_issue,
    qc_autofluorescence,
    qc_dark_frame,
    qc_registration,
    qc_segmentation,
    qc_white_light,
)
from mukha_pipeline.schema.models import (
    SYNTHETIC_NOTICE,
    AFAnalysis,
    AFReference,
    CalibrationInfo,
    CalibrationStatus,
    CaptureMetadata,
    CaptureRecord,
    DataProvenance,
    Limitation,
    ModelComponent,
    ModelInfo,
    ModelStatus,
    Notices,
    ProcessingInfo,
    QCIssue,
    QCResult,
    QCSeverity,
    QCSummary,
    ReferenceType,
    RegistrationSummary,
    RiskLevel,
    ScreeningResult,
    SegmentationSummary,
    SiteArtifacts,
    SiteResult,
    StageTiming,
    VersionInfo,
    WLAnalysis,
    WorkingResolution,
)
from mukha_pipeline.sites import OralSite, site_info
from mukha_pipeline.spectral import (
    RuleBasedSpectralClassifier,
    SpectralClassifier,
    analyze_spectral,
)
from mukha_pipeline.types import (
    AFMaps,
    AnalysisOutput,
    ImageCalibration,
    Masks,
    RegistrationResult,
    ScreeningInput,
    SiteInput,
)
from mukha_pipeline.wl import LesionClassifier, UnavailableLesionClassifier

log = logging.getLogger(__name__)

JPEG_QUALITY = 90

ANALYSIS_ERROR_MESSAGE = "This site could not be analysed — retake, or examine clinically."
REFERENCE_SITE_SKIPPED_NOTE = "Site not analysed because its captures did not pass quality checks."


# --------------------------------------------------------------------------------------------
# Per-site working state
# --------------------------------------------------------------------------------------------


@dataclass
class _SiteState:
    inp: SiteInput
    scale: float = 1.0
    wl: np.ndarray | None = None  # corrected WL, working resolution
    af: np.ndarray | None = None  # corrected AF, working resolution (own frame)
    qc_parts: list[QCResult] = field(default_factory=list)
    captures: list[CaptureRecord] = field(default_factory=list)
    calibration: CalibrationInfo | None = None
    intensity_scale: float = 1.0
    registration: RegistrationResult | None = None
    registration_qc: QCResult | None = None
    masks: Masks | None = None
    segmentation: SegmentationSummary | None = None
    maps: AFMaps | None = None
    own_reference: float | None = None
    analysable: bool = False
    error: bool = False
    af_analysis: AFAnalysis | None = None
    wl_analysis: WLAnalysis | None = None
    artifacts: SiteArtifacts = field(default_factory=SiteArtifacts)

    @property
    def site(self) -> OralSite:
        return self.inp.site


class _Timer:
    def __init__(self) -> None:
        self.stages: list[StageTiming] = []

    @contextmanager
    def stage(self, name: str, site: OralSite | None = None) -> Iterator[None]:
        t0 = time.perf_counter()
        status = "ok"
        try:
            yield
        except Exception:
            status = "failed"
            raise
        finally:
            self.stages.append(
                StageTiming(
                    stage=name,
                    duration_ms=round((time.perf_counter() - t0) * 1000.0, 2),
                    status=status,
                    site=site,
                )
            )


# --------------------------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------------------------


def analyze_screening(
    inp: ScreeningInput,
    config: PipelineConfig | None = None,
    *,
    classifier: LesionClassifier | None = None,
    segmenter: Segmenter | None = None,
    spectral_classifier: SpectralClassifier | None = None,
    software_version: str | None = None,
    strict: bool = False,
) -> AnalysisOutput:
    """Run the full screening analysis.

    ``strict=True`` re-raises unexpected errors (tests); otherwise a site whose analysis crashes
    is reported as not analysed, so one bad capture cannot take down the whole screening and the
    triage falls back to INCONCLUSIVE rather than LOW for that site.
    """
    t_start = time.perf_counter()
    cfg = config or PipelineConfig.default()
    qc_cfg = cfg.qc.data
    an_cfg = cfg.analysis.data
    now = as_utc(inp.now) if inp.now is not None else datetime.now(UTC)
    classifier = classifier or UnavailableLesionClassifier()
    segmenter = segmenter or ClassicalSegmenter(an_cfg.get("segmentation"))
    spectral_classifier = spectral_classifier or RuleBasedSpectralClassifier(
        an_cfg.get("spectral", {})
    )
    max_side = int(an_cfg.get("working_max_side_px", 1024))
    timer = _Timer()
    artifacts: dict[str, bytes] = {}

    # All AF captures are brought onto the exposure of the first AF capture with known settings,
    # so a site and its contralateral reference are compared at equal exposure.
    ref_af_meta = next((s.af.meta for s in inp.sites if s.af.meta.exposure_time_ms), None)

    states = [_SiteState(inp=s) for s in inp.sites]

    # Pass 1: QC, corrections, registration, segmentation, AF maps, own reference level.
    for st in states:
        _guarded(
            st,
            strict,
            lambda st=st: _prepare_site(
                st, inp, qc_cfg, an_cfg, max_side, now, ref_af_meta, segmenter, timer, artifacts
            ),
        )

    # Pass 2: reference normalisation needs every other site's reference level.
    available = {
        st.site: st.own_reference for st in states if st.analysable and st.own_reference is not None
    }
    for st in states:
        _guarded(
            st,
            strict,
            lambda st=st: _analyse_site(st, available, an_cfg, classifier, timer, artifacts),
        )

    # Spectral readings (Confirm probe).
    spectral_results = []
    if inp.spectral:
        with timer.stage("spectral"):
            spectral_results = analyze_spectral(
                inp.spectral,
                inp.spectral_calibrations,
                qc_cfg,
                an_cfg,
                now,
                classifier=spectral_classifier,
            )

    site_results = [_site_result(st, spectral_results, an_cfg) for st in states]

    with timer.stage("triage"):
        triage_out = run_triage(
            site_results, spectral_results, inp.patient, inp.risk_factors, cfg.triage
        )
    site_results = [
        sr.model_copy(
            update={
                "site_risk_level": triage_out.site_levels.get(
                    sr.site,
                    RiskLevel.LOW if sr.analysis_status == "analysed" else RiskLevel.INCONCLUSIVE,
                )
            }
        )
        for sr in site_results
    ]

    synthetic = inp.data_provenance == DataProvenance.SYNTHETIC or _any_synthetic(inp)
    provenance = inp.data_provenance
    if synthetic and provenance == DataProvenance.REAL:
        provenance = DataProvenance.MIXED

    calibration_status = _overall_calibration(
        [sr.calibration.status for sr in site_results]
        + [r.calibration.status for r in spectral_results]
    )
    limitations = _limitations(
        triage_out.limitations, site_results, calibration_status, synthetic, cfg
    )

    models = [
        ModelInfo(
            component=ModelComponent.SEGMENTER,
            name=str(getattr(segmenter, "name", type(segmenter).__name__)),
            version=str(getattr(segmenter, "version", "unknown")),
            status=ModelStatus.BASELINE_RULES,
            notes="Classical colour and texture rules; a trained segmenter can replace it.",
        ),
        classifier.info,
        spectral_classifier.info,
        ModelInfo(
            component=ModelComponent.FUSION,
            name=cfg.triage.ref.name,
            version=cfg.triage.ref.version,
            status=ModelStatus.BASELINE_RULES,
            notes="Transparent YAML rules; baseline heuristic, not clinically validated.",
        ),
    ]

    result = ScreeningResult(
        result_id=inp.result_id or f"res-{uuid.uuid4()}",
        screening_id=inp.screening_id,
        result_version=inp.result_version,
        supersedes_result_id=inp.supersedes_result_id,
        created_at=now,
        data_provenance=provenance,
        synthetic=synthetic,
        notices=Notices(synthetic_notice=SYNTHETIC_NOTICE if synthetic else None),
        versions=VersionInfo(
            software=software_version or __version__,
            pipeline=__version__,
            rule_config=cfg.triage.ref,
            qc_config=cfg.qc.ref,
            analysis_config=cfg.analysis.ref,
            models=models,
        ),
        context=inp.context,
        patient=inp.patient,
        consent=inp.consent,
        risk_factors=inp.risk_factors,
        sites=site_results,
        spectral_readings=spectral_results,
        lesions=triage_out.lesions,
        triage=triage_out.triage,
        qc_summary=_qc_summary(site_results, spectral_results),
        calibration_status=calibration_status,
        limitations=limitations,
        processing=ProcessingInfo(
            working_max_side_px=max_side,
            duration_ms=round((time.perf_counter() - t_start) * 1000.0, 2),
            stages=timer.stages,
        ),
    )
    return AnalysisOutput(result=result, artifacts=artifacts)


# --------------------------------------------------------------------------------------------
# Pass 1
# --------------------------------------------------------------------------------------------


def _prepare_site(
    st: _SiteState,
    inp: ScreeningInput,
    qc_cfg: dict,
    an_cfg: dict,
    max_side: int,
    now: datetime,
    ref_af_meta: CaptureMetadata | None,
    segmenter: Segmenter,
    timer: _Timer,
    artifacts: dict[str, bytes],
) -> None:
    site = st.site
    prefix = _prefix(site)

    with timer.stage("resize", site):
        wl, st.scale = resize_max_side(_float_rgb(st.inp.wl.image), max_side)
        af, _ = resize_max_side(_float_rgb(st.inp.af.image), max_side)
        dark = None
        if st.inp.dark is not None:
            dark = cv2.resize(
                _float_rgb(st.inp.dark.image),
                (af.shape[1], af.shape[0]),
                interpolation=cv2.INTER_AREA,
            )

    with timer.stage("quality_check", site):
        wl_qc = qc_white_light(wl, qc_cfg)
        af_qc = qc_autofluorescence(af, qc_cfg)
        if dark is not None:
            dark_qc = qc_dark_frame(dark, qc_cfg)
        else:
            dark_qc = QCResult(passed=True, score=1.0, issues=[missing_dark_frame_issue()])

    with timer.stage("calibration", site):
        cal = _image_calibration(st.inp, inp.image_calibrations)
        st.calibration = image_calibration_status(cal, now)
        if dark is not None:
            af = subtract_dark(af, dark)
        if cal is not None:
            wl = apply_flat_field(wl, cal.flat_field)
            wl = apply_color_matrix(wl, cal.color_matrix)  # colour correction is WL-only
            af = apply_flat_field(af, cal.flat_field)
        st.intensity_scale = exposure_factor(st.inp.af.meta, ref_af_meta) * (
            float(cal.af_scale) if cal is not None else 1.0
        )
    st.wl, st.af = wl, af

    wl_key, af_key = f"{prefix}/wl.jpg", f"{prefix}/af.jpg"
    artifacts[wl_key] = _encode_jpeg(wl)
    artifacts[af_key] = _encode_jpeg(af)
    st.artifacts = st.artifacts.model_copy(update={"wl_image": wl_key, "af_image": af_key})
    st.captures = [
        CaptureRecord(metadata=st.inp.wl.meta, qc=wl_qc, artifact=wl_key),
        CaptureRecord(metadata=st.inp.af.meta, qc=af_qc, artifact=af_key),
    ]
    if st.inp.dark is not None:
        st.captures.append(CaptureRecord(metadata=st.inp.dark.meta, qc=dark_qc))
    st.qc_parts = [wl_qc, af_qc, dark_qc]
    if not (wl_qc.passed and af_qc.passed and dark_qc.passed):
        return

    with timer.stage("registration", site):
        st.registration = register_af_to_wl(wl, af, an_cfg.get("registration"))
        st.registration_qc = qc_registration(st.registration, qc_cfg)
    st.qc_parts.append(st.registration_qc)
    reg_key = f"{prefix}/af_registered.jpg"
    artifacts[reg_key] = _encode_jpeg(st.registration.registered)
    st.artifacts = st.artifacts.model_copy(update={"af_registered": reg_key})
    if not st.registration_qc.passed:
        return

    with timer.stage("segmentation", site):
        st.masks = segmenter.segment(wl)
        seg_qc = qc_segmentation(st.masks, qc_cfg)
        st.segmentation = segmentation_summary(st.masks, segmenter, an_cfg.get("segmentation", {}))
    st.qc_parts.append(seg_qc)
    mask_key = f"{prefix}/mucosa_mask.png"
    artifacts[mask_key] = encode_png(st.masks.mucosa.astype(np.uint8) * 255)
    st.artifacts = st.artifacts.model_copy(update={"mucosa_mask": mask_key})
    if not seg_qc.passed:
        return

    with timer.stage("af_maps", site):
        st.maps = compute_af_maps(
            st.registration.registered, st.registration.valid, intensity_scale=st.intensity_scale
        )
        st.own_reference = reference_green(st.maps, st.masks.mucosa, an_cfg.get("af"))
    st.analysable = True


# --------------------------------------------------------------------------------------------
# Pass 2
# --------------------------------------------------------------------------------------------


def _analyse_site(
    st: _SiteState,
    available: dict[OralSite, float],
    an_cfg: dict,
    classifier: LesionClassifier,
    timer: _Timer,
    artifacts: dict[str, bytes],
) -> None:
    af_cfg = an_cfg.get("af", {})
    if not st.analysable or st.error:
        shape = st.wl.shape[:2] if st.wl is not None else (1, 1)
        st.af_analysis = empty_fvl_output(
            shape,
            af_cfg,
            status="skipped_qc_failed",
            reference=AFReference(type=ReferenceType.NONE, note=REFERENCE_SITE_SKIPPED_NOTE),
        ).analysis
        st.wl_analysis = WLAnalysis(
            status="skipped_qc_failed", model=classifier.info, note=REFERENCE_SITE_SKIPPED_NOTE
        )
        return

    assert st.maps is not None and st.masks is not None and st.wl is not None
    site = st.site
    prefix = _prefix(site)
    others = {s: v for s, v in available.items() if s != site}

    with timer.stage("autofluorescence", site):
        reference = resolve_reference(
            site,
            st.own_reference,
            others,
            min_paired_ratio=float(af_cfg.get("reference_min_paired_ratio", 0.67)),
        )
        px_per_mm = st.inp.px_per_mm * st.scale if st.inp.px_per_mm else None
        fvl = detect_fvl(st.maps, st.masks.mucosa, reference, af_cfg, px_per_mm=px_per_mm)
        fvl = flag_confounders(fvl, st.wl, st.maps, st.masks, an_cfg)
        st.af_analysis = fvl.analysis

    with timer.stage("heatmap", site):
        heat = render_heatmap_overlay(
            st.wl, fvl.relative_af, fvl.labels, fvl.analysis.fvl_threshold, af_cfg
        )
        grid = relative_af_grid(
            fvl.relative_af,
            int(af_cfg.get("relative_af_grid_size", 96)),
            threshold=fvl.analysis.fvl_threshold,
        )
        rel = np.nan_to_num(fvl.relative_af, nan=0.0)
        rel_png = np.clip(rel / 1.5, 0.0, 1.0)
    heat_key = f"{prefix}/heatmap_overlay.jpg"
    rel_key = f"{prefix}/relative_af_map.png"
    grid_key = f"{prefix}/relative_af_grid.json"
    artifacts[heat_key] = _encode_jpeg(heat, quality=92)
    artifacts[rel_key] = encode_png(rel_png)
    artifacts[grid_key] = json.dumps(grid, allow_nan=False).encode("utf-8")
    updates: dict[str, str | None] = {
        "heatmap_overlay": heat_key,
        "relative_af_map": rel_key,
        "relative_af_grid": grid_key,
    }

    with timer.stage("wl_classifier", site):
        wl_analysis = classifier.classify(st.wl, st.masks.mucosa)
        saliency_fn = getattr(classifier, "saliency", None)
        if wl_analysis.status == "ok" and callable(saliency_fn):
            sal = saliency_fn(st.wl, st.masks.mucosa)
            if sal is not None:
                sal_key = f"{prefix}/wl_saliency.png"  # clinician-only explainability map
                artifacts[sal_key] = encode_png(np.clip(sal, 0.0, 1.0))
                wl_analysis = wl_analysis.model_copy(update={"saliency_artifact": sal_key})
                updates["wl_saliency"] = sal_key
    st.wl_analysis = wl_analysis
    st.artifacts = st.artifacts.model_copy(update=updates)


# --------------------------------------------------------------------------------------------
# Result assembly
# --------------------------------------------------------------------------------------------


def _site_result(st: _SiteState, spectral_results: list, an_cfg: dict) -> SiteResult:
    info = site_info(st.site)
    qc_parts = list(st.qc_parts)
    if st.error:
        qc_parts.append(
            QCResult(
                passed=False,
                score=0.0,
                issues=[
                    QCIssue(
                        code="analysis_error",
                        severity=QCSeverity.ERROR,
                        message=ANALYSIS_ERROR_MESSAGE,
                        message_key="qc.analysis_error",
                    )
                ],
            )
        )
    qc = combine_qc(qc_parts) if qc_parts else QCResult(passed=False, score=0.0)
    analysed = st.analysable and not st.error
    registration = None
    if st.registration is not None:
        registration = RegistrationSummary(
            method=st.registration.method,
            confidence=float(np.clip(st.registration.confidence, 0.0, 1.0)),
            passed=bool(st.registration_qc.passed if st.registration_qc else False),
            warp_matrix=[[float(v) for v in row] for row in np.asarray(st.registration.warp)],
            note=st.registration.note,
        )
    working = None
    if st.wl is not None:
        h, w = st.wl.shape[:2]
        working = WorkingResolution(width_px=w, height_px=h, scale_from_original=st.scale)
    return SiteResult(
        site=st.site,
        site_label=info.label,
        role=st.inp.role,
        paired_site=info.paired_site,
        pairing=info.pairing,
        visual_flag=st.inp.visual_flag,
        diascopy=st.inp.diascopy,
        captures=st.captures,
        qc=qc,
        analysis_status="analysed" if analysed else "inconclusive_qc_failed",
        calibration=st.calibration
        or CalibrationInfo(status=CalibrationStatus.UNCALIBRATED, note="Not assessed."),
        registration=registration,
        segmentation=st.segmentation,
        af=st.af_analysis,
        wl=st.wl_analysis,
        spectral_reading_ids=[r.reading_id for r in spectral_results if r.site == st.site],
        artifacts=st.artifacts,
        working_resolution=working,
        px_per_mm=st.inp.px_per_mm * st.scale if st.inp.px_per_mm else None,
        site_risk_level=RiskLevel.INCONCLUSIVE,  # replaced after triage
    )


def _qc_summary(sites: list[SiteResult], spectral: list) -> QCSummary:
    failed_sites = [s.site for s in sites if s.analysis_status != "analysed"]
    captures_failed = sum(1 for s in sites for c in s.captures if not c.qc.passed)
    captures_failed += sum(1 for r in spectral if not r.qc.passed)
    retakes: list[str] = []
    issues: list[QCIssue] = []
    seen: set[tuple[str, str]] = set()
    for s in sites:
        for issue in s.qc.issues:
            if issue.severity == QCSeverity.ERROR:
                retakes.append(f"{s.site_label}: {issue.message}")
            key = (issue.code, issue.message)
            if issue.severity != QCSeverity.INFO and key not in seen:
                seen.add(key)
                issues.append(issue)
    for r in spectral:
        for issue in r.qc.issues:
            if issue.severity == QCSeverity.ERROR:
                retakes.append(f"Probe reading {r.reading_id} ({r.site.value}): {issue.message}")
            key = (issue.code, issue.message)
            if issue.severity != QCSeverity.INFO and key not in seen:
                seen.add(key)
                issues.append(issue)
    passed_all = not failed_sites and all(r.qc.passed for r in spectral)
    return QCSummary(
        passed_all=passed_all,
        sites_failed=failed_sites,
        captures_failed=captures_failed,
        retakes_recommended=retakes,
        issues=issues,
    )


def _overall_calibration(statuses: list[CalibrationStatus]) -> CalibrationStatus:
    if not statuses:
        return CalibrationStatus.UNCALIBRATED
    n_cal = sum(1 for s in statuses if s == CalibrationStatus.CALIBRATED)
    if n_cal == len(statuses):
        return CalibrationStatus.CALIBRATED
    if n_cal == 0:
        if all(s == CalibrationStatus.EXPIRED for s in statuses):
            return CalibrationStatus.EXPIRED
        return CalibrationStatus.UNCALIBRATED
    return CalibrationStatus.PARTIAL


def _limitations(
    from_triage: list[Limitation],
    sites: list[SiteResult],
    calibration: CalibrationStatus,
    synthetic: bool,
    cfg: PipelineConfig,
) -> list[Limitation]:
    out: list[Limitation] = []
    if synthetic:
        out.append(Limitation(code="synthetic_data", message=SYNTHETIC_NOTICE))
    if calibration == CalibrationStatus.UNCALIBRATED:
        out.append(
            Limitation(
                code="uncalibrated",
                message=(
                    "No valid device calibration: absolute fluorescence and reflectance are not "
                    "standardised. Comparisons within this patient (left vs right) remain usable."
                ),
            )
        )
    elif calibration == CalibrationStatus.EXPIRED:
        out.append(
            Limitation(
                code="calibration_expired",
                message="Device calibration has expired; recalibrate before the next screening.",
            )
        )
    elif calibration == CalibrationStatus.PARTIAL:
        out.append(
            Limitation(
                code="calibration_partial",
                message="Some captures were not covered by a valid device calibration.",
            )
        )
    no_dark = [
        s.site_label
        for s in sites
        if not any(c.metadata.kind.value == "dark_frame" for c in s.captures)
    ]
    if no_dark:
        out.append(
            Limitation(
                code="no_dark_frame",
                message=(
                    "No dark frame for "
                    + ", ".join(no_dark)
                    + "; room light could not be checked or subtracted."
                ),
            )
        )
    out.extend(from_triage)
    deduped: list[Limitation] = []
    seen: set[str] = set()
    for lim in out:
        if lim.code not in seen:
            seen.add(lim.code)
            deduped.append(lim)
    return deduped


# --------------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------------


def _guarded(st: _SiteState, strict: bool, fn: Callable[[], None]) -> None:
    try:
        fn()
    except Exception:
        if strict:
            raise
        log.exception("analysis failed for site %s", st.site)
        st.error = True
        st.analysable = False


def _image_calibration(
    site: SiteInput, calibrations: dict[str, ImageCalibration]
) -> ImageCalibration | None:
    for meta in (site.af.meta, site.wl.meta):
        if meta.calibration_id:
            for cal in calibrations.values():
                if cal.calibration_id == meta.calibration_id:
                    return cal
    device_id = site.af.meta.device_id or site.wl.meta.device_id
    return calibrations.get(device_id) if device_id else None


def _any_synthetic(inp: ScreeningInput) -> bool:
    for s in inp.sites:
        metas = [s.wl.meta, s.af.meta] + ([s.dark.meta] if s.dark is not None else [])
        if any(m.synthetic for m in metas):
            return True
    return any(r.meta.synthetic for r in inp.spectral)


def _float_rgb(img: np.ndarray) -> np.ndarray:
    arr = np.asarray(img)
    if arr.dtype == np.uint8:
        arr = arr.astype(np.float32) / 255.0
    elif arr.dtype == np.uint16:
        arr = arr.astype(np.float32) / 65535.0
    else:
        arr = arr.astype(np.float32)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    if arr.shape[2] > 3:
        arr = arr[:, :, :3]
    return np.clip(np.nan_to_num(arr, nan=0.0), 0.0, 1.0)


def _encode_jpeg(img: np.ndarray, quality: int = JPEG_QUALITY) -> bytes:
    rgb = to_uint8(img)
    ok, buf = cv2.imencode(
        ".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality]
    )
    if not ok:  # pragma: no cover - OpenCV only fails on invalid input
        raise ValueError("JPEG encoding failed")
    return buf.tobytes()


def _prefix(site: OralSite) -> str:
    return f"sites/{OralSite(site).value}"


__all__ = ["analyze_screening"]
