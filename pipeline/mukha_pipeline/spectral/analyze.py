"""Spectral stage: QC -> summary -> features -> lesion-vs-reference delta -> classifier.

The delta is computed on the per-channel lesion / reference quotient
(:func:`mukha_pipeline.spectral.features.reference_delta`), which cancels the probe's LED spectrum,
filter responses and gain, so calibrated and uncalibrated readings are scored on the same scale.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from typing import Any

from mukha_pipeline.schema.models import (
    CalibrationInfo,
    CalibrationStatus,
    QCIssue,
    QCResult,
    QCSeverity,
    SpectralClassifierOutput,
    SpectralFeatures,
    SpectralReadingResult,
)
from mukha_pipeline.spectral.classifier import RuleBasedSpectralClassifier, SpectralClassifier
from mukha_pipeline.spectral.features import compute_features, reference_delta
from mukha_pipeline.spectral.pairing import auto_pair_inputs
from mukha_pipeline.spectral.summary import summarise_reading
from mukha_pipeline.types import SpectralCalibration, SpectralFrame, SpectralInput

QCFn = Callable[[list[SpectralFrame], list[SpectralFrame], dict], QCResult]

REFERENCE_QC_FAILED = (
    "The reference reading '{ref}' did not pass quality checks, so it was not used for comparison."
)
REFERENCE_MISSING = "The reference reading '{ref}' was not found among the probe readings."
REFERENCE_UNUSABLE = (
    "The reference reading '{ref}' has no usable signal in the 515-630 nm bands, so it was not "
    "used for comparison."
)
SIGNAL_MISMATCH = (
    "The lesion and reference readings were not processed the same way (one calibrated, one "
    "not), so they cannot be compared."
)


def _default_qc_fn() -> QCFn:
    """Lazy import: qc/ is a separate module and must not be needed to import this one."""
    import importlib

    for module_name in ("mukha_pipeline.qc", "mukha_pipeline.qc.spectral"):
        try:
            fn = getattr(importlib.import_module(module_name), "qc_spectral", None)
        except ImportError:
            continue
        if callable(fn):
            return fn  # type: ignore[no-any-return]
    return _qc_unavailable


def _failed_qc(code: str, message: str) -> QCResult:
    return QCResult(
        passed=False,
        score=0.0,
        issues=[
            QCIssue(
                code=code,
                severity=QCSeverity.ERROR,
                message=message,
                message_key=f"qc.{code}",
            )
        ],
    )


def _qc_unavailable(lit: list[SpectralFrame], dark: list[SpectralFrame], cfg: dict) -> QCResult:
    # Without quality checks we cannot vouch for the reading, so it is not interpreted.
    return _failed_qc(
        "spectral_qc_unavailable",
        "Spectral quality checks could not run, so this probe reading was not interpreted.",
    )


def _run_qc(qc_fn: QCFn, inp: SpectralInput, qc_cfg: dict) -> QCResult:
    try:
        return qc_fn(list(inp.lit), list(inp.dark), qc_cfg)
    except Exception:
        return _failed_qc(
            "spectral_qc_error",
            "Spectral quality checks failed on this probe reading; please retake it.",
        )


def find_calibration(
    inp: SpectralInput, calibrations: Mapping[str, SpectralCalibration]
) -> SpectralCalibration | None:
    """The calibration recorded on the reading (same session) first, then the device's latest."""
    cal_id = inp.meta.calibration_id
    if cal_id:
        for cal in calibrations.values():
            if cal.calibration_id == cal_id:
                return cal
    if inp.meta.device_id:
        return calibrations.get(inp.meta.device_id)
    return None


def _spectral_section(analysis_cfg: Mapping[str, Any] | None) -> Mapping[str, Any]:
    cfg = analysis_cfg or {}
    section = cfg.get("spectral")
    return section if isinstance(section, Mapping) else cfg


class _Summary:
    """Per-reading intermediate, before pairing."""

    def __init__(self, inp: SpectralInput, qc: QCResult) -> None:
        self.inp = inp
        self.qc = qc
        self.values: dict[str, float] = {}
        self.std: dict[str, float] = {}
        self.signal_type = "relative_signal"
        self.calibration = CalibrationInfo(status=CalibrationStatus.UNCALIBRATED)
        self.features: SpectralFeatures | None = None


def _summarise(
    inp: SpectralInput,
    calibrations: Mapping[str, SpectralCalibration],
    qc_fn: QCFn,
    qc_cfg: dict,
    now: datetime,
) -> _Summary:
    s = _Summary(inp, _run_qc(qc_fn, inp, qc_cfg))
    try:
        s.values, s.std, s.signal_type, s.calibration = summarise_reading(
            inp, find_calibration(inp, calibrations), now
        )
        s.features = compute_features(s.values)
    except Exception:
        s.qc = _failed_qc(
            "spectral_processing_error",
            "This probe reading could not be processed; please retake it.",
        )
    return s


def _compare(
    s: _Summary, ref_id: str | None, by_id: Mapping[str, _Summary]
) -> tuple[SpectralFeatures | None, list[str]]:
    """Delta vs the reference reading, or None with the reason it is unavailable."""
    if ref_id is None:
        return None, []
    ref = by_id.get(ref_id)
    if ref is None or ref.inp.role != "reference":
        return None, [REFERENCE_MISSING.format(ref=ref_id)]
    if not ref.qc.passed or ref.features is None:
        return None, [REFERENCE_QC_FAILED.format(ref=ref_id)]
    if ref.signal_type != s.signal_type:
        return None, [SIGNAL_MISMATCH]
    delta = reference_delta(s.values, ref.values)
    if delta is None:
        return None, [REFERENCE_UNUSABLE.format(ref=ref_id)]
    return delta, []


def _classify(
    classifier: SpectralClassifier,
    s: _Summary,
    delta: SpectralFeatures | None,
    extra_reasons: list[str],
) -> SpectralClassifierOutput | None:
    if s.inp.role != "lesion":
        return None  # reference readings are the comparison, not a finding
    if not s.qc.passed or s.features is None:
        skipped = getattr(classifier, "skipped_qc_failed", None)
        if callable(skipped):
            return skipped()  # type: ignore[no-any-return]
        return SpectralClassifierOutput(
            status="skipped_qc_failed",
            method=getattr(classifier, "method", classifier.info.name),
            reasons=["The probe reading did not pass quality checks; retake it."],
            model=classifier.info,
        )
    out = classifier.classify(s.features, delta)
    if extra_reasons:
        out = out.model_copy(update={"reasons": [*extra_reasons, *out.reasons]})
    return out


def analyze_spectral(
    inputs: Sequence[SpectralInput],
    calibrations: Mapping[str, SpectralCalibration],
    qc_cfg: dict,
    analysis_cfg: dict,
    now: datetime,
    classifier: SpectralClassifier | None = None,
    qc_fn: QCFn | None = None,
) -> list[SpectralReadingResult]:
    """Analyse every probe reading; results are in the same order as ``inputs``.

    Lesion readings are compared with their reference reading (explicit ``reference_reading_id``
    or the unique contralateral/analogous reference). The classifier is skipped when the reading
    fails QC; a failed or missing reference leaves the lesion without a comparison.
    """
    clf = classifier or RuleBasedSpectralClassifier(_spectral_section(analysis_cfg))
    run_qc = qc_fn or _default_qc_fn()
    summaries = [_summarise(inp, calibrations, run_qc, qc_cfg, now) for inp in inputs]
    by_id = {s.inp.reading_id: s for s in summaries}
    auto = auto_pair_inputs(inputs)

    results: list[SpectralReadingResult] = []
    for s in summaries:
        inp = s.inp
        ref_id = inp.reference_reading_id if inp.role == "lesion" else None
        if inp.role == "lesion" and ref_id is None and inp.reading_id in auto:
            ref_id = auto[inp.reading_id].reference_reading_id
        delta, extra = _compare(s, ref_id, by_id) if s.features is not None else (None, [])
        meta = inp.meta
        if s.calibration.calibration_id and meta.calibration_id != s.calibration.calibration_id:
            meta = meta.model_copy(update={"calibration_id": s.calibration.calibration_id})
        results.append(
            SpectralReadingResult(
                reading_id=inp.reading_id,
                site=inp.site,
                role=inp.role,
                lesion_ref=inp.lesion_ref,
                capture=meta,
                n_readings=len(inp.lit),
                signal_type=s.signal_type,  # type: ignore[arg-type]
                calibration=s.calibration,
                values=s.values,
                values_std=s.std,
                qc=s.qc,
                features=s.features,
                reference_reading_id=ref_id,
                delta_vs_reference=delta,
                classifier=_classify(clf, s, delta, extra),
            )
        )
    return results
