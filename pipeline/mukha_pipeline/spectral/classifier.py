"""Spectral classifier interface and the rule-based placeholder.

The rules compare a lesion reading with the same patient's reference reading. They are
heuristics, not a trained model: thresholds live in ``analysis.v1.yaml`` (``spectral.classifier``)
and the score is a monotone transform of the deltas, not a probability. A learned model (logistic
regression / small MLP from ``ml/``) can replace this behind the same ``SpectralClassifier``
interface.

Clinical stance: the rules can raise suspicion but cannot clear a lesion. A "not suspicious"
spectral label always carries the reminder that it does not rule out disease.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

from mukha_pipeline.schema.models import (
    ModelComponent,
    ModelInfo,
    ModelStatus,
    SpectralClassifierOutput,
    SpectralFeatures,
)

NO_REFERENCE_REASON = (
    "No reference reading from the opposite side; spectral comparison not possible."
)
SCORE_NOTE = "Heuristic suspicion score {score:.2f} (not a calibrated probability)."
QC_FAILED_REASON = "The probe reading did not pass quality checks; retake it before interpreting."

_DEFAULTS: dict[str, float | str] = {
    "method": "rules_v1",
    "hb_proxy_delta_suspicious": 0.06,
    "hb_proxy_delta_indeterminate": 0.03,
    "ratio_555_630_delta_suspicious": -0.10,
    "ratio_555_630_delta_indeterminate": -0.05,
}


@runtime_checkable
class SpectralClassifier(Protocol):
    info: ModelInfo

    def classify(
        self, features: SpectralFeatures, delta: SpectralFeatures | None
    ) -> SpectralClassifierOutput: ...


def _rules(cfg: Mapping[str, Any] | None) -> dict[str, Any]:
    """Accept the full analysis config, its 'spectral' section, or the 'classifier' block."""
    section: Mapping[str, Any] = cfg or {}
    if "spectral" in section and isinstance(section["spectral"], Mapping):
        section = section["spectral"]
    if "classifier" in section and isinstance(section["classifier"], Mapping):
        section = section["classifier"]
    return {**_DEFAULTS, **dict(section)}


class RuleBasedSpectralClassifier:
    """Thresholds on lesion-minus-reference deltas of the haemoglobin dip and 555/630 ratio."""

    def __init__(self, cfg: Mapping[str, Any] | None = None) -> None:
        rules = _rules(cfg)
        self.method = str(rules["method"])
        self.hb_suspicious = float(rules["hb_proxy_delta_suspicious"])
        self.hb_indeterminate = float(rules["hb_proxy_delta_indeterminate"])
        self.ratio_suspicious = float(rules["ratio_555_630_delta_suspicious"])
        self.ratio_indeterminate = float(rules["ratio_555_630_delta_indeterminate"])
        self.info = ModelInfo(
            component=ModelComponent.SPECTRAL_CLASSIFIER,
            name="spectral-rules",
            version="1.0.0",
            status=ModelStatus.PLACEHOLDER,
            validated=False,
            notes="Heuristic rules on lesion-vs-reference deltas; not trained or validated.",
        )

    # ----------------------------------------------------------------------------------------

    def placeholder(self, reasons: list[str]) -> SpectralClassifierOutput:
        return SpectralClassifierOutput(
            status="placeholder",
            method=self.method,
            suspicion_score=None,
            label="indeterminate",
            reasons=reasons,
            model=self.info,
        )

    def skipped_qc_failed(self) -> SpectralClassifierOutput:
        return SpectralClassifierOutput(
            status="skipped_qc_failed",
            method=self.method,
            reasons=[QC_FAILED_REASON],
            model=self.info,
        )

    def score(self, hb_delta: float, ratio_delta: float) -> float:
        """0.5 at a 'suspicious' threshold, 1.0 at twice it; the larger of the two cues."""
        s_hb = 0.5 * hb_delta / self.hb_suspicious if self.hb_suspicious > 0 else 0.0
        s_ratio = 0.5 * ratio_delta / self.ratio_suspicious if self.ratio_suspicious < 0 else 0.0
        return float(min(1.0, max(0.0, s_hb, s_ratio)))

    def classify(
        self, features: SpectralFeatures, delta: SpectralFeatures | None
    ) -> SpectralClassifierOutput:
        if delta is None:
            return self.placeholder([NO_REFERENCE_REASON])
        hb, ratio = float(delta.hb_absorption_proxy), float(delta.ratio_555_630)
        if not (math.isfinite(hb) and math.isfinite(ratio)):
            return self.placeholder([NO_REFERENCE_REASON])

        reasons: list[str] = []
        hb_text = (
            f"The 540-580 nm haemoglobin absorption dip is {hb:+.3f} versus the reference reading"
        )
        ratio_text = f"The 555/630 nm ratio is {ratio:+.3f} versus the reference reading"
        if hb >= self.hb_suspicious:
            reasons.append(
                f"{hb_text} (suspicious at >= {self.hb_suspicious:+.3f}), consistent with "
                "increased blood volume."
            )
        elif hb >= self.hb_indeterminate:
            reasons.append(f"{hb_text} (borderline at >= {self.hb_indeterminate:+.3f}).")
        if ratio <= self.ratio_suspicious:
            reasons.append(
                f"{ratio_text} (suspicious at <= {self.ratio_suspicious:+.3f}), consistent with "
                "stronger green absorption by haemoglobin."
            )
        elif ratio <= self.ratio_indeterminate:
            reasons.append(f"{ratio_text} (borderline at <= {self.ratio_indeterminate:+.3f}).")

        if hb >= self.hb_suspicious or ratio <= self.ratio_suspicious:
            label = "suspicious"
        elif hb >= self.hb_indeterminate or ratio <= self.ratio_indeterminate:
            label = "indeterminate"
        else:
            label = "not_suspicious"
            reasons.append(
                "No spectral difference from the reference reading above the heuristic "
                "thresholds. This does not rule out disease; the visual and fluorescence "
                "findings still apply."
            )
        score = self.score(hb, ratio)
        reasons.append(SCORE_NOTE.format(score=score))
        return SpectralClassifierOutput(
            status="ok",
            method=self.method,
            suspicion_score=score,
            label=label,  # type: ignore[arg-type]
            reasons=reasons,
            model=self.info,
        )
