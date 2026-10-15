"""Spectral classifier interface and the rule-based placeholder.

The rules compare a lesion reading with the same patient's reference reading through the
device-invariant delta of :func:`mukha_pipeline.spectral.features.reference_delta`: the depth of
the 540-590 nm haemoglobin dip of the lesion / reference quotient, and the relative change of the
555/630 nm ratio. They are heuristics, not a trained model: thresholds live in
``analysis.v1.yaml`` (``spectral.classifier``, the same cut-offs as the triage tiers in
``triage_rules.v1.yaml`` ``lesion.spectral``) and the score is a monotone transform of the deltas,
not a probability. A learned model (logistic regression / small MLP from ``ml/``) can replace this
behind the same ``SpectralClassifier`` interface.

Every classification (and the no-reference output) has status ``placeholder``, never ``ok``;
a classification's first reason says it is a placeholder heuristic, not a trained model, not
validated.

Clinical stance: the rules can raise suspicion but cannot clear a lesion. The ``not_suspicious``
label means only "no spectral difference detected (this does not rule out disease)" and is
always worded that way; it is never described as normal.
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
PLACEHOLDER_REASON = (
    "Placeholder heuristic: fixed rules on the difference between this reading and the reference "
    "reading; not a trained model, not validated."
)
NO_DIFFERENCE_REASON = (
    "No spectral difference detected versus the reference reading (this does not rule out "
    "disease). The visual and fluorescence findings still apply."
)
MODEL_NOTES = (
    "Heuristic rules on lesion-vs-reference differences; placeholder, not a trained model, "
    "not validated."
)
MODEL_VERSION = "2.0.0"  # 2.0.0: deltas on the lesion / reference quotient (device-invariant)

_DEFAULTS: dict[str, float | str] = {
    "method": "rules_v2",
    "hb_proxy_delta_suspicious": 0.06,
    "hb_proxy_delta_indeterminate": 0.025,
    "ratio_555_630_delta_suspicious": -0.10,
    "ratio_555_630_delta_indeterminate": -0.04,
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
    """Placeholder thresholds on the lesion-vs-reference haemoglobin dip and 555/630 ratio."""

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
            version=MODEL_VERSION,
            status=ModelStatus.PLACEHOLDER,
            validated=False,
            notes=MODEL_NOTES,
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

        reasons: list[str] = [PLACEHOLDER_REASON]
        hb_text = (
            f"The 540-580 nm haemoglobin absorption dip versus the reference reading is {hb:+.3f}"
        )
        ratio_text = f"The 555/630 nm ratio is {ratio:+.1%} relative to the reference reading"
        if hb >= self.hb_suspicious:
            reasons.append(
                f"{hb_text} (suspicious at >= {self.hb_suspicious:+.3f}), consistent with "
                "increased blood volume."
            )
        elif hb >= self.hb_indeterminate:
            reasons.append(f"{hb_text} (borderline at >= {self.hb_indeterminate:+.3f}).")
        if ratio <= self.ratio_suspicious:
            reasons.append(
                f"{ratio_text} (suspicious at <= {self.ratio_suspicious:+.1%}), consistent with "
                "stronger green absorption by haemoglobin."
            )
        elif ratio <= self.ratio_indeterminate:
            reasons.append(f"{ratio_text} (borderline at <= {self.ratio_indeterminate:+.1%}).")

        if hb >= self.hb_suspicious or ratio <= self.ratio_suspicious:
            label = "suspicious"
        elif hb >= self.hb_indeterminate or ratio <= self.ratio_indeterminate:
            label = "indeterminate"
        else:
            label = "not_suspicious"
            reasons.append(NO_DIFFERENCE_REASON)
        score = self.score(hb, ratio)
        reasons.append(SCORE_NOTE.format(score=score))
        return SpectralClassifierOutput(
            # A placeholder heuristic, so never reported as a model output ("ok").
            status="placeholder",
            method=self.method,
            suspicion_score=score,
            label=label,  # type: ignore[arg-type]
            reasons=reasons,
            model=self.info,
        )
