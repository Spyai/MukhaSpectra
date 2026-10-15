"""QC issue catalogue: stable codes, default severities and plain-language retake prompts.

Every prompt is written for a health worker holding the phone at a screening camp: imperative,
at most 12 words, says what went wrong and what to do next. The i18n key is always
``qc.<code>``; translations live in the frontend. Prompts describe capture quality only and
never say anything about the patient's condition.

Severity semantics (see ``QCSeverity``): ERROR blocks analysis of the capture, WARNING lets the
analysis run but is reported as a limitation, INFO is recorded only.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from mukha_pipeline.schema.models import QCIssue, QCSeverity

_E, _W = QCSeverity.ERROR, QCSeverity.WARNING

MAX_PROMPT_WORDS = 12


@dataclass(frozen=True)
class IssueSpec:
    code: str
    severity: QCSeverity  # default; checks may escalate or relax it via make_issue(severity=)
    message: str

    @property
    def message_key(self) -> str:
        return f"qc.{self.code}"


_SPECS: tuple[IssueSpec, ...] = (
    # --- any image: input could not be checked properly -------------------------------------
    IssueSpec("image_invalid", _E, "Photo could not be checked — retake the photo."),
    IssueSpec("image_too_small", _E, "Photo too small — retake at full camera resolution."),
    IssueSpec("image_grayscale", _E, "Photo has no colour — switch camera to colour, retake."),
    # --- white light ---------------------------------------------------------------------------
    IssueSpec("blur", _E, "White-light photo blurry — hold the phone steady and retake."),
    IssueSpec("too_dark", _E, "White-light photo too dark — switch on the light, retake."),
    IssueSpec("too_bright", _E, "White-light photo too bright — move back slightly and retake."),
    IssueSpec("clipping_dark", _W, "Parts of photo too dark — light the whole area evenly."),
    IssueSpec("clipping_bright", _E, "Parts of photo washed out — reduce the light and retake."),
    IssueSpec("glare", _E, "Too much shine — tilt the phone slightly and retake."),
    IssueSpec("glare_minor", _W, "Some shine on tissue — tilt the phone slightly next time."),
    # --- autofluorescence (405 nm violet excitation) -----------------------------------------
    IssueSpec("af_signal_low", _E, "Violet-light photo too dim — switch on violet light, retake."),
    IssueSpec("af_filter_leak", _E, "Violet-light photo too blue — fit the yellow filter, retake."),
    IssueSpec("af_overexposed", _E, "Violet-light photo too bright — move back slightly, retake."),
    IssueSpec("af_blur", _E, "Violet-light photo blurry — hold the phone steady and retake."),
    IssueSpec("af_white_light", _E, "Looks like a white-light photo — retake with violet light."),
    # Default warning (weakly green); escalated to an error when green is not dominant at all
    # (grey from auto white balance or a greyscale file, or red-dominant).
    IssueSpec(
        "af_not_green",
        _W,
        "Violet-light photo not green — lock colour settings, check filter, retake.",
    ),
    IssueSpec(
        "af_auto_white_balance",
        _W,
        "Lock white balance for fluorescence photos — auto white balance alters colours.",
    ),
    # --- dark frame ----------------------------------------------------------------------------
    IssueSpec("ambient_light", _E, "Dark photo too bright — block room light and retake."),
    IssueSpec("ambient_light_minor", _W, "Some room light seen — dim lights or shade the mouth."),
    IssueSpec("no_dark_frame", _W, "Dark photo missing — take one with all lights off."),
    # --- registration / segmentation -----------------------------------------------------------
    IssueSpec("misaligned", _E, "Photos do not line up — keep still between both photos."),
    IssueSpec("alignment_marginal", _W, "Photos line up poorly — keep steadier between photos."),
    IssueSpec("insufficient_mucosa", _E, "Too little tissue in view — open wider, move closer."),
    IssueSpec(
        "segmentation_failed",
        _E,
        "Tissue not recognised in photo — retake in good light, mouth open.",
    ),
    IssueSpec(
        "segmentation_low_confidence",
        _W,
        "Tissue outline unclear — retake in even light if possible.",
    ),
    # --- spectral probe (AS7341) ---------------------------------------------------------------
    IssueSpec("spectral_no_readings", _E, "No probe readings — repeat the probe measurement."),
    IssueSpec(
        "spectral_few_readings", _W, "Too few probe readings — repeat the probe measurement."
    ),
    IssueSpec(
        "spectral_channels_missing", _E, "Probe data incomplete — reconnect the probe, repeat."
    ),
    IssueSpec("spectral_checks_skipped", _W, "Probe contact not checked — hold it flat on tissue."),
    IssueSpec("spectral_saturated", _E, "Probe signal too strong — lower probe gain and repeat."),
    IssueSpec(
        "spectral_ambient_leak", _E, "Room light reaching probe — press it gently on tissue."
    ),
    IssueSpec("probe_no_contact", _E, "Probe not touching tissue — place it flat and repeat."),
    IssueSpec("spectral_signal_low", _W, "Probe signal too weak — check probe light and repeat."),
    IssueSpec(
        "spectral_poor_repeatability", _W, "Probe readings unsteady — hold the probe still, repeat."
    ),
    IssueSpec(
        "spectral_settings_mismatch", _W, "Probe settings changed mid-reading — repeat measurement."
    ),
)

ISSUE_CATALOGUE: Mapping[str, IssueSpec] = MappingProxyType({s.code: s for s in _SPECS})


def issue_spec(code: str) -> IssueSpec:
    """Look up a catalogue entry; an unknown code is a programming error, so fail loudly."""
    try:
        return ISSUE_CATALOGUE[code]
    except KeyError:
        raise KeyError(f"unknown QC issue code {code!r}") from None


def _finite_or_none(x: float | None) -> float | None:
    if x is None:
        return None
    x = float(x)
    return x if math.isfinite(x) else None


def make_issue(
    code: str,
    *,
    metric: str | None = None,
    value: float | None = None,
    threshold: float | None = None,
    severity: QCSeverity | None = None,
) -> QCIssue:
    """Build a QCIssue from the catalogue. Non-finite numbers become None (JSON-safe)."""
    spec = issue_spec(code)
    return QCIssue(
        code=spec.code,
        severity=severity or spec.severity,
        message=spec.message,
        message_key=spec.message_key,
        metric=metric,
        value=_finite_or_none(value),
        threshold=_finite_or_none(threshold),
    )


def missing_dark_frame_issue() -> QCIssue:
    """No dark frame: analysis still runs, but room light cannot be checked or subtracted."""
    return make_issue("no_dark_frame")
