"""Spectral (Confirm probe, AS7341) calibration and diffuse reflectance.

Reflectance R = (S - D) / (W - Dw): the tissue reading minus its dark reading, divided by the
white reflectance standard minus the dark reading taken with it. This removes the LED spectrum,
the sensor's per-channel responsivity and ambient leak, so the 540-580 nm haemoglobin dip can be
compared between sites and sessions. All readings must share gain and integration time (the
firmware fixes them per session); otherwise counts must be rescaled before calling this.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from datetime import datetime

from mukha_pipeline.calibration.common import (
    CalibrationError,
    new_calibration_id,
    validity_window,
)
from mukha_pipeline.types import SpectralCalibration

# White-minus-dark spans at or below this are treated as no signal (division would explode).
SPAN_EPS = 1e-6
# Diffuse reflectance of tissue is well below 1; values up to 1.5 tolerate specular contribution
# and white-standard ageing. Anything beyond is an artefact and is clipped.
REFLECTANCE_MIN = 0.0
REFLECTANCE_MAX = 1.5


def _to_float(value: object) -> float:
    """Float conversion that maps junk (None, strings, NaN, inf) to NaN instead of raising."""
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return math.nan
    return out if math.isfinite(out) else math.nan


def _optional_float(value: object) -> float | None:
    """Optional metadata number; junk or non-finite becomes None (unknown), never NaN."""
    if value is None:
        return None
    out = _to_float(value)
    return None if math.isnan(out) else out


def _clean(channels: Mapping[str, object] | None) -> dict[str, float]:
    """Channel dict with float values; non-numeric values become NaN."""
    if not channels:
        return {}
    return {str(k): _to_float(v) for k, v in channels.items()}


def reflectance(
    sample: Mapping[str, float],
    dark: Mapping[str, float],
    white: Mapping[str, float],
    white_dark: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Per-channel reflectance ``(S - D) / (W - Dw)`` for every channel of ``sample``.

    ``Dw`` defaults to ``D``. A channel is NaN when any term is missing or non-finite, or when
    ``W - Dw <= SPAN_EPS`` (no white signal: dividing would turn noise into a number). Finite
    results are clipped to [0, 1.5]. NaN never becomes a plausible-looking value.
    """
    s_vals = _clean(sample)
    d_vals = _clean(dark)
    w_vals = _clean(white)
    dw_vals = _clean(white_dark) if white_dark is not None else d_vals
    out: dict[str, float] = {}
    for ch, s in s_vals.items():
        d = d_vals.get(ch, math.nan)
        w = w_vals.get(ch, math.nan)
        dw = dw_vals.get(ch, math.nan)
        span = w - dw
        if math.isnan(s) or math.isnan(d) or math.isnan(span) or span <= SPAN_EPS:
            out[ch] = math.nan
            continue
        out[ch] = min(max((s - d) / span, REFLECTANCE_MIN), REFLECTANCE_MAX)
    return out


def create_spectral_calibration(
    device_id: str,
    dark: Mapping[str, float],
    white: Mapping[str, float],
    *,
    now: datetime | None,
    valid_hours: float = 24,
    gain: float | None = None,
    integration_ms: float | None = None,
    synthetic: bool = False,
) -> SpectralCalibration:
    """Record a white-standard calibration (``dark`` = dark reading taken with the white one).

    Raises ``CalibrationError`` when no channel has white signal above dark, which means the
    probe was not seated on the standard or the LED did not fire.
    """
    dark_vals = _clean(dark)
    white_vals = _clean(white)
    usable = [
        ch
        for ch, w in white_vals.items()
        if not math.isnan(w)
        and not math.isnan(dark_vals.get(ch, math.nan))
        and w - dark_vals[ch] > SPAN_EPS
    ]
    if not usable:
        raise CalibrationError(
            "White reference reading is not brighter than the dark reading on any channel; "
            "seat the probe flat on the white standard and repeat the calibration."
        )
    created_at, expires_at = validity_window(now, valid_hours)
    return SpectralCalibration(
        calibration_id=new_calibration_id(),
        device_id=device_id,
        created_at=created_at,
        expires_at=expires_at,
        dark=dark_vals,
        white=white_vals,
        gain=_optional_float(gain),
        integration_ms=_optional_float(integration_ms),
        synthetic=synthetic,
    )
