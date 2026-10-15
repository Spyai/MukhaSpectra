"""Spectral features of oral mucosa from AS7341 band values, and the lesion-vs-reference delta.

Optics behind each feature (diffuse reflectance of mucosa under a white LED):

* Haemoglobin dominates visible absorption. Oxy-haemoglobin has the Soret band at ~415 nm and the
  Q bands at 542 and 577 nm; absorption falls by two orders of magnitude beyond ~600 nm. More
  blood (angiogenesis, inflammation) therefore darkens 415 and 540-590 nm while 630-680 nm stays
  bright, so 555/630 and 415/555 fall and the 540-580 nm "dip" deepens.
* ``hb_absorption_proxy`` measures that dip against a straight baseline from 515 to 630 nm, so a
  uniform brightness change (probe pressure, gain) cancels.
* Log-reflectance slopes summarise the spectral shape: 480->555 nm tracks haemoglobin absorption
  rising into the Q bands, 590->680 nm the red recovery that scattering and oxygenation shape.

Absolute features (:func:`compute_features`, the ``features`` field of a reading)
---------------------------------------------------------------------------------
Computed on the reading's own values. For calibrated reflectance they describe the tissue. For an
uncalibrated relative signal every channel still carries the probe's per-channel gain
``K_c`` = LED output x filter transmission x sensor response in that band, so the absolute ratios
and the haemoglobin dip of a relative-signal reading depend on the device (its LED batch, its
filters) and must not be compared with fixed thresholds or with another probe's readings.

Lesion-vs-reference delta (:func:`reference_delta`, the ``delta_vs_reference`` field)
-------------------------------------------------------------------------------------
Every channel of the lesion reading is divided by the same channel of the same patient's
reference reading, giving the quotient spectrum ``q_c = lesion_c / reference_c``. Both readings
were taken with the same probe and processed the same way, so for a relative signal

    q_c = (K_c R_lesion,c / S_lesion) / (K_c R_reference,c / S_reference)
        = (R_lesion,c / R_reference,c) x (S_reference / S_lesion)

``K_c`` cancels exactly, and what is left is the tissue reflectance quotient times one constant
that is the same for every channel. For calibrated reflectance ``q_c = R_lesion,c / R_reference,c``
directly. Every feature below is unchanged by an overall scale, so calibrated and uncalibrated
readings of the same tissue give the same delta (to within measurement noise) and fixed thresholds
mean the same thing on every probe. (Left over is only a second-order effect: within one filter
band the LED shape weights the tissue reflectance slightly differently from device to device.)

The delta is ``features(q)`` minus ``features`` of an all-ones spectrum (no difference):

* ratios: ``ratio(q) - 1 = ratio_lesion / ratio_reference - 1``, the relative change of the ratio
  (``ratio_555_630 = -0.10`` means the lesion's 555/630 ratio is 10 % below the reference's);
* ``hb_absorption_proxy``: the haemoglobin dip of ``q`` itself (positive = the lesion has a deeper
  540-590 nm dip than the reference; 0 = same shape);
* slopes: ``slope(q) = slope_lesion - slope_reference`` (log-slopes subtract exactly).

A feature whose channels are missing from either reading is 0 in the delta (no evidence either
way); without usable 515, 555, 590 and 630 nm signal in both readings there is no delta at all.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from mukha_pipeline.schema.models import AS7341_WAVELENGTH_NM, SpectralFeatures

EPS = 1e-6
FEATURE_NAMES: tuple[str, ...] = tuple(SpectralFeatures.model_fields)
_BAND_480_555 = ("F3_480", "F4_515", "F5_555")
_BAND_590_680 = ("F6_590", "F7_630", "F8_680")
# Channels behind the scored features (haemoglobin dip and 555/630 ratio).
REQUIRED_CHANNELS: tuple[str, ...] = ("F4_515", "F5_555", "F6_590", "F7_630")


def _value(values: Mapping[str, float], channel: str) -> float:
    try:
        v = float(values.get(channel, np.nan))
    except (TypeError, ValueError):
        return float("nan")
    return v if np.isfinite(v) else float("nan")


def _ratio(values: Mapping[str, float], num: str, den: str) -> float:
    """num / den; 0.0 when either is missing or the denominator has no signal."""
    a, b = _value(values, num), _value(values, den)
    if not (np.isfinite(a) and np.isfinite(b)) or b <= EPS:
        return 0.0
    return float(a / b)


def hb_absorption_proxy(values: Mapping[str, float]) -> float:
    """1 - (v555 + v590) / (b555 + b590), b = straight line through (515, v515) and (630, v630).

    0 means no dip below the baseline; larger values mean deeper haemoglobin absorption.
    """
    v515, v555 = _value(values, "F4_515"), _value(values, "F5_555")
    v590, v630 = _value(values, "F6_590"), _value(values, "F7_630")
    if not all(np.isfinite(v) for v in (v515, v555, v590, v630)):
        return 0.0
    slope = (v630 - v515) / (630.0 - 515.0)
    baseline = (v515 + slope * (555.0 - 515.0)) + (v515 + slope * (590.0 - 515.0))
    if baseline <= EPS:
        return 0.0
    return float(1.0 - (v555 + v590) / baseline)


def log_slope(values: Mapping[str, float], channels: Sequence[str]) -> float:
    """Least-squares slope of ln(value) against wavelength, per 100 nm.

    Channels without a positive finite value are skipped; fewer than two usable channels -> 0.0.
    """
    pts = [(AS7341_WAVELENGTH_NM[c] / 100.0, _value(values, c)) for c in channels]
    pts = [(x, y) for x, y in pts if np.isfinite(y) and y > EPS]
    if len(pts) < 2:
        return 0.0
    x = np.array([p[0] for p in pts])
    y = np.log(np.array([p[1] for p in pts]))
    slope = float(np.polyfit(x, y, 1)[0])
    return slope if np.isfinite(slope) else 0.0


def compute_features(values: Mapping[str, float]) -> SpectralFeatures:
    """Band ratios, haemoglobin dip and log-slopes. NaN-safe: undefined features are 0.0."""
    return SpectralFeatures(
        ratio_555_630=_ratio(values, "F5_555", "F7_630"),
        ratio_590_680=_ratio(values, "F6_590", "F8_680"),
        ratio_515_630=_ratio(values, "F4_515", "F7_630"),
        ratio_415_555=_ratio(values, "F1_415", "F5_555"),
        hb_absorption_proxy=hb_absorption_proxy(values),
        slope_480_555=log_slope(values, _BAND_480_555),
        slope_590_680=log_slope(values, _BAND_590_680),
    )


def delta_features(lesion: SpectralFeatures, reference: SpectralFeatures) -> SpectralFeatures:
    """``lesion - reference``, field by field.

    A plain difference of two feature sets. Lesion-vs-reference comparisons use
    :func:`reference_delta`, which is device-invariant; this difference of absolute features is
    not, for relative-signal readings.
    """
    return SpectralFeatures(
        **{
            name: float(getattr(lesion, name)) - float(getattr(reference, name))
            for name in FEATURE_NAMES
        }
    )


def quotient_spectrum(
    lesion: Mapping[str, float], reference: Mapping[str, float]
) -> dict[str, float]:
    """Per-channel ``lesion / reference``.

    Channels missing from either reading, not finite, or with no reference signal are left out.
    """
    out: dict[str, float] = {}
    for channel in lesion:
        num, den = _value(lesion, channel), _value(reference, channel)
        if np.isfinite(num) and np.isfinite(den) and den > EPS:
            out[channel] = max(num, 0.0) / den
    return out


def reference_delta(
    lesion: Mapping[str, float], reference: Mapping[str, float]
) -> SpectralFeatures | None:
    """Device-invariant lesion-vs-reference delta (see the module docstring).

    ``lesion`` and ``reference`` are per-channel values of two readings processed the same way
    (both reflectance or both relative signal). Returns ``features(q) - features(ones)`` with
    ``q = lesion / reference`` per channel, or None when the reference has no usable signal in a
    channel the scored features need.
    """
    q = quotient_spectrum(lesion, reference)
    if any(channel not in q for channel in REQUIRED_CHANNELS):
        return None
    # The all-ones spectrum over the same channels: a feature whose channels are missing is 0 in
    # both, so its delta is 0 rather than a spurious -1.
    return delta_features(compute_features(q), compute_features(dict.fromkeys(q, 1.0)))
