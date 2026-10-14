"""Spectral features of oral mucosa from AS7341 band values.

Optics behind each feature (diffuse reflectance of mucosa under a white LED):

* Haemoglobin dominates visible absorption. Oxy-haemoglobin has the Soret band at ~415 nm and the
  Q bands at 542 and 577 nm; absorption falls by two orders of magnitude beyond ~600 nm. More
  blood (angiogenesis, inflammation) therefore darkens 415 and 540-590 nm while 630-680 nm stays
  bright, so 555/630 and 415/555 fall and the 540-580 nm "dip" deepens.
* ``hb_absorption_proxy`` measures that dip against a straight baseline from 515 to 630 nm, so a
  uniform brightness change (probe pressure, gain) cancels.
* Log-reflectance slopes summarise the spectral shape: 480->555 nm tracks haemoglobin absorption
  rising into the Q bands, 590->680 nm the red recovery that scattering and oxygenation shape.

Every feature is a ratio or a log-slope, so it is equally meaningful for calibrated reflectance
and for the uncalibrated relative signal when compared with the same patient's reference reading.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from mukha_pipeline.schema.models import AS7341_WAVELENGTH_NM, SpectralFeatures

EPS = 1e-6
FEATURE_NAMES: tuple[str, ...] = tuple(SpectralFeatures.model_fields)
_BAND_480_555 = ("F3_480", "F4_515", "F5_555")
_BAND_590_680 = ("F6_590", "F7_630", "F8_680")


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
    """Lesion minus reference, field by field (positive hb delta = deeper dip in the lesion)."""
    return SpectralFeatures(
        **{
            name: float(getattr(lesion, name)) - float(getattr(reference, name))
            for name in FEATURE_NAMES
        }
    )
