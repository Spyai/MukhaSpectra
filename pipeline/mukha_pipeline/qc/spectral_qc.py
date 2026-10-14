"""Spectral QC for the Confirm probe (AS7341: F1-F8 visible bands, CLEAR, NIR).

A reading is N illuminated (lit) frames plus dark frames taken with the LED off. The checks
follow the optics of a contact diffuse-reflectance probe:

* saturation    - a clipped channel flattens the 540-580 nm haemoglobin dip we want to measure;
* ambient leak  - room light reaching the sensor shows up in the dark CLEAR counts;
* contact       - in contact, the LED's light comes back from tissue, so lit-dark CLEAR is large;
                  room light is rich in NIR, so a high dark NIR means the probe is not sealed;
* signal        - each band needs enough net counts above the dark level to be trusted. F1_415 is
                  often the weakest (Soret-band haemoglobin absorption, little LED output at
                  415 nm), so one or two weak bands are a warning and a mostly weak spectrum is an
                  error;
* repeatability - frames should agree when the probe is held still (coefficient of variation).

Frames may lack channels (old firmware, partial CSV rows) or carry NaN / junk; that becomes an
issue, never an exception.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from mukha_pipeline.qc.messages import make_issue
from mukha_pipeline.qc.result import build_result
from mukha_pipeline.schema.models import (
    AS7341_CHANNELS,
    AS7341_SPECTRAL_CHANNELS,
    QCIssue,
    QCResult,
    QCSeverity,
)
from mukha_pipeline.types import SpectralFrame

_IDX = {name: i for i, name in enumerate(AS7341_CHANNELS)}
_BANDS = np.array([_IDX[name] for name in AS7341_SPECTRAL_CHANNELS])


def _num(x: object) -> float:
    try:
        v = float(x)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return math.nan
    return v if math.isfinite(v) else math.nan


def _value(channels: object, name: str) -> float:
    try:
        return _num(channels[name])  # type: ignore[index]
    except (KeyError, TypeError, IndexError):
        return math.nan


def frame_matrix(frames: Sequence[SpectralFrame], names: Sequence[str]) -> np.ndarray:
    """(n_frames, n_channels) float64 of raw counts; NaN where a channel is missing or bad."""
    rows = [[_value(getattr(f, "channels", None), n) for n in names] for f in frames]
    return np.array(rows, dtype=np.float64).reshape(len(frames), len(names))


def _nanmean(m: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN column -> NaN, by design
        return np.nanmean(m, axis=0)


def _flag(
    when: bool,
    code: str,
    metric: str,
    value: float,
    threshold: float,
    severity: QCSeverity | None = None,
) -> list[QCIssue]:
    if not when:
        return []
    return [make_issue(code, metric=metric, value=value, threshold=threshold, severity=severity)]


@dataclass(frozen=True)
class _Reading:
    """Raw counts of one reading: lit (N, 10), dark (M, 10) and their per-channel means."""

    lit: np.ndarray
    dark: np.ndarray
    lit_mean: np.ndarray
    dark_mean: np.ndarray

    @classmethod
    def of(cls, lit: Sequence[SpectralFrame], dark: Sequence[SpectralFrame]) -> _Reading:
        L, D = frame_matrix(lit, AS7341_CHANNELS), frame_matrix(dark, AS7341_CHANNELS)
        return cls(L, D, _nanmean(L), _nanmean(D))

    def present(self, name: str) -> bool:
        i = _IDX[name]
        return bool(np.isfinite(self.lit_mean[i]) and np.isfinite(self.dark_mean[i]))

    @property
    def net_bands(self) -> np.ndarray:
        return self.lit_mean[_BANDS] - self.dark_mean[_BANDS]


def _check_bands(r: _Reading) -> list[QCIssue]:
    missing = sum(not r.present(n) for n in AS7341_SPECTRAL_CHANNELS)
    return _flag(missing > 0, "spectral_channels_missing", "spectral.missing_bands", missing, 0)


def _check_saturation(r: _Reading, c: Mapping, metrics: dict[str, float]) -> list[QCIssue]:
    raw = np.concatenate([r.lit.ravel(), r.dark.ravel()])
    raw = raw[np.isfinite(raw)]
    fraction = (float(raw.max()) if raw.size else 0.0) / float(c["adc_full_scale"])
    metrics["spectral.max_raw_fraction"] = fraction
    limit = c["saturation_fraction"]
    return _flag(
        fraction >= limit, "spectral_saturated", "spectral.max_raw_fraction", fraction, limit
    )


def _check_clear(r: _Reading, c: Mapping, metrics: dict[str, float]) -> list[QCIssue]:
    """Ambient leak (dark/lit CLEAR) and contact (lit-dark CLEAR)."""
    lit_clear, dark_clear = r.lit_mean[_IDX["CLEAR"]], r.dark_mean[_IDX["CLEAR"]]
    # With nothing coming back while lit, any dark light at all is pure leak.
    fallback = math.inf if dark_clear > 0 else 0.0
    ratio = dark_clear / lit_clear if lit_clear > 0 else fallback
    net = lit_clear - dark_clear
    metrics["spectral.ambient_ratio"], metrics["spectral.clear_net"] = ratio, net
    leak_max, contact_min = c["ambient_leak_ratio_max"], c["contact_clear_min"]
    return _flag(
        ratio > leak_max, "spectral_ambient_leak", "spectral.ambient_ratio", ratio, leak_max
    ) + _flag(net < contact_min, "probe_no_contact", "spectral.clear_net", net, contact_min)


def _check_dark_nir(nir: float, c: Mapping, metrics: dict[str, float]) -> list[QCIssue]:
    metrics["spectral.dark_nir"] = nir
    limit = c["contact_nir_dark_max"]
    return _flag(nir > limit, "probe_no_contact", "spectral.dark_nir", nir, limit)


def _check_signal(net: np.ndarray, c: Mapping, metrics: dict[str, float]) -> list[QCIssue]:
    available = np.isfinite(net)
    if not available.any():
        return []
    weak = available & (net < c["signal_min"])
    lowest = float(np.min(net[available]))
    metrics["spectral.min_net_signal"], metrics["spectral.weak_bands"] = lowest, float(weak.sum())
    mostly_weak = weak.sum() > available.sum() / 2
    return _flag(
        bool(weak.any()),
        "spectral_signal_low",
        "spectral.min_net_signal",
        lowest,
        c["signal_min"],
        QCSeverity.ERROR if mostly_weak else None,
    )


def max_band_cv(r: _Reading, signal_min: float) -> float:
    """Largest coefficient of variation of dark-subtracted lit counts across the N frames.

    Bands below ``signal_min`` net counts are left out: their CV explodes from noise alone and
    they are already reported by the signal check. NaN when fewer than 2 frames are usable.
    """
    cvs: list[float] = []
    for j in _BANDS:
        col = r.lit[:, j] - r.dark_mean[j]
        col = col[np.isfinite(col)]
        if col.size >= 2 and col.mean() >= signal_min:
            cvs.append(float(col.std(ddof=1) / col.mean()))
    return max(cvs, default=math.nan)


def _settings_differ(frames: Sequence[SpectralFrame]) -> bool:
    """Dark subtraction is only valid at the same gain and integration time as the lit frames."""
    for attr in ("gain", "integration_ms"):
        values = {round(v, 6) for f in frames if math.isfinite(v := _num(getattr(f, attr, None)))}
        if len(values) > 1:
            return True
    return False


def qc_spectral(
    lit: Sequence[SpectralFrame], dark: Sequence[SpectralFrame], cfg: Mapping
) -> QCResult:
    """Quality of one probe reading (N lit + dark frames). ``cfg`` is the full qc config.

    Issues are ordered by what the health worker should fix first. When the CLEAR check already
    says the probe is not touching tissue, the dark-NIR and weak-signal prompts are consequences
    of the same fault and are not repeated.
    """
    c = cfg["spectral"]
    lit, dark = list(lit or []), list(dark or [])
    metrics: dict[str, float] = {"spectral.n_lit": len(lit), "spectral.n_dark": len(dark)}
    if not lit or not dark:
        metric = "spectral.n_lit" if not lit else "spectral.n_dark"
        issue = make_issue(
            "spectral_no_readings", metric=metric, value=min(len(lit), len(dark)), threshold=1
        )
        return build_result([issue], metrics, cfg)

    r = _Reading.of(lit, dark)
    issues = _check_bands(r) + _check_saturation(r, c, metrics)
    contact: list[QCIssue] = []
    if r.present("CLEAR"):
        contact = _check_clear(r, c, metrics)
        issues += contact
    nir_dark = r.dark_mean[_IDX["NIR"]]
    no_contact = any(i.code == "probe_no_contact" for i in contact)
    if np.isfinite(nir_dark):
        nir_issues = _check_dark_nir(float(nir_dark), c, metrics)
        issues += [] if no_contact else nir_issues
    signal_issues = _check_signal(r.net_bands, c, metrics)
    issues += [] if no_contact else signal_issues

    if len(lit) >= 2:
        cv = max_band_cv(r, c["signal_min"])
        metrics["spectral.max_cv"] = cv
        limit = c["repeatability_cv_max"]
        issues += _flag(cv > limit, "spectral_poor_repeatability", "spectral.max_cv", cv, limit)
    n_min = c["min_readings"]
    issues += _flag(len(lit) < n_min, "spectral_few_readings", "spectral.n_lit", len(lit), n_min)
    skipped = (not r.present("CLEAR")) + (not np.isfinite(nir_dark))
    # Contact / ambient checks could not run: say so rather than pass silently.
    issues += _flag(
        skipped > 0, "spectral_checks_skipped", "spectral.missing_aux_channels", skipped, 0
    )
    if _settings_differ(lit + dark):
        issues.append(make_issue("spectral_settings_mismatch", metric="spectral.settings"))
    return build_result(issues, metrics, cfg)
