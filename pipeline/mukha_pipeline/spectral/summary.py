"""Turn N raw AS7341 frames into one spectrum per reading.

Calibrated path (white standard measured in the same session):
    reflectance = (S - D) / (W - Dw)
removes the LED spectrum, the filter responses and the sensor gain, so values are comparable
across devices and days. Expired calibrations are still applied (the optics rarely drift much in a
day or two) but the status says ``expired`` so the report can flag it.

Uncalibrated path: each frame's dark-subtracted signal is divided by its own sum over F1..F8.
Absolute levels are lost, and every channel still carries the probe's own gain (LED spectrum x
filter response x sensor gain in that band), so absolute relative-signal values and their band
ratios differ from probe to probe. The lesion-vs-reference comparison removes that gain by
dividing the lesion by its reference channel by channel
(:func:`mukha_pipeline.spectral.features.reference_delta`); both readings must come from the same
probe and the same path.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime

import numpy as np

from mukha_pipeline.schema.models import (
    AS7341_CHANNELS,
    AS7341_SPECTRAL_CHANNELS,
    CalibrationInfo,
    CalibrationStatus,
)
from mukha_pipeline.types import SpectralCalibration, SpectralFrame, SpectralInput

EPS = 1e-9
REFLECTANCE_RANGE = (0.0, 1.5)  # tissue diffuse reflectance is < 1; allow specular headroom
MIN_WHITE_NET_COUNTS = 1.0  # white minus dark below this means the white reading is unusable


def as_utc(dt: datetime) -> datetime:
    """Naive datetimes are treated as UTC (the pipeline's clock convention)."""
    return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt.astimezone(UTC)


def calibration_info(cal: SpectralCalibration | None, now: datetime) -> CalibrationInfo:
    """Status of a spectral calibration record at time ``now``."""
    if cal is None:
        return CalibrationInfo(
            status=CalibrationStatus.UNCALIBRATED,
            note="No white-reference calibration for this probe; values are relative signal "
            "(each channel divided by the spectral sum).",
        )
    expires = as_utc(cal.expires_at)
    if as_utc(now) > expires:
        return CalibrationInfo(
            status=CalibrationStatus.EXPIRED,
            calibration_id=cal.calibration_id,
            expires_at=expires,
            note=f"Spectral calibration expired at {expires:%Y-%m-%d %H:%M} UTC; reflectance was "
            "computed with it. Recalibrate the probe with the white standard.",
        )
    return CalibrationInfo(
        status=CalibrationStatus.CALIBRATED,
        calibration_id=cal.calibration_id,
        expires_at=expires,
    )


def _channels_present(lit: Sequence[SpectralFrame]) -> list[str]:
    return [c for c in AS7341_CHANNELS if lit and all(c in f.channels for f in lit)]


def _dark_mean(dark: Sequence[SpectralFrame], channels: Sequence[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for c in channels:
        vals = np.array([f.channels.get(c, np.nan) for f in dark], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        out[c] = float(vals.mean()) if vals.size else 0.0
    return out


def _net_matrix(
    lit: Sequence[SpectralFrame], dark: dict[str, float], channels: Sequence[str]
) -> np.ndarray:
    """(n_frames, n_channels) lit - dark counts; NaN where a count is not finite."""
    rows = [[float(f.channels.get(c, np.nan)) - dark[c] for c in channels] for f in lit]
    return np.asarray(rows, dtype=np.float64).reshape(len(lit), len(channels))


def _exposure(gain: float | None, integration_ms: float | None) -> float | None:
    if gain is None or integration_ms is None or gain <= 0 or integration_ms <= 0:
        return None
    return float(gain) * float(integration_ms)


def white_scale(frame: SpectralFrame, cal: SpectralCalibration) -> float:
    """Factor bringing the white-standard counts to this frame's gain x integration time.

    AS7341 counts scale linearly with gain and integration time, so a white reading taken at
    16x / 50 ms predicts 4x the counts at 64x / 50 ms. 1.0 when either setting is unknown.
    """
    frame_exp = _exposure(frame.gain, frame.integration_ms)
    cal_exp = _exposure(cal.gain, cal.integration_ms)
    if frame_exp is None or cal_exp is None:
        return 1.0
    return frame_exp / cal_exp


def _white_net(cal: SpectralCalibration, channels: Sequence[str]) -> dict[str, float]:
    """White minus white-dark counts per channel; NaN where unusable (missing or not above dark)."""
    out: dict[str, float] = {}
    for c in channels:
        v = float(cal.white.get(c, np.nan)) - float(cal.dark.get(c, 0.0))
        out[c] = v if np.isfinite(v) and v >= MIN_WHITE_NET_COUNTS else float("nan")
    return out


def _mean_std(per_frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """NaN-aware mean and sample std (ddof=1) across frames; 0 where undefined."""
    n_valid = np.sum(np.isfinite(per_frame), axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        filled = np.where(np.isfinite(per_frame), per_frame, 0.0)
        total = filled.sum(axis=0)
        mean = np.where(n_valid > 0, total / np.maximum(n_valid, 1), 0.0)
        sq = np.where(np.isfinite(per_frame), (per_frame - mean) ** 2, 0.0).sum(axis=0)
        std = np.where(n_valid > 1, np.sqrt(sq / np.maximum(n_valid - 1, 1)), 0.0)
    return np.nan_to_num(mean), np.nan_to_num(std)


def _relative(net: np.ndarray, channels: Sequence[str]) -> np.ndarray:
    spectral_idx = [i for i, c in enumerate(channels) if c in AS7341_SPECTRAL_CHANNELS]
    clipped = np.clip(net, 0.0, None)  # lit < dark is noise, not negative light
    total = np.nansum(clipped[:, spectral_idx], axis=1, keepdims=True) if spectral_idx else None
    if total is None:
        return np.full_like(net, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        rel = np.where(total > EPS, clipped / np.where(total > EPS, total, 1.0), np.nan)
    return rel


def summarise_reading(
    inp: SpectralInput, cal: SpectralCalibration | None, now: datetime
) -> tuple[dict[str, float], dict[str, float], str, CalibrationInfo]:
    """Mean and std per channel across the N lit frames, the signal type and calibration info."""
    channels = _channels_present(inp.lit)
    info = calibration_info(cal, now)
    if not channels:
        return {}, {}, "relative_signal", info
    dark = _dark_mean(inp.dark, channels)
    net = _net_matrix(inp.lit, dark, channels)

    if cal is not None:
        white_net = _white_net(cal, channels)
        bad = [c for c in AS7341_SPECTRAL_CHANNELS if not np.isfinite(white_net.get(c, np.nan))]
        if not bad:
            # CLEAR / NIR are optional: drop them rather than report a meaningless reflectance.
            keep = [i for i, c in enumerate(channels) if np.isfinite(white_net[c])]
            channels = [channels[i] for i in keep]
            white = np.array([white_net[c] for c in channels], dtype=np.float64)
            scales = np.array([white_scale(f, cal) for f in inp.lit], dtype=np.float64)[:, None]
            refl = np.clip(net[:, keep] / (white[None, :] * scales), *REFLECTANCE_RANGE)
            mean, std = _mean_std(refl)
            if np.any(np.abs(scales - 1.0) > 1e-9):
                factor = float(np.median(scales))
                note = (
                    f"White reference scaled by {factor:.3g} to match this reading's gain x "
                    "integration time."
                )
                info = info.model_copy(update={"note": _join(info.note, note)})
            return _as_dict(channels, mean), _as_dict(channels, std), "reflectance", info
        info = CalibrationInfo(
            status=CalibrationStatus.UNCALIBRATED,
            calibration_id=cal.calibration_id,
            expires_at=as_utc(cal.expires_at),
            note=f"Calibration {cal.calibration_id} has an unusable white reading (white not "
            f"above dark) on {', '.join(bad)}; relative signal used instead.",
        )

    mean, std = _mean_std(_relative(net, channels))
    return _as_dict(channels, mean), _as_dict(channels, std), "relative_signal", info


def _join(a: str | None, b: str) -> str:
    return f"{a} {b}" if a else b


def _as_dict(channels: Sequence[str], arr: np.ndarray) -> dict[str, float]:
    return {c: float(v) if np.isfinite(v) else 0.0 for c, v in zip(channels, arr, strict=True)}
