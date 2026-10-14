"""Synthetic AS7341 readings from the Confirm probe (raw ADC counts), labelled SYNTHETIC.

Forward model, per wavelength on a 1 nm grid:

    counts(channel) = dark + K(channel) * ∫ LED(λ) · S_channel(λ) · R(λ) dλ

* ``R(λ)``: oral-mucosa diffuse reflectance, a scattering baseline rising from ~0.25 at
  415 nm to ~0.55 at 680 nm, multiplied by ``exp(-hb · ε(λ))``. ``ε`` is an oxy-haemoglobin
  extinction *shape*: a strong Soret band at 415 nm and the Q bands at 542 and 577 nm. ``hb``
  is a dimensionless blood-volume index (about 0.25 for normal mucosa, up to ~0.6 for a
  vascular lesion such as erythroplakia). More blood deepens the 540-580 nm dip, which lowers
  F5 (555) and F6 (590) relative to their shoulders F4 (515) and F7 (630).
* ``LED(λ)``: phosphor-converted white LED (blue pump at ~450 nm plus a broad phosphor hump
  at 550-600 nm, with a weak violet tail so F1 still gets light).
* ``S_channel``: AS7341 filter responses (Gaussian approximations of the datasheet FWHM);
  CLEAR is the broadband photodiode, NIR sits at ~910 nm where the white LED emits little.
* ``K(channel)`` is set so that a 99 % white reflectance standard reads the counts in
  ``WHITE_COUNTS`` at gain 64x and 50 ms: thousands of counts, well below the 65535 ADC
  ceiling, as a careful firmware auto-gain would aim for.

All numbers are illustrative. They exercise the spectral pipeline (dark subtraction,
reflectance or relative-signal path, Hb-dip features, QC) and are never evidence of anything.
"""

from __future__ import annotations

import csv
import io
from collections.abc import Sequence
from datetime import datetime, timedelta

import numpy as np

from mukha_pipeline.schema.models import (
    AS7341_CHANNELS,
    CaptureKind,
    CaptureMetadata,
    LedMode,
    SourceAdapter,
)
from mukha_pipeline.sites import OralSite
from mukha_pipeline.types import SpectralCalibration, SpectralFrame, SpectralInput

SYNTHETIC_PROBE_ID = "SYN-CONFIRM-01"
SYNTHETIC_FIRMWARE = "synthetic"
ADC_FULL_SCALE = 65535
REFERENCE_GAIN = 64.0
REFERENCE_INTEGRATION_MS = 50.0
WHITE_REFLECTANCE = 0.99

CSV_COLUMNS: tuple[str, ...] = (
    "reading_id",
    "site",
    "role",
    "kind",
    "repeat",
    *AS7341_CHANNELS,
    "gain",
    "integration_ms",
    "reference_reading_id",
    "lesion_ref",
    "device_id",
)
CALIBRATION_READING_ID = "calibration"

# Target counts for the white standard at the reference gain and integration time.
WHITE_COUNTS: dict[str, float] = {
    "F1_415": 4200.0,
    "F2_445": 26000.0,
    "F3_480": 14000.0,
    "F4_515": 17500.0,
    "F5_555": 30000.0,
    "F6_590": 33000.0,
    "F7_630": 26000.0,
    "F8_680": 12500.0,
    "CLEAR": 50000.0,
    "NIR": 600.0,
}
DARK_COUNTS: dict[str, float] = {
    "F1_415": 210.0,
    "F2_445": 240.0,
    "F3_480": 250.0,
    "F4_515": 260.0,
    "F5_555": 280.0,
    "F6_590": 290.0,
    "F7_630": 300.0,
    "F8_680": 310.0,
    "CLEAR": 380.0,
    "NIR": 260.0,
}

_WL = np.arange(380.0, 1001.0, 1.0)
# (centre nm, FWHM nm) of the AS7341 channels; CLEAR and NIR are handled separately.
_FILTERS: dict[str, tuple[float, float]] = {
    "F1_415": (415.0, 26.0),
    "F2_445": (445.0, 30.0),
    "F3_480": (480.0, 36.0),
    "F4_515": (515.0, 39.0),
    "F5_555": (555.0, 39.0),
    "F6_590": (590.0, 40.0),
    "F7_630": (630.0, 50.0),
    "F8_680": (680.0, 52.0),
    "NIR": (910.0, 40.0),
}


def _gauss(x: np.ndarray, mu: float, fwhm: float) -> np.ndarray:
    sigma = fwhm / 2.3548
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def led_spectrum(wl: np.ndarray = _WL) -> np.ndarray:
    """Relative emission of a phosphor-converted white LED (peak-normalised)."""
    pump = _gauss(wl, 450.0, 22.0)
    phosphor = 0.62 * _gauss(wl, 565.0, 120.0)
    violet_tail = 0.10 * _gauss(wl, 420.0, 40.0)
    far_red = 0.01 * _gauss(wl, 760.0, 200.0)
    out = pump + phosphor + violet_tail + far_red
    return out / out.max()


def channel_response(channel: str, wl: np.ndarray = _WL) -> np.ndarray:
    """Spectral response of one AS7341 channel (peak-normalised)."""
    if channel == "CLEAR":
        # Unfiltered silicon photodiode: broad, rising into the red, cut by the IR filter.
        return np.clip((wl - 350.0) / 450.0, 0.0, 1.0) * (
            1.0 - 1.0 / (1.0 + np.exp(-(wl - 880) / 25))
        )
    centre, fwhm = _FILTERS[channel]
    return _gauss(wl, centre, fwhm)


def hb_extinction_shape(wl: np.ndarray = _WL) -> np.ndarray:
    """Oxy-haemoglobin absorption shape: Soret (415 nm) and Q bands (542, 577 nm)."""
    soret = 1.8 * _gauss(wl, 415.0, 33.0)
    q542 = 0.85 * _gauss(wl, 542.0, 26.0)
    q577 = 0.95 * _gauss(wl, 577.0, 21.0)
    between = 0.10 * _gauss(wl, 560.0, 90.0)  # Q bands overlap; the valley is not empty
    return soret + q542 + q577 + between


def tissue_reflectance(hb: float, wl: np.ndarray = _WL) -> np.ndarray:
    """Diffuse reflectance of oral mucosa with blood-volume index ``hb``."""
    base = np.interp(wl, [380.0, 415.0, 680.0, 1000.0], [0.23, 0.25, 0.55, 0.60])
    return base * np.exp(-hb * hb_extinction_shape(wl))


def _band_values(reflectance: np.ndarray) -> dict[str, float]:
    """LED- and filter-weighted mean reflectance per channel."""
    led = led_spectrum()
    out: dict[str, float] = {}
    for ch in AS7341_CHANNELS:
        weight = led * channel_response(ch)
        if ch == "NIR":
            weight = channel_response(ch)  # the LED barely reaches 910 nm; shape only
        out[ch] = float(np.sum(weight * reflectance) / np.sum(weight))
    return out


def expected_signal(
    hb: float, *, gain: float = REFERENCE_GAIN, integration_ms: float = REFERENCE_INTEGRATION_MS
) -> dict[str, float]:
    """Noise-free dark-subtracted counts for tissue with blood-volume index ``hb``."""
    tissue = _band_values(tissue_reflectance(hb))
    white = _band_values(np.full_like(_WL, WHITE_REFLECTANCE))
    scale = (gain / REFERENCE_GAIN) * (integration_ms / REFERENCE_INTEGRATION_MS)
    return {ch: WHITE_COUNTS[ch] * scale * tissue[ch] / white[ch] for ch in AS7341_CHANNELS}


def _frame(
    rng: np.random.Generator,
    signal: dict[str, float],
    *,
    gain: float,
    integration_ms: float,
    noise: float,
    lit: bool,
) -> SpectralFrame:
    """One read: dark offset + signal with multiplicative, common-mode and read noise."""
    scale = (gain / REFERENCE_GAIN) * (integration_ms / REFERENCE_INTEGRATION_MS)
    common = 1.0 + rng.normal(0.0, noise * 0.5) if lit else 1.0  # probe pressure / angle
    channels: dict[str, float] = {}
    for ch in AS7341_CHANNELS:
        dark = DARK_COUNTS[ch] * (0.5 + 0.5 * scale) + rng.normal(0.0, 4.0)
        value = dark
        if lit:
            value += signal[ch] * common * (1.0 + rng.normal(0.0, noise)) + rng.normal(0.0, 3.0)
        channels[ch] = float(np.clip(np.round(value), 0, ADC_FULL_SCALE))
    return SpectralFrame(channels=channels, gain=gain, integration_ms=integration_ms)


def generate_spectral_reading(
    rng: np.random.Generator,
    *,
    hb: float | None = None,
    hb_dip: float | None = None,
    n: int = 5,
    reading_id: str,
    site: OralSite | str,
    role: str,
    reference_reading_id: str | None = None,
    device_id: str = SYNTHETIC_PROBE_ID,
    gain: float = REFERENCE_GAIN,
    integration_ms: float = REFERENCE_INTEGRATION_MS,
    noise: float = 0.01,
    n_dark: int = 3,
    captured_at: datetime | None = None,
    calibration_id: str | None = None,
    lesion_ref: str | None = None,
) -> SpectralInput:
    """N lit + ``n_dark`` dark AS7341 reads at one spot, as a SYNTHETIC ``SpectralInput``.

    ``hb`` is the blood-volume index (``hb_dip`` is accepted as an alias, the name used in
    docs/pipeline_contracts.md). ``role`` is "lesion" or "reference".
    """
    if hb is None:
        hb = hb_dip
    if hb is None or not 0.0 <= hb <= 2.0:
        raise ValueError(f"hb must be given and within 0..2, got {hb}")
    if role not in ("lesion", "reference"):
        raise ValueError(f"role must be 'lesion' or 'reference', got {role!r}")
    if n < 1 or n_dark < 1:
        raise ValueError("need at least one lit and one dark reading")
    site = OralSite(site)
    signal = expected_signal(hb, gain=gain, integration_ms=integration_ms)
    dark = [
        _frame(rng, signal, gain=gain, integration_ms=integration_ms, noise=noise, lit=False)
        for _ in range(n_dark)
    ]
    lit = [
        _frame(rng, signal, gain=gain, integration_ms=integration_ms, noise=noise, lit=True)
        for _ in range(n)
    ]
    meta = CaptureMetadata(
        capture_id=f"{reading_id}-spectral",
        kind=CaptureKind.SPECTRAL,
        oral_site=site,
        source_adapter=SourceAdapter.SYNTHETIC,
        device_id=device_id,
        firmware_version=SYNTHETIC_FIRMWARE,
        led_mode=LedMode.WHITE,
        exposure_time_ms=integration_ms,
        captured_at=captured_at,
        calibration_id=calibration_id,
        synthetic=True,
        extra={"gain": gain, "generator": "mukha_pipeline.synthetic.spectra"},
    )
    return SpectralInput(
        reading_id=reading_id,
        site=site,
        role=role,  # type: ignore[arg-type]
        lit=lit,
        dark=dark,
        meta=meta,
        reference_reading_id=reference_reading_id,
        lesion_ref=lesion_ref,
    )


def generate_white_reference(
    rng: np.random.Generator,
    *,
    n: int = 5,
    n_dark: int = 3,
    gain: float = REFERENCE_GAIN,
    integration_ms: float = REFERENCE_INTEGRATION_MS,
    noise: float = 0.004,
) -> tuple[list[SpectralFrame], list[SpectralFrame]]:
    """(white frames, white_dark frames) on a ~99 % diffuse white reflectance standard."""
    scale = (gain / REFERENCE_GAIN) * (integration_ms / REFERENCE_INTEGRATION_MS)
    signal = {ch: WHITE_COUNTS[ch] * scale for ch in AS7341_CHANNELS}
    white = [
        _frame(rng, signal, gain=gain, integration_ms=integration_ms, noise=noise, lit=True)
        for _ in range(n)
    ]
    white_dark = [
        _frame(rng, signal, gain=gain, integration_ms=integration_ms, noise=noise, lit=False)
        for _ in range(n_dark)
    ]
    return white, white_dark


def _mean_channels(frames: Sequence[SpectralFrame]) -> dict[str, float]:
    return {ch: float(np.mean([f.channels[ch] for f in frames])) for ch in AS7341_CHANNELS}


def make_spectral_calibration(
    white: Sequence[SpectralFrame],
    white_dark: Sequence[SpectralFrame],
    *,
    created_at: datetime,
    device_id: str = SYNTHETIC_PROBE_ID,
    valid_hours: float = 24.0,
    calibration_id: str | None = None,
) -> SpectralCalibration:
    """SYNTHETIC white-reference calibration: means of the white and white-dark frames."""
    if not white or not white_dark:
        raise ValueError("need at least one white and one white_dark frame")
    return SpectralCalibration(
        calibration_id=calibration_id or f"SYN-SPECCAL-{created_at:%Y%m%dT%H%M}",
        device_id=device_id,
        created_at=created_at,
        expires_at=created_at + timedelta(hours=valid_hours),
        dark=_mean_channels(white_dark),
        white=_mean_channels(white),
        gain=white[0].gain,
        integration_ms=white[0].integration_ms,
        synthetic=True,
    )


# --------------------------------------------------------------------------------------------
# CSV (spectral file format v1, docs/pipeline_contracts.md)
# --------------------------------------------------------------------------------------------


def _fmt(value: float | str | None) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    f = float(value)
    return str(int(f)) if f.is_integer() else repr(f)


def _row(
    *,
    reading_id: str,
    site: str,
    role: str,
    kind: str,
    repeat: int,
    frame: SpectralFrame,
    reference_reading_id: str | None,
    lesion_ref: str | None,
    device_id: str | None,
) -> dict[str, str]:
    row = {
        "reading_id": reading_id,
        "site": site,
        "role": role,
        "kind": kind,
        "repeat": str(repeat),
        "gain": _fmt(frame.gain),
        "integration_ms": _fmt(frame.integration_ms),
        "reference_reading_id": reference_reading_id or "",
        "lesion_ref": lesion_ref or "",
        "device_id": device_id or "",
    }
    row.update({ch: _fmt(frame.channels[ch]) for ch in AS7341_CHANNELS})
    return row


def spectral_inputs_to_csv(
    readings: Sequence[SpectralInput],
    calibration: SpectralCalibration
    | tuple[Sequence[SpectralFrame], Sequence[SpectralFrame]]
    | None = None,
) -> str:
    """Serialise readings (and optionally the white reference) as spectral CSV v1.

    ``calibration`` may be the raw ``(white, white_dark)`` frames from
    :func:`generate_white_reference` (one row per frame, preferred) or a
    :class:`SpectralCalibration` (one mean row each). White rows use ``reading_id =
    calibration`` with empty site and role, as the format requires.
    """
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(CSV_COLUMNS), lineterminator="\n")
    writer.writeheader()
    if isinstance(calibration, SpectralCalibration):
        white_sets = {
            "white": [
                SpectralFrame(calibration.white, calibration.gain, calibration.integration_ms)
            ],
            "white_dark": [
                SpectralFrame(calibration.dark, calibration.gain, calibration.integration_ms)
            ],
        }
        cal_device: str | None = calibration.device_id
    elif calibration is not None:
        white_sets = {"white": list(calibration[0]), "white_dark": list(calibration[1])}
        cal_device = readings[0].meta.device_id if readings else SYNTHETIC_PROBE_ID
    else:
        white_sets, cal_device = {}, None
    for kind, frames in white_sets.items():
        for i, frame in enumerate(frames):
            writer.writerow(
                _row(
                    reading_id=CALIBRATION_READING_ID,
                    site="",
                    role="",
                    kind=kind,
                    repeat=i,
                    frame=frame,
                    reference_reading_id=None,
                    lesion_ref=None,
                    device_id=cal_device,
                )
            )
    for r in readings:
        for kind, frames in (("dark", r.dark), ("lit", r.lit)):
            for i, frame in enumerate(frames):
                writer.writerow(
                    _row(
                        reading_id=r.reading_id,
                        site=r.site.value,
                        role=r.role,
                        kind=kind,
                        repeat=i,
                        frame=frame,
                        reference_reading_id=r.reference_reading_id,
                        lesion_ref=r.lesion_ref,
                        device_id=r.meta.device_id,
                    )
                )
    return buf.getvalue()
