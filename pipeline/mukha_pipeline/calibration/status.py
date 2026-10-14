"""Calibration status for a capture: calibrated, uncalibrated or expired.

The status travels into the result and the report, which marks uncalibrated or expired captures.
It is a statement about intensity standardisation only: an uncalibrated capture is still analysed
(relative AF against a reference site cancels most device gain), but absolute intensities and
cross-device comparisons are flagged as not standardised.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from mukha_pipeline.calibration.common import as_utc
from mukha_pipeline.schema.models import CalibrationInfo, CalibrationStatus
from mukha_pipeline.types import ImageCalibration, SpectralCalibration

UNCALIBRATED_NOTE = "No calibration record for this device; intensities are not standardised."

# A record created noticeably after the capture usually means a wrong device clock. It is still
# used, but the note asks for the clock to be checked because expiry relies on it.
CLOCK_SKEW_TOLERANCE = timedelta(minutes=10)


def image_calibration_status(cal: ImageCalibration | None, now: datetime | None) -> CalibrationInfo:
    """Status of the daily image calibration (white target + fluorescence card) at ``now``."""
    if cal is None:
        return CalibrationInfo(status=CalibrationStatus.UNCALIBRATED, note=UNCALIBRATED_NOTE)
    return _status(
        cal.calibration_id,
        cal.created_at,
        cal.expires_at,
        now,
        redo="recalibrate the camera with the white target and the fluorescence reference card",
    )


def spectral_calibration_status(
    cal: SpectralCalibration | None, now: datetime | None
) -> CalibrationInfo:
    """Status of the probe's white-reference calibration at ``now``.

    Without it, spectra are reported as relative signals instead of reflectance.
    """
    if cal is None:
        return CalibrationInfo(status=CalibrationStatus.UNCALIBRATED, note=UNCALIBRATED_NOTE)
    return _status(
        cal.calibration_id,
        cal.created_at,
        cal.expires_at,
        now,
        redo="recalibrate the probe on the white reflectance standard",
    )


def _status(
    calibration_id: str,
    created_at: datetime,
    expires_at: datetime,
    now: datetime | None,
    *,
    redo: str,
) -> CalibrationInfo:
    """Shared expiry logic. Naive datetimes are UTC; expiry is strict (``now > expires_at``)."""
    now_utc = as_utc(now)
    expires_utc = as_utc(expires_at)
    created_utc = as_utc(created_at)
    if now_utc > expires_utc:
        return CalibrationInfo(
            status=CalibrationStatus.EXPIRED,
            calibration_id=calibration_id,
            expires_at=expires_utc,
            note=(
                f"Calibration expired at {expires_utc:%Y-%m-%d %H:%M} UTC; intensities may have "
                f"drifted since. Please {redo} before the next session."
            ),
        )
    note = None
    if created_utc - now_utc > CLOCK_SKEW_TOLERANCE:
        note = (
            "Calibration record is dated after this capture; check the device clock, because "
            "calibration expiry depends on it."
        )
    return CalibrationInfo(
        status=CalibrationStatus.CALIBRATED,
        calibration_id=calibration_id,
        expires_at=expires_utc,
        note=note,
    )
