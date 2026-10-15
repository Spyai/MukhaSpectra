"""Small shared helpers for the calibration package: clock handling, ids and errors."""

from __future__ import annotations

import math
import uuid
from datetime import UTC, datetime, timedelta

import numpy as np

CALIBRATION_ID_PREFIX = "cal-"


class CalibrationError(ValueError):
    """A calibration capture cannot be turned into a trustworthy calibration record.

    Raised only by the ``create_*`` / deserialisation entry points, which run at calibration
    time with an operator present who can repeat the capture. The message is plain language and
    tells the operator what to do. Analysis-time helpers never raise; they fall back to neutral
    values (scale 1, identity matrix) and the record's status carries the uncertainty.
    """


def as_utc(dt: datetime | None) -> datetime:
    """Return ``dt`` as an aware UTC datetime.

    Naive datetimes are treated as UTC: device clocks, database rows and the injected pipeline
    clock are all UTC by convention, and comparing naive with aware datetimes raises TypeError.
    ``None`` means "now" so a missing injected clock never crashes a status check.
    """
    if dt is None:
        return datetime.now(UTC)
    if dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def new_calibration_id() -> str:
    """Globally unique calibration id (``cal-`` + uuid4 hex); stored on every capture."""
    return f"{CALIBRATION_ID_PREFIX}{uuid.uuid4().hex}"


def validity_window(now: datetime | None, valid_hours: float) -> tuple[datetime, datetime]:
    """(created_at, expires_at) in UTC. Calibrations drift with LED ageing and temperature, so
    every record must expire; a non-positive validity is a programming error."""
    try:
        hours = float(valid_hours)
    except (TypeError, ValueError):
        hours = math.nan
    if not math.isfinite(hours):
        raise CalibrationError("Calibration validity must be a finite number of hours.")
    if hours <= 0:
        raise CalibrationError("Calibration validity must be longer than zero hours.")
    created = as_utc(now)
    return created, created + timedelta(hours=hours)


def as_float_image(img: np.ndarray) -> np.ndarray:
    """Float32 copy in [0, 1] semantics; integer inputs are scaled by their dtype maximum.

    Non-finite values are kept (callers decide how to fill them) so robust statistics can
    ignore them instead of being biased by an arbitrary fill value.
    """
    arr = np.asarray(img)
    if np.issubdtype(arr.dtype, np.integer):
        return arr.astype(np.float32) / float(np.iinfo(arr.dtype).max)
    if arr.dtype == np.bool_:
        return arr.astype(np.float32)
    return arr.astype(np.float32, copy=True)


def luminance(img: np.ndarray) -> np.ndarray:
    """Rec. 601 luma for RGB, the image itself for single-channel input. Float32 (H, W)."""
    arr = as_float_image(img)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3 and arr.shape[2] >= 3:
        r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
        return (0.299 * r + 0.587 * g + 0.114 * b).astype(np.float32)
    if arr.ndim == 3 and arr.shape[2] >= 1:
        return arr[..., 0]
    raise CalibrationError("Calibration image must be a 2-D grayscale or (H, W, 3) RGB array.")


def central_region(arr: np.ndarray, fraction: float = 0.5) -> np.ndarray:
    """Central crop covering ``fraction`` of each dimension (at least one pixel).

    Calibration targets are aimed at the centre of the field; the rim is where vignetting,
    target edges and the device housing contaminate the statistics.
    """
    h, w = arr.shape[:2]
    ch = max(1, int(round(h * fraction)))
    cw = max(1, int(round(w * fraction)))
    y0 = max(0, (h - ch) // 2)
    x0 = max(0, (w - cw) // 2)
    return arr[y0 : y0 + ch, x0 : x0 + cw]


def finite_median(values: np.ndarray) -> float | None:
    """Median of the finite entries, or None when there are none."""
    flat = np.asarray(values, dtype=np.float64).ravel()
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        return None
    return float(np.median(flat))
