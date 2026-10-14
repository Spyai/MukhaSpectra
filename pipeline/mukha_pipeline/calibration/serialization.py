"""Byte serialisation of ``ImageCalibration`` for the backend's calibration store.

Format: ``np.savez_compressed`` with float arrays plus a ``metadata`` entry holding UTF-8 JSON
as a uint8 array. Loading uses ``allow_pickle=False``: a calibration blob may come from a device
or an upload, and pickle would let it execute code.
"""

from __future__ import annotations

import io
import json
import zipfile
from datetime import datetime
from typing import Any

import numpy as np

from mukha_pipeline.calibration.common import CalibrationError, as_utc
from mukha_pipeline.types import ImageCalibration

FORMAT = "mukhaspectra.image_calibration"
FORMAT_VERSION = 1


def image_calibration_to_bytes(cal: ImageCalibration) -> bytes:
    """Serialise ``cal`` (arrays + JSON metadata) into a compressed ``.npz`` blob."""
    meta: dict[str, Any] = {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "calibration_id": cal.calibration_id,
        "device_id": cal.device_id,
        "created_at": as_utc(cal.created_at).isoformat(),
        "expires_at": as_utc(cal.expires_at).isoformat(),
        "af_scale": float(cal.af_scale),
        "synthetic": bool(cal.synthetic),
        "has_flat_field": cal.flat_field is not None,
    }
    arrays: dict[str, np.ndarray] = {
        "metadata": np.frombuffer(json.dumps(meta).encode("utf-8"), dtype=np.uint8),
        "color_matrix": np.asarray(cal.color_matrix, dtype=np.float64).reshape(3, 3),
    }
    if cal.flat_field is not None:
        arrays["flat_field"] = np.asarray(cal.flat_field, dtype=np.float32)
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    return buf.getvalue()


def image_calibration_from_bytes(data: bytes) -> ImageCalibration:
    """Inverse of ``image_calibration_to_bytes``; raises ``CalibrationError`` on bad blobs."""
    try:
        with np.load(io.BytesIO(data), allow_pickle=False) as npz:
            meta = json.loads(bytes(npz["metadata"].astype(np.uint8)).decode("utf-8"))
            color_matrix = np.asarray(npz["color_matrix"], dtype=np.float64).reshape(3, 3)
            flat = (
                np.asarray(npz["flat_field"], dtype=np.float32) if "flat_field" in npz else None
            )
    except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile, UnicodeDecodeError) as e:
        raise CalibrationError("Calibration file is damaged or not a calibration file.") from e
    if not isinstance(meta, dict) or meta.get("format") != FORMAT:
        raise CalibrationError("Calibration file is not a MukhaSpectra image calibration.")
    if int(meta.get("format_version", 0)) > FORMAT_VERSION:
        raise CalibrationError("Calibration file was written by a newer version; update first.")
    try:
        return ImageCalibration(
            calibration_id=str(meta["calibration_id"]),
            device_id=str(meta["device_id"]),
            created_at=as_utc(datetime.fromisoformat(meta["created_at"])),
            expires_at=as_utc(datetime.fromisoformat(meta["expires_at"])),
            flat_field=flat,
            color_matrix=color_matrix,
            af_scale=float(meta.get("af_scale", 1.0)),
            synthetic=bool(meta.get("synthetic", False)),
        )
    except (KeyError, TypeError, ValueError) as e:
        raise CalibrationError("Calibration file metadata is incomplete.") from e
