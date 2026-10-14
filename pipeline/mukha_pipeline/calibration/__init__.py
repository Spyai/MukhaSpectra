"""Per-device calibration: image flat field / colour / AF scale, spectral reflectance, expiry.

Calibration standardises intensities; it does not change what the pipeline can detect. A capture
without a valid calibration is still analysed, and the result marks it ``uncalibrated`` or
``expired`` so the clinician knows absolute intensities were not standardised.
"""

from mukha_pipeline.calibration.common import CalibrationError, as_utc
from mukha_pipeline.calibration.image import (
    compute_af_scale,
    compute_color_matrix,
    compute_flat_field,
    create_image_calibration,
)
from mukha_pipeline.calibration.serialization import (
    image_calibration_from_bytes,
    image_calibration_to_bytes,
)
from mukha_pipeline.calibration.spectral import create_spectral_calibration, reflectance
from mukha_pipeline.calibration.status import (
    UNCALIBRATED_NOTE,
    image_calibration_status,
    spectral_calibration_status,
)

__all__ = [
    "UNCALIBRATED_NOTE",
    "CalibrationError",
    "as_utc",
    "compute_af_scale",
    "compute_color_matrix",
    "compute_flat_field",
    "create_image_calibration",
    "create_spectral_calibration",
    "image_calibration_from_bytes",
    "image_calibration_status",
    "image_calibration_to_bytes",
    "reflectance",
    "spectral_calibration_status",
]
