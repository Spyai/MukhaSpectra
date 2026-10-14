"""Per-device daily image calibration: flat field, colour matrix and AF intensity scale.

Physics in one paragraph. A phone camera behind the Scout clip sees (a) an uneven illumination
profile from the LED ring plus lens vignetting, (b) a white balance that depends on the LED
spectrum and the phone's ISP, and (c) a violet-LED output that drifts with temperature and age.
Under 405 nm excitation, healthy mucosa emits green autofluorescence (collagen cross-links in
the stroma, NADH/FAD in the epithelium); dysplasia lowers it (fluorescence visualisation loss,
FVL). Comparing AF brightness between days, sites or devices therefore needs (a) a flat field
from a uniform white target, (b) a colour matrix for the WL photo, and (c) a scale from a
fluorescence reference card of known green emission. The colour matrix is for WL captures only:
white-balancing an AF image would distort the green/red ratio that separates tissue green from
~635 nm porphyrin red (plaque, bacteria).
"""

from __future__ import annotations

from datetime import datetime

import cv2
import numpy as np

from mukha_pipeline.calibration.common import (
    CalibrationError,
    as_float_image,
    central_region,
    finite_median,
    luminance,
    new_calibration_id,
    validity_window,
)
from mukha_pipeline.types import ImageCalibration

FLAT_MIN = 0.2  # below 20 % of mean illumination the target is not filling the field;
FLAT_MAX = 5.0  # correcting harder would mostly amplify noise.
GAIN_MIN = 0.2  # white-balance gains outside this range mean a bad capture, not a real cast
GAIN_MAX = 5.0
SATURATED = 0.98  # channel value treated as clipped by the sensor / ISP
EPS = 1e-6
AF_CARD_SIGNAL_MIN = 1e-3  # median green below this is sensor noise, not card emission

# Calibration-time acceptance limits (``create_image_calibration`` raises below these).
WHITE_TARGET_MIN_LEVEL = 0.05
WHITE_TARGET_MAX_SATURATED_FRACTION = 0.5
CARD_MIN_GREEN = 0.02
CARD_MAX_GREEN = 0.97


def compute_flat_field(white_target: np.ndarray, blur_sigma_px: float = 25) -> np.ndarray:
    """Illumination profile from a uniform white-target capture, float32 (H, W), mean 1.

    Grayscale -> heavy Gaussian blur (keeps only the low-frequency LED/vignetting profile, not
    dust or target texture) -> divide by the mean -> clip to [0.2, 5]. Dividing an image by this
    map evens out the illumination. A dark or empty capture yields all ones (no correction),
    because dividing by noise would be worse than not correcting.
    """
    gray = luminance(white_target)
    if gray.size == 0:
        return np.ones((max(gray.shape[0], 1), max(gray.shape[1], 1)), dtype=np.float32)
    fill = finite_median(gray)
    if fill is None:
        return np.ones(gray.shape, dtype=np.float32)
    gray = np.where(np.isfinite(gray), gray, np.float32(fill)).astype(np.float32)
    sigma = float(blur_sigma_px) if np.isfinite(blur_sigma_px) else 0.0
    if sigma > 0:
        gray = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma, borderType=cv2.BORDER_REFLECT)
    mean = float(np.mean(gray))
    if not np.isfinite(mean) or mean <= EPS:
        return np.ones(gray.shape, dtype=np.float32)
    return np.clip(gray / mean, FLAT_MIN, FLAT_MAX).astype(np.float32)


def compute_color_matrix(
    white_target: np.ndarray | None,
    chart_measured: np.ndarray | None = None,
    chart_reference: np.ndarray | None = None,
) -> np.ndarray:
    """3x3 colour matrix ``M`` (float64) such that ``corrected = img @ M.T``.

    With a colour chart (N x 3 measured vs N x 3 reference patch values, N >= 3, full rank) the
    matrix is the least-squares fit ``reference ~= measured @ M.T``. Otherwise a diagonal
    white balance makes the white target neutral, normalised so the green gain is 1 (green
    carries most of the signal and is the AF channel of interest, so it is left untouched).
    Degenerate input falls back to the identity.
    """
    if chart_measured is not None and chart_reference is not None:
        fitted = _chart_matrix(chart_measured, chart_reference)
        if fitted is not None:
            return fitted
    if white_target is None:
        return np.eye(3, dtype=np.float64)
    return _white_balance_matrix(white_target)


def _chart_matrix(measured: np.ndarray, reference: np.ndarray) -> np.ndarray | None:
    """Least-squares 3x3 fit on finite chart patches; None when under-determined."""
    try:
        meas = np.asarray(measured, dtype=np.float64).reshape(-1, 3)
        ref = np.asarray(reference, dtype=np.float64).reshape(-1, 3)
    except ValueError:
        return None
    if meas.shape != ref.shape:
        return None
    keep = np.all(np.isfinite(meas), axis=1) & np.all(np.isfinite(ref), axis=1)
    meas, ref = meas[keep], ref[keep]
    if meas.shape[0] < 3 or np.linalg.matrix_rank(meas) < 3:
        return None
    solution, *_ = np.linalg.lstsq(meas, ref, rcond=None)
    matrix = solution.T
    if not np.all(np.isfinite(matrix)):
        return None
    return matrix.astype(np.float64)


def _white_balance_matrix(white_target: np.ndarray) -> np.ndarray:
    """Diagonal gains from robust per-channel medians of the target centre (unclipped pixels)."""
    img = as_float_image(white_target)
    if img.ndim != 3 or img.shape[2] < 3 or img.size == 0:
        return np.eye(3, dtype=np.float64)
    centre = central_region(img[..., :3]).reshape(-1, 3)
    centre = centre[np.all(np.isfinite(centre), axis=1)]
    unclipped = centre[np.all(centre < SATURATED, axis=1)]
    # Clipped pixels report the sensor limit, not the colour of the light; ignore them when
    # enough unclipped pixels remain.
    sample = unclipped if unclipped.shape[0] >= 16 else centre
    if sample.shape[0] == 0:
        return np.eye(3, dtype=np.float64)
    medians = np.median(sample, axis=0).astype(np.float64)
    green = medians[1]
    gains = np.ones(3, dtype=np.float64)
    if green > EPS:
        for ch in (0, 2):
            if medians[ch] > EPS:
                gains[ch] = np.clip(green / medians[ch], GAIN_MIN, GAIN_MAX)
    return np.diag(gains)


def compute_af_scale(fluorescence_card: np.ndarray, expected_green: float) -> float:
    """Multiplier that brings measured AF green onto the reference card's known level.

    ``expected_green / median(green of the central 50 % of the card)``. The median ignores dust,
    glare and the card edge. Returns 1.0 (no rescaling) when the card signal is at noise level or
    ``expected_green`` is not a positive number, rather than producing an absurd multiplier.
    """
    img = as_float_image(fluorescence_card)
    if img.size == 0:
        return 1.0
    green = img[..., 1] if img.ndim == 3 and img.shape[2] >= 2 else luminance(img)
    measured = finite_median(central_region(green))
    try:
        expected = float(expected_green)
    except (TypeError, ValueError):
        return 1.0
    if measured is None or measured <= AF_CARD_SIGNAL_MIN:
        return 1.0
    if not np.isfinite(expected) or expected <= 0:
        return 1.0
    return float(expected / measured)


def create_image_calibration(
    device_id: str,
    white_target: np.ndarray,
    fluorescence_card: np.ndarray | None,
    *,
    now: datetime | None,
    valid_hours: float = 24,
    expected_card_green: float = 0.5,
    synthetic: bool = False,
    chart_measured: np.ndarray | None = None,
    chart_reference: np.ndarray | None = None,
    flat_blur_sigma_px: float = 25,
) -> ImageCalibration:
    """Build a daily image calibration record from calibration captures.

    Raises ``CalibrationError`` (plain-language message) when a capture is unusable, so the
    operator repeats it instead of the device silently claiming standardised intensities. A
    missing fluorescence card gives ``af_scale = 1.0`` (flat field and colour still apply).
    """
    _check_white_target(white_target)
    af_scale = 1.0
    if fluorescence_card is not None:
        _check_fluorescence_card(fluorescence_card)
        af_scale = compute_af_scale(fluorescence_card, expected_card_green)
    created_at, expires_at = validity_window(now, valid_hours)
    return ImageCalibration(
        calibration_id=new_calibration_id(),
        device_id=device_id,
        created_at=created_at,
        expires_at=expires_at,
        flat_field=compute_flat_field(white_target, flat_blur_sigma_px),
        color_matrix=compute_color_matrix(white_target, chart_measured, chart_reference),
        af_scale=af_scale,
        synthetic=synthetic,
    )


def _check_white_target(white_target: np.ndarray | None) -> None:
    """Reject empty, dark or mostly clipped white-target captures."""
    if white_target is None or np.asarray(white_target).size == 0:
        raise CalibrationError("White target capture is missing or empty; capture it again.")
    gray = luminance(white_target)
    level = finite_median(central_region(gray))
    if level is None or level < WHITE_TARGET_MIN_LEVEL:
        raise CalibrationError(
            "White target capture is too dark; check that the white LED is on and the target "
            "fills the view, then capture it again."
        )
    centre = central_region(gray)
    saturated = float(np.mean(centre >= SATURATED))
    if saturated > WHITE_TARGET_MAX_SATURATED_FRACTION:
        raise CalibrationError(
            "White target capture is overexposed (most pixels clipped); lower the exposure or "
            "move the target slightly away, then capture it again."
        )


def _check_fluorescence_card(card: np.ndarray) -> None:
    """Reject empty, dark (violet LED off / blocked) or saturated card captures."""
    img = as_float_image(card)
    if img.size == 0:
        raise CalibrationError("Fluorescence card capture is empty; capture it again.")
    green = img[..., 1] if img.ndim == 3 and img.shape[2] >= 2 else luminance(img)
    level = finite_median(central_region(green))
    if level is None or level < CARD_MIN_GREEN:
        raise CalibrationError(
            "Fluorescence card capture is too dark; check that the violet LED is on and the card "
            "is in view, then capture it again."
        )
    if level > CARD_MAX_GREEN:
        raise CalibrationError(
            "Fluorescence card capture is overexposed; lower the exposure, then capture it again."
        )
