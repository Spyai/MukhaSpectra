"""Internal data types shared by pipeline modules.

Conventions (all modules follow these):
* Images are ``np.ndarray`` float32, RGB channel order, values in [0, 1], shape (H, W, 3).
  Loaders convert to this; encoders convert back.
* Masks are ``np.ndarray`` bool, shape (H, W).
* Analysis runs at a *working resolution* (longest side ``analysis.working_max_side_px``) in the
  white-light (WL) frame. The AF image is warped into the WL frame before any per-pixel comparison.
* Pure functions: no file or network I/O inside analysis code. The orchestrator returns artifacts
  as bytes and the caller decides where to store them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

import numpy as np

from mukha_pipeline.schema.models import (
    AFAnalysis,
    CaptureMetadata,
    ConsentInfo,
    DataProvenance,
    Diascopy,
    PatientInfo,
    RiskFactors,
    ScreeningContext,
    ScreeningResult,
    SiteRole,
    VisualFlag,
)
from mukha_pipeline.sites import OralSite

# --------------------------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------------------------


@dataclass
class ImageCapture:
    # float32 RGB [0,1], as decoded. Uploads are shrunk while decoding (longest side at most
    # UPLOAD_DECODE_MAX_SIDE); meta.width_px/height_px keep the original capture size and
    # meta.extra['decode_scale'] = decoded width / original width.
    image: np.ndarray
    meta: CaptureMetadata


@dataclass
class SiteInput:
    site: OralSite
    wl: ImageCapture
    af: ImageCapture
    dark: ImageCapture | None = None  # captured with all LEDs off, same exposure as AF
    role: SiteRole = SiteRole.EXAMINED
    visual_flag: VisualFlag = field(default_factory=VisualFlag)
    diascopy: Diascopy = Diascopy.NOT_DONE
    px_per_mm: float | None = None  # for the decoded WL image's pixels, if a scale is known


@dataclass
class SpectralFrame:
    """One AS7341 read: raw counts per channel name (see AS7341_CHANNELS)."""

    channels: dict[str, float]
    gain: float | None = None
    integration_ms: float | None = None


@dataclass
class SpectralInput:
    reading_id: str
    site: OralSite
    role: Literal["lesion", "reference"]
    lit: list[SpectralFrame]  # N illuminated readings
    dark: list[SpectralFrame]  # dark readings (LED off), at least one
    meta: CaptureMetadata
    reference_reading_id: str | None = None  # for a lesion reading: its contralateral normal
    lesion_ref: str | None = None  # optional FVL region id / free label this reading targets


@dataclass
class ImageCalibration:
    """Per-device daily image calibration from a white target and a fluorescence reference card.

    ``flat_field`` is the WHITE-LED illumination profile (white target) and is applied to WL
    captures only. ``af_flat_field`` is the VIOLET-LED excitation profile (fluorescence card,
    green emission, linear light) and is applied to linearised AF captures only: the 405 nm LEDs
    sit elsewhere on the clip and light the field differently, so the white profile would leave
    (or invent) an AF fall-off that reads as fluorescence loss. ``None`` means not measured; AF
    is then not flat-fielded at all.
    """

    calibration_id: str
    device_id: str
    created_at: datetime
    expires_at: datetime
    flat_field: np.ndarray | None  # float32 (H, W), mean 1.0; resized to the image on use
    color_matrix: np.ndarray  # (3, 3); corrected = img @ color_matrix.T
    af_scale: float = 1.0  # multiply LINEAR AF intensities by this to reach the card scale
    synthetic: bool = False
    af_flat_field: np.ndarray | None = None  # float32 (H, W), mean 1.0, linear; AF only


@dataclass
class SpectralCalibration:
    """Dark reading + white reflectance standard, so reflectance = (S - D) / (W - D)."""

    calibration_id: str
    device_id: str
    created_at: datetime
    expires_at: datetime
    dark: dict[str, float]
    white: dict[str, float]
    gain: float | None = None
    integration_ms: float | None = None
    synthetic: bool = False


@dataclass
class ScreeningInput:
    screening_id: str
    patient: PatientInfo
    consent: ConsentInfo
    risk_factors: RiskFactors
    context: ScreeningContext
    sites: list[SiteInput]
    spectral: list[SpectralInput] = field(default_factory=list)
    image_calibrations: dict[str, ImageCalibration] = field(default_factory=dict)  # by device_id
    spectral_calibrations: dict[str, SpectralCalibration] = field(default_factory=dict)
    data_provenance: DataProvenance = DataProvenance.REAL
    now: datetime | None = None  # injected clock (calibration expiry, created_at); UTC
    result_version: int = 1
    supersedes_result_id: str | None = None
    result_id: str | None = None  # generated when None


# --------------------------------------------------------------------------------------------
# Intermediates
# --------------------------------------------------------------------------------------------


@dataclass
class Masks:
    mucosa: np.ndarray  # bool: analysable oral mucosa (excludes teeth, background, specular)
    teeth: np.ndarray
    background: np.ndarray
    specular: np.ndarray
    confidence: float  # 0..1


@dataclass
class RegistrationResult:
    warp: np.ndarray  # (3, 3) float64, maps AF pixel coords -> WL pixel coords
    method: Literal["ecc_affine", "ecc_homography", "orb_homography", "identity"]
    confidence: float  # 0..1
    passed: bool
    registered: np.ndarray  # AF image warped into the WL frame, float32 (H, W, 3)
    valid: np.ndarray  # bool (H, W): pixels covered by the warped AF image
    note: str | None = None


@dataclass
class AFMaps:
    """Per-pixel fluorescence maps in the WL frame (exposure- and calibration-normalised)."""

    green: np.ndarray  # float32 (H, W)
    red: np.ndarray
    blue: np.ndarray
    gr_ratio: np.ndarray  # green / (red + eps)
    af_index: np.ndarray  # green / (red + green + blue + eps)


@dataclass
class FVLOutput:
    analysis: AFAnalysis
    relative_af: np.ndarray  # float32 (H, W); NaN outside analysable mucosa
    labels: np.ndarray  # int32 (H, W); 0 = background, k = region k (region_id "r{k}")


@dataclass
class AnalysisOutput:
    result: ScreeningResult
    # Artifact key -> bytes. Keys are referenced from the result, e.g.
    # "sites/buccal_mucosa_left/heatmap_overlay.png".
    artifacts: dict[str, bytes] = field(default_factory=dict)
