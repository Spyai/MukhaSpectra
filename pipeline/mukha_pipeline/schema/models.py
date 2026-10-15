"""ScreeningResult v1: the single structured output of a MukhaSpectra screening.

This module is the source of truth for the contract between the pipeline, the API, the report
engine and the PWA. The JSON Schema in ``screening_result.v1.schema.json`` is generated from it
(``python -m mukha_pipeline.schema.export``) and a test fails if the two drift apart.

Positioning rules encoded here:
* Outputs are risk levels and suspicious findings, never a diagnosis.
* ``requires_clinician_review`` is always true and every result carries the prototype banner and
  the "screening aid, not a diagnosis" disclaimer.
* Model components carry ``validated = False`` until a documented clinical validation exists.
  Code never sets it to True.

Versioning: additive, backwards-compatible changes bump the minor version; anything that removes
or renames a field bumps the major version and gets a new schema file.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from mukha_pipeline import SCHEMA_VERSION
from mukha_pipeline.sites import OralSite

PROTOTYPE_BANNER = "Research prototype — not for clinical use"
DISCLAIMER = (
    "Screening aid — not a diagnosis. MukhaSpectra flags suspicious findings and assigns a risk "
    "level to support triage. Final diagnosis requires examination by a clinician and "
    "histopathology."
)
SYNTHETIC_NOTICE = (
    "SYNTHETIC DATA — generated for demonstration and software testing. Not from a real patient. "
    "Must never be used to support any performance claim."
)

AS7341_CHANNELS: tuple[str, ...] = (
    "F1_415",
    "F2_445",
    "F3_480",
    "F4_515",
    "F5_555",
    "F6_590",
    "F7_630",
    "F8_680",
    "CLEAR",
    "NIR",
)
AS7341_SPECTRAL_CHANNELS: tuple[str, ...] = AS7341_CHANNELS[:8]
AS7341_WAVELENGTH_NM: dict[str, int] = {
    "F1_415": 415,
    "F2_445": 445,
    "F3_480": 480,
    "F4_515": 515,
    "F5_555": 555,
    "F6_590": 590,
    "F7_630": 630,
    "F8_680": 680,
    "NIR": 910,
}

Probability = Annotated[float, Field(ge=0.0, le=1.0)]
UnitFloat = Annotated[float, Field(ge=0.0, le=1.0)]


class _Model(BaseModel):
    # allow_inf_nan=False: NaN/Infinity are not valid JSON and must never reach a stored result.
    model_config = ConfigDict(
        extra="forbid", use_enum_values=False, populate_by_name=True, allow_inf_nan=False
    )


# --------------------------------------------------------------------------------------------
# Enumerations
# --------------------------------------------------------------------------------------------


class RiskLevel(StrEnum):
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    INCONCLUSIVE = "INCONCLUSIVE"


RISK_ORDER: dict[RiskLevel, int] = {
    RiskLevel.LOW: 0,
    RiskLevel.MODERATE: 1,
    RiskLevel.HIGH: 2,
}


class DataProvenance(StrEnum):
    REAL = "real"
    SYNTHETIC = "synthetic"
    PHANTOM = "phantom"
    EX_VIVO = "ex_vivo"
    VOLUNTEER = "volunteer"
    MIXED = "mixed"


class CaptureKind(StrEnum):
    WHITE_LIGHT = "white_light"
    AUTOFLUORESCENCE = "autofluorescence"
    DARK_FRAME = "dark_frame"
    SPECTRAL = "spectral"


class LedMode(StrEnum):
    OFF = "OFF"
    WHITE = "WHITE"
    VIOLET = "VIOLET"


class SourceAdapter(StrEnum):
    UPLOAD = "upload"
    PHONE_CAMERA = "phone_camera"
    SCOUT_BLE = "scout_ble"
    DEDICATED_CAMERA = "dedicated_camera"
    CONFIRM_PROBE = "confirm_probe"
    MOCK_DEVICE = "mock_device"
    SYNTHETIC = "synthetic"


class QCSeverity(StrEnum):
    ERROR = "error"  # blocks analysis of this capture
    WARNING = "warning"  # analysis runs, limitation is reported
    INFO = "info"


class CalibrationStatus(StrEnum):
    CALIBRATED = "calibrated"
    UNCALIBRATED = "uncalibrated"  # no calibration record
    EXPIRED = "expired"  # record exists but is past its expiry date
    PARTIAL = "partial"  # some captures calibrated, some not (patient-level only)


class SiteRole(StrEnum):
    EXAMINED = "examined"
    REFERENCE_ONLY = "reference_only"


class VisualAppearance(StrEnum):
    WHITE_PATCH = "white_patch"
    RED_PATCH = "red_patch"
    MIXED_RED_WHITE = "mixed_red_white"
    ULCER = "ulcer"
    GROWTH = "growth"
    OTHER = "other"


class Diascopy(StrEnum):
    """Pressing a glass slide on a dark (FVL) area. If fluorescence returns, the darkness was
    most likely blood/inflammation rather than a tissue change."""

    NOT_DONE = "not_done"
    FVL_PERSISTS = "fvl_persists"
    FVL_RESOLVES = "fvl_resolves"


class ReferenceType(StrEnum):
    CONTRALATERAL = "contralateral"
    ANALOGOUS_SITE = "analogous_site"
    IN_IMAGE_AUTO = "in_image_auto"
    NONE = "none"


class ConfounderType(StrEnum):
    PORPHYRIN_RED = "porphyrin_red"  # bacterial porphyrins / plaque, ~635 nm red fluorescence
    PIGMENTATION = "pigmentation"  # melanin absorbs excitation and emission
    BLOOD = "blood"  # haemoglobin absorbs; haemorrhage or inflammation
    GLARE = "glare"  # specular highlight, no tissue signal


class ModelStatus(StrEnum):
    AVAILABLE = "available"
    MODEL_UNAVAILABLE = "model_unavailable"
    PLACEHOLDER = "placeholder"
    BASELINE_RULES = "baseline_rules"


class ModelComponent(StrEnum):
    SEGMENTER = "segmenter"
    WL_LESION_CLASSIFIER = "wl_lesion_classifier"
    SPECTRAL_CLASSIFIER = "spectral_classifier"
    FUSION = "fusion"


class FactorSource(StrEnum):
    AUTOFLUORESCENCE = "autofluorescence"
    WL_MODEL = "wl_model"
    SPECTRAL = "spectral"
    CLINICAL_VISUAL = "clinical_visual"
    CLINICAL_HISTORY = "clinical_history"
    SYMPTOM = "symptom"
    SITE = "site"
    QC = "qc"
    OVERRIDE = "override"


class LesionOrigin(StrEnum):
    AF_FVL = "af_fvl"
    VISUAL = "visual"
    WL_MODEL = "wl_model"
    SPECTRAL = "spectral"


class ReferralUrgency(StrEnum):
    NONE = "none"
    REEXAMINE = "reexamine"
    URGENT = "urgent"
    RETAKE = "retake"


class ActionCode(StrEnum):
    ROUTINE_RESCREEN = "routine_rescreen_and_cessation"
    REEXAMINE_2_WEEKS = "remove_irritants_reexamine_2_weeks"
    URGENT_REFERRAL = "urgent_referral_biopsy"
    RETAKE_OR_EXAMINE = "retake_or_clinical_exam"


class Sex(StrEnum):
    FEMALE = "female"
    MALE = "male"
    OTHER = "other"
    UNDISCLOSED = "undisclosed"


class Language(StrEnum):
    EN = "en"
    HI = "hi"
    KN = "kn"


class WLLabel(StrEnum):
    NORMAL = "normal"
    WHITE_PATCH = "white_patch"
    RED_PATCH = "red_patch"
    ULCER_MASS = "ulcer_mass"
    OTHER = "other"


# --------------------------------------------------------------------------------------------
# Shared small models
# --------------------------------------------------------------------------------------------


class NormPoint(_Model):
    """Point in normalised image coordinates of the white-light frame (0..1, origin top-left)."""

    x: UnitFloat
    y: UnitFloat


class NormBox(_Model):
    x: UnitFloat
    y: UnitFloat
    w: UnitFloat
    h: UnitFloat


class ConfigRef(_Model):
    name: str
    version: str
    sha256: str = Field(description="SHA-256 of the exact config file used.")


class ModelInfo(_Model):
    component: ModelComponent
    name: str
    version: str
    status: ModelStatus
    validated: Literal[False] = Field(
        default=False,
        description=(
            "Always false in schema v1: no MukhaSpectra model has a documented clinical "
            "validation. A validated model requires a new schema version."
        ),
    )
    model_card: str | None = Field(default=None, description="Path or URI of the model card.")
    sha256: str | None = None
    notes: str | None = None


# --------------------------------------------------------------------------------------------
# QC
# --------------------------------------------------------------------------------------------


class QCIssue(_Model):
    code: str = Field(description="Stable machine code, e.g. 'blur', 'glare', 'af_signal_low'.")
    severity: QCSeverity
    message: str = Field(description="Plain-language retake prompt in English.")
    message_key: str = Field(description="i18n key for the retake prompt, e.g. 'qc.blur'.")
    metric: str | None = None
    value: float | None = None
    threshold: float | None = None


class QCResult(_Model):
    passed: bool = Field(description="False if any issue has severity 'error'.")
    score: UnitFloat = Field(description="0..1 overall capture quality (1 = no concerns).")
    issues: list[QCIssue] = Field(default_factory=list)
    metrics: dict[str, float] = Field(default_factory=dict)


# --------------------------------------------------------------------------------------------
# Capture metadata
# --------------------------------------------------------------------------------------------


class CaptureMetadata(_Model):
    capture_id: str
    kind: CaptureKind
    oral_site: OralSite
    source_adapter: SourceAdapter
    device_id: str | None = None
    firmware_version: str | None = None
    led_mode: LedMode | None = None
    exposure_time_ms: float | None = None
    iso: float | None = None
    white_balance: str | None = Field(default=None, description="e.g. 'locked:5000K', 'auto'.")
    focus: str | None = None
    captured_at: datetime | None = None
    calibration_id: str | None = None
    width_px: int | None = None
    height_px: int | None = None
    file_name: str | None = None
    file_sha256: str | None = None
    mime_type: str | None = None
    synthetic: bool = False
    extra: dict[str, str | float | int | bool | None] = Field(default_factory=dict)


class CaptureRecord(_Model):
    metadata: CaptureMetadata
    qc: QCResult
    artifact: str | None = Field(default=None, description="Artifact key of the stored image.")


class CalibrationInfo(_Model):
    status: CalibrationStatus
    calibration_id: str | None = None
    expires_at: datetime | None = None
    note: str | None = None


# --------------------------------------------------------------------------------------------
# Patient, consent, risk factors, context
# --------------------------------------------------------------------------------------------


class PatientInfo(_Model):
    """Minimised clinical view of the patient. Name and contact details live in a separate PII
    store keyed by ``pseudonymous_id`` and are joined only when a report is rendered."""

    pseudonymous_id: str
    age_years: int | None = Field(default=None, ge=0, le=120)
    sex: Sex = Sex.UNDISCLOSED


class ConsentInfo(_Model):
    consent_recorded: bool
    method: Literal["checkbox", "signature", "thumbprint", "none"] = "none"
    language: Language = Language.EN
    research_use_opt_in: bool = Field(
        default=False, description="Separate opt-in to use de-identified data to improve the tool."
    )
    consent_id: str | None = None
    recorded_at: datetime | None = None


SmokelessType = Literal[
    "gutka", "khaini", "zarda", "paan_with_tobacco", "mawa", "mishri", "snuff", "other"
]
SmokingType = Literal["bidi", "cigarette", "hookah", "chutta_reverse", "other"]


class SmokelessTobaccoUse(_Model):
    uses: bool = False
    former: bool = False
    types: list[SmokelessType] = Field(default_factory=list)
    years: float | None = Field(default=None, ge=0)
    per_day: float | None = Field(default=None, ge=0)


class SmokingUse(_Model):
    uses: bool = False
    former: bool = False
    types: list[SmokingType] = Field(default_factory=list)
    years: float | None = Field(default=None, ge=0)
    per_day: float | None = Field(default=None, ge=0)


class ArecaNutUse(_Model):
    """Areca nut without tobacco (supari, paan without tobacco). Linked to submucous fibrosis."""

    uses: bool = False
    former: bool = False
    years: float | None = Field(default=None, ge=0)
    per_day: float | None = Field(default=None, ge=0)


class AlcoholUse(_Model):
    uses: bool = False
    frequency: Literal["never", "occasional", "weekly", "daily"] = "never"


class Symptoms(_Model):
    non_healing_ulcer_over_2_weeks: bool = False
    white_patch: bool = False
    red_patch: bool = False
    lump: bool = False
    pain: bool = False
    difficulty_opening_mouth: bool = False
    bleeding: bool = False
    burning_sensation: bool = False
    neck_lump: bool = False


class Irritants(_Model):
    sharp_tooth: bool = False
    ill_fitting_denture: bool = False


class RiskFactors(_Model):
    smokeless_tobacco: SmokelessTobaccoUse = Field(default_factory=SmokelessTobaccoUse)
    smoking: SmokingUse = Field(default_factory=SmokingUse)
    areca_nut: ArecaNutUse = Field(default_factory=ArecaNutUse)
    alcohol: AlcoholUse = Field(default_factory=AlcoholUse)
    prior_oral_lesion: bool = False
    prior_oral_cancer: bool = False
    symptoms: Symptoms = Field(default_factory=Symptoms)
    irritants: Irritants = Field(default_factory=Irritants)


class ScreeningContext(_Model):
    screening_site_name: str | None = Field(default=None, description="Camp or clinic name.")
    camp_id: str | None = None
    health_worker_id: str | None = None
    health_worker_name: str | None = None
    device_ids: list[str] = Field(default_factory=list)
    firmware_versions: list[str] = Field(default_factory=list)
    captured_at: datetime | None = None
    language: Language = Language.EN


class VisualFlag(_Model):
    """What the health worker saw with their own eyes at this site."""

    lesion_seen: bool = False
    appearance: list[VisualAppearance] = Field(default_factory=list)
    description: str | None = Field(default=None, max_length=1000)


# --------------------------------------------------------------------------------------------
# Per-site analysis
# --------------------------------------------------------------------------------------------


class RegistrationSummary(_Model):
    method: Literal["ecc_affine", "ecc_homography", "orb_homography", "identity"]
    confidence: UnitFloat
    passed: bool
    warp_matrix: list[list[float]] = Field(
        description="3x3 matrix mapping AF pixel coordinates into the WL frame (working res)."
    )
    note: str | None = None


class SegmentationSummary(_Model):
    method: str
    version: str
    status: Literal["ok", "low_confidence", "failed"]
    mucosa_fraction: UnitFloat
    teeth_fraction: UnitFloat
    specular_fraction: UnitFloat
    confidence: UnitFloat


class AFReference(_Model):
    type: ReferenceType
    reference_site: OralSite | None = None
    reference_green: float | None = Field(
        default=None, description="Median exposure-normalised green fluorescence of the reference."
    )
    note: str | None = None


class AFSiteMedians(_Model):
    green: float
    gr_ratio: float
    af_index: float
    relative_af: float


class ConfounderFlag(_Model):
    type: ConfounderType
    overlap_fraction: UnitFloat
    note: str


class FVLRegion(_Model):
    region_id: str
    area_px: int = Field(ge=0)
    area_mm2: float | None = Field(default=None, ge=0)
    area_fraction_of_mucosa: UnitFloat
    centroid: NormPoint
    bbox: NormBox
    mean_relative_af: float = Field(ge=0)
    min_relative_af: float = Field(ge=0, description="5th percentile (robust minimum).")
    contrast_vs_surround: float = Field(
        description="Mean relative AF of a ring around the region / region mean. >1 = darker."
    )
    border_irregularity: UnitFloat = Field(
        description="Border raggedness: 1 - convex hull perimeter / perimeter, on an outline "
        "simplified with a 1 px tolerance (0 = convex outline at any elongation)."
    )
    solidity: UnitFloat
    confounders: list[ConfounderFlag] = Field(default_factory=list)


class AFAnalysis(_Model):
    status: Literal["ok", "skipped_qc_failed", "insufficient_mucosa", "no_reference"]
    reference: AFReference
    fvl_threshold: float = Field(description="Relative-AF threshold below which tissue is FVL.")
    min_region_area_px: int
    site_medians: AFSiteMedians | None = None
    fvl_area_fraction: UnitFloat = 0.0
    porphyrin_red_fraction: UnitFloat = Field(
        default=0.0, description="Fraction of mucosa with red (porphyrin) fluorescence."
    )
    regions: list[FVLRegion] = Field(default_factory=list)


class WLPrediction(_Model):
    label: WLLabel
    probability: Probability


class WLAnalysis(_Model):
    status: Literal["ok", "model_unavailable", "skipped_qc_failed", "error"]
    model: ModelInfo
    predictions: list[WLPrediction] = Field(default_factory=list)
    top_label: WLLabel | None = None
    saliency_artifact: str | None = Field(
        default=None, description="Clinician-only explainability map (artifact key)."
    )
    note: str | None = None


class SiteArtifacts(_Model):
    """Artifact keys (relative paths in the result's artifact store)."""

    wl_image: str | None = None
    af_image: str | None = None
    af_registered: str | None = None
    heatmap_overlay: str | None = None
    relative_af_map: str | None = None
    relative_af_grid: str | None = Field(
        default=None, description="Downsampled relative-AF grid (JSON) for 3D/interactive views."
    )
    mucosa_mask: str | None = None
    wl_saliency: str | None = None


class WorkingResolution(_Model):
    width_px: int
    height_px: int
    scale_from_original: float = Field(description="working width / original width")


class SiteResult(_Model):
    site: OralSite
    site_label: str
    role: SiteRole = SiteRole.EXAMINED
    paired_site: OralSite | None = None
    pairing: Literal["contralateral", "analogous"] | None = None
    visual_flag: VisualFlag = Field(default_factory=VisualFlag)
    diascopy: Diascopy = Diascopy.NOT_DONE
    captures: list[CaptureRecord] = Field(default_factory=list)
    qc: QCResult
    analysis_status: Literal["analysed", "inconclusive_qc_failed", "not_analysed"]
    calibration: CalibrationInfo
    registration: RegistrationSummary | None = None
    segmentation: SegmentationSummary | None = None
    af: AFAnalysis | None = None
    wl: WLAnalysis | None = None
    spectral_reading_ids: list[str] = Field(default_factory=list)
    artifacts: SiteArtifacts = Field(default_factory=SiteArtifacts)
    working_resolution: WorkingResolution | None = None
    px_per_mm: float | None = Field(default=None, description="At working resolution, if known.")
    site_risk_level: RiskLevel


# --------------------------------------------------------------------------------------------
# Spectral (Confirm probe)
# --------------------------------------------------------------------------------------------


class SpectralFeatures(_Model):
    ratio_555_630: float
    ratio_590_680: float
    ratio_515_630: float
    ratio_415_555: float
    hb_absorption_proxy: float = Field(
        description="Depth of the 540-580 nm haemoglobin dip relative to a 515->630 nm baseline."
    )
    slope_480_555: float = Field(description="Log-reflectance slope per 100 nm.")
    slope_590_680: float = Field(description="Log-reflectance slope per 100 nm.")


class SpectralClassifierOutput(_Model):
    status: Literal["ok", "placeholder", "model_unavailable", "skipped_qc_failed"]
    method: str
    suspicion_score: Probability | None = Field(
        default=None, description="Heuristic 0..1 score. Not a calibrated probability."
    )
    label: Literal["not_suspicious", "indeterminate", "suspicious"] | None = None
    reasons: list[str] = Field(default_factory=list)
    model: ModelInfo


class SpectralReadingResult(_Model):
    reading_id: str
    site: OralSite
    role: Literal["lesion", "reference"]
    lesion_ref: str | None = Field(default=None, description="Region/lesion this reading targets.")
    capture: CaptureMetadata
    n_readings: int = Field(ge=0)
    signal_type: Literal["reflectance", "relative_signal"] = Field(
        description="'reflectance' when calibrated (S-D)/(W-D); otherwise dark-subtracted signal "
        "normalised to its own spectral sum."
    )
    calibration: CalibrationInfo
    values: dict[str, float] = Field(description="Mean per AS7341 channel.")
    values_std: dict[str, float] = Field(description="Std-dev across the N readings.")
    qc: QCResult
    features: SpectralFeatures | None = None
    reference_reading_id: str | None = None
    delta_vs_reference: SpectralFeatures | None = Field(
        default=None,
        description="Features of the per-channel lesion / reference quotient minus the features "
        "of a flat spectrum (spectral/features.py reference_delta): ratios become relative "
        "changes, hb_absorption_proxy the haemoglobin dip of the quotient, slopes differences. "
        "Independent of the probe's LED, filters and gain.",
    )
    classifier: SpectralClassifierOutput | None = None


# --------------------------------------------------------------------------------------------
# Triage
# --------------------------------------------------------------------------------------------


class ContributingFactor(_Model):
    code: str = Field(description="Stable machine code, e.g. 'af.fvl_present'.")
    label: str = Field(description="Human-readable reason.")
    source: FactorSource
    points: float = Field(description="Contribution to the rule score; may be negative.")
    detail: str | None = None


class RecommendedAction(_Model):
    code: ActionCode
    summary: str
    steps: list[str] = Field(default_factory=list)
    timeline_days: int | None = Field(default=None, ge=0)
    referral_urgency: ReferralUrgency


class LesionAssessment(_Model):
    lesion_id: str
    site: OralSite
    origins: list[LesionOrigin]
    region_ids: list[str] = Field(default_factory=list)
    spectral_reading_ids: list[str] = Field(default_factory=list)
    location: NormPoint | None = None
    risk_level: RiskLevel
    score: float = Field(description="Rule score. Always read together with the factors.")
    contributing_factors: list[ContributingFactor]
    confounders: list[ConfounderFlag] = Field(default_factory=list)
    recommended_action: RecommendedAction
    summary: str = Field(description="One plain-language sentence.")


class ScoreThresholds(_Model):
    moderate: float
    high: float


class TriageResult(_Model):
    overall_risk_level: RiskLevel
    score: float | None = None
    thresholds: ScoreThresholds
    contributing_factors: list[ContributingFactor]
    overrides_applied: list[ContributingFactor] = Field(
        default_factory=list,
        description="Clinical red-flag floors. The device can never lower these.",
    )
    recommended_action: RecommendedAction
    highest_risk_lesion_id: str | None = None
    rule_config: ConfigRef
    explanation: str = Field(description="Plain-language explanation of the level.")
    requires_clinician_review: Literal[True] = True
    review_status: Literal["pending_review", "reviewed"] = "pending_review"


# --------------------------------------------------------------------------------------------
# Top level
# --------------------------------------------------------------------------------------------


class Notices(_Model):
    prototype_banner: str = PROTOTYPE_BANNER
    disclaimer: str = DISCLAIMER
    synthetic_notice: str | None = None


class VersionInfo(_Model):
    software: str
    pipeline: str
    schema_version: str = SCHEMA_VERSION
    rule_config: ConfigRef
    qc_config: ConfigRef
    analysis_config: ConfigRef
    models: list[ModelInfo]


class QCSummary(_Model):
    passed_all: bool
    sites_failed: list[OralSite] = Field(default_factory=list)
    captures_failed: int = 0
    retakes_recommended: list[str] = Field(default_factory=list)
    issues: list[QCIssue] = Field(default_factory=list)


class Limitation(_Model):
    code: str
    message: str


class StageTiming(_Model):
    stage: str
    duration_ms: float
    status: Literal["ok", "skipped", "failed"] = "ok"
    site: OralSite | None = None


class ProcessingInfo(_Model):
    working_max_side_px: int
    duration_ms: float
    stages: list[StageTiming] = Field(default_factory=list)


class ClinicianReview(_Model):
    reviewer_name: str
    reviewer_id: str | None = None
    agreed: bool
    modified_risk_level: RiskLevel | None = None
    notes: str | None = None
    signed_at: datetime


class ScreeningResult(_Model):
    """Everything known about one screening, one analysis version."""

    schema_version: Literal["1.0.0"] = "1.0.0"
    result_id: str
    screening_id: str
    result_version: int = Field(ge=1, description="Re-analysis creates a new version.")
    supersedes_result_id: str | None = None
    created_at: datetime
    data_provenance: DataProvenance
    synthetic: bool = Field(description="True if any input is synthetic.")
    notices: Notices = Field(default_factory=Notices)
    versions: VersionInfo
    context: ScreeningContext
    patient: PatientInfo
    consent: ConsentInfo
    risk_factors: RiskFactors
    sites: list[SiteResult]
    spectral_readings: list[SpectralReadingResult] = Field(default_factory=list)
    lesions: list[LesionAssessment] = Field(default_factory=list)
    triage: TriageResult
    qc_summary: QCSummary
    calibration_status: CalibrationStatus
    limitations: list[Limitation] = Field(default_factory=list)
    processing: ProcessingInfo
    review: ClinicianReview | None = None
