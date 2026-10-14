/* eslint-disable */
/**
 * GENERATED FILE — do not edit by hand.
 * Source: pipeline/mukha_pipeline/schema/screening_result.v1.schema.json (ScreeningResult v1.0.0)
 * Regenerate with: npm run gen:types
 */

/**
 * This interface was referenced by `ScreeningResult`'s JSON-Schema
 * via the `definition` "DataProvenance".
 */
export type DataProvenance = 'real' | 'synthetic' | 'phantom' | 'ex_vivo' | 'volunteer' | 'mixed';
/**
 * This interface was referenced by `ScreeningResult`'s JSON-Schema
 * via the `definition` "ModelComponent".
 */
export type ModelComponent =
  'segmenter' | 'wl_lesion_classifier' | 'spectral_classifier' | 'fusion';
/**
 * This interface was referenced by `ScreeningResult`'s JSON-Schema
 * via the `definition` "ModelStatus".
 */
export type ModelStatus = 'available' | 'model_unavailable' | 'placeholder' | 'baseline_rules';
export type Language = 'en' | 'hi' | 'kn';
export type Sex = 'female' | 'male' | 'other' | 'undisclosed';
export type Language1 = 'en' | 'hi' | 'kn';
/**
 * This interface was referenced by `ScreeningResult`'s JSON-Schema
 * via the `definition` "OralSite".
 */
export type OralSite =
  | 'upper_lip_mucosa'
  | 'lower_lip_mucosa'
  | 'buccal_mucosa_left'
  | 'buccal_mucosa_right'
  | 'tongue_dorsum'
  | 'lateral_tongue_left'
  | 'lateral_tongue_right'
  | 'ventral_tongue_floor_of_mouth'
  | 'hard_palate'
  | 'soft_palate'
  | 'upper_gingiva'
  | 'lower_gingiva'
  | 'retromolar_left'
  | 'retromolar_right';
export type SiteRole = 'examined' | 'reference_only';
/**
 * This interface was referenced by `ScreeningResult`'s JSON-Schema
 * via the `definition` "VisualAppearance".
 */
export type VisualAppearance =
  'white_patch' | 'red_patch' | 'mixed_red_white' | 'ulcer' | 'growth' | 'other';
/**
 * Pressing a glass slide on a dark (FVL) area. If fluorescence returns, the darkness was
 * most likely blood/inflammation rather than a tissue change.
 */
export type Diascopy = 'not_done' | 'fvl_persists' | 'fvl_resolves';
/**
 * This interface was referenced by `ScreeningResult`'s JSON-Schema
 * via the `definition` "CaptureKind".
 */
export type CaptureKind = 'white_light' | 'autofluorescence' | 'dark_frame' | 'spectral';
/**
 * This interface was referenced by `ScreeningResult`'s JSON-Schema
 * via the `definition` "SourceAdapter".
 */
export type SourceAdapter =
  | 'upload'
  | 'phone_camera'
  | 'scout_ble'
  | 'dedicated_camera'
  | 'confirm_probe'
  | 'mock_device'
  | 'synthetic';
/**
 * This interface was referenced by `ScreeningResult`'s JSON-Schema
 * via the `definition` "LedMode".
 */
export type LedMode = 'OFF' | 'WHITE' | 'VIOLET';
/**
 * This interface was referenced by `ScreeningResult`'s JSON-Schema
 * via the `definition` "QCSeverity".
 */
export type QCSeverity = 'error' | 'warning' | 'info';
/**
 * This interface was referenced by `ScreeningResult`'s JSON-Schema
 * via the `definition` "CalibrationStatus".
 */
export type CalibrationStatus = 'calibrated' | 'uncalibrated' | 'expired' | 'partial';
/**
 * This interface was referenced by `ScreeningResult`'s JSON-Schema
 * via the `definition` "ReferenceType".
 */
export type ReferenceType = 'contralateral' | 'analogous_site' | 'in_image_auto' | 'none';
/**
 * This interface was referenced by `ScreeningResult`'s JSON-Schema
 * via the `definition` "ConfounderType".
 */
export type ConfounderType = 'porphyrin_red' | 'pigmentation' | 'blood' | 'glare';
/**
 * This interface was referenced by `ScreeningResult`'s JSON-Schema
 * via the `definition` "WLLabel".
 */
export type WLLabel = 'normal' | 'white_patch' | 'red_patch' | 'ulcer_mass' | 'other';
/**
 * This interface was referenced by `ScreeningResult`'s JSON-Schema
 * via the `definition` "RiskLevel".
 */
export type RiskLevel = 'LOW' | 'MODERATE' | 'HIGH' | 'INCONCLUSIVE';
/**
 * This interface was referenced by `ScreeningResult`'s JSON-Schema
 * via the `definition` "LesionOrigin".
 */
export type LesionOrigin = 'af_fvl' | 'visual' | 'wl_model' | 'spectral';
/**
 * This interface was referenced by `ScreeningResult`'s JSON-Schema
 * via the `definition` "FactorSource".
 */
export type FactorSource =
  | 'autofluorescence'
  | 'wl_model'
  | 'spectral'
  | 'clinical_visual'
  | 'clinical_history'
  | 'symptom'
  | 'site'
  | 'qc'
  | 'override';
/**
 * This interface was referenced by `ScreeningResult`'s JSON-Schema
 * via the `definition` "ActionCode".
 */
export type ActionCode =
  | 'routine_rescreen_and_cessation'
  | 'remove_irritants_reexamine_2_weeks'
  | 'urgent_referral_biopsy'
  | 'retake_or_clinical_exam';
/**
 * This interface was referenced by `ScreeningResult`'s JSON-Schema
 * via the `definition` "ReferralUrgency".
 */
export type ReferralUrgency = 'none' | 'reexamine' | 'urgent' | 'retake';
/**
 * Pressing a glass slide on a dark (FVL) area. If fluorescence returns, the darkness was
 * most likely blood/inflammation rather than a tissue change.
 *
 * This interface was referenced by `ScreeningResult`'s JSON-Schema
 * via the `definition` "Diascopy".
 */
export type Diascopy1 = 'not_done' | 'fvl_persists' | 'fvl_resolves';
/**
 * This interface was referenced by `ScreeningResult`'s JSON-Schema
 * via the `definition` "Language".
 */
export type Language2 = 'en' | 'hi' | 'kn';
/**
 * This interface was referenced by `ScreeningResult`'s JSON-Schema
 * via the `definition` "Sex".
 */
export type Sex1 = 'female' | 'male' | 'other' | 'undisclosed';
/**
 * This interface was referenced by `ScreeningResult`'s JSON-Schema
 * via the `definition` "SiteRole".
 */
export type SiteRole1 = 'examined' | 'reference_only';

/**
 * Everything known about one screening, one analysis version.
 */
export interface ScreeningResult {
  schema_version?: '1.0.0';
  result_id: string;
  screening_id: string;
  /**
   * Re-analysis creates a new version.
   */
  result_version: number;
  supersedes_result_id?: string | null;
  created_at: string;
  data_provenance: DataProvenance;
  /**
   * True if any input is synthetic.
   */
  synthetic: boolean;
  notices?: Notices;
  versions: VersionInfo;
  context: ScreeningContext;
  patient: PatientInfo;
  consent: ConsentInfo;
  risk_factors: RiskFactors;
  sites: SiteResult[];
  spectral_readings?: SpectralReadingResult[];
  lesions?: LesionAssessment[];
  triage: TriageResult;
  qc_summary: QCSummary;
  calibration_status: CalibrationStatus;
  limitations?: Limitation[];
  processing: ProcessingInfo;
  review?: ClinicianReview | null;
}
/**
 * This interface was referenced by `ScreeningResult`'s JSON-Schema
 * via the `definition` "Notices".
 */
export interface Notices {
  prototype_banner?: string;
  disclaimer?: string;
  synthetic_notice?: string | null;
}
/**
 * This interface was referenced by `ScreeningResult`'s JSON-Schema
 * via the `definition` "VersionInfo".
 */
export interface VersionInfo {
  software: string;
  pipeline: string;
  schema_version?: string;
  rule_config: ConfigRef;
  qc_config: ConfigRef;
  analysis_config: ConfigRef;
  models: ModelInfo[];
}
/**
 * This interface was referenced by `ScreeningResult`'s JSON-Schema
 * via the `definition` "ConfigRef".
 */
export interface ConfigRef {
  name: string;
  version: string;
  /**
   * SHA-256 of the exact config file used.
   */
  sha256: string;
}
/**
 * This interface was referenced by `ScreeningResult`'s JSON-Schema
 * via the `definition` "ModelInfo".
 */
export interface ModelInfo {
  component: ModelComponent;
  name: string;
  version: string;
  status: ModelStatus;
  /**
   * True only after a documented clinical validation. Never set by code.
   */
  validated?: boolean;
  /**
   * Path or URI of the model card.
   */
  model_card?: string | null;
  sha256?: string | null;
  notes?: string | null;
}
/**
 * This interface was referenced by `ScreeningResult`'s JSON-Schema
 * via the `definition` "ScreeningContext".
 */
export interface ScreeningContext {
  /**
   * Camp or clinic name.
   */
  screening_site_name?: string | null;
  camp_id?: string | null;
  health_worker_id?: string | null;
  health_worker_name?: string | null;
  device_ids?: string[];
  firmware_versions?: string[];
  captured_at?: string | null;
  language?: Language;
}
/**
 * Minimised clinical view of the patient. Name and contact details live in a separate PII
 * store keyed by ``pseudonymous_id`` and are joined only when a report is rendered.
 *
 * This interface was referenced by `ScreeningResult`'s JSON-Schema
 * via the `definition` "PatientInfo".
 */
export interface PatientInfo {
  pseudonymous_id: string;
  age_years?: number | null;
  sex?: Sex;
}
/**
 * This interface was referenced by `ScreeningResult`'s JSON-Schema
 * via the `definition` "ConsentInfo".
 */
export interface ConsentInfo {
  consent_recorded: boolean;
  method?: 'checkbox' | 'signature' | 'thumbprint' | 'none';
  language?: Language1;
  /**
   * Separate opt-in to use de-identified data to improve the tool.
   */
  research_use_opt_in?: boolean;
  consent_id?: string | null;
  recorded_at?: string | null;
}
/**
 * This interface was referenced by `ScreeningResult`'s JSON-Schema
 * via the `definition` "RiskFactors".
 */
export interface RiskFactors {
  smokeless_tobacco?: SmokelessTobaccoUse;
  smoking?: SmokingUse;
  areca_nut?: ArecaNutUse;
  alcohol?: AlcoholUse;
  prior_oral_lesion?: boolean;
  prior_oral_cancer?: boolean;
  symptoms?: Symptoms;
  irritants?: Irritants;
}
/**
 * This interface was referenced by `ScreeningResult`'s JSON-Schema
 * via the `definition` "SmokelessTobaccoUse".
 */
export interface SmokelessTobaccoUse {
  uses?: boolean;
  former?: boolean;
  types?: (
    'gutka' | 'khaini' | 'zarda' | 'paan_with_tobacco' | 'mawa' | 'mishri' | 'snuff' | 'other'
  )[];
  years?: number | null;
  per_day?: number | null;
}
/**
 * This interface was referenced by `ScreeningResult`'s JSON-Schema
 * via the `definition` "SmokingUse".
 */
export interface SmokingUse {
  uses?: boolean;
  former?: boolean;
  types?: ('bidi' | 'cigarette' | 'hookah' | 'chutta_reverse' | 'other')[];
  years?: number | null;
  per_day?: number | null;
}
/**
 * Areca nut without tobacco (supari, paan without tobacco). Linked to submucous fibrosis.
 *
 * This interface was referenced by `ScreeningResult`'s JSON-Schema
 * via the `definition` "ArecaNutUse".
 */
export interface ArecaNutUse {
  uses?: boolean;
  former?: boolean;
  years?: number | null;
  per_day?: number | null;
}
/**
 * This interface was referenced by `ScreeningResult`'s JSON-Schema
 * via the `definition` "AlcoholUse".
 */
export interface AlcoholUse {
  uses?: boolean;
  frequency?: 'never' | 'occasional' | 'weekly' | 'daily';
}
/**
 * This interface was referenced by `ScreeningResult`'s JSON-Schema
 * via the `definition` "Symptoms".
 */
export interface Symptoms {
  non_healing_ulcer_over_2_weeks?: boolean;
  white_patch?: boolean;
  red_patch?: boolean;
  lump?: boolean;
  pain?: boolean;
  difficulty_opening_mouth?: boolean;
  bleeding?: boolean;
  burning_sensation?: boolean;
  neck_lump?: boolean;
}
/**
 * This interface was referenced by `ScreeningResult`'s JSON-Schema
 * via the `definition` "Irritants".
 */
export interface Irritants {
  sharp_tooth?: boolean;
  ill_fitting_denture?: boolean;
}
/**
 * This interface was referenced by `ScreeningResult`'s JSON-Schema
 * via the `definition` "SiteResult".
 */
export interface SiteResult {
  site: OralSite;
  site_label: string;
  role?: SiteRole;
  paired_site?: OralSite | null;
  pairing?: ('contralateral' | 'analogous') | null;
  visual_flag?: VisualFlag;
  diascopy?: Diascopy;
  captures?: CaptureRecord[];
  qc: QCResult;
  analysis_status: 'analysed' | 'inconclusive_qc_failed' | 'not_analysed';
  calibration: CalibrationInfo;
  registration?: RegistrationSummary | null;
  segmentation?: SegmentationSummary | null;
  af?: AFAnalysis | null;
  wl?: WLAnalysis | null;
  spectral_reading_ids?: string[];
  artifacts?: SiteArtifacts;
  working_resolution?: WorkingResolution | null;
  /**
   * At working resolution, if known.
   */
  px_per_mm?: number | null;
  site_risk_level: RiskLevel;
}
/**
 * What the health worker saw with their own eyes at this site.
 *
 * This interface was referenced by `ScreeningResult`'s JSON-Schema
 * via the `definition` "VisualFlag".
 */
export interface VisualFlag {
  lesion_seen?: boolean;
  appearance?: VisualAppearance[];
  description?: string | null;
}
/**
 * This interface was referenced by `ScreeningResult`'s JSON-Schema
 * via the `definition` "CaptureRecord".
 */
export interface CaptureRecord {
  metadata: CaptureMetadata;
  qc: QCResult;
  /**
   * Artifact key of the stored image.
   */
  artifact?: string | null;
}
/**
 * This interface was referenced by `ScreeningResult`'s JSON-Schema
 * via the `definition` "CaptureMetadata".
 */
export interface CaptureMetadata {
  capture_id: string;
  kind: CaptureKind;
  oral_site: OralSite;
  source_adapter: SourceAdapter;
  device_id?: string | null;
  firmware_version?: string | null;
  led_mode?: LedMode | null;
  exposure_time_ms?: number | null;
  iso?: number | null;
  /**
   * e.g. 'locked:5000K', 'auto'.
   */
  white_balance?: string | null;
  focus?: string | null;
  captured_at?: string | null;
  calibration_id?: string | null;
  width_px?: number | null;
  height_px?: number | null;
  file_name?: string | null;
  file_sha256?: string | null;
  mime_type?: string | null;
  synthetic?: boolean;
  extra?: {
    [k: string]: string | number | boolean | null;
  };
}
/**
 * This interface was referenced by `ScreeningResult`'s JSON-Schema
 * via the `definition` "QCResult".
 */
export interface QCResult {
  /**
   * False if any issue has severity 'error'.
   */
  passed: boolean;
  /**
   * 0..1 overall capture quality (1 = no concerns).
   */
  score: number;
  issues?: QCIssue[];
  metrics?: {
    [k: string]: number;
  };
}
/**
 * This interface was referenced by `ScreeningResult`'s JSON-Schema
 * via the `definition` "QCIssue".
 */
export interface QCIssue {
  /**
   * Stable machine code, e.g. 'blur', 'glare', 'af_signal_low'.
   */
  code: string;
  severity: QCSeverity;
  /**
   * Plain-language retake prompt in English.
   */
  message: string;
  /**
   * i18n key for the retake prompt, e.g. 'qc.blur'.
   */
  message_key: string;
  metric?: string | null;
  value?: number | null;
  threshold?: number | null;
}
/**
 * This interface was referenced by `ScreeningResult`'s JSON-Schema
 * via the `definition` "CalibrationInfo".
 */
export interface CalibrationInfo {
  status: CalibrationStatus;
  calibration_id?: string | null;
  expires_at?: string | null;
  note?: string | null;
}
/**
 * This interface was referenced by `ScreeningResult`'s JSON-Schema
 * via the `definition` "RegistrationSummary".
 */
export interface RegistrationSummary {
  method: 'ecc_affine' | 'ecc_homography' | 'orb_homography' | 'identity';
  confidence: number;
  passed: boolean;
  /**
   * 3x3 matrix mapping AF pixel coordinates into the WL frame (working res).
   */
  warp_matrix: number[][];
  note?: string | null;
}
/**
 * This interface was referenced by `ScreeningResult`'s JSON-Schema
 * via the `definition` "SegmentationSummary".
 */
export interface SegmentationSummary {
  method: string;
  version: string;
  status: 'ok' | 'low_confidence' | 'failed';
  mucosa_fraction: number;
  teeth_fraction: number;
  specular_fraction: number;
  confidence: number;
}
/**
 * This interface was referenced by `ScreeningResult`'s JSON-Schema
 * via the `definition` "AFAnalysis".
 */
export interface AFAnalysis {
  status: 'ok' | 'skipped_qc_failed' | 'insufficient_mucosa' | 'no_reference';
  reference: AFReference;
  /**
   * Relative-AF threshold below which tissue is FVL.
   */
  fvl_threshold: number;
  min_region_area_px: number;
  site_medians?: AFSiteMedians | null;
  fvl_area_fraction?: number;
  /**
   * Fraction of mucosa with red (porphyrin) fluorescence.
   */
  porphyrin_red_fraction?: number;
  regions?: FVLRegion[];
}
/**
 * This interface was referenced by `ScreeningResult`'s JSON-Schema
 * via the `definition` "AFReference".
 */
export interface AFReference {
  type: ReferenceType;
  reference_site?: OralSite | null;
  /**
   * Median exposure-normalised green fluorescence of the reference.
   */
  reference_green?: number | null;
  note?: string | null;
}
/**
 * This interface was referenced by `ScreeningResult`'s JSON-Schema
 * via the `definition` "AFSiteMedians".
 */
export interface AFSiteMedians {
  green: number;
  gr_ratio: number;
  af_index: number;
  relative_af: number;
}
/**
 * This interface was referenced by `ScreeningResult`'s JSON-Schema
 * via the `definition` "FVLRegion".
 */
export interface FVLRegion {
  region_id: string;
  area_px: number;
  area_mm2?: number | null;
  area_fraction_of_mucosa: number;
  centroid: NormPoint;
  bbox: NormBox;
  mean_relative_af: number;
  /**
   * 5th percentile (robust minimum).
   */
  min_relative_af: number;
  /**
   * Mean relative AF of a ring around the region / region mean. >1 = darker.
   */
  contrast_vs_surround: number;
  /**
   * 1 - circularity (4*pi*A/P^2).
   */
  border_irregularity: number;
  solidity: number;
  confounders?: ConfounderFlag[];
}
/**
 * Point in normalised image coordinates of the white-light frame (0..1, origin top-left).
 *
 * This interface was referenced by `ScreeningResult`'s JSON-Schema
 * via the `definition` "NormPoint".
 */
export interface NormPoint {
  x: number;
  y: number;
}
/**
 * This interface was referenced by `ScreeningResult`'s JSON-Schema
 * via the `definition` "NormBox".
 */
export interface NormBox {
  x: number;
  y: number;
  w: number;
  h: number;
}
/**
 * This interface was referenced by `ScreeningResult`'s JSON-Schema
 * via the `definition` "ConfounderFlag".
 */
export interface ConfounderFlag {
  type: ConfounderType;
  overlap_fraction: number;
  note: string;
}
/**
 * This interface was referenced by `ScreeningResult`'s JSON-Schema
 * via the `definition` "WLAnalysis".
 */
export interface WLAnalysis {
  status: 'ok' | 'model_unavailable' | 'skipped_qc_failed' | 'error';
  model: ModelInfo;
  predictions?: WLPrediction[];
  top_label?: WLLabel | null;
  /**
   * Clinician-only explainability map (artifact key).
   */
  saliency_artifact?: string | null;
  note?: string | null;
}
/**
 * This interface was referenced by `ScreeningResult`'s JSON-Schema
 * via the `definition` "WLPrediction".
 */
export interface WLPrediction {
  label: WLLabel;
  probability: number;
}
/**
 * Artifact keys (relative paths in the result's artifact store).
 *
 * This interface was referenced by `ScreeningResult`'s JSON-Schema
 * via the `definition` "SiteArtifacts".
 */
export interface SiteArtifacts {
  wl_image?: string | null;
  af_image?: string | null;
  af_registered?: string | null;
  heatmap_overlay?: string | null;
  relative_af_map?: string | null;
  /**
   * Downsampled relative-AF grid (JSON) for 3D/interactive views.
   */
  relative_af_grid?: string | null;
  mucosa_mask?: string | null;
  wl_saliency?: string | null;
}
/**
 * This interface was referenced by `ScreeningResult`'s JSON-Schema
 * via the `definition` "WorkingResolution".
 */
export interface WorkingResolution {
  width_px: number;
  height_px: number;
  /**
   * working width / original width
   */
  scale_from_original: number;
}
/**
 * This interface was referenced by `ScreeningResult`'s JSON-Schema
 * via the `definition` "SpectralReadingResult".
 */
export interface SpectralReadingResult {
  reading_id: string;
  site: OralSite;
  role: 'lesion' | 'reference';
  /**
   * Region/lesion this reading targets.
   */
  lesion_ref?: string | null;
  capture: CaptureMetadata;
  n_readings: number;
  /**
   * 'reflectance' when calibrated (S-D)/(W-D); otherwise dark-subtracted signal normalised to its own spectral sum.
   */
  signal_type: 'reflectance' | 'relative_signal';
  calibration: CalibrationInfo;
  /**
   * Mean per AS7341 channel.
   */
  values: {
    [k: string]: number;
  };
  /**
   * Std-dev across the N readings.
   */
  values_std: {
    [k: string]: number;
  };
  qc: QCResult;
  features?: SpectralFeatures | null;
  reference_reading_id?: string | null;
  /**
   * Lesion features minus reference features.
   */
  delta_vs_reference?: SpectralFeatures | null;
  classifier?: SpectralClassifierOutput | null;
}
/**
 * This interface was referenced by `ScreeningResult`'s JSON-Schema
 * via the `definition` "SpectralFeatures".
 */
export interface SpectralFeatures {
  ratio_555_630: number;
  ratio_590_680: number;
  ratio_515_630: number;
  ratio_415_555: number;
  /**
   * Depth of the 540-580 nm haemoglobin dip relative to a 515->630 nm baseline.
   */
  hb_absorption_proxy: number;
  /**
   * Log-reflectance slope per 100 nm.
   */
  slope_480_555: number;
  /**
   * Log-reflectance slope per 100 nm.
   */
  slope_590_680: number;
}
/**
 * This interface was referenced by `ScreeningResult`'s JSON-Schema
 * via the `definition` "SpectralClassifierOutput".
 */
export interface SpectralClassifierOutput {
  status: 'ok' | 'placeholder' | 'model_unavailable' | 'skipped_qc_failed';
  method: string;
  /**
   * Heuristic 0..1 score. Not a calibrated probability.
   */
  suspicion_score?: number | null;
  label?: ('not_suspicious' | 'indeterminate' | 'suspicious') | null;
  reasons?: string[];
  model: ModelInfo;
}
/**
 * This interface was referenced by `ScreeningResult`'s JSON-Schema
 * via the `definition` "LesionAssessment".
 */
export interface LesionAssessment {
  lesion_id: string;
  site: OralSite;
  origins: LesionOrigin[];
  region_ids?: string[];
  spectral_reading_ids?: string[];
  location?: NormPoint | null;
  risk_level: RiskLevel;
  /**
   * Rule score. Always read together with the factors.
   */
  score: number;
  contributing_factors: ContributingFactor[];
  confounders?: ConfounderFlag[];
  recommended_action: RecommendedAction;
  /**
   * One plain-language sentence.
   */
  summary: string;
}
/**
 * This interface was referenced by `ScreeningResult`'s JSON-Schema
 * via the `definition` "ContributingFactor".
 */
export interface ContributingFactor {
  /**
   * Stable machine code, e.g. 'af.fvl_present'.
   */
  code: string;
  /**
   * Human-readable reason.
   */
  label: string;
  source: FactorSource;
  /**
   * Contribution to the rule score; may be negative.
   */
  points: number;
  detail?: string | null;
}
/**
 * This interface was referenced by `ScreeningResult`'s JSON-Schema
 * via the `definition` "RecommendedAction".
 */
export interface RecommendedAction {
  code: ActionCode;
  summary: string;
  steps?: string[];
  timeline_days?: number | null;
  referral_urgency: ReferralUrgency;
}
/**
 * This interface was referenced by `ScreeningResult`'s JSON-Schema
 * via the `definition` "TriageResult".
 */
export interface TriageResult {
  overall_risk_level: RiskLevel;
  score?: number | null;
  thresholds: ScoreThresholds;
  contributing_factors: ContributingFactor[];
  /**
   * Clinical red-flag floors. The device can never lower these.
   */
  overrides_applied?: ContributingFactor[];
  recommended_action: RecommendedAction;
  highest_risk_lesion_id?: string | null;
  rule_config: ConfigRef;
  /**
   * Plain-language explanation of the level.
   */
  explanation: string;
  requires_clinician_review?: boolean;
  review_status?: 'pending_review' | 'reviewed';
}
/**
 * This interface was referenced by `ScreeningResult`'s JSON-Schema
 * via the `definition` "ScoreThresholds".
 */
export interface ScoreThresholds {
  moderate: number;
  high: number;
}
/**
 * This interface was referenced by `ScreeningResult`'s JSON-Schema
 * via the `definition` "QCSummary".
 */
export interface QCSummary {
  passed_all: boolean;
  sites_failed?: OralSite[];
  captures_failed?: number;
  retakes_recommended?: string[];
  issues?: QCIssue[];
}
/**
 * This interface was referenced by `ScreeningResult`'s JSON-Schema
 * via the `definition` "Limitation".
 */
export interface Limitation {
  code: string;
  message: string;
}
/**
 * This interface was referenced by `ScreeningResult`'s JSON-Schema
 * via the `definition` "ProcessingInfo".
 */
export interface ProcessingInfo {
  working_max_side_px: number;
  duration_ms: number;
  stages?: StageTiming[];
}
/**
 * This interface was referenced by `ScreeningResult`'s JSON-Schema
 * via the `definition` "StageTiming".
 */
export interface StageTiming {
  stage: string;
  duration_ms: number;
  status?: 'ok' | 'skipped' | 'failed';
  site?: OralSite | null;
}
/**
 * This interface was referenced by `ScreeningResult`'s JSON-Schema
 * via the `definition` "ClinicianReview".
 */
export interface ClinicianReview {
  reviewer_name: string;
  reviewer_id?: string | null;
  agreed: boolean;
  modified_risk_level?: RiskLevel | null;
  notes?: string | null;
  signed_at: string;
}
