/**
 * SYNTHETIC ScreeningResult fixtures for mock mode and tests. Typed against the generated
 * contract (src/api/types.gen.ts) and validated against the JSON Schema in fixtures.test.ts.
 *
 * Scenarios
 *  - HIGH: right lateral tongue with deep glow loss + white patch seen by the health worker,
 *    spectral lesion vs contralateral reference, left buccal finding partly explained by red
 *    porphyrin (plaque), lower gingiva failed QC with plain retake prompts, red-flag overrides.
 *  - LOW: both cheeks glow evenly; no finding.
 *  - INCONCLUSIVE: one examined site failed QC, so LOW is not allowed.
 *  - MODERATE: derived from HIGH (left buccal finding only) for the moderate demo.
 * None of these come from a patient. They must never support a performance claim.
 */
import en from '../i18n/en.json';
import type {
  CaptureKind,
  CaptureRecord,
  ConfigRef,
  ContributingFactor,
  ModelInfo,
  Notices,
  OralSite,
  QCIssue,
  QCResult,
  QCSeverity,
  RecommendedAction,
  RiskFactors,
  ScreeningResult,
  SiteResult,
  StageTiming,
  VersionInfo,
} from '../api/types';

export const PROTOTYPE_BANNER = 'Research prototype — not for clinical use';
export const DISCLAIMER =
  'Screening aid — not a diagnosis. MukhaSpectra flags suspicious findings and assigns a risk ' +
  'level to support triage. Final diagnosis requires examination by a clinician and ' +
  'histopathology.';
export const SYNTHETIC_NOTICE =
  'SYNTHETIC DATA — generated for demonstration and software testing. Not from a real patient. ' +
  'Must never be used to support any performance claim.';

const NOTICES: Notices = {
  prototype_banner: PROTOTYPE_BANNER,
  disclaimer: DISCLAIMER,
  synthetic_notice: SYNTHETIC_NOTICE,
};

const RULES: ConfigRef = {
  name: 'mukhaspectra-triage-rules',
  version: '1.0.0',
  sha256: '3f9c2a71d8e04b6a9c51f0e2d7a84b3c6e19f5a2b7c8d0e1f2a3b4c5d6e7f809',
};

const MODELS: ModelInfo[] = [
  {
    component: 'segmenter',
    name: 'classical_hsv_lab',
    version: '1.0.0',
    status: 'baseline_rules',
    validated: false,
    model_card: null,
    sha256: null,
    notes: 'Classical colour rules with hole filling.',
  },
  {
    component: 'wl_lesion_classifier',
    name: 'wl_lesion_classifier',
    version: '0.0.0',
    status: 'model_unavailable',
    validated: false,
    model_card: null,
    sha256: null,
    notes: 'No weights installed; white-light images are not scored by a model.',
  },
  {
    component: 'spectral_classifier',
    name: 'rules_v1',
    version: '1.0.0',
    status: 'placeholder',
    validated: false,
    model_card: null,
    sha256: null,
    notes: 'Heuristic thresholds on lesion-minus-reference deltas.',
  },
  {
    component: 'fusion',
    name: 'mukhaspectra-triage-rules',
    version: '1.0.0',
    status: 'baseline_rules',
    validated: false,
    model_card: null,
    sha256: null,
    notes: 'baseline_heuristic_unvalidated',
  },
];

const VERSIONS: VersionInfo = {
  software: '0.1.0',
  pipeline: '0.1.0',
  schema_version: '1.0.0',
  rule_config: RULES,
  qc_config: {
    name: 'mukhaspectra-qc',
    version: '1.0.0',
    sha256: 'a41be0c9d2f37e58b6a0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c9d0e1f2',
  },
  analysis_config: {
    name: 'mukhaspectra-analysis',
    version: '1.0.0',
    sha256: '9d7e1c3b5a4f2e8d0c6b9a7f5e3d1c2b4a6f8e0d9c7b5a3f1e2d4c6b8a0f9e7d',
  },
  models: MODELS,
};

const QC_TEXT = en.qc as Record<string, string>;

function issue(
  code: string,
  severity: QCSeverity,
  metric?: string,
  value?: number,
  threshold?: number,
): QCIssue {
  return {
    code,
    severity,
    message: QC_TEXT[code] ?? code,
    message_key: `qc.${code}`,
    metric: metric ?? null,
    value: value ?? null,
    threshold: threshold ?? null,
  };
}

function qc(score: number, issues: QCIssue[] = [], metrics: Record<string, number> = {}): QCResult {
  return { passed: !issues.some((i) => i.severity === 'error'), score, issues, metrics };
}

const LED: Record<CaptureKind, 'WHITE' | 'VIOLET' | 'OFF'> = {
  white_light: 'WHITE',
  autofluorescence: 'VIOLET',
  dark_frame: 'OFF',
  spectral: 'WHITE',
};

function capture(
  site: OralSite,
  kind: CaptureKind,
  id: string,
  result: QCResult,
  artifact: string | null,
  capturedAt: string,
): CaptureRecord {
  return {
    metadata: {
      capture_id: id,
      kind,
      oral_site: site,
      source_adapter: 'synthetic',
      device_id: 'scout-sim-01',
      firmware_version: '0.0.0-sim',
      led_mode: LED[kind],
      exposure_time_ms: kind === 'autofluorescence' ? 66.7 : kind === 'dark_frame' ? 66.7 : 16.7,
      iso: kind === 'white_light' ? 100 : 400,
      white_balance: 'locked:5000K',
      focus: 'locked',
      captured_at: capturedAt,
      calibration_id: 'cal-sim-0412',
      width_px: 2560,
      height_px: 1920,
      file_name: `SYNTHETIC_${site}_${kind === 'white_light' ? 'wl' : kind === 'autofluorescence' ? 'af' : 'dark'}.jpg`,
      file_sha256: null,
      mime_type: 'image/jpeg',
      synthetic: true,
      extra: {},
    },
    qc: result,
    artifact,
  };
}

const CALIBRATED = {
  status: 'calibrated' as const,
  calibration_id: 'cal-sim-0412',
  expires_at: '2026-10-13T08:00:00Z',
  note: null,
};

const ACTIONS: Record<'LOW' | 'MODERATE' | 'HIGH' | 'INCONCLUSIVE', RecommendedAction> = {
  LOW: {
    code: 'routine_rescreen_and_cessation',
    summary: 'No high-risk features found by this screen. This does not rule out disease.',
    steps: [
      'Routine re-screening in 12 months, or sooner if any symptom appears.',
      'Tobacco and areca-nut cessation counselling.',
    ],
    timeline_days: 365,
    referral_urgency: 'none',
  },
  MODERATE: {
    code: 'remove_irritants_reexamine_2_weeks',
    summary: 'Suspicious finding. Remove irritants and re-examine in 2 weeks.',
    steps: [
      'Remove local irritants (sharp tooth, ill-fitting denture) and stop tobacco and areca nut.',
      'Re-examine the same site in 2 weeks.',
      'If the lesion is still present at re-examination, refer for clinical examination.',
    ],
    timeline_days: 14,
    referral_urgency: 'reexamine',
  },
  HIGH: {
    code: 'urgent_referral_biopsy',
    summary: 'Urgent referral for specialist examination and biopsy.',
    steps: [
      'Refer urgently to an oral medicine / head-and-neck specialist.',
      'Specialist examination and biopsy as soon as possible, within 2 weeks.',
      'Tobacco and areca-nut cessation counselling.',
    ],
    timeline_days: 14,
    referral_urgency: 'urgent',
  },
  INCONCLUSIVE: {
    code: 'retake_or_clinical_exam',
    summary: 'Images could not be analysed reliably. Retake, or refer for clinical examination.',
    steps: [
      'Retake the failed captures now, following the on-screen prompts.',
      'If a lesion is visible or a retake is not possible, refer for clinical examination.',
    ],
    timeline_days: 0,
    referral_urgency: 'retake',
  },
};

const IDENTITY_WARP = [
  [1, 0, 0],
  [0, 1, 0],
  [0, 0, 1],
];

function stages(sites: OralSite[], spectral: boolean, failed: OralSite[] = []): StageTiming[] {
  const out: StageTiming[] = [];
  const seed = [212, 188, 241, 197];
  sites.forEach((site, i) => {
    const base = seed[i % seed.length]!;
    const bad = failed.includes(site);
    out.push({ stage: 'decode', duration_ms: 38 + i * 3, status: 'ok', site });
    out.push({ stage: 'qc', duration_ms: base, status: bad ? 'failed' : 'ok', site });
    out.push({
      stage: 'calibration',
      duration_ms: 24 + i * 2,
      status: bad ? 'skipped' : 'ok',
      site,
    });
    out.push({
      stage: 'registration',
      duration_ms: bad ? 0 : 412 + i * 37,
      status: bad ? 'skipped' : 'ok',
      site,
    });
    out.push({
      stage: 'segmentation',
      duration_ms: bad ? 0 : 131 + i * 11,
      status: bad ? 'skipped' : 'ok',
      site,
    });
    out.push({
      stage: 'af_analysis',
      duration_ms: bad ? 0 : 286 + i * 23,
      status: bad ? 'skipped' : 'ok',
      site,
    });
    out.push({ stage: 'wl_classifier', duration_ms: 1.2, status: 'skipped', site });
    out.push({
      stage: 'artifacts',
      duration_ms: bad ? 0 : 164 + i * 9,
      status: bad ? 'skipped' : 'ok',
      site,
    });
  });
  out.push({
    stage: 'spectral',
    duration_ms: spectral ? 46 : 0,
    status: spectral ? 'ok' : 'skipped',
    site: null,
  });
  out.push({ stage: 'triage', duration_ms: 9.4, status: 'ok', site: null });
  return out;
}

const total = (s: StageTiming[]) => Math.round(s.reduce((a, b) => a + b.duration_ms, 0) * 10) / 10;

const f = (
  code: string,
  label: string,
  source: ContributingFactor['source'],
  points: number,
  detail: string | null = null,
): ContributingFactor => ({ code, label, source, points, detail });

const NO_RISK_FACTORS: RiskFactors = {
  smokeless_tobacco: { uses: false, former: false, types: [], years: null, per_day: null },
  smoking: { uses: false, former: false, types: [], years: null, per_day: null },
  areca_nut: { uses: false, former: false, years: null, per_day: null },
  alcohol: { uses: false, frequency: 'never' },
  prior_oral_lesion: false,
  prior_oral_cancer: false,
  symptoms: {
    non_healing_ulcer_over_2_weeks: false,
    white_patch: false,
    red_patch: false,
    lump: false,
    pain: false,
    difficulty_opening_mouth: false,
    bleeding: false,
    burning_sensation: false,
    neck_lump: false,
  },
  irritants: { sharp_tooth: false, ill_fitting_denture: false },
};

// ---------------------------------------------------------------------------------------------
// HIGH
// ---------------------------------------------------------------------------------------------

const HIGH_AT = '2026-10-12T09:38:05Z';

const highTongueRight: SiteResult = {
  site: 'lateral_tongue_right',
  site_label: 'Right lateral tongue',
  role: 'examined',
  paired_site: 'lateral_tongue_left',
  pairing: 'contralateral',
  visual_flag: {
    lesion_seen: true,
    appearance: ['white_patch'],
    description: 'Rough white patch on the right side of the tongue; does not rub off.',
  },
  diascopy: 'not_done',
  captures: [
    capture(
      'lateral_tongue_right',
      'white_light',
      'cap-ltr-wl',
      qc(0.93, [issue('glare_minor', 'warning', 'glare_fraction', 0.021, 0.015)]),
      'high/lateral_tongue_right/wl.jpg',
      HIGH_AT,
    ),
    capture(
      'lateral_tongue_right',
      'autofluorescence',
      'cap-ltr-af',
      qc(0.96),
      'high/lateral_tongue_right/af.jpg',
      HIGH_AT,
    ),
    capture('lateral_tongue_right', 'dark_frame', 'cap-ltr-dark', qc(1), null, HIGH_AT),
  ],
  qc: qc(0.93, [issue('glare_minor', 'warning', 'glare_fraction', 0.021, 0.015)]),
  analysis_status: 'analysed',
  calibration: CALIBRATED,
  registration: {
    method: 'ecc_affine',
    confidence: 0.91,
    passed: true,
    warp_matrix: [
      [0.9712, -0.0339, 21.4],
      [0.0339, 0.9712, -2.8],
      [0, 0, 1],
    ],
    note: null,
  },
  segmentation: {
    method: 'classical_hsv_lab',
    version: '1.0.0',
    status: 'ok',
    mucosa_fraction: 0.71,
    teeth_fraction: 0.0,
    specular_fraction: 0.012,
    confidence: 0.84,
  },
  af: {
    status: 'ok',
    reference: {
      type: 'contralateral',
      reference_site: 'lateral_tongue_left',
      reference_green: 0.412,
      note: null,
    },
    fvl_threshold: 0.75,
    min_region_area_px: 150,
    site_medians: { green: 0.381, gr_ratio: 4.12, af_index: 0.377, relative_af: 0.93 },
    fvl_area_fraction: 0.071,
    porphyrin_red_fraction: 0.004,
    regions: [
      {
        region_id: 'r1',
        area_px: 13955,
        area_mm2: null,
        area_fraction_of_mucosa: 0.071,
        centroid: { x: 0.429, y: 0.561 },
        bbox: { x: 0.336, y: 0.379, w: 0.217, h: 0.344 },
        mean_relative_af: 0.455,
        min_relative_af: 0.391,
        contrast_vs_surround: 2.14,
        border_irregularity: 0.42,
        solidity: 0.81,
        confounders: [],
      },
    ],
  },
  wl: {
    status: 'model_unavailable',
    model: MODELS[1]!,
    predictions: [],
    top_label: null,
    saliency_artifact: null,
    note: 'White-light classifier not available; not used in triage.',
  },
  spectral_reading_ids: ['sp-l1'],
  artifacts: {
    wl_image: 'high/lateral_tongue_right/wl.jpg',
    af_image: 'high/lateral_tongue_right/af.jpg',
    af_registered: 'high/lateral_tongue_right/af_registered.jpg',
    heatmap_overlay: 'high/lateral_tongue_right/heatmap_overlay.jpg',
    relative_af_map: null,
    relative_af_grid: 'high/lateral_tongue_right/relative_af_grid.json',
    mucosa_mask: null,
    wl_saliency: null,
  },
  working_resolution: { width_px: 1024, height_px: 768, scale_from_original: 0.4 },
  px_per_mm: null,
  site_risk_level: 'HIGH',
};

const highTongueLeft: SiteResult = {
  site: 'lateral_tongue_left',
  site_label: 'Left lateral tongue',
  role: 'reference_only',
  paired_site: 'lateral_tongue_right',
  pairing: 'contralateral',
  visual_flag: { lesion_seen: false, appearance: [], description: null },
  diascopy: 'not_done',
  captures: [
    capture(
      'lateral_tongue_left',
      'white_light',
      'cap-ltl-wl',
      qc(0.97),
      'high/lateral_tongue_left/wl.jpg',
      HIGH_AT,
    ),
    capture(
      'lateral_tongue_left',
      'autofluorescence',
      'cap-ltl-af',
      qc(0.95),
      'high/lateral_tongue_left/af.jpg',
      HIGH_AT,
    ),
  ],
  qc: qc(0.95, [issue('no_dark_frame', 'warning')]),
  analysis_status: 'analysed',
  calibration: CALIBRATED,
  registration: {
    method: 'ecc_affine',
    confidence: 0.94,
    passed: true,
    warp_matrix: IDENTITY_WARP,
    note: null,
  },
  segmentation: {
    method: 'classical_hsv_lab',
    version: '1.0.0',
    status: 'ok',
    mucosa_fraction: 0.74,
    teeth_fraction: 0,
    specular_fraction: 0.009,
    confidence: 0.86,
  },
  af: {
    status: 'ok',
    reference: {
      type: 'in_image_auto',
      reference_site: null,
      reference_green: 0.412,
      note: 'Reference site for the right lateral tongue.',
    },
    fvl_threshold: 0.75,
    min_region_area_px: 150,
    site_medians: { green: 0.408, gr_ratio: 4.31, af_index: 0.405, relative_af: 0.99 },
    fvl_area_fraction: 0,
    porphyrin_red_fraction: 0.002,
    regions: [],
  },
  wl: null,
  spectral_reading_ids: ['sp-r1'],
  artifacts: {
    wl_image: 'high/lateral_tongue_left/wl.jpg',
    af_image: 'high/lateral_tongue_left/af.jpg',
    af_registered: null,
    heatmap_overlay: null,
    relative_af_map: null,
    relative_af_grid: null,
    mucosa_mask: null,
    wl_saliency: null,
  },
  working_resolution: { width_px: 1024, height_px: 768, scale_from_original: 0.4 },
  px_per_mm: null,
  site_risk_level: 'LOW',
};

const highBuccalLeft: SiteResult = {
  site: 'buccal_mucosa_left',
  site_label: 'Left buccal mucosa',
  role: 'examined',
  paired_site: 'buccal_mucosa_right',
  pairing: 'contralateral',
  visual_flag: {
    lesion_seen: true,
    appearance: ['other'],
    description: 'Faint brownish-red area near the bite line.',
  },
  diascopy: 'not_done',
  captures: [
    capture(
      'buccal_mucosa_left',
      'white_light',
      'cap-bml-wl',
      qc(0.95),
      'high/buccal_mucosa_left/wl.jpg',
      HIGH_AT,
    ),
    capture(
      'buccal_mucosa_left',
      'autofluorescence',
      'cap-bml-af',
      qc(0.92),
      'high/buccal_mucosa_left/af.jpg',
      HIGH_AT,
    ),
    capture('buccal_mucosa_left', 'dark_frame', 'cap-bml-dark', qc(1), null, HIGH_AT),
  ],
  qc: qc(0.92),
  analysis_status: 'analysed',
  calibration: CALIBRATED,
  registration: {
    method: 'ecc_affine',
    confidence: 0.88,
    passed: true,
    warp_matrix: [
      [0.9988, -0.0349, 13.1],
      [0.0349, 0.9988, -7.6],
      [0, 0, 1],
    ],
    note: null,
  },
  segmentation: {
    method: 'classical_hsv_lab',
    version: '1.0.0',
    status: 'ok',
    mucosa_fraction: 0.79,
    teeth_fraction: 0,
    specular_fraction: 0.006,
    confidence: 0.82,
  },
  af: {
    status: 'ok',
    reference: {
      type: 'in_image_auto',
      reference_site: null,
      reference_green: 0.398,
      note: 'Right buccal mucosa was not captured; brighter tissue in the same photo was used (less reliable than the other side).',
    },
    fvl_threshold: 0.75,
    min_region_area_px: 150,
    site_medians: { green: 0.379, gr_ratio: 3.66, af_index: 0.371, relative_af: 0.95 },
    fvl_area_fraction: 0.058,
    porphyrin_red_fraction: 0.031,
    regions: [
      {
        region_id: 'r1',
        area_px: 5565,
        area_mm2: null,
        area_fraction_of_mucosa: 0.058,
        centroid: { x: 0.6, y: 0.46 },
        bbox: { x: 0.508, y: 0.354, w: 0.175, h: 0.192 },
        mean_relative_af: 0.653,
        min_relative_af: 0.625,
        contrast_vs_surround: 1.46,
        border_irregularity: 0.31,
        solidity: 0.88,
        confounders: [
          {
            type: 'porphyrin_red',
            overlap_fraction: 0.31,
            note: 'Red porphyrin fluorescence (plaque or bacteria) overlaps this region.',
          },
        ],
      },
    ],
  },
  wl: {
    status: 'model_unavailable',
    model: MODELS[1]!,
    predictions: [],
    top_label: null,
    saliency_artifact: null,
    note: 'White-light classifier not available; not used in triage.',
  },
  spectral_reading_ids: [],
  artifacts: {
    wl_image: 'high/buccal_mucosa_left/wl.jpg',
    af_image: 'high/buccal_mucosa_left/af.jpg',
    af_registered: 'high/buccal_mucosa_left/af_registered.jpg',
    heatmap_overlay: 'high/buccal_mucosa_left/heatmap_overlay.jpg',
    relative_af_map: null,
    relative_af_grid: 'high/buccal_mucosa_left/relative_af_grid.json',
    mucosa_mask: null,
    wl_saliency: null,
  },
  working_resolution: { width_px: 1024, height_px: 768, scale_from_original: 0.4 },
  px_per_mm: null,
  site_risk_level: 'MODERATE',
};

const gingivaWlIssues = [
  issue('blur', 'error', 'laplacian_var', 11.2, 40),
  issue('glare', 'error', 'glare_fraction', 0.083, 0.04),
];
const gingivaAfIssues = [issue('af_signal_low', 'error', 'green_p90', 0.07, 0.12)];

const highGingiva: SiteResult = {
  site: 'lower_gingiva',
  site_label: 'Lower gingiva',
  role: 'examined',
  paired_site: 'upper_gingiva',
  pairing: 'analogous',
  visual_flag: { lesion_seen: false, appearance: [], description: null },
  diascopy: 'not_done',
  captures: [
    capture(
      'lower_gingiva',
      'white_light',
      'cap-lg-wl',
      qc(0.2, gingivaWlIssues, { laplacian_var: 11.2 }),
      'high/lower_gingiva/wl.jpg',
      HIGH_AT,
    ),
    capture(
      'lower_gingiva',
      'autofluorescence',
      'cap-lg-af',
      qc(0.5, gingivaAfIssues, { green_p90: 0.07 }),
      'high/lower_gingiva/af.jpg',
      HIGH_AT,
    ),
    capture(
      'lower_gingiva',
      'dark_frame',
      'cap-lg-dark',
      qc(0.85, [issue('ambient_light_minor', 'warning', 'dark_mean', 0.041, 0.03)]),
      null,
      HIGH_AT,
    ),
  ],
  qc: qc(0.12, [...gingivaWlIssues, ...gingivaAfIssues]),
  analysis_status: 'inconclusive_qc_failed',
  calibration: CALIBRATED,
  registration: null,
  segmentation: null,
  af: null,
  wl: null,
  spectral_reading_ids: [],
  artifacts: {
    wl_image: 'high/lower_gingiva/wl.jpg',
    af_image: 'high/lower_gingiva/af.jpg',
    af_registered: null,
    heatmap_overlay: null,
    relative_af_map: null,
    relative_af_grid: null,
    mucosa_mask: null,
    wl_saliency: null,
  },
  working_resolution: null,
  px_per_mm: null,
  site_risk_level: 'INCONCLUSIVE',
};

const SPECTRAL_CAPTURE = (site: OralSite, id: string) => ({
  capture_id: id,
  kind: 'spectral' as const,
  oral_site: site,
  source_adapter: 'synthetic' as const,
  device_id: 'confirm-sim-01',
  firmware_version: '0.0.0-sim',
  led_mode: 'WHITE' as const,
  exposure_time_ms: 50,
  iso: null,
  white_balance: null,
  focus: null,
  captured_at: HIGH_AT,
  calibration_id: 'spcal-sim-0412',
  width_px: null,
  height_px: null,
  file_name: 'SYNTHETIC_spectrum.csv',
  file_sha256: null,
  mime_type: 'text/csv',
  synthetic: true,
  extra: { gain: 64 },
});

const SPECTRAL_MODEL = MODELS[2]!;

const highSpectral: ScreeningResult['spectral_readings'] = [
  {
    reading_id: 'sp-l1',
    site: 'lateral_tongue_right',
    role: 'lesion',
    lesion_ref: 'r1',
    capture: SPECTRAL_CAPTURE('lateral_tongue_right', 'cap-sp-l1'),
    n_readings: 5,
    signal_type: 'reflectance',
    calibration: {
      status: 'calibrated',
      calibration_id: 'spcal-sim-0412',
      expires_at: '2026-10-13T08:00:00Z',
      note: null,
    },
    values: {
      F1_415: 0.162,
      F2_445: 0.236,
      F3_480: 0.289,
      F4_515: 0.318,
      F5_555: 0.262,
      F6_590: 0.344,
      F7_630: 0.497,
      F8_680: 0.541,
      CLEAR: 0.402,
      NIR: 0.588,
    },
    values_std: {
      F1_415: 0.011,
      F2_445: 0.012,
      F3_480: 0.013,
      F4_515: 0.012,
      F5_555: 0.014,
      F6_590: 0.015,
      F7_630: 0.016,
      F8_680: 0.017,
      CLEAR: 0.012,
      NIR: 0.019,
    },
    qc: qc(0.94),
    features: {
      ratio_555_630: 0.527,
      ratio_590_680: 0.636,
      ratio_515_630: 0.64,
      ratio_415_555: 0.618,
      hb_absorption_proxy: 0.299,
      slope_480_555: -0.131,
      slope_590_680: 0.501,
    },
    reference_reading_id: 'sp-r1',
    delta_vs_reference: {
      ratio_555_630: -0.071,
      ratio_590_680: -0.041,
      ratio_515_630: -0.022,
      ratio_415_555: 0.018,
      hb_absorption_proxy: 0.047,
      slope_480_555: -0.112,
      slope_590_680: 0.067,
    },
    classifier: {
      status: 'placeholder',
      method: 'rules_v1',
      suspicion_score: 0.62,
      label: 'indeterminate',
      reasons: [
        'Haemoglobin dip is 0.047 deeper than the reference (indeterminate band 0.03–0.06).',
        '555/630 nm ratio is 0.071 lower than the reference (indeterminate band −0.05 to −0.10).',
      ],
      model: SPECTRAL_MODEL,
    },
  },
  {
    reading_id: 'sp-r1',
    site: 'lateral_tongue_left',
    role: 'reference',
    lesion_ref: null,
    capture: SPECTRAL_CAPTURE('lateral_tongue_left', 'cap-sp-r1'),
    n_readings: 5,
    signal_type: 'reflectance',
    calibration: {
      status: 'calibrated',
      calibration_id: 'spcal-sim-0412',
      expires_at: '2026-10-13T08:00:00Z',
      note: null,
    },
    values: {
      F1_415: 0.188,
      F2_445: 0.262,
      F3_480: 0.311,
      F4_515: 0.342,
      F5_555: 0.318,
      F6_590: 0.392,
      F7_630: 0.532,
      F8_680: 0.567,
      CLEAR: 0.441,
      NIR: 0.602,
    },
    values_std: {
      F1_415: 0.009,
      F2_445: 0.01,
      F3_480: 0.011,
      F4_515: 0.011,
      F5_555: 0.012,
      F6_590: 0.012,
      F7_630: 0.013,
      F8_680: 0.014,
      CLEAR: 0.01,
      NIR: 0.015,
    },
    qc: qc(0.97),
    features: {
      ratio_555_630: 0.598,
      ratio_590_680: 0.677,
      ratio_515_630: 0.662,
      ratio_415_555: 0.6,
      hb_absorption_proxy: 0.252,
      slope_480_555: -0.019,
      slope_590_680: 0.434,
    },
    reference_reading_id: null,
    delta_vs_reference: null,
    classifier: null,
  },
];

const HIGH_PATIENT_FACTORS: ContributingFactor[] = [
  f(
    'patient.smokeless_tobacco',
    'Uses smokeless tobacco (gutka) daily',
    'clinical_history',
    1.0,
    '20 years, about 6 sachets a day',
  ),
  f('patient.areca_nut', 'Chews areca nut', 'clinical_history', 0.75),
  f('patient.age_40_plus', 'Age 40 or older', 'clinical_history', 0.5),
  f('patient.smoking_former', 'Former bidi smoker (half points)', 'clinical_history', 0.25),
];

const L1_FACTORS: ContributingFactor[] = [
  f(
    'af.fvl_present',
    'Dark area under violet light (glow loss)',
    'autofluorescence',
    1.5,
    'Region r1',
  ),
  f(
    'af.depth',
    'Glow at 46% of the other side — well below the 75% threshold',
    'autofluorescence',
    2.0,
    'Mean relative AF 0.455; 0.75 points per 0.1 below threshold, max 2.0',
  ),
  f('af.area', 'Dark area covers 7% of the visible tissue', 'autofluorescence', 0.5),
  f('af.border', 'Irregular border', 'autofluorescence', 0.5, 'Border irregularity 0.42 (≥ 0.35)'),
  f(
    'af.contrast',
    'Clearly darker than the surrounding tissue',
    'autofluorescence',
    0.5,
    'Contrast 2.14 (≥ 1.4)',
  ),
  f(
    'site.high_risk',
    'Common site for oral lesions in tobacco users (glow points ×1.15)',
    'site',
    0.75,
  ),
  f(
    'spectral.hb_dip',
    'Deeper haemoglobin dip than the other side',
    'spectral',
    0.5,
    'Δ 0.047 (≥ 0.03)',
  ),
  f(
    'spectral.ratio_555_630',
    'Lower 555/630 nm ratio than the other side',
    'spectral',
    0.5,
    'Δ −0.071 (≤ −0.05)',
  ),
  f('visual.white_patch', 'White patch seen by the health worker', 'clinical_visual', 2.0),
];

const L2_FACTORS: ContributingFactor[] = [
  f(
    'af.fvl_present',
    'Dark area under violet light (glow loss)',
    'autofluorescence',
    1.5,
    'Region r1',
  ),
  f(
    'af.depth',
    'Glow at 65% of reference — below the 75% threshold',
    'autofluorescence',
    0.75,
    'Mean relative AF 0.653',
  ),
  f('af.area', 'Dark area covers 6% of the visible tissue', 'autofluorescence', 0.5),
  f(
    'af.contrast',
    'Darker than the surrounding tissue',
    'autofluorescence',
    0.5,
    'Contrast 1.46 (≥ 1.4)',
  ),
  f(
    'af.reference_in_image',
    'Reference from the same photo, not the other cheek (×0.75)',
    'autofluorescence',
    -0.81,
  ),
  f(
    'af.confounder_porphyrin',
    'Red porphyrin glow (plaque) overlaps 31% of the area',
    'autofluorescence',
    -0.45,
    'Glow points ×0.81',
  ),
  f(
    'site.high_risk',
    'Common site for oral lesions in tobacco users (glow points ×1.15)',
    'site',
    0.3,
  ),
  f('visual.other', 'Change seen by the health worker', 'clinical_visual', 1.0),
];

const sum = (fs: ContributingFactor[]) =>
  Math.round(fs.reduce((a, b) => a + b.points, 0) * 100) / 100;

const highStages = stages(
  ['lateral_tongue_right', 'lateral_tongue_left', 'buccal_mucosa_left', 'lower_gingiva'],
  true,
  ['lower_gingiva'],
);

export const highResult: ScreeningResult = {
  schema_version: '1.0.0',
  result_id: 'res_7f3a_v1',
  screening_id: 'scr_demo_high_7f3a',
  result_version: 1,
  supersedes_result_id: null,
  created_at: '2026-10-12T09:41:22Z',
  data_provenance: 'synthetic',
  synthetic: true,
  notices: NOTICES,
  versions: VERSIONS,
  context: {
    screening_site_name: 'Synthetic demo camp',
    camp_id: 'demo',
    health_worker_id: 'hw-demo',
    health_worker_name: null,
    device_ids: ['scout-sim-01', 'confirm-sim-01'],
    firmware_versions: ['0.0.0-sim'],
    captured_at: HIGH_AT,
    language: 'en',
  },
  patient: { pseudonymous_id: 'MS-7F3A-K2', age_years: 58, sex: 'male' },
  consent: {
    consent_recorded: true,
    method: 'checkbox',
    language: 'en',
    research_use_opt_in: false,
    consent_id: 'cons-7f3a',
    recorded_at: HIGH_AT,
  },
  risk_factors: {
    ...NO_RISK_FACTORS,
    smokeless_tobacco: { uses: true, former: false, types: ['gutka'], years: 20, per_day: 6 },
    smoking: { uses: false, former: true, types: ['bidi'], years: 12, per_day: null },
    areca_nut: { uses: true, former: false, years: 25, per_day: 3 },
    alcohol: { uses: true, frequency: 'occasional' },
    symptoms: { ...NO_RISK_FACTORS.symptoms, non_healing_ulcer_over_2_weeks: true },
    irritants: { sharp_tooth: true, ill_fitting_denture: false },
  },
  sites: [highTongueRight, highTongueLeft, highBuccalLeft, highGingiva],
  spectral_readings: highSpectral,
  lesions: [
    {
      lesion_id: 'L1',
      site: 'lateral_tongue_right',
      origins: ['af_fvl', 'visual', 'spectral'],
      region_ids: ['r1'],
      spectral_reading_ids: ['sp-l1'],
      location: { x: 0.429, y: 0.561 },
      risk_level: 'HIGH',
      score: sum(L1_FACTORS),
      contributing_factors: L1_FACTORS,
      confounders: [],
      recommended_action: ACTIONS.HIGH,
      summary:
        'Deep glow loss with an irregular border at a white patch on the right side of the tongue, with a deeper haemoglobin dip than the other side.',
    },
    {
      lesion_id: 'L2',
      site: 'buccal_mucosa_left',
      origins: ['af_fvl', 'visual'],
      region_ids: ['r1'],
      spectral_reading_ids: [],
      location: { x: 0.6, y: 0.46 },
      risk_level: 'MODERATE',
      score: sum(L2_FACTORS),
      contributing_factors: L2_FACTORS,
      confounders: [
        {
          type: 'porphyrin_red',
          overlap_fraction: 0.31,
          note: 'Red porphyrin fluorescence (plaque or bacteria) overlaps this region.',
        },
      ],
      recommended_action: ACTIONS.MODERATE,
      summary:
        'Shallow glow loss on the left cheek, partly explained by plaque; re-examine after cleaning.',
    },
  ],
  triage: {
    overall_risk_level: 'HIGH',
    score: sum([...L1_FACTORS, ...HIGH_PATIENT_FACTORS]),
    thresholds: { moderate: 3.0, high: 6.0 },
    contributing_factors: [...L1_FACTORS, ...HIGH_PATIENT_FACTORS],
    overrides_applied: [
      f(
        'symptom.non_healing_ulcer',
        'Non-healing ulcer for more than 2 weeks',
        'override',
        0,
        'Floor: HIGH',
      ),
      f(
        'visual.white_patch',
        'White patch seen by the health worker',
        'override',
        0,
        'Floor: MODERATE',
      ),
    ],
    recommended_action: ACTIONS.HIGH,
    highest_risk_lesion_id: 'L1',
    rule_config: RULES,
    explanation:
      'Glow loss on the right side of the tongue, a white patch seen by the health worker and a deeper haemoglobin dip than the other side give a score of 11.25, above the HIGH threshold of 6. The patient also reports an ulcer that has not healed for more than 2 weeks, a clinical red flag that sets a HIGH floor on its own.',
    requires_clinician_review: true,
    review_status: 'pending_review',
  },
  qc_summary: {
    passed_all: false,
    sites_failed: ['lower_gingiva'],
    captures_failed: 2,
    retakes_recommended: [
      'Lower gingiva: ' + QC_TEXT.blur,
      'Lower gingiva: ' + QC_TEXT.glare,
      'Lower gingiva: ' + QC_TEXT.af_signal_low,
    ],
    issues: [...gingivaWlIssues, ...gingivaAfIssues],
  },
  calibration_status: 'calibrated',
  limitations: [
    {
      code: 'wl_model_unavailable',
      message:
        'White-light classifier not available; white-light photos were not scored by a model.',
    },
    {
      code: 'site_qc_failed',
      message:
        'Lower gingiva could not be analysed because of image quality. Retake it or examine it clinically.',
    },
    {
      code: 'reference_in_image',
      message:
        'Left buccal mucosa used brighter tissue in the same photo as its reference because the right cheek was not captured.',
    },
    {
      code: 'mm_scale_unknown',
      message:
        'No millimetre scale was available, so areas are reported as a share of visible tissue.',
    },
    {
      code: 'rules_unvalidated',
      message: 'Triage rules are a baseline heuristic and have not been clinically validated.',
    },
    { code: 'synthetic_inputs', message: 'All inputs are synthetic.' },
  ],
  processing: { working_max_side_px: 1024, duration_ms: total(highStages), stages: highStages },
  review: null,
};

// ---------------------------------------------------------------------------------------------
// LOW
// ---------------------------------------------------------------------------------------------

const LOW_AT = '2026-10-11T15:12:40Z';
const lowStages = stages(['buccal_mucosa_right', 'buccal_mucosa_left'], false);

export const lowResult: ScreeningResult = {
  ...highResult,
  result_id: 'res_2c91_v1',
  screening_id: 'scr_demo_low_2c91',
  created_at: '2026-10-11T15:14:02Z',
  context: { ...highResult.context, captured_at: LOW_AT, device_ids: ['scout-sim-01'] },
  patient: { pseudonymous_id: 'MS-2C91-A7', age_years: 34, sex: 'female' },
  consent: {
    ...highResult.consent,
    consent_id: 'cons-2c91',
    recorded_at: LOW_AT,
    research_use_opt_in: true,
  },
  risk_factors: {
    ...NO_RISK_FACTORS,
    areca_nut: { uses: false, former: true, years: 3, per_day: null },
  },
  sites: [
    {
      site: 'buccal_mucosa_right',
      site_label: 'Right buccal mucosa',
      role: 'examined',
      paired_site: 'buccal_mucosa_left',
      pairing: 'contralateral',
      visual_flag: { lesion_seen: false, appearance: [], description: null },
      diascopy: 'not_done',
      captures: [
        capture(
          'buccal_mucosa_right',
          'white_light',
          'cap-low-bmr-wl',
          qc(0.97),
          'low/buccal_mucosa_right/wl.jpg',
          LOW_AT,
        ),
        capture(
          'buccal_mucosa_right',
          'autofluorescence',
          'cap-low-bmr-af',
          qc(0.96),
          'low/buccal_mucosa_right/af.jpg',
          LOW_AT,
        ),
        capture('buccal_mucosa_right', 'dark_frame', 'cap-low-bmr-dark', qc(1), null, LOW_AT),
      ],
      qc: qc(0.96),
      analysis_status: 'analysed',
      calibration: CALIBRATED,
      registration: {
        method: 'ecc_affine',
        confidence: 0.93,
        passed: true,
        warp_matrix: IDENTITY_WARP,
        note: null,
      },
      segmentation: {
        method: 'classical_hsv_lab',
        version: '1.0.0',
        status: 'ok',
        mucosa_fraction: 0.81,
        teeth_fraction: 0,
        specular_fraction: 0.007,
        confidence: 0.87,
      },
      af: {
        status: 'ok',
        reference: {
          type: 'contralateral',
          reference_site: 'buccal_mucosa_left',
          reference_green: 0.402,
          note: null,
        },
        fvl_threshold: 0.75,
        min_region_area_px: 150,
        site_medians: { green: 0.399, gr_ratio: 4.4, af_index: 0.396, relative_af: 0.99 },
        fvl_area_fraction: 0,
        porphyrin_red_fraction: 0.003,
        regions: [],
      },
      wl: {
        status: 'model_unavailable',
        model: MODELS[1]!,
        predictions: [],
        top_label: null,
        saliency_artifact: null,
        note: 'White-light classifier not available; not used in triage.',
      },
      spectral_reading_ids: [],
      artifacts: {
        wl_image: 'low/buccal_mucosa_right/wl.jpg',
        af_image: 'low/buccal_mucosa_right/af.jpg',
        af_registered: 'low/buccal_mucosa_right/af_registered.jpg',
        heatmap_overlay: 'low/buccal_mucosa_right/heatmap_overlay.jpg',
        relative_af_map: null,
        relative_af_grid: 'low/buccal_mucosa_right/relative_af_grid.json',
        mucosa_mask: null,
        wl_saliency: null,
      },
      working_resolution: { width_px: 1024, height_px: 768, scale_from_original: 0.4 },
      px_per_mm: null,
      site_risk_level: 'LOW',
    },
    {
      site: 'buccal_mucosa_left',
      site_label: 'Left buccal mucosa',
      role: 'reference_only',
      paired_site: 'buccal_mucosa_right',
      pairing: 'contralateral',
      visual_flag: { lesion_seen: false, appearance: [], description: null },
      diascopy: 'not_done',
      captures: [
        capture(
          'buccal_mucosa_left',
          'white_light',
          'cap-low-bml-wl',
          qc(0.96),
          'low/buccal_mucosa_left/wl.jpg',
          LOW_AT,
        ),
        capture(
          'buccal_mucosa_left',
          'autofluorescence',
          'cap-low-bml-af',
          qc(0.95),
          'low/buccal_mucosa_left/af.jpg',
          LOW_AT,
        ),
      ],
      qc: qc(0.95, [issue('no_dark_frame', 'warning')]),
      analysis_status: 'analysed',
      calibration: CALIBRATED,
      registration: {
        method: 'ecc_affine',
        confidence: 0.92,
        passed: true,
        warp_matrix: IDENTITY_WARP,
        note: null,
      },
      segmentation: null,
      af: null,
      wl: null,
      spectral_reading_ids: [],
      artifacts: {
        wl_image: 'low/buccal_mucosa_left/wl.jpg',
        af_image: 'low/buccal_mucosa_left/af.jpg',
        af_registered: null,
        heatmap_overlay: null,
        relative_af_map: null,
        relative_af_grid: null,
        mucosa_mask: null,
        wl_saliency: null,
      },
      working_resolution: { width_px: 1024, height_px: 768, scale_from_original: 0.4 },
      px_per_mm: null,
      site_risk_level: 'LOW',
    },
  ],
  spectral_readings: [],
  lesions: [],
  triage: {
    overall_risk_level: 'LOW',
    score: 0,
    thresholds: { moderate: 3.0, high: 6.0 },
    contributing_factors: [],
    overrides_applied: [],
    recommended_action: ACTIONS.LOW,
    highest_risk_lesion_id: null,
    rule_config: RULES,
    explanation:
      'Both cheeks glow evenly under violet light and the health worker saw no lesion, so no points were added. This screen does not rule out disease; re-screen routinely and sooner if anything changes.',
    requires_clinician_review: true,
    review_status: 'pending_review',
  },
  qc_summary: {
    passed_all: true,
    sites_failed: [],
    captures_failed: 0,
    retakes_recommended: [],
    issues: [],
  },
  calibration_status: 'calibrated',
  limitations: [
    {
      code: 'wl_model_unavailable',
      message:
        'White-light classifier not available; white-light photos were not scored by a model.',
    },
    {
      code: 'single_site',
      message: 'Only the right cheek was examined. Other sites were not screened.',
    },
    {
      code: 'rules_unvalidated',
      message: 'Triage rules are a baseline heuristic and have not been clinically validated.',
    },
    { code: 'synthetic_inputs', message: 'All inputs are synthetic.' },
  ],
  processing: { working_max_side_px: 1024, duration_ms: total(lowStages), stages: lowStages },
};

// ---------------------------------------------------------------------------------------------
// INCONCLUSIVE
// ---------------------------------------------------------------------------------------------

const INC_AT = '2026-10-10T11:02:10Z';
const incStages = stages(['buccal_mucosa_left', 'buccal_mucosa_right'], false, [
  'buccal_mucosa_left',
]);
const incWlIssues = [issue('blur', 'error', 'laplacian_var', 14.8, 40)];
const incAfIssues = [
  issue('af_signal_low', 'error', 'green_p90', 0.08, 0.12),
  issue('af_not_green', 'warning'),
];

export const inconclusiveResult: ScreeningResult = {
  ...lowResult,
  result_id: 'res_b4e0_v1',
  screening_id: 'scr_demo_inconclusive_b4e0',
  created_at: '2026-10-10T11:03:31Z',
  context: { ...lowResult.context, captured_at: INC_AT },
  patient: { pseudonymous_id: 'MS-B4E0-Q5', age_years: 45, sex: 'female' },
  consent: {
    ...lowResult.consent,
    consent_id: 'cons-b4e0',
    recorded_at: INC_AT,
    research_use_opt_in: false,
  },
  risk_factors: {
    ...NO_RISK_FACTORS,
    smokeless_tobacco: { uses: true, former: false, types: ['khaini'], years: 8, per_day: 4 },
  },
  sites: [
    {
      ...highGingiva,
      site: 'buccal_mucosa_left',
      site_label: 'Left buccal mucosa',
      paired_site: 'buccal_mucosa_right',
      pairing: 'contralateral',
      captures: [
        capture(
          'buccal_mucosa_left',
          'white_light',
          'cap-inc-bml-wl',
          qc(0.3, incWlIssues),
          'high/lower_gingiva/wl.jpg',
          INC_AT,
        ),
        capture(
          'buccal_mucosa_left',
          'autofluorescence',
          'cap-inc-bml-af',
          qc(0.4, incAfIssues),
          'high/lower_gingiva/af.jpg',
          INC_AT,
        ),
      ],
      qc: qc(0.2, [...incWlIssues, ...incAfIssues]),
      site_risk_level: 'INCONCLUSIVE',
    },
    {
      ...lowResult.sites[0]!,
      af: {
        ...lowResult.sites[0]!.af!,
        reference: {
          type: 'in_image_auto',
          reference_site: null,
          reference_green: 0.4,
          note: 'The paired site failed quality checks.',
        },
      },
    },
  ],
  triage: {
    ...lowResult.triage,
    overall_risk_level: 'INCONCLUSIVE',
    score: 0,
    recommended_action: ACTIONS.INCONCLUSIVE,
    explanation:
      'The left cheek photos failed quality checks, so that site could not be analysed. A low risk level is only given when every examined site was analysed, so this screening is inconclusive. Retake the photos now; if a lesion is visible or a retake is not possible, refer for clinical examination.',
  },
  qc_summary: {
    passed_all: false,
    sites_failed: ['buccal_mucosa_left'],
    captures_failed: 2,
    retakes_recommended: [
      'Left buccal mucosa: ' + QC_TEXT.blur,
      'Left buccal mucosa: ' + QC_TEXT.af_signal_low,
    ],
    issues: [...incWlIssues, ...incAfIssues],
  },
  limitations: [
    {
      code: 'site_qc_failed',
      message: 'Left buccal mucosa could not be analysed because of image quality.',
    },
    {
      code: 'wl_model_unavailable',
      message:
        'White-light classifier not available; white-light photos were not scored by a model.',
    },
    {
      code: 'rules_unvalidated',
      message: 'Triage rules are a baseline heuristic and have not been clinically validated.',
    },
    { code: 'synthetic_inputs', message: 'All inputs are synthetic.' },
  ],
  processing: { working_max_side_px: 1024, duration_ms: total(incStages), stages: incStages },
};

// ---------------------------------------------------------------------------------------------
// MODERATE (derived)
// ---------------------------------------------------------------------------------------------

const MOD_PATIENT_FACTORS: ContributingFactor[] = [
  f('patient.smokeless_tobacco', 'Uses smokeless tobacco (khaini) daily', 'clinical_history', 1.0),
  f('patient.age_40_plus', 'Age 40 or older', 'clinical_history', 0.5),
];
const modStages = stages(['buccal_mucosa_left'], false);

export const moderateResult: ScreeningResult = {
  ...highResult,
  result_id: 'res_5d18_v1',
  screening_id: 'scr_demo_moderate_5d18',
  created_at: '2026-10-12T12:20:45Z',
  patient: { pseudonymous_id: 'MS-5D18-H3', age_years: 47, sex: 'female' },
  risk_factors: {
    ...NO_RISK_FACTORS,
    smokeless_tobacco: { uses: true, former: false, types: ['khaini'], years: 15, per_day: 5 },
  },
  sites: [highBuccalLeft],
  spectral_readings: [],
  lesions: [highResult.lesions![1]!],
  triage: {
    ...highResult.triage,
    overall_risk_level: 'MODERATE',
    score: sum([...L2_FACTORS, ...MOD_PATIENT_FACTORS]),
    contributing_factors: [...L2_FACTORS, ...MOD_PATIENT_FACTORS],
    overrides_applied: [],
    recommended_action: ACTIONS.MODERATE,
    highest_risk_lesion_id: 'L2',
    explanation:
      'A shallow area of glow loss on the left cheek, a change seen by the health worker and daily khaini use give a score of 4.79, between the MODERATE (3) and HIGH (6) thresholds. Red porphyrin glow from plaque explains part of the darkness, so its points were reduced.',
  },
  qc_summary: {
    passed_all: true,
    sites_failed: [],
    captures_failed: 0,
    retakes_recommended: [],
    issues: [],
  },
  limitations: [
    {
      code: 'wl_model_unavailable',
      message:
        'White-light classifier not available; white-light photos were not scored by a model.',
    },
    {
      code: 'reference_in_image',
      message:
        'Left buccal mucosa used brighter tissue in the same photo as its reference because the right cheek was not captured.',
    },
    {
      code: 'rules_unvalidated',
      message: 'Triage rules are a baseline heuristic and have not been clinically validated.',
    },
    { code: 'synthetic_inputs', message: 'All inputs are synthetic.' },
  ],
  processing: { working_max_side_px: 1024, duration_ms: total(modStages), stages: modStages },
};

export const FIXTURES = {
  high: highResult,
  moderate: moderateResult,
  low: lowResult,
  inconclusive: inconclusiveResult,
} as const;
