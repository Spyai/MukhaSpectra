/**
 * API-level types. `ScreeningResult` and everything under it come from the generated
 * `types.gen.ts` (JSON Schema → TS). The envelopes below (site catalogue, screening detail,
 * summaries, manifest) are defined by docs/architecture.md §6 and normalised defensively in
 * `normalize.ts`, because the backend is developed in parallel.
 */
import type {
  DataProvenance,
  Diascopy,
  ModelInfo,
  OralSite,
  RiskFactors,
  RiskLevel,
  ScreeningResult,
  Sex,
  VisualAppearance,
} from './types.gen';

export type * from './types.gen';

export type DemoKind = 'low' | 'moderate' | 'high';

export interface SiteShape {
  cx: number;
  cy: number;
  rx: number;
  ry: number;
}

export interface SiteCatalogueEntry {
  site: OralSite;
  label: string;
  short_label: string;
  paired_site: OralSite | null;
  pairing: 'contralateral' | 'analogous' | null;
  keratinized: boolean;
  shape: SiteShape;
}

export interface SiteCatalogue {
  viewBox: [number, number, number, number];
  sites: SiteCatalogueEntry[];
}

export interface VersionResponse {
  software?: string;
  pipeline?: string;
  schema_version?: string;
  models?: ModelInfo[];
  [key: string]: unknown;
}

export interface ResultVersionSummary {
  result_version: number;
  result_id?: string;
  created_at?: string;
  overall_risk_level?: RiskLevel;
}

export interface ScreeningDetail {
  screening_id: string;
  created_at: string;
  latest: ScreeningResult;
  versions: ResultVersionSummary[];
}

export interface ScreeningSummary {
  screening_id: string;
  created_at: string;
  overall_risk_level?: RiskLevel;
  synthetic?: boolean;
  pseudonymous_id?: string;
  age_years?: number | null;
  sex?: Sex;
  latest_version?: number;
  sites?: OralSite[];
  site_count?: number;
}

/** Relative-AF grid artifact (`relative_af_grid`, JSON). `null` = outside tissue. */
export interface RelativeAfGrid {
  width: number;
  height: number;
  values: Array<Array<number | null>>;
  min?: number;
  max?: number;
  threshold?: number;
}

export interface ManifestSite {
  site: OralSite;
  role?: 'examined' | 'reference_only';
  wl: string;
  af: string;
  dark?: string;
  visual_flag?: {
    lesion_seen: boolean;
    appearance: VisualAppearance[];
    description?: string;
  };
  diascopy?: Diascopy;
  px_per_mm?: number | null;
  device_id?: string | null;
}

export interface UploadManifest {
  patient: { age_years: number | null; sex: Sex; external_id?: string };
  consent: {
    consent_recorded: boolean;
    method: 'checkbox';
    language: 'en' | 'hi' | 'kn';
    research_use_opt_in: boolean;
  };
  risk_factors: RiskFactors;
  context: { screening_site_name?: string; health_worker_id?: string };
  data_provenance: DataProvenance;
  sites: ManifestSite[];
  spectral_file?: string;
}

export interface UploadRequest {
  manifest: UploadManifest;
  /** Multipart file fields, keyed by the names used in the manifest. */
  files: Record<string, Blob & { name?: string }>;
}

export interface ApiClient {
  readonly mode: 'live' | 'mock';
  getSites(): Promise<SiteCatalogue>;
  getVersion(): Promise<VersionResponse>;
  listScreenings(): Promise<ScreeningSummary[]>;
  getScreening(id: string): Promise<ScreeningDetail>;
  getResultVersion(id: string, version: number): Promise<ScreeningResult>;
  createScreening(req: UploadRequest): Promise<ScreeningDetail>;
  runDemo(kind: DemoKind): Promise<ScreeningDetail>;
  reanalyze(id: string): Promise<ScreeningDetail>;
  getArtifactJson<T = unknown>(id: string, key: string, version?: number): Promise<T>;
  artifactUrl(id: string, key: string, version?: number): string;
  reportUrl(id: string, version?: number): string;
  sitemapUrl(id: string): string;
}
