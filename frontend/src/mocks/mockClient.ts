/**
 * In-memory mock of the MukhaSpectra API (VITE_MOCK=1). Seeded with SYNTHETIC fixtures; demo
 * runs, uploads and re-analyses add screenings / versions for the life of the page. Results are
 * canned: an upload in mock mode never analyses the photos you choose.
 */
import { ApiError } from '../api/client';
import { normalizeScreeningList } from '../api/normalize';
import type {
  ApiClient,
  DemoKind,
  ResultVersionSummary,
  ScreeningDetail,
  ScreeningResult,
  UploadRequest,
  VersionResponse,
} from '../api/types';
import { BUILTIN_SITE_CATALOGUE } from '../lib/sites';
import { MOCK_ASSETS, MOCK_REPORT_URL } from './assets';
import { FIXTURES } from './fixtures';

export interface MockClientOptions {
  /** Delay for analysis calls (upload, demo, re-analyse). */
  analysisLatencyMs?: number;
  /** Delay for simple reads. */
  readLatencyMs?: number;
  now?: () => Date;
}

interface Entry {
  created_at: string;
  versions: ScreeningResult[];
}

const clone = <T>(v: T): T => JSON.parse(JSON.stringify(v)) as T;
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

function randomId(n = 4): string {
  const alphabet = '0123456789abcdef';
  let s = '';
  for (let i = 0; i < n; i++) s += alphabet[Math.floor(Math.random() * 16)];
  return s;
}

export function createMockClient(options: MockClientOptions = {}): ApiClient {
  const analysisLatency = options.analysisLatencyMs ?? 1500;
  const readLatency = options.readLatencyMs ?? 180;
  const now = options.now ?? (() => new Date());
  const db = new Map<string, Entry>();

  for (const r of [FIXTURES.high, FIXTURES.low, FIXTURES.inconclusive]) {
    db.set(r.screening_id, { created_at: r.created_at, versions: [clone(r)] });
  }

  const summary = (r: ScreeningResult): ResultVersionSummary => ({
    result_version: r.result_version,
    result_id: r.result_id,
    created_at: r.created_at,
    overall_risk_level: r.triage.overall_risk_level,
  });

  const detail = (id: string): ScreeningDetail => {
    const e = db.get(id);
    if (!e) throw new ApiError('http', 'HTTP 404', 404, 'Screening not found');
    const latest = e.versions[e.versions.length - 1]!;
    return {
      screening_id: id,
      created_at: e.created_at,
      latest: clone(latest),
      versions: e.versions.map(summary),
    };
  };

  const createFrom = (template: ScreeningResult, tweak?: (r: ScreeningResult) => void) => {
    const id = `${template.screening_id.replace(/_[0-9a-f]{4}$/, '')}_${randomId()}`;
    const r = clone(template);
    const ts = now().toISOString();
    r.screening_id = id;
    r.result_id = `res_${randomId(6)}_v1`;
    r.created_at = ts;
    r.result_version = 1;
    r.supersedes_result_id = null;
    r.patient.pseudonymous_id = `MS-${randomId(4).toUpperCase()}-${randomId(2).toUpperCase()}`;
    tweak?.(r);
    db.set(id, { created_at: ts, versions: [r] });
    return detail(id);
  };

  return {
    mode: 'mock',
    async getSites() {
      await sleep(readLatency / 2);
      return BUILTIN_SITE_CATALOGUE;
    },
    async getVersion(): Promise<VersionResponse> {
      await sleep(readLatency / 2);
      const v = FIXTURES.high.versions;
      return {
        software: v.software,
        pipeline: v.pipeline,
        schema_version: v.schema_version,
        models: v.models,
      };
    },
    async listScreenings() {
      await sleep(readLatency);
      return normalizeScreeningList(
        [...db.entries()].map(([id, e]) => {
          const latest = e.versions[e.versions.length - 1]!;
          return { screening_id: id, created_at: e.created_at, latest };
        }),
      );
    },
    async getScreening(id) {
      await sleep(readLatency);
      return detail(id);
    },
    async getResultVersion(id, version) {
      await sleep(readLatency);
      const e = db.get(id);
      const r = e?.versions.find((v) => v.result_version === version);
      if (!r) throw new ApiError('http', 'HTTP 404', 404, 'Result version not found');
      return clone(r);
    },
    async createScreening(req: UploadRequest) {
      await sleep(analysisLatency);
      const m = req.manifest;
      if (!m.consent.consent_recorded) {
        throw new ApiError('http', 'HTTP 422', 422, 'Consent must be recorded.');
      }
      return createFrom(FIXTURES.high, (r) => {
        r.patient.age_years = m.patient.age_years;
        r.patient.sex = m.patient.sex;
        r.consent.research_use_opt_in = m.consent.research_use_opt_in;
        r.risk_factors = { ...r.risk_factors, ...m.risk_factors };
        if (m.context.screening_site_name)
          r.context.screening_site_name = m.context.screening_site_name;
      });
    },
    async runDemo(kind: DemoKind) {
      await sleep(analysisLatency);
      return createFrom(FIXTURES[kind]);
    },
    async reanalyze(id) {
      await sleep(analysisLatency);
      const e = db.get(id);
      if (!e) throw new ApiError('http', 'HTTP 404', 404, 'Screening not found');
      const prev = e.versions[e.versions.length - 1]!;
      const next = clone(prev);
      next.result_version = prev.result_version + 1;
      next.result_id = `${prev.result_id.replace(/_v\d+$/, '')}_v${next.result_version}`;
      next.supersedes_result_id = prev.result_id;
      next.created_at = now().toISOString();
      next.processing = {
        ...next.processing,
        stages: (next.processing.stages ?? []).map((s) => ({
          ...s,
          duration_ms: Math.round(s.duration_ms * (0.85 + Math.random() * 0.3) * 10) / 10,
        })),
      };
      next.processing.duration_ms =
        Math.round((next.processing.stages ?? []).reduce((a, s) => a + s.duration_ms, 0) * 10) / 10;
      e.versions.push(next);
      return detail(id);
    },
    async getArtifactJson<T>(_id: string, key: string): Promise<T> {
      const url = MOCK_ASSETS[key];
      if (!url) throw new ApiError('http', 'HTTP 404', 404, `Artifact ${key} not found`);
      const res = await fetch(url);
      return (await res.json()) as T;
    },
    artifactUrl: (_id, key) => MOCK_ASSETS[key] ?? '',
    reportUrl: () => MOCK_REPORT_URL,
    sitemapUrl: () => '',
  };
}
