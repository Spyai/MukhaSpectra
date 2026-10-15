/**
 * API client. `createHttpClient` talks to the FastAPI backend (/api/v1, docs/architecture.md §6);
 * `api` switches to the in-memory mock (src/mocks/) when the app is built or served with
 * VITE_MOCK=1. The mock module is loaded with a dynamic import, so it never ships in a live build.
 */
import { loadMockClient } from '@mock-loader';
import {
  isScreeningResult,
  normalizeScreeningDetail,
  normalizeScreeningList,
  normalizeSiteCatalogue,
} from './normalize';
import type {
  ApiClient,
  DemoKind,
  ScreeningDetail,
  ScreeningResult,
  UploadRequest,
  VersionResponse,
} from './types';

export const API_PREFIX = '/api/v1';

export type ApiErrorKind = 'network' | 'http' | 'bad-response';

export class ApiError extends Error {
  readonly kind: ApiErrorKind;
  readonly status: number | undefined;
  readonly detail: string | undefined;

  constructor(kind: ApiErrorKind, message: string, status?: number, detail?: string) {
    super(message);
    this.name = 'ApiError';
    this.kind = kind;
    this.status = status;
    this.detail = detail;
  }
}

/** Encode an artifact key segment by segment so keys with '/' keep working as path params. */
export function encodeKey(key: string): string {
  return key
    .split('/')
    .filter((s) => s.length > 0)
    .map(encodeURIComponent)
    .join('/');
}

function withVersion(path: string, version?: number): string {
  return version === undefined ? path : `${path}?version=${encodeURIComponent(String(version))}`;
}

/** Build the multipart body: a `manifest` JSON string plus one file field per manifest name. */
export function buildUploadFormData(req: UploadRequest): FormData {
  const form = new FormData();
  form.append('manifest', JSON.stringify(req.manifest));
  const referenced = new Set<string>();
  for (const s of req.manifest.sites) {
    referenced.add(s.wl);
    referenced.add(s.af);
    if (s.dark) referenced.add(s.dark);
  }
  if (req.manifest.spectral_file) referenced.add(req.manifest.spectral_file);
  for (const field of referenced) {
    const file = req.files[field];
    if (!file) throw new ApiError('bad-response', `missing file for manifest field "${field}"`);
    const name = (file as File).name || field;
    form.append(field, file, name);
  }
  return form;
}

function extractDetail(body: unknown): string | undefined {
  if (!body || typeof body !== 'object') return typeof body === 'string' ? body : undefined;
  const detail = (body as { detail?: unknown }).detail;
  if (typeof detail === 'string') return detail;
  if (Array.isArray(detail)) {
    return detail
      .map((d) =>
        d && typeof d === 'object' && 'msg' in d ? String((d as { msg: unknown }).msg) : String(d),
      )
      .join('; ');
  }
  return undefined;
}

export interface HttpClientOptions {
  baseUrl?: string;
  fetch?: typeof fetch;
}

export function createHttpClient(options: HttpClientOptions = {}): ApiClient {
  const base = (options.baseUrl ?? API_PREFIX).replace(/\/$/, '');
  const doFetch: typeof fetch = options.fetch ?? ((...args) => globalThis.fetch(...args));
  const url = (path: string) => `${base}${path}`;

  async function request<T>(path: string, init?: RequestInit): Promise<T> {
    let res: Response;
    try {
      res = await doFetch(url(path), {
        ...init,
        headers: { Accept: 'application/json', ...(init?.headers ?? {}) },
      });
    } catch (err) {
      throw new ApiError('network', err instanceof Error ? err.message : 'network error');
    }
    const text = await res.text();
    let body: unknown = undefined;
    if (text) {
      try {
        body = JSON.parse(text);
      } catch {
        body = text;
      }
    }
    if (!res.ok) {
      // A dev/preview proxy answers 500 with an empty body (or 502-504 with an HTML/text page)
      // when the API is down. The API itself answers 503 with a JSON detail when it is busy:
      // that is an HTTP error with a message to show, not a network failure.
      const apiDetail = body && typeof body === 'object' ? extractDetail(body) : undefined;
      if (!apiDetail && (res.status >= 502 || (res.status === 500 && !text))) {
        throw new ApiError('network', `HTTP ${res.status}`, res.status);
      }
      throw new ApiError('http', `HTTP ${res.status}`, res.status, extractDetail(body));
    }
    return body as T;
  }

  const detail = async (p: Promise<unknown>): Promise<ScreeningDetail> => {
    const raw = await p;
    try {
      return normalizeScreeningDetail(raw);
    } catch {
      // Some endpoints may answer with only {screening_id}; follow up with a GET.
      const id =
        raw && typeof raw === 'object' && 'screening_id' in raw
          ? String((raw as { screening_id: unknown }).screening_id)
          : undefined;
      if (id)
        return normalizeScreeningDetail(await request(`/screenings/${encodeURIComponent(id)}`));
      throw new ApiError('bad-response', 'unexpected response shape');
    }
  };

  return {
    mode: 'live',
    getSites: async () => normalizeSiteCatalogue(await request('/sites')),
    getVersion: () => request<VersionResponse>('/version'),
    listScreenings: async () => normalizeScreeningList(await request('/screenings')),
    getScreening: (id) => detail(request(`/screenings/${encodeURIComponent(id)}`)),
    getResultVersion: async (id, version) => {
      const raw = await request<unknown>(
        `/screenings/${encodeURIComponent(id)}/results/${encodeURIComponent(String(version))}`,
      );
      if (isScreeningResult(raw)) return raw;
      if (
        raw &&
        typeof raw === 'object' &&
        isScreeningResult((raw as { result?: unknown }).result)
      ) {
        return (raw as { result: ScreeningResult }).result;
      }
      throw new ApiError('bad-response', 'unexpected result shape');
    },
    createScreening: (req: UploadRequest) =>
      detail(request('/screenings', { method: 'POST', body: buildUploadFormData(req) })),
    runDemo: (kind: DemoKind) =>
      detail(request(`/demo/screenings/${encodeURIComponent(kind)}`, { method: 'POST' })),
    reanalyze: (id) =>
      detail(request(`/screenings/${encodeURIComponent(id)}/reanalyze`, { method: 'POST' })),
    getArtifactJson: <T>(id: string, key: string, version?: number) =>
      request<T>(
        withVersion(`/screenings/${encodeURIComponent(id)}/artifacts/${encodeKey(key)}`, version),
      ),
    artifactUrl: (id, key, version) =>
      url(
        withVersion(`/screenings/${encodeURIComponent(id)}/artifacts/${encodeKey(key)}`, version),
      ),
    reportUrl: (id, version) =>
      url(withVersion(`/screenings/${encodeURIComponent(id)}/report.pdf`, version)),
    sitemapUrl: (id) => url(`/screenings/${encodeURIComponent(id)}/sitemap.svg`),
  };
}

/**
 * Lazily loads the mock client. Synchronous URL builders delegate to the mock once it has
 * loaded — which is always the case by the time a result (and therefore an artifact key) exists.
 */
export function createLazyMockClient(): ApiClient {
  let loaded: ApiClient | null = null;
  let pending: Promise<ApiClient> | null = null;
  const get = () =>
    (pending ??= loadMockClient().then((c) => {
      loaded = c;
      return c;
    }));
  return {
    mode: 'mock',
    getSites: () => get().then((c) => c.getSites()),
    getVersion: () => get().then((c) => c.getVersion()),
    listScreenings: () => get().then((c) => c.listScreenings()),
    getScreening: (id) => get().then((c) => c.getScreening(id)),
    getResultVersion: (id, v) => get().then((c) => c.getResultVersion(id, v)),
    createScreening: (req) => get().then((c) => c.createScreening(req)),
    runDemo: (kind) => get().then((c) => c.runDemo(kind)),
    reanalyze: (id) => get().then((c) => c.reanalyze(id)),
    getArtifactJson: <T>(id: string, key: string, version?: number) =>
      get().then((c) => c.getArtifactJson<T>(id, key, version)),
    artifactUrl: (id, key, version) => loaded?.artifactUrl(id, key, version) ?? '',
    reportUrl: (id, version) => loaded?.reportUrl(id, version) ?? '',
    sitemapUrl: (id) => loaded?.sitemapUrl(id) ?? '',
  };
}

export const IS_MOCK: boolean = typeof __MUKHA_MOCK__ !== 'undefined' && __MUKHA_MOCK__;

export function createDefaultClient(): ApiClient {
  if (__MUKHA_MOCK__) return createLazyMockClient();
  return createHttpClient();
}
