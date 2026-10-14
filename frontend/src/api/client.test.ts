import { describe, expect, it, vi } from 'vitest';
import { highResult } from '../mocks/fixtures';
import { createMockClient } from '../mocks/mockClient';
import { ApiError, buildUploadFormData, createHttpClient, encodeKey } from './client';
import { normalizeScreeningDetail, normalizeSiteCatalogue } from './normalize';
import type { UploadRequest } from './types';

function jsonResponse(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  });
}

const detail = {
  screening_id: highResult.screening_id,
  created_at: highResult.created_at,
  latest: highResult,
  versions: [{ result_version: 1 }],
};

function makeRequest(): UploadRequest {
  const file = (name: string) => new File(['x'], name, { type: 'image/jpeg' });
  return {
    manifest: {
      patient: { age_years: 52, sex: 'male' },
      consent: {
        consent_recorded: true,
        method: 'checkbox',
        language: 'en',
        research_use_opt_in: false,
      },
      risk_factors: highResult.risk_factors,
      context: {},
      data_provenance: 'real',
      sites: [
        { site: 'lateral_tongue_right', role: 'examined', wl: 'wl', af: 'af', dark: 'dark' },
        { site: 'lateral_tongue_left', role: 'reference_only', wl: 'ref_wl', af: 'ref_af' },
      ],
      spectral_file: 'spectrum',
    },
    files: {
      wl: file('a_wl.jpg'),
      af: file('a_af.jpg'),
      dark: file('a_dark.jpg'),
      ref_wl: file('b_wl.jpg'),
      ref_af: file('b_af.jpg'),
      spectrum: new File(['reading_id'], 'spectrum.csv', { type: 'text/csv' }),
    },
  };
}

describe('http client URL building', () => {
  const api = createHttpClient({ fetch: vi.fn() });

  it('builds artifact URLs with the version query and per-segment encoding', () => {
    expect(api.artifactUrl('scr 1', 'sites/lateral_tongue_right/wl image.png', 2)).toBe(
      '/api/v1/screenings/scr%201/artifacts/sites/lateral_tongue_right/wl%20image.png?version=2',
    );
    expect(api.artifactUrl('s1', 'wl.png')).toBe('/api/v1/screenings/s1/artifacts/wl.png');
    expect(encodeKey('/a//b c/')).toBe('a/b%20c');
  });

  it('builds report and site-map URLs', () => {
    expect(api.reportUrl('s1', 3)).toBe('/api/v1/screenings/s1/report.pdf?version=3');
    expect(api.reportUrl('s1')).toBe('/api/v1/screenings/s1/report.pdf');
    expect(api.sitemapUrl('s1')).toBe('/api/v1/screenings/s1/sitemap.svg');
  });

  it('honours a custom base URL', () => {
    const remote = createHttpClient({ baseUrl: 'https://api.example.org/api/v1/', fetch: vi.fn() });
    expect(remote.reportUrl('x')).toBe('https://api.example.org/api/v1/screenings/x/report.pdf');
  });
});

describe('http client requests', () => {
  it('GETs a screening and normalises the envelope', async () => {
    const fetch = vi.fn().mockResolvedValue(jsonResponse(detail));
    const api = createHttpClient({ fetch });
    const d = await api.getScreening('scr/1');
    expect(fetch).toHaveBeenCalledWith(
      '/api/v1/screenings/scr%2F1',
      expect.objectContaining({ headers: expect.any(Object) }),
    );
    expect(d.latest.result_id).toBe(highResult.result_id);
    expect(d.versions.map((v) => v.result_version)).toEqual([1]);
  });

  it('POSTs demo runs and re-analysis to the documented paths', async () => {
    const fetch = vi.fn().mockImplementation(() => Promise.resolve(jsonResponse(detail)));
    const api = createHttpClient({ fetch });
    await api.runDemo('high');
    await api.reanalyze('s1');
    expect(fetch.mock.calls[0]![0]).toBe('/api/v1/demo/screenings/high');
    expect(fetch.mock.calls[0]![1].method).toBe('POST');
    expect(fetch.mock.calls[1]![0]).toBe('/api/v1/screenings/s1/reanalyze');
    expect(fetch.mock.calls[1]![1].method).toBe('POST');
  });

  it('sends the upload as multipart: a manifest JSON string plus one field per manifest name', async () => {
    const fetch = vi.fn().mockResolvedValue(jsonResponse(highResult));
    const api = createHttpClient({ fetch });
    const req = makeRequest();
    const d = await api.createScreening(req);
    expect(d.screening_id).toBe(highResult.screening_id);
    const [url, init] = fetch.mock.calls[0]!;
    expect(url).toBe('/api/v1/screenings');
    expect(init.method).toBe('POST');
    const body = init.body as FormData;
    expect(body).toBeInstanceOf(FormData);
    expect(typeof body.get('manifest')).toBe('string');
    const manifest = JSON.parse(body.get('manifest') as string);
    expect(manifest.sites[0]).toMatchObject({
      site: 'lateral_tongue_right',
      wl: 'wl',
      af: 'af',
      dark: 'dark',
    });
    expect(manifest.consent.consent_recorded).toBe(true);
    for (const field of ['wl', 'af', 'dark', 'ref_wl', 'ref_af', 'spectrum']) {
      expect(body.get(field)).toBeInstanceOf(File);
    }
    expect((body.get('wl') as File).name).toBe('a_wl.jpg');
    // No Content-Type header: the browser must set the multipart boundary itself.
    expect((init.headers as Record<string, string>)['Content-Type']).toBeUndefined();
  });

  it('refuses to build a multipart body when a manifest file is missing', () => {
    const req = makeRequest();
    delete req.files.af;
    expect(() => buildUploadFormData(req)).toThrow(/af/);
  });

  it('turns FastAPI errors into ApiError with the plain detail', async () => {
    const fetch = vi
      .fn()
      .mockResolvedValue(jsonResponse({ detail: 'Consent must be recorded.' }, 422));
    const api = createHttpClient({ fetch });
    await expect(api.listScreenings()).rejects.toMatchObject({
      kind: 'http',
      status: 422,
      detail: 'Consent must be recorded.',
    });
  });

  it('reports network failures as kind=network', async () => {
    const api = createHttpClient({
      fetch: vi.fn().mockRejectedValue(new TypeError('Failed to fetch')),
    });
    const err = await api.getVersion().catch((e: unknown) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect((err as ApiError).kind).toBe('network');
  });

  it('treats an empty 500 / 502 from the proxy as the API being unreachable', async () => {
    const api = createHttpClient({
      fetch: vi.fn().mockResolvedValue(new Response('', { status: 500 })),
    });
    await expect(api.listScreenings()).rejects.toMatchObject({ kind: 'network' });
    const api2 = createHttpClient({
      fetch: vi.fn().mockResolvedValue(new Response('Bad gateway', { status: 502 })),
    });
    await expect(api2.listScreenings()).rejects.toMatchObject({ kind: 'network' });
  });

  it('follows up with a GET when an endpoint only returns the screening id', async () => {
    const fetch = vi
      .fn()
      .mockResolvedValueOnce(jsonResponse({ screening_id: 'abc' }))
      .mockResolvedValueOnce(jsonResponse(detail));
    const api = createHttpClient({ fetch });
    const d = await api.runDemo('low');
    expect(fetch.mock.calls[1]![0]).toBe('/api/v1/screenings/abc');
    expect(d.latest.result_id).toBe(highResult.result_id);
  });
});

describe('normalisers', () => {
  it('accepts a bare ScreeningResult as a detail', () => {
    const d = normalizeScreeningDetail(highResult);
    expect(d.screening_id).toBe(highResult.screening_id);
    expect(d.versions).toHaveLength(1);
  });

  it('accepts {version} objects and plain numbers in versions[]', () => {
    const d = normalizeScreeningDetail({
      ...detail,
      latest: { ...highResult, result_version: 2 },
      versions: [{ version: 1 }, 2],
    });
    expect(d.versions.map((v) => v.result_version)).toEqual([1, 2]);
  });

  it('falls back to the built-in site catalogue and fills missing fields', () => {
    expect(normalizeSiteCatalogue({}).sites).toHaveLength(14);
    const cat = normalizeSiteCatalogue({
      view_box: '0 0 400 320',
      sites: [{ site: 'hard_palate', label: 'Palate', cx: 1, cy: 2, rx: 3, ry: 4 }],
    });
    expect(cat.viewBox).toEqual([0, 0, 400, 320]);
    expect(cat.sites[0]).toMatchObject({
      site: 'hard_palate',
      label: 'Palate',
      shape: { cx: 1, cy: 2, rx: 3, ry: 4 },
    });
  });
});

describe('mock client', () => {
  it('serves seeded synthetic screenings and runs demos', async () => {
    const api = createMockClient({ analysisLatencyMs: 0, readLatencyMs: 0 });
    const before = await api.listScreenings();
    expect(before.length).toBe(3);
    expect(before.every((s) => s.synthetic)).toBe(true);
    const d = await api.runDemo('moderate');
    expect(d.latest.triage.overall_risk_level).toBe('MODERATE');
    expect(d.latest.synthetic).toBe(true);
    expect((await api.listScreenings()).length).toBe(4);
  });

  it('re-analysis creates a new version and never overwrites the old one', async () => {
    const api = createMockClient({ analysisLatencyMs: 0, readLatencyMs: 0 });
    const d2 = await api.reanalyze(highResult.screening_id);
    expect(d2.latest.result_version).toBe(2);
    expect(d2.latest.supersedes_result_id).toBe(highResult.result_id);
    expect(d2.versions.map((v) => v.result_version)).toEqual([1, 2]);
    const v1 = await api.getResultVersion(highResult.screening_id, 1);
    expect(v1.result_id).toBe(highResult.result_id);
  });

  it('rejects uploads without consent', async () => {
    const api = createMockClient({ analysisLatencyMs: 0, readLatencyMs: 0 });
    const req = makeRequest();
    req.manifest.consent.consent_recorded = false;
    await expect(api.createScreening(req)).rejects.toMatchObject({ status: 422 });
  });

  it('resolves artifact keys to bundled synthetic assets', () => {
    const api = createMockClient();
    const key = highResult.sites[0]!.artifacts!.wl_image!;
    expect(api.artifactUrl('x', key)).toMatch(/synthetic_tongue_r_wl/);
    expect(api.reportUrl('x')).toMatch(/\.pdf/);
  });
});
