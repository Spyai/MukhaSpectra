/**
 * Defensive normalisers for API envelopes. The ScreeningResult itself is a strict, versioned
 * contract; the envelopes around it (lists, detail, catalogue) are normalised here so small
 * shape differences between backend revisions never crash the console.
 */
import { BUILTIN_SITE_CATALOGUE, isOralSite } from '../lib/sites';
import type {
  ResultVersionSummary,
  RiskLevel,
  ScreeningDetail,
  ScreeningResult,
  ScreeningSummary,
  SiteCatalogue,
  SiteCatalogueEntry,
} from './types';

type Json = Record<string, unknown>;

const isObj = (v: unknown): v is Json => typeof v === 'object' && v !== null && !Array.isArray(v);
const num = (v: unknown): number | undefined => (typeof v === 'number' && Number.isFinite(v) ? v : undefined);
const str = (v: unknown): string | undefined => (typeof v === 'string' ? v : undefined);
const RISK = new Set(['LOW', 'MODERATE', 'HIGH', 'INCONCLUSIVE']);
const risk = (v: unknown): RiskLevel | undefined => (typeof v === 'string' && RISK.has(v) ? (v as RiskLevel) : undefined);

export function isScreeningResult(v: unknown): v is ScreeningResult {
  return isObj(v) && typeof v.result_id === 'string' && isObj(v.triage) && Array.isArray(v.sites);
}

export function normalizeSiteCatalogue(raw: unknown): SiteCatalogue {
  const list: unknown[] = Array.isArray(raw)
    ? raw
    : isObj(raw) && Array.isArray(raw.sites)
      ? raw.sites
      : [];
  const sites: SiteCatalogueEntry[] = [];
  for (const item of list) {
    if (!isObj(item) || !isOralSite(item.site ?? item.id)) continue;
    const site = (item.site ?? item.id) as SiteCatalogueEntry['site'];
    const fallback = BUILTIN_SITE_CATALOGUE.sites.find((s) => s.site === site)!;
    const shapeRaw = isObj(item.shape) ? item.shape : isObj(item.ellipse) ? item.ellipse : item;
    const cx = num(shapeRaw.cx);
    const cy = num(shapeRaw.cy);
    const rx = num(shapeRaw.rx);
    const ry = num(shapeRaw.ry);
    const paired = item.paired_site;
    sites.push({
      site,
      label: str(item.label) ?? fallback.label,
      short_label: str(item.short_label) ?? fallback.short_label,
      paired_site: isOralSite(paired) ? paired : paired === null ? null : fallback.paired_site,
      pairing:
        item.pairing === 'contralateral' || item.pairing === 'analogous'
          ? item.pairing
          : item.pairing === null
            ? null
            : fallback.pairing,
      keratinized: typeof item.keratinized === 'boolean' ? item.keratinized : fallback.keratinized,
      shape:
        cx !== undefined && cy !== undefined && rx !== undefined && ry !== undefined
          ? { cx, cy, rx, ry }
          : fallback.shape,
    });
  }
  if (sites.length === 0) return BUILTIN_SITE_CATALOGUE;
  let viewBox = BUILTIN_SITE_CATALOGUE.viewBox;
  const vb = isObj(raw) ? (raw.viewBox ?? raw.viewbox ?? raw.view_box) : undefined;
  if (Array.isArray(vb) && vb.length === 4 && vb.every((n) => typeof n === 'number')) {
    viewBox = vb as SiteCatalogue['viewBox'];
  } else if (typeof vb === 'string') {
    const parts = vb.trim().split(/[\s,]+/).map(Number);
    if (parts.length === 4 && parts.every(Number.isFinite)) viewBox = parts as SiteCatalogue['viewBox'];
  }
  return { viewBox, sites };
}

function normalizeVersions(raw: unknown, latest: ScreeningResult): ResultVersionSummary[] {
  const out: ResultVersionSummary[] = [];
  if (Array.isArray(raw)) {
    for (const v of raw) {
      if (typeof v === 'number') {
        out.push({ result_version: v });
      } else if (isObj(v)) {
        const n = num(v.result_version) ?? num(v.version);
        if (n === undefined) continue;
        out.push({
          result_version: n,
          result_id: str(v.result_id),
          created_at: str(v.created_at),
          overall_risk_level: risk(v.overall_risk_level) ?? risk(v.risk_level),
        });
      }
    }
  }
  if (!out.some((v) => v.result_version === latest.result_version)) {
    out.push({
      result_version: latest.result_version,
      result_id: latest.result_id,
      created_at: latest.created_at,
      overall_risk_level: latest.triage.overall_risk_level,
    });
  }
  return out.sort((a, b) => a.result_version - b.result_version);
}

/** Accepts `{screening_id, created_at, latest, versions}` or a bare ScreeningResult. */
export function normalizeScreeningDetail(raw: unknown): ScreeningDetail {
  if (isScreeningResult(raw)) {
    return {
      screening_id: raw.screening_id,
      created_at: raw.created_at,
      latest: raw,
      versions: normalizeVersions([], raw),
    };
  }
  if (isObj(raw)) {
    const latest = raw.latest ?? raw.result ?? raw.latest_result;
    if (isScreeningResult(latest)) {
      return {
        screening_id: str(raw.screening_id) ?? str(raw.id) ?? latest.screening_id,
        created_at: str(raw.created_at) ?? latest.created_at,
        latest,
        versions: normalizeVersions(raw.versions, latest),
      };
    }
  }
  throw new Error('bad-response');
}

export function normalizeScreeningList(raw: unknown): ScreeningSummary[] {
  const list: unknown[] = Array.isArray(raw)
    ? raw
    : isObj(raw) && Array.isArray(raw.items)
      ? raw.items
      : isObj(raw) && Array.isArray(raw.screenings)
        ? raw.screenings
        : [];
  const out: ScreeningSummary[] = [];
  for (const item of list) {
    if (!isObj(item)) continue;
    const latest = isScreeningResult(item.latest) ? item.latest : undefined;
    const id = str(item.screening_id) ?? str(item.id) ?? latest?.screening_id;
    if (!id) continue;
    const patient = isObj(item.patient) ? item.patient : undefined;
    const sitesRaw = Array.isArray(item.sites) ? item.sites : undefined;
    out.push({
      screening_id: id,
      created_at: str(item.created_at) ?? latest?.created_at ?? '',
      overall_risk_level:
        risk(item.overall_risk_level) ?? risk(item.risk_level) ?? latest?.triage.overall_risk_level,
      synthetic: typeof item.synthetic === 'boolean' ? item.synthetic : latest?.synthetic,
      pseudonymous_id:
        str(item.pseudonymous_id) ?? str(patient?.pseudonymous_id) ?? latest?.patient.pseudonymous_id,
      age_years: num(item.age_years) ?? num(patient?.age_years) ?? latest?.patient.age_years ?? null,
      sex: (str(item.sex) ?? str(patient?.sex) ?? latest?.patient.sex) as ScreeningSummary['sex'],
      latest_version: num(item.latest_version) ?? num(item.result_version) ?? latest?.result_version,
      sites: sitesRaw
        ? sitesRaw
            .map((s) => (isObj(s) ? s.site : s))
            .filter(isOralSite)
        : latest?.sites.map((s) => s.site),
    });
  }
  return out.sort((a, b) => (a.created_at < b.created_at ? 1 : -1));
}
