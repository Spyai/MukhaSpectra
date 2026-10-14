/** Formatting helpers. Numbers use tabular figures in the UI; keep output short and stable. */

export function formatPoints(points: number, digits = 2): string {
  const v = Math.round(points * 10 ** digits) / 10 ** digits;
  const s =
    Math.abs(v)
      .toFixed(digits)
      .replace(/\.?0+$/, '') || '0';
  if (v > 0) return `+${s}`;
  if (v < 0) return `−${s}`;
  return '0';
}

export function formatNumber(v: number | null | undefined, digits = 2): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return '—';
  return v.toFixed(digits);
}

export function formatSigned(v: number | null | undefined, digits = 3): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return '—';
  const s = Math.abs(v).toFixed(digits);
  return v > 0 ? `+${s}` : v < 0 ? `−${s}` : s;
}

export function formatPercent(fraction: number | null | undefined, digits = 0): string {
  if (fraction === null || fraction === undefined || !Number.isFinite(fraction)) return '—';
  return `${(fraction * 100).toFixed(digits)}%`;
}

export function formatDuration(ms: number): string {
  if (ms >= 1000) return `${(ms / 1000).toFixed(ms >= 10000 ? 0 : 1)} s`;
  return `${Math.round(ms)} ms`;
}

export function formatDateTime(iso: string | null | undefined, locale = 'en-IN'): string {
  if (!iso) return '—';
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return new Intl.DateTimeFormat(locale, { dateStyle: 'medium', timeStyle: 'short' }).format(d);
}

export function formatDate(iso: string | null | undefined, locale = 'en-IN'): string {
  if (!iso) return '—';
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return new Intl.DateTimeFormat(locale, { dateStyle: 'medium' }).format(d);
}

export function shortSha(sha: string | null | undefined, n = 8): string {
  return sha ? sha.slice(0, n) : '—';
}

export function round(v: number, digits = 2): number {
  const k = 10 ** digits;
  return Math.round(v * k) / k;
}
