/**
 * Formatting helpers. Numbers follow the active i18n language (Intl), use a true minus sign
 * (U+2212) everywhere and keep output short and stable.
 */
import i18n from 'i18next';

const LOCALES: Record<string, string> = { en: 'en-IN', hi: 'hi-IN', kn: 'kn-IN' };
const MINUS = '−';

/** BCP 47 locale for the active UI language (Indian formats for every supported language). */
export function currentLocale(): string {
  const lang = i18n.language || 'en';
  return LOCALES[lang] ?? lang;
}

const formatters = new Map<string, Intl.NumberFormat>();
function nf(maxDigits: number, minDigits: number): Intl.NumberFormat {
  const locale = currentLocale();
  const key = `${locale}|${minDigits}|${maxDigits}`;
  let f = formatters.get(key);
  if (!f) {
    f = new Intl.NumberFormat(locale, {
      minimumFractionDigits: minDigits,
      maximumFractionDigits: maxDigits,
      useGrouping: false,
    });
    formatters.set(key, f);
  }
  return f;
}

/** Absolute value, localised, rounded to `digits`; returns [text, sign of the rounded value]. */
function magnitude(v: number, maxDigits: number, minDigits: number): [string, -1 | 0 | 1] {
  const k = 10 ** maxDigits;
  const r = Math.round(v * k) / k;
  return [nf(maxDigits, minDigits).format(Math.abs(r)), r > 0 ? 1 : r < 0 ? -1 : 0];
}

export function formatPoints(points: number, digits = 2): string {
  const [s, sign] = magnitude(points, digits, 0);
  if (sign > 0) return `+${s}`;
  if (sign < 0) return `${MINUS}${s}`;
  return nf(0, 0).format(0);
}

export function formatNumber(v: number | null | undefined, digits = 2): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return '—';
  const [s, sign] = magnitude(v, digits, digits);
  return sign < 0 ? `${MINUS}${s}` : s;
}

export function formatSigned(v: number | null | undefined, digits = 3): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return '—';
  const [s, sign] = magnitude(v, digits, digits);
  return sign > 0 ? `+${s}` : sign < 0 ? `${MINUS}${s}` : s;
}

export function formatPercent(fraction: number | null | undefined, digits = 0): string {
  if (fraction === null || fraction === undefined || !Number.isFinite(fraction)) return '—';
  return new Intl.NumberFormat(currentLocale(), {
    style: 'percent',
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  })
    .format(fraction)
    .replace('-', MINUS);
}

export function formatDuration(ms: number): string {
  if (ms >= 1000) {
    const d = ms >= 10000 ? 0 : 1;
    return i18n.t('common.seconds', { value: nf(d, d).format(ms / 1000) });
  }
  return i18n.t('common.ms', { value: nf(0, 0).format(Math.round(ms)) });
}

export function formatDateTime(iso: string | null | undefined, locale = currentLocale()): string {
  if (!iso) return '—';
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return new Intl.DateTimeFormat(locale, { dateStyle: 'medium', timeStyle: 'short' }).format(d);
}

export function formatDate(iso: string | null | undefined, locale = currentLocale()): string {
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
