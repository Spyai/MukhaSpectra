import { describe, expect, it } from 'vitest';
import en from '../i18n/en.json';
import { FIXTURES } from '../mocks/fixtures';
import { findPositioningViolations, PERFORMANCE_CLAIM_PATTERN } from './positioning';

function flatten(obj: unknown, prefix = ''): Array<[string, string]> {
  if (typeof obj === 'string') return [[prefix, obj]];
  if (obj && typeof obj === 'object') {
    return Object.entries(obj).flatMap(([k, v]) => flatten(v, prefix ? `${prefix}.${k}` : k));
  }
  return [];
}

// Every source file the UI is built from (string literals and JSX text included), minus tests and
// the guard itself, which necessarily spells out the forbidden words.
const sources = import.meta.glob(
  ['../**/*.{ts,tsx}', '!../**/*.test.{ts,tsx}', '!./positioning.ts'],
  {
    query: '?raw',
    import: 'default',
    eager: true,
  },
) as Record<string, string>;

describe('positioning guard (port of mukha_pipeline/positioning.py)', () => {
  it('detects forbidden phrasing and allows the mandated disclaimers', () => {
    expect(findPositioningViolations('Cancer detected on the tongue')).toHaveLength(1);
    expect(findPositioningViolations('This lesion is malignant')).toHaveLength(1);
    expect(findPositioningViolations('No cancer here. Patient is cancer-free.')).toHaveLength(2);
    expect(findPositioningViolations('Our diagnostic AI')).toHaveLength(1);
    expect(findPositioningViolations('Screening aid — not a diagnosis.')).toEqual([]);
    expect(
      findPositioningViolations(
        'Final diagnosis requires examination by a clinician and histopathology.',
      ),
    ).toEqual([]);
    expect(findPositioningViolations('Previous oral cancer')).toEqual([]);
  });

  it('en.json contains no forbidden phrasing', () => {
    const bad = flatten(en).flatMap(([key, text]) =>
      findPositioningViolations(text).map((v) => `${key}: ${v}`),
    );
    expect(bad).toEqual([]);
  });

  it('en.json makes no performance claims (no accuracy / sensitivity / specificity)', () => {
    const bad = flatten(en).filter(([, text]) =>
      new RegExp(PERFORMANCE_CLAIM_PATTERN.source, 'i').test(text),
    );
    expect(bad).toEqual([]);
  });

  it('no .ts/.tsx source (string literals, JSX text, fixtures) contains forbidden phrasing', () => {
    expect(Object.keys(sources).length).toBeGreaterThan(30);
    const bad = Object.entries(sources).flatMap(([file, text]) =>
      findPositioningViolations(text).map((v) => `${file}: ${v}`),
    );
    expect(bad).toEqual([]);
  });

  it('mock fixtures carry the disclaimer and no forbidden phrasing', () => {
    for (const r of Object.values(FIXTURES)) {
      expect(r.notices?.disclaimer).toMatch(/^Screening aid — not a diagnosis\./);
      expect(findPositioningViolations(JSON.stringify(r))).toEqual([]);
      expect(r.triage.requires_clinician_review).toBe(true);
      expect(r.versions.models.every((m) => m.validated === false)).toBe(true);
    }
  });
});
