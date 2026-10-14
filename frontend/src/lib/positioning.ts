/**
 * Positioning guard, ported from pipeline/mukha_pipeline/positioning.py.
 *
 * MukhaSpectra is a screening and triage aid. No UI text may say "diagnosis", "malignant",
 * "cancer detected", "no cancer" or "cancer-free"; the only allowed uses of the word family are
 * the mandated disclaimer phrases below. Keep these lists in sync with the Python module.
 */

export const ALLOWED_PHRASES: readonly string[] = [
  'not a diagnosis',
  'not a diagnostic device',
  'never a diagnostic device',
  'final diagnosis requires',
  'diagnosis requires a clinician',
];

export const FORBIDDEN_PATTERNS: ReadonlyArray<readonly [string, RegExp]> = [
  ['diagnosis', /\bdiagnos[a-z]*/gi],
  ['malignant', /\bmalignan[a-z]*/gi],
  ['cancer detected', /\bcancer\s+(?:was\s+|is\s+)?(?:detected|found|present|confirmed)/gi],
  ['no cancer', /\bno\s+(?:sign\s+of\s+)?cancer\b/gi],
  ['cancer-free', /\bcancer[\s-]free\b/gi],
];

/** Performance claims are not allowed anywhere in the UI (no validation data exists). */
export const PERFORMANCE_CLAIM_PATTERN = /\b(?:accuracy|sensitivity|specificity|AUC|PPV|NPV)\b/gi;

function escapeRegExp(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

/** Return human-readable violations found in `text` (empty when clean). */
export function findPositioningViolations(text: string): string[] {
  let scrubbed = text;
  for (const phrase of ALLOWED_PHRASES) {
    scrubbed = scrubbed.replace(new RegExp(escapeRegExp(phrase), 'gi'), ' ');
  }
  const violations: string[] = [];
  for (const [name, pattern] of FORBIDDEN_PATTERNS) {
    for (const m of scrubbed.matchAll(new RegExp(pattern.source, pattern.flags))) {
      const start = Math.max(0, (m.index ?? 0) - 40);
      const context = scrubbed.slice(start, (m.index ?? 0) + m[0].length + 40).replace(/\n/g, ' ');
      violations.push(`'${name}': …${context}…`);
    }
  }
  return violations;
}
