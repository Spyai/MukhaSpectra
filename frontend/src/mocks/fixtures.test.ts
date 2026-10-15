import Ajv2020 from 'ajv/dist/2020';
import { describe, expect, it } from 'vitest';
import schema from '../../../pipeline/mukha_pipeline/schema/screening_result.v1.schema.json';
import { FIXTURES } from './fixtures';

describe('mock fixtures conform to the ScreeningResult v1 JSON Schema', () => {
  const ajv = new Ajv2020({ allErrors: true, strict: false, validateFormats: false });
  const validate = ajv.compile(schema);

  it.each(Object.entries(FIXTURES))('%s', (_name, result) => {
    const ok = validate(JSON.parse(JSON.stringify(result)));
    expect(validate.errors ?? []).toEqual([]);
    expect(ok).toBe(true);
    expect(result.synthetic).toBe(true);
  });

  it('the HIGH fixture exercises the hard cases', () => {
    const r = FIXTURES.high;
    expect(r.lesions).toHaveLength(2);
    expect(
      r.lesions!.some((l) => (l.confounders ?? []).some((c) => c.type === 'porphyrin_red')),
    ).toBe(true);
    expect(r.spectral_readings!.map((s) => s.role).sort()).toEqual(['lesion', 'reference']);
    expect(r.sites.some((s) => s.wl?.status === 'model_unavailable')).toBe(true);
    expect(
      r.sites.some((s) => !s.qc.passed && s.analysis_status === 'inconclusive_qc_failed'),
    ).toBe(true);
    expect(r.triage.overrides_applied!.length).toBeGreaterThan(0);
  });
});
