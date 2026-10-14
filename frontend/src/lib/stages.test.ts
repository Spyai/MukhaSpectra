import { describe, expect, it } from 'vitest';
import { highResult } from '../mocks/fixtures';
import { summariseStages } from './stages';

describe('summariseStages', () => {
  it('maps pipeline stage timings onto the seven display stages', () => {
    const s = summariseStages([
      { stage: 'qc', duration_ms: 100, status: 'ok', site: 'hard_palate' },
      { stage: 'qc', duration_ms: 50, status: 'failed', site: 'soft_palate' },
      { stage: 'registration', duration_ms: 300, status: 'ok' },
      { stage: 'segmentation', duration_ms: 20, status: 'ok' },
      { stage: 'af_analysis', duration_ms: 40, status: 'ok' },
      { stage: 'spectral', duration_ms: 0, status: 'skipped' },
      { stage: 'triage', duration_ms: 5, status: 'ok' },
    ]);
    expect(s.map((x) => x.stage)).toEqual([
      'quality',
      'align',
      'tissue',
      'af',
      'spectral',
      'triage',
      'report',
    ]);
    expect(s[0]).toEqual({ stage: 'quality', durationMs: 150, status: 'warn' });
    expect(s[4]!.status).toBe('skipped');
    expect(s[6]).toEqual({ stage: 'report', durationMs: null, status: 'not-timed' });
  });

  it('handles the fixture timings', () => {
    const s = summariseStages(highResult.processing.stages ?? []);
    expect(s.find((x) => x.stage === 'align')!.durationMs).toBeGreaterThan(0);
  });
});
