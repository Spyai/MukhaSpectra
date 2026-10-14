import type { StageTiming } from '../api/types';

export const DISPLAY_STAGES = [
  'quality',
  'align',
  'tissue',
  'af',
  'spectral',
  'triage',
  'report',
] as const;
export type DisplayStage = (typeof DISPLAY_STAGES)[number];

const STAGE_PATTERNS: Array<[DisplayStage, RegExp]> = [
  ['quality', /qc|quality|decode|calib|dark/i],
  ['align', /regist|align|warp/i],
  ['tissue', /segment|mucosa|tissue|mask/i],
  ['spectral', /spectr|probe/i],
  ['af', /(^|_)af|fluor|fvl|confound|heatmap|overlay|artifact|grid/i],
  ['triage', /triage|fusion|rule|lesion/i],
  ['report', /report|pdf|render/i],
];

export interface StageSummary {
  stage: DisplayStage;
  durationMs: number | null;
  status: 'done' | 'warn' | 'skipped' | 'not-timed';
}

/** Aggregate real pipeline stage timings (per site) into the seven display stages. */
export function summariseStages(stages: StageTiming[]): StageSummary[] {
  const buckets = new Map<DisplayStage, StageTiming[]>();
  for (const s of stages) {
    const hit = STAGE_PATTERNS.find(([, re]) => re.test(s.stage));
    if (!hit) continue;
    const list = buckets.get(hit[0]) ?? [];
    list.push(s);
    buckets.set(hit[0], list);
  }
  return DISPLAY_STAGES.map((stage) => {
    const list = buckets.get(stage);
    if (!list || list.length === 0)
      return { stage, durationMs: null, status: 'not-timed' as const };
    const durationMs = list.reduce((a, b) => a + b.duration_ms, 0);
    const status = list.some((s) => s.status === 'failed')
      ? ('warn' as const)
      : list.every((s) => s.status === 'skipped')
        ? ('skipped' as const)
        : ('done' as const);
    return { stage, durationMs, status };
  });
}
