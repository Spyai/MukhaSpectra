import type { ContributingFactor, FactorSource } from '../api/types';
import type { IconName } from '../components/ui/Icon';
import { round } from './format';

export const SOURCE_ICON: Record<FactorSource, IconName> = {
  autofluorescence: 'glow',
  wl_model: 'camera',
  spectral: 'wave',
  clinical_visual: 'eye',
  clinical_history: 'clipboard',
  symptom: 'pulse',
  site: 'pin',
  qc: 'shield',
  override: 'lock',
};

export interface WaterfallRow {
  factor: ContributingFactor;
  start: number;
  end: number;
}

/** Running sum: each factor's bar starts where the previous one ended. */
export function waterfall(factors: ContributingFactor[]): { rows: WaterfallRow[]; total: number } {
  let acc = 0;
  const rows = factors.map((factor) => {
    const start = acc;
    acc = round(acc + factor.points, 6);
    return { factor, start, end: acc };
  });
  return { rows, total: round(acc, 2) };
}
