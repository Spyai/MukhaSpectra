import type { RiskLevel } from '../api/types';

export const RISK_LEVELS: RiskLevel[] = ['LOW', 'MODERATE', 'HIGH', 'INCONCLUSIVE'];

export interface RiskStyle {
  /** CSS colour (token) for strokes, icons and text. */
  color: string;
  /** Tinted background token. */
  bg: string;
  /** Number of lit segments on the 3-step gauge (0 = dashed / unknown). */
  steps: 0 | 1 | 2 | 3;
  dashed: boolean;
}

export const RISK_STYLE: Record<RiskLevel, RiskStyle> = {
  LOW: { color: 'var(--risk-low)', bg: 'var(--risk-low-bg)', steps: 1, dashed: false },
  MODERATE: { color: 'var(--risk-moderate)', bg: 'var(--risk-moderate-bg)', steps: 2, dashed: false },
  HIGH: { color: 'var(--risk-high)', bg: 'var(--risk-high-bg)', steps: 3, dashed: false },
  INCONCLUSIVE: {
    color: 'var(--risk-inconclusive)',
    bg: 'var(--risk-inconclusive-bg)',
    steps: 0,
    dashed: true,
  },
};

export function isRiskLevel(v: unknown): v is RiskLevel {
  return typeof v === 'string' && (RISK_LEVELS as string[]).includes(v);
}
