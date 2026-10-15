import { screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import type { RiskLevel } from '../../api/types';
import { renderWithProviders } from '../../test/utils';
import { RiskBadge, RiskPill } from './RiskBadge';

const CASES: Array<[RiskLevel, string]> = [
  ['LOW', 'Low risk'],
  ['MODERATE', 'Moderate risk'],
  ['HIGH', 'High risk'],
  ['INCONCLUSIVE', 'Inconclusive'],
];

describe('RiskBadge', () => {
  it.each(CASES)('renders %s with a text label, not colour alone', (level, label) => {
    renderWithProviders(<RiskBadge level={level} />);
    const badge = screen.getByTestId('risk-badge');
    expect(badge).toHaveAttribute('data-level', level);
    expect(badge).toHaveAccessibleName(`Risk level: ${label}`);
    // The level word is visible text, and a glyph (svg) is always drawn.
    expect(badge.textContent).toContain(label.split(' ')[0]);
    expect(badge.querySelectorAll('svg').length).toBeGreaterThan(0);
  });

  it('draws INCONCLUSIVE as a dashed ring with a "?" glyph', () => {
    renderWithProviders(<RiskBadge level="INCONCLUSIVE" />);
    const badge = screen.getByTestId('risk-badge');
    expect(badge.querySelector('circle[stroke-dasharray]')).not.toBeNull();
  });

  it('lights one, two or three gauge segments for LOW, MODERATE and HIGH', () => {
    for (const [level, lit] of [
      ['LOW', 1],
      ['MODERATE', 2],
      ['HIGH', 3],
    ] as const) {
      const { unmount } = renderWithProviders(<RiskBadge level={level} />);
      const paths = screen
        .getByTestId('risk-badge')
        .querySelectorAll('svg[viewBox="0 0 120 120"] path');
      // 3 track segments + lit segments
      expect(paths.length).toBe(3 + lit);
      unmount();
    }
  });

  it('pill shows the level word in every case', () => {
    renderWithProviders(
      <>
        {CASES.map(([l]) => (
          <RiskPill key={l} level={l} />
        ))}
      </>,
    );
    const pills = screen.getAllByTestId('risk-pill');
    expect(pills.map((p) => p.textContent)).toEqual(['Low', 'Moderate', 'High', 'Inconclusive']);
  });
});
