import { screen, within } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { waterfall } from '../../lib/factors';
import { highResult } from '../../mocks/fixtures';
import { renderWithProviders } from '../../test/utils';
import { FactorBars } from './FactorBars';

describe('FactorBars', () => {
  it('waterfall bars sum exactly to the triage score', () => {
    const tri = highResult.triage;
    const { rows, total } = waterfall(tri.contributing_factors);
    expect(total).toBeCloseTo(tri.score!, 6);
    // every bar starts where the previous one ended
    rows.forEach((r, i) => {
      if (i > 0) expect(r.start).toBeCloseTo(rows[i - 1]!.end, 9);
      expect(r.end - r.start).toBeCloseTo(r.factor.points, 9);
    });
  });

  it('renders one row per factor, with points and the total', () => {
    const tri = highResult.triage;
    renderWithProviders(
      <FactorBars
        factors={tri.contributing_factors}
        thresholds={tri.thresholds}
        score={tri.score}
        level="HIGH"
      />,
    );
    const rows = screen.getAllByTestId('factor-row');
    expect(rows).toHaveLength(tri.contributing_factors.length);
    const sum = rows.reduce((a, r) => a + Number(r.getAttribute('data-points')), 0);
    expect(sum).toBeCloseTo(tri.score!, 6);
    const totalRow = screen.getByTestId('factor-total');
    expect(Number(totalRow.getAttribute('data-total'))).toBeCloseTo(tri.score!, 6);
    expect(within(totalRow).getByText('11.25')).toBeInTheDocument();
    // threshold markers are labelled
    expect(screen.getByText('Moderate from 3')).toBeInTheDocument();
    expect(screen.getByText('High from 6')).toBeInTheDocument();
  });

  it('shows negative contributions (confounder discounts) with a minus sign', () => {
    const lesion = highResult.lesions![1]!;
    renderWithProviders(
      <FactorBars
        factors={lesion.contributing_factors}
        score={lesion.score}
        level="MODERATE"
        compact
      />,
    );
    expect(screen.getByText('−0.81')).toBeInTheDocument();
    expect(screen.getByText('−0.45')).toBeInTheDocument();
    expect(Number(screen.getByTestId('factor-total').getAttribute('data-total'))).toBeCloseTo(
      3.29,
      6,
    );
  });

  it('every row has an accessible sentence with label, points and source', () => {
    const tri = highResult.triage;
    renderWithProviders(<FactorBars factors={tri.contributing_factors} score={tri.score} />);
    expect(
      screen.getByText(
        /White patch seen by the health worker: \+2 points \(Seen by health worker\)/,
      ),
    ).toBeInTheDocument();
  });

  it('explains an empty factor list instead of showing a bare score', () => {
    renderWithProviders(<FactorBars factors={[]} score={0} />);
    expect(screen.getByTestId('factor-bars-empty')).toHaveTextContent(
      /No suspicious features added points/,
    );
  });
});
