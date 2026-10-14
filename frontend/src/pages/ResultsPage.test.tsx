import { screen, within } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import type { ApiClient, ScreeningResult } from '../api/types';
import { highResult, inconclusiveResult, lowResult } from '../mocks/fixtures';
import { fastMockClient, renderRoute } from '../test/utils';

function clientWith(result: ScreeningResult): ApiClient {
  const base = fastMockClient();
  const detail = {
    screening_id: result.screening_id,
    created_at: result.created_at,
    latest: result,
    versions: [{ result_version: result.result_version }],
  };
  return { ...base, getScreening: async () => detail };
}

describe('Results page', () => {
  it('shows the SYNTHETIC DATA ribbon for synthetic results', async () => {
    renderRoute(`/screenings/${highResult.screening_id}`);
    await screen.findByTestId('risk-badge');
    const ribbon = screen.getByTestId('synthetic-ribbon');
    expect(ribbon).toHaveTextContent('SYNTHETIC DATA');
    expect(ribbon).toHaveTextContent(highResult.notices!.synthetic_notice!);
  });

  it('does not show the ribbon for real data', async () => {
    const real: ScreeningResult = {
      ...highResult,
      synthetic: false,
      data_provenance: 'real',
      notices: { ...highResult.notices, synthetic_notice: null },
    };
    renderRoute(`/screenings/${real.screening_id}`, clientWith(real));
    await screen.findByTestId('risk-badge');
    expect(screen.queryByTestId('synthetic-ribbon')).not.toBeInTheDocument();
  });

  it('never shows a score without its reasons, and is always pending clinician review', async () => {
    renderRoute(`/screenings/${highResult.screening_id}`);
    expect(await screen.findByTestId('risk-badge')).toHaveAttribute('data-level', 'HIGH');
    expect(screen.getByTestId('review-chip')).toHaveTextContent('Pending clinician review');
    expect(screen.getAllByTestId('factor-row').length).toBe(
      highResult.triage.contributing_factors.length,
    );
    expect(screen.getByTestId('explanation')).toHaveTextContent(highResult.triage.explanation);
    expect(screen.getByTestId('timeline')).toHaveTextContent('14 days');
    const overrides = screen.getByTestId('overrides');
    expect(overrides).toHaveTextContent('Clinical red flag — the device cannot lower this.');
    expect(overrides).toHaveTextContent('Non-healing ulcer for more than 2 weeks');
    expect(screen.getByTestId('disclaimer')).toHaveTextContent(highResult.notices!.disclaimer!);
  });

  it('states model statuses honestly and lists retake prompts for failed captures', async () => {
    renderRoute(`/screenings/${highResult.screening_id}`);
    await screen.findByTestId('risk-badge');
    const models = screen.getByTestId('model-statuses');
    expect(models).toHaveTextContent('White-light classifier: not available — not used in triage');
    expect(models).toHaveTextContent('Spectral classifier: placeholder heuristic');
    expect(models).toHaveTextContent('Rules: baseline heuristic, unvalidated');
    const prompts = screen.getAllByTestId('retake-prompt').map((p) => p.textContent);
    expect(prompts).toContain('White-light photo blurry — hold the phone steady and retake.');
    expect(screen.getByTestId('spectral-classifier')).toHaveTextContent(/placeholder/i);
  });

  it('shows only pseudonymous id, age and sex for the patient', async () => {
    renderRoute(`/screenings/${highResult.screening_id}`);
    await screen.findByTestId('risk-badge');
    expect(screen.getByTestId('patient-id')).toHaveTextContent(
      `Patient ${highResult.patient.pseudonymous_id}`,
    );
  });

  it('renders INCONCLUSIVE with a retake action and LOW with routine re-screening', async () => {
    const { unmount } = renderRoute(`/screenings/${inconclusiveResult.screening_id}`);
    expect(await screen.findByTestId('risk-badge')).toHaveAttribute('data-level', 'INCONCLUSIVE');
    expect(screen.getByTestId('timeline')).toHaveTextContent('Now');
    expect(
      screen.getByText(/No score: the images could not be analysed reliably/),
    ).toBeInTheDocument();
    unmount();
    renderRoute(`/screenings/${lowResult.screening_id}`);
    expect(await screen.findByTestId('risk-badge')).toHaveAttribute('data-level', 'LOW');
    expect(screen.getByTestId('recommended-action')).toHaveTextContent(
      'This does not rule out disease.',
    );
  });

  it('links the PDF report for the version being viewed', async () => {
    renderRoute(`/screenings/${highResult.screening_id}`);
    await screen.findByTestId('risk-badge');
    const link = screen.getByTestId('report-link');
    expect(link).toHaveAttribute('href', expect.stringMatching(/\.pdf/));
    expect(
      within(screen.getByTestId('version-switcher')).getByRole('button', { name: 'Version 1' }),
    ).toHaveAttribute('aria-pressed', 'true');
  });
});
