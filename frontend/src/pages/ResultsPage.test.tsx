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
    expect(screen.getByTestId('timeline')).toHaveTextContent('Before the person leaves');
    expect(
      screen.getByText(/No score: the images could not be analysed reliably/),
    ).toBeInTheDocument();
    unmount();
    renderRoute(`/screenings/${lowResult.screening_id}`);
    expect(await screen.findByTestId('risk-badge')).toHaveAttribute('data-level', 'LOW');
    expect(screen.getByTestId('timeline')).toHaveTextContent('in 12 months');
    expect(screen.getByText('Next screening')).toBeInTheDocument();
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

  it('puts retake prompts right under the risk header and makes retaking the next step', async () => {
    renderRoute(`/screenings/${inconclusiveResult.screening_id}`);
    await screen.findByTestId('risk-badge');
    const callout = screen.getByTestId('retake-callout');
    expect(callout).toHaveAttribute('data-primary', 'true');
    expect(callout).toHaveTextContent('1 site could not be analysed');
    expect(callout).toHaveTextContent(
      'White-light photo blurry — hold the phone steady and retake.',
    );
    const link = within(callout).getByRole('link', {
      name: 'Retake photos for Left buccal mucosa',
    });
    expect(link).toHaveAttribute('href', '/upload?site=buccal_mucosa_left');
    // Re-analysing the same bad photos cannot help: demoted, with the reason.
    expect(screen.getByTestId('reanalyse')).toHaveAttribute('data-demoted', 'true');
    expect(screen.getByTestId('reanalyse')).toHaveAccessibleDescription(
      'Re-analysing the same photos will not fix quality problems.',
    );
    // "No suspicious findings" is never shown alone when a site was not analysed.
    expect(screen.getByTestId('lesions-not-analysed')).toHaveTextContent(
      '1 site could not be analysed — see the retake prompts.',
    );
  });

  it('a HIGH result with a failed site keeps re-analyse primary but still lists the retake', async () => {
    renderRoute(`/screenings/${highResult.screening_id}`);
    await screen.findByTestId('risk-badge');
    const callout = screen.getByTestId('retake-callout');
    expect(callout).toHaveAttribute('data-primary', 'false');
    expect(within(callout).getByRole('link', { name: /Lower gingiva/ })).toHaveAttribute(
      'href',
      '/upload?site=lower_gingiva',
    );
    expect(screen.getByTestId('reanalyse')).toHaveAttribute('data-demoted', 'false');
  });

  it('every finding score is shown with its main reasons, without expanding anything', async () => {
    renderRoute(`/screenings/${highResult.screening_id}`);
    await screen.findByTestId('risk-badge');
    const cards = screen.getAllByTestId('lesion-card');
    expect(cards).toHaveLength(highResult.lesions!.length);
    for (const card of cards) {
      const reasons = within(card).getByTestId('lesion-reasons');
      expect(reasons).toHaveTextContent(/Finding score/);
      expect(within(reasons).getAllByRole('listitem').length).toBeGreaterThan(0);
    }
  });

  it('groups "Why this level" into the finding and risk factors and says only the top finding counts', async () => {
    renderRoute(`/screenings/${highResult.screening_id}`);
    await screen.findByTestId('risk-badge');
    const groups = screen.getAllByTestId('factor-group');
    expect(groups.map((g) => g.getAttribute('data-group'))).toEqual(['finding', 'risk']);
    expect(groups[0]).toHaveTextContent('Finding L1 · Right lateral tongue');
    expect(screen.getByTestId('factor-note')).toHaveTextContent(
      'Only the highest-scoring finding (L1) counts towards this total',
    );
    expect(screen.getByTestId('factor-note')).toHaveTextContent('L2 · 3.29');
  });

  it('shows the ID with a referral-slip hint and a copy button', async () => {
    renderRoute(`/screenings/${highResult.screening_id}`);
    await screen.findByTestId('risk-badge');
    expect(screen.getByText('Write this ID on the person’s referral slip')).toBeInTheDocument();
    expect(screen.getByTestId('copy-id')).toHaveAccessibleName('Copy ID');
  });

  it('keeps the DOM (focus) order Why, Map, Viewer, Findings, Probe, QC', async () => {
    renderRoute(`/screenings/${highResult.screening_id}`);
    await screen.findByTestId('risk-badge');
    const ids = ['why', 'map', 'viewer', 'lesions', 'spectral', 'qc'];
    const els = ids.map((id) => document.getElementById(id)!);
    for (let i = 1; i < els.length; i++) {
      expect(
        els[i - 1]!.compareDocumentPosition(els[i]!) & Node.DOCUMENT_POSITION_FOLLOWING,
      ).toBeTruthy();
    }
  });
});
