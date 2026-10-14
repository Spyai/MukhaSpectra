import { act, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { highResult, lowResult } from '../../mocks/fixtures';
import { renderWithProviders } from '../../test/utils';
import { AnalysisOverlay } from './AnalysisOverlay';

const realMatchMedia = window.matchMedia;
/** Pretend the person has NOT asked for reduced motion (jsdom setup defaults to reduced). */
function allowMotion() {
  window.matchMedia = ((query: string) => ({
    ...realMatchMedia(query),
    matches: false,
  })) as typeof window.matchMedia;
}

afterEach(() => {
  vi.useRealTimers();
  window.matchMedia = realMatchMedia;
});

describe('AnalysisOverlay', () => {
  it('never marks a step as done before the server answers', () => {
    vi.useFakeTimers();
    renderWithProviders(
      <AnalysisOverlay open mode="upload" onContinue={() => {}} onClose={() => {}} />,
    );
    act(() => vi.advanceTimersByTime(5000));
    const states = screen.getAllByTestId(/^stage-/).map((el) => el.getAttribute('data-state'));
    expect(states).toHaveLength(7);
    expect(states.every((s) => s === 'waiting' || s === 'pulse')).toBe(true);
    expect(screen.queryByText('Done')).toBeNull();
  });

  it('shows measured stage timings and a SYNTHETIC tag once a synthetic result arrives', () => {
    renderWithProviders(
      <AnalysisOverlay
        open
        mode="demo"
        result={lowResult}
        onContinue={() => {}}
        onClose={() => {}}
      />,
    );
    const quality = screen.getByTestId('stage-quality');
    expect(quality).toHaveAttribute('data-state', 'done');
    expect(screen.getAllByTestId('synthetic-tag').length).toBeGreaterThan(0);
    expect(screen.getByTestId('analysis-thumbnail')).toHaveAttribute(
      'src',
      expect.stringContaining('wl'),
    );
  });

  it('auto-continues to the results when nothing needs a retake (motion allowed)', () => {
    allowMotion();
    vi.useFakeTimers();
    const onContinue = vi.fn();
    renderWithProviders(
      <AnalysisOverlay
        open
        mode="demo"
        result={lowResult}
        onContinue={onContinue}
        onClose={() => {}}
        autoContinueMs={100}
      />,
    );
    act(() => vi.advanceTimersByTime(1000));
    expect(onContinue).toHaveBeenCalledTimes(1);
  });

  it('calls out sites that need a retake and does not auto-continue past them', () => {
    allowMotion();
    vi.useFakeTimers();
    const onContinue = vi.fn();
    renderWithProviders(
      <AnalysisOverlay
        open
        mode="demo"
        result={highResult}
        onContinue={onContinue}
        onClose={() => {}}
        autoContinueMs={100}
      />,
    );
    expect(screen.getByTestId('analysis-retake')).toHaveTextContent('1 site needs a retake');
    act(() => vi.advanceTimersByTime(1000));
    expect(onContinue).not.toHaveBeenCalled();
  });

  it('is a modal rendered outside the app root', () => {
    const root = document.createElement('div');
    root.id = 'root';
    document.body.appendChild(root);
    const { unmount } = renderWithProviders(
      <AnalysisOverlay open mode="upload" onContinue={() => {}} onClose={() => {}} />,
    );
    const dialog = screen.getByRole('dialog');
    expect(root.contains(dialog)).toBe(false);
    expect(root.inert).toBe(true);
    unmount();
    expect(root.inert).toBe(false);
    root.remove();
  });
});
