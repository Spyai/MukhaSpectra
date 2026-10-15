import { fireEvent, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import { renderWithProviders } from '../../test/utils';
import { WipeSlider } from './WipeSlider';

function setup(onChange = vi.fn()) {
  renderWithProviders(
    <WipeSlider
      leftSrc="wl.png"
      rightSrc="af.png"
      leftAlt="White light"
      rightAlt="Violet light"
      leftLabel="WL"
      rightLabel="AF"
      onChange={onChange}
    />,
  );
  return { slider: screen.getByRole('slider'), onChange };
}

describe('WipeSlider', () => {
  it('is an accessible slider starting at 50%', () => {
    const { slider } = setup();
    expect(slider).toHaveAttribute('aria-valuemin', '0');
    expect(slider).toHaveAttribute('aria-valuemax', '100');
    expect(slider).toHaveAttribute('aria-valuenow', '50');
    expect(slider).toHaveAttribute('aria-valuetext', '50% white light');
    expect(slider).toHaveAttribute('tabindex', '0');
    expect(slider).toHaveAccessibleName(/Comparison position/);
  });

  it('moves with arrows, Shift+arrows, PageUp/PageDown, Home and End', () => {
    const { slider, onChange } = setup();
    fireEvent.keyDown(slider, { key: 'ArrowRight' });
    expect(slider).toHaveAttribute('aria-valuenow', '52');
    fireEvent.keyDown(slider, { key: 'ArrowLeft', shiftKey: true });
    expect(slider).toHaveAttribute('aria-valuenow', '42');
    fireEvent.keyDown(slider, { key: 'PageUp' });
    expect(slider).toHaveAttribute('aria-valuenow', '52');
    fireEvent.keyDown(slider, { key: 'ArrowDown' });
    expect(slider).toHaveAttribute('aria-valuenow', '50');
    fireEvent.keyDown(slider, { key: 'End' });
    expect(slider).toHaveAttribute('aria-valuenow', '100');
    fireEvent.keyDown(slider, { key: 'ArrowRight' });
    expect(slider).toHaveAttribute('aria-valuenow', '100');
    fireEvent.keyDown(slider, { key: 'Home' });
    expect(slider).toHaveAttribute('aria-valuenow', '0');
    expect(onChange).toHaveBeenLastCalledWith(0);
  });

  it('clips the white-light image to the slider position', () => {
    const { slider } = setup();
    fireEvent.keyDown(slider, { key: 'PageDown' });
    const wl = screen.getByAltText('White light');
    expect(wl.style.clipPath).toBe('inset(0 60% 0 0)');
  });
});
