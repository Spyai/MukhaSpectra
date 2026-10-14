import { screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { highResult } from '../../mocks/fixtures';
import { renderRoute } from '../../test/utils';

const PATHS = [
  '/',
  '/upload',
  '/screenings',
  `/screenings/${highResult.screening_id}`,
  '/does-not-exist',
];

describe('prototype banner', () => {
  it.each(PATHS)('is present and cannot be dismissed on %s', async (path) => {
    renderRoute(path);
    const banner = await screen.findByTestId('prototype-banner');
    expect(banner).toHaveTextContent('Research prototype — not for clinical use');
    expect(banner.querySelector('button')).toBeNull();
  });
});
