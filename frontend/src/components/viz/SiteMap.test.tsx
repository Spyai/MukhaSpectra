import { fireEvent, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import { BUILTIN_SITE_CATALOGUE } from '../../lib/sites';
import { nextSiteInDirection } from '../../lib/siteNav';
import { renderWithProviders } from '../../test/utils';
import { SiteMap } from './SiteMap';

const byId = (id: string) => BUILTIN_SITE_CATALOGUE.sites.find((s) => s.site === id)!;

describe('SiteMap', () => {
  it('renders all 14 oral sites as buttons in select mode', () => {
    renderWithProviders(
      <SiteMap
        catalogue={BUILTIN_SITE_CATALOGUE}
        mode="select"
        selected={null}
        onSelect={() => {}}
        label="Map"
      />,
    );
    const map = screen.getByRole('group', { name: 'Map' });
    const buttons = map.querySelectorAll('[role="button"]');
    expect(buttons).toHaveLength(14);
    expect(screen.getByRole('button', { name: /Right lateral tongue/ })).toBeInTheDocument();
    expect(
      screen.getByRole('button', { name: /Ventral tongue \/ floor of mouth/ }),
    ).toBeInTheDocument();
  });

  it('uses one tab stop and moves focus spatially with the arrow keys; Enter selects', () => {
    const onSelect = vi.fn();
    renderWithProviders(
      <SiteMap
        catalogue={BUILTIN_SITE_CATALOGUE}
        mode="select"
        selected="lateral_tongue_right"
        onSelect={onSelect}
        label="Map"
      />,
    );
    const start = screen.getByRole('button', { name: /Right lateral tongue/ });
    const tabStops = screen.getByRole('group', { name: 'Map' }).querySelectorAll('[tabindex="0"]');
    expect(tabStops).toHaveLength(1);
    expect(tabStops[0]).toBe(start);
    start.focus();
    // Patient's right is on the viewer's left: moving right goes to the tongue dorsum.
    fireEvent.keyDown(start, { key: 'ArrowRight' });
    const dorsum = screen.getByRole('button', { name: /Tongue dorsum/ });
    expect(dorsum).toHaveFocus();
    fireEvent.keyDown(dorsum, { key: 'Enter' });
    expect(onSelect).toHaveBeenCalledWith('tongue_dorsum');
    fireEvent.keyDown(dorsum, { key: 'ArrowDown' });
    const under = screen.getByRole('button', { name: /Ventral tongue/ });
    expect(under).toHaveFocus();
    fireEvent.keyDown(under, { key: ' ' });
    expect(onSelect).toHaveBeenLastCalledWith('ventral_tongue_floor_of_mouth');
  });

  it('marks the selected site as pressed and announces the suggested reference', () => {
    renderWithProviders(
      <SiteMap
        catalogue={BUILTIN_SITE_CATALOGUE}
        mode="select"
        selected="buccal_mucosa_left"
        reference="buccal_mucosa_right"
        onSelect={() => {}}
        label="Map"
      />,
    );
    expect(screen.getByRole('button', { name: /^Left buccal mucosa/ })).toHaveAttribute(
      'aria-pressed',
      'true',
    );
    expect(
      screen.getByRole('button', { name: /Right buccal mucosa: Reference/ }),
    ).toBeInTheDocument();
  });

  it('results mode: only sites in the result are interactive and carry their risk level in words', () => {
    renderWithProviders(
      <SiteMap
        catalogue={BUILTIN_SITE_CATALOGUE}
        mode="results"
        selected="lateral_tongue_right"
        statuses={{
          lateral_tongue_right: { level: 'HIGH', role: 'examined' },
          lateral_tongue_left: { level: 'LOW', role: 'reference_only' },
        }}
        onSelect={() => {}}
        label="Map"
      />,
    );
    const buttons = screen.getByRole('group', { name: 'Map' }).querySelectorAll('[role="button"]');
    expect(buttons).toHaveLength(2);
    expect(
      screen.getByRole('button', { name: 'Right lateral tongue: High risk' }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole('button', { name: 'Left lateral tongue: Reference' }),
    ).toBeInTheDocument();
  });

  it('spatial navigation picks the nearest site in the pressed direction', () => {
    const sites = BUILTIN_SITE_CATALOGUE.sites;
    expect(nextSiteInDirection(sites, byId('buccal_mucosa_right'), 'left')).toBeUndefined();
    expect(nextSiteInDirection(sites, byId('hard_palate'), 'up')?.site).toBe('upper_gingiva');
    expect(nextSiteInDirection(sites, byId('lateral_tongue_left'), 'right')?.site).toBe(
      'buccal_mucosa_left',
    );
  });
});
