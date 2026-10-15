import type { SiteCatalogueEntry } from '../api/types';

export type Dir = 'left' | 'right' | 'up' | 'down';

/** Spatial arrow-key navigation: nearest site in the pressed direction. */
export function nextSiteInDirection(
  sites: SiteCatalogueEntry[],
  current: SiteCatalogueEntry,
  dir: Dir,
): SiteCatalogueEntry | undefined {
  const { cx, cy } = current.shape;
  let best: SiteCatalogueEntry | undefined;
  let bestScore = Infinity;
  for (const s of sites) {
    if (s.site === current.site) continue;
    const dx = s.shape.cx - cx;
    const dy = s.shape.cy - cy;
    const primary = dir === 'left' ? -dx : dir === 'right' ? dx : dir === 'up' ? -dy : dy;
    const ortho = dir === 'left' || dir === 'right' ? Math.abs(dy) : Math.abs(dx);
    if (primary <= 2) continue;
    const score = primary + ortho * 1.6;
    if (score < bestScore) {
      bestScore = score;
      best = s;
    }
  }
  return best;
}
