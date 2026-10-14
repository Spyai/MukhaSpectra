import { AnimatePresence, m } from 'motion/react';
import { useCallback, useEffect, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import type { FVLRegion } from '../../api/types';
import { formatNumber, formatPercent } from '../../lib/format';
import { Icon } from '../ui/Icon';

type Active = { id: string; pinned: boolean } | null;

/**
 * Region outlines (r1…) with detail cards. Coordinates are normalised to the WL frame.
 * WCAG 1.4.13: the card appears on hover or focus, stays while the pointer is over it, closes
 * with Escape, and a click (or Enter) pins it open until Escape, the close button or a click
 * elsewhere.
 */
export function RegionMarkers({
  regions,
  threshold,
}: {
  regions: FVLRegion[];
  threshold?: number;
}) {
  const { t } = useTranslation();
  const [active, setActive] = useState<Active>(null);
  const closeTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const root = useRef<HTMLDivElement>(null);

  const cancelClose = () => {
    if (closeTimer.current) clearTimeout(closeTimer.current);
    closeTimer.current = null;
  };
  const show = (id: string) => {
    cancelClose();
    setActive((a) => (a?.pinned ? a : { id, pinned: false }));
  };
  // Grace period so the pointer can travel from the marker to its card.
  const hideSoon = useCallback(() => {
    cancelClose();
    closeTimer.current = setTimeout(() => setActive((a) => (a?.pinned ? a : null)), 450);
  }, []);
  const togglePin = (id: string) => {
    cancelClose();
    setActive((a) => (a?.id === id && a.pinned ? null : { id, pinned: true }));
  };

  useEffect(() => () => cancelClose(), []);

  useEffect(() => {
    if (!active) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setActive(null);
    };
    const onDown = (e: PointerEvent) => {
      if (!active.pinned) return;
      if (root.current && e.target instanceof Node && root.current.contains(e.target)) return;
      setActive(null);
    };
    document.addEventListener('keydown', onKey);
    document.addEventListener('pointerdown', onDown);
    return () => {
      document.removeEventListener('keydown', onKey);
      document.removeEventListener('pointerdown', onDown);
    };
  }, [active]);

  return (
    <div ref={root} className="contents">
      {regions.map((r) => {
        const open = active?.id === r.region_id;
        const pinned = open && active.pinned;
        // Dock the card in the image corner opposite the region so it never hides it or clips.
        const dockLeft = r.centroid.x >= 0.5;
        const cardId = `region-card-${r.region_id}`;
        return (
          <div key={r.region_id}>
            <div
              aria-hidden="true"
              className="pointer-events-none absolute rounded-[40%] border-2 border-dashed border-white/85 shadow-[0_0_0_1px_oklch(0_0_0/0.35)]"
              style={{
                left: `${r.bbox.x * 100}%`,
                top: `${r.bbox.y * 100}%`,
                width: `${r.bbox.w * 100}%`,
                height: `${r.bbox.h * 100}%`,
              }}
            />
            <button
              type="button"
              data-region-marker
              data-testid={`region-${r.region_id}`}
              aria-expanded={open}
              aria-pressed={pinned}
              aria-controls={open ? cardId : undefined}
              aria-describedby={open && !pinned ? cardId : undefined}
              aria-label={t('results.viewer.regionTitle', { id: r.region_id })}
              title={t('results.viewer.pinHint')}
              className={`absolute z-10 flex h-8 min-w-8 -translate-x-1/2 -translate-y-1/2 items-center justify-center rounded-full border-2 border-white px-2 text-xs font-bold text-white shadow-lg backdrop-blur hover:bg-violet focus-visible:bg-violet ${
                pinned ? 'bg-violet ring-2 ring-white/70' : 'bg-[oklch(0.25_0.05_290/0.85)]'
              }`}
              style={{ left: `${r.bbox.x * 100}%`, top: `${r.bbox.y * 100}%` }}
              onMouseEnter={() => show(r.region_id)}
              onMouseLeave={hideSoon}
              onFocus={() => show(r.region_id)}
              onBlur={hideSoon}
              onClick={() => togglePin(r.region_id)}
            >
              {r.region_id}
            </button>
            <AnimatePresence>
              {open ? (
                <m.div
                  id={cardId}
                  role={pinned ? 'dialog' : 'tooltip'}
                  aria-label={
                    pinned ? t('results.viewer.regionTitle', { id: r.region_id }) : undefined
                  }
                  data-region-marker
                  data-testid="region-card"
                  data-pinned={pinned}
                  initial={{ opacity: 0, scale: 0.96, x: dockLeft ? -6 : 6 }}
                  animate={{ opacity: 1, scale: 1, x: 0 }}
                  exit={{ opacity: 0, scale: 0.97 }}
                  transition={{ type: 'spring', stiffness: 420, damping: 32 }}
                  onMouseEnter={cancelClose}
                  onMouseLeave={hideSoon}
                  onPointerDown={(e) => e.stopPropagation()}
                  className={`absolute top-3 z-20 max-h-[calc(100%-1.5rem)] w-[min(16rem,calc(100%-1.5rem))] overflow-y-auto rounded-xl border bg-surface/95 p-3 text-xs text-fg shadow-card backdrop-blur-md ${
                    pinned ? 'border-violet' : 'border-line'
                  } ${dockLeft ? 'left-3' : 'right-3'}`}
                >
                  <div className="mb-2 flex items-start justify-between gap-2">
                    <p className="text-sm font-bold">
                      {t('results.viewer.regionTitle', { id: r.region_id })}
                    </p>
                    {pinned ? (
                      <button
                        type="button"
                        className="btn btn-ghost -mt-1 -mr-1 min-h-8 w-8 !p-0"
                        aria-label={t('results.viewer.closeCard')}
                        onClick={() => setActive(null)}
                      >
                        <Icon name="x" size={14} />
                      </button>
                    ) : null}
                  </div>
                  <dl className="grid grid-cols-[minmax(0,1fr)_auto] gap-x-3 gap-y-1 tabular">
                    <dt className="text-fg-subtle">{t('results.viewer.area')}</dt>
                    <dd className="text-right font-semibold">
                      {r.area_mm2 != null
                        ? `${formatNumber(r.area_mm2, 1)} mm²`
                        : t('results.viewer.areaFraction', {
                            value: formatNumber(r.area_fraction_of_mucosa * 100, 1),
                          })}
                    </dd>
                    <dt className="text-fg-subtle">{t('results.viewer.meanRelAf')}</dt>
                    <dd className="text-right font-semibold">
                      {formatNumber(r.mean_relative_af, 2)} ({formatPercent(r.mean_relative_af)})
                    </dd>
                    <dt className="text-fg-subtle">{t('results.viewer.minRelAf')}</dt>
                    <dd className="text-right font-semibold">
                      {formatNumber(r.min_relative_af, 2)}
                    </dd>
                    <dt className="text-fg-subtle">{t('results.viewer.contrast')}</dt>
                    <dd className="text-right font-semibold">
                      {formatNumber(r.contrast_vs_surround, 2)}×
                    </dd>
                    <dt className="text-fg-subtle">{t('results.viewer.border')}</dt>
                    <dd className="text-right font-semibold">
                      {formatNumber(r.border_irregularity, 2)}
                    </dd>
                  </dl>
                  {threshold !== undefined ? (
                    <p className="mt-2 text-fg-subtle">
                      {t('results.viewer.threshold', { value: threshold })}
                    </p>
                  ) : null}
                  <p className="mt-2 font-semibold text-fg-muted">
                    {t('results.viewer.confounders')}
                  </p>
                  {(r.confounders ?? []).length === 0 ? (
                    <p className="text-fg-subtle">{t('results.viewer.noConfounders')}</p>
                  ) : (
                    <ul className="mt-0.5 list-disc pl-4 text-warn">
                      {(r.confounders ?? []).map((c) => (
                        <li key={c.type}>
                          {t(`results.confounder.${c.type}`)} ·{' '}
                          {t('results.confounder.overlap', {
                            value: Math.round(c.overlap_fraction * 100),
                          })}
                        </li>
                      ))}
                    </ul>
                  )}
                </m.div>
              ) : null}
            </AnimatePresence>
          </div>
        );
      })}
    </div>
  );
}
