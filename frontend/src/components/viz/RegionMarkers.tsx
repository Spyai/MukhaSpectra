import { AnimatePresence, m } from 'motion/react';
import { useState } from 'react';
import { useTranslation } from 'react-i18next';
import type { FVLRegion } from '../../api/types';
import { formatNumber, formatPercent } from '../../lib/format';

/** Region outlines (r1…) with hover/focus cards. Coordinates are normalised to the WL frame. */
export function RegionMarkers({ regions, threshold }: { regions: FVLRegion[]; threshold?: number }) {
  const { t } = useTranslation();
  const [active, setActive] = useState<string | null>(null);

  return (
    <>
      {regions.map((r) => {
        const open = active === r.region_id;
        const right = r.centroid.x > 0.55;
        const below = r.bbox.y < 0.3;
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
              aria-describedby={open ? `region-card-${r.region_id}` : undefined}
              aria-label={t('results.viewer.regionTitle', { id: r.region_id })}
              className="absolute z-10 flex h-8 min-w-8 -translate-x-1/2 -translate-y-1/2 items-center justify-center rounded-full border-2 border-white bg-[oklch(0.25_0.05_290/0.85)] px-2 text-xs font-bold text-white shadow-lg backdrop-blur hover:bg-violet focus-visible:bg-violet"
              style={{ left: `${r.bbox.x * 100}%`, top: `${r.bbox.y * 100}%` }}
              onMouseEnter={() => setActive(r.region_id)}
              onMouseLeave={() => setActive((a) => (a === r.region_id ? null : a))}
              onFocus={() => setActive(r.region_id)}
              onBlur={() => setActive((a) => (a === r.region_id ? null : a))}
              onClick={() => setActive((a) => (a === r.region_id ? null : r.region_id))}
              onKeyDown={(e) => {
                if (e.key === 'Escape') setActive(null);
              }}
            >
              {r.region_id}
            </button>
            <AnimatePresence>
              {open ? (
                <m.div
                  id={`region-card-${r.region_id}`}
                  role="tooltip"
                  initial={{ opacity: 0, scale: 0.96, y: below ? -4 : 4 }}
                  animate={{ opacity: 1, scale: 1, y: 0 }}
                  exit={{ opacity: 0, scale: 0.97 }}
                  transition={{ type: 'spring', stiffness: 420, damping: 32 }}
                  className="pointer-events-none absolute z-20 w-56 rounded-xl border border-line bg-surface/95 p-3 text-xs text-fg shadow-card backdrop-blur-md"
                  style={{
                    left: right ? undefined : `calc(${r.bbox.x * 100}% + 18px)`,
                    right: right ? `calc(${(1 - r.bbox.x) * 100}% + 18px)` : undefined,
                    top: below ? `calc(${r.bbox.y * 100}% + 8px)` : undefined,
                    bottom: below ? undefined : `calc(${(1 - r.bbox.y) * 100}% - 32px)`,
                  }}
                >
                  <p className="mb-2 text-sm font-bold">{t('results.viewer.regionTitle', { id: r.region_id })}</p>
                  <dl className="grid grid-cols-[1fr_auto] gap-x-3 gap-y-1 tabular">
                    <dt className="text-fg-subtle">{t('results.viewer.area')}</dt>
                    <dd className="text-right font-semibold">
                      {r.area_mm2 != null
                        ? `${formatNumber(r.area_mm2, 1)} mm²`
                        : t('results.viewer.areaFraction', { value: (r.area_fraction_of_mucosa * 100).toFixed(1) })}
                    </dd>
                    <dt className="text-fg-subtle">{t('results.viewer.meanRelAf')}</dt>
                    <dd className="text-right font-semibold">
                      {formatNumber(r.mean_relative_af, 2)} ({formatPercent(r.mean_relative_af)})
                    </dd>
                    <dt className="text-fg-subtle">{t('results.viewer.minRelAf')}</dt>
                    <dd className="text-right font-semibold">{formatNumber(r.min_relative_af, 2)}</dd>
                    <dt className="text-fg-subtle">{t('results.viewer.contrast')}</dt>
                    <dd className="text-right font-semibold">{formatNumber(r.contrast_vs_surround, 2)}×</dd>
                    <dt className="text-fg-subtle">{t('results.viewer.border')}</dt>
                    <dd className="text-right font-semibold">{formatNumber(r.border_irregularity, 2)}</dd>
                  </dl>
                  {threshold !== undefined ? (
                    <p className="mt-2 text-fg-subtle">{t('results.viewer.threshold', { value: threshold })}</p>
                  ) : null}
                  <p className="mt-2 font-semibold text-fg-muted">{t('results.viewer.confounders')}</p>
                  {(r.confounders ?? []).length === 0 ? (
                    <p className="text-fg-subtle">{t('results.viewer.noConfounders')}</p>
                  ) : (
                    <ul className="mt-0.5 list-disc pl-4 text-warn">
                      {(r.confounders ?? []).map((c) => (
                        <li key={c.type}>
                          {t(`results.confounder.${c.type}`)} ·{' '}
                          {t('results.confounder.overlap', { value: Math.round(c.overlap_fraction * 100) })}
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
    </>
  );
}
