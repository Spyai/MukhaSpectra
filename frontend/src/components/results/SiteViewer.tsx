import { lazy, Suspense, useEffect, useRef, useState, type KeyboardEvent } from 'react';
import { useTranslation } from 'react-i18next';
import { useApi } from '../../api/ApiContext';
import { useArtifactJson } from '../../api/queries';
import type { RelativeAfGrid, ScreeningResult, SiteResult } from '../../api/types';
import { LOSS_GRADIENT_CSS } from '../../lib/colormap';
import { formatNumber } from '../../lib/format';
import { useTheme } from '../../lib/theme';
import { usePrefersReducedMotion } from '../../lib/useMediaQuery';
import { hasWebGL } from '../../lib/webgl';
import { SyntheticTag } from '../layout/SyntheticRibbon';
import { Icon } from '../ui/Icon';
import { RegionMarkers } from '../viz/RegionMarkers';
import { WipeSlider } from '../viz/WipeSlider';
import { RiskPill } from './RiskBadge';

const Terrain3D = lazy(() => import('../../three/Terrain3D'));

type Tab = 'wipe' | 'terrain';

export function SiteViewer({ result, site }: { result: ScreeningResult; site: SiteResult }) {
  const { t } = useTranslation();
  const api = useApi();
  const [tab, setTab] = useState<Tab>('wipe');
  const [heat, setHeat] = useState(false);
  const [opacity, setOpacity] = useState(70);
  const tabRefs = useRef<Record<Tab, HTMLButtonElement | null>>({ wipe: null, terrain: null });

  const art = site.artifacts ?? {};
  const url = (key: string | null | undefined) =>
    key ? api.artifactUrl(result.screening_id, key, result.result_version) : null;
  const wl = url(art.wl_image);
  const af = url(art.af_registered ?? art.af_image);
  const heatmap = url(art.heatmap_overlay);
  const hasGrid = Boolean(art.relative_af_grid);
  const wr = site.working_resolution;
  const aspect = wr && wr.width_px > 0 && wr.height_px > 0 ? wr.width_px / wr.height_px : 4 / 3;
  const siteName = t(`sites.${site.site}`);
  const regions = site.af?.status === 'ok' ? (site.af.regions ?? []) : [];
  const threshold = site.af?.fvl_threshold;

  const tabs: Tab[] = hasGrid ? ['wipe', 'terrain'] : ['wipe'];
  const onTabKey = (e: KeyboardEvent<HTMLButtonElement>) => {
    if (e.key !== 'ArrowRight' && e.key !== 'ArrowLeft' && e.key !== 'Home' && e.key !== 'End')
      return;
    e.preventDefault();
    const i = tabs.indexOf(tab);
    const next =
      e.key === 'Home'
        ? tabs[0]!
        : e.key === 'End'
          ? tabs[tabs.length - 1]!
          : tabs[(i + (e.key === 'ArrowRight' ? 1 : -1) + tabs.length) % tabs.length]!;
    setTab(next);
    tabRefs.current[next]?.focus();
  };

  return (
    <div data-testid="site-viewer" data-site={site.site}>
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex min-w-0 flex-wrap items-center gap-2">
          <h3 className="text-base font-bold">{siteName}</h3>
          {site.role === 'reference_only' ? (
            <span className="chip border-dashed border-ref text-ref">
              {t('results.map.reference')}
            </span>
          ) : (
            <RiskPill level={site.site_risk_level} />
          )}
        </div>
        {tabs.length > 1 ? (
          <div
            role="tablist"
            aria-label={t('results.sections.viewer')}
            className="flex rounded-xl border border-line bg-surface-2 p-1"
          >
            {tabs.map((id) => (
              <button
                key={id}
                ref={(el) => {
                  tabRefs.current[id] = el;
                }}
                type="button"
                role="tab"
                id={`tab-${id}`}
                aria-selected={tab === id}
                aria-controls={`panel-${id}`}
                tabIndex={tab === id ? 0 : -1}
                onClick={() => setTab(id)}
                onKeyDown={onTabKey}
                data-testid={`tab-${id}`}
                className={`flex min-h-10 items-center gap-1.5 rounded-lg px-3 text-sm font-semibold transition-colors ${
                  tab === id ? 'bg-surface text-fg shadow-card' : 'text-fg-muted hover:text-fg'
                }`}
              >
                <Icon name={id === 'wipe' ? 'layers' : 'cube'} size={16} />
                {id === 'wipe' ? t('results.viewer.wipe') : t('results.viewer.terrain')}
              </button>
            ))}
          </div>
        ) : null}
      </div>

      {site.role === 'reference_only' ? (
        <p className="mt-3 flex items-start gap-2 text-sm text-fg-muted">
          <Icon name="target" size={16} className="mt-0.5 shrink-0 text-ref" />
          {t('results.viewer.referenceSite', {
            site: site.paired_site ? t(`sites.${site.paired_site}`) : '—',
          })}
        </p>
      ) : null}
      {site.analysis_status !== 'analysed' ? (
        <p className="mt-3 flex items-start gap-2 rounded-xl border border-dashed border-risk-inconclusive bg-risk-inconclusive-bg p-3 text-sm">
          <Icon name="alert" size={16} className="mt-0.5 shrink-0" />
          {t('results.viewer.qcFailedSite')}
        </p>
      ) : null}

      <div className="mt-4">
        {tab === 'wipe' ? (
          <div
            role={tabs.length > 1 ? 'tabpanel' : undefined}
            id="panel-wipe"
            aria-labelledby={tabs.length > 1 ? 'tab-wipe' : undefined}
          >
            {wl && af ? (
              <WipeSlider
                leftSrc={wl}
                rightSrc={af}
                leftAlt={t('results.viewer.imageAlt', {
                  kind: t('results.viewer.wl'),
                  site: siteName,
                })}
                rightAlt={t('results.viewer.imageAlt', {
                  kind: t('results.viewer.af'),
                  site: siteName,
                })}
                leftLabel={t('results.viewer.wl')}
                rightLabel={t('results.viewer.af')}
                aspect={aspect}
                overlaySrc={heat ? heatmap : null}
                overlayOpacity={opacity / 100}
              >
                <RegionMarkers regions={regions} threshold={threshold} />
                {result.synthetic ? (
                  <SyntheticTag
                    solid
                    className="pointer-events-none absolute right-2 bottom-2 z-10"
                  />
                ) : null}
              </WipeSlider>
            ) : (
              <p className="flex aspect-[4/3] items-center justify-center rounded-xl bg-image-well p-6 text-center text-sm text-[oklch(0.8_0.02_268)]">
                {t('results.viewer.noImages')}
              </p>
            )}

            {heatmap ? (
              <div className="mt-4 flex flex-col gap-3">
                <div className="flex flex-wrap items-center gap-x-6 gap-y-3">
                  <label className="inline-flex min-h-11 cursor-pointer items-center gap-2.5 text-sm font-semibold">
                    <span className="relative inline-flex">
                      <input
                        type="checkbox"
                        role="switch"
                        aria-checked={heat}
                        checked={heat}
                        onChange={(e) => setHeat(e.target.checked)}
                        className="peer sr-only"
                        data-testid="heatmap-toggle"
                      />
                      <span className="h-6 w-11 rounded-full bg-surface-3 ring-1 ring-line-strong transition-colors ring-inset peer-checked:bg-violet peer-checked:ring-violet peer-focus-visible:outline-2 peer-focus-visible:outline-offset-2 peer-focus-visible:outline-focus" />
                      <span className="absolute top-0.5 left-0.5 h-5 w-5 rounded-full bg-white shadow transition-transform peer-checked:translate-x-5" />
                    </span>
                    {t('results.viewer.heatmap')}
                  </label>
                  {heat ? (
                    <label className="flex min-w-48 flex-1 items-center gap-3 text-sm">
                      <span className="shrink-0 font-medium text-fg-muted">
                        {t('results.viewer.opacity')}
                      </span>
                      <input
                        type="range"
                        min={0}
                        max={100}
                        step={5}
                        value={opacity}
                        onChange={(e) => setOpacity(Number(e.target.value))}
                        className="range h-6 w-full"
                        aria-valuetext={`${opacity}%`}
                        data-testid="heatmap-opacity"
                      />
                      <span className="w-10 text-right text-fg-muted tabular">{opacity}%</span>
                    </label>
                  ) : null}
                </div>
                {heat ? <LossLegend threshold={threshold} testId="heatmap-legend" /> : null}
              </div>
            ) : null}
          </div>
        ) : (
          <div role="tabpanel" id="panel-terrain" aria-labelledby="tab-terrain">
            <TerrainPanel result={result} site={site} />
          </div>
        )}
      </div>

      <ul className="mt-4 grid grid-cols-1 gap-x-6 gap-y-1.5 text-xs text-fg-subtle sm:grid-cols-2">
        {site.af ? (
          <li>
            {t('results.viewer.reference', {
              type: t(`results.refType.${site.af.reference.type}`),
            })}
            {site.af.reference.reference_site
              ? ` · ${t(`sites.${site.af.reference.reference_site}`)}`
              : ''}
          </li>
        ) : null}
        {site.registration ? (
          <li>
            {t('results.viewer.registration', {
              method: t(`results.viewer.registrationMethod.${site.registration.method}`, {
                defaultValue: site.registration.method.replace(/_/g, ' '),
              }),
              value: formatNumber(site.registration.confidence, 2),
            })}
          </li>
        ) : null}
        <li className="sm:col-span-2">{t('results.viewer.orientation')}</li>
      </ul>
    </div>
  );
}

function TerrainPanel({ result, site }: { result: ScreeningResult; site: SiteResult }) {
  const { t } = useTranslation();
  const reduced = usePrefersReducedMotion();
  const [theme] = useTheme();
  const [webgl, setWebgl] = useState<boolean | null>(null);
  useEffect(() => setWebgl(hasWebGL()), []);
  const q = useArtifactJson<RelativeAfGrid>(
    result.screening_id,
    site.artifacts?.relative_af_grid,
    result.result_version,
  );
  const threshold = q.data?.threshold ?? site.af?.fvl_threshold ?? 0.75;
  const regions = site.af?.status === 'ok' ? (site.af.regions ?? []) : [];

  return (
    <div>
      <div className="relative h-[320px] overflow-hidden rounded-xl border border-line bg-[radial-gradient(80%_80%_at_50%_30%,var(--surface-2),var(--surface))] sm:h-[420px]">
        {webgl === false ? (
          <p className="flex h-full items-center justify-center p-6 text-center text-sm text-fg-muted">
            {t('results.viewer.terrainUnavailable')}
          </p>
        ) : q.isPending || webgl === null ? (
          <p
            className="flex h-full items-center justify-center text-sm text-fg-muted"
            aria-busy="true"
          >
            {t('results.viewer.loadingTerrain')}
          </p>
        ) : q.isError || !q.data ? (
          <p className="flex h-full items-center justify-center p-6 text-center text-sm text-fg-muted">
            {t('results.viewer.noGrid')}
          </p>
        ) : (
          <Suspense
            fallback={
              <p className="flex h-full items-center justify-center text-sm text-fg-muted">
                {t('results.viewer.loadingTerrain')}
              </p>
            }
          >
            <Terrain3D
              grid={q.data}
              threshold={threshold}
              reducedMotion={reduced}
              dark={theme === 'dark'}
              label={t('results.viewer.terrainLabel')}
              regions={regions}
              topLabel={t('results.viewer.topOfPhoto')}
            />
          </Suspense>
        )}
        {result.synthetic ? (
          <SyntheticTag solid className="pointer-events-none absolute bottom-2 left-2 z-10" />
        ) : null}
      </div>
      <LossLegend threshold={threshold} plane className="mt-3" />
      <p className="mt-2 text-xs text-fg-subtle">{t('results.viewer.terrainHint')}</p>
    </div>
  );
}

/** Colour key for AF loss, shared by the 2D heatmap and the 3D terrain. */
function LossLegend({
  threshold,
  plane = false,
  className = '',
  testId,
}: {
  threshold?: number;
  plane?: boolean;
  className?: string;
  testId?: string;
}) {
  const { t } = useTranslation();
  return (
    <div
      className={`flex flex-wrap items-center gap-x-6 gap-y-2 text-xs text-fg-muted ${className}`}
      data-testid={testId}
    >
      <div className="flex items-center gap-2">
        <span className="sr-only">{t('results.viewer.heatmapLegend')}:</span>
        <span>{t('results.viewer.lossLegendLow')}</span>
        <span
          className="h-2.5 w-32 rounded-full"
          style={{ background: LOSS_GRADIENT_CSS }}
          aria-hidden="true"
        />
        <span>{t('results.viewer.lossLegendHigh')}</span>
      </div>
      {threshold !== undefined ? (
        <div className="flex items-center gap-2">
          {plane ? (
            <span
              className="h-3 w-5 rounded-sm border border-[#a48bff] bg-[#7c5cff]/25"
              aria-hidden="true"
            />
          ) : null}
          <span>{t('results.viewer.threshold', { value: threshold })}</span>
        </div>
      ) : null}
    </div>
  );
}
