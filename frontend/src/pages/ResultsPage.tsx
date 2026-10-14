import { useCallback, useMemo, useRef, useState, type ReactNode } from 'react';
import { useTranslation } from 'react-i18next';
import { Link, useParams, useSearchParams } from 'react-router';
import { useApi } from '../api/ApiContext';
import { useReanalyze, useResultVersion, useScreening, useSiteCatalogue } from '../api/queries';
import type { OralSite, RiskFactors, ScreeningResult } from '../api/types';
import { AnalysisOverlay } from '../components/analysis/AnalysisOverlay';
import { SyntheticRibbon } from '../components/layout/SyntheticRibbon';
import { FactorBars } from '../components/results/FactorBars';
import { LesionList } from '../components/results/LesionList';
import { QcPanel } from '../components/results/QcPanel';
import { ResultActions, VersionSwitcher } from '../components/results/ResultActions';
import { RiskPill } from '../components/results/RiskBadge';
import { RiskHeader } from '../components/results/RiskHeader';
import { SiteViewer } from '../components/results/SiteViewer';
import { SpectralPanel } from '../components/results/SpectralPanel';
import { VersionsFooter } from '../components/results/VersionsFooter';
import { Card } from '../components/ui/Card';
import { Icon } from '../components/ui/Icon';
import { SiteMap, type SiteStatus } from '../components/viz/SiteMap';
import { errorMessage } from '../lib/errors';
import { formatDateTime } from '../lib/format';
import { RISK_LEVELS } from '../lib/risk';
import { usePrefersReducedMotion } from '../lib/useMediaQuery';

function defaultSite(r: ScreeningResult): OralSite | null {
  const top = r.lesions?.find((l) => l.lesion_id === r.triage.highest_risk_lesion_id);
  if (top) return top.site;
  const examined = r.sites.find((s) => s.role !== 'reference_only');
  return examined?.site ?? r.sites[0]?.site ?? null;
}

function riskFactorChips(rf: RiskFactors, t: ReturnType<typeof useTranslation>['t']): string[] {
  const out: string[] = [];
  const habit = (uses: boolean, former: boolean | undefined, label: string, types?: string[]) => {
    if (uses) out.push(types && types.length ? `${label} (${types.join(', ')})` : label);
    else if (former) out.push(`${label} · ${t('results.factor.former')}`);
  };
  habit(
    rf.smokeless_tobacco?.uses ?? false,
    rf.smokeless_tobacco?.former,
    t('results.factor.smokeless'),
    rf.smokeless_tobacco?.types,
  );
  habit(
    rf.smoking?.uses ?? false,
    rf.smoking?.former,
    t('results.factor.smoking'),
    rf.smoking?.types,
  );
  habit(rf.areca_nut?.uses ?? false, rf.areca_nut?.former, t('results.factor.areca'));
  if (rf.alcohol?.uses)
    out.push(
      `${t('results.factor.alcohol')} (${t(`results.factor.alcoholFrequency.${rf.alcohol.frequency ?? 'occasional'}`)})`,
    );
  if (rf.prior_oral_lesion) out.push(t('results.factor.priorLesion'));
  if (rf.prior_oral_cancer) out.push(t('results.factor.priorCancer'));
  return out;
}

export default function ResultsPage() {
  const { t } = useTranslation();
  const { id = '' } = useParams();
  const api = useApi();
  const [params, setParams] = useSearchParams();
  const catalogue = useSiteCatalogue();
  const reduced = usePrefersReducedMotion();
  const detail = useScreening(id);
  const latest = detail.data?.latest;
  const vParam = Number(params.get('v')) || undefined;
  const wantOld = vParam !== undefined && latest !== undefined && vParam !== latest.result_version;
  const old = useResultVersion(id, vParam, wantOld);
  const result = wantOld ? old.data : latest;

  const reanalyse = useReanalyze(id);
  const [overlay, setOverlay] = useState(false);

  const setVersion = useCallback(
    (v: number) => {
      const next = new URLSearchParams(params);
      if (latest && v === latest.result_version) next.delete('v');
      else next.set('v', String(v));
      setParams(next, { replace: true });
    },
    [latest, params, setParams],
  );

  const startReanalyse = () => {
    setOverlay(true);
    reanalyse.mutate();
  };
  const onReanalysed = useCallback(() => {
    setOverlay(false);
    const next = new URLSearchParams(params);
    next.delete('v');
    setParams(next, { replace: true });
    reanalyse.reset();
  }, [params, reanalyse, setParams]);

  if (detail.isPending || (wantOld && old.isPending)) {
    return (
      <div className="mx-auto max-w-7xl px-4 py-10 sm:px-6" aria-busy="true">
        <p className="sr-only">{t('results.loading')}</p>
        <div className="h-10 w-72 animate-pulse rounded-xl bg-surface-2" />
        <div className="mt-6 h-64 animate-pulse rounded-[1.25rem] bg-surface-2" />
        <div className="mt-5 grid gap-5 lg:grid-cols-2">
          <div className="h-80 animate-pulse rounded-[1.25rem] bg-surface-2" />
          <div className="h-80 animate-pulse rounded-[1.25rem] bg-surface-2" />
        </div>
      </div>
    );
  }
  if (detail.isError || !result || !detail.data) {
    return (
      <div className="mx-auto max-w-3xl px-4 py-16 text-center sm:px-6" role="alert">
        <Icon name="alert" size={36} className="mx-auto text-warn" />
        <h1 className="mt-4 text-2xl font-bold">{t('common.error')}</h1>
        <p className="mt-2 text-fg-muted">{errorMessage(detail.error ?? old.error, t)}</p>
        <div className="mt-6 flex justify-center gap-3">
          <button type="button" className="btn btn-secondary" onClick={() => void detail.refetch()}>
            <Icon name="refresh" size={16} />
            {t('common.retry')}
          </button>
          <Link to="/" className="btn btn-ghost">
            {t('app.backHome')}
          </Link>
        </div>
      </div>
    );
  }

  return (
    <ResultView
      key={result.result_id}
      result={result}
      latestVersion={latest!.result_version}
      versions={detail.data.versions}
      catalogue={catalogue}
      reportUrl={api.reportUrl(result.screening_id, result.result_version)}
      onVersion={setVersion}
      onReanalyse={startReanalyse}
      reanalysing={reanalyse.isPending}
      reduced={reduced}
      overlay={
        <AnalysisOverlay
          open={overlay}
          mode="reanalyse"
          result={reanalyse.data?.latest ?? null}
          error={reanalyse.isError ? errorMessage(reanalyse.error, t) : null}
          onContinue={onReanalysed}
          onClose={() => {
            setOverlay(false);
            reanalyse.reset();
          }}
        />
      }
    />
  );
}

function ResultView({
  result,
  latestVersion,
  versions,
  catalogue,
  reportUrl,
  onVersion,
  onReanalyse,
  reanalysing,
  reduced,
  overlay,
}: {
  result: ScreeningResult;
  latestVersion: number;
  versions: Parameters<typeof VersionSwitcher>[0]['versions'];
  catalogue: ReturnType<typeof useSiteCatalogue>;
  reportUrl: string;
  onVersion: (v: number) => void;
  onReanalyse: () => void;
  reanalysing: boolean;
  reduced: boolean;
  overlay: ReactNode;
}) {
  const { t } = useTranslation();
  const [selected, setSelected] = useState<OralSite | null>(() => defaultSite(result));
  const viewerRef = useRef<HTMLDivElement>(null);
  const site = result.sites.find((s) => s.site === selected) ?? result.sites[0];
  const statuses = useMemo(() => {
    const out: Partial<Record<OralSite, SiteStatus>> = {};
    for (const s of result.sites)
      out[s.site] = { level: s.site_risk_level, role: s.role ?? 'examined' };
    return out;
  }, [result.sites]);
  const factors = riskFactorChips(result.risk_factors, t);
  const p = result.patient;

  const selectSite = (s: OralSite) => {
    setSelected(s);
    if (typeof window !== 'undefined' && window.innerWidth < 1024) {
      viewerRef.current?.scrollIntoView({ behavior: reduced ? 'auto' : 'smooth', block: 'start' });
    }
  };

  return (
    <div
      className="mx-auto max-w-7xl px-4 pt-6 pb-16 sm:px-6"
      data-testid="results-page"
      data-synthetic={result.synthetic}
    >
      <div className="flex flex-col gap-4">
        <Link to="/screenings" className="btn btn-ghost -ml-3 self-start px-3 text-sm">
          <Icon name="arrowLeft" size={16} />
          {t('app.nav.screenings')}
        </Link>
        {result.synthetic ? <SyntheticRibbon notice={result.notices?.synthetic_notice} /> : null}

        {/* Mobile: patient → risk → actions. Desktop: patient | actions, then the risk header. */}
        <div className="flex flex-col gap-4 lg:grid lg:grid-cols-[minmax(0,1fr)_auto] lg:items-end">
          <header className="min-w-0 lg:col-start-1 lg:row-start-1">
            <p className="eyebrow">{t('results.pageTitle')}</p>
            <h1
              className="mt-1.5 text-2xl font-extrabold tracking-tight tabular sm:text-3xl"
              data-testid="patient-id"
            >
              {t('results.patient', { id: p.pseudonymous_id })}
            </h1>
            <p className="mt-1 flex flex-wrap gap-x-3 gap-y-1 text-sm text-fg-muted tabular">
              <span>
                {t('results.patientMeta', {
                  age:
                    p.age_years != null
                      ? t('common.years', { count: p.age_years })
                      : t('results.ageUnknown'),
                  sex: t(`upload.sex.${p.sex ?? 'undisclosed'}`),
                })}
              </span>
              <span aria-hidden="true">·</span>
              <span>
                {t('results.capturedAt', {
                  date: formatDateTime(result.context.captured_at ?? result.created_at),
                })}
              </span>
              {result.context.screening_site_name ? (
                <>
                  <span aria-hidden="true">·</span>
                  <span>{t('results.camp', { name: result.context.screening_site_name })}</span>
                </>
              ) : null}
            </p>
            <div className="mt-2.5 flex flex-wrap items-center gap-1.5">
              <span className="mr-1 text-xs font-semibold text-fg-subtle">
                {t('results.patientFactors')}
              </span>
              {factors.length === 0 ? (
                <span className="chip">{t('results.factor.none')}</span>
              ) : (
                factors.map((f) => (
                  <span key={f} className="chip">
                    {f}
                  </span>
                ))
              )}
            </div>
          </header>
          <div className="order-3 flex flex-col items-start gap-3 lg:order-none lg:col-start-2 lg:row-start-1 lg:items-end">
            <VersionSwitcher
              versions={versions}
              current={result.result_version}
              latest={latestVersion}
              onChange={onVersion}
            />
            <ResultActions
              reportUrl={reportUrl}
              onReanalyse={onReanalyse}
              reanalysing={reanalysing}
            />
          </div>

          {result.result_version !== latestVersion ? (
            <div
              className="order-1 flex flex-wrap items-center gap-3 rounded-xl border border-warn/60 bg-risk-moderate-bg p-3 text-sm lg:order-none lg:col-span-2"
              role="status"
            >
              <Icon name="history" size={18} className="text-warn" />
              <span className="flex-1">
                {t('results.superseded', { version: result.result_version, latest: latestVersion })}
              </span>
              <button
                type="button"
                className="btn btn-secondary min-h-9 py-1 text-sm"
                onClick={() => onVersion(latestVersion)}
              >
                {t('results.viewLatest')}
              </button>
            </div>
          ) : null}

          <div className="order-2 lg:order-none lg:col-span-2">
            <RiskHeader result={result} />
          </div>
        </div>
      </div>

      <div className="mt-5 flex flex-col gap-5 lg:grid lg:grid-cols-[minmax(0,1.25fr)_minmax(0,1fr)] lg:items-start">
        <div className="contents lg:flex lg:flex-col lg:gap-5">
          <Card
            id="why"
            title={t('results.sections.why')}
            icon="layers"
            className="order-1 lg:order-none"
            subtitle={t('results.why.intro')}
          >
            {result.triage.overall_risk_level === 'INCONCLUSIVE' &&
            (result.triage.contributing_factors ?? []).length === 0 ? (
              <p className="rounded-xl border border-dashed border-risk-inconclusive p-4 text-sm text-fg-muted">
                {t('results.why.noScore')}
              </p>
            ) : (
              <FactorBars
                factors={result.triage.contributing_factors}
                thresholds={result.triage.thresholds}
                score={result.triage.score}
                level={result.triage.overall_risk_level}
              />
            )}
          </Card>

          <div ref={viewerRef} className="order-3 scroll-mt-28 lg:order-none">
            <Card id="viewer" title={t('results.sections.viewer')} icon="eye">
              {site ? <SiteViewer key={site.site} result={result} site={site} /> : null}
            </Card>
          </div>

          <Card
            id="spectral"
            title={t('results.sections.spectral')}
            icon="wave"
            className="order-5 lg:order-none"
          >
            <SpectralPanel result={result} />
          </Card>
        </div>

        <div className="contents lg:flex lg:flex-col lg:gap-5">
          <Card
            id="map"
            title={t('results.sections.map')}
            icon="pin"
            className="order-2 lg:order-none"
            subtitle={t('results.map.hint')}
          >
            <SiteMap
              catalogue={catalogue}
              mode="results"
              selected={site?.site ?? null}
              statuses={statuses}
              onSelect={selectSite}
              label={t('results.map.label')}
              className="mx-auto max-w-md"
            />
            <ul className="mt-4 flex flex-wrap gap-2" aria-label={t('results.map.legend')}>
              {RISK_LEVELS.map((l) => (
                <li key={l}>
                  <RiskPill level={l} />
                </li>
              ))}
              <li>
                <span className="inline-flex items-center gap-1.5 rounded-full border-[1.5px] border-dashed border-ref px-2.5 py-1 text-xs font-bold tracking-wide text-ref uppercase">
                  <Icon name="target" size={13} />
                  {t('results.map.reference')}
                </span>
              </li>
            </ul>
          </Card>

          <Card
            id="lesions"
            title={t('results.sections.lesions')}
            icon="target"
            className="order-4 lg:order-none"
          >
            <LesionList
              result={result}
              onSelectSite={selectSite}
              selectedSite={site?.site ?? null}
            />
          </Card>

          <Card
            id="qc"
            title={t('results.sections.qc')}
            icon="shield"
            className="order-6 lg:order-none"
          >
            <QcPanel result={result} />
          </Card>
        </div>
      </div>

      <div className="mt-5">
        <Card id="versions" title={t('results.sections.versions')} icon="history">
          <VersionsFooter result={result} />
        </Card>
      </div>

      {overlay}
    </div>
  );
}
