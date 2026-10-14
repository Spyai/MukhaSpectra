import { useCallback, useMemo, useRef, useState, type MouseEvent, type ReactNode } from 'react';
import { useTranslation } from 'react-i18next';
import { Link, useParams, useSearchParams } from 'react-router';
import { useApi } from '../api/ApiContext';
import { useReanalyze, useResultVersion, useScreening, useSiteCatalogue } from '../api/queries';
import type { ContributingFactor, OralSite, RiskFactors, ScreeningResult } from '../api/types';
import { AnalysisOverlay } from '../components/analysis/AnalysisOverlay';
import { SyntheticRibbon } from '../components/layout/SyntheticRibbon';
import { FactorBars } from '../components/results/FactorBars';
import { LesionList } from '../components/results/LesionList';
import { QcPanel } from '../components/results/QcPanel';
import { RetakeCallout } from '../components/results/RetakeCallout';
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
import { formatDateTime, formatPoints } from '../lib/format';
import { failedSites } from '../lib/qc';
import { wlThumbnail } from '../lib/thumbnail';
import { RISK_LEVELS } from '../lib/risk';
import { useMediaQuery, usePrefersReducedMotion } from '../lib/useMediaQuery';

function defaultSite(r: ScreeningResult): OralSite | null {
  const top = r.lesions?.find((l) => l.lesion_id === r.triage.highest_risk_lesion_id);
  if (top) return top.site;
  const examined = r.sites.find((s) => s.role !== 'reference_only');
  return examined?.site ?? r.sites[0]?.site ?? null;
}

type T = ReturnType<typeof useTranslation>['t'];

function riskFactorChips(rf: RiskFactors, t: T): string[] {
  const out: string[] = [];
  const habit = (
    use:
      | { uses?: boolean; former?: boolean; years?: number | null; per_day?: number | null }
      | undefined,
    label: string,
    types: string[] = [],
  ) => {
    if (!use || (!use.uses && !use.former)) return;
    const detail = [
      ...types.map((x) => t(`upload.product.${x as 'gutka'}`, { defaultValue: x })),
      ...(use.years != null ? [t('results.factor.yearsShort', { count: use.years })] : []),
      ...(use.per_day != null ? [t('results.factor.perDayShort', { count: use.per_day })] : []),
    ];
    const head = use.uses ? label : `${label} · ${t('results.factor.former')}`;
    out.push(detail.length ? `${head} (${detail.join(' · ')})` : head);
  };
  habit(rf.smokeless_tobacco, t('results.factor.smokeless'), rf.smokeless_tobacco?.types);
  habit(rf.smoking, t('results.factor.smoking'), rf.smoking?.types);
  habit(rf.areca_nut, t('results.factor.areca'));
  if (rf.alcohol?.uses)
    out.push(
      `${t('results.factor.alcohol')} (${t(`results.factor.alcoholFrequency.${rf.alcohol.frequency ?? 'occasional'}`)})`,
    );
  if (rf.prior_oral_lesion) out.push(t('results.factor.priorLesion'));
  if (rf.prior_oral_cancer) out.push(t('results.factor.priorCancer'));
  return out;
}

const RISK_FACTOR_SOURCES = new Set<ContributingFactor['source']>(['clinical_history', 'symptom']);

/** The pseudonymous ID, with a copy button and the hand-off instruction. */
function PatientId({ id }: { id: string }) {
  const { t } = useTranslation();
  const [copied, setCopied] = useState(false);
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(id);
      setCopied(true);
      setTimeout(() => setCopied(false), 1600);
    } catch {
      /* clipboard unavailable: the ID is visible to copy by hand */
    }
  };
  return (
    <div className="mt-1.5">
      <div className="flex flex-wrap items-center gap-x-3 gap-y-2">
        <h1 className="text-2xl font-extrabold tracking-tight sm:text-3xl" data-testid="patient-id">
          {t('results.patientLabel')} <span className="id-text whitespace-nowrap">{id}</span>
        </h1>
        <button
          type="button"
          className="btn btn-secondary min-h-10 min-w-10 px-2.5 py-1.5 text-sm sm:px-3"
          onClick={() => void copy()}
          aria-label={copied ? t('results.copiedId') : t('results.copyId')}
          data-testid="copy-id"
        >
          <Icon name={copied ? 'check' : 'copy'} size={16} />
          <span className="hidden sm:inline" aria-hidden="true">
            {copied ? t('results.copiedId') : t('results.copyId')}
          </span>
        </button>
        <span className="sr-only" aria-live="polite">
          {copied ? t('results.copiedId') : ''}
        </span>
      </div>
      <p className="mt-1 flex items-center gap-1.5 text-sm font-medium text-fg-muted">
        <Icon name="clipboard" size={15} className="shrink-0 text-violet-text" />
        {t('results.idHint')}
      </p>
    </div>
  );
}

const JUMPS = ['why', 'map', 'viewer', 'lesions', 'spectral', 'qc'] as const;

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
          thumbnailUrl={wlThumbnail(api, result).url}
          syntheticHint={result.synthetic}
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
  const desktop = useMediaQuery('(min-width: 1024px)');
  const [selected, setSelected] = useState<OralSite | null>(() => defaultSite(result));
  const [openSections, setOpenSections] = useState<Record<string, boolean>>({});
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
  const inconclusive = result.triage.overall_risk_level === 'INCONCLUSIVE';
  const needsRetake = failedSites(result).length > 0;

  // Phones: long, secondary sections start collapsed (always open on desktop).
  const collapsible = !desktop;
  const isOpen = (id: string) => desktop || (openSections[id] ?? false);
  const setOpen = (id: string, v: boolean) => setOpenSections((o) => ({ ...o, [id]: v }));

  // "Why this level": the finding vs the person's risk factors, with subtotals.
  const lesions = result.lesions ?? [];
  const topLesion = [...lesions].sort((a, b) => b.score - a.score)[0];
  const groupBy = (f: ContributingFactor) =>
    RISK_FACTOR_SOURCES.has(f.source)
      ? { key: 'risk', label: t('results.why.groupRiskFactors') }
      : topLesion
        ? {
            key: 'finding',
            label: t('results.why.groupFinding', {
              id: topLesion.lesion_id,
              site: t(`sites.${topLesion.site}`),
            }),
          }
        : { key: 'other', label: t('results.why.groupOther') };
  const others = lesions.filter((l) => l !== topLesion);
  const whyNote =
    topLesion && others.length > 0
      ? t('results.why.onlyTopFinding', {
          id: topLesion.lesion_id,
          others: others
            .map((l) => `${l.lesion_id} · ${formatPoints(l.score).replace('+', '')}`)
            .join(', '),
        })
      : undefined;

  const selectSite = (s: OralSite) => {
    setSelected(s);
    if (typeof window !== 'undefined' && window.innerWidth < 1024) {
      viewerRef.current?.scrollIntoView({ behavior: reduced ? 'auto' : 'smooth', block: 'start' });
    }
  };

  const jump = (e: MouseEvent<HTMLAnchorElement>, id: string) => {
    e.preventDefault();
    if (!isOpen(id)) setOpen(id, true);
    requestAnimationFrame(() => {
      const el = document.getElementById(id);
      if (!el) return;
      el.focus({ preventScroll: true });
      el.scrollIntoView({ behavior: reduced ? 'auto' : 'smooth', block: 'start' });
    });
  };

  const cards = {
    why: (
      <Card
        id="why"
        title={t('results.sections.why')}
        icon="layers"
        subtitle={t('results.why.intro')}
      >
        {inconclusive && (result.triage.contributing_factors ?? []).length === 0 ? (
          <p className="rounded-xl border border-dashed border-risk-inconclusive p-4 text-sm text-fg-muted">
            {t('results.why.noScore')}
          </p>
        ) : (
          <FactorBars
            factors={result.triage.contributing_factors}
            thresholds={result.triage.thresholds}
            score={result.triage.score}
            level={result.triage.overall_risk_level}
            groupBy={groupBy}
            note={whyNote}
          />
        )}
      </Card>
    ),
    map: (
      <Card id="map" title={t('results.sections.map')} icon="pin" subtitle={t('results.map.hint')}>
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
          <li>
            <span className="inline-flex items-center gap-1.5 rounded-full border-[1.5px] border-line px-2.5 py-1 text-xs font-bold tracking-wide text-fg-subtle uppercase opacity-80">
              <span className="h-3 w-3 rounded-full border border-line-strong" aria-hidden="true" />
              {t('results.map.notExamined')}
            </span>
          </li>
        </ul>
      </Card>
    ),
    viewer: (
      <div ref={viewerRef} className="scroll-mt-28">
        <Card id="viewer" title={t('results.sections.viewer')} icon="eye">
          {site ? <SiteViewer key={site.site} result={result} site={site} /> : null}
        </Card>
      </div>
    ),
    lesions: (
      <Card id="lesions" title={t('results.sections.lesions')} icon="target">
        <LesionList result={result} onSelectSite={selectSite} selectedSite={site?.site ?? null} />
      </Card>
    ),
    spectral: (
      <Card
        id="spectral"
        title={t('results.sections.spectral')}
        icon="wave"
        collapsible={collapsible}
        open={isOpen('spectral')}
        onOpenChange={(v) => setOpen('spectral', v)}
      >
        <SpectralPanel result={result} />
      </Card>
    ),
    qc: (
      <Card id="qc" title={t('results.sections.qc')} icon="shield">
        <QcPanel result={result} />
      </Card>
    ),
  };

  const actions = (
    <div
      className="flex flex-wrap items-center gap-2 lg:max-w-md lg:flex-col lg:items-end lg:gap-3"
      data-testid="result-actions"
    >
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
        demoteReanalyse={inconclusive && needsRetake}
      />
    </div>
  );

  return (
    <div
      className="mx-auto max-w-7xl px-4 pt-4 pb-16 sm:px-6 sm:pt-6"
      data-testid="results-page"
      data-synthetic={result.synthetic}
    >
      <div className="flex flex-col gap-4">
        {/* Phones reach Screenings from the tab bar; the back link would only push the result down. */}
        <Link
          to="/screenings"
          className="btn btn-ghost -ml-3 hidden self-start px-3 text-sm sm:inline-flex"
        >
          <Icon name="arrowLeft" size={16} />
          {t('app.nav.screenings')}
        </Link>
        {result.synthetic ? <SyntheticRibbon notice={result.notices?.synthetic_notice} /> : null}

        {/* DOM order = visual order. Desktop: patient | actions, then risk and retake. Phones:
            patient, risk, retake, then actions, so the risk level is on the first screen. */}
        <div className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
          <header className="min-w-0">
            <p className="eyebrow">{t('results.pageTitle')}</p>
            <PatientId id={p.pseudonymous_id} />
            <p className="mt-2 flex flex-col gap-y-0.5 text-sm text-fg-muted tabular sm:flex-row sm:flex-wrap sm:gap-x-3">
              <span>
                {t('results.patientMeta', {
                  age:
                    p.age_years != null
                      ? t('common.years', { count: p.age_years })
                      : t('results.ageUnknown'),
                  sex: t(`upload.sex.${p.sex ?? 'undisclosed'}`),
                })}
              </span>
              <span aria-hidden="true" className="hidden sm:inline">
                ·
              </span>
              <span>
                {t('results.capturedAt', {
                  date: formatDateTime(result.context.captured_at ?? result.created_at),
                })}
              </span>
              {result.context.screening_site_name ? (
                <>
                  <span aria-hidden="true" className="hidden sm:inline">
                    ·
                  </span>
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
          {desktop ? <div className="shrink-0">{actions}</div> : null}
        </div>

        {result.result_version !== latestVersion ? (
          <div
            className="flex flex-wrap items-center gap-3 rounded-xl border border-warn/60 bg-risk-moderate-bg p-3 text-sm"
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

        <RiskHeader result={result} />
        <RetakeCallout result={result} />
        {desktop ? null : actions}
      </div>

      {/* Phones and tablets: a sticky section index so the long page stays navigable. */}
      <nav
        aria-label={t('results.jump.label')}
        className="sticky top-[5.5rem] z-20 -mx-4 mt-5 border-y border-line/70 bg-bg/90 px-4 py-2 backdrop-blur-xl sm:-mx-6 sm:px-6 lg:hidden"
        data-testid="section-jump"
      >
        <ul className="flex justify-between gap-1 overflow-x-auto [scrollbar-width:none] sm:justify-start sm:gap-1.5">
          {JUMPS.map((id) => (
            <li key={id} className="shrink-0">
              <a
                href={`#${id}`}
                onClick={(e) => jump(e, id)}
                className="inline-flex min-h-9 items-center rounded-full border border-line bg-surface-2 px-2 text-xs font-semibold text-fg-muted hover:text-fg sm:px-3 sm:text-sm"
              >
                {t(`results.jump.${id}`)}
              </a>
            </li>
          ))}
        </ul>
      </nav>

      {/* Phones: one column in reading order (Why, Map, Viewer, Findings, Probe, QC).
          Desktop: two real columns, each read top to bottom. The DOM follows what is on
          screen at each size, so keyboard focus never jumps around (no CSS `order`). */}
      {desktop ? (
        <div className="mt-5 grid grid-cols-[minmax(0,1.25fr)_minmax(0,1fr)] items-start gap-5">
          <div className="flex min-w-0 flex-col gap-5">
            {cards.why}
            {cards.viewer}
            {cards.spectral}
          </div>
          <div className="flex min-w-0 flex-col gap-5">
            {cards.map}
            {cards.lesions}
            {cards.qc}
          </div>
        </div>
      ) : (
        <div className="mt-5 flex flex-col gap-5">
          {cards.why}
          {cards.map}
          {cards.viewer}
          {cards.lesions}
          {cards.spectral}
          {cards.qc}
        </div>
      )}

      <div className="mt-5">
        <Card
          id="versions"
          title={t('results.sections.versions')}
          icon="history"
          collapsible={collapsible}
          open={isOpen('versions')}
          onOpenChange={(v) => setOpen('versions', v)}
        >
          <VersionsFooter result={result} />
        </Card>
      </div>

      {overlay}
    </div>
  );
}
