import { AnimatePresence, m } from 'motion/react';
import { useState } from 'react';
import { useTranslation } from 'react-i18next';
import type { LesionAssessment, OralSite, ScreeningResult } from '../../api/types';
import { SOURCE_ICON, topFactors } from '../../lib/factors';
import { formatPoints } from '../../lib/format';
import { failedSites } from '../../lib/qc';
import { Icon, type IconName } from '../ui/Icon';
import { FactorBars } from './FactorBars';
import { RiskPill } from './RiskBadge';

const ORIGIN_ICON: Record<LesionAssessment['origins'][number], IconName> = {
  af_fvl: 'glow',
  visual: 'eye',
  wl_model: 'camera',
  spectral: 'wave',
};

function NotAnalysedNote({ count }: { count: number }) {
  const { t } = useTranslation();
  if (count === 0) return null;
  return (
    <p
      className="flex items-start gap-2 rounded-xl border border-dashed border-risk-inconclusive bg-risk-inconclusive-bg p-3.5 text-sm text-fg"
      data-testid="lesions-not-analysed"
    >
      <Icon name="camera" size={16} className="mt-0.5 shrink-0" />
      {t('results.lesions.notAnalysed', { count })}
    </p>
  );
}

export function LesionList({
  result,
  onSelectSite,
  selectedSite,
}: {
  result: ScreeningResult;
  onSelectSite: (site: OralSite) => void;
  selectedSite: OralSite | null;
}) {
  const { t } = useTranslation();
  const lesions = result.lesions ?? [];
  const [open, setOpen] = useState<string | null>(null);
  const notAnalysed = failedSites(result).length;

  if (lesions.length === 0) {
    return (
      <div className="flex flex-col gap-3">
        <p className="rounded-xl border border-dashed border-line p-4 text-sm text-fg-muted">
          {t('results.lesions.none')}
        </p>
        <NotAnalysedNote count={notAnalysed} />
      </div>
    );
  }
  return (
    <div className="flex flex-col gap-3">
      <ul className="flex flex-col gap-3" data-testid="lesion-list">
        {lesions.map((l) => {
          const expanded = open === l.lesion_id;
          const active = selectedSite === l.site;
          const top = topFactors(l.contributing_factors);
          const rest = l.contributing_factors.length - top.length;
          return (
            <li
              key={l.lesion_id}
              className={`rounded-2xl border p-4 transition-colors ${active ? 'border-violet/60 bg-violet-soft/40' : 'border-line bg-surface-2/50'}`}
              data-testid="lesion-card"
            >
              <div className="flex flex-wrap items-start justify-between gap-2">
                <button
                  type="button"
                  onClick={() => onSelectSite(l.site)}
                  className="flex min-h-10 min-w-0 items-center gap-2 rounded-lg text-left font-semibold hover:text-violet-text"
                >
                  <Icon name="pin" size={16} className="shrink-0 text-fg-subtle" />
                  <span className="min-w-0">{t(`sites.${l.site}`)}</span>
                  <span className="text-xs font-normal text-fg-subtle">{l.lesion_id}</span>
                </button>
                <RiskPill level={l.risk_level} />
              </div>
              <p className="mt-1 text-sm leading-relaxed text-fg-muted">{l.summary}</p>

              {/* A score is never shown without its reasons: the main ones are always visible. */}
              <div
                className="mt-3 rounded-xl border border-line bg-surface p-3"
                data-testid="lesion-reasons"
              >
                <div className="flex items-baseline justify-between gap-3">
                  <p className="text-xs font-semibold text-fg-subtle">
                    {t('results.lesions.topReasons')}
                  </p>
                  <p className="text-xs font-semibold text-fg-muted tabular">
                    {t('results.why.lesionScore', {
                      value: formatPoints(l.score).replace('+', ''),
                    })}
                  </p>
                </div>
                <ul className="mt-1.5 space-y-1">
                  {top.map((f) => (
                    <li key={f.code} className="flex items-start gap-2 text-sm">
                      <Icon
                        name={SOURCE_ICON[f.source]}
                        size={15}
                        className="mt-0.5 shrink-0 text-fg-subtle"
                      />
                      <span className="min-w-0 flex-1">{f.label}</span>
                      <span
                        className={`shrink-0 font-bold tabular ${f.points >= 0 ? 'text-violet-text' : 'text-glow-text'}`}
                      >
                        {formatPoints(f.points)}
                      </span>
                    </li>
                  ))}
                </ul>
                {rest > 0 ? (
                  <p className="mt-1 text-xs text-fg-subtle">
                    {t('results.lesions.moreReasons', { count: rest })}
                  </p>
                ) : null}
              </div>

              <div className="mt-3 flex flex-wrap items-center gap-1.5">
                <span className="mr-1 text-xs font-semibold text-fg-subtle">
                  {t('results.lesions.origins')}
                </span>
                {l.origins.map((o) => (
                  <span key={o} className="chip py-0.5 text-xs">
                    <Icon name={ORIGIN_ICON[o]} size={13} />
                    {t(`results.lesions.origin.${o}`)}
                  </span>
                ))}
              </div>
              {(l.confounders ?? []).length > 0 ? (
                <ul className="mt-2 space-y-1 text-xs text-warn">
                  {(l.confounders ?? []).map((c) => (
                    <li key={c.type} className="flex items-start gap-1.5">
                      <Icon name="alert" size={14} className="mt-px shrink-0" />
                      <span>
                        {t(`results.confounder.${c.type}`)} ·{' '}
                        {t('results.confounder.overlap', {
                          value: Math.round(c.overlap_fraction * 100),
                        })}
                      </span>
                    </li>
                  ))}
                </ul>
              ) : null}
              <button
                type="button"
                className="btn btn-ghost mt-2 min-h-10 px-2 text-sm"
                aria-expanded={expanded}
                aria-controls={`lesion-factors-${l.lesion_id}`}
                onClick={() => setOpen(expanded ? null : l.lesion_id)}
              >
                <Icon
                  name="chevronDown"
                  size={16}
                  className={`transition-transform ${expanded ? 'rotate-180' : ''}`}
                />
                {expanded
                  ? t('results.lesions.hideFactors')
                  : t('results.lesions.showFactors', { count: l.contributing_factors.length })}
              </button>
              <AnimatePresence initial={false}>
                {expanded ? (
                  <m.div
                    id={`lesion-factors-${l.lesion_id}`}
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: 'auto' }}
                    exit={{ opacity: 0, height: 0 }}
                    className="overflow-hidden"
                  >
                    <div className="pt-3">
                      <FactorBars
                        factors={l.contributing_factors}
                        score={l.score}
                        level={l.risk_level}
                        thresholds={result.triage.thresholds}
                        compact
                      />
                    </div>
                  </m.div>
                ) : null}
              </AnimatePresence>
            </li>
          );
        })}
      </ul>
      <NotAnalysedNote count={notAnalysed} />
    </div>
  );
}
