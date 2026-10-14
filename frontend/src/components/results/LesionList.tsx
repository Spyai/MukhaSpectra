import { AnimatePresence, m } from 'motion/react';
import { useState } from 'react';
import { useTranslation } from 'react-i18next';
import type { LesionAssessment, OralSite, ScreeningResult } from '../../api/types';
import { formatPoints } from '../../lib/format';
import { Icon, type IconName } from '../ui/Icon';
import { FactorBars } from './FactorBars';
import { RiskPill } from './RiskBadge';

const ORIGIN_ICON: Record<LesionAssessment['origins'][number], IconName> = {
  af_fvl: 'glow',
  visual: 'eye',
  wl_model: 'camera',
  spectral: 'wave',
};

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

  if (lesions.length === 0) {
    return (
      <p className="rounded-xl border border-dashed border-line p-4 text-sm text-fg-muted">
        {t('results.lesions.none')}
      </p>
    );
  }
  return (
    <ul className="flex flex-col gap-3" data-testid="lesion-list">
      {lesions.map((l) => {
        const expanded = open === l.lesion_id;
        const active = selectedSite === l.site;
        return (
          <li
            key={l.lesion_id}
            className={`rounded-2xl border p-4 transition-colors ${active ? 'border-violet/60 bg-violet-soft/40' : 'border-line bg-surface-2/50'}`}
          >
            <div className="flex flex-wrap items-start justify-between gap-2">
              <button
                type="button"
                onClick={() => onSelectSite(l.site)}
                className="flex min-w-0 items-center gap-2 rounded-lg text-left font-semibold hover:text-violet-text"
              >
                <Icon name="pin" size={16} className="shrink-0 text-fg-subtle" />
                <span className="truncate">{t(`sites.${l.site}`)}</span>
                <span className="text-xs font-normal text-fg-subtle">{l.lesion_id}</span>
              </button>
              <div className="flex items-center gap-2">
                <span className="text-xs text-fg-subtle tabular">
                  {t('results.why.lesionScore', { value: formatPoints(l.score).replace('+', '') })}
                </span>
                <RiskPill level={l.risk_level} />
              </div>
            </div>
            <p className="mt-2 text-sm leading-relaxed text-fg-muted">{l.summary}</p>
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
              {expanded ? t('results.lesions.hideFactors') : t('results.lesions.showFactors')}
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
  );
}
