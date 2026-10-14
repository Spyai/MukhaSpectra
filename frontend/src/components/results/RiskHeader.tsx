import { m } from 'motion/react';
import { useTranslation } from 'react-i18next';
import type { ScreeningResult } from '../../api/types';
import { RISK_STYLE, isRiskLevel, overrideFloor } from '../../lib/risk';
import { SyntheticCorner } from '../layout/SyntheticRibbon';
import { Icon } from '../ui/Icon';
import { RiskBadge, RiskPill } from './RiskBadge';

export function RiskHeader({ result }: { result: ScreeningResult }) {
  const { t } = useTranslation();
  const tri = result.triage;
  const level = tri.overall_risk_level;
  const style = RISK_STYLE[level];
  const action = tri.recommended_action;
  const days = action.timeline_days;
  const overrides = tri.overrides_applied ?? [];
  const reviewed = tri.review_status === 'reviewed';

  return (
    <section
      className="card relative overflow-hidden p-5 sm:p-8"
      aria-labelledby="risk-header-title"
      data-testid="risk-header"
    >
      {result.synthetic ? <SyntheticCorner /> : null}
      <div
        aria-hidden="true"
        className="pointer-events-none absolute -top-24 -left-24 h-72 w-72 rounded-full opacity-40 blur-3xl"
        style={{ background: style.bg }}
      />
      <h2 id="risk-header-title" className="sr-only">
        {t('risk.eyebrow')}
      </h2>
      <div className="relative grid gap-6 lg:grid-cols-[auto_minmax(0,1fr)] lg:gap-10">
        <div className="flex flex-col gap-4">
          <RiskBadge level={level} />
          <div className="flex flex-wrap gap-2">
            <span
              className={`chip ${reviewed ? '' : 'border-warn/60 text-fg'}`}
              data-testid="review-chip"
            >
              <Icon
                name={reviewed ? 'check' : 'clock'}
                size={14}
                className={reviewed ? 'text-ok' : 'text-warn'}
              />
              {reviewed ? t('results.reviewed') : t('results.pendingReview')}
            </span>
            <span className="chip">
              <Icon name="alert" size={14} />
              {t(`results.urgency.${action.referral_urgency}`)}
            </span>
          </div>
        </div>

        <div className={`min-w-0 ${result.synthetic ? 'lg:pr-16' : ''}`}>
          <p className="eyebrow">{t('results.action')}</p>
          <m.p
            className="mt-1.5 text-xl leading-snug font-bold sm:text-2xl"
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.35, type: 'spring', stiffness: 220, damping: 26 }}
            data-testid="recommended-action"
          >
            {action.summary}
          </m.p>
          <div className="mt-4 flex flex-wrap items-stretch gap-4">
            <div
              className="flex min-w-36 flex-col justify-center rounded-2xl border px-4 py-3"
              style={{ borderColor: style.color, background: style.bg }}
            >
              <span className="text-xs font-semibold text-fg-muted">{t('results.timeline')}</span>
              <span
                className="text-2xl font-extrabold tabular"
                style={{ color: style.color }}
                data-testid="timeline"
              >
                {days === null || days === undefined
                  ? '—'
                  : days === 0
                    ? t('results.timelineNow')
                    : t('results.timelineDays', { count: days })}
              </span>
            </div>
            {(action.steps ?? []).length > 0 ? (
              <ol className="min-w-0 flex-1 basis-64 list-decimal space-y-1 pl-5 text-sm text-fg-muted marker:font-semibold marker:text-fg-subtle">
                {(action.steps ?? []).map((s) => (
                  <li key={s}>{s}</li>
                ))}
              </ol>
            ) : null}
          </div>

          <div className="mt-5">
            <p className="eyebrow">{t('results.explanation')}</p>
            <p className="mt-1.5 max-w-3xl leading-relaxed text-fg" data-testid="explanation">
              {tri.explanation}
            </p>
          </div>

          {overrides.length > 0 ? (
            <div
              className="mt-5 rounded-2xl border-2 border-risk-high/60 bg-risk-high-bg p-4"
              data-testid="overrides"
            >
              <p className="flex items-center gap-2 text-sm font-bold">
                <Icon name="lock" size={18} className="text-risk-high" />
                {t('results.overridesTitle')}
              </p>
              <ul className="mt-2 space-y-2">
                {overrides.map((o) => {
                  const floor = overrideFloor(o);
                  return (
                    <li
                      key={o.code}
                      className="flex flex-wrap items-center gap-x-3 gap-y-1 text-sm"
                    >
                      <span className="font-semibold">{o.label}</span>
                      {floor && isRiskLevel(floor) ? (
                        <span className="inline-flex items-center gap-1.5 text-xs text-fg-muted">
                          {t('results.overrideFloor', { level: t(`risk.level.${floor}`) })}
                          <RiskPill level={floor} />
                        </span>
                      ) : null}
                    </li>
                  );
                })}
              </ul>
              <p className="mt-3 text-sm font-semibold text-fg">{t('results.overrideNote')}</p>
            </div>
          ) : null}
        </div>
      </div>

      <p
        className="relative mt-6 flex items-start gap-2.5 rounded-xl border border-line bg-surface-2 p-3.5 text-sm leading-relaxed text-fg-muted"
        role="note"
        aria-label={t('disclaimer.label')}
        data-testid="disclaimer"
      >
        <Icon name="info" size={18} className="mt-0.5 shrink-0 text-violet-text" />
        <span>{result.notices?.disclaimer ?? t('disclaimer.default')}</span>
      </p>
    </section>
  );
}
