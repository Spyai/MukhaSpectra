import { m } from 'motion/react';
import type { ReactNode } from 'react';
import { useTranslation } from 'react-i18next';
import type { ContributingFactor, RiskLevel, ScoreThresholds } from '../../api/types';
import { SOURCE_ICON, waterfall, type WaterfallRow } from '../../lib/factors';
import { formatPoints, round } from '../../lib/format';
import { RISK_STYLE } from '../../lib/risk';
import { Icon } from '../ui/Icon';

interface FactorBarsProps {
  factors: ContributingFactor[];
  thresholds?: ScoreThresholds;
  /** The score reported by the rules engine (shown as the total). */
  score?: number | null;
  level?: RiskLevel;
  compact?: boolean;
  id?: string;
  /** Optional grouping of contiguous rows, e.g. the finding vs the person's risk factors. */
  groupBy?: (f: ContributingFactor) => { key: string; label: string };
  /** Shown under the total, e.g. that only the highest-scoring finding counts. */
  note?: ReactNode;
}

export function FactorBars({
  factors,
  thresholds,
  score,
  level,
  compact = false,
  id,
  groupBy,
  note,
}: FactorBarsProps) {
  const { t } = useTranslation();
  const { rows, total } = waterfall(factors);
  const shownTotal = score ?? total;
  const lows = rows.flatMap((r) => [r.start, r.end]);
  const min = Math.min(0, ...lows);
  const max = Math.max(shownTotal, ...lows, thresholds ? thresholds.high * 1.12 : 0, 1);
  const span = max - min || 1;
  const pct = (v: number) => ((v - min) / span) * 100;
  const levelColor = level ? RISK_STYLE[level].color : 'var(--violet)';

  const ticks = thresholds
    ? [
        {
          key: 'moderate',
          value: thresholds.moderate,
          label: t('results.why.thresholdModerate', { value: thresholds.moderate }),
          color: 'var(--risk-moderate)',
        },
        {
          key: 'high',
          value: thresholds.high,
          label: t('results.why.thresholdHigh', { value: thresholds.high }),
          color: 'var(--risk-high)',
        },
      ]
    : [];

  const track = (children?: ReactNode) => (
    <div className="relative h-6 w-full rounded-md bg-surface-2">
      {min < 0 ? (
        <div className="absolute inset-y-0 w-px bg-line-strong" style={{ left: `${pct(0)}%` }} />
      ) : null}
      {ticks.map((tk) => (
        <div
          key={tk.key}
          className="absolute inset-y-[-3px] w-0 border-l-[1.5px] border-dashed"
          style={{ left: `${pct(tk.value)}%`, borderColor: tk.color, opacity: 0.75 }}
        />
      ))}
      {children}
    </div>
  );

  const renderRow = ({ factor, start, end }: WaterfallRow, i: number) => {
    const left = pct(Math.min(start, end));
    const width = Math.max(pct(Math.max(start, end)) - left, 0.6);
    const positive = factor.points >= 0;
    return (
      <li
        key={`${factor.code}-${i}`}
        data-testid="factor-row"
        data-points={factor.points}
        className="grid grid-cols-[minmax(0,1fr)_3.5rem] items-center gap-x-3 gap-y-1.5 sm:grid-cols-[minmax(0,1.15fr)_minmax(0,1fr)_4rem] sm:gap-x-4"
      >
        <div className="col-start-1 row-start-1 flex min-w-0 items-start gap-2 sm:col-start-auto sm:row-start-auto">
          <span
            className="mt-0.5 flex h-6 w-6 shrink-0 items-center justify-center rounded-md bg-surface-2 text-fg-muted"
            title={t(`results.source.${factor.source}`)}
          >
            <Icon name={SOURCE_ICON[factor.source]} size={15} />
          </span>
          <div className="min-w-0">
            <span className="sr-only">
              {t('results.why.factorAria', {
                label: factor.label,
                points: formatPoints(factor.points),
                source: t(`results.source.${factor.source}`),
              })}
            </span>
            <p aria-hidden="true" className="text-sm leading-snug font-medium text-fg">
              {factor.label}
            </p>
            {!compact && factor.detail ? (
              <p aria-hidden="true" className="mt-0.5 text-xs leading-snug text-fg-subtle">
                {factor.detail}
              </p>
            ) : null}
          </div>
        </div>
        <div
          className="col-span-2 col-start-1 row-start-2 sm:col-span-1 sm:col-start-auto sm:row-start-auto"
          aria-hidden="true"
        >
          {track(
            <m.div
              className="absolute inset-y-1 rounded-[5px]"
              style={{
                left: `${left}%`,
                width: `${width}%`,
                background: positive ? 'var(--violet)' : 'var(--green)',
                opacity: positive ? 0.85 : 0.9,
                transformOrigin: positive ? 'left center' : 'right center',
              }}
              initial={{ scaleX: 0 }}
              animate={{ scaleX: 1 }}
              transition={{
                type: 'spring',
                stiffness: 260,
                damping: 30,
                delay: Math.min(i * 0.045, 0.5),
              }}
            />,
          )}
        </div>
        <span
          aria-hidden="true"
          className={`col-start-2 row-start-1 text-right text-sm font-bold tabular sm:col-start-auto sm:row-start-auto ${positive ? 'text-violet-text' : 'text-glow-text'}`}
        >
          {formatPoints(factor.points)}
        </span>
      </li>
    );
  };

  // Contiguous groups (e.g. "Finding L1" then "Risk factors"), each with its subtotal.
  const groups: Array<{
    key: string;
    label: string;
    first: number;
    subtotal: number;
    rows: Array<{ row: WaterfallRow; index: number }>;
  }> = [];
  rows.forEach((row, index) => {
    const g = groupBy ? groupBy(row.factor) : { key: 'all', label: '' };
    const last = groups[groups.length - 1];
    if (last && last.key === g.key) {
      last.rows.push({ row, index });
      last.subtotal = round(last.subtotal + row.factor.points, 6);
    } else {
      groups.push({ ...g, first: index, subtotal: row.factor.points, rows: [{ row, index }] });
    }
  });

  if (factors.length === 0) {
    return (
      <p
        className="rounded-xl border border-dashed border-line p-4 text-sm text-fg-muted"
        data-testid="factor-bars-empty"
      >
        {t('results.why.noFactors')}
      </p>
    );
  }

  return (
    <div data-testid="factor-bars" id={id}>
      {ticks.length > 0 && !compact ? (
        <div className="mb-1 grid grid-cols-1 sm:grid-cols-[minmax(0,1.15fr)_minmax(0,1fr)_4rem] sm:gap-x-4">
          <div className="hidden sm:block" />
          <div className="relative h-9">
            {ticks.map((tk) => (
              <span
                key={tk.key}
                className="absolute bottom-0 -translate-x-1/2 rounded px-1 text-[11px] leading-tight font-semibold whitespace-nowrap"
                style={{
                  left: `${pct(tk.value)}%`,
                  color: tk.color,
                  top: tk.key === 'moderate' ? 0 : undefined,
                }}
              >
                {tk.label}
              </span>
            ))}
          </div>
        </div>
      ) : null}
      <div role="group" aria-label={t('results.why.chartLabel')} className="flex flex-col gap-4">
        {groups.map((g) => (
          <div key={`${g.key}-${g.first}`} data-testid="factor-group" data-group={g.key}>
            {g.label ? (
              <div className="mb-2 flex items-baseline justify-between gap-3 border-b border-line/70 pb-1.5">
                <p className="text-xs font-bold tracking-wide text-fg-muted uppercase">{g.label}</p>
                <p className="text-xs font-semibold text-fg-subtle tabular">
                  {t('results.why.subtotal', { value: formatPoints(g.subtotal) })}
                </p>
              </div>
            ) : null}
            <ol aria-label={g.label || undefined} className="flex flex-col gap-2.5">
              {g.rows.map(({ row, index }) => renderRow(row, index))}
            </ol>
          </div>
        ))}
      </div>
      <div
        className="mt-4 grid grid-cols-[minmax(0,1fr)_3.5rem] items-center gap-x-3 gap-y-1.5 border-t border-line pt-4 sm:grid-cols-[minmax(0,1.15fr)_minmax(0,1fr)_4rem] sm:gap-x-4"
        data-testid="factor-total"
        data-total={total}
      >
        <p className="col-start-1 row-start-1 text-sm font-bold sm:col-start-auto sm:row-start-auto">
          {compact ? t('results.why.score') : t('results.why.total')}
        </p>
        <div
          className="col-span-2 col-start-1 row-start-2 sm:col-span-1 sm:col-start-auto sm:row-start-auto"
          aria-hidden="true"
        >
          {track(
            <m.div
              className="absolute inset-y-0.5 rounded-[6px]"
              style={{
                left: `${pct(0)}%`,
                width: `${Math.max(pct(shownTotal) - pct(0), 0.6)}%`,
                background: levelColor,
                transformOrigin: 'left center',
              }}
              initial={{ scaleX: 0 }}
              animate={{ scaleX: 1 }}
              transition={{ type: 'spring', stiffness: 160, damping: 26, delay: 0.2 }}
            />,
          )}
        </div>
        <span
          className="col-start-2 row-start-1 text-right text-lg font-extrabold tabular sm:col-start-auto sm:row-start-auto"
          style={{ color: levelColor }}
        >
          {formatPoints(shownTotal).replace('+', '')}
        </span>
      </div>
      {!compact ? <p className="mt-3 text-xs text-fg-subtle">{t('results.why.sumNote')}</p> : null}
      {note ? (
        <p
          className="mt-2 flex items-start gap-1.5 text-xs text-fg-muted"
          data-testid="factor-note"
        >
          <Icon name="info" size={14} className="mt-px shrink-0" />
          <span>{note}</span>
        </p>
      ) : null}
    </div>
  );
}
