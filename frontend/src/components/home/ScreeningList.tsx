import { m } from 'motion/react';
import { useTranslation } from 'react-i18next';
import { Link } from 'react-router';
import { useScreenings } from '../../api/queries';
import { formatDateTime } from '../../lib/format';
import { errorMessage } from '../../lib/errors';
import { RiskPill } from '../results/RiskBadge';
import { SyntheticTag } from '../layout/SyntheticRibbon';
import { Icon } from '../ui/Icon';

export function ScreeningList({ limit }: { limit?: number }) {
  const { t } = useTranslation();
  const q = useScreenings();

  if (q.isPending) {
    return (
      <ul className="flex flex-col gap-2" aria-busy="true" aria-label={t('common.loading')}>
        {[0, 1, 2].map((i) => (
          <li key={i} className="h-[72px] animate-pulse rounded-2xl bg-surface-2" />
        ))}
      </ul>
    );
  }
  if (q.isError) {
    return (
      <div
        className="rounded-2xl border border-dashed border-line p-5 text-sm text-fg-muted"
        role="alert"
      >
        <p>{t('home.recentError')}</p>
        <p className="mt-1 text-fg-subtle">{errorMessage(q.error, t)}</p>
        <button type="button" className="btn btn-secondary mt-3" onClick={() => void q.refetch()}>
          <Icon name="refresh" size={16} />
          {t('common.retry')}
        </button>
      </div>
    );
  }
  const items = limit ? q.data.slice(0, limit) : q.data;
  if (items.length === 0) {
    return (
      <p className="rounded-2xl border border-dashed border-line p-5 text-sm text-fg-muted">
        {t('home.recentEmpty')}
      </p>
    );
  }
  return (
    <div>
      {/* Desktop column headings (the rows below are links, read as one sentence each). */}
      <div
        aria-hidden="true"
        className="hidden grid-cols-[2.5rem_minmax(0,1.3fr)_minmax(0,1fr)_minmax(0,1fr)_8.5rem_1.25rem] gap-x-4 px-4 pb-2 text-xs font-semibold tracking-wide text-fg-subtle uppercase lg:grid"
      >
        <span />
        <span>{t('screenings.columns.patient')}</span>
        <span>{t('screenings.columns.date')}</span>
        <span>{t('screenings.columns.details')}</span>
        <span>{t('screenings.columns.risk')}</span>
        <span />
      </div>
      <m.ul
        className="flex flex-col gap-2"
        initial="hidden"
        animate="show"
        variants={{ hidden: {}, show: { transition: { staggerChildren: 0.05 } } }}
        data-testid="screening-list"
      >
        {items.map((s) => {
          const details = [
            s.age_years != null ? t('common.years', { count: s.age_years }) : null,
            s.sex ? t(`upload.sex.${s.sex}`) : null,
            s.site_count ? t('home.sitesCount', { count: s.site_count }) : null,
            s.latest_version && s.latest_version > 1
              ? t('common.version', { version: s.latest_version })
              : null,
          ].filter(Boolean);
          return (
            <m.li
              key={s.screening_id}
              variants={{ hidden: { opacity: 0, y: 8 }, show: { opacity: 1, y: 0 } }}
            >
              <Link
                to={`/screenings/${encodeURIComponent(s.screening_id)}`}
                className="group grid min-h-[72px] grid-cols-[2.5rem_minmax(0,1fr)_1.25rem] items-center gap-x-3 gap-y-1.5 rounded-2xl border border-line bg-surface px-4 py-3 transition-colors hover:border-line-strong hover:bg-surface-2 lg:grid-cols-[2.5rem_minmax(0,1.3fr)_minmax(0,1fr)_minmax(0,1fr)_8.5rem_1.25rem] lg:gap-x-4"
              >
                <span className="row-span-2 flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-surface-2 text-fg-muted lg:row-span-1">
                  <Icon name="user" size={20} />
                </span>
                <span className="flex min-w-0 flex-wrap items-center gap-2">
                  <span className="id-text font-semibold whitespace-nowrap">
                    {s.pseudonymous_id ?? s.screening_id}
                  </span>
                  {s.synthetic ? <SyntheticTag /> : null}
                </span>
                <Icon
                  name="chevronRight"
                  size={18}
                  className="col-start-3 row-span-2 row-start-1 shrink-0 justify-self-end text-fg-subtle transition-transform group-hover:translate-x-0.5 lg:col-start-6 lg:row-span-1"
                />
                {/* Phones: date + details on one line, risk level under them. */}
                <span className="col-start-2 min-w-0 text-xs text-fg-subtle tabular lg:col-start-3 lg:row-start-1 lg:text-sm lg:text-fg-muted">
                  {formatDateTime(s.created_at)}
                  <span className="lg:hidden">
                    {details.length ? ` · ${details.join(' · ')}` : ''}
                  </span>
                </span>
                <span className="hidden min-w-0 text-sm text-fg-muted tabular lg:col-start-4 lg:row-start-1 lg:block">
                  {details.join(' · ')}
                </span>
                {s.overall_risk_level ? (
                  <span className="col-start-2 lg:col-start-5 lg:row-start-1">
                    <RiskPill level={s.overall_risk_level} />
                  </span>
                ) : null}
              </Link>
            </m.li>
          );
        })}
      </m.ul>
    </div>
  );
}
