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
    <m.ul
      className="flex flex-col gap-2"
      initial="hidden"
      animate="show"
      variants={{ hidden: {}, show: { transition: { staggerChildren: 0.05 } } }}
      data-testid="screening-list"
    >
      {items.map((s) => (
        <m.li
          key={s.screening_id}
          variants={{ hidden: { opacity: 0, y: 8 }, show: { opacity: 1, y: 0 } }}
        >
          <Link
            to={`/screenings/${encodeURIComponent(s.screening_id)}`}
            className="group flex min-h-[72px] items-center gap-3 rounded-2xl border border-line bg-surface px-4 py-3 transition-colors hover:border-line-strong hover:bg-surface-2"
          >
            <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-surface-2 text-fg-muted">
              <Icon name="user" size={20} />
            </span>
            <span className="min-w-0 flex-1">
              <span className="flex flex-wrap items-center gap-2">
                <span className="font-semibold tabular">{s.pseudonymous_id ?? s.screening_id}</span>
                {s.synthetic ? <SyntheticTag /> : null}
              </span>
              <span className="mt-0.5 block text-xs text-fg-subtle tabular">
                {formatDateTime(s.created_at)}
                {s.age_years != null ? ` · ${t('common.years', { count: s.age_years })}` : ''}
                {s.sex ? ` · ${t(`upload.sex.${s.sex}`)}` : ''}
                {s.site_count ? ` · ${t('home.sitesCount', { count: s.site_count })}` : ''}
                {s.latest_version && s.latest_version > 1
                  ? ` · ${t('common.version', { version: s.latest_version })}`
                  : ''}
              </span>
            </span>
            {s.overall_risk_level ? <RiskPill level={s.overall_risk_level} /> : null}
            <Icon
              name="chevronRight"
              size={18}
              className="shrink-0 text-fg-subtle transition-transform group-hover:translate-x-0.5"
            />
          </Link>
        </m.li>
      ))}
    </m.ul>
  );
}
