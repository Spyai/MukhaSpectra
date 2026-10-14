import { useTranslation } from 'react-i18next';

/** Notice strip shown at the top of every view of a synthetic result. */
export function SyntheticRibbon({ notice }: { notice?: string | null }) {
  const { t } = useTranslation();
  return (
    <div
      data-testid="synthetic-ribbon"
      role="note"
      aria-label={t('synthetic.ribbon')}
      className="synthetic-stripes flex flex-col items-start gap-x-3 gap-y-1.5 rounded-2xl px-3 py-2.5 text-sm font-semibold shadow-card sm:flex-row sm:items-center sm:px-4"
    >
      <span className="rounded-md bg-synthetic-ink px-2 py-0.5 text-xs font-bold tracking-widest text-synthetic">
        {t('synthetic.ribbon')}
      </span>
      <span className="min-w-0 flex-1 rounded-md bg-synthetic px-1.5 py-0.5 text-[13px] leading-snug sm:text-sm">
        {notice ?? t('synthetic.defaultNotice')}
      </span>
    </div>
  );
}

/** Diagonal corner ribbon. Place inside a `relative overflow-hidden` container. */
export function SyntheticCorner() {
  const { t } = useTranslation();
  return (
    <div
      aria-hidden="true"
      className="pointer-events-none absolute top-0 right-0 z-10 h-36 w-36 overflow-hidden"
    >
      <div className="synthetic-stripes absolute top-[38px] -right-[50px] w-[200px] rotate-45 py-1 text-center text-[10px] font-extrabold tracking-[0.18em] shadow-card">
        {t('synthetic.ribbon')}
      </div>
    </div>
  );
}

export function SyntheticTag({ className = '' }: { className?: string }) {
  const { t } = useTranslation();
  return (
    <span
      className={`inline-flex items-center rounded-md border border-synthetic/70 bg-synthetic/20 px-1.5 py-0.5 text-[11px] font-bold tracking-widest text-fg ${className}`}
    >
      {t('synthetic.tag')}
    </span>
  );
}
