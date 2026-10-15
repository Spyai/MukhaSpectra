import { m } from 'motion/react';
import type { ReactNode } from 'react';
import { useTranslation } from 'react-i18next';
import { Icon, type IconName } from './Icon';

export function Card({
  id,
  title,
  icon,
  actions,
  children,
  className = '',
  subtitle,
  collapsible = false,
  open = true,
  onOpenChange,
}: {
  id: string;
  title: string;
  icon?: IconName;
  actions?: ReactNode;
  children: ReactNode;
  className?: string;
  subtitle?: ReactNode;
  /** Phones: the header becomes a disclosure button (content stays mounted, just hidden). */
  collapsible?: boolean;
  open?: boolean;
  onOpenChange?: (open: boolean) => void;
}) {
  const { t } = useTranslation();
  const isOpen = !collapsible || open;
  const heading = (
    <span className="flex items-center gap-2.5">
      {icon ? (
        <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-violet-soft text-violet-text">
          <Icon name={icon} size={18} />
        </span>
      ) : null}
      {title}
    </span>
  );
  return (
    <m.section
      aria-labelledby={`${id}-title`}
      className={`card scroll-mt-16 p-4 outline-none sm:p-6 lg:scroll-mt-0 ${className}`}
      tabIndex={-1}
      initial={{ opacity: 0, y: 14 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ type: 'spring', stiffness: 220, damping: 28 }}
      id={id}
      data-open={isOpen}
    >
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0 flex-1">
          <h2 id={`${id}-title`} className="text-lg font-bold tracking-tight">
            {collapsible ? (
              <button
                type="button"
                className="-m-1 flex w-[calc(100%+0.5rem)] items-center justify-between gap-3 rounded-lg p-1 text-left"
                aria-expanded={isOpen}
                aria-controls={`${id}-body`}
                onClick={() => onOpenChange?.(!isOpen)}
                data-testid={`${id}-toggle`}
              >
                {heading}
                <span className="flex items-center gap-1.5 text-sm font-semibold text-fg-muted">
                  <span className="sr-only">
                    {isOpen
                      ? t('results.card.collapse', { title })
                      : t('results.card.expand', { title })}
                  </span>
                  <Icon
                    name="chevronDown"
                    size={20}
                    className={`transition-transform ${isOpen ? 'rotate-180' : ''}`}
                  />
                </span>
              </button>
            ) : (
              heading
            )}
          </h2>
          {subtitle && isOpen ? (
            <div className="mt-1.5 text-sm text-fg-muted">{subtitle}</div>
          ) : null}
        </div>
        {actions ? <div className="flex flex-wrap items-center gap-2">{actions}</div> : null}
      </div>
      <div id={`${id}-body`} className="mt-5" hidden={!isOpen}>
        {children}
      </div>
    </m.section>
  );
}
