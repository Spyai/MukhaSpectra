import { m } from 'motion/react';
import type { ReactNode } from 'react';
import { Icon, type IconName } from './Icon';

export function Card({
  id,
  title,
  icon,
  actions,
  children,
  className = '',
  subtitle,
}: {
  id: string;
  title: string;
  icon?: IconName;
  actions?: ReactNode;
  children: ReactNode;
  className?: string;
  subtitle?: ReactNode;
}) {
  return (
    <m.section
      aria-labelledby={`${id}-title`}
      className={`card p-4 sm:p-6 ${className}`}
      initial={{ opacity: 0, y: 14 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ type: 'spring', stiffness: 220, damping: 28 }}
      id={id}
    >
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <h2
            id={`${id}-title`}
            className="flex items-center gap-2.5 text-lg font-bold tracking-tight"
          >
            {icon ? (
              <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-violet-soft text-violet-text">
                <Icon name={icon} size={18} />
              </span>
            ) : null}
            {title}
          </h2>
          {subtitle ? <div className="mt-1.5 text-sm text-fg-muted">{subtitle}</div> : null}
        </div>
        {actions ? <div className="flex flex-wrap items-center gap-2">{actions}</div> : null}
      </div>
      <div className="mt-5">{children}</div>
    </m.section>
  );
}
