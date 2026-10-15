import { useTranslation } from 'react-i18next';
import { Icon } from '../ui/Icon';

/** Persistent, non-dismissable notice shown on every screen. */
export function PrototypeBanner() {
  const { t } = useTranslation();
  return (
    <div
      role="note"
      aria-label={t('banner.label')}
      data-testid="prototype-banner"
      className="sticky top-0 z-50 flex min-h-8 items-center justify-center gap-2 border-b border-line bg-surface-3 px-4 py-1.5 text-center text-[13px] font-semibold text-fg"
    >
      <Icon name="alert" size={15} className="shrink-0 text-warn" />
      <span>{t('banner.prototype')}</span>
    </div>
  );
}
