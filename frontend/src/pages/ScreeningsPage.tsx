import { useTranslation } from 'react-i18next';
import { ScreeningList } from '../components/home/ScreeningList';

export default function ScreeningsPage() {
  const { t } = useTranslation();
  return (
    <div className="mx-auto max-w-4xl px-4 pt-8 pb-16 sm:px-6">
      <p className="eyebrow">{t('app.consoleName')}</p>
      <h1 className="mt-2 text-3xl font-extrabold tracking-tight">{t('screenings.title')}</h1>
      <p className="mt-2 text-fg-muted">{t('screenings.intro')}</p>
      <div className="mt-6">
        <ScreeningList />
      </div>
    </div>
  );
}
