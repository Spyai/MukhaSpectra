import { useTranslation } from 'react-i18next';
import { Link } from 'react-router';
import { Icon } from '../components/ui/Icon';

export default function NotFoundPage() {
  const { t } = useTranslation();
  return (
    <div className="mx-auto max-w-xl px-4 py-20 text-center">
      <Icon name="question" size={40} className="mx-auto text-fg-subtle" />
      <h1 className="mt-4 text-2xl font-bold">{t('app.notFoundTitle')}</h1>
      <p className="mt-2 text-fg-muted">{t('app.notFoundBody')}</p>
      <Link to="/" className="btn btn-primary mt-6">
        {t('app.backHome')}
      </Link>
    </div>
  );
}
