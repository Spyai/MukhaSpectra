import { NavLink, Outlet, useLocation } from 'react-router';
import { useTranslation } from 'react-i18next';
import { useEffect, useRef } from 'react';
import { IS_MOCK } from '../../api/client';
import { useTheme } from '../../lib/theme';
import { Icon, type IconName } from '../ui/Icon';
import { LogoMark } from './Logo';
import { PrototypeBanner } from './PrototypeBanner';

const NAV: Array<{ to: string; key: 'home' | 'upload' | 'screenings'; icon: IconName; end?: boolean }> = [
  { to: '/', key: 'home', icon: 'glow', end: true },
  { to: '/upload', key: 'upload', icon: 'upload' },
  { to: '/screenings', key: 'screenings', icon: 'history' },
];

export function AppShell() {
  const { t } = useTranslation();
  const [theme, toggleTheme] = useTheme();
  const location = useLocation();
  const mainRef = useRef<HTMLElement>(null);
  const first = useRef(true);

  // Move focus to the main landmark on route change so screen readers announce the new page.
  useEffect(() => {
    if (first.current) {
      first.current = false;
      return;
    }
    mainRef.current?.focus({ preventScroll: true });
    window.scrollTo({ top: 0 });
  }, [location.pathname]);

  return (
    <div className="flex min-h-dvh flex-col">
      <a
        href="#main"
        className="sr-only z-[60] rounded-lg bg-violet px-4 py-2 font-semibold text-on-violet focus:not-sr-only focus:fixed focus:top-12 focus:left-4"
      >
        {t('app.skipToContent')}
      </a>
      <PrototypeBanner />
      <header className="sticky top-8 z-30 border-b border-line/70 bg-bg/80 backdrop-blur-xl supports-[backdrop-filter]:bg-bg/65">
        <div className="mx-auto flex h-14 max-w-7xl items-center gap-3 px-4 sm:px-6">
          <NavLink to="/" className="flex items-center gap-2.5 rounded-lg pr-1" aria-label={t('app.name')}>
            <LogoMark />
            <span className="flex flex-col leading-none">
              <span className="text-[15px] font-bold tracking-tight">{t('app.name')}</span>
              <span className="mt-0.5 text-[11px] font-medium text-fg-subtle">{t('app.consoleName')}</span>
            </span>
          </NavLink>
          <nav aria-label={t('app.nav.label')} className="ml-4 hidden items-center gap-1 sm:flex">
            {NAV.map((item) => (
              <NavLink
                key={item.to}
                to={item.to}
                end={item.end}
                className={({ isActive }) =>
                  `rounded-xl px-3 py-2 text-sm font-semibold transition-colors ${
                    isActive ? 'bg-violet-soft text-violet-text' : 'text-fg-muted hover:bg-surface-2 hover:text-fg'
                  }`
                }
              >
                {t(`app.nav.${item.key}`)}
              </NavLink>
            ))}
          </nav>
          <div className="ml-auto flex items-center gap-2">
            {IS_MOCK ? (
              <span
                className="chip border-violet/40 bg-violet-soft text-violet-text"
                title={t('app.mockModeHint')}
                data-testid="mock-pill"
              >
                <Icon name="flask" size={14} />
                {t('app.mockMode')}
              </span>
            ) : null}
            <button
              type="button"
              onClick={toggleTheme}
              className="btn btn-ghost h-11 w-11 !p-0"
              aria-label={theme === 'dark' ? t('app.theme.toLight') : t('app.theme.toDark')}
              title={theme === 'dark' ? t('app.theme.toLight') : t('app.theme.toDark')}
            >
              <Icon name={theme === 'dark' ? 'sun' : 'moon'} />
            </button>
          </div>
        </div>
      </header>

      <main id="main" ref={mainRef} tabIndex={-1} className="flex-1 outline-none">
        <Outlet />
      </main>

      <footer className="mb-16 border-t border-line px-4 py-6 text-center text-xs text-fg-subtle sm:mb-0">
        <p>{t('app.footer')}</p>
      </footer>

      <nav
        aria-label={t('app.nav.label')}
        className="fixed inset-x-0 bottom-0 z-30 grid grid-cols-3 border-t border-line bg-bg/90 pb-[env(safe-area-inset-bottom)] backdrop-blur-xl sm:hidden"
      >
        {NAV.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            end={item.end}
            className={({ isActive }) =>
              `flex min-h-14 flex-col items-center justify-center gap-0.5 text-[11px] font-semibold ${
                isActive ? 'text-violet-text' : 'text-fg-subtle'
              }`
            }
          >
            <Icon name={item.icon} size={22} />
            {t(`app.nav.${item.key}`)}
          </NavLink>
        ))}
      </nav>
    </div>
  );
}
