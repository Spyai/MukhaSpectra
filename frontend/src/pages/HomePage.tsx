import { m } from 'motion/react';
import { useCallback, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { Link, useNavigate } from 'react-router';
import { useRunDemo } from '../api/queries';
import type { DemoKind, RiskLevel } from '../api/types';
import { AnalysisOverlay } from '../components/analysis/AnalysisOverlay';
import { Hero } from '../components/home/Hero';
import { ScreeningList } from '../components/home/ScreeningList';
import { SyntheticTag } from '../components/layout/SyntheticRibbon';
import { RiskGlyph } from '../components/results/RiskBadge';
import { Icon, type IconName } from '../components/ui/Icon';
import { errorMessage } from '../lib/errors';
import { RISK_STYLE } from '../lib/risk';

const DEMOS: Array<{ kind: DemoKind; level: RiskLevel }> = [
  { kind: 'low', level: 'LOW' },
  { kind: 'moderate', level: 'MODERATE' },
  { kind: 'high', level: 'HIGH' },
];

const HOW: Array<{ key: 'capture' | 'analyse' | 'triage'; icon: IconName }> = [
  { key: 'capture', icon: 'camera' },
  { key: 'analyse', icon: 'glow' },
  { key: 'triage', icon: 'layers' },
];

export default function HomePage() {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const demo = useRunDemo();
  const [running, setRunning] = useState<DemoKind | null>(null);

  const start = (kind: DemoKind) => {
    setRunning(kind);
    demo.mutate(kind);
  };
  const onContinue = useCallback(() => {
    if (demo.data) navigate(`/screenings/${encodeURIComponent(demo.data.screening_id)}`);
  }, [demo.data, navigate]);
  const onClose = useCallback(() => {
    demo.reset();
    setRunning(null);
  }, [demo]);

  return (
    <>
      <Hero />

      <section className="mx-auto max-w-7xl px-4 pt-10 sm:px-6" aria-labelledby="how-title">
        <h2 id="how-title" className="eyebrow">
          {t('home.howTitle')}
        </h2>
        <ol className="mt-4 grid gap-3 sm:grid-cols-3">
          {HOW.map((h, i) => (
            <m.li
              key={h.key}
              className="card flex gap-4 p-5"
              initial={{ opacity: 0, y: 12 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: '-40px' }}
              transition={{ type: 'spring', stiffness: 220, damping: 28, delay: i * 0.07 }}
            >
              <span className="flex h-11 w-11 shrink-0 items-center justify-center rounded-xl bg-violet-soft text-violet-text">
                <Icon name={h.icon} size={22} />
              </span>
              <div>
                <p className="font-bold">
                  <span className="mr-1.5 text-fg-subtle tabular">{i + 1}.</span>
                  {t(`home.how.${h.key}Title`)}
                </p>
                <p className="mt-1 text-sm leading-relaxed text-fg-muted">{t(`home.how.${h.key}Body`)}</p>
              </div>
            </m.li>
          ))}
        </ol>
      </section>

      <section id="demos" className="mx-auto max-w-7xl scroll-mt-28 px-4 pt-12 sm:px-6" aria-labelledby="demo-title">
        <div className="flex flex-wrap items-end justify-between gap-3">
          <div>
            <h2 id="demo-title" className="text-2xl font-bold tracking-tight">
              {t('home.demoTitle')}
            </h2>
            <p className="mt-1 max-w-2xl text-sm text-fg-muted">{t('home.demoBody')}</p>
          </div>
        </div>
        <div className="mt-5 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {DEMOS.map((d, i) => {
            const style = RISK_STYLE[d.level];
            const name = t(`home.demo.${d.kind}.title`);
            const busy = demo.isPending && running === d.kind;
            return (
              <m.article
                key={d.kind}
                className="card relative flex flex-col overflow-hidden p-5"
                initial={{ opacity: 0, y: 14 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true, margin: '-40px' }}
                transition={{ type: 'spring', stiffness: 220, damping: 28, delay: i * 0.07 }}
                whileHover={{ y: -3 }}
                data-testid={`demo-card-${d.kind}`}
              >
                <div className="absolute inset-x-0 top-0 h-1" style={{ background: style.color }} />
                <div className="flex items-center justify-between gap-2">
                  <span
                    className="flex h-10 w-10 items-center justify-center rounded-xl"
                    style={{ background: style.bg, color: style.color }}
                  >
                    <RiskGlyph level={d.level} size={22} />
                  </span>
                  <SyntheticTag />
                </div>
                <h3 className="mt-4 text-lg font-bold">{name}</h3>
                <p className="mt-1 flex-1 text-sm leading-relaxed text-fg-muted">{t(`home.demo.${d.kind}.body`)}</p>
                <button
                  type="button"
                  className="btn btn-secondary mt-5 w-full"
                  onClick={() => start(d.kind)}
                  disabled={demo.isPending}
                  aria-label={t('home.demo.runLabel', { name })}
                >
                  <Icon name="play" size={14} />
                  {busy ? t('home.demo.running') : t('home.demo.run')}
                </button>
              </m.article>
            );
          })}
          <m.article
            className="card relative flex flex-col overflow-hidden border-violet/40 bg-[linear-gradient(160deg,var(--violet-soft),transparent_60%)] p-5"
            initial={{ opacity: 0, y: 14 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: '-40px' }}
            transition={{ type: 'spring', stiffness: 220, damping: 28, delay: 0.21 }}
            whileHover={{ y: -3 }}
          >
            <span className="flex h-10 w-10 items-center justify-center rounded-xl bg-violet text-on-violet">
              <Icon name="upload" size={20} />
            </span>
            <h3 className="mt-4 text-lg font-bold">{t('home.uploadCard.title')}</h3>
            <p className="mt-1 flex-1 text-sm leading-relaxed text-fg-muted">{t('home.uploadCard.body')}</p>
            <Link to="/upload" className="btn btn-primary mt-5 w-full">
              {t('home.uploadCard.cta')}
              <Icon name="arrowRight" size={18} />
            </Link>
          </m.article>
        </div>
      </section>

      <section className="mx-auto max-w-7xl px-4 pt-12 pb-16 sm:px-6" aria-labelledby="recent-title">
        <div className="flex items-end justify-between gap-3">
          <h2 id="recent-title" className="text-2xl font-bold tracking-tight">
            {t('home.recentTitle')}
          </h2>
          <Link to="/screenings" className="btn btn-ghost text-sm">
            {t('app.nav.screenings')}
            <Icon name="arrowRight" size={16} />
          </Link>
        </div>
        <div className="mt-4">
          <ScreeningList limit={6} />
        </div>
      </section>

      <AnalysisOverlay
        open={running !== null}
        mode="demo"
        result={demo.data?.latest ?? null}
        error={demo.isError ? errorMessage(demo.error, t) : null}
        onContinue={onContinue}
        onClose={onClose}
      />
    </>
  );
}
