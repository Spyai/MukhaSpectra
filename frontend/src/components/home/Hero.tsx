import { lazy, Suspense, useEffect, useState } from 'react';
import { m } from 'motion/react';
import { useTranslation } from 'react-i18next';
import { Link } from 'react-router';
import { hasWebGL } from '../../lib/webgl';
import { usePrefersReducedMotion } from '../../lib/useMediaQuery';
import { Icon } from '../ui/Icon';

const HeroScene = lazy(() => import('../../three/HeroScene'));

/** Static illustration of the same principle (no WebGL, reduced motion, or while loading). */
export function HeroFallback() {
  return (
    <svg viewBox="0 0 640 400" className="h-full w-full" aria-hidden="true" preserveAspectRatio="xMidYMid slice">
      <defs>
        <radialGradient id="hf-glow" cx="42%" cy="62%" r="55%">
          <stop offset="0%" stopColor="oklch(0.82 0.2 150)" stopOpacity="0.95" />
          <stop offset="55%" stopColor="oklch(0.62 0.17 150)" stopOpacity="0.55" />
          <stop offset="100%" stopColor="oklch(0.3 0.05 150)" stopOpacity="0" />
        </radialGradient>
        <linearGradient id="hf-beam" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="oklch(0.75 0.2 290)" stopOpacity="0.85" />
          <stop offset="100%" stopColor="oklch(0.55 0.25 290)" stopOpacity="0.08" />
        </linearGradient>
        <filter id="hf-blur">
          <feGaussianBlur stdDeviation="10" />
        </filter>
        <pattern id="hf-tex" width="9" height="9" patternUnits="userSpaceOnUse">
          <circle cx="2" cy="2" r="0.9" fill="oklch(1 0 0 / 0.06)" />
          <circle cx="6.5" cy="6" r="0.7" fill="oklch(1 0 0 / 0.05)" />
        </pattern>
      </defs>
      <path d="M0 250 C 120 205, 230 300, 360 250 S 560 205, 640 245 L640 400 L0 400 Z" fill="oklch(0.2 0.05 330 / 0.9)" />
      <ellipse cx="300" cy="300" rx="300" ry="95" fill="url(#hf-glow)" filter="url(#hf-blur)" />
      <path d="M0 250 C 120 205, 230 300, 360 250 S 560 205, 640 245 L640 400 L0 400 Z" fill="url(#hf-tex)" />
      <path
        d="M402 286 c 18 -14 52 -16 66 0 c 14 16 6 34 -14 40 c -22 6 -48 2 -58 -14 c -6 -10 -2 -18 6 -26 Z"
        fill="oklch(0.14 0.03 160)"
        opacity="0.92"
      />
      <ellipse cx="436" cy="305" rx="62" ry="40" fill="none" stroke="white" strokeOpacity="0.8" strokeWidth="2" strokeDasharray="7 7" />
      <path d="M300 30 L 210 280 L 390 280 Z" fill="url(#hf-beam)" opacity="0.55" />
      <circle cx="300" cy="30" r="6" fill="oklch(0.85 0.12 290)" />
    </svg>
  );
}

export function Hero() {
  const { t } = useTranslation();
  const reduced = usePrefersReducedMotion();
  const [webgl, setWebgl] = useState<boolean | null>(null);
  useEffect(() => setWebgl(hasWebGL()), []);
  const show3d = webgl === true && !reduced;

  return (
    <section className="relative overflow-hidden border-b border-line" aria-labelledby="hero-title">
      <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(60%_80%_at_80%_10%,var(--violet-soft),transparent_60%),radial-gradient(50%_70%_at_10%_100%,var(--green-soft),transparent_65%)]" />
      <div className="relative mx-auto grid max-w-7xl gap-6 px-4 pt-8 pb-10 sm:px-6 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.05fr)] lg:items-center lg:gap-10 lg:pt-14 lg:pb-16">
        <m.div
          initial="hidden"
          animate="show"
          variants={{ hidden: {}, show: { transition: { staggerChildren: 0.09 } } }}
          className="order-2 lg:order-1"
        >
          <m.p variants={rise} className="eyebrow">
            {t('home.eyebrow')}
          </m.p>
          <h1 id="hero-title" className="mt-3 text-[2.1rem] leading-[1.08] font-extrabold tracking-tight sm:text-5xl lg:text-[3.4rem]">
            <m.span variants={rise} className="block violet-text">
              {t('home.heroTitle1')}
            </m.span>
            <m.span variants={rise} className="block glow-text">
              {t('home.heroTitle2')}
            </m.span>
            <m.span variants={rise} className="mt-1 block text-[1.45rem] leading-snug font-bold text-fg sm:text-3xl">
              {t('home.heroTitle3')}
            </m.span>
          </h1>
          <m.p variants={rise} className="mt-5 max-w-xl text-base leading-relaxed text-fg-muted sm:text-lg">
            {t('home.heroBody')}
          </m.p>
          <m.div variants={rise} className="mt-7 flex flex-wrap gap-3">
            <Link to="/upload" className="btn btn-primary min-h-12 px-5 text-base">
              <Icon name="upload" />
              {t('home.ctaUpload')}
            </Link>
            <a href="#demos" className="btn btn-secondary min-h-12 px-5 text-base">
              <Icon name="play" size={16} />
              {t('home.ctaDemo')}
            </a>
          </m.div>
        </m.div>

        <div className="order-1 lg:order-2">
          <div
            className="relative aspect-[16/10] w-full overflow-hidden rounded-[1.75rem] border border-line bg-[oklch(0.12_0.02_268)] shadow-card dark:bg-[oklch(0.1_0.02_268)]"
            role="img"
            aria-label={t('home.sceneLabel')}
          >
            {show3d ? (
              <Suspense fallback={<HeroFallback />}>
                <HeroScene reduced={reduced} dark={true} />
              </Suspense>
            ) : (
              <HeroFallback />
            )}
            <div className="pointer-events-none absolute inset-x-3 bottom-3 flex flex-wrap gap-2 text-[11px] font-semibold text-white sm:text-xs">
              <span className="flex items-center gap-1.5 rounded-full bg-black/55 px-2.5 py-1 backdrop-blur">
                <span className="h-2.5 w-2.5 rounded-full bg-[oklch(0.62_0.25_290)]" />
                {t('home.sceneLegendBeam')}
              </span>
              <span className="flex items-center gap-1.5 rounded-full bg-black/55 px-2.5 py-1 backdrop-blur">
                <span className="h-2.5 w-2.5 rounded-full bg-[oklch(0.82_0.2_150)] shadow-[0_0_8px_oklch(0.82_0.2_150)]" />
                {t('home.sceneLegendGlow')}
              </span>
              <span className="flex items-center gap-1.5 rounded-full bg-black/55 px-2.5 py-1 backdrop-blur">
                <span className="h-2.5 w-2.5 rounded-full border border-dashed border-white bg-[oklch(0.16_0.03_160)]" />
                {t('home.sceneLegendLoss')}
              </span>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

const rise = {
  hidden: { opacity: 0, y: 14 },
  show: { opacity: 1, y: 0, transition: { type: 'spring' as const, stiffness: 220, damping: 28 } },
};
