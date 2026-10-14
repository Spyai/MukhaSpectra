import { AnimatePresence, m } from 'motion/react';
import { useEffect, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import type { ScreeningResult, StageTiming } from '../../api/types';
import { formatDuration } from '../../lib/format';
import { usePrefersReducedMotion } from '../../lib/useMediaQuery';
import { Icon } from '../ui/Icon';

export const DISPLAY_STAGES = ['quality', 'align', 'tissue', 'af', 'spectral', 'triage', 'report'] as const;
export type DisplayStage = (typeof DISPLAY_STAGES)[number];

const STAGE_PATTERNS: Array<[DisplayStage, RegExp]> = [
  ['quality', /qc|quality|decode|calib|dark/i],
  ['align', /regist|align|warp/i],
  ['tissue', /segment|mucosa|tissue|mask/i],
  ['spectral', /spectr|probe/i],
  ['af', /(^|_)af|fluor|fvl|confound|heatmap|overlay|artifact|grid/i],
  ['triage', /triage|fusion|rule|lesion/i],
  ['report', /report|pdf|render/i],
];

export interface StageSummary {
  stage: DisplayStage;
  durationMs: number | null;
  status: 'done' | 'warn' | 'skipped' | 'not-timed';
}

/** Aggregate real pipeline stage timings (per site) into the seven display stages. */
export function summariseStages(stages: StageTiming[]): StageSummary[] {
  const buckets = new Map<DisplayStage, StageTiming[]>();
  for (const s of stages) {
    const hit = STAGE_PATTERNS.find(([, re]) => re.test(s.stage));
    if (!hit) continue;
    const list = buckets.get(hit[0]) ?? [];
    list.push(s);
    buckets.set(hit[0], list);
  }
  return DISPLAY_STAGES.map((stage) => {
    const list = buckets.get(stage);
    if (!list || list.length === 0) return { stage, durationMs: null, status: 'not-timed' as const };
    const durationMs = list.reduce((a, b) => a + b.duration_ms, 0);
    const status = list.some((s) => s.status === 'failed')
      ? ('warn' as const)
      : list.every((s) => s.status === 'skipped')
        ? ('skipped' as const)
        : ('done' as const);
    return { stage, durationMs, status };
  });
}

// Expected relative pacing while we wait (purely cosmetic; never shown as numbers).
const PACE_MS = [500, 650, 450, 700, 350, 300];

export interface AnalysisOverlayProps {
  open: boolean;
  mode: 'upload' | 'demo' | 'reanalyse';
  thumbnailUrl?: string | null;
  result?: ScreeningResult | null;
  error?: string | null;
  onContinue: () => void;
  onClose: () => void;
  autoContinueMs?: number;
}

export function AnalysisOverlay(props: AnalysisOverlayProps) {
  return <AnimatePresence>{props.open ? <OverlayBody {...props} /> : null}</AnimatePresence>;
}

function OverlayBody({
  mode,
  thumbnailUrl,
  result,
  error,
  onContinue,
  onClose,
  autoContinueMs = 2200,
}: AnalysisOverlayProps) {
  const { t } = useTranslation();
  const reduced = usePrefersReducedMotion();
  const [active, setActive] = useState(0);
  const titleRef = useRef<HTMLHeadingElement>(null);
  const continueRef = useRef<HTMLButtonElement>(null);
  const done = Boolean(result);
  const failed = Boolean(error);

  // Advance the waiting animation; hold on "Triage" until the server answers.
  useEffect(() => {
    if (done || failed) return;
    let i = 0;
    let timer: ReturnType<typeof setTimeout>;
    const tick = () => {
      if (i >= PACE_MS.length - 1) return;
      timer = setTimeout(() => {
        i += 1;
        setActive(i);
        tick();
      }, PACE_MS[i]);
    };
    tick();
    return () => clearTimeout(timer);
  }, [done, failed]);

  useEffect(() => {
    titleRef.current?.focus();
    const prev = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    return () => {
      document.body.style.overflow = prev;
    };
  }, []);

  useEffect(() => {
    if (!done) return;
    continueRef.current?.focus();
    const timer = setTimeout(onContinue, autoContinueMs);
    return () => clearTimeout(timer);
  }, [done, onContinue, autoContinueMs]);

  const summary = useMemo(() => (result ? summariseStages(result.processing.stages ?? []) : null), [result]);
  const title =
    mode === 'demo' ? t('analysis.titleDemo') : mode === 'reanalyse' ? t('analysis.titleReanalyse') : t('analysis.title');

  return (
    <m.div
      className="fixed inset-0 z-[70] flex items-end justify-center bg-[oklch(0.1_0.02_268/0.72)] p-0 backdrop-blur-sm sm:items-center sm:p-6"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.2 }}
    >
      <m.div
        role="dialog"
        aria-modal="true"
        aria-labelledby="analysis-title"
        data-testid="analysis-overlay"
        className="card max-h-[92dvh] w-full max-w-3xl overflow-y-auto rounded-b-none p-5 sm:rounded-b-[1.25rem] sm:p-7"
        initial={{ y: 32, opacity: 0, scale: 0.98 }}
        animate={{ y: 0, opacity: 1, scale: 1 }}
        exit={{ y: 24, opacity: 0 }}
        transition={{ type: 'spring', stiffness: 300, damping: 30 }}
        onKeyDown={(e) => {
          if (e.key === 'Escape' && (done || failed)) (done ? onContinue : onClose)();
        }}
      >
        <div className="grid gap-6 sm:grid-cols-[minmax(0,0.9fr)_minmax(0,1.1fr)]">
          <div>
            <h2 id="analysis-title" ref={titleRef} tabIndex={-1} className="text-xl font-bold tracking-tight outline-none">
              {failed ? t('analysis.failed') : done ? t('analysis.done') : title}
            </h2>
            <p className="mt-1 text-sm text-fg-muted">
              {failed ? error : done ? t('analysis.doneBody') : t('analysis.subtitle')}
            </p>
            <div className="relative mt-5 aspect-[4/3] overflow-hidden rounded-xl border border-line bg-image-well">
              {thumbnailUrl ? (
                <img src={thumbnailUrl} alt={t('analysis.thumbnailAlt')} className="h-full w-full object-cover" />
              ) : (
                <div
                  aria-hidden="true"
                  className="h-full w-full bg-[radial-gradient(60%_55%_at_45%_55%,oklch(0.62_0.12_10),oklch(0.32_0.08_10)_70%,oklch(0.14_0.03_268))]"
                />
              )}
              {!done && !failed ? (
                <>
                  <m.div
                    aria-hidden="true"
                    className="absolute inset-y-0 left-0 bg-[oklch(0.8_0.2_150/0.22)] mix-blend-screen"
                    initial={{ width: '0%' }}
                    animate={reduced ? { width: '50%' } : { width: ['0%', '100%'] }}
                    transition={reduced ? { duration: 0 } : { duration: 1.9, repeat: Infinity, ease: 'linear' }}
                  />
                  <m.div
                    aria-hidden="true"
                    className="scan-line absolute inset-y-0 w-10 -translate-x-1/2"
                    initial={{ left: '0%' }}
                    animate={reduced ? { left: '50%' } : { left: ['0%', '100%'] }}
                    transition={reduced ? { duration: 0 } : { duration: 1.9, repeat: Infinity, ease: 'linear' }}
                  />
                </>
              ) : null}
              {done ? (
                <m.div
                  aria-hidden="true"
                  className="absolute inset-0 flex items-center justify-center bg-[oklch(0.8_0.2_150/0.16)]"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                >
                  <m.span
                    className="flex h-14 w-14 items-center justify-center rounded-full bg-[oklch(0.8_0.2_150)] text-[oklch(0.2_0.05_150)] shadow-[0_0_40px_oklch(0.8_0.2_150/0.6)]"
                    initial={{ scale: 0.5 }}
                    animate={{ scale: 1 }}
                    transition={{ type: 'spring', stiffness: 380, damping: 16 }}
                  >
                    <Icon name="check" size={30} strokeWidth={2.5} />
                  </m.span>
                </m.div>
              ) : null}
            </div>
          </div>

          <div>
            <ol aria-label={t('analysis.progressLabel')} className="relative flex flex-col gap-1">
              {DISPLAY_STAGES.map((stage, i) => {
                const s = summary?.[i];
                const state: 'waiting' | 'running' | 'done' | 'warn' | 'skipped' | 'not-timed' | 'failed' = failed
                  ? i <= active
                    ? i === active
                      ? 'failed'
                      : 'done'
                    : 'waiting'
                  : s
                    ? s.status
                    : i < active
                      ? 'done'
                      : i === active
                        ? 'running'
                        : 'waiting';
                return (
                  <li key={stage} className="relative flex items-center gap-3 rounded-xl px-2 py-2" data-testid={`stage-${stage}`} data-state={state}>
                    {i < DISPLAY_STAGES.length - 1 ? (
                      <span aria-hidden="true" className="absolute top-10 left-[1.4rem] h-[calc(100%-1.5rem)] w-px bg-line" />
                    ) : null}
                    <StageDot state={state} />
                    <span className={`flex-1 text-sm font-semibold ${state === 'waiting' ? 'text-fg-subtle' : 'text-fg'}`}>
                      {t(`analysis.stages.${stage}`)}
                    </span>
                    <span className="text-xs text-fg-subtle tabular">
                      {s
                        ? s.durationMs !== null && s.status !== 'skipped'
                          ? formatDuration(s.durationMs)
                          : s.status === 'skipped'
                            ? t('analysis.stageSkipped')
                            : t('analysis.notMeasured')
                        : state === 'running'
                          ? t('analysis.stageRunning')
                          : state === 'done'
                            ? t('analysis.stageDone')
                            : state === 'failed'
                              ? t('analysis.stageFailed')
                              : ''}
                    </span>
                  </li>
                );
              })}
            </ol>
            <p className="sr-only" aria-live="polite">
              {failed
                ? t('analysis.failed')
                : done
                  ? t('analysis.done')
                  : t(`analysis.stages.${DISPLAY_STAGES[active]!}`)}
            </p>
            {result ? (
              <p className="mt-3 border-t border-line pt-3 text-sm text-fg-muted tabular">
                {t('analysis.totalTime', { value: formatDuration(result.processing.duration_ms) })}
              </p>
            ) : null}
            <div className="mt-5 flex justify-end gap-3">
              {failed ? (
                <button type="button" className="btn btn-secondary" onClick={onClose}>
                  <Icon name="arrowLeft" size={18} />
                  {t('analysis.back')}
                </button>
              ) : null}
              {done ? (
                <button ref={continueRef} type="button" className="btn btn-primary" onClick={onContinue}>
                  {t('analysis.viewResults')}
                  <Icon name="arrowRight" size={18} />
                </button>
              ) : null}
            </div>
          </div>
        </div>
      </m.div>
    </m.div>
  );
}

function StageDot({ state }: { state: string }) {
  const base = 'relative z-10 flex h-7 w-7 shrink-0 items-center justify-center rounded-full border-2';
  if (state === 'done')
    return (
      <span className={`${base} border-glow bg-glow-soft text-glow-text`}>
        <Icon name="check" size={15} strokeWidth={2.6} />
      </span>
    );
  if (state === 'warn')
    return (
      <span className={`${base} border-warn bg-risk-moderate-bg text-warn`}>
        <Icon name="alert" size={14} strokeWidth={2.2} />
      </span>
    );
  if (state === 'failed')
    return (
      <span className={`${base} border-danger text-danger`}>
        <Icon name="x" size={14} strokeWidth={2.6} />
      </span>
    );
  if (state === 'running')
    return (
      <span className={`${base} border-violet bg-violet-soft`}>
        <span className="h-2.5 w-2.5 animate-ping rounded-full bg-violet" />
      </span>
    );
  if (state === 'skipped' || state === 'not-timed')
    return <span className={`${base} border-dashed border-line-strong text-fg-subtle`}>–</span>;
  return <span className={`${base} border-line`} />;
}
