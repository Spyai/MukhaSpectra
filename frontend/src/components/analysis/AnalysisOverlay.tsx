import { AnimatePresence, m } from 'motion/react';
import { useEffect, useMemo, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { useTranslation } from 'react-i18next';
import { useApi } from '../../api/ApiContext';
import type { ScreeningResult } from '../../api/types';
import { wlThumbnail } from '../../lib/thumbnail';
import { formatDuration } from '../../lib/format';
import { failedSites } from '../../lib/qc';
import { DISPLAY_STAGES, summariseStages } from '../../lib/stages';
import { usePrefersReducedMotion } from '../../lib/useMediaQuery';
import { SyntheticTag } from '../layout/SyntheticRibbon';
import { Icon } from '../ui/Icon';

// Cosmetic pace of the "working" pulse while we wait. It never marks a step as done: ticks,
// warnings and timings appear only once the server's measured stages arrive.
const PULSE_MS = 650;

export interface AnalysisOverlayProps {
  open: boolean;
  mode: 'upload' | 'demo' | 'reanalyse';
  thumbnailUrl?: string | null;
  /** Known before the server answers (demo runs, re-analysis of a synthetic result). */
  syntheticHint?: boolean;
  result?: ScreeningResult | null;
  error?: string | null;
  onContinue: () => void;
  onClose: () => void;
  autoContinueMs?: number;
}

export function AnalysisOverlay(props: AnalysisOverlayProps) {
  if (typeof document === 'undefined') return null;
  return createPortal(
    <AnimatePresence>{props.open ? <OverlayBody {...props} /> : null}</AnimatePresence>,
    document.body,
  );
}

function OverlayBody({
  mode,
  thumbnailUrl,
  syntheticHint = false,
  result,
  error,
  onContinue,
  onClose,
  autoContinueMs = 2200,
}: AnalysisOverlayProps) {
  const { t } = useTranslation();
  const reduced = usePrefersReducedMotion();
  const [pulse, setPulse] = useState(0);
  const titleRef = useRef<HTMLHeadingElement>(null);
  const continueRef = useRef<HTMLButtonElement>(null);
  const dialogRef = useRef<HTMLDivElement>(null);
  const done = Boolean(result);
  const failed = Boolean(error);
  const synthetic = result?.synthetic ?? syntheticHint;
  const api = useApi();
  const fromResult = useMemo(() => wlThumbnail(api, result), [api, result]);
  const thumb = thumbnailUrl ?? fromResult.url;
  const retakeSites = useMemo(() => (result ? failedSites(result) : []), [result]);
  const needsRetake = retakeSites.length > 0;
  const autoContinue = done && !needsRetake && !reduced;

  // A pulse walks along the steps while waiting (cosmetic only, never "done").
  useEffect(() => {
    if (done || failed || reduced) return;
    const timer = setInterval(() => setPulse((p) => (p + 1) % DISPLAY_STAGES.length), PULSE_MS);
    return () => clearInterval(timer);
  }, [done, failed, reduced]);

  // Modal: the app behind is inert (no focus, no clicks) and does not scroll.
  useEffect(() => {
    titleRef.current?.focus();
    const root = document.getElementById('root');
    const prevInert = root?.inert ?? false;
    if (root) root.inert = true;
    const prev = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    return () => {
      if (root) root.inert = prevInert;
      document.body.style.overflow = prev;
    };
  }, []);

  useEffect(() => {
    if (!done) return;
    continueRef.current?.focus();
    if (!autoContinue) return;
    const timer = setTimeout(onContinue, autoContinueMs);
    return () => clearTimeout(timer);
  }, [done, autoContinue, onContinue, autoContinueMs]);

  const summary = useMemo(
    () => (result ? summariseStages(result.processing.stages ?? []) : null),
    [result],
  );
  const title =
    mode === 'demo'
      ? t('analysis.titleDemo')
      : mode === 'reanalyse'
        ? t('analysis.titleReanalyse')
        : t('analysis.title');

  return (
    <m.div
      className="fixed inset-0 z-[70] flex items-end justify-center bg-[oklch(0.1_0.02_268/0.72)] p-0 backdrop-blur-sm sm:items-center sm:p-6"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.2 }}
    >
      <m.div
        ref={dialogRef}
        role="dialog"
        aria-modal="true"
        aria-labelledby="analysis-title"
        data-testid="analysis-overlay"
        data-synthetic={synthetic}
        className="card max-h-[92dvh] w-full max-w-3xl overflow-y-auto rounded-b-none p-5 sm:rounded-b-[1.25rem] sm:p-7"
        initial={{ y: 32, opacity: 0, scale: 0.98 }}
        animate={{ y: 0, opacity: 1, scale: 1 }}
        exit={{ y: 24, opacity: 0 }}
        transition={{ type: 'spring', stiffness: 300, damping: 30 }}
        onKeyDown={(e) => {
          if (e.key === 'Escape' && (done || failed)) (done ? onContinue : onClose)();
          // Focus trap: the backdrop app is inert, so only wrap Tab within the dialog.
          if (e.key === 'Tab' && dialogRef.current) {
            const items = dialogRef.current.querySelectorAll<HTMLElement>(
              'button:not([disabled]), [href], [tabindex]:not([tabindex="-1"])',
            );
            const first = items[0];
            const last = items[items.length - 1];
            if (!first || !last) {
              e.preventDefault();
              titleRef.current?.focus();
            } else if (e.shiftKey && document.activeElement === first) {
              e.preventDefault();
              last.focus();
            } else if (!e.shiftKey && document.activeElement === last) {
              e.preventDefault();
              first.focus();
            }
          }
        }}
      >
        <div className="grid grid-cols-1 gap-6 sm:grid-cols-[minmax(0,0.9fr)_minmax(0,1.1fr)]">
          <div className="min-w-0">
            <div className="flex flex-wrap items-center gap-2">
              <h2
                id="analysis-title"
                ref={titleRef}
                tabIndex={-1}
                className="text-xl font-bold tracking-tight outline-none"
              >
                {failed ? t('analysis.failed') : done ? t('analysis.done') : title}
              </h2>
              {synthetic ? <SyntheticTag /> : null}
            </div>
            <p className="mt-1 text-sm text-fg-muted">
              {failed
                ? error
                : done
                  ? autoContinue
                    ? t('analysis.doneBodyAuto')
                    : t('analysis.doneBody')
                  : t('analysis.subtitle')}
            </p>
            <div className="relative mt-5 aspect-[4/3] overflow-hidden rounded-xl border border-line bg-image-well">
              {thumb ? (
                <img
                  src={thumb}
                  alt={
                    fromResult.site && !thumbnailUrl
                      ? t('analysis.thumbnailArtifactAlt', {
                          site: t(`sites.${fromResult.site}`),
                        })
                      : t('analysis.thumbnailAlt')
                  }
                  className="h-full w-full object-cover"
                  data-testid="analysis-thumbnail"
                />
              ) : (
                <div
                  className="flex h-full w-full flex-col items-center justify-center gap-2 text-[oklch(0.75_0.03_268)]"
                  data-testid="analysis-thumbnail-placeholder"
                >
                  <Icon name="camera" size={32} />
                  <span className="text-xs font-semibold">
                    {synthetic ? t('analysis.placeholderThumb') : t('analysis.thumbnailAlt')}
                  </span>
                </div>
              )}
              {synthetic ? <SyntheticTag solid className="absolute top-2 left-2" /> : null}
              {!done && !failed ? (
                <>
                  <m.div
                    aria-hidden="true"
                    className="absolute inset-y-0 left-0 bg-[oklch(0.8_0.2_150/0.22)] mix-blend-screen"
                    initial={{ width: '0%' }}
                    animate={reduced ? { width: '50%' } : { width: ['0%', '100%'] }}
                    transition={
                      reduced
                        ? { duration: 0 }
                        : { duration: 1.9, repeat: Infinity, ease: 'linear' }
                    }
                  />
                  <m.div
                    aria-hidden="true"
                    className="scan-line absolute inset-y-0 w-10 -translate-x-1/2"
                    initial={{ left: '0%' }}
                    animate={reduced ? { left: '50%' } : { left: ['0%', '100%'] }}
                    transition={
                      reduced
                        ? { duration: 0 }
                        : { duration: 1.9, repeat: Infinity, ease: 'linear' }
                    }
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

          <div className="min-w-0">
            <ol aria-label={t('analysis.progressLabel')} className="relative flex flex-col gap-1">
              {DISPLAY_STAGES.map((stage, i) => {
                const s = summary?.[i];
                const state:
                  'waiting' | 'pulse' | 'done' | 'warn' | 'skipped' | 'not-timed' | 'failed' =
                  failed ? 'waiting' : s ? s.status : !reduced && i === pulse ? 'pulse' : 'waiting';
                return (
                  <li
                    key={stage}
                    className="relative flex items-center gap-3 rounded-xl px-2 py-2"
                    data-testid={`stage-${stage}`}
                    data-state={state}
                  >
                    {i < DISPLAY_STAGES.length - 1 ? (
                      <span
                        aria-hidden="true"
                        className="absolute top-10 left-[1.4rem] h-[calc(100%-1.5rem)] w-px bg-line"
                      />
                    ) : null}
                    <StageDot state={state} />
                    <span
                      className={`flex-1 text-sm font-semibold ${s ? 'text-fg' : 'text-fg-muted'}`}
                    >
                      {t(`analysis.stages.${stage}`)}
                    </span>
                    <span className="text-xs text-fg-subtle tabular">
                      {s
                        ? s.durationMs !== null && s.status !== 'skipped'
                          ? formatDuration(s.durationMs)
                          : s.status === 'skipped'
                            ? t('analysis.stageSkipped')
                            : t('analysis.notMeasured')
                        : ''}
                    </span>
                  </li>
                );
              })}
            </ol>
            {!done && !failed ? (
              <p className="mt-3 flex items-center gap-2 border-t border-line pt-3 text-xs text-fg-subtle">
                <Icon name="clock" size={14} />
                {t('analysis.waitingNote')}
              </p>
            ) : null}
            <p className="sr-only" aria-live="polite">
              {failed
                ? t('analysis.failed')
                : done
                  ? needsRetake
                    ? `${t('analysis.done')}. ${t('analysis.retakeNeeded', { count: retakeSites.length })}`
                    : t('analysis.done')
                  : t('analysis.waiting')}
            </p>
            {result ? (
              <p className="mt-3 border-t border-line pt-3 text-sm text-fg-muted tabular">
                {t('analysis.totalTime', { value: formatDuration(result.processing.duration_ms) })}
              </p>
            ) : null}
            {needsRetake ? (
              <div
                className="mt-3 flex items-start gap-2.5 rounded-xl border border-warn/60 bg-risk-moderate-bg p-3 text-sm"
                data-testid="analysis-retake"
              >
                <Icon name="camera" size={18} className="mt-0.5 shrink-0 text-warn" />
                <span>
                  <span className="font-semibold">
                    {t('analysis.retakeNeeded', { count: retakeSites.length })}
                  </span>
                  <span className="block text-fg-muted">
                    {retakeSites.map((s) => t(`sites.${s}`)).join(', ')} ·{' '}
                    {t('analysis.retakeNeededBody')}
                  </span>
                </span>
              </div>
            ) : null}
            <div className="mt-5 flex justify-end gap-3">
              {failed ? (
                <button type="button" className="btn btn-secondary" onClick={onClose}>
                  <Icon name="arrowLeft" size={18} />
                  {t('analysis.back')}
                </button>
              ) : null}
              {done ? (
                <button
                  ref={continueRef}
                  type="button"
                  className="btn btn-primary"
                  onClick={onContinue}
                >
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
  const base =
    'relative z-10 flex h-7 w-7 shrink-0 items-center justify-center rounded-full border-2';
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
  if (state === 'pulse')
    return (
      <span className={`${base} border-violet/70`}>
        <span className="h-2.5 w-2.5 animate-ping rounded-full bg-violet" />
      </span>
    );
  if (state === 'skipped' || state === 'not-timed')
    return <span className={`${base} border-dashed border-line-strong text-fg-subtle`}>–</span>;
  return <span className={`${base} border-line`} />;
}
