import { m } from 'motion/react';
import { useEffect, useId, useRef, useState, type DragEvent } from 'react';
import { useTranslation } from 'react-i18next';
import { Icon, type IconName } from '../ui/Icon';
import { IMAGE_ACCEPT, middleTruncate, PREVIEWABLE, SPECTRUM_ACCEPT } from './fileTypes';

function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(0)} kB`;
  return `${(n / 1024 / 1024).toFixed(1)} MB`;
}

export interface DropZoneProps {
  id: string;
  label: string;
  hint?: string;
  kind: 'image' | 'spectrum';
  file: File | undefined;
  onFile: (file: File | undefined) => void;
  error?: string;
  required?: boolean;
  icon?: IconName;
  accent?: 'violet' | 'green' | 'neutral';
  compact?: boolean;
}

export function DropZone({
  id,
  label,
  hint,
  kind,
  file,
  onFile,
  error,
  required,
  icon,
  accent = 'neutral',
  compact = false,
}: DropZoneProps) {
  const { t } = useTranslation();
  const input = useRef<HTMLInputElement>(null);
  const [over, setOver] = useState(false);
  const [preview, setPreview] = useState<string | null>(null);
  const hintId = useId();
  const errId = `${id}-error`;

  useEffect(() => {
    if (
      !file ||
      kind !== 'image' ||
      !PREVIEWABLE.test(file.name) ||
      typeof URL.createObjectURL !== 'function'
    ) {
      setPreview(null);
      return;
    }
    const url = URL.createObjectURL(file);
    setPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [file, kind]);

  const onDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setOver(false);
    const f = e.dataTransfer.files?.[0];
    if (f) onFile(f);
  };

  const ring =
    accent === 'violet'
      ? 'from-[oklch(0.55_0.25_290/0.16)]'
      : accent === 'green'
        ? 'from-[oklch(0.8_0.2_150/0.14)]'
        : 'from-transparent';

  return (
    <div className="min-w-0">
      <div className="mb-1.5 flex items-baseline justify-between gap-2">
        <label htmlFor={id} className="text-sm font-semibold text-fg">
          {label}
          {required ? (
            <span className="ml-1 text-danger" aria-hidden="true">
              *
            </span>
          ) : (
            <span className="ml-1.5 text-xs font-normal text-fg-subtle">
              ({t('common.optional')})
            </span>
          )}
        </label>
      </div>
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setOver(true);
        }}
        onDragLeave={() => setOver(false)}
        onDrop={onDrop}
        className={`relative overflow-hidden rounded-2xl border-2 border-dashed bg-gradient-to-br has-[input:focus-visible]:outline-2 has-[input:focus-visible]:outline-offset-2 has-[input:focus-visible]:outline-focus ${ring} to-transparent transition-colors ${
          error
            ? 'border-danger'
            : over
              ? 'border-violet bg-violet-soft'
              : file
                ? 'border-line-strong border-solid'
                : 'border-line hover:border-line-strong'
        }`}
        data-testid={`dropzone-${id}`}
      >
        <input
          ref={input}
          id={id}
          type="file"
          accept={kind === 'image' ? IMAGE_ACCEPT : SPECTRUM_ACCEPT}
          className="sr-only"
          aria-label={label}
          aria-invalid={error ? true : undefined}
          aria-required={required || undefined}
          aria-describedby={
            [hint ? hintId : null, error ? errId : null].filter(Boolean).join(' ') || undefined
          }
          data-testid={`file-${id}`}
          // The input is visually hidden: bring the whole drop zone clear of the sticky bars.
          onFocus={(e) =>
            e.currentTarget
              .closest('[data-testid^="dropzone"]')
              ?.scrollIntoView({ block: 'nearest' })
          }
          onChange={(e) => {
            const f = e.target.files?.[0];
            onFile(f ?? undefined);
            e.target.value = '';
          }}
        />
        {file ? (
          <m.div
            className={`flex min-w-0 items-start gap-3 p-3 ${compact ? '' : 'sm:p-4'}`}
            initial={{ opacity: 0, scale: 0.98 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ type: 'spring', stiffness: 400, damping: 30 }}
          >
            <div className="flex h-16 w-20 shrink-0 items-center justify-center overflow-hidden rounded-xl bg-image-well">
              {preview ? (
                <img src={preview} alt="" className="h-full w-full object-cover" />
              ) : (
                <Icon
                  name={kind === 'image' ? 'camera' : 'wave'}
                  size={24}
                  className="text-fg-subtle"
                />
              )}
            </div>
            <div className="min-w-0 flex-1">
              {hint ? (
                <span id={hintId} className="sr-only">
                  {hint}
                </span>
              ) : null}
              <p
                className="line-clamp-2 text-sm leading-snug font-semibold break-all"
                title={file.name}
                data-testid={`filename-${id}`}
              >
                {middleTruncate(file.name)}
              </p>
              <p className="mt-0.5 text-xs text-fg-subtle tabular">
                {formatBytes(file.size)}
                {kind === 'image' && !preview ? ` · ${t('upload.captures.noPreview')}` : ''}
              </p>
              <div className="mt-1 -ml-2 flex flex-wrap gap-1">
                <button
                  type="button"
                  className="btn btn-ghost min-h-9 px-2 py-1 text-xs"
                  onClick={() => input.current?.click()}
                >
                  <Icon name="refresh" size={14} />
                  {t('upload.captures.replace')}
                </button>
                <button
                  type="button"
                  className="btn btn-ghost min-h-9 px-2 py-1 text-xs"
                  onClick={() => onFile(undefined)}
                  aria-label={`${t('common.remove')}: ${label}`}
                >
                  <Icon name="x" size={14} />
                  {t('common.remove')}
                </button>
              </div>
            </div>
          </m.div>
        ) : (
          <button
            type="button"
            onClick={() => input.current?.click()}
            className={`flex w-full flex-col items-center justify-center gap-1.5 px-4 text-center ${compact ? 'min-h-24 py-4' : 'min-h-32 py-6'}`}
            aria-describedby={hint ? hintId : undefined}
            tabIndex={-1}
          >
            <span className="flex h-10 w-10 items-center justify-center rounded-xl bg-surface-2 text-fg-muted">
              <Icon name={icon ?? (kind === 'image' ? 'camera' : 'wave')} size={20} />
            </span>
            <span className="text-sm text-fg-muted">
              {t('upload.captures.drop')}{' '}
              <span className="font-semibold text-violet-text underline underline-offset-2">
                {t('upload.captures.browse')}
              </span>
            </span>
            <span id={hintId} className="text-xs text-fg-subtle">
              {hint ? `${hint} · ` : ''}
              {kind === 'image'
                ? t('upload.captures.acceptedImages')
                : t('upload.captures.acceptedSpectrum')}
            </span>
          </button>
        )}
      </div>
      {error ? (
        <p
          id={errId}
          className="mt-1.5 flex items-start gap-1.5 text-sm font-medium text-danger"
          data-testid="field-error"
        >
          <Icon name="alert" size={16} className="mt-0.5 shrink-0" />
          <span>{error}</span>
        </p>
      ) : null}
    </div>
  );
}
