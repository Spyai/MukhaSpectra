import {
  useCallback,
  useRef,
  useState,
  type KeyboardEvent,
  type PointerEvent,
  type ReactNode,
} from 'react';
import { useTranslation } from 'react-i18next';

export interface WipeSliderProps {
  leftSrc: string;
  rightSrc: string;
  leftAlt: string;
  rightAlt: string;
  leftLabel: string;
  rightLabel: string;
  /** Width / height of the images. */
  aspect?: number;
  /** Optional overlay image (e.g. heatmap) drawn over both sides. */
  overlaySrc?: string | null;
  overlayOpacity?: number;
  /** Absolutely positioned content over the images (region markers). */
  children?: ReactNode;
  initial?: number;
  value?: number;
  onChange?: (value: number) => void;
}

const clamp = (v: number) => Math.min(100, Math.max(0, v));

/**
 * WL/AF wipe comparison. The handle is a WAI-ARIA slider: arrows move 2%, Shift+arrows or
 * PageUp/PageDown move 10%, Home/End jump to the ends. Pointer dragging works anywhere on the image.
 */
export function WipeSlider({
  leftSrc,
  rightSrc,
  leftAlt,
  rightAlt,
  leftLabel,
  rightLabel,
  aspect = 4 / 3,
  overlaySrc,
  overlayOpacity = 0.7,
  children,
  initial = 50,
  value,
  onChange,
}: WipeSliderProps) {
  const { t } = useTranslation();
  const [inner, setInner] = useState(initial);
  const pos = value ?? inner;
  const box = useRef<HTMLDivElement>(null);
  const dragging = useRef(false);

  const set = useCallback(
    (v: number) => {
      const next = Math.round(clamp(v) * 10) / 10;
      if (value === undefined) setInner(next);
      onChange?.(next);
    },
    [onChange, value],
  );

  const fromPointer = (clientX: number) => {
    const r = box.current?.getBoundingClientRect();
    if (!r || r.width === 0) return;
    set(((clientX - r.left) / r.width) * 100);
  };

  const onPointerDown = (e: PointerEvent<HTMLDivElement>) => {
    if ((e.target as HTMLElement).closest('[data-region-marker]')) return;
    dragging.current = true;
    (e.currentTarget as HTMLDivElement).setPointerCapture?.(e.pointerId);
    fromPointer(e.clientX);
  };
  const onPointerMove = (e: PointerEvent<HTMLDivElement>) => {
    if (dragging.current) fromPointer(e.clientX);
  };
  const onPointerUp = () => {
    dragging.current = false;
  };

  const onKeyDown = (e: KeyboardEvent<HTMLDivElement>) => {
    const big = e.shiftKey ? 10 : 2;
    const map: Record<string, number> = {
      ArrowLeft: pos - big,
      ArrowDown: pos - big,
      ArrowRight: pos + big,
      ArrowUp: pos + big,
      PageDown: pos - 10,
      PageUp: pos + 10,
      Home: 0,
      End: 100,
    };
    if (e.key in map) {
      e.preventDefault();
      set(map[e.key]!);
    }
  };

  return (
    <div
      ref={box}
      className="relative w-full touch-pan-y overflow-hidden rounded-xl bg-image-well select-none"
      style={{ aspectRatio: String(aspect) }}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      onPointerCancel={onPointerUp}
      data-testid="wipe-slider"
    >
      <img
        src={rightSrc}
        alt={rightAlt}
        className="absolute inset-0 h-full w-full object-contain"
        draggable={false}
      />
      <img
        src={leftSrc}
        alt={leftAlt}
        className="absolute inset-0 h-full w-full object-contain"
        style={{ clipPath: `inset(0 ${100 - pos}% 0 0)` }}
        draggable={false}
      />
      {overlaySrc ? (
        <img
          src={overlaySrc}
          alt=""
          aria-hidden="true"
          className="pointer-events-none absolute inset-0 h-full w-full object-contain transition-opacity duration-200"
          style={{ opacity: overlayOpacity }}
          draggable={false}
          data-testid="heatmap-layer"
        />
      ) : null}

      <span className="pointer-events-none absolute top-2.5 left-2.5 rounded-md bg-black/60 px-2 py-1 text-xs font-semibold text-white">
        {leftLabel}
      </span>
      <span className="pointer-events-none absolute top-2.5 right-2.5 rounded-md bg-black/60 px-2 py-1 text-xs font-semibold text-[oklch(0.9_0.16_150)]">
        {rightLabel}
      </span>

      {children}

      {/* divider + handle */}
      <div className="pointer-events-none absolute inset-y-0 w-0" style={{ left: `${pos}%` }}>
        <div className="absolute inset-y-0 -left-px w-0.5 bg-white/90 shadow-[0_0_12px_2px_oklch(0.55_0.25_290/0.7)]" />
      </div>
      <div
        role="slider"
        tabIndex={0}
        aria-label={t('results.viewer.sliderLabel')}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={pos}
        aria-valuetext={t('results.viewer.sliderValue', { value: Math.round(pos) })}
        aria-orientation="horizontal"
        onKeyDown={onKeyDown}
        className="absolute top-1/2 flex h-11 w-11 -translate-x-1/2 -translate-y-1/2 cursor-ew-resize items-center justify-center rounded-full border-2 border-white bg-violet text-white shadow-lg outline-none focus-visible:ring-4 focus-visible:ring-focus/60"
        style={{ left: `${pos}%` }}
        data-testid="wipe-handle"
      >
        <svg
          width="20"
          height="20"
          viewBox="0 0 24 24"
          aria-hidden="true"
          fill="none"
          stroke="currentColor"
          strokeWidth="2.2"
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <path d="m9 7-5 5 5 5M15 7l5 5-5 5" />
        </svg>
      </div>
    </div>
  );
}
