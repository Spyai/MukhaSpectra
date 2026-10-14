import { m } from 'motion/react';
import { useTranslation } from 'react-i18next';
import type { RiskLevel } from '../../api/types';
import { RISK_STYLE } from '../../lib/risk';

/**
 * Level glyph that never relies on colour: ascending bars (1/2/3 lit) for LOW/MODERATE/HIGH and
 * a "?" in a dashed circle for INCONCLUSIVE.
 */
export function RiskGlyph({ level, size = 20 }: { level: RiskLevel; size?: number }) {
  const style = RISK_STYLE[level];
  if (level === 'INCONCLUSIVE') {
    return (
      <svg width={size} height={size} viewBox="0 0 24 24" aria-hidden="true" focusable="false">
        <circle
          cx="12"
          cy="12"
          r="9.5"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.8"
          strokeDasharray="3 2.6"
        />
        <path
          d="M9.4 9.4a2.7 2.7 0 1 1 3.9 2.4c-.8.4-1.3 1-1.3 1.9v.5"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.9"
          strokeLinecap="round"
        />
        <circle cx="12" cy="17.2" r="1.15" fill="currentColor" />
      </svg>
    );
  }
  const bars = [
    { x: 4, h: 7 },
    { x: 10, h: 11 },
    { x: 16, h: 15 },
  ];
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" aria-hidden="true" focusable="false">
      {bars.map((b, i) => (
        <rect
          key={b.x}
          x={b.x}
          y={20 - b.h}
          width="4.2"
          height={b.h}
          rx="1.2"
          fill={i < style.steps ? 'currentColor' : 'none'}
          stroke="currentColor"
          strokeWidth="1.4"
          opacity={i < style.steps ? 1 : 0.45}
        />
      ))}
    </svg>
  );
}

/** Compact pill: glyph + word. Used in lists, maps and lesion cards. */
export function RiskPill({ level, className = '' }: { level: RiskLevel; className?: string }) {
  const { t } = useTranslation();
  const style = RISK_STYLE[level];
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-bold tracking-wide uppercase ${className}`}
      style={{
        color: style.color,
        background: style.bg,
        border: `1.5px ${style.dashed ? 'dashed' : 'solid'} ${style.color}`,
      }}
      data-testid="risk-pill"
      data-level={level}
    >
      <RiskGlyph level={level} size={14} />
      {t(`risk.level.${level}`)}
    </span>
  );
}

const R = 52;
const C = 60;
const polar = (deg: number) => {
  const rad = ((deg - 90) * Math.PI) / 180;
  return [C + R * Math.cos(rad), C + R * Math.sin(rad)] as const;
};
const arc = (from: number, to: number) => {
  const [x1, y1] = polar(from);
  const [x2, y2] = polar(to);
  const large = to - from > 180 ? 1 : 0;
  return `M ${x1.toFixed(2)} ${y1.toFixed(2)} A ${R} ${R} 0 ${large} 1 ${x2.toFixed(2)} ${y2.toFixed(2)}`;
};
// 270° gauge from -135° to +135°, three segments with 8° gaps.
const SEGMENTS = [arc(-135, -49), arc(-41, 41), arc(49, 135)];

/**
 * Large risk badge: a three-step level gauge (not a probability) that draws itself in,
 * the level glyph in the centre and the level in words.
 */
export function RiskBadge({ level, size = 132 }: { level: RiskLevel; size?: number }) {
  const { t } = useTranslation();
  const style = RISK_STYLE[level];
  const label = t(`risk.levelLong.${level}`);
  return (
    <div
      className="flex items-center gap-4"
      data-testid="risk-badge"
      data-level={level}
      role="img"
      aria-label={t('risk.badgeLabel', { level: label })}
    >
      <div className="relative shrink-0" style={{ width: size, height: size }}>
        <m.div
          className="absolute inset-3 rounded-full blur-2xl"
          style={{ background: style.color }}
          initial={{ opacity: 0 }}
          animate={{ opacity: 0.22 }}
          transition={{ duration: 0.6, delay: 0.25 }}
        />
        <svg
          viewBox="0 0 120 120"
          width={size}
          height={size}
          className="relative"
          aria-hidden="true"
        >
          {style.dashed ? (
            <m.circle
              cx={C}
              cy={C}
              r={R}
              fill="none"
              stroke={style.color}
              strokeWidth="7"
              strokeDasharray="6 7"
              strokeLinecap="round"
              initial={{ opacity: 0, rotate: -40 }}
              animate={{ opacity: 1, rotate: 0 }}
              transition={{ type: 'spring', stiffness: 120, damping: 20 }}
              style={{ originX: '50%', originY: '50%' }}
            />
          ) : (
            SEGMENTS.map((d, i) => (
              <g key={d}>
                <path
                  d={d}
                  fill="none"
                  stroke="var(--line)"
                  strokeWidth="7"
                  strokeLinecap="round"
                />
                {i < style.steps ? (
                  <m.path
                    d={d}
                    fill="none"
                    stroke={style.color}
                    strokeWidth="7"
                    strokeLinecap="round"
                    initial={{ pathLength: 0, opacity: 0 }}
                    animate={{ pathLength: 1, opacity: 1 }}
                    transition={{
                      duration: 0.38,
                      delay: 0.12 + i * 0.16,
                      ease: [0.22, 1, 0.36, 1],
                    }}
                  />
                ) : null}
              </g>
            ))
          )}
        </svg>
        <m.div
          className="absolute inset-0 flex items-center justify-center"
          style={{ color: style.color }}
          initial={{ scale: 0.6, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ type: 'spring', stiffness: 300, damping: 18, delay: 0.2 }}
        >
          <RiskGlyph level={level} size={Math.round(size * 0.36)} />
        </m.div>
      </div>
      <div className="min-w-0">
        <div className="eyebrow">{t('risk.eyebrow')}</div>
        <m.div
          className="text-4xl leading-tight font-extrabold tracking-tight sm:text-5xl"
          style={{ color: style.color }}
          initial={{ opacity: 0, x: -8 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ type: 'spring', stiffness: 260, damping: 26, delay: 0.3 }}
        >
          {t(`risk.level.${level}`)}
        </m.div>
      </div>
    </div>
  );
}
