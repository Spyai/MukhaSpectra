/** Brand mark: a violet beam meets tissue that glows green, except a dark patch (FVL). */
export function LogoMark({ size = 28 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 32 32" aria-hidden="true" focusable="false">
      <defs>
        <radialGradient id="lm-glow" cx="50%" cy="50%" r="50%">
          <stop offset="0%" stopColor="oklch(0.88 0.2 150)" />
          <stop offset="70%" stopColor="oklch(0.72 0.2 150)" />
          <stop offset="100%" stopColor="oklch(0.55 0.16 150)" />
        </radialGradient>
        <linearGradient id="lm-beam" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stopColor="oklch(0.62 0.25 290 / 0)" />
          <stop offset="100%" stopColor="oklch(0.62 0.25 290)" />
        </linearGradient>
      </defs>
      <path d="M2 2 L15 13 L11 17 Z" fill="url(#lm-beam)" />
      <circle cx="19" cy="19" r="10" fill="url(#lm-glow)" />
      <ellipse
        cx="22"
        cy="21.5"
        rx="3.6"
        ry="2.6"
        fill="oklch(0.25 0.04 160)"
        transform="rotate(-20 22 21.5)"
      />
    </svg>
  );
}
