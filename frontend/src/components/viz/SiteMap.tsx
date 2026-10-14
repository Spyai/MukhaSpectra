import { useMemo, useRef, useState, type KeyboardEvent } from 'react';
import { useTranslation } from 'react-i18next';
import type { OralSite, RiskLevel, SiteCatalogue, SiteCatalogueEntry } from '../../api/types';
import { RISK_STYLE } from '../../lib/risk';
import { nextSiteInDirection, type Dir } from '../../lib/siteNav';

export interface SiteStatus {
  level?: RiskLevel;
  role?: 'examined' | 'reference_only';
}

export interface SiteMapProps {
  catalogue: SiteCatalogue;
  mode: 'select' | 'results';
  selected: OralSite | null;
  onSelect: (site: OralSite) => void;
  /** Suggested / used reference site: drawn with a dashed blue outline. */
  reference?: OralSite | null;
  /** Results mode: per-site risk level and role. Sites without a status are not interactive. */
  statuses?: Partial<Record<OralSite, SiteStatus>>;
  label: string;
  className?: string;
}

const KEY_DIR: Record<string, Dir> = {
  ArrowLeft: 'left',
  ArrowRight: 'right',
  ArrowUp: 'up',
  ArrowDown: 'down',
};

export function SiteMap({
  catalogue,
  mode,
  selected,
  onSelect,
  reference,
  statuses,
  label,
  className = '',
}: SiteMapProps) {
  const { t } = useTranslation();
  const refs = useRef(new Map<OralSite, SVGGElement>());
  const interactive = useMemo(
    () => catalogue.sites.filter((s) => mode === 'select' || Boolean(statuses?.[s.site])),
    [catalogue.sites, mode, statuses],
  );
  const [focusSite, setFocusSite] = useState<OralSite | null>(null);
  // Roving tabindex: the selected site (or the first interactive one) is the tab stop.
  const tabStop =
    focusSite && interactive.some((s) => s.site === focusSite)
      ? focusSite
      : selected && interactive.some((s) => s.site === selected)
        ? selected
        : (interactive[0]?.site ?? null);

  const [vx, vy, vw, vh] = catalogue.viewBox;

  const onKeyDown = (e: KeyboardEvent<SVGGElement>, site: SiteCatalogueEntry) => {
    let target: SiteCatalogueEntry | undefined;
    const dir = KEY_DIR[e.key];
    if (dir) target = nextSiteInDirection(interactive, site, dir);
    else if (e.key === 'Home') target = interactive[0];
    else if (e.key === 'End') target = interactive[interactive.length - 1];
    else if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      onSelect(site.site);
      return;
    } else return;
    e.preventDefault();
    if (target) {
      setFocusSite(target.site);
      refs.current.get(target.site)?.focus();
    }
  };

  const statusText = (s: SiteCatalogueEntry): string => {
    if (mode === 'select') {
      if (s.site === selected) return t('upload.site.selected', { site: t(`sites.${s.site}`) });
      if (s.site === reference) return t('results.map.reference');
      return '';
    }
    const st = statuses?.[s.site];
    if (!st) return t('results.map.notExamined');
    if (st.role === 'reference_only') return t('results.map.reference');
    return st.level ? t(`risk.levelLong.${st.level}`) : '';
  };

  return (
    <div className={className}>
      <svg
        viewBox={`${vx} ${vy} ${vw} ${vh}`}
        className="h-auto w-full select-none"
        role="group"
        aria-label={label}
        data-testid="site-map"
      >
        <defs>
          <radialGradient id="sm-cavity" cx="50%" cy="50%" r="60%">
            <stop offset="0%" stopColor="var(--surface-3)" />
            <stop offset="100%" stopColor="var(--surface-2)" />
          </radialGradient>
          <filter id="sm-glow" x="-50%" y="-50%" width="200%" height="200%">
            <feGaussianBlur stdDeviation="5" />
          </filter>
        </defs>
        {/* Oral cavity silhouette for orientation */}
        <ellipse
          cx={200}
          cy={165}
          rx={196}
          ry={158}
          fill="url(#sm-cavity)"
          stroke="var(--line)"
          strokeWidth={1.5}
        />
        <text
          x={14}
          y={312}
          className="fill-fg-subtle text-[11px] font-semibold"
          aria-hidden="true"
        >
          {t('results.map.patientRight')}
        </text>
        <text
          x={386}
          y={312}
          textAnchor="end"
          className="fill-fg-subtle text-[11px] font-semibold"
          aria-hidden="true"
        >
          {t('results.map.patientLeft')}
        </text>

        {catalogue.sites.map((s) => {
          const { cx, cy, rx, ry } = s.shape;
          const isInteractive = interactive.some((i) => i.site === s.site);
          const isSelected = s.site === selected;
          const st = statuses?.[s.site];
          const isRef = mode === 'select' ? s.site === reference : st?.role === 'reference_only';
          let fill = 'var(--surface)';
          let stroke = 'var(--line-strong)';
          let dash: string | undefined;
          let opacity = 1;
          if (mode === 'results') {
            if (!st) {
              opacity = 0.45;
              fill = 'transparent';
              stroke = 'var(--line)';
            } else if (isRef) {
              fill = 'color-mix(in oklch, var(--ref-blue) 14%, transparent)';
              stroke = 'var(--ref-blue)';
              dash = '5 4';
            } else if (st.level) {
              const rs = RISK_STYLE[st.level];
              fill = rs.bg;
              stroke = rs.color;
              dash = rs.dashed ? '5 4' : undefined;
            }
          } else {
            if (isSelected) {
              fill = 'var(--violet)';
              stroke = 'var(--violet-strong)';
            } else if (isRef) {
              fill = 'color-mix(in oklch, var(--ref-blue) 16%, transparent)';
              stroke = 'var(--ref-blue)';
              dash = '5 4';
            }
          }
          const showLabel = rx >= 28 && ry >= 12;
          const glyphLevel = mode === 'results' && st?.level && !isRef ? st.level : null;
          const status = statusText(s);
          const name = t(`sites.${s.site}`);
          return (
            <g
              key={s.site}
              ref={(el) => {
                if (el) refs.current.set(s.site, el);
                else refs.current.delete(s.site);
              }}
              role={isInteractive ? 'button' : undefined}
              aria-pressed={isInteractive ? isSelected : undefined}
              aria-disabled={!isInteractive ? true : undefined}
              aria-label={status ? t('results.map.siteAria', { site: name, status }) : name}
              tabIndex={isInteractive ? (s.site === tabStop ? 0 : -1) : undefined}
              data-site={s.site}
              data-testid={`site-${s.site}`}
              className={`group outline-none ${isInteractive ? 'cursor-pointer' : ''}`}
              style={{ opacity }}
              onClick={isInteractive ? () => onSelect(s.site) : undefined}
              onKeyDown={isInteractive ? (e) => onKeyDown(e, s) : undefined}
              onFocus={() => setFocusSite(s.site)}
            >
              <title>{status ? `${name} — ${status}` : name}</title>
              {isSelected ? (
                <ellipse
                  cx={cx}
                  cy={cy}
                  rx={rx + 4}
                  ry={ry + 4}
                  fill={mode === 'select' ? 'var(--violet)' : 'var(--fg)'}
                  opacity={mode === 'select' ? 0.45 : 0.18}
                  filter="url(#sm-glow)"
                />
              ) : null}
              <ellipse
                cx={cx}
                cy={cy}
                rx={rx}
                ry={ry}
                fill={fill}
                stroke={stroke}
                strokeWidth={isSelected ? 2.5 : 1.6}
                strokeDasharray={dash}
                className={
                  isInteractive
                    ? 'transition-[filter,stroke-width] duration-150 group-hover:brightness-110 group-hover:[stroke-width:2.6]'
                    : undefined
                }
              />
              {mode === 'results' && isSelected ? (
                <ellipse
                  cx={cx}
                  cy={cy}
                  rx={rx + 3.5}
                  ry={ry + 3.5}
                  fill="none"
                  stroke="var(--fg)"
                  strokeWidth={2}
                />
              ) : null}
              {/* keyboard focus ring */}
              <ellipse
                cx={cx}
                cy={cy}
                rx={rx + 6}
                ry={ry + 6}
                fill="none"
                stroke="var(--focus)"
                strokeWidth={2.5}
                className="opacity-0 group-focus-visible:opacity-100"
              />
              {glyphLevel ? (
                <g
                  transform={
                    showLabel && ry >= 20
                      ? `translate(${cx - 7} ${cy - ry + 8})`
                      : showLabel
                        ? `translate(${cx - rx + 8} ${cy - 7})`
                        : `translate(${cx - 7} ${cy - 7})`
                  }
                  style={{ color: RISK_STYLE[glyphLevel].color }}
                  aria-hidden="true"
                >
                  <GlyphMini level={glyphLevel} />
                </g>
              ) : null}
              {showLabel ? (
                <text
                  x={cx}
                  y={glyphLevel && ry >= 20 ? cy + 8 : cy + 4}
                  textAnchor="middle"
                  aria-hidden="true"
                  className={`pointer-events-none text-[10.5px] font-semibold ${
                    mode === 'select' && isSelected ? 'fill-on-violet' : 'fill-fg-muted'
                  }`}
                >
                  {s.short_label}
                </text>
              ) : null}
            </g>
          );
        })}
      </svg>
    </div>
  );
}

function GlyphMini({ level }: { level: RiskLevel }) {
  if (level === 'INCONCLUSIVE') {
    return (
      <g>
        <circle
          cx="7"
          cy="7"
          r="6"
          fill="var(--surface)"
          stroke="currentColor"
          strokeWidth="1.4"
          strokeDasharray="2 1.6"
        />
        <text x="7" y="10.2" textAnchor="middle" fontSize="9" fontWeight="800" fill="currentColor">
          ?
        </text>
      </g>
    );
  }
  const steps = RISK_STYLE[level].steps;
  return (
    <g>
      <rect x="0" y="0" width="14" height="14" rx="3.5" fill="var(--surface)" opacity="0.9" />
      {[0, 1, 2].map((i) => (
        <rect
          key={i}
          x={2 + i * 3.6}
          y={11 - (4 + i * 2.5)}
          width="2.6"
          height={4 + i * 2.5}
          rx="0.8"
          fill={i < steps ? 'currentColor' : 'none'}
          stroke="currentColor"
          strokeWidth="0.8"
          opacity={i < steps ? 1 : 0.45}
        />
      ))}
    </g>
  );
}
