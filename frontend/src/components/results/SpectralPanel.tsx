import { m } from 'motion/react';
import { useId, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';
import type { ScreeningResult, SpectralFeatures, SpectralReadingResult } from '../../api/types';
import { monotonePath, SPECTRAL_CHANNELS } from '../../lib/chart';
import { formatNumber, formatSigned } from '../../lib/format';
import { Icon } from '../ui/Icon';

const FEATURES: Array<keyof SpectralFeatures> = [
  'hb_absorption_proxy',
  'ratio_555_630',
  'ratio_590_680',
  'ratio_515_630',
  'ratio_415_555',
  'slope_480_555',
  'slope_590_680',
];

const W = 560;
const H = 300;
const PAD = { l: 48, r: 16, t: 18, b: 40 };
const X0 = 405;
const X1 = 690;

function SpectrumChart({
  lesion,
  reference,
}: {
  lesion: SpectralReadingResult;
  reference?: SpectralReadingResult;
}) {
  const { t } = useTranslation();
  const titleId = useId();
  const series = [lesion, reference].filter(Boolean) as SpectralReadingResult[];
  const maxY =
    Math.max(
      ...series.flatMap((s) =>
        SPECTRAL_CHANNELS.map(([k]) => (s.values[k] ?? 0) + (s.values_std[k] ?? 0)),
      ),
      0.1,
    ) * 1.12;
  const x = (nm: number) => PAD.l + ((nm - X0) / (X1 - X0)) * (W - PAD.l - PAD.r);
  const y = (v: number) => PAD.t + (1 - v / maxY) * (H - PAD.t - PAD.b);
  const step = maxY > 0.6 ? 0.2 : maxY > 0.3 ? 0.1 : 0.05;
  const yTicks: number[] = [];
  for (let v = 0; v <= maxY + 1e-9; v += step) yTicks.push(Math.round(v * 100) / 100);

  const build = (s: SpectralReadingResult) => {
    const pts = SPECTRAL_CHANNELS.filter(([k]) => typeof s.values[k] === 'number').map(
      ([k, nm]) => ({
        nm,
        k,
        v: s.values[k]!,
        sd: s.values_std[k] ?? 0,
      }),
    );
    const line = monotonePath(pts.map((p) => ({ x: x(p.nm), y: y(p.v) })));
    const upper = pts.map((p) => ({ x: x(p.nm), y: y(p.v + p.sd) }));
    const lower = pts.map((p) => ({ x: x(p.nm), y: y(Math.max(0, p.v - p.sd)) })).reverse();
    const band = `${monotonePath(upper)} ${monotonePath(lower).replace(/^M/, 'L')} Z`;
    return { pts, line, band };
  };
  const L = build(lesion);
  const R = reference ? build(reference) : null;
  const yLabel = t(`results.spectral.${lesion.signal_type}`);

  return (
    <figure>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="h-auto w-full"
        role="img"
        aria-labelledby={titleId}
        data-testid="spectral-chart"
      >
        <title id={titleId}>{t('results.spectral.chartLabel')}</title>
        {/* Hb dip band */}
        <rect
          x={x(540)}
          y={PAD.t}
          width={x(580) - x(540)}
          height={H - PAD.t - PAD.b}
          fill="var(--risk-high)"
          opacity={0.08}
        />
        <text
          x={(x(540) + x(580)) / 2}
          y={PAD.t + 12}
          textAnchor="middle"
          className="fill-fg-subtle text-[10px] font-semibold"
        >
          {t('results.spectral.hbBand')}
        </text>
        {yTicks.map((v) => (
          <g key={v}>
            <line
              x1={PAD.l}
              x2={W - PAD.r}
              y1={y(v)}
              y2={y(v)}
              stroke="var(--line)"
              strokeWidth={1}
            />
            <text
              x={PAD.l - 8}
              y={y(v) + 4}
              textAnchor="end"
              className="fill-fg-subtle text-[10.5px] tabular"
            >
              {v.toFixed(step < 0.1 ? 2 : 1)}
            </text>
          </g>
        ))}
        {SPECTRAL_CHANNELS.map(([k, nm]) => (
          <text
            key={k}
            x={x(nm)}
            y={H - PAD.b + 18}
            textAnchor="middle"
            className="fill-fg-subtle text-[10.5px] tabular"
          >
            {nm}
          </text>
        ))}
        <text
          x={(PAD.l + W - PAD.r) / 2}
          y={H - 4}
          textAnchor="middle"
          className="fill-fg-muted text-[11px] font-semibold"
        >
          {t('results.spectral.wavelength')}
        </text>
        <text
          transform={`translate(12 ${(PAD.t + H - PAD.b) / 2}) rotate(-90)`}
          textAnchor="middle"
          className="fill-fg-muted text-[11px] font-semibold"
        >
          {yLabel}
        </text>

        {R ? (
          <>
            <m.path
              d={R.band}
              fill="var(--ref-blue)"
              initial={{ opacity: 0 }}
              animate={{ opacity: 0.14 }}
              transition={{ delay: 0.5 }}
            />
            <m.path
              d={R.line}
              fill="none"
              stroke="var(--ref-blue)"
              strokeWidth={2.4}
              strokeDasharray="7 5"
              initial={{ pathLength: 0 }}
              animate={{ pathLength: 1 }}
              transition={{ duration: 0.9, ease: [0.22, 1, 0.36, 1] }}
            />
            {R.pts.map((p) => (
              <circle
                key={p.k}
                cx={x(p.nm)}
                cy={y(p.v)}
                r={3.2}
                fill="var(--surface)"
                stroke="var(--ref-blue)"
                strokeWidth={2}
              >
                <title>{`${t('results.spectral.referenceReading')} ${p.nm} nm: ${p.v.toFixed(3)} ± ${p.sd.toFixed(3)}`}</title>
              </circle>
            ))}
          </>
        ) : null}
        <m.path
          d={L.band}
          fill="var(--violet)"
          initial={{ opacity: 0 }}
          animate={{ opacity: 0.18 }}
          transition={{ delay: 0.6 }}
        />
        <m.path
          d={L.line}
          fill="none"
          stroke="var(--violet)"
          strokeWidth={2.8}
          initial={{ pathLength: 0 }}
          animate={{ pathLength: 1 }}
          transition={{ duration: 1.1, ease: [0.22, 1, 0.36, 1], delay: 0.15 }}
        />
        {L.pts.map((p) => (
          <circle
            key={p.k}
            cx={x(p.nm)}
            cy={y(p.v)}
            r={3.6}
            fill="var(--violet)"
            stroke="var(--surface)"
            strokeWidth={1.5}
          >
            <title>{`${t('results.spectral.lesion')} ${p.nm} nm: ${p.v.toFixed(3)} ± ${p.sd.toFixed(3)}`}</title>
          </circle>
        ))}
      </svg>
      <figcaption className="mt-2 flex flex-wrap gap-x-5 gap-y-1.5 text-xs text-fg-muted">
        <span className="flex items-center gap-2">
          <svg width="26" height="8" aria-hidden="true">
            <line x1="0" x2="26" y1="4" y2="4" stroke="var(--violet)" strokeWidth="3" />
          </svg>
          {t('results.spectral.lesion')} · {t(`sites.${lesion.site}`)} ·{' '}
          {t('results.spectral.readings', { count: lesion.n_readings })}
        </span>
        {reference ? (
          <span className="flex items-center gap-2">
            <svg width="26" height="8" aria-hidden="true">
              <line
                x1="0"
                x2="26"
                y1="4"
                y2="4"
                stroke="var(--ref-blue)"
                strokeWidth="3"
                strokeDasharray="6 4"
              />
            </svg>
            {t('results.spectral.referenceReading')} · {t(`sites.${reference.site}`)} ·{' '}
            {t('results.spectral.readings', { count: reference.n_readings })}
          </span>
        ) : null}
        <span className="flex items-center gap-2">
          <span className="h-3 w-6 rounded-sm bg-violet/20" aria-hidden="true" />
          {t('results.spectral.band')}
        </span>
      </figcaption>
      <div className="sr-only">
        <table>
          <caption>{t('results.spectral.chartLabel')}</caption>
          <thead>
            <tr>
              <th scope="col">{t('results.spectral.wavelength')}</th>
              <th scope="col">{t('results.spectral.lesion')}</th>
              {reference ? <th scope="col">{t('results.spectral.referenceReading')}</th> : null}
            </tr>
          </thead>
          <tbody>
            {SPECTRAL_CHANNELS.map(([k, nm]) => (
              <tr key={k}>
                <th scope="row">{nm}</th>
                <td>{formatNumber(lesion.values[k], 3)}</td>
                {reference ? <td>{formatNumber(reference.values[k], 3)}</td> : null}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </figure>
  );
}

export function SpectralPanel({ result }: { result: ScreeningResult }) {
  const { t } = useTranslation();
  const readings = useMemo(() => result.spectral_readings ?? [], [result.spectral_readings]);
  const lesions = readings.filter((r) => r.role === 'lesion');
  const [activeId, setActiveId] = useState<string | null>(lesions[0]?.reading_id ?? null);
  const lesion = lesions.find((r) => r.reading_id === activeId) ?? lesions[0];

  if (!lesion) {
    const only = readings[0];
    return only ? (
      <SpectrumChart lesion={only} />
    ) : (
      <p className="rounded-xl border border-dashed border-line p-4 text-sm text-fg-muted">
        {t('results.spectral.none')}
      </p>
    );
  }
  const reference = readings.find((r) => r.reading_id === lesion.reference_reading_id);
  const cls = lesion.classifier;

  return (
    <div data-testid="spectral-panel">
      <p className="text-sm text-fg-muted">{t('results.spectral.intro')}</p>
      {lesions.length > 1 ? (
        <div className="mt-3 flex flex-wrap gap-2">
          {lesions.map((l) => (
            <button
              key={l.reading_id}
              type="button"
              aria-pressed={l.reading_id === lesion.reading_id}
              onClick={() => setActiveId(l.reading_id)}
              className={`chip min-h-9 ${l.reading_id === lesion.reading_id ? 'border-violet bg-violet-soft text-fg' : ''}`}
            >
              {t(`sites.${l.site}`)} · {l.reading_id}
            </button>
          ))}
        </div>
      ) : null}
      <div className="mt-4">
        <SpectrumChart lesion={lesion} reference={reference} />
      </div>

      {lesion.features ? (
        <div className="mt-5 overflow-x-auto">
          <table className="w-full min-w-[420px] text-sm tabular">
            <thead>
              <tr className="border-b border-line text-left text-xs text-fg-subtle">
                <th scope="col" className="py-2 pr-3 font-semibold">
                  {t('results.spectral.feature')}
                </th>
                <th scope="col" className="py-2 pr-3 text-right font-semibold">
                  {t('results.spectral.lesionValue')}
                </th>
                <th scope="col" className="py-2 pr-3 text-right font-semibold">
                  {t('results.spectral.refValue')}
                </th>
                <th scope="col" className="py-2 text-right font-semibold">
                  {t('results.spectral.delta')}
                </th>
              </tr>
            </thead>
            <tbody>
              {FEATURES.map((f) => (
                <tr key={f} className="border-b border-line/60 last:border-0">
                  <th scope="row" className="py-2 pr-3 text-left font-medium text-fg">
                    {t(`results.spectral.features.${f}`)}
                  </th>
                  <td className="py-2 pr-3 text-right">{formatNumber(lesion.features?.[f], 3)}</td>
                  <td className="py-2 pr-3 text-right text-fg-muted">
                    {formatNumber(reference?.features?.[f], 3)}
                  </td>
                  <td className="py-2 text-right font-semibold">
                    {formatSigned(lesion.delta_vs_reference?.[f], 3)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : null}

      {cls ? (
        <div
          className="mt-5 rounded-xl border border-dashed border-line-strong p-4"
          data-testid="spectral-classifier"
        >
          <div className="flex flex-wrap items-center gap-2">
            <Icon name="flask" size={18} className="text-fg-subtle" />
            <span className="font-semibold">{t('results.spectral.classifier')}</span>
            <span className="rounded-md border border-warn/60 px-1.5 py-0.5 text-[11px] font-bold tracking-wide text-warn uppercase">
              {t('results.spectral.placeholder')}
            </span>
            {cls.label ? (
              <span className="chip">{t(`results.spectral.classifierLabel.${cls.label}`)}</span>
            ) : null}
          </div>
          {cls.suspicion_score != null ? (
            <p className="mt-2 text-sm text-fg-muted">
              {t('results.spectral.heuristicScore', { value: cls.suspicion_score.toFixed(2) })}
            </p>
          ) : null}
          {(cls.reasons ?? []).length > 0 ? (
            <ul className="mt-2 list-disc space-y-1 pl-5 text-sm text-fg-muted">
              {(cls.reasons ?? []).map((r) => (
                <li key={r}>{r}</li>
              ))}
            </ul>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}
