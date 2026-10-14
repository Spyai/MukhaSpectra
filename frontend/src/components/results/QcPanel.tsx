import { useTranslation } from 'react-i18next';
import type { CaptureRecord, ModelInfo, ScreeningResult } from '../../api/types';
import { issueText } from '../../lib/qc';
import { Icon } from '../ui/Icon';

function CaptureChip({ c }: { c: CaptureRecord }) {
  const { t } = useTranslation();
  const passed = c.qc.passed;
  const warn = passed && (c.qc.issues ?? []).some((i) => i.severity === 'warning');
  return (
    <span
      className={`chip ${passed ? (warn ? 'border-warn/60' : 'border-ok/50') : 'border-danger/70 bg-risk-high-bg text-fg'}`}
      data-testid="capture-chip"
      data-passed={passed}
    >
      <Icon
        name={passed ? (warn ? 'alert' : 'check') : 'x'}
        size={14}
        strokeWidth={2.4}
        className={passed ? (warn ? 'text-warn' : 'text-ok') : 'text-danger'}
      />
      {t(`results.qc.kind.${c.metadata.kind}`)}
      <span className="font-semibold">
        ·{' '}
        {passed
          ? warn
            ? t('results.qc.warning')
            : t('results.qc.passed')
          : t('results.qc.failed')}
      </span>
    </span>
  );
}

function modelLine(m: ModelInfo, t: ReturnType<typeof useTranslation>['t']): string {
  return t(`results.models.${m.component}.${m.status}`);
}

export function QcPanel({ result }: { result: ScreeningResult }) {
  const { t } = useTranslation();
  const models = result.versions.models ?? [];
  const limitations = result.limitations ?? [];

  return (
    <div className="flex flex-col gap-6" data-testid="qc-panel">
      <div>
        <h3 className="text-sm font-bold">{t('results.qc.captures')}</h3>
        {result.qc_summary.passed_all ? (
          <p className="mt-1 flex items-center gap-1.5 text-sm text-fg-muted">
            <Icon name="check" size={16} className="text-ok" />
            {t('results.qc.allPassed')}
          </p>
        ) : null}
        <ul className="mt-3 flex flex-col gap-3">
          {result.sites.map((s) => {
            const errors = (s.captures ?? []).flatMap((c) =>
              (c.qc.issues ?? []).filter((i) => i.severity !== 'info'),
            );
            return (
              <li key={s.site} className="rounded-xl border border-line p-3">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <span className="text-sm font-semibold">{t(`sites.${s.site}`)}</span>
                  <span
                    className={`text-xs font-semibold ${s.analysis_status === 'analysed' ? 'text-fg-subtle' : 'text-danger'}`}
                  >
                    {t(`results.qc.analysisStatus.${s.analysis_status}`)}
                  </span>
                </div>
                <div className="mt-2 flex flex-wrap gap-1.5">
                  {(s.captures ?? []).map((c) => (
                    <CaptureChip key={c.metadata.capture_id} c={c} />
                  ))}
                </div>
                {errors.length > 0 ? (
                  <ul className="mt-2.5 space-y-1.5" aria-label={t('results.qc.retakes')}>
                    {errors.map((i, idx) => (
                      <li
                        key={`${i.code}-${idx}`}
                        className={`flex items-start gap-2 text-sm ${i.severity === 'error' ? 'font-semibold text-fg' : 'text-fg-muted'}`}
                        data-testid="retake-prompt"
                      >
                        <Icon
                          name={i.severity === 'error' ? 'camera' : 'info'}
                          size={16}
                          className={`mt-0.5 shrink-0 ${i.severity === 'error' ? 'text-danger' : 'text-warn'}`}
                        />
                        {issueText(i, t)}
                      </li>
                    ))}
                  </ul>
                ) : null}
              </li>
            );
          })}
        </ul>
      </div>

      <div className="grid gap-6 sm:grid-cols-2">
        <div>
          <h3 className="text-sm font-bold">{t('results.qc.calibration')}</h3>
          <p className="mt-1.5 flex items-center gap-2 text-sm">
            <Icon
              name={result.calibration_status === 'calibrated' ? 'check' : 'alert'}
              size={16}
              className={result.calibration_status === 'calibrated' ? 'text-ok' : 'text-warn'}
            />
            {t(`results.qc.calibrationStatus.${result.calibration_status}`)}
          </p>
        </div>
        <div>
          <h3 className="text-sm font-bold">{t('results.qc.models')}</h3>
          <ul className="mt-1.5 space-y-1.5 text-sm" data-testid="model-statuses">
            {models.map((m) => (
              <li key={`${m.component}-${m.name}`} className="flex items-start gap-2">
                <Icon
                  name={
                    m.status === 'model_unavailable'
                      ? 'x'
                      : m.status === 'placeholder'
                        ? 'flask'
                        : 'layers'
                  }
                  size={15}
                  className="mt-0.5 shrink-0 text-fg-subtle"
                />
                <span>
                  {modelLine(m, t)}
                  {!m.validated ? (
                    <span className="block text-xs text-fg-subtle">
                      {t('results.qc.unvalidated')}
                    </span>
                  ) : null}
                </span>
              </li>
            ))}
          </ul>
        </div>
      </div>

      <div>
        <h3 className="text-sm font-bold">{t('results.qc.limitations')}</h3>
        {limitations.length === 0 ? (
          <p className="mt-1.5 text-sm text-fg-muted">{t('results.qc.noLimitations')}</p>
        ) : (
          <ul
            className="mt-1.5 list-disc space-y-1 pl-5 text-sm text-fg-muted"
            data-testid="limitations"
          >
            {limitations.map((l) => (
              <li key={l.code}>{l.message}</li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}
