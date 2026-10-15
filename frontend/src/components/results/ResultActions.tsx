import { useTranslation } from 'react-i18next';
import type { ResultVersionSummary } from '../../api/types';
import { Icon } from '../ui/Icon';

export function VersionSwitcher({
  versions,
  current,
  latest,
  onChange,
}: {
  versions: ResultVersionSummary[];
  current: number;
  latest: number;
  onChange: (v: number) => void;
}) {
  const { t } = useTranslation();
  if (versions.length === 0) return null;
  return (
    <div
      role="group"
      aria-label={t('results.actions.versionSwitcher')}
      className="flex max-w-full overflow-x-auto rounded-xl border border-line bg-surface-2 p-1"
      data-testid="version-switcher"
    >
      {versions.map((v) => {
        const on = v.result_version === current;
        return (
          <button
            key={v.result_version}
            type="button"
            aria-pressed={on}
            aria-label={t('results.actions.versionOption', { version: v.result_version })}
            onClick={() => onChange(v.result_version)}
            className={`flex min-h-9 shrink-0 items-center gap-1 rounded-lg px-3 text-sm font-semibold tabular transition-colors ${
              on ? 'bg-surface text-fg shadow-card' : 'text-fg-muted hover:text-fg'
            }`}
          >
            {t('common.version', { version: v.result_version })}
            {v.result_version === latest ? (
              <span className="text-[10px] font-bold tracking-wide text-violet-text uppercase">
                {t('common.latest')}
              </span>
            ) : null}
          </button>
        );
      })}
    </div>
  );
}

export function ResultActions({
  reportUrl,
  onReanalyse,
  reanalysing,
  demoteReanalyse = false,
}: {
  reportUrl: string;
  onReanalyse: () => void;
  reanalysing: boolean;
  /** INCONCLUSIVE because of bad photos: retaking is the next step, not re-analysing. */
  demoteReanalyse?: boolean;
}) {
  const { t } = useTranslation();
  return (
    <div className="flex flex-wrap items-center gap-2">
      <a
        href={reportUrl}
        target="_blank"
        rel="noopener"
        className="btn btn-primary"
        data-testid="report-link"
        title={t('results.actions.pdfHint')}
      >
        <Icon name="file" size={18} />
        {t('results.actions.pdf')}
      </a>
      <button
        type="button"
        className={`btn ${demoteReanalyse ? 'btn-ghost' : 'btn-secondary'}`}
        onClick={onReanalyse}
        disabled={reanalysing}
        title={
          demoteReanalyse
            ? t('results.actions.reanalyseDemoted')
            : t('results.actions.reanalyseHint')
        }
        aria-describedby={demoteReanalyse ? 'reanalyse-demoted' : undefined}
        data-testid="reanalyse"
        data-demoted={demoteReanalyse}
      >
        <Icon name="refresh" size={18} className={reanalysing ? 'animate-spin' : ''} />
        {t('results.actions.reanalyse')}
      </button>
      {demoteReanalyse ? (
        <span id="reanalyse-demoted" className="basis-full text-xs text-fg-subtle lg:text-right">
          {t('results.actions.reanalyseDemoted')}
        </span>
      ) : null}
    </div>
  );
}
