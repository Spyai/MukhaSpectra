import { useState } from 'react';
import { useTranslation } from 'react-i18next';
import type { ConfigRef, ScreeningResult } from '../../api/types';
import { shortSha } from '../../lib/format';
import { Icon } from '../ui/Icon';

function Row({ label, value, mono }: { label: string; value: string; mono?: string }) {
  return (
    <div className="flex flex-col gap-0.5 rounded-xl bg-surface-2 px-3 py-2.5">
      <dt className="text-[11px] font-semibold tracking-wide text-fg-subtle uppercase">{label}</dt>
      <dd className="text-sm font-semibold tabular">
        {value}
        {mono ? (
          <span className="ml-1.5 font-mono text-xs font-normal text-fg-subtle">{mono}</span>
        ) : null}
      </dd>
    </div>
  );
}

const cfg = (c: ConfigRef) => `${c.version}`;

export function VersionsFooter({ result }: { result: ScreeningResult }) {
  const { t } = useTranslation();
  const v = result.versions;
  const [copied, setCopied] = useState(false);
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(`${result.screening_id} / ${result.result_id}`);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      /* clipboard unavailable */
    }
  };
  return (
    <div data-testid="versions-footer">
      <dl className="grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-6">
        <Row label={t('results.versions.software')} value={v.software} />
        <Row label={t('results.versions.pipeline')} value={v.pipeline} />
        <Row
          label={t('results.versions.schema')}
          value={v.schema_version ?? result.schema_version ?? '1.0.0'}
        />
        <Row
          label={t('results.versions.rules')}
          value={cfg(v.rule_config)}
          mono={shortSha(v.rule_config.sha256)}
        />
        <Row
          label={t('results.versions.qcConfig')}
          value={cfg(v.qc_config)}
          mono={shortSha(v.qc_config.sha256)}
        />
        <Row
          label={t('results.versions.analysisConfig')}
          value={cfg(v.analysis_config)}
          mono={shortSha(v.analysis_config.sha256)}
        />
      </dl>
      <div className="mt-3 flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-fg-subtle">
        <span>
          {t('results.versions.screeningId')}:{' '}
          <span className="font-mono">{result.screening_id}</span>
        </span>
        <span>
          {t('results.versions.resultId')}: <span className="font-mono">{result.result_id}</span>
        </span>
        <button
          type="button"
          className="btn btn-ghost min-h-8 px-2 py-1 text-xs"
          onClick={() => void copy()}
        >
          <Icon name={copied ? 'check' : 'copy'} size={14} />
          {copied ? t('results.versions.copied') : t('results.versions.copy')}
        </button>
      </div>
    </div>
  );
}
