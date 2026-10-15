import { m } from 'motion/react';
import { useTranslation } from 'react-i18next';
import { Link } from 'react-router';
import type { ScreeningResult } from '../../api/types';
import { failedSites, issueText, retakeIssues } from '../../lib/qc';
import { Icon } from '../ui/Icon';

/**
 * Camp workflow: when a site failed quality checks, say so right under the risk header with the
 * plain retake prompts and a one-tap way back to the upload form for that site. For an
 * INCONCLUSIVE result this is THE next step (re-analysing the same photos cannot help).
 */
export function RetakeCallout({ result }: { result: ScreeningResult }) {
  const { t } = useTranslation();
  const failed = failedSites(result);
  if (failed.length === 0) return null;
  const primary = result.triage.overall_risk_level === 'INCONCLUSIVE';

  return (
    <m.section
      aria-labelledby="retake-title"
      className={`rounded-[1.25rem] border-2 p-4 sm:p-6 ${
        primary
          ? 'border-risk-inconclusive bg-risk-inconclusive-bg'
          : 'border-warn/60 bg-risk-moderate-bg'
      }`}
      data-testid="retake-callout"
      data-primary={primary}
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ type: 'spring', stiffness: 240, damping: 28, delay: 0.1 }}
    >
      <div className="flex items-start gap-3">
        <span
          className={`flex h-10 w-10 shrink-0 items-center justify-center rounded-xl ${
            primary ? 'bg-surface text-fg' : 'bg-surface text-warn'
          }`}
        >
          <Icon name="camera" size={20} />
        </span>
        <div className="min-w-0 flex-1">
          <p className="eyebrow">{t('results.retake.label')}</p>
          <h2 id="retake-title" className="mt-0.5 text-lg font-bold tracking-tight">
            {t('results.retake.title', { count: failed.length })}
          </h2>
          <p className="mt-1 text-sm text-fg-muted">
            {primary ? t('results.retake.bodyInconclusive') : t('results.retake.body')}
          </p>
        </div>
      </div>
      <ul className="mt-4 flex flex-col gap-3">
        {failed.map((siteId) => {
          const site = result.sites.find((s) => s.site === siteId);
          const issues = site ? retakeIssues(site) : [];
          const name = t(`sites.${siteId}`);
          return (
            <li
              key={siteId}
              className="flex flex-col gap-3 rounded-2xl border border-line bg-surface p-3.5 sm:flex-row sm:items-start sm:justify-between"
            >
              <div className="min-w-0">
                <p className="font-semibold">{name}</p>
                {issues.length > 0 ? (
                  <ul className="mt-1.5 space-y-1">
                    {issues.map((i) => (
                      <li
                        key={i.code}
                        className={`flex items-start gap-2 text-sm ${i.severity === 'error' ? 'text-fg' : 'text-fg-muted'}`}
                        data-testid="retake-callout-prompt"
                      >
                        <Icon
                          name={i.severity === 'error' ? 'camera' : 'info'}
                          size={15}
                          className={`mt-0.5 shrink-0 ${i.severity === 'error' ? 'text-danger' : 'text-warn'}`}
                        />
                        {issueText(i, t)}
                      </li>
                    ))}
                  </ul>
                ) : null}
              </div>
              <Link
                to={`/upload?site=${encodeURIComponent(siteId)}`}
                className={`btn shrink-0 ${primary ? 'btn-primary' : 'btn-secondary'}`}
                data-testid={`retake-link-${siteId}`}
              >
                <Icon name="camera" size={18} />
                {t('results.retake.button', { site: name })}
              </Link>
            </li>
          );
        })}
      </ul>
    </m.section>
  );
}
