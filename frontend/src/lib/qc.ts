import type { TFunction } from 'i18next';
import type { OralSite, QCIssue, ScreeningResult, SiteResult } from '../api/types';

/** Sites that could not be analysed (quality check failed), in capture order. */
export function failedSites(result: ScreeningResult): OralSite[] {
  const listed = new Set(result.qc_summary.sites_failed ?? []);
  return result.sites
    .filter((s) => listed.has(s.site) || s.analysis_status !== 'analysed')
    .map((s) => s.site);
}

/** Plain retake prompts (errors first, then warnings) for one site, without duplicates. */
export function retakeIssues(site: SiteResult): QCIssue[] {
  const all = [
    ...(site.captures ?? []).flatMap((c) => c.qc.issues ?? []),
    ...(site.qc?.issues ?? []),
  ].filter((i) => i.severity !== 'info');
  const seen = new Set<string>();
  const unique = all.filter((i) => {
    if (seen.has(i.code)) return false;
    seen.add(i.code);
    return true;
  });
  return unique.sort((a, b) => (a.severity === b.severity ? 0 : a.severity === 'error' ? -1 : 1));
}

/** Translated retake prompt (i18n key qc.<code>), falling back to the pipeline's English. */
export function issueText(issue: QCIssue, t: TFunction): string {
  const key = issue.message_key?.startsWith('qc.') ? issue.message_key : `qc.${issue.code}`;
  return t(key as 'qc.blur', { defaultValue: issue.message });
}
