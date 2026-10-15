import type { ApiClient, OralSite, ScreeningResult } from '../api/types';

/** Top site's white-light artifact: the real photo, never a stand-in. */
export function wlThumbnail(
  api: ApiClient,
  result: ScreeningResult | null | undefined,
): { url: string | null; site: OralSite | null } {
  if (!result) return { url: null, site: null };
  const topLesion = result.lesions?.find(
    (l) => l.lesion_id === result.triage.highest_risk_lesion_id,
  );
  const site =
    result.sites.find((s) => s.site === topLesion?.site && s.artifacts?.wl_image) ??
    result.sites.find((s) => s.role !== 'reference_only' && s.artifacts?.wl_image) ??
    result.sites.find((s) => s.artifacts?.wl_image);
  const key = site?.artifacts?.wl_image;
  return key
    ? { url: api.artifactUrl(result.screening_id, key, result.result_version), site: site.site }
    : { url: null, site: null };
}
