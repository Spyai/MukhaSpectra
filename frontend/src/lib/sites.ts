import type { OralSite, SiteCatalogue, SiteCatalogueEntry } from '../api/types';

/**
 * Built-in copy of pipeline/mukha_pipeline/sites.py (catalogue + site-map geometry). The live
 * catalogue from GET /api/v1/sites replaces it when reachable; this copy keeps the site map
 * working offline and in mock mode. Drawn facing the patient: patient's RIGHT is on the
 * viewer's LEFT. viewBox 0 0 400 320.
 */
type Row = [OralSite, string, string, OralSite | null, 'contralateral' | 'analogous' | null, boolean, number, number, number, number];

const ROWS: Row[] = [
  ['upper_lip_mucosa', 'Upper lip mucosa', 'U lip', 'lower_lip_mucosa', 'analogous', false, 200, 28, 92, 16],
  ['upper_gingiva', 'Upper gingiva', 'U gum', 'lower_gingiva', 'analogous', true, 200, 60, 112, 12],
  ['hard_palate', 'Hard palate', 'H palate', null, null, true, 200, 98, 70, 22],
  ['soft_palate', 'Soft palate', 'S palate', null, null, false, 200, 136, 54, 13],
  ['retromolar_right', 'Right retromolar area', 'R retro', 'retromolar_left', 'contralateral', true, 112, 132, 17, 14],
  ['retromolar_left', 'Left retromolar area', 'L retro', 'retromolar_right', 'contralateral', true, 288, 132, 17, 14],
  ['buccal_mucosa_right', 'Right buccal mucosa', 'R cheek', 'buccal_mucosa_left', 'contralateral', false, 62, 170, 30, 72],
  ['buccal_mucosa_left', 'Left buccal mucosa', 'L cheek', 'buccal_mucosa_right', 'contralateral', false, 338, 170, 30, 72],
  ['tongue_dorsum', 'Tongue dorsum', 'Tongue top', null, null, true, 200, 190, 56, 38],
  ['lateral_tongue_right', 'Right lateral tongue', 'R tongue side', 'lateral_tongue_left', 'contralateral', false, 128, 194, 13, 36],
  ['lateral_tongue_left', 'Left lateral tongue', 'L tongue side', 'lateral_tongue_right', 'contralateral', false, 272, 194, 13, 36],
  ['ventral_tongue_floor_of_mouth', 'Ventral tongue / floor of mouth', 'Under tongue', null, null, false, 200, 246, 62, 13],
  ['lower_gingiva', 'Lower gingiva', 'L gum', 'upper_gingiva', 'analogous', true, 200, 274, 112, 12],
  ['lower_lip_mucosa', 'Lower lip mucosa', 'L lip', 'upper_lip_mucosa', 'analogous', false, 200, 300, 92, 16],
];

export const BUILTIN_SITE_CATALOGUE: SiteCatalogue = {
  viewBox: [0, 0, 400, 320],
  sites: ROWS.map(
    ([site, label, short_label, paired_site, pairing, keratinized, cx, cy, rx, ry]): SiteCatalogueEntry => ({
      site,
      label,
      short_label,
      paired_site,
      pairing,
      keratinized,
      shape: { cx, cy, rx, ry },
    }),
  ),
};

export const ALL_SITES: OralSite[] = BUILTIN_SITE_CATALOGUE.sites.map((s) => s.site);

export function findSite(catalogue: SiteCatalogue, site: OralSite | null | undefined) {
  return site ? catalogue.sites.find((s) => s.site === site) : undefined;
}

export function isOralSite(value: unknown): value is OralSite {
  return typeof value === 'string' && (ALL_SITES as string[]).includes(value);
}
