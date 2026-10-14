import type {
  DataProvenance,
  Diascopy,
  OralSite,
  RiskFactors,
  Sex,
  UploadManifest,
  UploadRequest,
  VisualAppearance,
} from '../../api/types';
import { isImageFile, isSpectrumFile } from './fileTypes';

export type FileSlot = 'wl' | 'af' | 'dark' | 'refWl' | 'refAf' | 'spectrum';

export interface UploadFormState {
  site: OralSite | null;
  referenceSite: OralSite | null;
  useReference: boolean;
  files: Partial<Record<FileSlot, File>>;
  lesionSeen: boolean;
  appearance: VisualAppearance[];
  note: string;
  diascopy: Diascopy;
  age: string;
  sex: Sex;
  smokeless: boolean;
  smoking: boolean;
  areca: boolean;
  alcoholDaily: boolean;
  ulcer: boolean;
  whitePatch: boolean;
  redPatch: boolean;
  lump: boolean;
  trismus: boolean;
  neckLump: boolean;
  sharpTooth: boolean;
  denture: boolean;
  consent: boolean;
  research: boolean;
  provenance: DataProvenance;
  provenanceTouched: boolean;
  campName: string;
  workerId: string;
}

export const initialUploadForm: UploadFormState = {
  site: null,
  referenceSite: null,
  useReference: true,
  files: {},
  lesionSeen: false,
  appearance: [],
  note: '',
  diascopy: 'not_done',
  age: '',
  sex: 'undisclosed',
  smokeless: false,
  smoking: false,
  areca: false,
  alcoholDaily: false,
  ulcer: false,
  whitePatch: false,
  redPatch: false,
  lump: false,
  trismus: false,
  neckLump: false,
  sharpTooth: false,
  denture: false,
  consent: false,
  research: false,
  provenance: 'real',
  provenanceTouched: false,
  campName: '',
  workerId: '',
};

/** Field → i18n key of a plain-language error. Order = order shown in the summary. */
export type UploadErrors = Partial<
  Record<'site' | 'wl' | 'af' | 'refPair' | 'spectrum' | 'age' | 'consent' | 'dark', string>
>;

export function validateUpload(form: UploadFormState): UploadErrors {
  const e: UploadErrors = {};
  if (!form.site) e.site = 'upload.validation.site';
  if (!form.files.wl) e.wl = 'upload.validation.wl';
  else if (!isImageFile(form.files.wl)) e.wl = 'upload.validation.imageType';
  if (!form.files.af) e.af = 'upload.validation.af';
  else if (!isImageFile(form.files.af)) e.af = 'upload.validation.imageType';
  if (form.files.dark && !isImageFile(form.files.dark)) e.dark = 'upload.validation.imageType';
  if (form.useReference && form.referenceSite) {
    const hasWl = Boolean(form.files.refWl);
    const hasAf = Boolean(form.files.refAf);
    if (hasWl !== hasAf) e.refPair = 'upload.validation.refPair';
    else if (
      (form.files.refWl && !isImageFile(form.files.refWl)) ||
      (form.files.refAf && !isImageFile(form.files.refAf))
    )
      e.refPair = 'upload.validation.imageType';
  }
  if (form.files.spectrum && !isSpectrumFile(form.files.spectrum))
    e.spectrum = 'upload.validation.spectrumType';
  if (form.age.trim() !== '') {
    const n = Number(form.age);
    if (!Number.isInteger(n) || n < 0 || n > 120) e.age = 'upload.validation.age';
  }
  if (!form.consent) e.consent = 'upload.validation.consent';
  return e;
}

export function filesLookSynthetic(files: Partial<Record<FileSlot, File>>): boolean {
  return Object.values(files).some((f) => f && /synthetic/i.test(f.name));
}

export function buildRiskFactors(form: UploadFormState): RiskFactors {
  return {
    smokeless_tobacco: {
      uses: form.smokeless,
      former: false,
      types: [],
      years: null,
      per_day: null,
    },
    smoking: { uses: form.smoking, former: false, types: [], years: null, per_day: null },
    areca_nut: { uses: form.areca, former: false, years: null, per_day: null },
    alcohol: { uses: form.alcoholDaily, frequency: form.alcoholDaily ? 'daily' : 'never' },
    prior_oral_lesion: false,
    prior_oral_cancer: false,
    symptoms: {
      non_healing_ulcer_over_2_weeks: form.ulcer,
      white_patch: form.whitePatch,
      red_patch: form.redPatch,
      lump: form.lump,
      pain: false,
      difficulty_opening_mouth: form.trismus,
      bleeding: false,
      burning_sensation: false,
      neck_lump: form.neckLump,
    },
    irritants: { sharp_tooth: form.sharpTooth, ill_fitting_denture: form.denture },
  };
}

/** Manifest + multipart files exactly as POST /api/v1/screenings expects (architecture §6). */
export function buildUploadRequest(
  form: UploadFormState,
  language: 'en' | 'hi' | 'kn' = 'en',
): UploadRequest {
  if (!form.site || !form.files.wl || !form.files.af) throw new Error('form is not valid');
  const files: UploadRequest['files'] = { wl: form.files.wl, af: form.files.af };
  const examined: UploadManifest['sites'][number] = {
    site: form.site,
    role: 'examined',
    wl: 'wl',
    af: 'af',
    visual_flag: {
      lesion_seen: form.lesionSeen,
      appearance: form.lesionSeen ? form.appearance : [],
      ...(form.lesionSeen && form.note.trim()
        ? { description: form.note.trim().slice(0, 1000) }
        : {}),
    },
    diascopy: form.diascopy,
    px_per_mm: null,
    device_id: null,
  };
  if (form.files.dark) {
    examined.dark = 'dark';
    files.dark = form.files.dark;
  }
  const sites: UploadManifest['sites'] = [examined];
  if (form.useReference && form.referenceSite && form.files.refWl && form.files.refAf) {
    sites.push({ site: form.referenceSite, role: 'reference_only', wl: 'ref_wl', af: 'ref_af' });
    files.ref_wl = form.files.refWl;
    files.ref_af = form.files.refAf;
  }
  const manifest: UploadManifest = {
    patient: { age_years: form.age.trim() === '' ? null : Number(form.age), sex: form.sex },
    consent: {
      consent_recorded: form.consent,
      method: 'checkbox',
      language,
      research_use_opt_in: form.research,
    },
    risk_factors: buildRiskFactors(form),
    context: {
      ...(form.campName.trim() ? { screening_site_name: form.campName.trim() } : {}),
      ...(form.workerId.trim() ? { health_worker_id: form.workerId.trim() } : {}),
    },
    data_provenance: form.provenance,
    sites,
  };
  if (form.files.spectrum) {
    manifest.spectral_file = 'spectrum';
    files.spectrum = form.files.spectrum;
  }
  return { manifest, files };
}
