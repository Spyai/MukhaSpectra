import type {
  AlcoholUse,
  ArecaNutUse,
  DataProvenance,
  Diascopy,
  OralSite,
  RiskFactors,
  Sex,
  SmokelessTobaccoUse,
  SmokingUse,
  UploadManifest,
  UploadRequest,
  VisualAppearance,
} from '../../api/types';
import { isImageFile, isSpectrumFile } from './fileTypes';

export type FileSlot = 'wl' | 'af' | 'dark' | 'refWl' | 'refAf' | 'spectrum';

/** "unknown" = the health worker did not ask. It is never sent as "no". */
export type HabitStatus = 'unknown' | 'current' | 'former' | 'never';
export type AlcoholAnswer = 'unknown' | NonNullable<AlcoholUse['frequency']>;
export type SmokelessProduct = NonNullable<SmokelessTobaccoUse['types']>[number];
export type SmokingProduct = NonNullable<SmokingUse['types']>[number];

export const HABIT_STATUSES: HabitStatus[] = ['unknown', 'current', 'former', 'never'];
export const ALCOHOL_ANSWERS: AlcoholAnswer[] = [
  'unknown',
  'never',
  'occasional',
  'weekly',
  'daily',
];
export const SMOKELESS_PRODUCTS: SmokelessProduct[] = [
  'gutka',
  'khaini',
  'zarda',
  'mawa',
  'mishri',
  'paan_with_tobacco',
  'snuff',
  'other',
];
export const SMOKING_PRODUCTS: SmokingProduct[] = [
  'bidi',
  'cigarette',
  'chutta_reverse',
  'hookah',
  'other',
];

export interface HabitAnswer<P extends string = never> {
  status: HabitStatus;
  products: P[];
  /** Free numeric text from the form; empty = not asked. */
  years: string;
  perDay: string;
}

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
  smokeless: HabitAnswer<SmokelessProduct>;
  smoking: HabitAnswer<SmokingProduct>;
  areca: HabitAnswer;
  alcohol: AlcoholAnswer;
  ulcer: boolean;
  whitePatch: boolean;
  redPatch: boolean;
  lump: boolean;
  pain: boolean;
  bleeding: boolean;
  burning: boolean;
  trismus: boolean;
  neckLump: boolean;
  priorLesion: boolean;
  priorCancer: boolean;
  sharpTooth: boolean;
  denture: boolean;
  consent: boolean;
  research: boolean;
  provenance: DataProvenance;
  provenanceTouched: boolean;
  campName: string;
  workerId: string;
}

const emptyHabit = { status: 'unknown' as const, products: [], years: '', perDay: '' };

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
  smokeless: emptyHabit,
  smoking: emptyHabit,
  areca: emptyHabit,
  alcohol: 'unknown',
  ulcer: false,
  whitePatch: false,
  redPatch: false,
  lump: false,
  pain: false,
  bleeding: false,
  burning: false,
  trismus: false,
  neckLump: false,
  priorLesion: false,
  priorCancer: false,
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

/** Number typed by the health worker, or null when left empty (= not asked). */
function optionalNumber(text: string): number | null {
  const s = text.trim();
  if (s === '') return null;
  const n = Number(s);
  return Number.isFinite(n) && n >= 0 ? n : null;
}

function habit(a: HabitAnswer<string>) {
  const used = a.status === 'current' || a.status === 'former';
  return {
    uses: a.status === 'current',
    former: a.status === 'former',
    years: used ? optionalNumber(a.years) : null,
    per_day: used ? optionalNumber(a.perDay) : null,
  };
}

/**
 * Risk factors exactly as answered. A habit or alcohol question left on "Not asked" is OMITTED
 * from the manifest instead of being recorded as "no"; years / per day left empty are sent as
 * null. Symptom and history chips are shown to every person, so an unticked chip means "no".
 */
export function buildRiskFactors(form: UploadFormState): RiskFactors {
  const rf: RiskFactors = {};
  if (form.smokeless.status !== 'unknown') {
    const v: SmokelessTobaccoUse = {
      ...habit(form.smokeless),
      types: form.smokeless.status === 'never' ? [] : form.smokeless.products,
    };
    rf.smokeless_tobacco = v;
  }
  if (form.smoking.status !== 'unknown') {
    const v: SmokingUse = {
      ...habit(form.smoking),
      types: form.smoking.status === 'never' ? [] : form.smoking.products,
    };
    rf.smoking = v;
  }
  if (form.areca.status !== 'unknown') {
    const v: ArecaNutUse = habit(form.areca);
    rf.areca_nut = v;
  }
  if (form.alcohol !== 'unknown') {
    rf.alcohol = { uses: form.alcohol !== 'never', frequency: form.alcohol };
  }
  rf.prior_oral_lesion = form.priorLesion;
  rf.prior_oral_cancer = form.priorCancer;
  rf.symptoms = {
    non_healing_ulcer_over_2_weeks: form.ulcer,
    white_patch: form.whitePatch,
    red_patch: form.redPatch,
    lump: form.lump,
    pain: form.pain,
    difficulty_opening_mouth: form.trismus,
    bleeding: form.bleeding,
    burning_sensation: form.burning,
    neck_lump: form.neckLump,
  };
  rf.irritants = { sharp_tooth: form.sharpTooth, ill_fitting_denture: form.denture };
  return rf;
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
