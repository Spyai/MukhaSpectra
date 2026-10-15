import { describe, expect, it } from 'vitest';
import {
  buildUploadRequest,
  filesLookSynthetic,
  initialUploadForm,
  validateUpload,
  type UploadFormState,
} from './uploadForm';

const img = (name: string) => new File(['x'], name, { type: 'image/jpeg' });

function valid(): UploadFormState {
  return {
    ...initialUploadForm,
    site: 'lateral_tongue_right',
    referenceSite: 'lateral_tongue_left',
    files: { wl: img('wl.jpg'), af: img('af.jpg') },
    consent: true,
  };
}

describe('upload form validation', () => {
  it('an empty form needs a site, both photos and consent', () => {
    const e = validateUpload(initialUploadForm);
    expect(Object.keys(e).sort()).toEqual(['af', 'consent', 'site', 'wl']);
    expect(e.consent).toBe('upload.validation.consent');
  });

  it('consent is required even when everything else is present', () => {
    const e = validateUpload({ ...valid(), consent: false });
    expect(e).toEqual({ consent: 'upload.validation.consent' });
  });

  it('WL and AF are both required', () => {
    expect(validateUpload({ ...valid(), files: { af: img('af.jpg') } })).toEqual({
      wl: 'upload.validation.wl',
    });
    expect(validateUpload({ ...valid(), files: { wl: img('wl.jpg') } })).toEqual({
      af: 'upload.validation.af',
    });
  });

  it('a reference pair must be complete', () => {
    const e = validateUpload({ ...valid(), files: { ...valid().files, refWl: img('r.jpg') } });
    expect(e).toEqual({ refPair: 'upload.validation.refPair' });
  });

  it('rejects non-image photos, bad spectra and impossible ages', () => {
    const e = validateUpload({
      ...valid(),
      age: '150',
      files: {
        wl: new File(['x'], 'notes.txt', { type: 'text/plain' }),
        af: img('af.jpg'),
        spectrum: img('s.jpg'),
      },
    });
    expect(e).toEqual({
      wl: 'upload.validation.imageType',
      spectrum: 'upload.validation.spectrumType',
      age: 'upload.validation.age',
    });
  });

  it('a complete form is valid', () => {
    expect(validateUpload(valid())).toEqual({});
  });
});

describe('manifest building', () => {
  it('names every file field in the manifest and records consent', () => {
    const form: UploadFormState = {
      ...valid(),
      age: '52',
      sex: 'male',
      lesionSeen: true,
      appearance: ['white_patch'],
      note: '  rough patch  ',
      smokeless: { status: 'current', products: ['gutka'], years: '20', perDay: '6' },
      smoking: { status: 'former', products: ['bidi'], years: '12', perDay: '' },
      alcohol: 'weekly',
      ulcer: true,
      burning: true,
      research: true,
      files: {
        wl: img('wl.jpg'),
        af: img('af.jpg'),
        dark: img('dark.jpg'),
        refWl: img('rwl.jpg'),
        refAf: img('raf.jpg'),
        spectrum: new File(['x'], 'spectrum.csv', { type: 'text/csv' }),
      },
    };
    const { manifest, files } = buildUploadRequest(form);
    expect(manifest.sites).toEqual([
      expect.objectContaining({
        site: 'lateral_tongue_right',
        role: 'examined',
        wl: 'wl',
        af: 'af',
        dark: 'dark',
        visual_flag: { lesion_seen: true, appearance: ['white_patch'], description: 'rough patch' },
        diascopy: 'not_done',
      }),
      { site: 'lateral_tongue_left', role: 'reference_only', wl: 'ref_wl', af: 'ref_af' },
    ]);
    expect(manifest.spectral_file).toBe('spectrum');
    expect(Object.keys(files).sort()).toEqual(['af', 'dark', 'ref_af', 'ref_wl', 'spectrum', 'wl']);
    expect(manifest.consent).toEqual({
      consent_recorded: true,
      method: 'checkbox',
      language: 'en',
      research_use_opt_in: true,
    });
    expect(manifest.patient).toEqual({ age_years: 52, sex: 'male' });
    expect(manifest.risk_factors.smokeless_tobacco).toEqual({
      uses: true,
      former: false,
      types: ['gutka'],
      years: 20,
      per_day: 6,
    });
    // A former bidi smoker is recorded as former; an unanswered "per day" is null, not 0.
    expect(manifest.risk_factors.smoking).toEqual({
      uses: false,
      former: true,
      types: ['bidi'],
      years: 12,
      per_day: null,
    });
    // A weekly drinker is recorded as weekly (never collapsed to "never").
    expect(manifest.risk_factors.alcohol).toEqual({ uses: true, frequency: 'weekly' });
    expect(manifest.risk_factors.symptoms?.non_healing_ulcer_over_2_weeks).toBe(true);
    expect(manifest.risk_factors.symptoms?.burning_sensation).toBe(true);
    // No name or contact details are ever sent from the console.
    expect(JSON.stringify(manifest)).not.toMatch(/"name"/);
  });

  it('never records "no" for a habit that was not asked', () => {
    const rf = buildUploadRequest(valid()).manifest.risk_factors;
    expect(rf.smokeless_tobacco).toBeUndefined();
    expect(rf.smoking).toBeUndefined();
    expect(rf.areca_nut).toBeUndefined();
    expect(rf.alcohol).toBeUndefined();
    const never = buildUploadRequest({
      ...valid(),
      areca: { status: 'never', products: [], years: '9', perDay: '9' },
      alcohol: 'never',
    }).manifest.risk_factors;
    expect(never.areca_nut).toEqual({ uses: false, former: false, years: null, per_day: null });
    expect(never.alcohol).toEqual({ uses: false, frequency: 'never' });
  });

  it('omits the reference site when the reference toggle is off', () => {
    const form = {
      ...valid(),
      useReference: false,
      files: { ...valid().files, refWl: img('a.jpg'), refAf: img('b.jpg') },
    };
    expect(buildUploadRequest(form).manifest.sites).toHaveLength(1);
  });

  it('detects SYNTHETIC sample files by name', () => {
    expect(filesLookSynthetic({ wl: img('SYNTHETIC_lateral_tongue_right_wl.jpg') })).toBe(true);
    expect(filesLookSynthetic({ wl: img('IMG_0001.jpg') })).toBe(false);
  });
});
