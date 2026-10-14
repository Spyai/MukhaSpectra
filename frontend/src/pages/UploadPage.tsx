import { AnimatePresence, m } from 'motion/react';
import { useCallback, useEffect, useMemo, useRef, useState, type FormEvent } from 'react';
import { useTranslation } from 'react-i18next';
import { useNavigate } from 'react-router';
import { IS_MOCK } from '../api/client';
import { useCreateScreening, useSiteCatalogue } from '../api/queries';
import type { DataProvenance, Diascopy, OralSite, Sex, VisualAppearance } from '../api/types';
import { AnalysisOverlay } from '../components/analysis/AnalysisOverlay';
import { DropZone } from '../components/upload/DropZone';
import { PREVIEWABLE } from '../components/upload/fileTypes';
import {
  buildUploadRequest,
  filesLookSynthetic,
  initialUploadForm,
  validateUpload,
  type FileSlot,
  type UploadErrors,
  type UploadFormState,
} from '../components/upload/uploadForm';
import { Checkbox, FieldError, Section, Segmented, ToggleChip } from '../components/ui/controls';
import { Icon } from '../components/ui/Icon';
import { SiteMap } from '../components/viz/SiteMap';
import { errorMessage } from '../lib/errors';
import { findSite } from '../lib/sites';

const APPEARANCES: VisualAppearance[] = [
  'white_patch',
  'red_patch',
  'mixed_red_white',
  'ulcer',
  'growth',
  'other',
];
const SEXES: Sex[] = ['female', 'male', 'other', 'undisclosed'];
const DIASCOPY: Diascopy[] = ['not_done', 'fvl_persists', 'fvl_resolves'];
const PROVENANCE: DataProvenance[] = [
  'real',
  'synthetic',
  'phantom',
  'volunteer',
  'ex_vivo',
  'mixed',
];

const FIELD_ANCHOR: Record<keyof UploadErrors, string> = {
  site: 'site-select',
  wl: 'wl',
  af: 'af',
  dark: 'dark',
  refPair: 'refWl',
  spectrum: 'spectrum',
  age: 'age',
  consent: 'consent',
};

export default function UploadPage() {
  const { t, i18n } = useTranslation();
  const navigate = useNavigate();
  const catalogue = useSiteCatalogue();
  const create = useCreateScreening();
  const [form, setForm] = useState<UploadFormState>(initialUploadForm);
  const [submitted, setSubmitted] = useState(false);
  const [overlay, setOverlay] = useState(false);
  const [thumb, setThumb] = useState<string | null>(null);
  const summaryRef = useRef<HTMLDivElement>(null);

  const errors = useMemo(() => (submitted ? validateUpload(form) : {}), [form, submitted]);
  const errorEntries = Object.entries(errors) as Array<[keyof UploadErrors, string]>;
  const set = <K extends keyof UploadFormState>(key: K, value: UploadFormState[K]) =>
    setForm((f) => ({ ...f, [key]: value }));

  const siteInfo = findSite(catalogue, form.site);
  const refInfo = findSite(catalogue, form.referenceSite);

  const chooseSite = (site: OralSite) => {
    const info = findSite(catalogue, site);
    setForm((f) => ({
      ...f,
      site,
      referenceSite: info?.paired_site ?? null,
      files:
        info?.paired_site === f.referenceSite
          ? f.files
          : { ...f.files, refWl: undefined, refAf: undefined },
    }));
  };

  const setFile = (slot: FileSlot, file: File | undefined) =>
    setForm((f) => {
      const files = { ...f.files, [slot]: file };
      const synthetic = filesLookSynthetic(files);
      return {
        ...f,
        files,
        provenance: f.provenanceTouched ? f.provenance : synthetic ? 'synthetic' : 'real',
      };
    });

  useEffect(
    () => () => {
      if (thumb) URL.revokeObjectURL(thumb);
    },
    [thumb],
  );

  const onSubmit = (e: FormEvent) => {
    e.preventDefault();
    setSubmitted(true);
    const errs = validateUpload(form);
    if (Object.keys(errs).length > 0) {
      requestAnimationFrame(() => {
        summaryRef.current?.focus({ preventScroll: true });
        summaryRef.current?.scrollIntoView({ block: 'center' });
      });
      return;
    }
    const wl = form.files.wl!;
    setThumb(
      PREVIEWABLE.test(wl.name) && typeof URL.createObjectURL === 'function'
        ? URL.createObjectURL(wl)
        : null,
    );
    setOverlay(true);
    create.mutate(buildUploadRequest(form, (i18n.language as 'en') ?? 'en'));
  };

  const onContinue = useCallback(() => {
    if (create.data) navigate(`/screenings/${encodeURIComponent(create.data.screening_id)}`);
  }, [create.data, navigate]);
  const onClose = useCallback(() => {
    setOverlay(false);
    create.reset();
  }, [create]);

  const synthetic = filesLookSynthetic(form.files);

  return (
    <div className="mx-auto max-w-7xl px-4 pt-8 pb-16 sm:px-6">
      <header className="max-w-3xl">
        <p className="eyebrow">{t('app.consoleName')}</p>
        <h1 className="mt-2 text-3xl font-extrabold tracking-tight sm:text-4xl">
          {t('upload.title')}
        </h1>
        <p className="mt-2 text-fg-muted">{t('upload.intro')}</p>
        {IS_MOCK ? (
          <p className="mt-3 inline-flex items-center gap-2 rounded-xl bg-violet-soft px-3 py-2 text-sm text-violet-text">
            <Icon name="flask" size={16} />
            {t('app.mockModeHint')}
          </p>
        ) : null}
      </header>

      <form
        noValidate
        onSubmit={onSubmit}
        className="mt-8 grid gap-5 lg:grid-cols-[minmax(0,0.9fr)_minmax(0,1.1fr)] lg:items-start"
      >
        <div className="lg:sticky lg:top-28">
          <Section id="site" step={1} title={t('upload.site.title')}>
            <p className="text-sm text-fg-muted">{t('upload.site.hint')}</p>
            <SiteMap
              className="mx-auto mt-4 max-w-md"
              catalogue={catalogue}
              mode="select"
              selected={form.site}
              reference={form.useReference ? form.referenceSite : null}
              onSelect={chooseSite}
              label={t('upload.site.mapLabel')}
            />
            <div className="mt-4">
              <label htmlFor="site-select" className="field-label">
                {t('upload.site.listLabel')}
              </label>
              <select
                id="site-select"
                className="input"
                value={form.site ?? ''}
                onChange={(e) => e.target.value && chooseSite(e.target.value as OralSite)}
                aria-invalid={errors.site ? true : undefined}
                aria-describedby={errors.site ? 'site-error' : undefined}
              >
                <option value="">{t('upload.site.placeholder')}</option>
                {catalogue.sites.map((s) => (
                  <option key={s.site} value={s.site}>
                    {t(`sites.${s.site}`)}
                  </option>
                ))}
              </select>
              <FieldError id="site-error">
                {errors.site ? t(errors.site as 'upload.validation.site') : null}
              </FieldError>
            </div>
            <AnimatePresence initial={false}>
              {siteInfo ? (
                <m.div
                  key={siteInfo.site}
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  exit={{ opacity: 0, height: 0 }}
                  className="overflow-hidden"
                >
                  <p
                    className="mt-4 rounded-xl border border-line bg-surface-2 p-3 text-sm text-fg-muted"
                    aria-live="polite"
                  >
                    <span className="font-semibold text-fg">
                      {t('upload.site.selected', { site: t(`sites.${siteInfo.site}`) })}
                    </span>
                    <br />
                    {refInfo && siteInfo.pairing
                      ? t('upload.site.referenceSuggested', {
                          site: t(`sites.${refInfo.site}`),
                          pairing: t(`upload.pairing.${siteInfo.pairing}`),
                        })
                      : t('upload.site.noPair')}
                  </p>
                </m.div>
              ) : null}
            </AnimatePresence>
          </Section>
        </div>

        <div className="flex flex-col gap-5">
          <Section id="captures" step={2} title={t('upload.captures.title')}>
            <div className="grid gap-4 sm:grid-cols-2">
              <DropZone
                id="wl"
                label={t('upload.captures.wl')}
                hint={t('upload.captures.wlHint')}
                kind="image"
                required
                icon="camera"
                file={form.files.wl}
                onFile={(f) => setFile('wl', f)}
                error={errors.wl ? t(errors.wl as 'upload.validation.wl') : undefined}
              />
              <DropZone
                id="af"
                label={t('upload.captures.af')}
                hint={t('upload.captures.afHint')}
                kind="image"
                required
                icon="glow"
                accent="green"
                file={form.files.af}
                onFile={(f) => setFile('af', f)}
                error={errors.af ? t(errors.af as 'upload.validation.af') : undefined}
              />
              <DropZone
                id="dark"
                label={t('upload.captures.dark')}
                hint={t('upload.captures.darkHint')}
                kind="image"
                compact
                file={form.files.dark}
                onFile={(f) => setFile('dark', f)}
                error={errors.dark ? t(errors.dark as 'upload.validation.imageType') : undefined}
              />
              <DropZone
                id="spectrum"
                label={t('upload.captures.spectrum')}
                hint={t('upload.captures.spectrumHint')}
                kind="spectrum"
                compact
                icon="wave"
                file={form.files.spectrum}
                onFile={(f) => setFile('spectrum', f)}
                error={
                  errors.spectrum
                    ? t(errors.spectrum as 'upload.validation.spectrumType')
                    : undefined
                }
              />
            </div>

            {refInfo ? (
              <div className="mt-5 rounded-2xl border border-dashed border-ref/60 p-4">
                <ToggleChip
                  pressed={form.useReference}
                  onChange={(v) => set('useReference', v)}
                  icon="target"
                  testId="use-reference"
                >
                  {t('upload.site.useReference', { site: t(`sites.${refInfo.site}`) })}
                </ToggleChip>
                <AnimatePresence initial={false}>
                  {form.useReference ? (
                    <m.div
                      initial={{ opacity: 0, height: 0 }}
                      animate={{ opacity: 1, height: 'auto' }}
                      exit={{ opacity: 0, height: 0 }}
                      className="overflow-hidden"
                    >
                      <div className="grid gap-4 pt-4 sm:grid-cols-2">
                        <DropZone
                          id="refWl"
                          label={t('upload.captures.refWl')}
                          kind="image"
                          compact
                          file={form.files.refWl}
                          onFile={(f) => setFile('refWl', f)}
                        />
                        <DropZone
                          id="refAf"
                          label={t('upload.captures.refAf')}
                          kind="image"
                          compact
                          accent="green"
                          file={form.files.refAf}
                          onFile={(f) => setFile('refAf', f)}
                        />
                      </div>
                      <FieldError id="refPair-error">
                        {errors.refPair ? t(errors.refPair as 'upload.validation.refPair') : null}
                      </FieldError>
                    </m.div>
                  ) : null}
                </AnimatePresence>
              </div>
            ) : null}

            {synthetic ? (
              <p className="mt-4 flex items-start gap-2 rounded-xl border border-synthetic/60 bg-synthetic/15 p-3 text-sm">
                <Icon name="info" size={18} className="mt-0.5 shrink-0" />
                {t('upload.captures.syntheticDetected')}
              </p>
            ) : null}
          </Section>

          <Section id="seen" step={3} title={t('upload.seen.title')}>
            <ToggleChip
              pressed={form.lesionSeen}
              onChange={(v) => set('lesionSeen', v)}
              icon="eye"
              testId="lesion-seen"
            >
              {t('upload.seen.lesionSeen')}
            </ToggleChip>
            <AnimatePresence initial={false}>
              {form.lesionSeen ? (
                <m.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  exit={{ opacity: 0, height: 0 }}
                  className="overflow-hidden"
                >
                  <fieldset className="mt-4">
                    <legend className="field-label">{t('upload.seen.appearance')}</legend>
                    <div className="flex flex-wrap gap-2">
                      {APPEARANCES.map((a) => (
                        <ToggleChip
                          key={a}
                          pressed={form.appearance.includes(a)}
                          onChange={(on) =>
                            set(
                              'appearance',
                              on ? [...form.appearance, a] : form.appearance.filter((x) => x !== a),
                            )
                          }
                        >
                          {t(`upload.appearance.${a}`)}
                        </ToggleChip>
                      ))}
                    </div>
                  </fieldset>
                  <div className="mt-4">
                    <label htmlFor="note" className="field-label">
                      {t('upload.seen.note')}{' '}
                      <span className="text-xs font-normal text-fg-subtle">
                        ({t('common.optional')})
                      </span>
                    </label>
                    <textarea
                      id="note"
                      className="input min-h-20 resize-y"
                      maxLength={1000}
                      value={form.note}
                      placeholder={t('upload.seen.notePlaceholder')}
                      onChange={(e) => set('note', e.target.value)}
                    />
                  </div>
                  <div className="mt-4">
                    <Segmented
                      label={t('upload.seen.diascopy')}
                      value={form.diascopy}
                      onChange={(v) => set('diascopy', v)}
                      options={DIASCOPY.map((d) => ({
                        value: d,
                        label: t(`upload.diascopy.${d}`),
                      }))}
                    />
                    <p className="mt-2 text-xs text-fg-subtle">{t('upload.seen.diascopyHint')}</p>
                  </div>
                </m.div>
              ) : null}
            </AnimatePresence>
          </Section>

          <Section id="risk" step={4} title={t('upload.risk.title')}>
            <div className="grid gap-4 sm:grid-cols-[8rem_minmax(0,1fr)]">
              <div>
                <label htmlFor="age" className="field-label">
                  {t('upload.risk.age')}
                </label>
                <div className="relative">
                  <input
                    id="age"
                    className="input pr-14 tabular"
                    inputMode="numeric"
                    pattern="[0-9]*"
                    value={form.age}
                    onChange={(e) => set('age', e.target.value.replace(/[^0-9]/g, '').slice(0, 3))}
                    aria-invalid={errors.age ? true : undefined}
                    aria-describedby={errors.age ? 'age-error' : undefined}
                  />
                  <span className="pointer-events-none absolute top-1/2 right-3 -translate-y-1/2 text-sm text-fg-subtle">
                    {t('upload.risk.ageUnit')}
                  </span>
                </div>
                <FieldError id="age-error">
                  {errors.age ? t(errors.age as 'upload.validation.age') : null}
                </FieldError>
              </div>
              <Segmented
                label={t('upload.risk.sex')}
                value={form.sex}
                onChange={(v) => set('sex', v)}
                options={SEXES.map((s) => ({ value: s, label: t(`upload.sex.${s}`) }))}
              />
            </div>

            <fieldset className="mt-5">
              <legend className="field-label">{t('upload.risk.habits')}</legend>
              <div className="grid gap-2 sm:grid-cols-2">
                <ToggleChip
                  pressed={form.smokeless}
                  onChange={(v) => set('smokeless', v)}
                  hint={t('upload.risk.smokelessHint')}
                >
                  {t('upload.risk.smokeless')}
                </ToggleChip>
                <ToggleChip
                  pressed={form.smoking}
                  onChange={(v) => set('smoking', v)}
                  hint={t('upload.risk.smokingHint')}
                >
                  {t('upload.risk.smoking')}
                </ToggleChip>
                <ToggleChip
                  pressed={form.areca}
                  onChange={(v) => set('areca', v)}
                  hint={t('upload.risk.arecaHint')}
                >
                  {t('upload.risk.areca')}
                </ToggleChip>
                <ToggleChip pressed={form.alcoholDaily} onChange={(v) => set('alcoholDaily', v)}>
                  {t('upload.risk.alcohol')}
                </ToggleChip>
              </div>
            </fieldset>

            <fieldset className="mt-5">
              <legend className="field-label">{t('upload.risk.symptoms')}</legend>
              <div className="flex flex-wrap gap-2">
                <ToggleChip pressed={form.ulcer} onChange={(v) => set('ulcer', v)}>
                  {t('upload.risk.ulcer')}
                </ToggleChip>
                <ToggleChip pressed={form.whitePatch} onChange={(v) => set('whitePatch', v)}>
                  {t('upload.risk.whitePatch')}
                </ToggleChip>
                <ToggleChip pressed={form.redPatch} onChange={(v) => set('redPatch', v)}>
                  {t('upload.risk.redPatch')}
                </ToggleChip>
                <ToggleChip pressed={form.lump} onChange={(v) => set('lump', v)}>
                  {t('upload.risk.lump')}
                </ToggleChip>
                <ToggleChip pressed={form.trismus} onChange={(v) => set('trismus', v)}>
                  {t('upload.risk.trismus')}
                </ToggleChip>
                <ToggleChip pressed={form.neckLump} onChange={(v) => set('neckLump', v)}>
                  {t('upload.risk.neckLump')}
                </ToggleChip>
              </div>
            </fieldset>

            <fieldset className="mt-5">
              <legend className="field-label">{t('upload.risk.irritants')}</legend>
              <div className="flex flex-wrap gap-2">
                <ToggleChip pressed={form.sharpTooth} onChange={(v) => set('sharpTooth', v)}>
                  {t('upload.risk.sharpTooth')}
                </ToggleChip>
                <ToggleChip pressed={form.denture} onChange={(v) => set('denture', v)}>
                  {t('upload.risk.denture')}
                </ToggleChip>
              </div>
            </fieldset>

            <div className="mt-5 grid gap-4 sm:grid-cols-3">
              <div>
                <label htmlFor="provenance" className="field-label">
                  {t('upload.risk.provenance')}
                </label>
                <select
                  id="provenance"
                  className="input"
                  value={form.provenance}
                  onChange={(e) =>
                    setForm((f) => ({
                      ...f,
                      provenance: e.target.value as DataProvenance,
                      provenanceTouched: true,
                    }))
                  }
                >
                  {PROVENANCE.map((p) => (
                    <option key={p} value={p}>
                      {t(`upload.provenance.${p}`)}
                    </option>
                  ))}
                </select>
              </div>
              <div>
                <label htmlFor="camp" className="field-label">
                  {t('upload.risk.campName')}
                </label>
                <input
                  id="camp"
                  className="input"
                  value={form.campName}
                  onChange={(e) => set('campName', e.target.value)}
                />
              </div>
              <div>
                <label htmlFor="worker" className="field-label">
                  {t('upload.risk.workerId')}
                </label>
                <input
                  id="worker"
                  className="input"
                  value={form.workerId}
                  onChange={(e) => set('workerId', e.target.value)}
                />
              </div>
            </div>
          </Section>

          <Section id="consent" step={5} title={t('upload.consent.title')}>
            <div className="flex flex-col gap-3">
              <Checkbox
                id="consent"
                checked={form.consent}
                onChange={(v) => set('consent', v)}
                invalid={Boolean(errors.consent)}
                describedBy={errors.consent ? 'consent-error' : undefined}
                required
                testId="consent"
              >
                <span className="font-semibold">{t('upload.consent.required')}</span>
                <span className="ml-1 text-danger" aria-hidden="true">
                  *
                </span>
              </Checkbox>
              <FieldError id="consent-error">
                {errors.consent ? t(errors.consent as 'upload.validation.consent') : null}
              </FieldError>
              <Checkbox
                id="research"
                checked={form.research}
                onChange={(v) => set('research', v)}
                testId="research"
              >
                {t('upload.consent.research')}
              </Checkbox>
              <p className="text-xs text-fg-subtle">{t('upload.consent.note')}</p>
            </div>
          </Section>

          {errorEntries.length > 0 ? (
            <div
              ref={summaryRef}
              tabIndex={-1}
              role="alert"
              data-testid="error-summary"
              className="scroll-mt-32 rounded-2xl border-2 border-danger bg-risk-high-bg p-4 text-sm outline-none"
            >
              <p className="font-semibold text-fg">
                {t('upload.errorsSummary', { count: errorEntries.length })}
              </p>
              <ul className="mt-1.5 list-disc pl-5">
                {errorEntries.map(([field, key]) => (
                  <li key={field}>
                    <a
                      href={`#${FIELD_ANCHOR[field]}`}
                      className="font-medium text-danger underline underline-offset-2"
                    >
                      {t(key as 'upload.validation.site')}
                    </a>
                  </li>
                ))}
              </ul>
            </div>
          ) : null}
          <div className="sticky bottom-16 z-20 rounded-2xl border border-line bg-bg/85 p-2 shadow-card backdrop-blur-xl sm:bottom-4">
            <button
              type="submit"
              className="btn btn-primary min-h-14 w-full text-base"
              disabled={create.isPending}
              data-testid="submit-upload"
            >
              <Icon name="glow" size={20} />
              {create.isPending ? t('upload.submitting') : t('upload.submit')}
            </button>
          </div>
        </div>
      </form>

      <AnalysisOverlay
        open={overlay}
        mode="upload"
        thumbnailUrl={thumb}
        result={create.data?.latest ?? null}
        error={create.isError ? errorMessage(create.error, t) : null}
        onContinue={onContinue}
        onClose={onClose}
      />
    </div>
  );
}
