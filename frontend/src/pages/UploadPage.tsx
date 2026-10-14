import { AnimatePresence, m } from 'motion/react';
import { useCallback, useEffect, useMemo, useRef, useState, type FormEvent } from 'react';
import { useTranslation } from 'react-i18next';
import { useNavigate, useSearchParams } from 'react-router';
import { IS_MOCK } from '../api/client';
import { useCreateScreening, useSiteCatalogue } from '../api/queries';
import type { DataProvenance, Diascopy, OralSite, Sex, VisualAppearance } from '../api/types';
import { AnalysisOverlay } from '../components/analysis/AnalysisOverlay';
import { DropZone } from '../components/upload/DropZone';
import { PREVIEWABLE } from '../components/upload/fileTypes';
import {
  ALCOHOL_ANSWERS,
  buildUploadRequest,
  filesLookSynthetic,
  HABIT_STATUSES,
  initialUploadForm,
  SMOKELESS_PRODUCTS,
  SMOKING_PRODUCTS,
  validateUpload,
  type FileSlot,
  type HabitAnswer,
  type SmokelessProduct,
  type SmokingProduct,
  type UploadErrors,
  type UploadFormState,
} from '../components/upload/uploadForm';
import { Checkbox, FieldError, Section, Segmented, ToggleChip } from '../components/ui/controls';
import { Icon } from '../components/ui/Icon';
import { SiteMap } from '../components/viz/SiteMap';
import { errorMessage } from '../lib/errors';
import { BUILTIN_SITE_CATALOGUE, findSite, isOralSite } from '../lib/sites';

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

/** `/upload?site=lower_gingiva` (from a retake prompt) preselects that site and its pair. */
function initialFormFor(site: string | null): UploadFormState {
  if (!isOralSite(site)) return initialUploadForm;
  const info = findSite(BUILTIN_SITE_CATALOGUE, site);
  return { ...initialUploadForm, site, referenceSite: info?.paired_site ?? null };
}

/** Focus the field behind an error-summary link and bring its whole control into view. */
function focusField(id: string) {
  const el = document.getElementById(id);
  if (!el) return;
  el.focus({ preventScroll: true });
  const box =
    el.closest('[data-testid^="dropzone"]') ?? el.closest('label') ?? el.closest('div') ?? el;
  box.scrollIntoView({ block: 'center' });
}

export default function UploadPage() {
  const { t, i18n } = useTranslation();
  const navigate = useNavigate();
  const [params] = useSearchParams();
  const catalogue = useSiteCatalogue();
  const create = useCreateScreening();
  const [form, setForm] = useState<UploadFormState>(() => initialFormFor(params.get('site')));
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
        className="mt-8 grid grid-cols-1 gap-5 lg:grid-cols-[minmax(0,0.9fr)_minmax(0,1.1fr)] lg:items-start"
      >
        <div className="min-w-0 lg:sticky lg:top-28">
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

        <div className="flex min-w-0 flex-col gap-5">
          <Section id="captures" step={2} title={t('upload.captures.title')}>
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
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
                      <div className="grid grid-cols-1 gap-4 pt-4 sm:grid-cols-2">
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
                </m.div>
              ) : null}
            </AnimatePresence>
            {/* Glow loss is judged on the violet-light screen, so the glass-slide press is
                offered whenever an AF photo is attached, lesion seen by eye or not. */}
            <AnimatePresence initial={false}>
              {form.files.af ? (
                <m.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  exit={{ opacity: 0, height: 0 }}
                  className="overflow-hidden"
                  data-testid="diascopy"
                >
                  <div className="mt-5 border-t border-line pt-5">
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
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-[8rem_minmax(0,1fr)]">
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
              </div>
              <Segmented
                label={t('upload.risk.sex')}
                value={form.sex}
                onChange={(v) => set('sex', v)}
                options={SEXES.map((s) => ({ value: s, label: t(`upload.sex.${s}`) }))}
              />
            </div>
            <FieldError id="age-error">
              {errors.age ? t(errors.age as 'upload.validation.age') : null}
            </FieldError>

            <fieldset className="mt-6">
              <legend className="field-label">{t('upload.risk.habits')}</legend>
              <p className="-mt-1 mb-3 text-xs text-fg-subtle">{t('upload.risk.habitsHint')}</p>
              <div className="flex flex-col gap-3">
                <HabitField
                  id="smokeless"
                  label={t('upload.risk.smokeless')}
                  hint={t('upload.risk.smokelessHint')}
                  perDayLabel={t('upload.risk.perDay.smokeless')}
                  products={SMOKELESS_PRODUCTS}
                  value={form.smokeless}
                  onChange={(v) => set('smokeless', v)}
                />
                <HabitField
                  id="smoking"
                  label={t('upload.risk.smoking')}
                  hint={t('upload.risk.smokingHint')}
                  perDayLabel={t('upload.risk.perDay.smoking')}
                  products={SMOKING_PRODUCTS}
                  value={form.smoking}
                  onChange={(v) => set('smoking', v)}
                />
                <HabitField
                  id="areca"
                  label={t('upload.risk.areca')}
                  hint={t('upload.risk.arecaHint')}
                  perDayLabel={t('upload.risk.perDay.areca')}
                  value={form.areca}
                  onChange={(v) => set('areca', v)}
                />
                <div className="rounded-2xl border border-line p-3.5">
                  <Segmented
                    label={t('upload.risk.alcohol')}
                    value={form.alcohol}
                    onChange={(v) => set('alcohol', v)}
                    options={ALCOHOL_ANSWERS.map((a) => ({
                      value: a,
                      label: t(`upload.risk.alcoholFrequency.${a}`),
                      muted: a === 'unknown',
                    }))}
                    compact
                  />
                </div>
              </div>
            </fieldset>

            <fieldset className="mt-6">
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
                <ToggleChip pressed={form.pain} onChange={(v) => set('pain', v)}>
                  {t('upload.risk.pain')}
                </ToggleChip>
                <ToggleChip pressed={form.bleeding} onChange={(v) => set('bleeding', v)}>
                  {t('upload.risk.bleeding')}
                </ToggleChip>
                <ToggleChip
                  pressed={form.burning}
                  onChange={(v) => set('burning', v)}
                  hint={t('upload.risk.burningHint')}
                >
                  {t('upload.risk.burning')}
                </ToggleChip>
                <ToggleChip pressed={form.trismus} onChange={(v) => set('trismus', v)}>
                  {t('upload.risk.trismus')}
                </ToggleChip>
                <ToggleChip pressed={form.neckLump} onChange={(v) => set('neckLump', v)}>
                  {t('upload.risk.neckLump')}
                </ToggleChip>
              </div>
            </fieldset>

            <div className="mt-6 grid grid-cols-1 gap-x-6 gap-y-5 sm:grid-cols-2">
              <fieldset>
                <legend className="field-label">{t('upload.risk.history')}</legend>
                <div className="flex flex-wrap gap-2">
                  <ToggleChip pressed={form.priorLesion} onChange={(v) => set('priorLesion', v)}>
                    {t('upload.risk.priorLesion')}
                  </ToggleChip>
                  <ToggleChip pressed={form.priorCancer} onChange={(v) => set('priorCancer', v)}>
                    {t('upload.risk.priorCancer')}
                  </ToggleChip>
                </div>
              </fieldset>
              <fieldset>
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
            </div>

            <div className="mt-6 grid grid-cols-1 gap-4 sm:grid-cols-3">
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
                      className="inline-block py-1 font-medium text-danger underline underline-offset-2"
                      onClick={(e) => {
                        e.preventDefault();
                        focusField(FIELD_ANCHOR[field]);
                      }}
                    >
                      {t(key as 'upload.validation.site')}
                    </a>
                  </li>
                ))}
              </ul>
            </div>
          ) : null}
          <div className="sticky bottom-[max(0.5rem,env(safe-area-inset-bottom))] z-20 rounded-2xl border border-line bg-bg/85 p-2 shadow-card backdrop-blur-xl sm:bottom-4">
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

type Product = SmokelessProduct | SmokingProduct;

/** One habit: Not asked / Current / Former / Never, then products, years and amount per day. */
function HabitField<P extends Product = never>({
  id,
  label,
  hint,
  perDayLabel,
  products,
  value,
  onChange,
}: {
  id: string;
  label: string;
  hint?: string;
  perDayLabel: string;
  products?: readonly P[];
  value: HabitAnswer<P>;
  onChange: (v: HabitAnswer<P>) => void;
}) {
  const { t } = useTranslation();
  const used = value.status === 'current' || value.status === 'former';
  const digits = (s: string, n: number) => s.replace(/[^0-9]/g, '').slice(0, n);
  return (
    <div className="rounded-2xl border border-line p-3.5" data-testid={`habit-${id}`}>
      <Segmented
        label={label}
        hint={hint}
        value={value.status}
        onChange={(status) => onChange({ ...value, status })}
        options={HABIT_STATUSES.map((s) => ({
          value: s,
          label: t(`upload.risk.habitStatus.${s}`),
          muted: s === 'unknown',
        }))}
        compact
      />
      <AnimatePresence initial={false}>
        {used ? (
          <m.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="overflow-hidden"
          >
            <div className="flex flex-col gap-3 pt-3">
              {products ? (
                <fieldset>
                  <legend className="mb-1.5 text-xs font-semibold text-fg-muted">
                    {t('upload.risk.products')}
                  </legend>
                  <div className="flex flex-wrap gap-1.5">
                    {products.map((p) => (
                      <ToggleChip
                        key={p}
                        size="sm"
                        pressed={value.products.includes(p)}
                        onChange={(on) =>
                          onChange({
                            ...value,
                            products: on
                              ? [...value.products, p]
                              : value.products.filter((x) => x !== p),
                          })
                        }
                      >
                        {t(`upload.product.${p as Product}`)}
                      </ToggleChip>
                    ))}
                  </div>
                </fieldset>
              ) : null}
              <div className="grid max-w-sm grid-cols-2 gap-3">
                <div>
                  <label
                    htmlFor={`${id}-years`}
                    className="mb-1 block text-xs font-semibold text-fg-muted"
                  >
                    {t('upload.risk.years')}{' '}
                    <span className="font-normal text-fg-subtle">({t('common.optional')})</span>
                  </label>
                  <input
                    id={`${id}-years`}
                    className="input tabular"
                    inputMode="numeric"
                    pattern="[0-9]*"
                    value={value.years}
                    onChange={(e) => onChange({ ...value, years: digits(e.target.value, 2) })}
                  />
                </div>
                <div>
                  <label
                    htmlFor={`${id}-perday`}
                    className="mb-1 block text-xs font-semibold text-fg-muted"
                  >
                    {perDayLabel}{' '}
                    <span className="font-normal text-fg-subtle">({t('common.optional')})</span>
                  </label>
                  <input
                    id={`${id}-perday`}
                    className="input tabular"
                    inputMode="numeric"
                    pattern="[0-9]*"
                    value={value.perDay}
                    onChange={(e) => onChange({ ...value, perDay: digits(e.target.value, 3) })}
                  />
                </div>
              </div>
            </div>
          </m.div>
        ) : null}
      </AnimatePresence>
    </div>
  );
}
