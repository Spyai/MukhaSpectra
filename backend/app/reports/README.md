# Report engine (clinician / referral report)

> Research prototype — not for clinical use. The report is a screening aid, not a diagnosis.

One `ScreeningResult` (the only clinical input) → Jinja2 HTML → WeasyPrint PDF (A4, 2–4 pages).
The HTML is self-contained: CSS is inlined and every image (artifacts, site map, spectral plot,
QR code, logo) is a data URI. That means the API can serve it as `report.html` and the PDF needs
nothing but the installed fonts (Inter, with DejaVu Sans as the fallback).

## Public API (`app.reports`)

```python
ArtifactReader = Callable[[str], bytes | None]          # missing -> None; exceptions are caught
PatientIdentity(name=None, external_id=None)            # PII, joined only at render time
Facility(name, address, phone=None, hours=None, notes=None, example=False,
         levels=[], capabilities=[])                     # levels: risk levels it serves
ProgrammeContact(name, phone=None)
ReportOptions(report_id, generated_at, record_url=None, facilities=[], report_version=1,
              display_tz="Asia/Kolkata", programme_contact=None)

render_clinician_report_html(result, artifacts, *, identity=None, options) -> str
render_clinician_report_pdf(result, artifacts, *, identity=None, options) -> bytes
render_site_map_svg(result=None, *, width=400, show_legend=True) -> str   # also /sitemap.svg
spectral_plot_svg(lesion, reference) -> str                               # matplotlib SVG
load_facilities(path=None) -> list[Facility]   # default: backend/app/config/facilities.yaml
```

## Report specification (brief §6.1)

Content order: header → disclaimer → patient → overall risk → site map → findings → QC →
review → referral. Page 1 carries everything needed to act (risk, action, timeline with due
date, coverage, reasons, map); later pages carry the evidence for each finding, then QC,
sign-off and referral. The header repeats the level, timeline and action next to the title so
the level is read within seconds (top 35 % of page 1, tested). All times are printed in
`ReportOptions.display_tz` (IST by default).

| # | Section | Contents |
|---|---|---|
| 1 | Header | placeholder logo, compact level badge + timeline (due date) + action, report id + version + generated time; camp, health worker, captured/analysed times, device IDs, firmware, calibration status, data provenance; result + screening id, software, pipeline, schema, rule/QC/analysis config versions + short SHA-256; one-line model summary ("all unvalidated · White-light classifier: not available — not used in triage (see §5)") |
| 2 | Disclaimer | "Screening aid — not a diagnosis" box using the result's own disclaimer text; the SYNTHETIC notice (red) or a phantom / ex vivo / volunteer notice (amber) as a second line |
| 3 | Patient and risk factors | name and masked external ID ("Patient ID (masked) XX-XXXX-XXXX-0123") only if a `PatientIdentity` is passed, pseudonymous ID, age, sex, consent; plain-words risk summary ("Gutka 20 y (6/day) · bidi 15 y · alcohol daily; non-healing ulcer > 2 weeks") |
| 4 | Overall risk | badge = colour + icon + text (LOW teal, MODERATE amber, HIGH red, INCONCLUSIVE slate dashed; icon encodes the level by shape: 1/2/3 bars or "?"), action summary, steps, timeline with due date, urgency; a clinician-reviewed level when the review changed it ("Clinician-reviewed level: MODERATE (device: HIGH) — Dr A. Rao, 20 Oct 2026", also in the header and referral); for INCONCLUSIVE a note for the receiving clinician; device coverage (sites imaged of 14, not examined, not assessed → "examine clinically"). "Why this level": explanation, rule score on a gauge against the thresholds (never shown without reasons), factor table (reason, source, points); red-flag overrides tagged "Red flag (floor)" with one footnote; reported symptoms say "site not recorded — examine the whole mouth and neck" |
| 5 | Oral site map | SVG from `mukha_pipeline.sites` geometry, viewed facing the patient, in a 74 mm column with every text ≥ 2.1 mm; upper/lower spelled out ("R"/"L" only mean right/left); examined sites filled by `site_risk_level` (INCONCLUSIVE hatched grey) and carrying the level icon, reference-only blue dashed, not examined light grey; numbered markers match finding numbers; HTML legend below the map |
| 6 | Suspicious findings | top 2 findings by severity then score in full (rest in a compact table; a 5-finding result stays within 4 pages). Each block may continue on the next page between its parts (header + images + measurements, spectral block and factor table never split): WL / AF / AF-loss heatmap cropped to the 58 × 30 mm frame around the flagged region (same crop for every panel in the WL frame; brackets mark the region, the heatmap keeps its own outline) with the heatmap colour key ("relative AF 0.85 → 0.20 · no colour = no loss · grey = not analysed"), measurements (location in the photo, area mm² or % of mucosa, mean/min relative AF, contrast, border irregularity, reference used + threshold, confounder flags), visual observation (or "no lesion recorded"), diascopy, spectral plot vs reference with ± SD bands + feature/delta table (features cited in the reasons in bold) + classifier label (placeholder scores hidden), WL classifier output or "not available — not used in triage", contributing factors. A reading shared by two findings is plotted once |
| 7 | QC and limitations | per-site QC / analysis / calibration / site level ("Reference only" for the healthy reference site), failed or warned captures with the plain-language retake prompt, recommended retakes, models with status and `validated: No`, all `result.limitations` |
| 8 | Clinician review | reviewer name and ID, agree / modify (the other levels), clinical findings on examination, biopsy taken (site) / planned / not indicated, follow-up date, notes, signature, date; pre-filled when `result.review` exists ("recorded in MukhaSpectra — not a digital signature") |
| 9 | Referral | urgency + timeline (and the reviewed level), a stand-alone slip (patient, reason: level, top finding, red flags), facilities filtered by `levels` (urgent / re-examine: up to 3 cards, first = "First-choice facility"; none / retake: one line "If a symptom appears or a patch persists > 2 weeks: …"); `example: true` entries are labelled and dropped for non-synthetic results; "Referral facility not configured — contact the screening programme" (and a logged warning) when none apply; screening and programme contact; QR code to `options.record_url` with the URL printed, or a note |

Every page: running header "Research prototype — not for clinical use" (+ "SYNTHETIC DATA" or
"PHANTOM DATA") and "Report <id> · <pseudonymous id> · <age> <sex>", footer with software /
pipeline / schema / rules version (short SHA) and "Page X of Y", and a diagonal translucent
"SYNTHETIC DATA" watermark when the result is synthetic. Printing `report.html` from a browser
keeps the colours (`print-color-adjust: exact`). Tabular figures are applied only to numeric
cells (Inter's tabular hyphen would space out IDs).

Rules enforced by tests (`backend/tests/test_reports*.py`): every section present and in order;
every fixture, including a 5-finding stress case, renders to 2–4 pages; level in the top 35 %
of page 1; findings heading never stranded; banner, report id, patient pseudonym and page footer
on every page; watermark on every page only for synthetic results; site-map text ≥ 2.1 mm; missing / failing / corrupt artifacts give an "Image not
available" placeholder; site-map colours match levels; positioning check
(`mukha_pipeline.positioning`) on HTML and extracted PDF text; no "accuracy / sensitivity /
specificity" anywhere; free text is HTML-escaped. At render time the engine also scans its own
output and logs a warning if pipeline-provided wording (labels, explanations, health-worker notes)
breaks the positioning rules; it never blocks a referral over wording.

## Files

```
reports/
  __init__.py      public API
  types.py         ArtifactReader, PatientIdentity, Facility, ReportOptions
  render.py        Jinja2 environment, HTML + PDF rendering, visible-text helper
  context.py       ScreeningResult -> template view model (all show/hide decisions)
  labels.py        every user-facing word and number format (audit positioning here)
  sitemap.py       oral site map SVG
  charts.py        spectral plot (matplotlib SVG), score gauge, risk icons
  images.py        artifact -> downscaled JPEG data URI, region brackets, placeholders
  qr.py            QR code (SVG)
  facilities.py    facilities.yaml loader
  preview.py       CLI preview
  templates/clinician_report.html.j2
  static/report.css, static/logo.svg (placeholder wordmark)
```

## Preview

From `backend/` (fixtures are SYNTHETIC and live in `backend/tests/report_fixtures`):

```bash
python -m app.reports.preview --fixture high --out /tmp/report.pdf --html --sitemap /tmp/map.svg
python -m app.reports.preview --fixture inconclusive --out /tmp/report.html   # no QR code
python -m app.reports.preview --fixture stress --out /tmp/stress.pdf          # 5 findings
python -m app.reports.preview --result path/to/result.json --artifacts path/to/v1/ --out /tmp/r.pdf
```

Fixtures: `high` (HIGH; two findings incl. a porphyrin confounder and a spectral pair; WL model
unavailable; one site failed QC), `low` (LOW, no findings), `inconclusive` (uncalibrated device,
failed AF capture, no QR code), `stress` (HIGH with five findings, WL classifier output, a
review changing the level to MODERATE, long names, ABHA-style ID). Fixture heatmaps are drawn
with the pipeline's own renderer (`mukha_pipeline.af.overlay`).

Fonts: names in Kannada or Devanagari need Noto Sans Kannada / Noto Sans Devanagari installed
(they are in the font stack); without them fontconfig falls back to FreeSans / FreeSerif.

Rendering a 4-page report takes about 2 s warm (≈ 3 s including first-call imports).
