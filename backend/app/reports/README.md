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
Facility(name, address, phone=None, hours=None, notes=None, example=False)
ReportOptions(report_id, generated_at, record_url=None, facilities=[], report_version=1)

render_clinician_report_html(result, artifacts, *, identity=None, options) -> str
render_clinician_report_pdf(result, artifacts, *, identity=None, options) -> bytes
render_site_map_svg(result=None, *, width=400, show_legend=True) -> str   # also /sitemap.svg
spectral_plot_svg(lesion, reference) -> str                               # matplotlib SVG
load_facilities(path=None) -> list[Facility]   # default: backend/app/config/facilities.yaml
```

## Report specification (brief §6.1)

Content order: header → disclaimer → patient → overall risk → site map → findings → QC →
review → referral. Page 1 carries everything needed to act (risk, action, timeline, reasons,
map); later pages carry the evidence for each finding, then QC, sign-off and referral.

| # | Section | Contents |
|---|---|---|
| 1 | Header | placeholder logo, report id + version, generated/captured/analysed times, result id + version, camp, health worker, device IDs, firmware, calibration status, data provenance; software, pipeline, schema, rule/QC/analysis config versions + short SHA-256, every model with honest status |
| 2 | Disclaimer | "Screening aid — not a diagnosis" box using the result's own disclaimer text; SYNTHETIC notice when `result.synthetic` |
| 3 | Patient and risk factors | name / external ID only if a `PatientIdentity` is passed, pseudonymous ID, age, sex, consent; one-line plain-words risk summary ("Gutka 20 y, bidi 15 y, alcohol daily; non-healing ulcer > 2 weeks") |
| 4 | Overall risk | badge = colour + icon + text (LOW teal, MODERATE amber, HIGH red, INCONCLUSIVE slate dashed; icon encodes the level by shape: 1/2/3 bars or "?"), action summary, steps, timeline in days, urgency. "Why this level": explanation, rule score on a gauge against the thresholds (never shown without reasons), factor table (reason, source, points); clinical red-flag overrides marked "Clinical red flag — the device cannot lower this" |
| 5 | Oral site map | SVG from `mukha_pipeline.sites` geometry, viewed facing the patient; examined sites filled by `site_risk_level` (INCONCLUSIVE hatched grey), reference-only blue dashed, not examined light grey; numbered markers match finding numbers |
| 6 | Suspicious findings | top 3 findings by severity then score (rest in a compact table). Each block (`break-inside: avoid`): WL / AF / AF-loss heatmap (brackets mark the region when the image is in the WL frame), measurements (location, area mm² or % of mucosa, mean/min relative AF, contrast, border irregularity, reference used + threshold, confounder flags), visual observation, diascopy, spectral plot vs reference with ± SD bands + feature/delta table + classifier label, reasons and "placeholder" status, WL classifier output or "not available — not used in triage", contributing factors |
| 7 | QC and limitations | per-site QC / analysis / calibration / site level, failed or warned captures with the plain-language retake prompt, recommended retakes, models with status and `validated: No`, all `result.limitations` |
| 8 | Clinician review | reviewer name and ID, agree / modify to LOW / MODERATE / HIGH, notes, signature, date; pre-filled when `result.review` exists |
| 9 | Referral | urgency + timeline, up to 3 configured facilities (first = nearest; `example: true` entries are labelled "Example facility — configure facilities.yaml"), screening contact, QR code to `options.record_url` with the URL printed; a note instead when no URL is configured |

Every page: running header "Research prototype — not for clinical use" + report id, footer with
software / pipeline / schema / rules version (short SHA) and "Page X of Y", and a diagonal
translucent "SYNTHETIC DATA" watermark when the result is synthetic.

Rules enforced by tests (`backend/tests/test_reports*.py`): every section present and in order;
HIGH fixture renders to 2–4 pages; banner, report id and page footer on every page; watermark on
every page only for synthetic results; missing / failing / corrupt artifacts give an "Image not
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
python -m app.reports.preview --fixture inconclusive --out /tmp/report.html
python -m app.reports.preview --result path/to/result.json --artifacts path/to/v1/ --out /tmp/r.pdf
```

Fixtures: `high` (HIGH; two findings incl. a porphyrin confounder and a spectral pair; WL model
unavailable; one site failed QC), `low` (LOW, no findings), `inconclusive` (uncalibrated device,
failed AF capture, no QR code).

Rendering a 4-page report takes about 2 s warm (≈ 3 s including first-call imports).
