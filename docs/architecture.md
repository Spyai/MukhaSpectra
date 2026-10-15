# MukhaSpectra architecture

> Research prototype, not for clinical use. MukhaSpectra is a **screening and triage aid**. It
> flags suspicious findings and assigns a risk level. It never diagnoses. Final diagnosis
> requires a clinician and histopathology.

## 1. What we are building

A health worker at a screening camp holds a phone with the **Scout** clip over a patient's mouth.
For each oral site, Scout captures a white-light (WL) photo and a 405 nm autofluorescence (AF)
photo. Healthy mucosa glows green under violet light; dysplastic tissue loses that glow
(fluorescence visualisation loss, FVL). At a flagged spot, the **Confirm** probe (AS7341, 8
visible channels + clear + NIR) measures diffuse reflectance, where extra blood volume deepens
the 540–580 nm haemoglobin dip.

The software turns those captures, plus a short risk questionnaire, into:

1. a per-lesion and per-patient **risk level** (`LOW` / `MODERATE` / `HIGH` / `INCONCLUSIVE`)
   with the reasons behind it,
2. a clinician referral **PDF report** and a one-page patient report, and
3. a growing, biopsy-labelled dataset. That dataset is the long-term moat.

Two rules shape every design decision:

* **Positioning.** No UI text, report or API field says "diagnosis", "cancer detected" or
  "malignant". We say "risk level", "suspicious finding", "refer for clinical examination /
  biopsy". The only allowed uses of "diagnosis" are the mandated disclaimers ("Screening aid —
  not a diagnosis", "final diagnosis requires a clinician and histopathology"). A test enforces it.
* **Honesty.** We have no trained clinical model and no validation data. No accuracy number is
  hard-coded or displayed. Every model ships as a labelled baseline or placeholder and carries
  `validated: false`. Synthetic data is labelled synthetic in pixels, file names, metadata and
  results, and never feeds a performance claim. Every screen and report shows
  **"Research prototype — not for clinical use"**.

## 2. System overview

```mermaid
flowchart LR
  subgraph Capture["Capture (PWA, offline-first)"]
    U[UploadAdapter]:::p1
    P[PhoneCameraAdapter]
    S[ScoutDeviceAdapter BLE]
    C[ConfirmProbeAdapter BLE/Serial]
    D[DedicatedCameraAdapter stub]
    M[MockDeviceAdapter]
  end
  subgraph Pipeline["pipeline/ (pure Python, unit-tested)"]
    QC[QC] --> CAL[Calibration] --> REG[WL↔AF registration] --> SEG[Mucosa segmentation]
    SEG --> AF[AF analysis: maps, reference, FVL, confounders]
    SEG --> WL[WL LesionClassifier]
    SP[Spectral features + classifier]
    AF & WL & SP --> FUS[Rule-based fusion + triage]
  end
  subgraph Backend["backend/ (FastAPI)"]
    API[REST API] --> ORCH[Orchestrator]
    ORCH --> DB[(SQLite / PostgreSQL)]
    ORCH --> OBJ[(Object storage)]
    API --> REP[Report engine: Jinja2 → WeasyPrint]
  end
  Capture -->|captures + metadata| API
  ORCH --> Pipeline
  REP -->|clinician PDF, patient report| Users[Health worker / clinician]
  classDef p1 fill:#5b3cc4,color:#fff
```

Phase 1 implements the upload path end to end: upload WL + AF (+ optional spectrum) → pipeline
→ `ScreeningResult` → clinician PDF.

## 3. Repository layout

```
mukhaspectra/
  pipeline/mukha_pipeline/   pure analysis library (no I/O beyond decoding bytes)
    schema/                  ScreeningResult v1 (pydantic) + generated JSON Schema
    config/                  qc.v1.yaml, analysis.v1.yaml, triage_rules.v1.yaml
    sites.py                 oral site catalogue, contralateral pairing, site-map geometry
    types.py                 internal dataclasses shared by modules
    io/                      image decoding (JPG/PNG/HEIC/DNG), spectral CSV/JSON parsing
    capture/                 CaptureSource abstraction, UploadAdapter, SyntheticSource
    qc/                      image + spectral quality control
    calibration/             flat-field, colour, AF scale, spectral reflectance, expiry
    preprocess/              dark subtraction, corrections, registration, segmentation
    af/                      fluorescence maps, reference normalisation, FVL, confounders, overlay
    wl/                      LesionClassifier interface, ONNX runner, saliency
    spectral/                reflectance summary, features, SpectralClassifier baseline
    fusion/                  rule-based triage
    synthetic/               synthetic WL/AF/spectra generator, demo patients (SYNTHETIC)
    orchestrator.py          analyze_screening(): inputs → ScreeningResult + artifacts
  backend/app/               FastAPI app, DB models, storage, report engine
  frontend/                  React + Vite + TypeScript (Phase 1: analysis console; Phase 2: PWA)
  firmware/                  ESP32 Scout + Confirm (Phase 3)
  ml/                        training/eval, ONNX export, model cards (Phase 4)
  sample_data/               synthetic demo patients (labelled) + bench data slots
  docs/                      this file, data model, report spec, risk register, requirements
```

## 4. The contract: `ScreeningResult` v1

`pipeline/mukha_pipeline/schema/models.py` is the source of truth. The JSON Schema
(`pipeline/mukha_pipeline/schema/screening_result.v1.schema.json`, draft 2020-12) is generated
from it and a test fails on drift. The frontend generates TypeScript types from the JSON Schema.

```
ScreeningResult
├── schema_version "1.0.0", result_id, screening_id, result_version, supersedes_result_id
├── data_provenance (real | synthetic | phantom | ex_vivo | volunteer | mixed), synthetic: bool
├── notices        prototype banner, disclaimer, synthetic notice
├── versions       software ("<version>+<build>"), pipeline, schema, rule/qc/analysis config {name, version, sha256},
│                  models[] {component, name, version, status, validated=false, model_card}
├── context        camp, health worker, device IDs, firmware, capture time, language
├── patient        pseudonymous_id, age, sex            (name/contact live in a separate PII table)
├── consent        recorded, method, language, research_use_opt_in
├── risk_factors   smokeless tobacco, smoking, areca nut, alcohol, history, symptoms, irritants
├── sites[]        per oral site
│   ├── visual_flag (lesion seen by eye + appearance), diascopy
│   ├── captures[] {metadata (adapter, device, firmware, LED, exposure, ISO, WB, site,
│   │              calibration id, sha256, synthetic), qc {passed, score, issues[]}}
│   ├── qc, analysis_status, calibration status
│   ├── registration {method, confidence, warp}, segmentation {mucosa fraction, confidence}
│   ├── af {reference (contralateral | analogous | in-image), medians, regions[] {area px/mm²,
│   │       mean/min relative AF, contrast, border raggedness (1 − hull/perimeter), confounders[]}}
│   ├── wl {status (model_unavailable until weights exist), predictions, saliency (clinician)}
│   ├── artifacts {wl, af, af_registered, heatmap_overlay, relative_af_map, relative_af_grid}
│   └── site_risk_level
├── spectral_readings[] {values ± std, reflectance vs relative signal, QC, features,
│                        delta vs reference (lesion/reference quotient), classifier (placeholder)}
├── lesions[]      {site, origins (af_fvl | visual | wl_model | spectral), risk_level, score,
│                   contributing_factors[] {code, label, source, points}, confounders,
│                   recommended_action}
├── triage         overall_risk_level, score + thresholds, factors, overrides_applied,
│                  recommended_action {code, steps, timeline_days, urgency}, explanation,
│                  requires_clinician_review = true, review_status = pending_review
├── qc_summary, calibration_status, limitations[]
├── processing     working resolution, per-stage timings
└── review         clinician sign-off (Phase 4)
```

Versioning: additive changes bump the minor version; breaking changes get a new major schema
file. Re-analysis never overwrites: it writes `result_version + 1` with `supersedes_result_id`.

## 5. Pipeline design

### 5.1 Conventions
* Images are float32 RGB in [0, 1], shape (H, W, 3). Masks are bool (H, W).
* Decoded images hold sRGB-encoded file values. WL stays encoded; AF and dark frames are
  converted to **linear light** before any radiometric step (dark subtraction, AF flat field,
  exposure scaling, every AF ratio). On encoded values a 2x exposure reads as ~1.37x and relative
  AF 0.75 would mean ~0.53 of the light. QC judges files as stored; images shown to people are
  re-encoded.
* Uploads are decoded at most 2048 px (longest side) while decoding; images over 50 MP are
  refused. `working_resolution.scale_from_original` is relative to the original capture.
* Analysis runs at a working resolution (longest side 1024 px) in the **WL frame**. AF is warped
  into the WL frame before any per-pixel comparison. The scale factor is recorded.
* Pure functions. The orchestrator returns artifacts as bytes; the backend decides storage.
* Every threshold lives in versioned YAML; every result records the config SHA-256.

### 5.2 Per-site flow (orchestrator)

```
for each site:
  QC(WL), QC(AF), QC(dark) on the stored values → failures block analysis of that site
                                           (AF not green / auto white balance / greyscale caught)
  WL: white-LED flat field + colour matrix  → calibration status: calibrated | uncalibrated | expired
  AF: sRGB → linear; − linear dark; ÷ violet-LED flat field (from the fluorescence card, never
      the white profile); intensity scale = exposure factor × calibration AF scale
  register AF → WL (ECC affine, ORB+RANSAC homography fallback) on the corrected AF re-encoded
      to sRGB → confidence → QC; registered AF linearised again
  segment WL → mucosa / teeth / background / specular (Segmenter interface; U-Net later)
  refine with AF: whitish areas that do not glow like enamel go back into the mucosa;
      segmentation failed → QC error, low_confidence → warning (site cannot give LOW)
  AF maps (linear): green, green/red ratio, AF index (exposure- and calibration-normalised)
  reference green: max(paired, own) where paired = contralateral > analogous site with no
      recorded lesion; otherwise in-image robust mean (50th–95th percentile, highlights set aside)
  relative AF = smoothed green / reference → FVL = relative AF < threshold, cleaned, min area
  per region: area (px, mm²), mean/min relative AF, contrast, border raggedness, solidity
  confounders: red porphyrin excess (plaque), pigmentation (brown, a* < 20), blood/red (a* ≥ 25),
      small specular glare → overlap per region
  heatmap overlay of AF loss on WL; downsampled relative-AF grid for interactive 3D view
  WL LesionClassifier (model_unavailable if no weights) + occlusion saliency (clinician only)
  a crash in either pass → site "not_analysed" (software error), never a QC failure
spectral: (S−D)/(W−D) or relative signal → features → quotient delta vs reference → placeholder
          classifier
fusion: lesions ← FVL regions ∪ visual flags ∪ WL model ∪ every lesion probe reading → rule
        scores → lesion floors (strong FVL, red area) → floors from clinical red flags, a seen
        lesion and reported symptoms → patient level → INCONCLUSIVE safety rule → actions
```

Two passes are needed for contralateral normalisation: pass 1 computes each site's own reference
green; pass 2 resolves each site's reference from the sites that passed QC and have no recorded
lesion, using max(paired, own) so a dimmer opposite side (exposure, angle, or change on both
sides) can never hide a loss. When the opposite side reads more than 10% dimmer the result adds
the limitation `possible_bilateral_change`.

### 5.3 Triage rules (v1, transparent)

`config/triage_rules.v1.yaml` (1.1.0), status `baseline_heuristic_unvalidated`. Points per
lesion come from AF (presence, depth, area, border raggedness, contrast, with multipliers for
reference quality, diascopy and confounders), the WL model (ignored while unavailable), spectral
quotient deltas, the health worker's own visual observation and site risk. Patient risk factors
add a capped amount only when a lesion exists. The multipliers order findings but are bounded:

* **Lesion floors:** strong fluorescence loss (≥ 3.0 AF points before discounts, or mean relative
  AF ≤ 0.55) and a red area under FVL are at least MODERATE. A discount for plaque or glare adds a
  "clean the area and retake" step (worded never to delay a HIGH referral). Low-specificity and
  in-image reference discounts are not compounded; diascopy "resolves" applies only when the site
  has exactly one dark region.
* **Patient floors** the device can never lower: non-healing ulcer > 2 weeks, red patch, growth,
  neck lump, lump with a lesion, previous oral cancer with a lesion → HIGH; white patch, ulcer,
  trismus, bleeding, reported red/white patch, lump, burning sensation, pain, previous oral
  cancer, and **any lesion the health worker saw** → MODERATE.
* **LOW** is only allowed when every examined site was analysed with a reliable mucosa outline
  (segmentation `ok`) and every lesion (including every lesion probe reading and lesions at
  unimaged sites) could be checked; otherwise the result is INCONCLUSIVE. A site lost to a
  software error is never sent for a retake; the worker is told to examine it clinically.
* When red flags or habits put the overall level above every finding's own level, the
  explanation and the report say why; each finding keeps its own level.

| Level | Action |
|---|---|
| LOW | Routine re-screening, tobacco and areca-nut cessation counselling. "This does not rule out disease." |
| MODERATE | Remove irritants, re-examine in 2 weeks; refer if the lesion persists. |
| HIGH | Urgent referral for specialist examination and biopsy. |
| INCONCLUSIVE | Retake now; if a lesion is visible or retake impossible, refer for clinical examination. |

### 5.4 Pluggable models
| Interface | Phase 1 implementation | Later |
|---|---|---|
| `Segmenter` | classical HSV/Lab rules + hole filling (up to 35% of the frame) + AF cross-check for teeth (`refine_masks_with_af`) | U-Net (ONNX) |
| `LesionClassifier` | ONNX runner; returns `model_unavailable` when no weights | MobileNetV3/EfficientNet from `ml/` |
| `SpectralClassifier` | `rules_v2` on lesion/reference quotient deltas, status `placeholder` | logistic regression / small MLP |
| fusion | YAML rules | learned fusion behind the same output |

## 6. Backend

FastAPI + Pydantic v2 + SQLAlchemy 2 + Alembic. SQLite in dev, PostgreSQL in prod.

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/v1/health` | liveness |
| GET | `/api/v1/version` | software, pipeline, schema, config and model versions |
| GET | `/api/v1/schema/screening-result` | the JSON Schema |
| GET | `/api/v1/sites` | site catalogue + site-map geometry |
| POST | `/api/v1/screenings` | multipart upload: `manifest` JSON + image/spectrum files → analysis → result |
| POST | `/api/v1/demo/screenings/{low\|moderate\|high}` | run a SYNTHETIC demo patient |
| GET | `/api/v1/screenings` | list summaries |
| GET | `/api/v1/screenings/{id}` | latest result + version list |
| GET | `/api/v1/screenings/{id}/results/{version}` | a specific result version |
| POST | `/api/v1/screenings/{id}/reanalyze` | new result version with current pipeline/config |
| GET | `/api/v1/screenings/{id}/artifacts/{key}` | stored artifact (PNG/JSON) |
| GET | `/api/v1/screenings/{id}/report.pdf` | clinician report (versioned, stored) |
| GET | `/api/v1/screenings/{id}/report.html` | same report as HTML (preview/debug) |
| GET | `/api/v1/screenings/{id}/sitemap.svg` | oral site map coloured by risk |
| DELETE | `/api/v1/screenings/{id}` | erase a screening: rows, inputs, results, cached PDFs (and the patient record if no other screening uses it) |
| DELETE | `/api/v1/patients/{pseudonymous_id}` | erase a patient and all their screenings |

**Access.** Phase 1 has no user accounts (Phase 5). With `MUKHA_API_KEY` set (≥ 16 characters)
every `/api/v1` route except `/health` needs the `X-API-Key` header (401 otherwise); without it
the server logs a warning and `/version` reports `authentication: "none"` with an access notice.
Compose publishes both services on 127.0.0.1 only and sets `MUKHA_ALLOWED_HOSTS` (Host-header
check against DNS rebinding). The console's nginx adds the key to `/api` requests server-side, so
the key protects the API port only; keep the console port local or behind network access control.

**Limits.** Byte limits are enforced while the body streams (`MUKHA_MAX_REQUEST_MB`, default
200; manifest ≤ 1 MB); images over 50 MP are refused; uploads are decoded at most 2048 px under a
per-request decoded-pixel budget (`MUKHA_MAX_DECODED_MEGAPIXELS`, default 180); at most
`MUKHA_MAX_CONCURRENT_ANALYSES` (default 2) analyses run at once, others get 503 with
Retry-After. Stored JPEG/PNG uploads have EXIF/XMP/IPTC/comments removed (HEIC, DNG, TIFF, WebP
and BMP are stored as uploaded). Naive times are read in `MUKHA_DEFAULT_TIMEZONE` (Asia/Kolkata)
and stored as UTC. `MUKHA_BUILD_ID` (or git) gives the build recorded in every result.

Upload manifest (`manifest` form field, JSON):

```json
{
  "patient": {"name": "optional", "external_id": "optional", "age_years": 52, "sex": "male"},
  "consent": {"consent_recorded": true, "method": "checkbox", "language": "en",
              "research_use_opt_in": false},
  "risk_factors": { "...": "RiskFactors" },
  "context": {"screening_site_name": "PHC Hoskote camp", "health_worker_id": "hw-17"},
  "data_provenance": "real",
  "sites": [
    {"site": "buccal_mucosa_left", "wl": "wl_bl", "af": "af_bl", "dark": "dark_bl",
     "visual_flag": {"lesion_seen": true, "appearance": ["white_patch"]},
     "diascopy": "not_done", "px_per_mm": null, "device_id": null},
    {"site": "buccal_mucosa_right", "role": "reference_only", "wl": "wl_br", "af": "af_br"}
  ],
  "spectral_file": "spectrum"
}
```

Each `wl`/`af`/`dark`/`spectral_file` value names a multipart file field. Consent must be
recorded with a method (`checkbox`, `signature` or `thumbprint`) or the request is rejected
before anything is stored; the server stamps `recorded_at` when the client did not. Captures are
marked synthetic when the provenance says so, the uploaded file name starts with `SYNTHETIC`, or
the file carries a SYNTHETIC marker.

Data model (Phase 1 subset): `patients` (pseudonymous id, age, sex) and `patient_identities`
(PII, separate table), `screenings`, `captures` (storage key, sha256, metadata, QC),
`analysis_results` (versioned JSON), `reports` (versioned PDFs), `calibration_records`,
`audit_events`. Files live in a storage backend (local filesystem in dev) under
`screenings/{id}/inputs/…`, `screenings/{id}/results/v{n}/…`, `screenings/{id}/reports/v{n}/…`.

## 7. Report engine

One `ScreeningResult` → Jinja2 HTML → WeasyPrint PDF. Charts (spectra) are matplotlib SVG; the
site map is SVG built from the shared geometry in `sites.py`; images are embedded as data URIs so
the PDF is self-contained. The clinician report (2–4 pages) follows §6.1 of the brief: header
with versions and calibration status, disclaimer box, minimised patient details and risk factors,
colour-coded risk badge with action and timeline, site map, per-lesion WL | AF | heatmap panels
with measurements, spectral plot, model status (placeholder heuristic, not validated) and
contributing factors, QC and limitations, clinician review block, referral details with a QR code
to the record (access control arrives in a later release). A cached PDF contains the patient's
name; the erase endpoints delete it. Synthetic results carry a diagonal
"SYNTHETIC DATA" watermark on every page.

## 8. Frontend (Phase 1 console → Phase 2 PWA)

React + Vite + TypeScript + Tailwind. Phase 1 ships an analysis console: run a synthetic demo
patient or upload WL + AF (+ spectrum), watch the analysis, inspect results (risk with reasons,
WL/AF wipe comparison, AF-loss heatmap, 3D relative-AF terrain, spectra, QC), and download the
PDF. Phase 2 adds the PWA shell, guided capture, i18n (EN/HI/KN), offline storage and ONNX
Runtime Web.

## 9. Phases

| Phase | Scope | Exit criteria |
|---|---|---|
| 1 | Scaffold, schema, QC, registration, segmentation, AF analysis, rule triage, synthetic data, clinician PDF, upload API + console | Upload WL + AF (+ spectrum CSV) → holistic PDF; pipeline coverage ≥ 80% |
| 2 | PWA, guided capture, site map, questionnaire, camera capture, QC feedback, patient report, i18n, offline | Guided flow works offline on Android Chrome |
| 3 | Firmware (Scout, Confirm), BLE adapters, synced capture, probe readings, calibration API, mock devices | Full flow with mock devices; firmware builds |
| 4 | Tele-review, follow-up + biopsy tracking, de-identified dataset export, ML scripts, ONNX, model cards | Signed review changes report; dataset export reproducible |
| 5 | Auth/RBAC, encryption at rest, audit, sync, dashboard, CI, docs, risk register | Security checklist + traceability complete |
