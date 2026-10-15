# MukhaSpectra

Point-of-care oral cancer **screening and triage** for community health workers and dental
screening camps in India. Built by SparkyAI Private Limited.

> **Research prototype — not for clinical use.** MukhaSpectra flags suspicious findings and
> assigns a risk level. It is a screening aid, not a diagnosis. Final diagnosis requires a
> clinician and histopathology. No model in this repository is trained or clinically validated,
> and no accuracy figure is claimed anywhere.

A phone clip (**Scout**) photographs each oral site twice: once in white light and once under
405 nm violet light. Healthy mucosa glows green under violet light, while dysplastic tissue
loses that glow (fluorescence visualisation loss, FVL). A contact probe (**Confirm**, AS7341)
measures reflectance, where more blood deepens the 540–580 nm haemoglobin dip. The software turns
these captures, plus a short risk questionnaire (gutka, khaini, bidi, areca nut, alcohol,
symptoms), into three outputs:

- **A risk level with its reasons:** LOW, MODERATE, HIGH or INCONCLUSIVE. Every point is
  itemised, and clinical red flags set floors the device can never lower.
- **A clinician referral PDF:** WL, AF and AF-loss heatmap panels, spectra, QC and a review
  block.
- **A versioned record** built to become biopsy-labelled training data later.

## 5-minute quickstart

```bash
docker compose up --build        # API on 127.0.0.1:8000, console on http://localhost:5173
```

1. Open http://localhost:5173.
2. Click **Run demo** on the *High-risk demo* card. Every demo patient is synthetic and labelled
   SYNTHETIC.
3. Look at the result: the risk level and why, the site map, the WL/AF comparison, the 3D
   glow-loss map, the probe spectrum and QC.
4. Click **Download PDF report**.

To upload your own captures, choose **New screening**, pick a site, add the white-light and
violet-light photos (plus an optional dark frame, the opposite side and a spectrum CSV), tick
consent and click **Analyse**. The files in `sample_data/synthetic/*/` work as inputs.

### Without Docker

```bash
make setup                       # Python 3.11 venv + editable pipeline/backend + npm ci
make dev-backend                 # http://127.0.0.1:8000/docs
make dev-frontend                # http://localhost:5173 (proxies /api)
```

Pipeline only, no server:

```bash
.venv/bin/python -m mukha_pipeline analyze --demo high --out out/high
.venv/bin/python -m mukha_pipeline analyze --manifest sample_data/synthetic/moderate/manifest.json --out out/mod
```

Each command writes `result.json` (a `ScreeningResult`) and every artifact.

## Hosted demo on Vercel

The console deploys to Vercel as a static site. By default it builds in **mock mode**: the whole
console runs on bundled synthetic patients with no backend, which is the right thing to put on
a public URL (no real data; every screen carries the prototype banner and the SYNTHETIC ribbon).

```bash
npx vercel login
npx vercel --prod          # from the repository root; prints the https://….vercel.app URL
```

Or, once this branch is on `main`, use the one-click button (it clones the default branch):

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/Spyai/MukhaSpectra&project-name=mukhaspectra&repository-name=MukhaSpectra)

Keep the project's *Root Directory* at the repository root; `vercel.json` builds `frontend/`.

To host the **live** console (real uploads and PDFs) instead, run the backend from
`backend/Dockerfile` on a container host (Fly.io, Railway, Render, a VM) and set these Vercel
build variables: `VITE_MOCK=0` and `VITE_API_BASE=https://<your-api-host>/api/v1`. On the backend
set `MUKHA_CORS_ORIGINS=["https://<your-app>.vercel.app"]`, `MUKHA_PUBLIC_BASE_URL` to the same
origin, and `MUKHA_API_KEY` (the console never holds the key: put a proxy such as the bundled
nginx config in front of the API, or wait for Phase 5 authentication before exposing uploads).
The backend itself cannot run on Vercel: WeasyPrint needs Pango, OpenCV/onnxruntime exceed the
function size limit, and an analysis runs for seconds per site.

## Repository

| Path | What |
|---|---|
| `pipeline/` | `mukha_pipeline`, the pure-Python analysis library: QC, calibration, WL↔AF registration, mucosa segmentation, AF analysis (FVL and confounders), spectral features, rule-based triage, synthetic data |
| `backend/` | FastAPI API, SQLAlchemy with Alembic (SQLite dev, PostgreSQL prod), storage, Jinja2 + WeasyPrint report engine |
| `frontend/` | React + Vite + TypeScript console. Phase 2 turns it into the offline PWA. |
| `sample_data/` | Three synthetic demo patients with upload manifests, plus slots for bench data |
| `docs/` | Architecture, module contracts, device and spectral protocol, risk register |
| `ml/`, `firmware/` | Model card template (Phase 1); training and firmware come in Phases 3–4 |

The contract between all parts is `ScreeningResult` v1:
`pipeline/mukha_pipeline/schema/models.py`, with the generated JSON Schema next to it.

## Tests and checks

```bash
make test        # pipeline + backend (pytest) and frontend (vitest)
make coverage    # pipeline coverage, gate 80%
make lint        # ruff, ruff format --check, eslint, tsc
make e2e         # Playwright: upload sample → analysis → PDF report, against the real backend
```

## Rules this code enforces

- **Positioning:** no output says "diagnosis", "malignant", "cancer detected", "no cancer" or
  similar. `mukha_pipeline.positioning` is the guard, and tests scan pipeline text, API output,
  report HTML/PDF and every UI string.
- **Honesty:**
  - Models carry `validated: false` by schema.
  - With no weights installed, the white-light classifier reports `model_unavailable`.
  - The spectral classifier is labelled a placeholder.
- **Synthetic data:** the label travels with the data. Pixels, file names, metadata, results,
  console and report pages all carry it.
- **Safety:**
  - LOW requires every examined site to be readable; otherwise the result is INCONCLUSIVE.
  - Red flags (non-healing ulcer, red patch, growth, neck lump, and others) set floors.
  - A strong loss of fluorescence is never discounted below MODERATE.
- **Traceability:** every result records software, pipeline, schema, model and config versions,
  plus the SHA-256 of each rules file.
- **Privacy:**
  - Consent is required before anything is stored.
  - Names live in a separate PII table, and EXIF location data is stripped from stored uploads.
  - Screenings and patients can be erased through the API.
  - Authentication and RBAC arrive in Phase 5. Until then, keep the API on localhost or set
    `MUKHA_API_KEY`. The console's nginx adds that key itself, so the key protects the API
    port only: keep the console port on localhost or behind network access control.

## Build phases

| Phase | Scope | Status |
|---|---|---|
| 1 | Core pipeline, `ScreeningResult` schema, QC, registration, segmentation, AF analysis, rule triage, synthetic data, clinician PDF, upload API and console | **this branch** |
| 2 | PWA, guided capture, site map, questionnaire, camera capture, patient report, EN/HI/KN, offline | next |
| 3 | Scout and Confirm firmware, BLE adapters, synced capture, calibration API, mock devices | |
| 4 | Clinician tele-review, follow-up and biopsy tracking, de-identified dataset export, ML training and ONNX export | |
| 5 | Auth/RBAC, encryption at rest, audit, sync, dashboard, hardening | |
