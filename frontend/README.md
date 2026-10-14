# MukhaSpectra — Analysis console (Phase 1)

> **Research prototype — not for clinical use.** Screening aid — not a diagnosis. The console shows
> risk levels, suspicious findings and the reasons behind them; final diagnosis requires
> examination by a clinician and histopathology.

React 19 + Vite + TypeScript (strict) + Tailwind CSS v4 + react-router + TanStack Query +
react-i18next + motion + three / react-three-fiber / drei (lazy). It talks to the FastAPI backend
(`/api/v1`, see `docs/architecture.md` §6) or, in **mock mode**, to an in-memory API seeded with
SYNTHETIC fixtures.

## Run

```bash
cd frontend
npm ci

# Live: needs the backend on http://localhost:8000 (vite proxies /api → :8000)
npm run dev                 # http://localhost:5173

# Mock mode: no backend, SYNTHETIC fixtures, ~1.5 s fake analysis latency
VITE_MOCK=1 npm run dev     # or: npm run dev:mock

# Production build + preview (preview also proxies /api → :8000)
npm run build && npm run preview          # http://localhost:4173
npm run build:mock                        # mock build in dist-mock/
```

Point the proxy elsewhere with `MUKHA_API_TARGET=http://host:port npm run dev`.

## Scripts

| Script                          | What it does                                                                                                                |
| ------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| `dev` / `dev:mock`              | Vite dev server (live / mock)                                                                                               |
| `build` / `build:mock`          | `tsc -b` + production build (`dist/` / `dist-mock/`)                                                                        |
| `preview`                       | Serve `dist/` on :4173 with the `/api` proxy                                                                                |
| `test`                          | Vitest unit/component tests (jsdom)                                                                                         |
| `lint` / `format` / `typecheck` | ESLint (flat config) / Prettier / `tsc -b --noEmit`                                                                         |
| `gen:types`                     | Regenerate `src/api/types.gen.ts` from `pipeline/mukha_pipeline/schema/screening_result.v1.schema.json` (commit the result) |
| `mock:assets`                   | Regenerate the SYNTHETIC mock images/grids/PDF (`scripts/make_mock_assets.py`, uses `../.venv/bin/python`)                  |
| `e2e`                           | Integration e2e: backend + live build → upload → result → PDF                                                               |
| `e2e:mock`                      | Mock-mode smoke e2e (no backend); screenshots in `e2e-screenshots/`                                                         |

## Tests

- **Unit** (`npm test`): API client (mock + real URL building, multipart manifest, error
  mapping, envelope normalisers), risk badge per level, factor bars sum to the score, site map
  (14 sites, spatial keyboard navigation), wipe slider keyboard control, upload validation
  (consent required, WL + AF required), prototype banner on every route, SYNTHETIC ribbon,
  results honesty (model statuses, retake prompts, pending review), fixtures validated against
  the JSON Schema (ajv, draft 2020-12), and the **positioning scan**: `en.json` plus every
  `.ts/.tsx` source are checked with the regexes ported from
  `pipeline/mukha_pipeline/positioning.py` (disclaimer phrases allowed) and `en.json` must make
  no accuracy/sensitivity/specificity claims.
- **E2E** — Playwright `@playwright/test` is pinned to **1.56.1**, whose Chromium build (1194)
  is already in `/opt/pw-browsers`. Do **not** run `playwright install`.

  ```bash
  export PLAYWRIGHT_BROWSERS_PATH=/opt/pw-browsers   # if unset, the config points at the bundled chrome
  npm run e2e:mock      # builds with VITE_MOCK=1, serves :4174, runs e2e/mock-smoke.spec.ts
  npm run e2e           # integration, runs e2e/upload-to-report.spec.ts
  ```

  The integration run starts `../.venv/bin/uvicorn app.main:app --port 8000` (cwd `../backend`,
  `MUKHA_DATABASE_URL=sqlite:////tmp/mukha-e2e/mukha.db`, `MUKHA_STORAGE_DIR=/tmp/mukha-e2e/storage`)
  and `vite preview` on :4173, both with `reuseExistingServer`. It uploads
  `../sample_data/synthetic/high/SYNTHETIC_lateral_tongue_right_{wl,af}.jpg` plus the left-side
  reference pair and `spectrum.csv`, ticks consent, waits for the risk badge, then fetches the
  report link and checks `application/pdf`, a `%PDF` header and size > 20 kB.

## Layout

```
src/
  api/          client.ts (live + lazy mock), normalize.ts (defensive envelopes), queries.ts
                (TanStack Query hooks), types.gen.ts (GENERATED), types.ts (envelopes, manifest)
  mocks/        fixtures.ts (HIGH / MODERATE / LOW / INCONCLUSIVE, all SYNTHETIC),
                mockClient.ts (in-memory API), assets/ (generated SYNTHETIC images + grids + PDF)
  i18n/         en.json (every user-facing string), index.ts; hi/kn arrive in Phase 2
  components/
    layout/     AppShell, PrototypeBanner, SyntheticRibbon, Logo
    home/       Hero (lazy r3f scene + static fallback), ScreeningList
    upload/     DropZone, uploadForm.ts (state, validation, manifest builder)
    analysis/   AnalysisOverlay (pipeline tracker with scan line + real stage timings)
    results/    RiskHeader, RiskBadge, FactorBars, SiteViewer, LesionList, SpectralPanel,
                QcPanel, VersionsFooter, ResultActions
    viz/        SiteMap, WipeSlider, RegionMarkers
  three/        HeroScene.tsx, Terrain3D.tsx (lazy chunks only)
  lib/          positioning guard, risk palette, formatting, colour map, theme, stages, …
  pages/        Home, Upload, Results, Screenings, NotFound (all but Home are lazy)
e2e/            upload-to-report.spec.ts (integration), mock-smoke.spec.ts
scripts/        gen-types.mjs, make_mock_assets.py
```

## Product rules baked into the UI

- Non-dismissable banner **"Research prototype — not for clinical use"** on every screen.
- No "diagnosis"/"malignant"/"cancer detected"/"no cancer"/"cancer-free" anywhere except the
  mandated disclaimer, which comes from `result.notices.disclaimer`. Enforced by a test.
- No accuracy/sensitivity/specificity numbers. Model statuses are shown as they are
  (white-light classifier not available — not used in triage; spectral classifier placeholder
  heuristic; rules baseline heuristic, unvalidated). The spectral "suspicion score" is labelled
  "not a probability".
- A score is never shown without its reasons: "Why this level" is a waterfall of every
  contributing factor (source icon, points, detail) adding up to the score, with the moderate
  and high thresholds marked; clinical red-flag floors are called out as "the device cannot lower
  this". Every result carries "Pending clinician review".
- Synthetic results get a "SYNTHETIC DATA" notice strip and corner ribbon on every view; demo
  cards and list rows carry a SYNTHETIC tag; uploads whose file names contain "SYNTHETIC" are
  recorded with `data_provenance: synthetic`.
- Privacy: the console never collects names; it shows pseudonymous id, age and sex only.
- Re-analysis creates a new version; the version switcher keeps v1, v2… and an older version is
  clearly marked as superseded.

## Design notes — "Violet in, green out"

- Tokens live in `src/styles/index.css` (OKLCH, light + dark, `data-theme` set before first
  paint, follows `prefers-color-scheme` until the toggle is used). Deep-ink surfaces make images
  readable; 405 nm violet is the action colour, fluorescence green the "healthy glow" accent.
- Risk levels always pair colour with a glyph and a word: ascending bars (1/2/3 lit) for
  LOW/MODERATE/HIGH; a dashed ring with "?" for INCONCLUSIVE. Reference sites are dashed blue.
  Text tokens meet WCAG AA (≥ 4.5:1) in both themes; control boundaries ≥ 3:1.
- Motion is spring-based and short (150–400 ms) via `motion/react` (`LazyMotion` +
  `domAnimation`), and `MotionConfig reducedMotion="user"`; with reduced motion the hero uses the
  static illustration and the 3D terrain stops auto-rotating.
- The site map, wipe slider, tabs and version switcher are keyboard operable (roving tab stop
  with spatial arrow-key navigation on the map; WAI-ARIA slider with arrows / Shift / PageUp /
  PageDown / Home / End).
- three / r3f / drei only load with the hero scene or the 3D tab (separate chunk).

## API assumptions

`src/api/normalize.ts` accepts the shapes the backend serves today (`ScreeningCreated
{screening_id, result_version, result}`, `ScreeningDetail {latest, versions[{version, …}]}`,
`ScreeningSummary[]` with `site_count` / `latest_result_version`, `SitesResponse {viewBox:
"0 0 400 320", sites[]}`) and tolerates small variations. The site map falls back to a built-in
copy of `pipeline/mukha_pipeline/sites.py` when `/api/v1/sites` is unreachable. Artifact keys are
encoded segment by segment (`/artifacts/{key:path}`), and every artifact/report URL carries
`?version=` for the result version on screen.
