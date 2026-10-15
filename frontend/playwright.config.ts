import { defineConfig, devices } from '@playwright/test';
import { existsSync } from 'node:fs';

/**
 * Two modes:
 *  - integration (default, `npm run e2e`): starts the FastAPI backend on :8000 and a production
 *    build of the console with `vite preview` on :4173 (which proxies /api to the backend), then
 *    runs e2e/upload-to-report.spec.ts.
 *  - mock (`npm run e2e:mock`, E2E_MODE=mock): builds the console with VITE_MOCK=1 and serves it
 *    on :4174 — no backend needed — then runs e2e/mock-smoke.spec.ts.
 *
 * Browsers: @playwright/test 1.56.x matches Chromium build 1194 in /opt/pw-browsers
 * (PLAYWRIGHT_BROWSERS_PATH). Never run `playwright install` here. If the env var is missing we
 * point straight at the bundled Chrome binary.
 */
const MOCK = process.env.E2E_MODE === 'mock';
const CI = Boolean(process.env.CI);
const BUNDLED_CHROME = '/opt/pw-browsers/chromium-1194/chrome-linux/chrome';
const executablePath =
  !process.env.PLAYWRIGHT_BROWSERS_PATH && existsSync(BUNDLED_CHROME) ? BUNDLED_CHROME : undefined;

const E2E_DIR = '/tmp/mukha-e2e';

export default defineConfig({
  testDir: './e2e',
  testMatch: MOCK ? /mock-smoke\.spec\.ts$/ : /upload-to-report\.spec\.ts$/,
  outputDir: './test-results',
  fullyParallel: false,
  workers: 1,
  retries: CI ? 1 : 0,
  timeout: MOCK ? 60_000 : 240_000,
  expect: { timeout: 10_000 },
  reporter: [['list']],
  use: {
    baseURL: MOCK ? 'http://localhost:4174' : 'http://localhost:4173',
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
    colorScheme: 'dark',
    launchOptions: {
      ...(executablePath ? { executablePath } : {}),
      // Software WebGL so the 3D views render in headless runs.
      args: ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'],
    },
  },
  projects: [
    {
      name: MOCK ? 'mock' : 'integration',
      use: { ...devices['Desktop Chrome'], viewport: { width: 1440, height: 900 } },
    },
  ],
  webServer: MOCK
    ? [
        {
          command:
            'npm run build:mock && npx vite preview --outDir dist-mock --port 4174 --strictPort',
          url: 'http://localhost:4174',
          reuseExistingServer: !CI,
          timeout: 180_000,
          env: { VITE_MOCK: '1' },
        },
      ]
    : [
        {
          command: `mkdir -p ${E2E_DIR}/storage && ../.venv/bin/uvicorn app.main:app --port 8000`,
          cwd: '../backend',
          url: 'http://localhost:8000/api/v1/health',
          reuseExistingServer: true,
          timeout: 120_000,
          env: {
            MUKHA_DATABASE_URL: `sqlite:///${E2E_DIR}/mukha.db`,
            MUKHA_STORAGE_DIR: `${E2E_DIR}/storage`,
          },
        },
        {
          command: 'npm run build && npx vite preview --port 4173 --strictPort',
          url: 'http://localhost:4173',
          reuseExistingServer: true,
          timeout: 180_000,
          env: { VITE_MOCK: '0', MUKHA_API_TARGET: 'http://localhost:8000' },
        },
      ],
});
