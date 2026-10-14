/**
 * Mock-mode smoke test: the whole console against SYNTHETIC fixtures, no backend.
 *
 *   npm run e2e:mock
 *
 * Screenshots land in e2e-screenshots/ for a quick visual check.
 */
import { expect, test, type Page } from '@playwright/test';

const SHOTS = 'e2e-screenshots';

async function noHorizontalScroll(page: Page) {
  const { sw, vw } = await page.evaluate(() => ({
    sw: document.documentElement.scrollWidth,
    vw: document.documentElement.clientWidth,
  }));
  expect(sw).toBeLessThanOrEqual(vw);
}

test('home → synthetic HIGH demo → results, viewer, 3D, re-analyse, PDF', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(e.message));

  await page.goto('/');
  await expect(page.getByTestId('prototype-banner')).toHaveText(
    'Research prototype — not for clinical use',
  );
  await expect(page.getByRole('heading', { level: 1 })).toContainText('Violet light in.');
  await expect(page.getByTestId('mock-pill')).toBeVisible();
  await page.waitForTimeout(800);
  await page.screenshot({ path: `${SHOTS}/01-home.png` });

  // Demo run with the animated pipeline tracker.
  await page.getByRole('button', { name: 'Run the High-risk demo' }).click();
  await expect(page.getByTestId('analysis-overlay')).toBeVisible();
  await expect(page.getByTestId('stage-quality')).toBeVisible();
  await page.screenshot({ path: `${SHOTS}/02-analysing.png` });
  await expect(page.getByRole('button', { name: 'View results' })).toBeVisible({ timeout: 10_000 });
  await page.getByRole('button', { name: 'View results' }).click();

  // Results.
  const badge = page.getByTestId('risk-badge');
  await expect(badge).toHaveAttribute('data-level', 'HIGH');
  await expect(page.getByTestId('synthetic-ribbon')).toContainText('SYNTHETIC DATA');
  await expect(page.getByTestId('review-chip')).toHaveText('Pending clinician review');
  await expect(page.getByTestId('overrides')).toContainText('the device cannot lower this');
  expect(await page.getByTestId('factor-row').count()).toBeGreaterThan(5);
  await expect(page.getByTestId('factor-total')).toContainText('11.25');
  await page.waitForTimeout(900);
  await page.screenshot({ path: `${SHOTS}/03-results.png`, fullPage: true });

  // Wipe slider is keyboard operable.
  const slider = page.getByRole('slider', { name: /Comparison position/ });
  await slider.focus();
  await page.keyboard.press('ArrowRight');
  await expect(slider).toHaveAttribute('aria-valuenow', '52');
  await page.keyboard.press('Home');
  await expect(slider).toHaveAttribute('aria-valuenow', '0');
  await page.keyboard.press('PageUp');
  await expect(slider).toHaveAttribute('aria-valuenow', '10');

  // Heatmap overlay + region hover card.
  await page.getByText('AF-loss heatmap').click();
  await expect(page.getByTestId('heatmap-layer')).toBeVisible();
  await page.getByTestId('region-r1').hover();
  await expect(page.getByRole('tooltip')).toContainText('Region r1');
  await page.locator('#viewer').screenshot({ path: `${SHOTS}/04-viewer.png` });

  // 3D terrain (lazy chunk). Software WebGL may be missing; then the honest fallback shows.
  await page.getByRole('tab', { name: /3D glow map/ }).click();
  await expect(
    page
      .locator('#panel-terrain canvas')
      .or(page.getByText('3D view is not available on this device.')),
  ).toBeVisible({ timeout: 20_000 });
  await page.waitForTimeout(1200);
  await page.locator('#viewer').screenshot({ path: `${SHOTS}/05-terrain.png` });

  // Site map: choose another site.
  await page.getByRole('button', { name: /^Left buccal mucosa: Moderate risk/ }).click();
  await expect(page.getByTestId('site-viewer')).toHaveAttribute('data-site', 'buccal_mucosa_left');

  // Spectral panel draws lesion vs reference.
  await expect(page.getByTestId('spectral-chart')).toBeVisible();
  await expect(page.getByTestId('spectral-classifier')).toContainText(/placeholder/i);

  // Clinician PDF (mock serves a small placeholder PDF).
  const href = await page.getByTestId('report-link').getAttribute('href');
  const pdf = await page.request.get(new URL(href!, page.url()).toString());
  expect(pdf.ok()).toBe(true);
  expect((await pdf.body()).subarray(0, 4).toString()).toBe('%PDF');

  // Re-analyse creates v2; v1 stays available.
  await page.getByTestId('reanalyse').click();
  await expect(page.getByRole('button', { name: 'View results' })).toBeVisible({ timeout: 10_000 });
  await page.getByRole('button', { name: 'View results' }).click();
  const switcher = page.getByTestId('version-switcher');
  await expect(switcher.getByRole('button', { name: 'Version 2' })).toHaveAttribute(
    'aria-pressed',
    'true',
  );
  await switcher.getByRole('button', { name: 'Version 1' }).click();
  await expect(page.getByText(/You are viewing version 1/)).toBeVisible();

  expect(errors).toEqual([]);
});

test('upload validation explains what is missing', async ({ page }) => {
  await page.goto('/upload');
  await page.getByTestId('submit-upload').click();
  const summary = page.getByTestId('error-summary');
  await expect(summary).toContainText('Choose the site you photographed.');
  await expect(summary).toContainText('Tick consent. We cannot analyse photos without it.');
  await page.screenshot({ path: `${SHOTS}/06-upload-errors.png`, fullPage: true });
});

test('360 px phone: results and upload fit without horizontal scrolling', async ({ page }) => {
  await page.setViewportSize({ width: 360, height: 780 });
  await page.goto('/screenings/scr_demo_high_7f3a');
  await expect(page.getByTestId('risk-badge')).toBeVisible();
  await page.waitForTimeout(800);
  await noHorizontalScroll(page);
  await page.screenshot({ path: `${SHOTS}/07-mobile-results.png` });
  await page.goto('/upload');
  await expect(page.getByTestId('submit-upload')).toBeVisible();
  await noHorizontalScroll(page);
  await page.screenshot({ path: `${SHOTS}/08-mobile-upload.png` });
});

test('light theme and INCONCLUSIVE result', async ({ page }) => {
  await page.emulateMedia({ colorScheme: 'light' });
  await page.goto('/screenings/scr_demo_inconclusive_b4e0');
  await expect(page.getByTestId('risk-badge')).toHaveAttribute('data-level', 'INCONCLUSIVE');
  await expect(page.getByTestId('timeline')).toHaveText('Now');
  expect(await page.getByTestId('retake-prompt').count()).toBeGreaterThan(0);
  await page.waitForTimeout(800);
  await page.screenshot({ path: `${SHOTS}/09-light-inconclusive.png`, fullPage: true });
});
