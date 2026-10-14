/**
 * Mock-mode smoke test: the whole console against SYNTHETIC fixtures, no backend.
 *
 *   npm run e2e:mock
 *
 * Screenshots land in e2e-screenshots/ for a quick visual check.
 */
import { expect, test, type Page } from '@playwright/test';
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const SHOTS = 'e2e-screenshots';
const here = path.dirname(fileURLToPath(import.meta.url));
const MOCK_WL = readFileSync(path.resolve(here, '../src/mocks/assets/synthetic_tongue_r_wl.jpg'));
const MOCK_AF = readFileSync(path.resolve(here, '../src/mocks/assets/synthetic_tongue_r_af.jpg'));

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
  // Click pins the card (WCAG 1.4.13); Escape closes it.
  await page.getByTestId('region-r1').click();
  await expect(page.getByTestId('region-card')).toHaveAttribute('data-pinned', 'true');
  await page.keyboard.press('Escape');
  await expect(page.getByTestId('region-card')).toHaveCount(0);
  await page.getByTestId('region-r1').hover();
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
  await expect(page.getByTestId('timeline')).toHaveText('Before the person leaves');
  await expect(page.getByTestId('retake-callout')).toContainText('Retake photos for');
  expect(await page.getByTestId('retake-prompt').count()).toBeGreaterThan(0);
  await page.waitForTimeout(800);
  await page.screenshot({ path: `${SHOTS}/09-light-inconclusive.png`, fullPage: true });
});

test('360 × 640 phone: upload stays inside the screen after a file is chosen', async ({ page }) => {
  await page.setViewportSize({ width: 360, height: 640 });
  await page.goto('/upload');
  await page.getByRole('button', { name: /^Right lateral tongue/ }).click();
  await page.getByTestId('file-wl').setInputFiles({
    name: 'SYNTHETIC_lateral_tongue_right_wl.jpg',
    mimeType: 'image/jpeg',
    buffer: MOCK_WL,
  });
  await page.getByTestId('file-af').setInputFiles({
    name: 'SYNTHETIC_lateral_tongue_right_af.jpg',
    mimeType: 'image/jpeg',
    buffer: MOCK_AF,
  });
  await expect(page.getByTestId('filename-wl')).toContainText('tongue_right_wl.jpg');
  await noHorizontalScroll(page);
  await expect(page.getByTestId('prototype-banner')).toBeInViewport();
  await expect(page.getByTestId('submit-upload')).toBeInViewport();

  // Submit without consent: the error summary takes focus, the banner stays on screen.
  await page.getByTestId('submit-upload').click();
  await expect(page.getByTestId('error-summary')).toBeFocused();
  await noHorizontalScroll(page);
  await expect(page.getByTestId('prototype-banner')).toBeInViewport();
  await page.screenshot({ path: `${SHOTS}/10-mobile-upload-filled.png` });

  // Keyboard focus is never hidden under the sticky header or the sticky submit bar.
  await page.evaluate(() => window.scrollTo(0, 0));
  await page.locator('h1').first().click();
  const checked: string[] = [];
  for (let i = 0; i < 60; i++) {
    await page.keyboard.press('Tab');
    const r = await page.evaluate(() => {
      const el = document.activeElement as HTMLElement | null;
      if (!el || el === document.body) return null;
      // Things inside the fixed / sticky bars are never obscured by them.
      if (el.closest('header, nav, [data-testid="submit-bar"], [href="#main"]')) return null;
      const target = el.classList.contains('sr-only') ? (el.closest('label') ?? el) : el;
      const box = target.getBoundingClientRect();
      const bar = document.querySelector('[data-testid="submit-bar"]')!.getBoundingClientRect();
      return { top: box.top, bottom: box.bottom, barTop: bar.top, name: `${el.tagName}#${el.id}` };
    });
    if (!r) continue;
    checked.push(r.name);
    expect(r.top, `${r.name} hidden under the header`).toBeGreaterThanOrEqual(88);
    expect(r.bottom, `${r.name} hidden under the submit bar`).toBeLessThanOrEqual(r.barTop);
  }
  expect(checked.length).toBeGreaterThan(20);
});
