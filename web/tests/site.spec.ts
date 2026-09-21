import { test, expect } from '@playwright/test';
import { readFileSync } from 'node:fs';
const base = process.env.NEXT_BASE_PATH ?? '';

test('article search, generated heading links, and mobile navigation', async ({ page }) => {
  await page.goto(`${base}/`);
  await page.getByRole('button', { name: 'Search', exact: false }).click();
  const dialog = page.getByRole('dialog', { name: 'Search documentation' });
  await expect(dialog).toBeVisible();
  await dialog.getByRole('searchbox').fill('API');
  await dialog.getByRole('link', { name: /API & TOML/ }).click();
  await expect(page.getByRole('heading', { name: 'API & TOML', exact: true })).toBeVisible();
  await page.setViewportSize({ width: 1500, height: 1000 });
  const contents = page.getByRole('navigation', { name: 'On this page' });
  await contents.getByRole('link', { name: 'Parameter studies' }).click();
  await expect(page).toHaveURL(/#parameter-studies$/);
  await expect(page.locator('#parameter-studies')).toBeVisible();
  // Both ends of the page must be exact: an intersection band cannot mark the
  // first heading when the page is scrolled fully up, nor the last when it is
  // scrolled fully down.
  const entries = contents.getByRole('link');
  await page.evaluate(() => window.scrollTo(0, 0));
  await expect(entries.first()).toHaveAttribute('aria-current', 'location');
  await page.evaluate(() => window.scrollTo(0, document.documentElement.scrollHeight));
  await expect(entries.last()).toHaveAttribute('aria-current', 'location');
  await page.setViewportSize({ width: 390, height: 844 });
  const menuButton = page.getByRole('button', { name: 'Open navigation' });
  await menuButton.click();
  const menu = page.getByRole('dialog', { name: 'Navigation', exact: true });
  await expect(menu).toBeVisible();
  await page.keyboard.press('Escape');
  await expect(menu).not.toBeVisible();
  await expect(menuButton).toBeFocused();
  await menuButton.click();
  await menu.getByRole('link', { name: 'Examples', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'A starting point for your next calculation' })).toBeVisible();
  await expect(menu).not.toBeVisible();
  await expect(page.locator('.wb-example-card img')).toHaveCount(7);
});

test('Cartesian object edits preserve fractional coordinates for an oblique non-unit cell', async ({ page }) => {
  const source = readFileSync('../examples/calculations/square-rods.toml', 'utf8')
    .replace('lattice = { type = "square", a = 1.0 }', 'lattice = { type = "custom", vectors = [[2.0, 0.0], [0.4, 1.6]] }')
    .replace('center = [0.0, 0.0]', 'center = [0.2, -0.3]');
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto(`${base}/workbench/`);
  await expect(page.getByRole('button', { name: 'Run', exact: true })).toBeEnabled();
  await page.locator('input[type=file]').setInputFiles({ name: 'coordinates.toml', mimeType: 'text/plain', buffer: Buffer.from(source) });
  await page.getByRole('button', { name: 'Apply', exact: true }).click();
  await page.getByRole('tab', { name: 'Geometry', exact: true }).click();
  await page.getByRole('button', { name: 'Cartesian', exact: true }).click();
  await expect(page.getByRole('textbox', { name: 'Center (cartesian) x', exact: true })).toHaveValue('0.68');
  await expect(page.getByRole('textbox', { name: 'Center (cartesian) y', exact: true })).toHaveValue('1.12');
  await page.getByRole('textbox', { name: 'Center (cartesian) x', exact: true }).fill('1.08');
  await page.keyboard.press('Tab');
  await expect(page.getByRole('button', { name: 'Fractional', exact: true })).toBeEnabled();
  await page.getByRole('button', { name: 'Fractional', exact: true }).click();
  await expect(page.getByRole('textbox', { name: 'Center (fractional) u', exact: true })).toHaveValue('0.4');
  await expect(page.getByRole('textbox', { name: 'Center (fractional) v', exact: true })).toHaveValue('0.7');
  await expect(page.getByRole('group', { name: 'Preview periods' }).getByRole('button', { name: '5 × 5' })).toHaveAttribute('aria-pressed', 'true');
});

test('pitch overlays independent data and calculates both polarizations', async ({ page }) => {
  await page.emulateMedia({ reducedMotion: 'reduce' });
  await page.goto(`${base}/pitch/`);
  const comparison = page.locator('.pitch-comparison');
  await comparison.scrollIntoViewIfNeeded();
  await expect(comparison.locator('[data-solver="MPB"] polyline')).toHaveCount(20);
  await expect(comparison.locator('[data-solver="Blaze"] polyline')).toHaveCount(20);
  const mpb = await comparison.locator('[data-solver="MPB"] polyline').nth(2).getAttribute('points');
  const blaze = await comparison.locator('[data-solver="Blaze"] polyline').nth(2).getAttribute('points');
  expect(mpb).not.toBe(blaze);
  await expect(page.locator('.pitch-speed [data-series="MPB"] rect')).toHaveCount(4);
  const demo = page.locator('.pitch-demo');
  await demo.scrollIntoViewIfNeeded();
  await demo.getByRole('button', { name: 'Calculate TM & TE' }).click();
  await expect(demo.getByRole('heading', { name: 'TM & TE bands', exact: true })).toBeVisible();
  await expect(demo.locator('.pitch-demo-status')).toContainText('completed', { timeout: 60_000 });
  await expect(demo.locator('[data-polarization="TM"] polyline')).toHaveCount(8);
  await expect(demo.locator('[data-polarization="TE"] polyline')).toHaveCount(8);
});
