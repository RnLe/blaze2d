import { test, expect } from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';
import { moireGeometry, rotate, type Lattice, type Point } from '../lib/moire';

const base = process.env.NEXT_BASE_PATH ?? '';

test('moire vectors are dual to reciprocal beat vectors and respect lattice symmetry', () => {
  const dot = (a: Point, b: Point) => a[0] * b[0] + a[1] * b[1];
  for (const lattice of ['triangular', 'square'] as Lattice[]) {
    const symmetry = lattice === 'triangular' ? 60 : 90;
    for (const angle of [0.1, 2, 8, symmetry / 2, symmetry - 2]) {
      const { basis: [a, b], vectors, period, reduced } = moireGeometry(lattice, angle);
      const determinant = a[0] * b[1] - a[1] * b[0];
      const reciprocal: Point[] = [[b[1], -b[0]], [-a[1], a[0]]].map(([x, y]) => [2 * Math.PI * x / determinant, 2 * Math.PI * y / determinant]);
      for (let i = 0; i < 2; i++) {
        const rotated = rotate(reciprocal[i], reduced * Math.PI / 180);
        const beat: Point = [reciprocal[i][0] - rotated[0], reciprocal[i][1] - rotated[1]];
        for (let j = 0; j < 2; j++) expect(dot(beat, vectors![j])).toBeCloseTo(i === j ? 2 * Math.PI : 0, 8);
        expect(Math.hypot(...vectors![i])).toBeCloseTo(period, 8);
      }
    }
    expect(moireGeometry(lattice, 0).vectors).toBeNull();
    expect(moireGeometry(lattice, symmetry).period).toBe(Infinity);
    expect(moireGeometry(lattice, 2).period).toBeCloseTo(28.64934425, 6);
    expect(moireGeometry(lattice, symmetry - 2).period).toBeCloseTo(28.64934425, 6);
  }
});

test('bilayer explorer responds to angle, lattice, symmetry endpoints and zoom', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.goto(`${base}/thesis/`);
  const explorer = page.getByRole('region', { name: 'Interactive bilayer explorer' });
  const slider = page.getByRole('slider', { name: 'Twist angle' });
  await expect(slider).toHaveAttribute('max', '60');
  await expect(explorer.locator('[data-moire-vector]')).toHaveCount(2);
  await expect(explorer.locator('[data-moire-period]')).toContainText('7.17');
  await slider.fill('2');
  await expect(explorer.locator('[data-moire-period]')).toContainText('28.65');
  await slider.focus();
  await page.keyboard.press('Home');
  await expect(explorer.locator('[data-moire-vector]')).toHaveCount(0);
  await expect(slider).toHaveAttribute('aria-valuetext', /Layers aligned/);
  await expect(explorer.locator('[data-moire-period]')).toContainText('∞');
  await page.keyboard.press('End');
  await expect(slider).toHaveValue('60');
  await expect(explorer.locator('[data-moire-vector]')).toHaveCount(0);
  await page.getByRole('radio', { name: 'Square', exact: true }).check();
  await expect(slider).toHaveAttribute('max', '90');
  await slider.fill('45');
  await expect(explorer.locator('[data-moire-period]')).toContainText('1.31');
  await slider.fill('90');
  await expect(explorer.locator('[data-moire-vector]')).toHaveCount(0);
  await page.getByRole('radio', { name: 'Triangular', exact: true }).check();
  await expect(slider).toHaveValue('60');
  await slider.fill('8');
  await slider.focus();
  await page.keyboard.press('ArrowUp');
  await expect(slider).toHaveValue('8.1');
  await page.keyboard.press('ArrowDown');
  await expect(slider).toHaveValue('8');
  await explorer.getByRole('button', { name: '64a Wider' }).click();
  await expect(explorer.getByRole('button', { name: '64a Wider' })).toHaveAttribute('aria-pressed', 'true');
  await expect.poll(() => explorer.locator('canvas').evaluate((element: HTMLCanvasElement) => element.width)).toBeGreaterThan(0);
  await expect(explorer.getByRole('img', { name: /primitive cell/ })).toHaveCount(2);
  expect(errors).toEqual([]);
});

test('explorer expands and centers by button, then collapses on scroll without reflow', async ({ page }) => {
  for (const [width, height] of [[1920, 1080], [1280, 720], [390, 844], [320, 700], [320, 568], [844, 390]]) {
    await page.setViewportSize({ width, height });
    await page.goto(`${base}/thesis/`);
    const slot = page.locator('.bilayer-slot');
    const explorer = page.getByRole('region', { name: 'Interactive bilayer explorer' });
    await expect(slot).toHaveAttribute('data-expanded', 'false');
    const positionBefore = await slot.evaluate(e => e.getBoundingClientRect().top + window.scrollY);
    await slot.evaluate(e => e.scrollIntoView({ block: 'start' }));
    await expect(slot).toHaveAttribute('data-expanded', 'false');
    await expect(page.locator('.site-sidebar')).toHaveCSS('opacity', '1');
    await explorer.getByRole('button', { name: 'Expand explorer' }).click();
    await expect(slot).toHaveAttribute('data-expanded', 'true');
    await expect(explorer.getByRole('button', { name: 'Collapse explorer' })).toHaveAttribute('aria-expanded', 'true');
    await expect.poll(() => explorer.evaluate(e => Math.round(e.getBoundingClientRect().width))).toBe(width - (width <= 640 ? 16 : 32));
    await expect.poll(() => explorer.evaluate(e => {
      const box = e.getBoundingClientRect();
      const headerHeight = document.querySelector('.site-header')!.getBoundingClientRect().height;
      return Math.abs((box.top - headerHeight) - (window.innerHeight - box.bottom));
    })).toBeLessThanOrEqual(2);
    const box = (await explorer.boundingBox())!;
    expect(box.height).toBeLessThanOrEqual(height * 0.9 + 1);
    expect(box.y + box.height).toBeLessThanOrEqual(height);
    expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBe(width);
    expect(await slot.evaluate(e => e.getBoundingClientRect().top + window.scrollY)).toBeCloseTo(positionBefore, 0);
    for (const selector of ['.site-sidebar', '.article-toc']) {
      expect(await page.locator(selector).evaluate((e: HTMLElement) => e.inert)).toBe(true);
      await expect(page.locator(selector)).toHaveCSS('opacity', '0');
    }
    const slider = explorer.getByRole('slider', { name: 'Twist angle' });
    await expect(slider).toHaveAttribute('aria-orientation', 'vertical');
    const sliderBox = (await slider.boundingBox())!;
    expect(sliderBox.height).toBeGreaterThan(sliderBox.width);
    expect(await explorer.locator('.bilayer-angle').evaluate(e => e.scrollHeight <= e.clientHeight + 1)).toBe(true);
    expect(await explorer.locator('.bilayer-controls').evaluate(e => e.scrollHeight <= e.clientHeight + 1)).toBe(true);
    const canvasBox = (await explorer.locator('canvas').boundingBox())!;
    expect(sliderBox.x + sliderBox.width).toBeLessThan(canvasBox.x);
    await explorer.getByRole('button', { name: 'Collapse explorer' }).click();
    await expect(slot).toHaveAttribute('data-expanded', 'false');
    await expect(page.locator('.site-sidebar')).toHaveCSS('opacity', '1');
    await explorer.getByRole('button', { name: 'Expand explorer' }).click();
    await expect(slot).toHaveAttribute('data-expanded', 'true');
    // Leaving below and returning must not trigger another automatic expansion.
    await slot.evaluate(e => window.scrollTo(0, window.scrollY + e.getBoundingClientRect().bottom));
    await expect(slot).toHaveAttribute('data-expanded', 'false');
    await slot.evaluate(e => e.scrollIntoView({ block: 'start' }));
    await expect(slot).toHaveAttribute('data-expanded', 'false');
    await page.evaluate(() => window.scrollTo(0, 0));
    await expect(slot).toHaveAttribute('data-expanded', 'false');
    for (const selector of ['.site-sidebar', '.article-toc']) {
      expect(await page.locator(selector).evaluate((e: HTMLElement) => e.inert)).toBe(false);
      await expect(page.locator(selector)).toHaveCSS('opacity', '1');
    }
  }
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.emulateMedia({ reducedMotion: 'reduce' });
  await page.locator('.bilayer-slot').evaluate(e => e.scrollIntoView({ block: 'start' }));
  await page.getByRole('button', { name: 'Expand explorer' }).click();
  await expect(page.locator('.docs-shell')).toHaveClass(/bilayer-focused/);
  expect(await page.locator('.bilayer-explorer').evaluate(e => parseFloat(getComputedStyle(e).transitionDuration))).toBeLessThan(0.01);
  // Navigate while the explorer is active to exercise cleanup in the shared shell.
  await page.locator('.site-brand').click();
  await expect(page).toHaveURL(new RegExp(`${base}/$`));
  await expect(page.locator('.docs-shell')).not.toHaveClass(/bilayer-focused/);
  expect(await page.locator('.site-sidebar').evaluate((e: HTMLElement) => e.inert)).toBe(false);
});

test('both complete Hamiltonians open, zoom, trap focus and return focus on dismissal', async ({ page }) => {
  await page.setViewportSize({ width: 1500, height: 1000 });
  await page.goto(`${base}/thesis/`);
  await expect(page.locator('img[src*="hamiltonian"]')).toHaveCount(0);
  for (const polarization of ['TE', 'TM']) {
    const trigger = page.getByRole('button', { name: new RegExp(`Full ${polarization} Hamiltonian`) });
    await trigger.click();
    const dialog = page.getByRole('dialog', { name: `Full ${polarization} effective Hamiltonian` });
    await expect(dialog).toBeVisible();
    await expect(dialog.getByRole('button', { name: 'Close figure' })).toBeFocused();
    await expect(dialog.locator('img')).toHaveAttribute('src', `${base}/figures/thesis/second-edition/${polarization.toLowerCase()}-hamiltonian.svg`);
    await expect.poll(() => dialog.locator('img').evaluate((image: HTMLImageElement) => image.naturalWidth)).toBeGreaterThan(1000);
    await dialog.getByRole('button', { name: 'Zoom in' }).click();
    await expect(dialog.locator('output')).toHaveText('150%');
    await expect.poll(() => dialog.locator('.thesis-dialog-viewport').evaluate(e => e.scrollWidth > e.clientWidth)).toBe(true);
    await dialog.getByRole('button', { name: 'Fit to view' }).click();
    await expect(dialog.locator('output')).toHaveText('Fit');
    for (let n = 0; n < 9; n++) {
      await page.keyboard.press('Tab');
      expect(await dialog.evaluate(e => e.contains(document.activeElement))).toBe(true);
    }
    await page.keyboard.press('Escape');
    await expect(dialog).not.toBeVisible();
    await expect(trigger).toBeFocused();
    expect(await page.evaluate(() => document.body.style.overflow)).not.toBe('hidden');
  }
  await page.getByRole('button', { name: /Full TE Hamiltonian/ }).click();
  await page.mouse.click(2, 2);
  await expect(page.locator('dialog[open]')).toHaveCount(0);
});

test('thesis figures, equations and mobile controls are accessible and fit the page', async ({ page, request }) => {
  await page.goto(`${base}/thesis/`);
  await expect(page.locator('.katex-error')).toHaveCount(0);
  expect(await page.locator('article').innerText()).not.toContain('\u2014');
  for (const image of await page.locator('article img').all()) {
    const src = await image.getAttribute('src');
    const response = await request.get(src!);
    expect(response.ok(), src!).toBe(true);
  }
  const pdf = await request.head(`${base}/reports/masters-thesis-second-edition.pdf`);
  expect(pdf.ok()).toBe(true);
  expect((await new AxeBuilder({ page }).include('.thesis-page').analyze()).violations).toEqual([]);
  for (const width of [390, 320]) {
    await page.setViewportSize({ width, height: 844 });
    expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(width);
    await page.getByRole('button', { name: /Full TM Hamiltonian/ }).click();
    await expect(page.getByRole('button', { name: 'Close figure' })).toBeInViewport();
    expect((await new AxeBuilder({ page }).include('.thesis-dialog[open]').analyze()).violations).toEqual([]);
    await page.getByRole('button', { name: 'Close figure' }).click();
  }
});
