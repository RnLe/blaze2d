import { test, expect } from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';
import { readdirSync, readFileSync, existsSync } from 'node:fs';
import path from 'node:path';

const base = process.env.NEXT_BASE_PATH ?? '';
const routes = ['', 'installation', 'introduction', 'configuration', 'examples', 'thesis', 'blaze', 'paper', 'roadmap', 'potential', 'workbench-guide', 'pitch', 'workbench',
  ...JSON.parse(readFileSync('../examples/calculations/catalog.json', 'utf8')).map((item: {slug: string}) => 'examples/' + item.slug)];

for (const width of [320, 1920]) test(`public routes reflow at ${width}px`, { tag: width === 320 ? '@smoke' : [] }, async ({ page }) => {
  test.setTimeout(180_000);
  await page.setViewportSize({ width, height: width === 320 ? 740 : 1080 });
  const failures: string[] = [];
  page.on('pageerror', error => failures.push(`${page.url()}: ${error.message}`));
  for (const route of routes) {
    const response = await page.goto(`${base}/${route}${route ? '/' : ''}`);
    expect(response?.status(), route).toBe(200);
    if (route === 'examples' || route.startsWith('examples/')) await expect(page).toHaveURL(/\/workbench\/?\?view=examples/);
    if (route === 'workbench' || route === 'examples' || route.startsWith('examples/'))
      await expect(page.getByRole('button', { name: 'Run', exact: true })).toBeEnabled();
    await page.evaluate(() => document.fonts.ready);
    // Let route prefetches finish before the next full navigation. WebKit reports
    // aborted requests from a departing document as access-control errors.
    await page.waitForLoadState('networkidle');
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth - innerWidth);
    if (overflow > 1) failures.push(`${route || '/'}: ${overflow}px`);
  }
  expect(failures).toEqual([]);
});

test('desktop, mobile, landscape, zoom, and pixel density', async ({ browser }) => {
  test.setTimeout(180_000);
  for (const [width, height] of [[1280,720],[1920,1080],[2560,1440],[3840,2160],[320,740],[360,800],[390,844],[430,932],[768,1024],[844,390]]) {
    for (const deviceScaleFactor of [1,2]) {
      const context = await browser.newContext({ viewport: { width, height }, deviceScaleFactor });
      const page = await context.newPage(); await page.goto(`${base}/workbench/`);
      await expect(page.getByRole('button', { name: 'Run', exact: true })).toBeEnabled();
      await page.getByRole('tab', { name: 'Study', exact: true }).click();
      expect(await page.evaluate(() => document.documentElement.scrollWidth), `${width} DPR${deviceScaleFactor}`).toBeLessThanOrEqual(width + 1);
      if (width >= 640) await page.evaluate(() => { document.documentElement.style.zoom = '2'; });
      expect(await page.evaluate(() => document.documentElement.scrollWidth), `${width} CSS zoom 200%`).toBeLessThanOrEqual(width + 2);
      await context.close();
    }
  }
});

test('primary workflows have accessible controls and contrast', async ({ page }) => {
  test.setTimeout(120_000);
  for (const route of ['', 'installation', 'configuration', 'workbench']) {
    await page.goto(`${base}/${route}${route ? '/' : ''}`);
    if (route === 'workbench') await expect(page.getByRole('button', { name: 'Run', exact: true })).toBeEnabled();
    for (const tab of route === 'workbench' ? ['Geometry', 'Study', 'Results', 'Examples'] : ['']) {
      if (tab) await page.getByRole('tab', { name: tab, exact: true }).click();
      const report = await new AxeBuilder({ page }).withTags(['wcag2a','wcag2aa','wcag21aa','wcag22aa']).analyze();
      expect(report.violations.map(v => ({ id:v.id, impact:v.impact, nodes:v.nodes.map(n=>n.target) })), `${route} ${tab}`).toEqual([]);
    }
    if (route === 'workbench') {
      await page.getByRole('tab', { name: 'TOML', exact: true }).click();
      const editorReport = await new AxeBuilder({ page }).withTags(['wcag2a','wcag2aa']).analyze();
      expect(editorReport.violations.map(v => ({ id: v.id, nodes: v.nodes.map(n => n.target) }))).toEqual([]);
    }
  }
});

test('static export keeps relocation pages and excludes Architecture', { tag: '@smoke' }, () => {
  const root = path.resolve('out');
  expect(existsSync(path.join(root, 'architecture'))).toBe(false);
  const pages = readdirSync(root, { recursive: true }).filter(file => typeof file === 'string' && file.endsWith('.html')) as string[];
  const broken = new Set<string>();
  for (const file of pages) {
    const html = readFileSync(path.join(root,file),'utf8');
    for (const match of html.matchAll(/(?:href|src)="([^"#?]+)(?:[?#][^"]*)?"/g)) {
      const url = match[1]; if (!url.startsWith('/') || url.startsWith('//')) continue;
      expect(url.startsWith(base + '/'), `${file}: ${url}`).toBe(true);
      const target = path.join(root,decodeURIComponent(url.slice(base.length)));
      if (!existsSync(target) && !existsSync(path.join(target, 'index.html'))) broken.add(`${file}: ${url}`);
    }
  }
  expect([...broken]).toEqual([]);
  expect(readFileSync(path.join(root, 'potential/index.html'),'utf8')).toMatch(/name="robots" content="noindex/);
});
