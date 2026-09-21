import { test, expect, type Locator, type Page } from '@playwright/test';
import { readFileSync } from 'node:fs';
import { initSync, applyConfig, defaultToml, validateToml } from '../public/wasm-blaze/blaze2d_backend_wasm.js';
import { CRYSTALS, DEFAULT_SETTINGS, demoConfig, analyticGeometry, visibleFrequencies, type DemoSettings } from '../app/pitch/demo-model';

const base = process.env.NEXT_BASE_PATH ?? '';
test.beforeAll(() => initSync({ module: Uint8Array.from(readFileSync('public/wasm-blaze/blaze2d_backend_wasm_bg.wasm')) }));

function prepare(settings: DemoSettings) {
  const source = applyConfig(defaultToml(), JSON.stringify(demoConfig(settings)));
  const report = validateToml(source);
  expect(report.errors).toEqual([]);
  expect(report.ok).toBe(true);
  if (!report.resolved) throw new Error('Expected a resolved band calculation.');
  return { source, resolved: report.resolved };
}

test('native crystal paths use reciprocal distances and solve four extra bands', () => {
  for (const crystal of CRYSTALS) for (const bands of [4, 8, 12]) {
    const { resolved } = prepare({ ...DEFAULT_SETTINGS, crystal, bands });
    expect(resolved.solved_bands).toBe(bands + 4);
    expect(resolved.config.geometry.background_epsilon).toBe(7);
    expect(resolved.config.geometry.objects?.every(item => item.epsilon === 1 && item.radius === 0.2)).toBe(true);
    const triangular = crystal === 'triangular' || crystal === 'honeycomb';
    const rectangular = crystal === 'rectangular';
    expect(resolved.config.geometry.objects).toHaveLength(crystal === 'honeycomb' ? 2 : 1);
    expect(resolved.k_labels).toEqual(triangular ? ['Γ', 'M', 'K', 'Γ'] : rectangular ? ['Γ', 'X', 'S', 'Y', 'Γ'] : ['Γ', 'X', 'M', 'Γ']);
    expect(resolved.k_label_indices).toEqual(rectangular ? [0, 15, 30, 45, 60] : [0, 15, 30, 45]);
    expect(resolved.distances.at(-1)).toBeCloseTo(triangular ? 2 * Math.PI * (1 + 1 / Math.sqrt(3)) : rectangular ? 3 * Math.PI : Math.PI * (2 + Math.sqrt(2)), 8);
    expect(resolved.k_points_fractional[30]).toEqual(triangular ? [2 / 3, 1 / 3] : [0.5, 0.5]);
  }
  for (const ratio of [0.5, 1, 2]) {
    const { resolved } = prepare({ ...DEFAULT_SETTINGS, crystal: 'rectangular', ratio });
    resolved.lattice_vectors.flat().forEach((value, index) => expect(value).toBeCloseTo([1, 0, 0, 1 / ratio][index], 12));
    const expected = [0, Math.PI, Math.PI * (1 + ratio), Math.PI * (2 + ratio), 2 * Math.PI * (1 + ratio)];
    resolved.k_label_indices.forEach((sample, i) => expect(resolved.distances[sample]).toBeCloseTo(expected[i], 8));
  }
  // Padding must be discarded per k-point row, including the final row.
  expect(visibleFrequencies([1, 2, 3, 4, 90, 91, 92, 93, 5, 6, 7, 8, 94, 95, 96, 97], 8, 4)).toEqual([1, 2, 3, 4, 5, 6, 7, 8]);
});

test('analytic preview tiles the native crystal geometry through all four viewport edges', () => {
  for (const crystal of CRYSTALS) for (const ratio of [0.5, 2]) for (const radius of [0.05, 0.45]) {
    const settings = { ...DEFAULT_SETTINGS, crystal, ratio, radius };
    const { resolved } = prepare(settings);
    const { span, circles } = analyticGeometry(settings);
    const [a, b] = resolved.lattice_vectors;
    expect(span).toBeCloseTo(7 * Math.min(Math.hypot(...a), Math.hypot(...b)));
    // Independently tile the native solver's vectors and basis over a wider region.
    const expected: string[] = [];
    const extent = span / 2 + radius;
    const key = (x: number, y: number) => `${Math.round(x * 1e8)}:${Math.round(y * 1e8)}`;
    for (let u = -16; u <= 16; u++) for (let v = -16; v <= 16; v++) {
      for (const object of resolved.config.geometry.objects ?? []) {
        const [cu, cv] = object.center!;
        const x = (u + cu) * a[0] + (v + cv) * b[0], y = (u + cu) * a[1] + (v + cv) * b[1];
        if (Math.abs(x) <= extent + 1e-10 && Math.abs(y) <= extent + 1e-10) expected.push(key(x, y));
      }
    }
    expect(circles.map(circle => key(circle.x, circle.y)).sort()).toEqual(expected.sort());
  }
});

async function open(page: Page) {
  await page.emulateMedia({ reducedMotion: 'reduce' });
  await page.goto(`${base}/pitch/`);
  const demo = page.locator('.pitch-demo');
  await demo.scrollIntoViewIfNeeded();
  await expect(demo.getByRole('button', { name: 'Calculate TM & TE' })).toBeEnabled();
  return demo;
}
async function ready(demo: Locator) { await expect(demo).toHaveAttribute('data-ready', 'true'); }

test('model buttons and sliders preserve the layout and use the latest settings', async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 1100 });
  const demo = await open(page);
  await expect(demo.locator('select')).toHaveCount(0);
  await expect(demo.locator('.pitch-demo-preview')).toHaveText('');
  await expect(demo.getByRole('button', { name: 'Overlaid', exact: true })).toHaveAttribute('aria-pressed', 'true');
  await expect(demo.getByRole('button', { name: 'Separate', exact: true })).toBeVisible();
  await expect(demo.locator('.pitch-demo-plot')).toHaveCount(1);
  await expect(demo.getByRole('slider')).toHaveCount(5);
  await expect(demo.getByRole('slider', { name: 'Background ε' })).toHaveValue('7');
  await expect(demo.getByRole('slider', { name: 'Object ε' })).toHaveValue('1');
  await expect(demo.getByRole('slider', { name: 'Rectangular a/b' })).toHaveValue('0.5');
  const sizes = await demo.evaluate(element => ({ height: element.clientHeight, widths: [...element.querySelectorAll('.pitch-demo-slider')].map(item => item.getBoundingClientRect().width) }));
  for (const crystal of CRYSTALS) {
    await demo.getByRole('button', { name: crystal, exact: false }).click();
    await ready(demo);
    const labels = await demo.locator('[data-k-label]').evaluateAll(elements => elements.map(element => element.getAttribute('data-k-label')));
    expect(labels).toEqual(crystal === 'square' ? ['Γ', 'X', 'M', 'Γ'] : crystal === 'rectangular' ? ['Γ', 'X', 'S', 'Y', 'Γ'] : ['Γ', 'M', 'K', 'Γ']);
    expect(await demo.evaluate(element => ({ height: element.clientHeight, widths: [...element.querySelectorAll('.pitch-demo-slider')].map(item => item.getBoundingClientRect().width) }))).toEqual(sizes);
  }
  await demo.getByRole('slider', { name: 'Rectangular a/b' }).fill('2');
  await demo.getByRole('slider', { name: 'Background ε' }).fill('13');
  await demo.getByRole('slider', { name: 'Object ε' }).fill('3');
  await demo.getByRole('slider', { name: 'Radius r/a' }).fill('0.25');
  await demo.getByRole('slider', { name: 'Bands shown' }).fill('4');
  await ready(demo);
  await expect(demo).toHaveAttribute('data-computed-bands', '8');
  await expect(demo.locator('.pitch-demo-preview svg')).toHaveAttribute('data-world-span', '3.5');
  const positions = await demo.locator('[data-k-label]').evaluateAll(elements => elements.map(element => Number(element.getAttribute('data-distance'))));
  [0, Math.PI, 3 * Math.PI, 4 * Math.PI, 6 * Math.PI].forEach((distance, index) => expect(positions[index]).toBeCloseTo(distance, 8));
  await demo.getByRole('button', { name: 'Honeycomb', exact: true }).click(); await ready(demo);
  await expect(demo.getByRole('slider', { name: 'Rectangular a/b' })).toBeDisabled();
  await expect(demo.getByRole('slider', { name: 'Background ε' })).toHaveValue('13');
  await expect(demo.getByRole('slider', { name: 'Object ε' })).toHaveValue('3');
  await expect(demo.getByRole('slider', { name: 'Radius r/a' })).toHaveValue('0.25');
});

test('streamed TM and TE share a fixed path and hide the four computed guard bands', async ({ page }) => {
  const sources: string[] = [];
  await page.exposeFunction('capturePitchSource', (source: string) => sources.push(source));
  await page.addInitScript(() => {
    const post = Worker.prototype.postMessage;
    Worker.prototype.postMessage = function (message: { action?: string; source?: string }, ...args: []) {
      if (message.action === 'start') void (window as unknown as { capturePitchSource: (source: string) => Promise<void> }).capturePitchSource(message.source!);
      return post.call(this, message, ...args);
    };
  });
  const demo = await open(page);
  await demo.getByRole('slider', { name: 'Bands shown' }).fill('12'); await ready(demo);
  await expect(demo).toHaveAttribute('data-computed-bands', '16');
  const axis = await demo.locator('.pitch-demo-plot-stage svg').getAttribute('data-x-max');
  await demo.evaluate(element => {
    const snapshots: string[] = [];
    const sample = () => snapshots.push(JSON.stringify([...element.querySelectorAll('.pitch-demo-plot-stage svg')].map(svg => ({ max: svg.getAttribute('data-x-max'), labels: [...svg.querySelectorAll('[data-k-label] text')].map(text => text.getAttribute('x')) }))));
    sample();
    const observer = new MutationObserver(sample);
    observer.observe(element.querySelector('.pitch-demo-charts')!, { subtree: true, attributes: true, childList: true });
    Object.assign(window, { pitchAxisSnapshots: snapshots, pitchAxisObserver: observer });
  });
  await demo.getByRole('button', { name: 'Calculate TM & TE' }).click();
  await expect(demo.locator('[data-polarization="TM"]')).toHaveAttribute('data-samples', /[1-9]\d*/, { timeout: 60_000 });
  await expect(demo.getByRole('slider', { name: 'Bands shown' })).toBeDisabled();
  await expect(demo.locator('.pitch-demo-status')).toContainText('completed', { timeout: 60_000 });
  const snapshots = await page.evaluate(() => {
    const state = window as unknown as { pitchAxisSnapshots: string[]; pitchAxisObserver: MutationObserver };
    state.pitchAxisObserver.disconnect(); return state.pitchAxisSnapshots;
  });
  expect(snapshots.length).toBeGreaterThan(3);
  expect(new Set(snapshots).size).toBe(1);
  expect(sources).toHaveLength(1);
  expect(validateToml(sources[0]).resolved?.solved_bands).toBe(16);
  for (const polarization of ['TM', 'TE']) {
    const curves = demo.locator(`[data-polarization="${polarization}"] polyline`);
    await expect(curves).toHaveCount(12);
    expect((await curves.first().getAttribute('points'))!.split(' ')).toHaveLength(46);
    expect(await curves.last().getAttribute('points')).not.toMatch(/NaN|Infinity/);
  }
  await demo.getByRole('button', { name: 'Separate', exact: true }).click();
  await expect(demo.locator('.pitch-demo-plot')).toHaveCount(2);
  for (const svg of await demo.locator('.pitch-demo-plot-stage svg').all()) await expect(svg).toHaveAttribute('data-x-max', axis!);
  await demo.getByRole('button', { name: 'Overlaid', exact: true }).click();
  await expect(demo.locator('.pitch-demo-plot')).toHaveCount(1);
  await demo.getByRole('button', { name: 'Triangular', exact: true }).click(); await ready(demo);
  await expect(demo.locator('[data-polarization]')).toHaveCount(0);
});


test('analytic preview updates immediately while native validation is pending', async ({ page }) => {
  await page.addInitScript(() => {
    const post = Worker.prototype.postMessage;
    Worker.prototype.postMessage = function (message: { action?: string }, ...args: []) {
      if (message.action === 'preview') throw new Error('Pitch must not request a dielectric grid.');
      if (message.action === 'edit') { setTimeout(() => post.call(this, message, ...args), 750); return; }
      return post.call(this, message, ...args);
    };
  });
  const demo = await open(page);
  const preview = demo.locator('.pitch-demo-preview svg');
  await demo.getByRole('slider', { name: 'Radius r/a' }).fill('0.35');
  await expect(preview).toHaveAttribute('data-radius', '0.35');
  await expect(preview.locator('circle').first()).toHaveAttribute('r', '0.35');
  await expect(demo).toHaveAttribute('data-ready', 'false');
  await demo.getByRole('slider', { name: 'Background ε' }).fill('13');
  await expect(preview.locator('rect')).toHaveAttribute('fill', 'rgb(89, 182, 255)');
  await expect(demo).toHaveAttribute('data-ready', 'false');
  await ready(demo);
  await expect(demo.getByRole('button', { name: 'Calculate TM & TE' })).toBeEnabled();
});
