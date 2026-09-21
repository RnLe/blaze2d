import { test, expect, type Page } from '@playwright/test';
import { readFileSync } from 'node:fs';
import { unzipSync, strFromU8 } from 'fflate';
import { editorSpan } from '../lib/compute/editor';
import { examples } from '../lib/examples/registry';
import { nearestBandPoint } from '../components/workbench/BandTooltip';

const base = process.env.NEXT_BASE_PATH ?? '';
const bands = readFileSync('../examples/calculations/square-rods.toml', 'utf8').replace('resolution = 32', 'resolution = [12, 16]').replace('intervals_per_segment = 15', 'intervals_per_segment = 2');
const operators = readFileSync('../examples/calculations/operator-point.toml', 'utf8').replace('[24, 32]', '[12, 16]');

async function open(page: Page) {
  await page.goto(`${base}/workbench/`);
  await expect(page.getByRole('button', { name: 'Run', exact: true })).toBeEnabled();
}
async function apply(page: Page, source: string, executable = true) {
  await page.locator('input[type=file]').setInputFiles({ name: 'fixture.toml', mimeType: 'text/plain', buffer: Buffer.from(source) });
  await expect(page.getByRole('button', { name: 'Apply', exact: true })).toBeEnabled();
  await page.getByRole('button', { name: 'Apply', exact: true }).click();
  if (executable) await expect(page.getByRole('button', { name: 'Run', exact: true })).toBeEnabled();
  await page.getByRole('tab', { name: 'Geometry', exact: true }).click();
}
async function run(page: Page) {
  await page.getByRole('button', { name: 'Run', exact: true }).click();
  await expect(page.locator('.wb-results').getByText(/completed · \d+ completed/)).toBeVisible({ timeout: 60_000 });
}
async function exported(page: Page, format: string) {
  const promise = page.waitForEvent('download');
  await page.getByRole('button', { name: `Export ${format.toUpperCase()}`, exact: true }).click();
  const download = await promise, path = await download.path();
  return readFileSync(path!);
}

test('UTF-8 diagnostic spans identify Unicode source accurately', () => {
  const source = '# Γ 🔬\nradius = invalid';
  const offset = Buffer.byteLength('# Γ 🔬\nradius = ');
  expect(editorSpan(source, [offset, offset + 7])).toEqual([source.indexOf('invalid'), source.length]);
});

test('invalid drafts survive reload and cannot overwrite the applied model', { tag: '@smoke' }, async ({ page }) => {
  await open(page); await apply(page, bands);
  const invalid = bands + '\n# Unicode Γ 🔬\n[unsupported]\nfield = true\n';
  await page.locator('input[type=file]').setInputFiles({ name: 'invalid.toml', mimeType: 'text/plain', buffer: Buffer.from(invalid) });
  await expect(page.getByRole('button', { name: 'Run', exact: true })).toBeDisabled();
  await expect(page.locator('.wb-validation')).toContainText('unknown field');
  await page.reload();
  await expect(page.getByRole('button', { name: 'Run', exact: true })).toBeDisabled();
  await expect(page.locator('.cm-content')).toContainText('[unsupported]');
  const download = page.waitForEvent('download'); await page.getByRole('button', { name: 'Download draft' }).click();
  expect(readFileSync((await (await download).path())!, 'utf8')).toBe(invalid);
  await page.getByRole('button', { name: 'Revert', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Run', exact: true })).toBeEnabled();
});

test('bands export complete arrays and preserve a historical snapshot', { tag: '@smoke' }, async ({ page }) => {
  await open(page); await apply(page, bands); await run(page);
  const first = JSON.parse((await exported(page, 'json')).toString());
  expect(first.results[0].arrays.frequencies.shape).toEqual([7, 8]);
  expect(first.results[0].arrays.k_points.data.slice(-2)).toEqual([0, 0]);
  const npz = unzipSync(await exported(page, 'npz'));
  const headerLength = new DataView(npz['manifest.npy'].buffer).getUint16(8, true);
  const manifest = JSON.parse(strFromU8(npz['manifest.npy'].subarray(10 + headerLength)));
  expect(manifest.results[0].arrays.frequencies.shape).toEqual([7, 8]);
  expect(Object.keys(npz)).toContain(manifest.results[0].arrays.frequencies.buffer + '.npy');
  await page.getByRole('tab', { name: 'Geometry', exact: true }).click();
  await apply(page, bands.replace('radius = 0.20', 'radius = 0.25'));
  await page.getByRole('tab', { name: 'Results', exact: true }).click();
  expect(JSON.parse((await exported(page, 'json')).toString()).config.geometry.objects[0].radius).toBe(.2);
  await page.reload(); await expect(page.getByRole('button', { name: 'Run', exact: true })).toBeEnabled();
  await page.getByRole('button', { name: 'History', exact: true }).click();
  await page.locator('.wb-history-open').first().click();
  await expect(page.getByRole('tab', { name: 'Results', exact: true })).toHaveAttribute('aria-selected', 'true');
  expect(JSON.parse((await exported(page, 'json')).toString()).results[0].arrays.frequencies.data).toEqual(first.results[0].arrays.frequencies.data);
});

test('cancel terminates a solve and permits an immediate replacement', async ({ page }) => {
  await page.addInitScript(() => {
    const NativeWorker = window.Worker;
    window.Worker = class extends NativeWorker {
      private previousRun = '';
      postMessage(message: unknown, transfer?: Transferable[] | StructuredSerializeOptions) {
        if (message && typeof message === 'object' && 'action' in message && message.action === 'start' && 'runId' in message) this.previousRun = String(message.runId);
        if (Array.isArray(transfer)) super.postMessage(message, transfer); else super.postMessage(message, transfer);
      }
      terminate() {
        super.terminate();
        const id = this.previousRun;
        if (id) setTimeout(() => this.onmessage?.call(this, new MessageEvent('message', { data: { runId: id, kind: 'event', event: { event: 'terminal', status: 'failed', completed: 0, failed: 1 } } })), 75);
      }
    };
  });
  await open(page); await apply(page, bands.replace('[12, 16]', '[128, 128]').replace('intervals_per_segment = 2', 'intervals_per_segment = 50'));
  await page.getByRole('button', { name: 'Run', exact: true }).click();
  await page.getByRole('button', { name: 'Cancel', exact: true }).click();
  await expect(page.locator('.wb-results')).toContainText('cancelled');
  await apply(page, bands); await run(page);
  const data = JSON.parse((await exported(page, 'json')).toString());
  expect(data.results).toHaveLength(1); expect(data.results[0].arrays.frequencies.shape).toEqual([7, 8]);
});

test('operator windows, registry stencils, and residual failures remain distinguishable', async ({ page }) => {
  await open(page); await apply(page, operators); await run(page);
  const point = JSON.parse((await exported(page, 'json')).toString()).results[0];
  expect(point.arrays.velocity_matrices.shape).toEqual([2, 2, 7]); expect(point.arrays.velocity_matrices.dtype).toBe('complex128');
  const source = operators + '\n[operators.registry]\nobject = "hole"\npoints = [[0.0, 0.0], [1.25, -0.25]]\nfd_step = 0.001\n\n[operators.k_stencil]\npoints_per_axis = 3\nhalf_width = 0.01\n';
  await apply(page, source); await run(page);
  const stencil = JSON.parse((await exported(page, 'json')).toString());
  expect(stencil.results).toHaveLength(2); expect(stencil.results[0].samples).toHaveLength(9);
  await apply(page, operators.replace('[operators]', '[operators]\nfail_on_residual = 1e-30'));
  await page.getByRole('button', { name: 'Run', exact: true }).click();
  await expect(page.locator('.wb-results')).toContainText('failed · 0 completed · 1 failed');
  const failure = JSON.parse((await exported(page, 'json')).toString());
  expect(failure.statistics.status).toBe('failed'); expect(failure.errors[0].partial_result.arrays.residuals.data).toHaveLength(7);
});

test('storage denial keeps results in memory and excessive memory blocks execution', async ({ page }) => {
  await page.addInitScript(() => { Object.defineProperty(window, 'indexedDB', { get() { throw new Error('Storage denied'); } }); });
  await open(page); await apply(page, bands); await run(page);
  await expect(page.locator('.wb-notice')).toContainText('History unavailable');
  expect(JSON.parse((await exported(page, 'ndjson')).toString()).results).toHaveLength(1);
  await apply(page, bands.replace('[12, 16]', '[4096, 4096]'), false);
  await expect(page.getByRole('button', { name: 'Run', exact: true })).toBeDisabled();
  await expect(page.locator('.wb-main').getByRole('alert')).toContainText('512 MiB');
});

test('mobile reflow and keyboard tabs preserve usable controls', async ({ page }) => {
  await page.setViewportSize({ width: 320, height: 740 }); await open(page);
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBe(320);
  await page.getByRole('tab', { name: 'Geometry', exact: true }).focus(); await page.keyboard.press('ArrowRight');
  await expect(page.getByRole('tab', { name: 'Study', exact: true })).toBeFocused();
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBe(320);
  await page.getByRole('button', { name: 'History', exact: true }).click();
  await expect(page.getByRole('dialog')).toBeVisible(); await page.keyboard.press('Escape');
  await expect(page.getByRole('button', { name: 'History', exact: true })).toBeFocused();
});

test('ordered sweeps preserve explicit sample indices and closing Gamma', async ({ page }) => {
  await open(page);
  await apply(page, bands + '\n[[sweeps]]\nname = "radius"\ntarget = "geometry.objects.rod.radius"\nlinspace = { start = 0.22, stop = 0.18, count = 3 }\n\n[[sweeps]]\nname = "pol"\ntarget = "polarization"\nvalues = ["TM", "TE"]\n');
  await run(page);
  const study = JSON.parse((await exported(page, 'json')).toString());
  expect(study.results.map((result: { metadata: {multi_index: number[]} }) => result.metadata.multi_index)).toEqual([[0,0],[0,1],[1,0],[1,1],[2,0],[2,1]]);
  expect(study.results[3].metadata.config.polarization).toBe('TE');
  await apply(page, operators + '\n[operators.k_stencil]\npoints_per_axis = 1\nhalf_width = 0.0\n');
  await run(page);
  expect(JSON.parse((await exported(page, 'json')).toString()).results[0].samples).toHaveLength(1);
});

test('structured fields retain invalid edits and block execution until corrected', async ({ page }) => {
  await open(page);
  const radius = page.getByRole('textbox', { name: 'Circle radius', exact: true });
  await radius.fill('bad'); await radius.press('Tab');
  await expect(radius).toHaveValue('bad');
  await expect(radius).toHaveAttribute('aria-invalid', 'true');
  await expect(page.getByRole('button', { name: 'Run', exact: true })).toBeDisabled();
  await radius.focus(); await radius.press('Escape');
  await expect(page.getByRole('button', { name: 'Run', exact: true })).toBeEnabled();
  await radius.fill('-1'); await radius.press('Enter');
  await expect(page.locator('.wb-error')).toContainText('radius');
  await expect(page.getByRole('button', { name: 'Run', exact: true })).toBeDisabled();
  await page.getByRole('button', { name: 'Revert to applied configuration' }).click();
  await expect(radius).toHaveValue('0.2');
  await expect(page.getByRole('button', { name: 'Run', exact: true })).toBeEnabled();
});

test('unavailable WASM fails visibly and can recover on reload', async ({ page }) => {
  await page.route('**/*.wasm', route => route.abort());
  await page.goto(`${base}/workbench/`);
  await expect(page.locator('.wb-error')).toBeVisible();
  await expect(page.getByRole('button', { name: 'Run', exact: true })).toBeDisabled();
  await page.unroute('**/*.wasm');
  await page.getByRole('button', { name: 'Reload', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Run', exact: true })).toBeEnabled();
});

test('quota exhaustion preserves the current result for export', async ({ page }) => {
  await page.addInitScript(() => {
    const put = IDBObjectStore.prototype.put;
    IDBObjectStore.prototype.put = function(...args) {
      if (this.name === 'records') throw new DOMException('Storage quota exhausted', 'QuotaExceededError');
      return put.apply(this, args);
    };
  });
  await open(page); await apply(page, bands); await run(page);
  await expect(page.locator('.wb-notice')).toContainText('History unavailable');
  expect(JSON.parse((await exported(page, 'json')).toString()).results).toHaveLength(1);
});

test('reload marks an unfinished calculation as interrupted', async ({ page }) => {
  await open(page); await apply(page, bands.replace('[12, 16]', '[128, 128]').replace('intervals_per_segment = 2', 'intervals_per_segment = 50'));
  await page.getByRole('button', { name: 'Run', exact: true }).click();
  await expect.poll(() => page.evaluate(() => new Promise<number>((resolve, reject) => {
    const request = indexedDB.open('blaze2d-workbench-v1');
    request.onsuccess = () => { const count = request.result.transaction('runs').objectStore('runs').count(); count.onsuccess = () => { resolve(count.result); request.result.close(); }; };
    request.onerror = () => reject(request.error);
  }))).toBe(1);
  await page.reload(); await expect(page.getByRole('button', { name: 'Run', exact: true })).toBeEnabled();
  await page.getByRole('button', { name: 'History', exact: true }).click();
  await expect(page.getByRole('dialog')).toContainText('interrupted');
});

test('all examples load their complete configuration in one workspace', async ({ page }) => {
  await page.goto(`${base}/workbench/?view=examples`);
  await expect(page.getByRole('button', { name: 'Run', exact: true })).toBeEnabled();
  await expect(page.locator('.wb-example-card')).toHaveCount(examples.length);
  for (const example of examples) {
    await page.getByRole('tab', { name: 'Examples', exact: true }).click();
    const back = page.getByRole('button', { name: 'All examples', exact: true });
    if (await back.isVisible()) await back.click();
    await page.locator('.wb-example-card').filter({ hasText: example.title }).click();
    await page.getByRole('button', { name: 'Load example', exact: true }).click();
    await expect(page.getByRole('tab', { name: 'Geometry', exact: true })).toHaveAttribute('aria-selected', 'true');
    await expect(page.getByRole('button', { name: 'Run', exact: true })).toBeEnabled();
    await page.getByRole('tab', { name: 'TOML', exact: true }).click();
    const promise = page.waitForEvent('download');
    await page.getByRole('button', { name: 'Download draft' }).click();
    expect(readFileSync((await (await promise).path())!, 'utf8')).toBe(example.source);
    expect(await page.locator('.workbench-shell').count()).toBe(1);
    expect(await page.locator('.workbench-shell select').count()).toBe(0);
  }
});

test('legacy example links open the inspectable library without a second workbench', { tag: '@smoke' }, async ({ page }) => {
  for (const slug of ['square-rods', 'first-band-diagram']) {
    await page.goto(`${base}/examples/${slug}/`);
    await expect(page).toHaveURL(/workbench\/?\?view=examples&inspect=square-rods/);
    await expect(page.getByRole('heading', { name: 'Square rods', exact: true })).toBeVisible();
    await expect(page.getByRole('button', { name: 'Load example' })).toBeEnabled();
    await expect(page.locator('.workbench-shell')).toHaveCount(1);
  }
});

test('split editors preserve view state and protect unfinished TOML', async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await open(page); await apply(page, bands); await run(page);
  await page.getByText('Arrays and dimensions', { exact: true }).click();
  const arrays = page.locator('.wb-results details').filter({ has: page.getByText('Arrays and dimensions', { exact: true }) });
  await page.getByRole('button', { name: 'Split view', exact: true }).click();
  await expect(page.getByRole('tabpanel', { name: 'Results', exact: true })).toBeVisible();
  await expect(page.getByRole('tabpanel', { name: 'Geometry', exact: true })).toBeVisible();
  const plots = await page.getByRole('tabpanel', { name: 'Results', exact: true }).boundingBox();
  const geometry = await page.getByRole('tabpanel', { name: 'Geometry', exact: true }).boundingBox();
  expect(geometry!.x).toBeGreaterThan(plots!.x + plots!.width);
  const separator = page.getByRole('separator', { name: 'Editor split' });
  await separator.focus(); await page.keyboard.press('ArrowRight');
  await expect(separator).toHaveAttribute('aria-valuenow', '55');
  const secondary = page.getByRole('tablist', { name: 'Split editor tabs' });
  await secondary.getByRole('tab', { name: 'TOML', exact: true }).click();
  const editor = page.getByRole('textbox', { name: 'Calculation TOML', exact: true });
  await editor.fill(bands + '\n[unsupported]\nfield = true\n');
  await expect(page.getByRole('button', { name: 'Run', exact: true })).toBeDisabled();
  await expect(arrays).toHaveAttribute('open', '');
  await page.getByRole('button', { name: 'Revert', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Run', exact: true })).toBeEnabled();
  await page.getByRole('button', { name: 'Close split editor' }).click();
  await expect(arrays).toHaveAttribute('open', '');
});

test('nearest band tooltip measures both axes and searches all polarizations', () => {
  const series = [
    { polarization: 'TM', distances: [0, 10, 20], frequencies: [0, 10, 100, 100, 20, 30], bands: 2 },
    { polarization: 'TE', distances: [0, 10, 20], frequencies: [50, 60, 30, 40, 40, 50], bands: 2 },
  ];
  const identity = (value: number) => value;
  // The nearest column (x=10) has no nearby band: the TM point at x=0 wins.
  expect(nearestBandPoint(series, { x: 9, y: 10 }, identity, identity)).toMatchObject({ sample: 0, band: 1, polarization: 'TM', frequency: 10 });
  expect(nearestBandPoint(series, { x: 11, y: 40 }, identity, identity)).toMatchObject({ sample: 1, band: 1, polarization: 'TE', frequency: 40 });
  expect(nearestBandPoint([], { x: 0, y: 0 }, identity, identity)).toBeUndefined();
});

test('instant band tooltips follow the actual point and job polarization without selection controls', async ({ page }) => {
  await open(page);
  await apply(page, bands + '\n[[sweeps]]\nname = "polarization"\ntarget = "polarization"\nvalues = ["TM", "TE"]\n');
  await run(page);
  const results = JSON.parse((await exported(page, 'json')).toString()).results;
  await expect(page.getByRole('spinbutton', { name: 'Readout band' })).toHaveCount(0);
  await expect(page.getByRole('slider', { name: 'Selected k-point' })).toHaveCount(0);
  for (const [job, polarization] of ['TM', 'TE'].entries()) {
    await page.getByRole('spinbutton', { name: 'Configuration sample', exact: true }).fill(String(job));
    for (const width of [390, 1920]) {
      await page.setViewportSize({ width, height: 1080 });
      const plot = page.locator('.wb-plot');
      await plot.scrollIntoViewIfNeeded();
      const curve = plot.locator('polyline').nth(6);
      const position = await curve.evaluate(element => {
        const [x, y] = element.getAttribute('points')!.split(' ')[3].split(',').map(Number);
        const point = new DOMPoint(x, y).matrixTransform((element as SVGPolylineElement).getScreenCTM()!);
        return { x: point.x, y: point.y };
      });
      await page.mouse.move(position.x, position.y);
      const tooltip = plot.getByRole('tooltip');
      await expect(tooltip).toContainText(`Band index 6 · ${polarization}`);
      await expect(tooltip).toContainText(`f = ${Number(results[job].arrays.frequencies.data[3 * 8 + 6]).toPrecision(6)}`);
      const marker = await tooltip.locator('circle').evaluate(element => {
        const point = new DOMPoint(Number(element.getAttribute('cx')), Number(element.getAttribute('cy'))).matrixTransform((element as SVGCircleElement).getScreenCTM()!);
        return { x: point.x, y: point.y };
      });
      expect(marker.x).toBeCloseTo(position.x, 2); expect(marker.y).toBeCloseTo(position.y, 2);
      const download = page.waitForEvent('download');
      await plot.getByRole('button', { name: 'Export plot' }).focus(); await page.keyboard.press('Enter');
      expect(readFileSync((await (await download).path())!, 'utf8')).not.toContain('data-interactive');
      await page.mouse.move(0, 0); await expect(tooltip).toHaveCount(0);
    }
  }
});

test('normalizing an edited valid draft preserves its calculation and remains runnable', { tag: '@smoke' }, async ({ page }) => {
  await open(page);
  await page.locator('input[type=file]').setInputFiles({ name: 'normalize.toml', mimeType: 'text/plain', buffer: Buffer.from(bands.replace('radius = 0.20', 'radius = 0.23')) });
  await expect(page.getByRole('button', { name: 'Normalize', exact: true })).toBeEnabled();
  await page.getByRole('button', { name: 'Normalize', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Run', exact: true })).toBeEnabled();
  await expect(page.locator('.wb-validation')).toContainText('Configuration applied');
  await run(page);
  const result = JSON.parse((await exported(page, 'json')).toString());
  expect(result.config.geometry.objects[0].radius).toBe(.23);
  expect(result.results[0].arrays.frequencies.shape).toEqual([7, 8]);
});

test('selected choices preserve settings and Run keeps an existing split arrangement', async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await open(page); await apply(page, bands);
  await page.getByRole('tab', { name: 'Study', exact: true }).click();
  await page.getByRole('group', { name: 'Task', exact: true }).getByRole('button', { name: 'Bands', exact: true }).click();
  await expect(page.getByRole('textbox', { name: 'Intervals per segment', exact: true })).toHaveValue('2');
  await page.getByRole('tab', { name: 'Geometry', exact: true }).click();
  await page.getByRole('button', { name: 'Split view', exact: true }).click();
  await run(page);
  await expect(page.getByRole('tablist', { name: 'Editor tabs', exact: true }).getByRole('tab', { name: 'Geometry', exact: true })).toHaveAttribute('aria-selected', 'true');
  await expect(page.getByRole('tablist', { name: 'Split editor tabs' }).getByRole('tab', { name: 'Results', exact: true })).toHaveAttribute('aria-selected', 'true');
});

test('streaming bands use the complete planned axis from the first samples', async ({ page }) => {
  await open(page);
  await apply(page, bands.replace('[12, 16]', '[32, 32]').replace('intervals_per_segment = 2', 'intervals_per_segment = 30'));
  // Observe real streamed frames, including those produced faster than a test can click.
  await page.evaluate(() => {
    const frames: { maximum: string | null; samples: number; labels: string[] }[] = [];
    Object.assign(window, { bandFrames: frames });
    new MutationObserver(() => {
      const svg = document.querySelector('.wb-plot svg[data-samples]');
      if (svg) frames.push({ maximum: svg.getAttribute('data-x-max'), samples: Number(svg.getAttribute('data-samples')), labels: [...svg.querySelectorAll('text')].map(label => label.textContent ?? '') });
    }).observe(document.querySelector('.wb-results') ?? document.getElementById('wb-panel-Results')!, { childList: true, subtree: true, attributes: true });
  });
  await run(page);
  const frames = await page.evaluate(() => (window as unknown as { bandFrames: { maximum: string; samples: number; labels: string[] }[] }).bandFrames);
  expect(frames.some(frame => frame.samples > 0 && frame.samples < 91)).toBe(true);
  expect(new Set(frames.map(frame => frame.maximum)).size).toBe(1);
  expect(frames.every(frame => frame.labels.includes('X') && frame.labels.includes('M'))).toBe(true);
});
