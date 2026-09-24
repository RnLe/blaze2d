// Actual wasm32 admission and public planning checks; no timing inference.
import {readFile, writeFile} from 'node:fs/promises';
import {pathToFileURL} from 'node:url';
import path from 'node:path';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';

const [directory, output] = process.argv.slice(2);
if (!directory || !output) throw Error('usage: node wasm_memory_check.mjs MODULE_DIRECTORY OUTPUT.json');
const binary = await readFile(path.join(directory, 'blaze2d_backend_wasm_bg.wasm'));
const wasm = await import(pathToFileURL(path.resolve(directory, 'blaze2d_backend_wasm.js')));
await wasm.default({module_or_path: binary});
const template = wasm.defaultToml();
const base = wasm.validateToml(template).config;
base.grid.resolution = 9;
base.bands.count = 5;
base.bands.path = {points: [[0, 0]], basis: 'reciprocal_fractional'};
base.bands.symmetry.plane_group = 'p4m';
base.bands.symmetry.sectors = 'adaptive_group';
base.bands.symmetry.max_sector_workspace_bytes = 8 * 1024 * 1024;
base.eigensolver.tolerance = 1e-14;
base.eigensolver.max_iterations = 500;
base.results.eigenvectors = true;
const plain = value => JSON.parse(JSON.stringify(value, (_, x) => x instanceof Float64Array ? Array.from(x) : x));
const records = [], planning = [];
function calculate(name, config) {
  const source = wasm.applyConfig(template, JSON.stringify(config));
  const calc = new wasm.Calculation(source);
  try {
    const result = calc.runJob(0, () => {});
    assert.equal(result.metadata.converged, true);
    const array_sha256 = {};
    for (const [key, value] of Object.entries(result.arrays)) {
      assert(value.data instanceof Float64Array, `${name}/${key}: expected typed f64 output`);
      assert.equal(value.data.length, value.shape.reduce((a, b) => a * b, 1) * (value.dtype === 'complex128' ? 2 : 1));
      assert(value.data.every(Number.isFinite), `${name}/${key}: nonfinite output`);
      array_sha256[key] = createHash('sha256').update(Buffer.from(value.data.buffer, value.data.byteOffset, value.data.byteLength)).digest('hex');
    }
    assert.deepEqual(result.arrays.eigenvectors.shape, [1, config.bands.count, 9, 9]);
    const frequencies = result.arrays.frequencies.data;
    const relative = result.arrays.residuals.data, absolute = result.arrays.absolute_residuals.data;
    assert(frequencies[0] < 1e-8 && absolute[0] <= 1e-7, `${name}: Gamma kernel certificate`);
    assert(relative.slice(1).every(x => x <= 1e-5), `${name}: nonzero-mode certificate`);
    assert(result.metadata.certification.b_orthogonality_defect <= 1e-8, `${name}: orthogonality`);
    const ledger = result.metadata.symmetry.points[0];
    assert.equal(result.metadata.iterations[0], ledger.solver_attempts.reduce((sum, a) => sum + a.iterations, 0));
    const row = plain({name, config, source, summary: calc.summary, array_sha256, result});
    records.push(row); return row;
  } finally { calc.free(); }
}
for (const polarization of ['TM', 'TE']) {
  const c = structuredClone(base); c.polarization = polarization;
  const admitted = calculate(`admitted-${polarization}`, c);
  const workspace = admitted.result.metadata.symmetry.points[0].sectors.workspace;
  assert.equal(workspace.model, 'rectangular-sector-payload/1');
  assert(workspace.peak_estimated_payload_bytes <= workspace.budget_bytes);
  c.bands.symmetry.max_sector_workspace_bytes = 0;
  const fallback = calculate(`zero-budget-${polarization}`, c);
  const ledger = fallback.result.metadata.symmetry.points[0];
  assert.equal(ledger.sectors, null);
  assert(ledger.solver_attempts[0].rejection.includes('above budget 0'));
  assert.equal(ledger.solver_attempts[0].iterations, 0);
  c.bands.symmetry.sectors = 'off';
  const control = calculate(`unrestricted-${polarization}`, c);
  assert.deepEqual(fallback.result.arrays, control.result.arrays);
  assert.deepEqual(fallback.array_sha256, control.array_sha256);
  assert.equal(fallback.summary.estimated_peak_bytes, control.summary.estimated_peak_bytes);
  for (let j = 0; j < c.bands.count; ++j) {
    const a = admitted.result.arrays.frequencies.data[j], b = control.result.arrays.frequencies.data[j];
    assert(Math.abs(a - b) / Math.max(1, Math.abs(b)) <= 1e-7, `admitted-${polarization}: spectrum disagreement`);
  }
}
for (const n of [65, 129, 257]) {
  const c = structuredClone(base); c.grid.resolution = n;
  c.bands.symmetry.max_sector_workspace_bytes = 512 * 1024 * 1024;
  const withCap = new wasm.Calculation(wasm.applyConfig(template, JSON.stringify(c)));
  let selected;
  try { selected = withCap.summary; } finally { withCap.free(); }
  c.bands.symmetry.sectors = 'off';
  const withoutCap = new wasm.Calculation(wasm.applyConfig(template, JSON.stringify(c)));
  let unrestricted;
  try { unrestricted = withoutCap.summary; } finally { withoutCap.free(); }
  assert.equal(selected.estimated_peak_bytes-unrestricted.estimated_peak_bytes, 512 * 1024 * 1024);
  planning.push({grid: [n, n], selected, unrestricted});
}
const report = {schema: 'blaze2d/wasm-memory-check/1', scope: 'correctness and planning; no performance inference',
  build: wasm.buildInfo(), node: process.version, v8: process.versions.v8,
  wasm_sha256: createHash('sha256').update(binary).digest('hex'),
  runner_sha256: createHash('sha256').update(await readFile(new URL(import.meta.url))).digest('hex'),
  records, planning};
await writeFile(output, JSON.stringify(report)+'\n', {flag: 'wx'});
console.log(JSON.stringify({calculations: records.length, large_grid_plans: planning.length, build: report.build}));
