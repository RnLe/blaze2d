// Actual wasm32 public-workflow correctness checks. This is not a timing study.
// Complete physical fields are retained for the independent Python FFT audit.
import {readFile, writeFile} from 'node:fs/promises';
import {pathToFileURL} from 'node:url';
import path from 'node:path';
import {createHash} from 'node:crypto';

const [directory, output] = process.argv.slice(2);
if (!directory || !output) throw Error('usage: node wasm_support_check.mjs MODULE_DIRECTORY OUTPUT.json');
const binary = await readFile(path.join(directory, 'blaze2d_backend_wasm_bg.wasm'));
const wasm = await import(pathToFileURL(path.resolve(directory, 'blaze2d_backend_wasm.js')));
await wasm.default({module_or_path: binary});
const source = await readFile(new URL('../../examples/calculations/metric-support.toml', import.meta.url), 'utf8');
const validated = wasm.validateToml(source);
if (!validated.ok) throw Error(JSON.stringify(validated.errors));
const base = validated.config;
base.grid.resolution = 17;
base.bands.count = 5;
base.bands.path = {points: [[0, 0], [.17, .09], [2 / 3, 1 / 3], [1, 0]], basis: 'reciprocal_fractional'};
base.bands.discretization.angular_cutoff = 2 * Math.PI * 3.1;
const records = [], negatives = [], issues = [], comparisons = [];
const check = (condition, message) => { if (!condition) issues.push(message); };
const plain = value => JSON.parse(JSON.stringify(value, (_, x) => x instanceof Float64Array ? Array.from(x) : x));
const bits = value => {
  const b = new ArrayBuffer(8), v = new DataView(b);
  v.setFloat64(0, value, false);
  return v.getBigUint64(0, false).toString(16).padStart(16, '0');
};
function unreduced(c) {
  c.bands.symmetry.sectors = 'off';
  c.bands.symmetry.sector_storage = 'full_grid';
  c.bands.symmetry.reconstruct_partners = false;
  delete c.bands.symmetry.sector_guard_budget;
  return c;
}
function validateArrays(record, label) {
  for (const [name, a] of Object.entries(record.arrays)) {
    const length = a.shape.reduce((x, y) => x * y, 1) * (a.dtype === 'complex128' ? 2 : 1);
    check(a.data instanceof Float64Array, `${label}: ${name} is not a typed array`);
    check(a.data.length === length && Array.from(a.data).every(Number.isFinite), `${label}: ${name} shape/finite data`);
  }
}
function run(name, config, expectedFailure = false) {
  const text = wasm.applyConfig(source, JSON.stringify(config));
  const calc = new wasm.Calculation(text), events = [];
  let result = null, failure = null, preview, summary;
  try {
    preview = calc.geometryPreview();
    summary = calc.summary;
    try { result = calc.runJob(0, e => events.push(e)); }
    catch (e) { failure = e; result = e.partial_result ?? null; }
  } finally { calc.free(); }
  check(Boolean(failure) === expectedFailure, `${name}: unexpected success/failure`);
  if (result) {
    validateArrays(result, name);
    const count = result.arrays.frequencies.shape[0];
    check(events.filter(e => e.event === 'progress').length === count, `${name}: progress and output disagree`);
    check(result.metadata.discretization.completed_points === count, `${name}: completed row count`);
    check(result.metadata.converged === !expectedFailure, `${name}: convergence metadata`);
    check(result.metadata.certification.pencil === 'selected_metric_fourier_support', `${name}: pencil identity`);
    check(result.arrays.eigenvectors.shape.join() === [count, config.bands.count, config.grid.resolution, config.grid.resolution].join(), `${name}: physical output shape`);
    for (const [i, r] of result.metadata.discretization.points.entries()) {
      check(r.support.bit_encoding === 'hexadecimal_u64', `${name}/${i}: lossy support identity encoding`);
      check(r.support.k_bits.every((x, j) => x === bits(result.arrays.k_points.data[2 * i + j])), `${name}/${i}: k identity bits`);
      check(r.selected.projected.residuals.every((x, j) => x === result.arrays.residuals.data[i * config.bands.count + j]), `${name}/${i}: returned certificate order`);
      check(r.selected.full_rectangle != null, `${name}/${i}: lost discretization diagnostic`);
      const ledger = result.metadata.symmetry.points[i];
      check(ledger.attempted_iterations === ledger.solver_attempts.reduce((sum, a) => sum + a.iterations, 0), `${name}/${i}: lost attempted iterations`);
      check(result.metadata.iterations[i] === ledger.attempted_iterations, `${name}/${i}: lost public iterations`);
      if (!expectedFailure) check(r.selected.accepted, `${name}/${i}: selected pencil failed certification`);
    }
  } else check(false, `${name}: no result or structured partial result`);
  const item = plain({name, config, source: text, expected_failure: expectedFailure, result, failure, preview, summary, events});
  records.push(item);
  return item;
}
function compare(candidate, reference, exact = false) {
  if (!candidate.result || !reference.result) return;
  const a = candidate.result.arrays.frequencies, b = reference.result.arrays.frequencies;
  let error = 0;
  for (let i = 0; i < a.shape[0]; ++i) {
    const width = a.shape[1];
    const x = a.data.slice(i * width, (i + 1) * width).sort((x, y) => x - y);
    const y = b.data.slice(i * width, (i + 1) * width).sort((x, y) => x - y);
    for (let j = 0; j < width; ++j) error = Math.max(error, Math.abs(x[j] - y[j]) / Math.max(1, Math.abs(y[j])));
    check(JSON.stringify(candidate.result.metadata.discretization.points[i].support) === JSON.stringify(reference.result.metadata.discretization.points[i].support), `${candidate.name}/${i}: fallback changed support`);
  }
  check(error < 1e-8, `${candidate.name}: same-pencil spectral disagreement ${error}`);
  const exactArrays = JSON.stringify(candidate.result.arrays) === JSON.stringify(reference.result.arrays);
  if (exact) check(exactArrays, `${candidate.name}: fallback changed numerical arrays`);
  comparisons.push({candidate: candidate.name, reference: reference.name, max_scaled_frequency_error: error, exact_arrays: exactArrays, exact_required: exact});
}
for (const polarization of ['TM', 'TE']) {
  const raw = structuredClone(base); raw.polarization = polarization;
  const reduced = run(`raw-${polarization}-sectors`, raw);
  const control = run(`raw-${polarization}-unreduced`, unreduced(structuredClone(raw)));
  compare(reduced, control);
  if (reduced.result) {
    const points = reduced.result.metadata.discretization.points;
    for (const i of [0, 2, 3]) check(points[i].sector_attempt?.accepted && points[i].returned_labels?.length === 5, `${reduced.name}/${i}: qualified sector path absent`);
    check(points[1].sector_attempt === null && points[1].selected.method === 'lobpcg', `${reduced.name}: generic point did not release constraint`);
    check(points[3].selected.analytic_kernel, `${reduced.name}: folded Gamma kernel missing`);
  }
  const analytic = structuredClone(raw);
  analytic.dielectric.smoothing = 'analytic'; analytic.bands.path.points = [[0, 0]];
  const fallback = run(`analytic-${polarization}-fallback`, analytic);
  const plainControl = run(`analytic-${polarization}-unreduced`, unreduced(structuredClone(analytic)));
  compare(fallback, plainControl, true);
  check(fallback.result?.metadata.discretization.points[0].qualification?.authorized === false, `${fallback.name}: incompatible material accepted`);

  const failing = structuredClone(raw);
  failing.grid.resolution = 33; failing.bands.count = 8;
  failing.bands.discretization.angular_cutoff = 2 * Math.PI * 7;
  failing.eigensolver.max_iterations = 1;
  const failed = run(`iteration-limit-${polarization}`, failing, true);
  check(failed.result?.arrays.frequencies.shape[0] === 1, `${failed.name}: partial width or count`);
  const ledger = failed.result?.metadata.failed_point.symmetry;
  check(ledger?.solver_attempts.length === 2 && ledger.attempted_iterations > ledger.retained_iterations, `${failed.name}: lost sector/fallback work`);
}
const growth = structuredClone(base);
growth.grid.resolution = 33; growth.bands.count = 8;
growth.bands.discretization.angular_cutoff = 2 * Math.PI * 7;
growth.bands.discretization.max_allocation_rounds = 1;
growth.bands.path.points = [[0, 0]];
const grown = run('growth-limit-TE-fallback', growth);
const grownControl = run('growth-limit-TE-unreduced', unreduced(structuredClone(growth)));
compare(grown, grownControl, true);
check(grown.result?.metadata.discretization.points[0].sector_attempt?.rejection?.includes('allocation round limit'), 'growth limit not exercised');
const empty = structuredClone(base); empty.bands.discretization.max_estimated_payload_bytes = 1;
const budget = run('empty-budget', empty, true);
check(budget.result?.arrays.frequencies.shape[0] === 0, 'driver budget produced fabricated rows');

for (const [name, change] of [
  ['precision', c => c.eigensolver.precision = 'f32'],
  ['reuse', c => c.bands.symmetry.k_reuse = 'unitary'],
  ['pt', c => c.bands.symmetry.arithmetic = 'pt_real'],
  ['traversal', c => c.bands.symmetry.k_traversal = 'nearest'],
  ['storage', c => c.bands.symmetry.sector_storage = 'full_grid'],
  ['external', c => c.dielectric.source = 'external'],
  ['clipped-support', c => c.bands.discretization.angular_cutoff = 1e3],
  ['overfull-band-count', c => c.bands.count = 100],
]) {
  const c = structuredClone(base); change(c);
  let error = null;
  try { const s = wasm.applyConfig(source, JSON.stringify(c)); const calc = new wasm.Calculation(s); calc.free(); }
  catch (e) { error = e; }
  check(error != null, `${name}: invalid request silently accepted`);
  negatives.push(plain({name, configuration: c, error}));
}
const report = {schema: 'blaze2d/wasm-support-check/1', scope: 'correctness and interface checks; no performance inference',
  build: wasm.buildInfo(), node: process.version, v8: process.versions.v8,
  wasm_sha256: createHash('sha256').update(binary).digest('hex'),
  runner_sha256: createHash('sha256').update(await readFile(new URL(import.meta.url))).digest('hex'),
  records, configuration_negatives: negatives, comparisons, issues};
await writeFile(output, JSON.stringify(report) + '\n', {flag: 'wx'});
console.log(JSON.stringify({records: records.length, configuration_negatives: negatives.length, comparisons, issues}));
if (issues.length) process.exitCode = 1;
