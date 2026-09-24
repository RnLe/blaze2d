// Actual wasm32 public PT/sector checks. All outcomes and fields are retained.
// Correctness only: no elapsed-time or memory-performance inference.
import {readFile, writeFile} from 'node:fs/promises';
import {pathToFileURL} from 'node:url';
import path from 'node:path';
import {createHash} from 'node:crypto';
const [directory, output] = process.argv.slice(2);
if (!directory || !output) throw Error('usage: node wasm_pt_sector_check.mjs MODULE_DIRECTORY OUTPUT.json');
const sha = bytes => createHash('sha256').update(bytes).digest('hex');
const binary = await readFile(path.join(directory, 'blaze2d_backend_wasm_bg.wasm'));
const wasm = await import(pathToFileURL(path.resolve(directory, 'blaze2d_backend_wasm.js')));
await wasm.default({module_or_path: binary});
const template = wasm.defaultToml();
const base = wasm.validateToml(template).config;
base.grid.resolution = 13;
base.bands.count = 6;
base.bands.path = {points: [[0, 0], [.17, 0], [0, 0]], basis: 'reciprocal_fractional'};
base.bands.symmetry.plane_group = 'p4m';
base.bands.symmetry.sectors = 'adaptive_group';
base.bands.symmetry.arithmetic = 'pt_real';
base.bands.symmetry.sector_guard_budget = 4;
base.bands.symmetry.max_sector_workspace_bytes = 512 * 1024 * 1024;
base.eigensolver.tolerance = 1e-14;
base.eigensolver.max_iterations = 500;
base.results.eigenvectors = true;
const records = [], issues = [], comparisons = [], planning = [], negatives = [];
const check = (condition, message) => { if (!condition) issues.push(message); };
const plain = value => JSON.parse(JSON.stringify(value, (_, x) => x instanceof Float64Array ? Array.from(x) : x));
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);

function run(name, config) {
  let calculation, source, result = null, failure = null, summary, preview;
  const events = [], array_sha256 = {};
  try {
    source = wasm.applyConfig(template, JSON.stringify(config));
    const normalized = wasm.validateToml(source);
    check(normalized.ok, `${name}: normalized source invalid`);
    check(same(wasm.normalizeToml(source), wasm.normalizeToml(wasm.applyConfig(source, JSON.stringify(normalized.config)))), `${name}: TOML roundtrip changed`);
    calculation = new wasm.Calculation(source);
    summary = calculation.summary; preview = calculation.geometryPreview();
    result = calculation.runJob(0, e => events.push(e));
    const width = config.bands.count, count = config.bands.path.points.length, n = config.grid.resolution;
    check(result.metadata.converged === true, `${name}: not converged`);
    check(result.metadata.certification.source === 'fresh_operator_application', `${name}: stale certificate`);
    check(result.metadata.certification.b_orthogonality_defect <= 1e-8, `${name}: orthogonality`);
    check(same(result.arrays.eigenvectors.shape, [count, width, n, n]), `${name}: full field shape`);
    check(result.arrays.eigenvectors.dtype === 'complex128', `${name}: full field dtype`);
    check(same(Array.from(result.arrays.k_points.data), config.bands.path.points.flat()), `${name}: requested point order`);
    for (const [key, a] of Object.entries(result.arrays)) {
      check(a.data instanceof Float64Array, `${name}/${key}: typed f64 array`);
      check(a.data.length === a.shape.reduce((a, b) => a * b, 1) * (a.dtype === 'complex128' ? 2 : 1), `${name}/${key}: payload shape`);
      check(a.data.every(Number.isFinite), `${name}/${key}: nonfinite payload`);
      array_sha256[key] = sha(Buffer.from(a.data.buffer, a.data.byteOffset, a.data.byteLength));
    }
    for (let i = 0; i < count * width; ++i) {
      const f = result.arrays.frequencies.data[i];
      check(f < 1e-8 ? result.arrays.absolute_residuals.data[i] <= 1e-7 : result.arrays.residuals.data[i] <= 1e-5, `${name}/${i}: physical residual`);
    }
    check(result.metadata.symmetry.points.length === count, `${name}: missing point report`);
    for (const [i, p] of result.metadata.symmetry.points.entries()) {
      const attempts = p.solver_attempts;
      check(p.attempted_iterations === attempts.reduce((sum, a) => sum + a.iterations, 0), `${name}/${i}: lost attempted work`);
      check(result.metadata.iterations[i] === p.attempted_iterations, `${name}/${i}: public iteration mismatch`);
      check(attempts.filter(a => a.retained).length === 1, `${name}/${i}: retained attempt count`);
      check(p.retained_iterations === attempts.filter(a => a.retained).reduce((sum, a) => sum + a.iterations, 0), `${name}/${i}: retained iteration mismatch`);
    }
  } catch (error) {
    failure = error instanceof Error ? {message: error.message, stack: error.stack} : error;
    result = result ?? error.partial_result ?? null;
    check(false, `${name}: ${JSON.stringify(failure)}`);
  } finally { calculation?.free(); }
  const item = plain({name, config, source, result, failure, summary, preview, array_sha256, events});
  records.push(item); return item;
}

function sectors(item) {
  if (!item.result) return;
  const group = item.config.bands.symmetry.plane_group;
  const isPT = item.config.bands.symmetry.arithmetic === 'pt_real';
  const n = item.config.grid.resolution, rows = n*n;
  for (const [i, p] of item.result.metadata.symmetry.points.entries()) {
    const label = `${item.name}/${i}`, k = item.config.bands.path.points[i];
    const gamma = k.every(x => x === 0), attempts = p.solver_attempts;
    if (group === 'p4' && !gamma) {
      check(p.sectors === null, `${label}: constraint retained outside little group`);
      if (!isPT) continue;
      check(attempts[0].method === 'fourier', `${label}: missing unrestricted PT attempt`);
      if (p.coordinates === null) {
        check(attempts.length === 2 && attempts[0].retained === false && attempts[0].iterations > 0
          && attempts[0].rejection?.includes('fresh physical quality gates')
          && attempts[1].method === 'unrestricted' && attempts[1].retained, `${label}: unaccounted continuation fallback`);
      } else check(p.coordinates.representation === 'pt_real_te' && attempts.length === 1 && attempts[0].retained, `${label}: incorrect PT path`);
      continue;
    }
    const s = p.sectors;
    if (!s) { check(false, `${label}: requested sectors absent`); continue; }
    check(attempts.length === 1 && attempts[0].method === 'sectors' && attempts[0].retained, `${label}: sector solve not retained alone`);
    check(s.finite_dimension === rows && s.sum_sector_dimensions === rows
      && s.sectors.reduce((sum, r) => sum + r.dimension*r.partner_rows, 0) === rows, `${label}: finite dimensions/multiplicity`);
    check(s.kernel_owners === Number(gamma), `${label}: kernel ownership`);
    check(s.requested === item.config.bands.count && s.computed >= s.requested, `${label}: incomplete output`);
    check(s.workspace?.peak_estimated_payload_bytes <= s.workspace?.budget_bytes, `${label}: memory admission`);
    let real = 0, complex = 0;
    for (const r of s.sectors) {
      check(r.leakage <= 1e-8, `${label}: sector leakage`);
      if (!isPT) { check(r.coordinates == null, `${label}: unexpected coordinate solve`); continue; }
      const c = r.coordinates;
      if (!c) { check(false, `${label}: ignored PT-sector request`); continue; }
      check(Number.isFinite(c.final_sector_projection_defect) && c.maximum_real_subspace_leakage <= 1e-10, `${label}: final projection/reality`);
      check(r.vector_column_bytes === c.bytes_per_column, `${label}: coordinate byte accounting`);
      if (c.representation === 'pt_real_te') {
        ++real;
        check(c.bytes_per_column === Math.ceil(rows/2)*16 && c.qualification.authorized
          && r.coordinate_rejection == null && r.pt_projector_defect <= 1e-10, `${label}: invalid real sector authorization`);
      } else {
        ++complex;
        check(c.representation === 'fourier_complex' && c.bytes_per_column === rows*16
          && r.coordinate_rejection && r.pt_projector_defect > 1e-10, `${label}: unexplained complex sector`);
      }
    }
    // The C4v E1 matrix-unit row uses a complex rotation eigenvector. Its
    // isotypic pair is PT-real, but this individual partner row is not.
    const expectedComplex = group === 'p4' ? 2 : (gamma && item.config.bands.symmetry.reconstruct_partners ? 1 : 0);
    if (isPT) check(real > 0 && complex === expectedComplex, `${label}: incorrect real/complex projector selection`);
    if (item.config.bands.symmetry.reconstruct_partners) check(s.sectors.some(r => r.partner_rows > 1), `${label}: partner reconstruction ignored`);
  }
}

function compare(a, b, exact = false) {
  if (!a.result || !b.result) return;
  check(same(a.preview, b.preview), `${a.name}: sampled preview changed`);
  check(same(a.result.arrays.k_points, b.result.arrays.k_points), `${a.name}: k output changed`);
  const width = a.config.bands.count, count = a.config.bands.path.points.length;
  let error = 0;
  for (let i = 0; i < count; ++i) {
    const aa = a.result.arrays.frequencies.data.slice(i*width, (i+1)*width).sort((x,y) => x-y);
    const bb = b.result.arrays.frequencies.data.slice(i*width, (i+1)*width).sort((x,y) => x-y);
    for (let j = 0; j < width; ++j) error = Math.max(error, Math.abs(aa[j]-bb[j])/Math.max(1, Math.abs(aa[j]), Math.abs(bb[j])));
  }
  check(error <= 1e-7, `${a.name}: spectrum disagreement ${error}`);
  const exactArrays = same(a.array_sha256, b.array_sha256) && same(a.result.arrays, b.result.arrays);
  if (exact) {
    check(exactArrays, `${a.name}: fallback changes full arrays`);
    check(same(a.result.metadata.iterations, b.result.metadata.iterations), `${a.name}: fallback iterations changed`);
    check(a.summary.estimated_peak_bytes === b.summary.estimated_peak_bytes, `${a.name}: rejected reservation persists`);
  }
  comparisons.push({candidate: a.name, reference: b.name, max_scaled_frequency_error: error, exact_required: exact, exact_arrays: exactArrays});
}

for (const smoothing of ['analytic', 'none']) for (const polarization of ['TM', 'TE']) for (const group of ['p4', 'p4m']) {
  const c = structuredClone(base); c.dielectric.smoothing = smoothing; c.polarization = polarization; c.bands.symmetry.plane_group = group;
  const pt = run(`${smoothing}-${polarization}-${group}-pt`, c); sectors(pt);
  c.bands.symmetry.arithmetic = 'spatial_complex';
  const control = run(`${smoothing}-${polarization}-${group}-spatial`, c); sectors(control);
  compare(pt, control);
}
for (const polarization of ['TM', 'TE']) {
  const c = structuredClone(base); c.polarization = polarization; c.bands.symmetry.max_sector_workspace_bytes = 0;
  const fallback = run(`budget-${polarization}-pt`, c);
  if (fallback.result) for (const [i, p] of fallback.result.metadata.symmetry.points.entries()) {
    check(p.sectors === null && p.coordinates?.representation === 'pt_real_te', `${fallback.name}/${i}: lost PT fallback`);
    check(p.solver_attempts.length === 2 && p.solver_attempts[0].method === 'sectors' && p.solver_attempts[0].iterations === 0
      && !p.solver_attempts[0].retained && p.solver_attempts[0].rejection?.includes('above budget 0')
      && p.solver_attempts[1].method === 'fourier' && p.solver_attempts[1].retained, `${fallback.name}/${i}: memory rejection ledger`);
  }
  c.bands.symmetry.sectors = 'off'; delete c.bands.symmetry.sector_guard_budget;
  const control = run(`budget-${polarization}-off`, c); compare(fallback, control, true);
  const p = structuredClone(base); p.polarization = polarization; p.bands.path.points = [[0,0]]; p.bands.symmetry.reconstruct_partners = true;
  const partners = run(`partners-${polarization}-pt`, p); sectors(partners);
  p.bands.symmetry.arithmetic = 'spatial_complex';
  const spatial = run(`partners-${polarization}-spatial`, p); sectors(spatial); compare(partners, spatial);
}
for (const n of [65,129,257]) {
  const c = structuredClone(base); c.grid.resolution = n;
  let selected, unrestricted;
  const a = new wasm.Calculation(wasm.applyConfig(template, JSON.stringify(c)));
  try { selected = a.summary; } finally { a.free(); }
  c.bands.symmetry.sectors = 'off'; delete c.bands.symmetry.sector_guard_budget;
  const b = new wasm.Calculation(wasm.applyConfig(template, JSON.stringify(c)));
  try { unrestricted = b.summary; } finally { b.free(); }
  check(selected.estimated_peak_bytes-unrestricted.estimated_peak_bytes === 512*1024*1024, `${n}: capped planning overflow`);
  planning.push({grid:[n,n], selected, unrestricted});
}
for (const kind of ['orbit', 'mirror-kernel', 'metric']) {
  const c = structuredClone(base);
  if (kind === 'metric') c.bands.discretization = {kind:'metric_ball', angular_cutoff: 20};
  else c.bands.symmetry.sector_storage = 'orbit';
  if (kind === 'mirror-kernel') c.bands.symmetry.sector_kernel = 'mirror_tm_te';
  let rejection = null;
  try { const calc = new wasm.Calculation(wasm.applyConfig(template, JSON.stringify(c))); calc.free(); }
  catch (e) { rejection = plain(e); }
  const message = JSON.stringify(rejection);
  check(rejection && (kind === 'metric' ? message.includes('PT') || message.includes('arithmetic') : message.includes('orbit storage')), `${kind}: unsupported combination accepted/wrong reason`);
  negatives.push({kind, config:c, rejection});
}
const report = {schema:'blaze2d/wasm-pt-sector-check/1',scope:'correctness and planning only; no timing inference',
  build:wasm.buildInfo(),node:process.version,v8:process.versions.v8,
  wasm_sha256:sha(binary),runner_sha256:sha(await readFile(new URL(import.meta.url))),
  records,comparisons,planning,configuration_negatives:negatives,issues};
await writeFile(output, JSON.stringify(report)+'\n', {flag:'wx'});
console.log(JSON.stringify({records:records.length,comparisons:comparisons.length,planning:planning.length,negatives:negatives.length,issues,build:report.build}));
process.exitCode = Number(issues.length > 0);
