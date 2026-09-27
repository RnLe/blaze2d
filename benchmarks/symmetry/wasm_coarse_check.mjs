// Actual wasm32 public coarse-initialization checks. All outcomes and fields are retained.
// Correctness only: no elapsed-time or memory-performance inference.
import {readFile, writeFile} from 'node:fs/promises';
import {pathToFileURL} from 'node:url';
import path from 'node:path';
import {createHash} from 'node:crypto';
const [directory, output] = process.argv.slice(2);
if (!directory || !output) throw Error('usage: node wasm_coarse_check.mjs MODULE_DIRECTORY OUTPUT.json');
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


base.bands.tracking = false;
base.bands.symmetry.sectors = 'off';
base.bands.symmetry.reconstruct_partners = false;
base.bands.symmetry.sector_guard_budget = null;
for (const smoothing of ['analytic','none']) for (const polarization of ['TM','TE']) {
  const cold = structuredClone(base); cold.polarization=polarization; cold.dielectric.smoothing=smoothing;
  const label=smoothing+'/'+polarization;
  cold.bands.symmetry.arithmetic='spatial_complex';
  const reference=run(label+'/cold',cold);
  for (const pt of [false,true]) for (const sector of [false,true]) {
    const c=structuredClone(cold);
    c.bands.symmetry.arithmetic=pt?'pt_real':'spatial_complex';
    c.bands.symmetry.sectors=sector?'adaptive_group':'off';
    c.bands.symmetry.sector_guard_budget=sector?4:null;
    c.bands.symmetry.coarse_start={resolution:[7,7],max_seed_field_bytes:1048576};
    const row=run(`${label}/coarse/pt-${pt}/sector-${sector}`,c);
    if (!row.result || !reference.result) continue;
    const reports=row.result.metadata.symmetry.points;
    check(reports[0].coarse_start?.used_as_seed===true, `${row.name}: coarse seed ignored`);
    check(reports[0].coarse_start?.iterations>0, `${row.name}: source iterations absent`);
    check(reports.slice(1).every(p=>p.coarse_start==null), `${row.name}: initialization repeated`);
    const a=reference.result.arrays.frequencies.data,b=row.result.arrays.frequencies.data;
    const error=Math.max(...a.map((v,i)=>Math.abs(v-b[i])/Math.max(1,Math.abs(v),Math.abs(b[i]))));
    comparisons.push({candidate:row.name,reference:reference.name,frequency_error:error});
    check(error<=1e-6,`${row.name}: spectrum changed`);
    check(row.summary.estimated_peak_bytes>reference.summary.estimated_peak_bytes, `${row.name}: planning omits preparation`);
    if(sector) check(reports.every(p=>p.solver_attempts.some(a=>a.method==='sectors')), `${row.name}: sector request ignored`);
    if(pt) check(reports.every(p=>p.sectors!=null || p.solver_attempts.some(a=>a.method==='fourier')), `${row.name}: PT request ignored`);
  }
  const refused=structuredClone(cold);
  refused.bands.symmetry.coarse_start={resolution:[7,7],max_seed_field_bytes:0};
  const fallback=run(label+'/budget-fallback',refused);
  check(same(fallback.array_sha256,reference.array_sha256),`${polarization}: zero-budget arrays changed`);
  if(fallback.result && reference.result) {
    check(same(fallback.result.metadata.iterations,reference.result.metadata.iterations),`${polarization}: zero-budget iterations changed`);
    check(fallback.result.metadata.symmetry.points[0].coarse_start?.used_as_seed===false,`${polarization}: rejected budget used`);
  }
}
for(const resolution of [[0,7],[14,7],[13,13]]) {
 const c=structuredClone(base);c.bands.symmetry.coarse_start={resolution,max_seed_field_bytes:1048576};
 let rejected=false;try{const source=wasm.applyConfig(template,JSON.stringify(c));rejected=!wasm.validateToml(source).ok;}catch{rejected=true;}
 negatives.push({resolution,rejected});check(rejected,'invalid coarse resolution accepted');
}
await writeFile(output,JSON.stringify({scope:'actual wasm32 correctness; no timing inference',module_sha256:sha(binary),build:wasm.buildInfo(),node:process.version,records,comparisons,negatives,issues}));
console.log(JSON.stringify({calculations:records.length,comparisons:comparisons.length,issues}));
process.exitCode=issues.length?1:0;
