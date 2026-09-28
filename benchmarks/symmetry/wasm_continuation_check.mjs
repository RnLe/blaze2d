// Portable public calculation objects must retain certified state between jobs.
import {readFile,writeFile} from 'node:fs/promises';
import {pathToFileURL} from 'node:url';
import {createHash} from 'node:crypto';
import path from 'node:path';
const [directory,output] = process.argv.slice(2);
const binary = await readFile(path.join(directory,'blaze2d_backend_wasm_bg.wasm'));
const wasm = await import(pathToFileURL(path.resolve(directory,'blaze2d_backend_wasm.js')));
await wasm.default({module_or_path:binary});
const original = wasm.defaultToml(), base = wasm.validateToml(original).config;
base.grid.resolution = 17; base.bands.count = 5; base.bands.tracking = false;
base.geometry.lattice = {type:'triangular',a:1};
base.geometry.objects[0].center = [0,0];
base.dielectric.smoothing = 'voronoi';
base.eigensolver.precision = 'f64'; base.eigensolver.tolerance = 1e-14; base.eigensolver.max_iterations = 600;
base.bands.path = {points:[[0,0],[.17,.09],[-.17,-.09],[0,0]],basis:'reciprocal_fractional'};
base.bands.discretization = {kind:'metric_ball',angular_cutoff:2*Math.PI*3.1};
base.results.eigenvectors = true;
base.sweeps = [{name:'radius',target:`geometry.objects.${base.geometry.objects[0].name}.radius`,values:[.18,.22]}];
const issues = [], records = [];
const check = (value,message) => {if (!value) issues.push(message);};
function run(config) {
  const calc = new wasm.Calculation(wasm.applyConfig(original,JSON.stringify(config)));
  try {return [0,1].map(index=>calc.runJob(index,()=>{}));}
  finally {calc.free();}
}
for (const polarization of ['TM','TE']) {
  const referenceConfig = structuredClone(base); referenceConfig.polarization = polarization;
  referenceConfig.bands.symmetry = {arithmetic:'fourier_complex'};
  const candidateConfig = structuredClone(referenceConfig);
  candidateConfig.bands.symmetry = {arithmetic:'pt_real',plane_group:'p6m',k_reuse:'time_reversal',
    continuation:'previous_geometry',max_continuation_bytes:1048576,
    coarse_start:{resolution:[9,9],max_seed_field_bytes:1048576}};
  const reference = run(referenceConfig), candidate = run(candidateConfig);
  let maxDifference=0,seeds=0,reuses=0,coarse=0;
  for (let job=0;job<2;job++) {
    const a=candidate[job],b=reference[job];
    check(a.metadata.converged,`${polarization}/${job}: convergence`);
    check(a.arrays.eigenvectors.data instanceof Float64Array,`${polarization}: typed fields`);
    check(a.arrays.eigenvectors.shape.join()===[4,5,17,17].join(),`${polarization}: physical field shape`);
    for (let row=0;row<4;row++) {
      const x=Array.from(a.arrays.frequencies.data.slice(row*5,row*5+5)).sort((x,y)=>x-y);
      const y=Array.from(b.arrays.frequencies.data.slice(row*5,row*5+5)).sort((x,y)=>x-y);
      x.forEach((v,i)=>maxDifference=Math.max(maxDifference,Math.abs(v-y[i])));
      const r=a.metadata.symmetry.points[row];
      seeds+=Number(Boolean(r.continuation?.seed)); reuses+=Number(r.source_index!==null);
      coarse+=Number(Boolean(r.coarse_start?.used_as_seed));
      check(r.continuation.cache_payload_bytes<=1048576,`${polarization}: continuation bound`);
      check(r.attempted_iterations===r.solver_attempts.reduce((n,v)=>n+v.iterations,0),`${polarization}: complete work ledger`);
    }
  }
  check(maxDifference<2e-7,`${polarization}: spectrum differs ${maxDifference}`);
  check(seeds===2 && reuses===4 && coarse===1,`${polarization}: mechanisms did not execute ${seeds}/${reuses}/${coarse}`);
  records.push({polarization,maxDifference,seeds,reuses,coarse});
}
const result={binary_sha256:createHash('sha256').update(binary).digest('hex'),build:wasm.buildInfo(),records,issues};
await writeFile(output,JSON.stringify(result,null,2)+'\n');
console.log(JSON.stringify(result));
if (issues.length) process.exitCode=1;
