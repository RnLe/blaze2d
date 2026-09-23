// Actual wasm32 execution under Node/V8. Instantiation and bridge work are
// reported separately; this does not include a browser UI or GPU transfers.
import {readFile} from 'node:fs/promises';
import {pathToFileURL} from 'node:url';
import path from 'node:path';
import {createHash} from 'node:crypto';
import {performance} from 'node:perf_hooks';
const [directory, caseName='square-path', nArg='31', bandsArg='8', pol='TM', mode='off', tolArg='1e-14'] = process.argv.slice(2);
const n=Number(nArg), bands=Number(bandsArg), tolerance=Number(tolArg);
if (![n,bands].every(x=>Number.isInteger(x)&&x>0) || !['TM','TE'].includes(pol) || !['off','pt','unitary','pt-unitary'].includes(mode) || !(tolerance>0&&Number.isFinite(tolerance))) throw Error('invalid benchmark arguments');
const loadStart=performance.now();
const binary=await readFile(path.join(directory,'blaze2d_backend_wasm_bg.wasm'));
const wasm=await import(pathToFileURL(path.resolve(directory,'blaze2d_backend_wasm.js')));
const exports=await wasm.default({module_or_path:binary});
const instantiationSeconds=(performance.now()-loadStart)/1000;
const cfg=wasm.validateToml(wasm.defaultToml()).config;
cfg.grid.resolution=[n,n]; cfg.bands.count=bands; cfg.bands.tracking=false;
cfg.polarization=pol; cfg.eigensolver.precision='f64'; cfg.eigensolver.tolerance=tolerance;
cfg.eigensolver.max_iterations=500; cfg.eigensolver.block_size=0;
const isExport=caseName.endsWith('-export');
const withoutExport=caseName.replace(/-export$/,'');
const fieldOutput=!withoutExport.endsWith('-values');
const baseCase=withoutExport.replace(/-values$/,'');
if(baseCase==='square-path') cfg.bands.path={preset:'square',intervals_per_segment:3,basis:'reciprocal_fractional'};
else if(baseCase==='square-star') cfg.bands.path={points:[[.13,.21],[-.21,.13],[-.13,-.21],[.21,-.13],[.13,-.21],[.21,.13],[-.13,.21],[-.21,-.13]],basis:'reciprocal_fractional'};
else if(baseCase==='repeated') cfg.bands.path={points:[[.13,.21],[.27,.09],[.13,.21],[.27,.09]],basis:'reciprocal_fractional'};
else throw Error('unknown case');
cfg.bands.symmetry.k_reuse=mode.includes('unitary')?'unitary':'off';
cfg.bands.symmetry.arithmetic=mode.startsWith('pt')?'pt_real':'spatial_complex';
cfg.results.eigenvectors=fieldOutput;
const source=wasm.applyConfig(wasm.defaultToml(),JSON.stringify(cfg));
const countersAvailable=typeof wasm.profileSnapshot==='function';
if(countersAvailable) wasm.resetProfile();
const memoryBefore=exports.memory.buffer.byteLength;
const begin=performance.now();
const calc=new wasm.Calculation(source);
let record;
try {record=calc.runJob(0,()=>{});} finally {calc.free();}
const bridgeSeconds=(performance.now()-begin)/1000;
const encodeStart=performance.now();
let serializedBytes=0, serializedSha256=null;
if(isExport) {
 const serialized=JSON.stringify(record,(_,v)=>v instanceof Float64Array?Array.from(v):v);
 serializedBytes=Buffer.byteLength(serialized);
 serializedSha256=createHash('sha256').update(serialized).digest('hex');
}
const serializationSeconds=isExport?(performance.now()-encodeStart)/1000:0;
const elapsed=(performance.now()-begin)/1000;
const peakRss=process.resourceUsage().maxRSS;
const profile=countersAvailable?wasm.profileSnapshot():{counters:{},stage_counters:{},timings:{}};
const arr=record.arrays, meta=record.metadata;
const nk=arr.frequencies.shape[0];
const rows=(values,width)=>Array.from({length:values.length/width},(_,i)=>Array.from(values.slice(i*width,(i+1)*width)));
const ks=rows(arr.k_points.data,2), res=rows(arr.residuals.data,bands), abs=rows(arr.absolute_residuals.data,bands);
const spectra=rows(arr.frequencies.data,bands).map(r=>r.map(x=>x*2*Math.PI));
const sym=meta.symmetry.points;
const points=ks.map((k,i)=>({k,iterations:meta.iterations[i],converged:meta.converged_per_point[i],
 residuals:res[i],absolute_residuals:abs[i],b_orthogonality_defect:meta.certification.b_orthogonality_per_point[i],
 field_count:arr.eigenvectors?.shape[1]??0,symmetry:sym[i]}));
const reusedPoints=sym.filter(s=>s.source_index!=null).length;
const policyMatches=sym.every(s=>s.requested===(mode.includes('unitary')?'unitary':'off'));
const pathSuccess=policyMatches && (mode.startsWith('pt') ? sym.every(s=>s.coordinates?.representation==='pt_real_te' || s.source_index!=null) : sym.every(s=>s.coordinates==null)) && (!(mode.includes('unitary') && ['square-star','repeated'].includes(baseCase)) || reusedPoints>0);
console.log(JSON.stringify({case:caseName,mode,grid:[n,n],bands,polarization:pol,precision:'f64',tolerance,
 seconds:elapsed,peak_rss_kib:peakRss,expected_points:nk,requested_path_success:pathSuccess,field_output:fieldOutput,
 arrays:Object.fromEntries(Object.entries(arr).map(([name,a])=>[name,{dtype:a.dtype,shape:a.shape,payload_bytes:a.data.byteLength}])),
 spectrum:spectra,points,...profile,counters_available:countersAvailable,linear_memory_bytes_before_job:memoryBefore,linear_memory_bytes_after_job:exports.memory.buffer.byteLength,backend:'wasm32-unknown-unknown / nalgebra / Node V8',
 configuration:cfg,reused_points:reusedPoints,node_version:process.version,v8_version:process.versions.v8,build:wasm.buildInfo(),wasm_sha256:createHash('sha256').update(binary).digest('hex'),
 instantiation_seconds:instantiationSeconds,bridge_seconds:bridgeSeconds,core_seconds:meta.elapsed_seconds,
 serialization_seconds:serializationSeconds,serialized_bytes:serializedBytes,serialized_sha256:serializedSha256,
 scope:'complete interface calculation including requested physical fields and WASM-to-JavaScript copies; optional JSON export with checksum; module instantiation separate; no browser UI or GPU'}));
