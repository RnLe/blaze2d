import { execFileSync } from 'node:child_process';
import { readFile, rm, writeFile } from 'node:fs/promises';

const repository = new URL('../..', import.meta.url), output = new URL('../public/wasm-blaze/', import.meta.url);
execFileSync('wasm-pack', ['build', '--target', 'web', '--out-dir', '../../web/public/wasm-blaze', '--release', '--', '--locked', '--features', 'bindings,wasm-linalg'], {
  cwd: new URL('../../crates/backend-wasm/', import.meta.url), stdio: 'inherit',
});
const wasm = await import(new URL('blaze2d_backend_wasm.js', output));
await wasm.default({ module_or_path: await readFile(new URL('blaze2d_backend_wasm_bg.wasm', output)) });
const info = wasm.buildInfo(), revision = execFileSync('git', ['rev-parse', 'HEAD'], { cwd: repository, encoding: 'utf8' }).trim();
if (info.source_revision !== revision || info.config_schema !== 'blaze2d/1' || info.result_schema !== 'blaze2d/result/1') {
  throw new Error('The WASM build revision or schema does not match this checkout.');
}
const report = wasm.validateToml(wasm.defaultToml());
if (!report.ok) throw new Error('The freshly built solver rejected its default calculation.');
await writeFile(new URL('build.json', output), JSON.stringify(info, null, 2) + '\n');
for (const name of ['.gitignore', 'package.json', 'README.md']) await rm(new URL(name, output), { force: true });
